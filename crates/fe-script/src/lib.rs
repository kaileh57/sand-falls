//! Game scripts and mods in WebAssembly.
//!
//! A script is a WebAssembly module that runs inside the deterministic tick. It
//! reads the world through host functions imported from module `fe`, queues
//! [`Command`]s that the engine applies that tick, and sets game state on the
//! [`Simulation`]: per-player tool masks and zones, and a status line for HUDs.
//!
//! Three rules keep scripts deterministic on every machine:
//!
//! 1. Float instructions are rejected when the module loads (the runtime is
//!    [`wasmi`] with floats disabled), so no result depends on how a CPU rounds
//!    or which NaN bits it produces.
//! 2. Every call gets a fixed fuel budget ([`TICK_FUEL`], [`INIT_FUEL`]). Running
//!    out is a script fault at the same instruction on every machine, never a hang.
//! 3. After every call the script's linear memory and mutable globals are saved
//!    into [`Simulation::script_state`], so the state hash and snapshots cover the
//!    script, and a late joiner restores it with [`Script::restore`].
//!
//! # Host API
//!
//! All parameters and results are `i32` or `i64`. Strings are `(ptr, len)` pairs
//! in the module's exported `memory`.
//!
//! | Import | Meaning |
//! | --- | --- |
//! | `tick() -> i64` | Tick about to run |
//! | `seed() -> i64` | World seed |
//! | `get_material(x, y) -> i32` | Material id at a cell |
//! | `material_id(ptr, len) -> i32` | Material id by name, -1 if unknown |
//! | `count(x0, y0, x1, y1, mat) -> i32` | Cells of a material in a rectangle (costs fuel) |
//! | `player_count() -> i32`, `player_id(i) -> i32` | Characters, sorted by id |
//! | `avatar_exists(id)`, `avatar_x(id)`, `avatar_y(id)` | Character feet position |
//! | `paint(x, y, r, mat)`, `fill(x0, y0, x1, y1, mat)`, `dig(x, y, r)` | Queue edits |
//! | `explode(x, y, r, power)`, `carve(x0, y0, x1, y1)` | Queue an explosion or a body |
//! | `add_player(id, x, y)`, `remove_player(id)` | Queue character changes |
//! | `set_tool_mask(player, mask)` | Allowed tools, one bit per tool number |
//! | `set_tool_zone(player, x0, y0, x1, y1)`, `clear_tool_zone(player)` | Where tools work |
//! | `set_status(ptr, len)`, `log(ptr, len)` | HUD text and debug output |
//!
//! A module may export `on_init()`, called once after scene setup, and
//! `on_tick()`, called at the start of every tick.
#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic)]

use fe_core::{CellPos, IRect, MaterialId, StableHasher};
use fe_engine::{Command, Simulation};
use wasmi::{
    Caller, CompilationMode, Config, Engine, Error, Extern, Global, Instance, Linker, Memory,
    Module, Store, StoreLimits, StoreLimitsBuilder, TrapCode, TypedFunc, Val,
};

/// Fuel for one `on_tick` call. Roughly a few milliseconds of script time.
pub const TICK_FUEL: u64 = 2_000_000;
/// Fuel for the `on_init` call.
pub const INIT_FUEL: u64 = 50_000_000;
/// Largest linear memory a script may grow to, in bytes.
pub const MAX_MEMORY: usize = 16 << 20;
/// Most commands one call may queue.
pub const MAX_COMMANDS: usize = 4096;
/// Largest rectangle `count`, `fill` and `carve` accept, in cells.
pub const MAX_AREA: i64 = 1 << 22;

const STATE_MAGIC: [u8; 4] = *b"FES1";
const MAX_TEXT: i32 = 4096;
const MAX_LOGS: usize = 256;
const PAGE: usize = 65_536;

/// Stable hash of a script's bytes, sent in the network handshake. 0 means no script.
pub fn mod_hash(wasm: &[u8]) -> u64 {
    if wasm.is_empty() {
        return 0;
    }
    let mut h = StableHasher::new();
    h.write_bytes(wasm);
    h.finish().max(1)
}

/// Why a script could not be loaded or restored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScriptError {
    /// The bytes are not a valid module, or the runtime refused it.
    Invalid(String),
    /// The module uses float instructions or types.
    Floats,
    /// The module breaks a rule that state saving depends on.
    Rules(String),
    /// Saved state does not fit this module.
    State(String),
}

impl std::fmt::Display for ScriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(e) => write!(f, "script is not a valid module: {e}"),
            Self::Floats => write!(
                f,
                "script uses floating-point instructions, which are not allowed because \
                 their results can differ between machines; use integer or fixed-point math"
            ),
            Self::Rules(e) => write!(f, "script breaks a scripting rule: {e}"),
            Self::State(e) => write!(f, "saved script state does not fit the module: {e}"),
        }
    }
}

impl std::error::Error for ScriptError {}

/// Store data: what host functions can reach during a call.
struct Host {
    /// The real simulation during a call; a spare empty one between calls.
    sim: Simulation,
    queued: Vec<Command>,
    logs: Vec<String>,
    materials: usize,
    limits: StoreLimits,
}

#[derive(Clone, Copy)]
enum Entry {
    Init,
    Tick,
}

/// A loaded game script.
pub struct Script {
    engine: Engine,
    module: Module,
    linker: Linker<Host>,
    store: Store<Host>,
    memory: Option<Memory>,
    globals: Vec<Global>,
    on_init: Option<TypedFunc<(), ()>>,
    on_tick: Option<TypedFunc<(), ()>>,
    faulted: bool,
    hash: u64,
    logs: Vec<String>,
}

fn config(floats: bool) -> Config {
    let mut c = Config::default();
    c.floats(floats)
        .consume_fuel(true)
        .allow_start_fn(false)
        .wasm_multi_memory(false)
        .compilation_mode(CompilationMode::Eager);
    c
}

/// Checks the rules that make saving memory and mutable globals enough to
/// capture the whole script state.
fn check_rules(wasm: &[u8]) -> Result<(), ScriptError> {
    use wasmparser::{ExternalKind, Operator, Parser, Payload, TypeRef, ValType};
    let bad = |e: wasmparser::BinaryReaderError| ScriptError::Invalid(e.to_string());
    let mut mutable: Vec<u32> = Vec::new();
    let mut globals = 0u32;
    let mut memories = 0u32;
    let mut exported_globals: Vec<u32> = Vec::new();
    let mut memory_exported = false;
    for payload in Parser::new(0).parse_all(wasm) {
        match payload.map_err(bad)? {
            Payload::ImportSection(r) => {
                for imp in r {
                    let imp = imp.map_err(bad)?;
                    match imp.ty {
                        TypeRef::Func(_) if imp.module == "fe" => {}
                        TypeRef::Func(_) => {
                            return Err(ScriptError::Rules(format!(
                                "imports '{}.{}', but only module 'fe' is provided",
                                imp.module, imp.name
                            )));
                        }
                        _ => {
                            return Err(ScriptError::Rules(format!(
                                "imports '{}.{}', but only functions can be imported",
                                imp.module, imp.name
                            )));
                        }
                    }
                }
            }
            Payload::GlobalSection(r) => {
                for g in r {
                    let g = g.map_err(bad)?;
                    if g.ty.mutable {
                        if !matches!(g.ty.content_type, ValType::I32 | ValType::I64) {
                            return Err(ScriptError::Rules(format!(
                                "mutable global {globals} is not i32 or i64"
                            )));
                        }
                        mutable.push(globals);
                    }
                    globals += 1;
                }
            }
            Payload::MemorySection(r) => memories += r.count(),
            Payload::ExportSection(r) => {
                for e in r {
                    let e = e.map_err(bad)?;
                    match e.kind {
                        ExternalKind::Global => exported_globals.push(e.index),
                        ExternalKind::Memory if e.name == "memory" => memory_exported = true,
                        _ => {}
                    }
                }
            }
            Payload::CodeSectionEntry(body) => {
                let mut ops = body.get_operators_reader().map_err(bad)?;
                while !ops.eof() {
                    let name = match ops.read().map_err(bad)? {
                        Operator::TableSet { .. } => "table.set",
                        Operator::TableGrow { .. } => "table.grow",
                        Operator::TableFill { .. } => "table.fill",
                        Operator::TableCopy { .. } => "table.copy",
                        Operator::TableInit { .. } => "table.init",
                        Operator::ElemDrop { .. } => "elem.drop",
                        Operator::DataDrop { .. } => "data.drop",
                        _ => continue,
                    };
                    return Err(ScriptError::Rules(format!(
                        "uses {name}, which changes state that is not saved"
                    )));
                }
            }
            _ => {}
        }
    }
    if memories > 0 && !memory_exported {
        return Err(ScriptError::Rules("the module's memory must be exported as 'memory'".into()));
    }
    if let Some(g) = mutable.iter().find(|g| !exported_globals.contains(g)) {
        return Err(ScriptError::Rules(format!(
            "mutable global {g} is not exported, so its value cannot be saved; export every \
             mutable global (for Rust, link with --export=__stack_pointer)"
        )));
    }
    Ok(())
}

fn trap(msg: impl Into<String>) -> Error {
    Error::new(msg.into())
}

fn memory_of(caller: &Caller<'_, Host>) -> Result<Memory, Error> {
    caller
        .get_export("memory")
        .and_then(Extern::into_memory)
        .ok_or_else(|| trap("the script has no exported memory"))
}

fn read_text(caller: &Caller<'_, Host>, ptr: i32, len: i32) -> Result<String, Error> {
    if !(0..=MAX_TEXT).contains(&len) || ptr < 0 {
        return Err(trap(format!("text length {len} is out of range")));
    }
    let mem = memory_of(caller)?;
    let data = mem.data(caller);
    let (a, b) = (ptr as usize, ptr as usize + len as usize);
    let bytes = data.get(a..b).ok_or_else(|| Error::from(TrapCode::MemoryOutOfBounds))?;
    Ok(String::from_utf8_lossy(bytes).into_owned())
}

fn charge(caller: &mut Caller<'_, Host>, fuel: u64) -> Result<(), Error> {
    let left = caller.get_fuel()?;
    if left < fuel {
        caller.set_fuel(0)?;
        return Err(Error::from(TrapCode::OutOfFuel));
    }
    caller.set_fuel(left - fuel)?;
    Ok(())
}

fn rect(x0: i32, y0: i32, x1: i32, y1: i32) -> Option<IRect> {
    let r = IRect::new(x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1));
    let area = i64::from(r.width()) * i64::from(r.height());
    (area <= MAX_AREA).then_some(r)
}

fn queue(caller: &mut Caller<'_, Host>, cmd: Command) -> Result<(), Error> {
    let host = caller.data_mut();
    if host.queued.len() >= MAX_COMMANDS {
        return Err(trap(format!("more than {MAX_COMMANDS} commands in one call")));
    }
    host.queued.push(cmd);
    Ok(())
}

fn note(caller: &mut Caller<'_, Host>, msg: String) {
    let host = caller.data_mut();
    if host.logs.len() < MAX_LOGS {
        host.logs.push(msg);
    }
}

fn material(caller: &mut Caller<'_, Host>, mat: i32, what: &str) -> Option<MaterialId> {
    let n = caller.data().materials;
    if mat < 0 || mat as usize >= n || mat as u16 == MaterialId::BOUNDARY.0 {
        note(caller, format!("{what}: material {mat} is not valid, ignored"));
        return None;
    }
    Some(MaterialId(mat as u16))
}

fn player(id: i32) -> Option<u8> {
    u8::try_from(id).ok()
}

fn avatar_field(caller: &Caller<'_, Host>, id: i32, f: fn(&fe_engine::Avatar) -> i32) -> i32 {
    player(id)
        .and_then(|p| caller.data().sim.avatars.iter().find(|a| a.id == p))
        .map_or(i32::MIN, f)
}

fn build_linker(engine: &Engine) -> Result<Linker<Host>, ScriptError> {
    let mut l: Linker<Host> = Linker::new(engine);
    let err = |e: wasmi::errors::LinkerError| ScriptError::Invalid(e.to_string());
    l.func_wrap("fe", "tick", |c: Caller<'_, Host>| c.data().sim.tick() as i64).map_err(err)?;
    l.func_wrap("fe", "seed", |c: Caller<'_, Host>| c.data().sim.world.seed() as i64)
        .map_err(err)?;
    l.func_wrap("fe", "get_material", |c: Caller<'_, Host>, x: i32, y: i32| {
        i32::from(c.data().sim.world.get(CellPos::new(x, y)).material().0)
    })
    .map_err(err)?;
    l.func_wrap("fe", "material_id", |c: Caller<'_, Host>, ptr: i32, len: i32| {
        let name = read_text(&c, ptr, len)?;
        Ok(c.data().sim.material(&name).map_or(-1, |m| i32::from(m.0)))
    })
    .map_err(err)?;
    l.func_wrap(
        "fe",
        "count",
        |mut c: Caller<'_, Host>, x0: i32, y0: i32, x1: i32, y1: i32, mat: i32| {
            let Some(r) = rect(x0, y0, x1, y1) else { return Ok(-1) };
            let area = i64::from(r.width()) * i64::from(r.height());
            charge(&mut c, area as u64 / 16 + 1)?;
            let Ok(m) = u16::try_from(mat) else { return Ok(0) };
            Ok(c.data().sim.world.count(r, MaterialId(m)) as i32)
        },
    )
    .map_err(err)?;
    l.func_wrap("fe", "player_count", |c: Caller<'_, Host>| c.data().sim.avatars.len() as i32)
        .map_err(err)?;
    l.func_wrap("fe", "player_id", |c: Caller<'_, Host>, i: i32| {
        usize::try_from(i)
            .ok()
            .and_then(|i| c.data().sim.avatars.get(i))
            .map_or(-1, |a| i32::from(a.id))
    })
    .map_err(err)?;
    l.func_wrap("fe", "avatar_exists", |c: Caller<'_, Host>, id: i32| {
        i32::from(avatar_field(&c, id, |_| 0) != i32::MIN)
    })
    .map_err(err)?;
    l.func_wrap("fe", "avatar_x", |c: Caller<'_, Host>, id: i32| {
        avatar_field(&c, id, |a| a.x.floor())
    })
    .map_err(err)?;
    l.func_wrap("fe", "avatar_y", |c: Caller<'_, Host>, id: i32| {
        avatar_field(&c, id, |a| a.y.floor())
    })
    .map_err(err)?;
    l.func_wrap("fe", "paint", |mut c: Caller<'_, Host>, x: i32, y: i32, r: i32, mat: i32| {
        let Some(material) = material(&mut c, mat, "paint") else { return Ok(()) };
        let cmd = Command::Paint { x, y, radius: r.clamp(0, 64), material, replace: false };
        queue(&mut c, cmd)
    })
    .map_err(err)?;
    l.func_wrap(
        "fe",
        "fill",
        |mut c: Caller<'_, Host>, x0: i32, y0: i32, x1: i32, y1: i32, mat: i32| {
            let Some(material) = material(&mut c, mat, "fill") else { return Ok(()) };
            let Some(rect) = rect(x0, y0, x1, y1) else {
                note(&mut c, "fill: rectangle too large, ignored".into());
                return Ok(());
            };
            queue(&mut c, Command::FillRect { rect, material })
        },
    )
    .map_err(err)?;
    l.func_wrap("fe", "dig", |mut c: Caller<'_, Host>, x: i32, y: i32, r: i32| {
        queue(&mut c, Command::Dig { x, y, radius: r.clamp(0, 64) })
    })
    .map_err(err)?;
    l.func_wrap("fe", "explode", |mut c: Caller<'_, Host>, x: i32, y: i32, r: i32, p: i32| {
        let cmd =
            Command::Explode { x, y, radius: r.clamp(1, 48) as u8, power: p.clamp(0, 255) as u8 };
        queue(&mut c, cmd)
    })
    .map_err(err)?;
    l.func_wrap("fe", "carve", |mut c: Caller<'_, Host>, x0: i32, y0: i32, x1: i32, y1: i32| {
        let Some(rect) = rect(x0, y0, x1, y1) else {
            note(&mut c, "carve: rectangle too large, ignored".into());
            return Ok(());
        };
        queue(&mut c, Command::CarveBody { rect })
    })
    .map_err(err)?;
    l.func_wrap("fe", "add_player", |mut c: Caller<'_, Host>, id: i32, x: i32, y: i32| {
        let Some(id) = player(id) else { return Ok(()) };
        queue(&mut c, Command::AddPlayer { id, x, y })
    })
    .map_err(err)?;
    l.func_wrap("fe", "remove_player", |mut c: Caller<'_, Host>, id: i32| {
        let Some(id) = player(id) else { return Ok(()) };
        queue(&mut c, Command::RemovePlayer { id })
    })
    .map_err(err)?;
    l.func_wrap("fe", "set_tool_mask", |mut c: Caller<'_, Host>, id: i32, mask: i32| {
        if let Some(id) = player(id) {
            c.data_mut().sim.set_tool_mask(id, mask as u16);
        }
    })
    .map_err(err)?;
    l.func_wrap(
        "fe",
        "set_tool_zone",
        |mut c: Caller<'_, Host>, id: i32, x0: i32, y0: i32, x1: i32, y1: i32| {
            if let Some(id) = player(id) {
                let z = IRect::new(x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1));
                c.data_mut().sim.set_tool_zone(id, Some(z));
            }
        },
    )
    .map_err(err)?;
    l.func_wrap("fe", "clear_tool_zone", |mut c: Caller<'_, Host>, id: i32| {
        if let Some(id) = player(id) {
            c.data_mut().sim.set_tool_zone(id, None);
        }
    })
    .map_err(err)?;
    l.func_wrap("fe", "set_status", |mut c: Caller<'_, Host>, ptr: i32, len: i32| {
        let text = read_text(&c, ptr, len)?;
        c.data_mut().sim.status = text;
        Ok(())
    })
    .map_err(err)?;
    l.func_wrap("fe", "log", |mut c: Caller<'_, Host>, ptr: i32, len: i32| {
        let text = read_text(&c, ptr, len)?;
        note(&mut c, text);
        Ok(())
    })
    .map_err(err)?;
    Ok(l)
}

struct Instanced {
    store: Store<Host>,
    memory: Option<Memory>,
    globals: Vec<Global>,
    on_init: Option<TypedFunc<(), ()>>,
    on_tick: Option<TypedFunc<(), ()>>,
}

fn instantiate(
    engine: &Engine,
    module: &Module,
    linker: &Linker<Host>,
    sim: &Simulation,
) -> Result<Instanced, ScriptError> {
    let host = Host {
        sim: Simulation::new(sim.materials().clone(), fe_engine::SimConfig::default()),
        queued: Vec::new(),
        logs: Vec::new(),
        materials: sim.materials().len(),
        limits: StoreLimitsBuilder::new()
            .memory_size(MAX_MEMORY)
            .instances(1)
            .trap_on_grow_failure(false)
            .build(),
    };
    let mut store = Store::new(engine, host);
    store.limiter(|h| &mut h.limits);
    let instance: Instance = linker
        .instantiate_and_start(&mut store, module)
        .map_err(|e| ScriptError::Invalid(e.to_string()))?;
    let memory = instance.get_memory(&store, "memory");
    let mut globals = Vec::new();
    for e in instance.exports(&store) {
        if let Some(g) = e.into_global()
            && g.ty(&store).mutability().is_mut()
        {
            globals.push(g);
        }
    }
    let func = |name: &str| -> Result<Option<TypedFunc<(), ()>>, ScriptError> {
        match instance.get_func(&store, name) {
            None => Ok(None),
            Some(f) => f.typed::<(), ()>(&store).map(Some).map_err(|_| {
                ScriptError::Rules(format!("{name} must take no parameters and return nothing"))
            }),
        }
    };
    let on_init = func("on_init")?;
    let on_tick = func("on_tick")?;
    Ok(Instanced { store, memory, globals, on_init, on_tick })
}

impl Script {
    /// Loads a module (binary WebAssembly) for a simulation.
    ///
    /// Fails if the module uses floats, breaks a scripting rule, or imports
    /// something the host does not provide.
    pub fn load(wasm: &[u8], sim: &Simulation) -> Result<Self, ScriptError> {
        let engine = Engine::new(&config(false));
        let module = match Module::new(&engine, wasm) {
            Ok(m) => m,
            Err(e) => {
                // Tell float use apart from other problems: the same bytes load
                // when floats are allowed.
                let with_floats = Engine::new(&config(true));
                if Module::new(&with_floats, wasm).is_ok() {
                    return Err(ScriptError::Floats);
                }
                return Err(ScriptError::Invalid(e.to_string()));
            }
        };
        check_rules(wasm)?;
        let linker = build_linker(&engine)?;
        let inst = instantiate(&engine, &module, &linker, sim)?;
        Ok(Self {
            engine,
            module,
            linker,
            store: inst.store,
            memory: inst.memory,
            globals: inst.globals,
            on_init: inst.on_init,
            on_tick: inst.on_tick,
            faulted: false,
            hash: mod_hash(wasm),
            logs: Vec::new(),
        })
    }

    /// [`mod_hash`] of the module's bytes.
    pub const fn mod_hash(&self) -> u64 {
        self.hash
    }

    /// True once a call has trapped or run out of fuel. A faulted script is
    /// never called again; the fault is described in [`Simulation::status`].
    pub const fn faulted(&self) -> bool {
        self.faulted
    }

    /// Calls `on_init` and saves the script state. Returns the queued commands,
    /// which the caller applies now.
    pub fn init(&mut self, sim: &mut Simulation) -> Vec<Command> {
        self.call(sim, Entry::Init)
    }

    /// Calls `on_tick` and saves the script state. Returns the queued commands,
    /// which the caller applies this tick before the simulation steps.
    pub fn tick(&mut self, sim: &mut Simulation) -> Vec<Command> {
        self.call(sim, Entry::Tick)
    }

    /// Log lines written since the last call to this method.
    pub fn take_logs(&mut self) -> Vec<String> {
        std::mem::take(&mut self.logs)
    }

    fn call(&mut self, sim: &mut Simulation, entry: Entry) -> Vec<Command> {
        let (func, fuel, name) = match entry {
            Entry::Init => (self.on_init, INIT_FUEL, "on_init"),
            Entry::Tick => (self.on_tick, TICK_FUEL, "on_tick"),
        };
        let Some(func) = func.filter(|_| !self.faulted) else {
            if sim.script_state.is_empty() {
                sim.script_state = self.save();
            }
            return Vec::new();
        };
        // Host functions reach the simulation through the store, so lend it
        // there for the call and take it back after.
        std::mem::swap(sim, &mut self.store.data_mut().sim);
        let result = self.store.set_fuel(fuel).and_then(|()| func.call(&mut self.store, ()));
        std::mem::swap(sim, &mut self.store.data_mut().sim);
        let host = self.store.data_mut();
        let queued = std::mem::take(&mut host.queued);
        self.logs.append(&mut host.logs);
        let out = match result {
            Ok(()) => queued,
            Err(e) => {
                self.faulted = true;
                sim.status = format!("script fault in {name} at tick {}: {e}", sim.tick());
                self.logs.push(sim.status.clone());
                Vec::new()
            }
        };
        sim.script_state = self.save();
        out
    }

    /// The script state: fault flag, mutable globals, and linear memory.
    pub fn save(&self) -> Vec<u8> {
        let mem = self.memory.map_or(&[][..], |m| m.data(&self.store));
        let mut out = Vec::with_capacity(16 + self.globals.len() * 9 + mem.len());
        out.extend_from_slice(&STATE_MAGIC);
        out.push(u8::from(self.faulted));
        out.extend_from_slice(&(self.globals.len() as u32).to_le_bytes());
        for g in &self.globals {
            let (tag, v) = match g.get(&self.store) {
                Val::I64(v) => (1u8, v as u64),
                Val::I32(v) => (0u8, u64::from(v as u32)),
                _ => (2u8, 0),
            };
            out.push(tag);
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&(mem.len() as u32).to_le_bytes());
        out.extend_from_slice(mem);
        out
    }

    /// Replaces the script's state with [`Simulation::script_state`], as saved by
    /// another peer or an earlier run, so it continues bit for bit from there.
    /// An empty state means the script has not run yet: it is reset to its
    /// freshly loaded state.
    pub fn restore(&mut self, sim: &Simulation) -> Result<(), ScriptError> {
        let mut inst = instantiate(&self.engine, &self.module, &self.linker, sim)?;
        let state = &sim.script_state;
        let mut faulted = false;
        if !state.is_empty() {
            let bad = |what: &str| ScriptError::State(what.to_string());
            let mut r = Reader(state);
            if r.take(4)? != STATE_MAGIC {
                return Err(bad("missing header"));
            }
            faulted = r.take(1)?[0] != 0;
            let n = r.u32()? as usize;
            if n != inst.globals.len() {
                return Err(bad("global count differs"));
            }
            for g in &inst.globals {
                let tag = r.take(1)?[0];
                let v = u64::from_le_bytes(r.array()?);
                let val = match tag {
                    0 => Val::I32(v as u32 as i32),
                    1 => Val::I64(v as i64),
                    _ => return Err(bad("unknown global type")),
                };
                g.set(&mut inst.store, val).map_err(|e| ScriptError::State(e.to_string()))?;
            }
            let len = r.u32()? as usize;
            let bytes = r.take(len)?;
            match inst.memory {
                Some(m) => {
                    let have = m.data(&inst.store).len();
                    if len < have || !len.is_multiple_of(PAGE) {
                        return Err(bad("memory size does not fit"));
                    }
                    if len > have {
                        m.grow(&mut inst.store, ((len - have) / PAGE) as u64)
                            .map_err(|e| ScriptError::State(e.to_string()))?;
                    }
                    m.data_mut(&mut inst.store).copy_from_slice(bytes);
                }
                None if len == 0 => {}
                None => return Err(bad("state has memory but the module has none")),
            }
        }
        self.store = inst.store;
        self.memory = inst.memory;
        self.globals = inst.globals;
        self.on_init = inst.on_init;
        self.on_tick = inst.on_tick;
        self.faulted = faulted;
        Ok(())
    }
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], ScriptError> {
        if self.0.len() < n {
            return Err(ScriptError::State("state is truncated".into()));
        }
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], ScriptError> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32, ScriptError> {
        Ok(u32::from_le_bytes(self.array()?))
    }
}

#[cfg(test)]
mod tests;
