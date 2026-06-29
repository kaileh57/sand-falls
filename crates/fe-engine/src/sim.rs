use crate::command::Command;
use crate::worldgen::WorldGen;
use fe_core::{Cell, CellPos, ChunkPos, IRect, LocalRect, MaterialId, StableHasher, isqrt};
use fe_materials::{MaterialSet, State};
use fe_physics::Physics;
use fe_sim::{
    Avatar, Blast, Event, Particles, PlayerInput, SUPPORT_RETRIES, SUPPORT_SEARCH_LIMIT,
    SupportSeed, World,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

/// Tool numbers carried in [`PlayerInput::tool`].
pub mod tools {
    /// No tool.
    pub const NONE: u8 = 0;
    /// Removes cells in a disc; `param` is the radius (default 4).
    pub const DIG: u8 = 1;
    /// Paints a material; `param` is the material id.
    pub const PAINT: u8 = 2;
    /// Explodes once per press; `param` is the radius (default 10).
    pub const EXPLODE: u8 = 3;
    /// Heats a disc.
    pub const HEAT: u8 = 4;
    /// Cools a disc.
    pub const FREEZE: u8 = 5;
    /// Drops a wooden crate once per press.
    pub const CRATE: u8 = 6;

    /// Tool-mask bit for a tool number.
    pub const fn bit(tool: u8) -> u16 {
        if tool < 16 { 1 << tool } else { 0 }
    }

    /// A mask that allows every tool. Players have it unless a game sets another.
    pub const ALL: u16 = u16::MAX;
}

/// Chunk streaming around player characters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Chunks kept loaded around each character, in chunks.
    pub radius: i32,
    /// Extra distance before a loaded chunk is unloaded, in chunks.
    pub margin: i32,
}

/// Settings for a new simulation.
#[derive(Clone, Debug)]
pub struct SimConfig {
    /// World seed.
    pub seed: u64,
    /// Worker threads for cell passes. Results do not depend on it.
    pub threads: usize,
    /// How new chunks are filled.
    pub worldgen: WorldGen,
    /// Streaming, or `None` for a fixed world.
    pub stream: Option<StreamConfig>,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self { seed: 1, threads: 1, worldgen: WorldGen::Empty, stream: None }
    }
}

/// Wall-clock time spent in each phase of the last tick, in microseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseTimes {
    /// Commands and tools.
    pub commands: u64,
    /// Cell passes.
    pub cells: u64,
    /// Explosions.
    pub explosions: u64,
    /// Particles.
    pub particles: u64,
    /// Support checks and detachment.
    pub support: u64,
    /// Body sync, stamping and the Rapier step.
    pub physics: u64,
    /// Characters.
    pub avatars: u64,
    /// Chunk streaming.
    pub streaming: u64,
    /// The whole tick.
    pub total: u64,
}

/// Counters from the last tick.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TickReport {
    /// Tick number that ran.
    pub tick: u64,
    /// Chunks with work.
    pub active_chunks: u32,
    /// Loaded chunks.
    pub loaded_chunks: u32,
    /// Cells visited by the rules.
    pub visited_cells: u64,
    /// Explosions processed.
    pub explosions: u32,
    /// Particles in flight after the tick.
    pub particles: u32,
    /// Rigid bodies after the tick.
    pub bodies: u32,
    /// Bodies created this tick.
    pub bodies_created: u32,
    /// Structure groups that lost support this tick.
    pub detached: u32,
    /// Joints broken this tick.
    pub joints_broken: u32,
    /// Phase timings.
    pub times: PhaseTimes,
}

/// A state hash split by subsystem.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HashParts {
    /// Cells, seed, tick and materials.
    pub cells: u64,
    /// Particles in flight.
    pub particles: u64,
    /// Rigid bodies and joints.
    pub bodies: u64,
    /// Characters.
    pub avatars: u64,
    /// Game state: tool masks and zones, status text, and script state.
    pub game: u64,
    /// The full [`Simulation::hash`].
    pub total: u64,
}

/// One chunk's cells and scheduling state, as sent by network repair.
///
/// The scheduling state is part of the deterministic state: visiting a cell
/// rewrites its parity flag, so two copies of a chunk with the same cells but
/// different pending rects drift apart.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkState {
    /// Chunk position.
    pub pos: ChunkPos,
    /// The 4,096 cells as packed bits, row-major.
    pub cells: Vec<u32>,
    /// Cells visited on the tick that just ran.
    pub dirty: LocalRect,
    /// Cells marked for the next tick.
    pub next: LocalRect,
    /// A static cell changed since terrain colliders were last rebuilt.
    pub solid_changed: bool,
}

/// The whole deterministic simulation.
pub struct Simulation {
    /// Cells and chunks.
    pub world: World,
    /// Cells in flight.
    pub particles: Particles,
    /// Rigid bodies.
    pub physics: Physics,
    /// Player characters, sorted by id.
    pub avatars: Vec<Avatar>,
    pub(crate) support_seeds: Vec<SupportSeed>,
    pub(crate) last_inputs: BTreeMap<u8, PlayerInput>,
    pub(crate) worldgen: WorldGen,
    pub(crate) stream: Option<StreamConfig>,
    pub(crate) stored: BTreeMap<ChunkPos, Vec<u32>>,
    pub(crate) tool_masks: BTreeMap<u8, u16>,
    pub(crate) tool_zones: BTreeMap<u8, IRect>,
    /// Text a game shows in its HUD, such as the phase and score. Set by game scripts.
    pub status: String,
    /// Saved state of the game script, kept here so hashes and snapshots cover it.
    /// The engine never reads it.
    pub script_state: Vec<u8>,
    report: TickReport,
}

/// Explosions set off on one tick, merged where they overlap.
#[derive(Clone, Copy)]
struct Pending {
    /// Sum of the merged cells' positions, for their center.
    sum: (i64, i64),
    cells: i64,
    radius: u8,
    power: u8,
}

impl Pending {
    fn center(&self) -> CellPos {
        let n = self.cells.max(1);
        CellPos::new(self.sum.0.div_euclid(n) as i32, self.sum.1.div_euclid(n) as i32)
    }

    fn overlaps(&self, at: CellPos, radius: u8) -> bool {
        let c = self.center();
        let (dx, dy) = (i64::from(c.x - at.x), i64::from(c.y - at.y));
        let reach = i64::from(self.radius.max(radius)) / 2;
        dx * dx + dy * dy <= reach * reach
    }

    /// Grows by another charge: a block of explosive cells makes one large
    /// blast at its middle instead of many overlapping small ones.
    fn absorb(&mut self, o: &Self) {
        self.sum = (self.sum.0 + o.sum.0, self.sum.1 + o.sum.1);
        self.cells += o.cells;
        let grow = u8::try_from(o.cells).unwrap_or(u8::MAX);
        self.radius = self.radius.max(o.radius).saturating_add(grow).min(48);
        self.power = self.power.max(o.power).saturating_add(grow.saturating_mul(6));
    }
}

/// Adds an explosion to this tick's list, merging it into one that overlaps.
fn merge_blast(list: &mut Vec<Pending>, pos: CellPos, radius: u8, power: u8) {
    let new = Pending { sum: (i64::from(pos.x), i64::from(pos.y)), cells: 1, radius, power };
    match list.iter_mut().find(|b| b.overlaps(pos, radius)) {
        Some(b) => b.absorb(&new),
        None => list.push(new),
    }
}

/// Merges blasts that grew to overlap each other, until none do.
fn settle_blasts(list: &mut Vec<Pending>) {
    let mut i = 0;
    while i < list.len() {
        let Some(j) =
            (i + 1..list.len()).find(|&j| list[i].overlaps(list[j].center(), list[j].radius))
        else {
            i += 1;
            continue;
        };
        let o = list.remove(j);
        list[i].absorb(&o);
        i = 0;
    }
}

fn us(t: Instant) -> u64 {
    t.elapsed().as_micros() as u64
}

impl Simulation {
    /// Creates an empty simulation.
    pub fn new(mats: Arc<MaterialSet>, cfg: SimConfig) -> Self {
        Self {
            world: World::new(mats, cfg.seed, cfg.threads),
            particles: Particles::default(),
            physics: Physics::new(),
            avatars: Vec::new(),
            support_seeds: Vec::new(),
            last_inputs: BTreeMap::new(),
            worldgen: cfg.worldgen,
            stream: cfg.stream,
            stored: BTreeMap::new(),
            tool_masks: BTreeMap::new(),
            tool_zones: BTreeMap::new(),
            status: String::new(),
            script_state: Vec::new(),
            report: TickReport::default(),
        }
    }

    /// The tools a player may use, one bit per tool number ([`tools::bit`]).
    pub fn tool_mask(&self, player: u8) -> u16 {
        self.tool_masks.get(&player).copied().unwrap_or(tools::ALL)
    }

    /// Sets the tools a player may use. [`tools::ALL`] restores the default.
    pub fn set_tool_mask(&mut self, player: u8, mask: u16) {
        if mask == tools::ALL {
            self.tool_masks.remove(&player);
        } else {
            self.tool_masks.insert(player, mask);
        }
    }

    /// The area a player's tools are limited to, if any.
    pub fn tool_zone(&self, player: u8) -> Option<IRect> {
        self.tool_zones.get(&player).copied()
    }

    /// Limits where a player's tools take effect: a tool aimed outside the zone
    /// does nothing. `None` removes the limit.
    pub fn set_tool_zone(&mut self, player: u8, zone: Option<IRect>) {
        match zone {
            Some(z) => {
                self.tool_zones.insert(player, z);
            }
            None => {
                self.tool_zones.remove(&player);
            }
        }
    }

    /// The material set.
    pub fn materials(&self) -> &Arc<MaterialSet> {
        self.world.materials()
    }

    /// Looks up a material id by name.
    pub fn material(&self, name: &str) -> Option<MaterialId> {
        self.world.materials().id(name)
    }

    /// The current tick.
    pub fn tick(&self) -> u64 {
        self.world.tick()
    }

    /// Counters and timings from the last tick.
    pub const fn report(&self) -> &TickReport {
        &self.report
    }

    /// The world generator.
    pub const fn worldgen(&self) -> &WorldGen {
        &self.worldgen
    }

    /// Loads (generating if needed) every chunk overlapping a rectangle.
    pub fn load_rect(&mut self, rect: IRect) {
        if rect.is_empty() {
            return;
        }
        let (a, b) = rect.chunk_range();
        for cy in a.y..b.y {
            for cx in a.x..b.x {
                self.load_chunk(ChunkPos::new(cx, cy));
            }
        }
    }

    fn load_chunk(&mut self, pos: ChunkPos) {
        if self.world.grid.contains(pos) {
            return;
        }
        let cells = match self.stored.remove(&pos) {
            Some(bits) => bits.into_iter().map(Cell::from_bits).collect(),
            None => self.worldgen.generate(self.world.materials(), self.world.seed(), pos),
        };
        let mut chunk = fe_sim::Chunk::new(pos, Cell::AIR);
        chunk.load_cells(&cells);
        self.world.grid.insert(chunk);
        // Wake each neighbor's whole facing edge so material can flow across the new border.
        let last = (fe_core::CHUNK_SIZE - 1) as u8;
        for (dx, dy, a, b) in [
            (-1, 0, (last, 0), (last, last)),
            (1, 0, (0, 0), (0, last)),
            (0, -1, (0, last), (last, last)),
            (0, 1, (0, 0), (last, 0)),
        ] {
            if let Some(n) = self.world.grid.chunk(ChunkPos::new(pos.x + dx, pos.y + dy)) {
                n.mark(a.0, a.1);
                n.mark(b.0, b.1);
            }
        }
    }

    fn unload_chunk(&mut self, pos: ChunkPos) {
        if let Some(c) = self.world.grid.remove(pos) {
            self.stored.insert(pos, c.to_vec().into_iter().map(Cell::to_bits).collect());
        }
    }

    /// Applies one command now. An explosion it sets off pushes rigid bodies at once.
    pub fn apply(&mut self, cmd: &Command) {
        let mut blasts = Vec::new();
        self.apply_with(cmd, &mut blasts);
        self.physics.apply_blasts(&blasts);
    }

    /// Applies one command, adding any explosion's blast to `blasts` so the
    /// caller decides when it pushes bodies.
    fn apply_with(&mut self, cmd: &Command, blasts: &mut Vec<Blast>) {
        match *cmd {
            Command::Paint { x, y, radius, material, replace } => {
                let mut painted = Vec::new();
                self.disc(x, y, radius, |sim, p| {
                    let c = sim.world.get(p);
                    let m = sim.world.materials().get(c.material());
                    if c.material() == MaterialId::BOUNDARY || c.has(Cell::FLAG_IN_BODY) {
                        return;
                    }
                    if c.is_air() || (replace && !m.indestructible) {
                        let n = sim.world.make_cell(material, p);
                        if let Some(Event::SupportLost(q)) = sim.world.set(p, n) {
                            sim.seed_support(q);
                        }
                        painted.push(p);
                    }
                });
                self.check_painted_support(&painted, material);
            }
            Command::FillRect { rect, material } => {
                self.load_rect(rect);
                for e in self.world.fill_rect(rect, material) {
                    if let Event::SupportLost(q) = e {
                        self.seed_support(q);
                    }
                }
            }
            Command::Dig { x, y, radius } => {
                self.disc(x, y, radius, |sim, p| {
                    let c = sim.world.get(p);
                    if c.is_air() || c.material() == MaterialId::BOUNDARY {
                        return;
                    }
                    if sim.world.materials().get(c.material()).indestructible {
                        return;
                    }
                    if let Some(Event::SupportLost(q)) = sim.world.set(p, Cell::AIR) {
                        sim.seed_support(q);
                    }
                });
            }
            Command::Explode { x, y, radius, power } => {
                self.run_explosion(CellPos::new(x, y), radius, power, blasts);
            }
            Command::Heat { x, y, radius, delta } => {
                self.disc(x, y, radius, |sim, p| {
                    let mut c = sim.world.get(p);
                    if c.material() == MaterialId::BOUNDARY {
                        return;
                    }
                    c.temp = (i32::from(c.temp) + i32::from(delta)).clamp(0, 255) as u8;
                    sim.world.grid.set(p, c);
                });
            }
            Command::CarveBody { rect } => {
                let mut lost = Vec::new();
                self.physics.carve_body(&mut self.world, rect, &mut lost);
                for q in lost {
                    self.seed_support(q);
                }
            }
            Command::Impulse { body, vx, vy } => {
                self.physics.push_body(body, vx, vy);
            }
            Command::Joint { a, b, x, y, kind, break_force } => {
                self.physics.add_joint(a, b, CellPos::new(x, y), kind, break_force);
            }
            Command::Pin { body, x, y, break_force } => {
                self.physics.pin_body(body, CellPos::new(x, y), break_force);
            }
            Command::AddPlayer { id, x, y } => {
                self.avatars.retain(|a| a.id != id);
                self.avatars.push(Avatar::new(id, CellPos::new(x, y)));
                self.avatars.sort_by_key(|a| a.id);
            }
            Command::RemovePlayer { id } => {
                self.avatars.retain(|a| a.id != id);
                self.last_inputs.remove(&id);
            }
        }
    }

    /// Removes the listed cells from the world and makes them one rigid body.
    /// Returns the body id, or `None` if too few cells can be part of a body.
    pub fn carve_cells(&mut self, cells: &[CellPos]) -> Option<u32> {
        let picked: Vec<(CellPos, Cell)> = cells
            .iter()
            .map(|&p| (p, self.world.get(p)))
            .filter(|&(_, c)| fe_physics::is_body_material(&self.world, c))
            .collect();
        let mut lost = Vec::new();
        let id = self.physics.carve_cells(&mut self.world, &picked, &mut lost);
        for q in lost {
            self.seed_support(q);
        }
        id
    }

    fn disc(&mut self, x: i32, y: i32, radius: i32, mut f: impl FnMut(&mut Self, CellPos)) {
        let r = radius.max(0);
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r {
                    f(self, CellPos::new(x + dx, y + dy));
                }
            }
        }
    }

    /// Structure painted in mid-air, touching no other structure, has nothing
    /// holding it up: queue a support check so it falls. Paint that touches
    /// existing structure is assumed to hang on it, which keeps the check cheap.
    fn check_painted_support(&mut self, painted: &[CellPos], material: MaterialId) {
        let m = self.world.materials().get(material);
        if painted.is_empty() || m.state != State::Static || !m.structural {
            return;
        }
        let mut sorted = painted.to_vec();
        sorted.sort_unstable();
        for &p in painted {
            for (dx, dy) in [(0, 1), (1, 0), (0, -1), (-1, 0)] {
                let q = CellPos::new(p.x + dx, p.y + dy);
                if sorted.binary_search(&q).is_ok() {
                    continue;
                }
                let c = self.world.get(q);
                if c.material() == MaterialId::BOUNDARY || c.has(Cell::FLAG_IN_BODY) {
                    return;
                }
                let cm = self.world.materials().get(c.material());
                if !c.is_air() && cm.state == State::Static && (cm.structural || cm.anchored) {
                    return;
                }
            }
        }
        // The seed's neighbors are painted cells, so the check starts inside the blob.
        for &p in painted {
            self.seed_support(p);
        }
    }

    fn seed_support(&mut self, p: CellPos) {
        self.support_seeds.push(SupportSeed { pos: p, retries: SUPPORT_RETRIES });
    }

    fn run_explosion(&mut self, pos: CellPos, radius: u8, power: u8, blasts: &mut Vec<Blast>) {
        let mut blast = Blast::default();
        let events =
            fe_sim::explode(&mut self.world, &mut self.particles, pos, radius, power, &mut blast);
        for e in events {
            if let Event::SupportLost(q) = e {
                self.seed_support(q);
            }
        }
        blasts.push(blast);
    }

    fn tool_commands(
        &self,
        avatar: &Avatar,
        input: &PlayerInput,
        prev: &PlayerInput,
    ) -> Vec<Command> {
        if !input.use_tool || self.tool_mask(avatar.id) & tools::bit(input.tool) == 0 {
            return Vec::new();
        }
        let t = avatar.aim_target(input);
        if self.tool_zone(avatar.id).is_some_and(|z| !z.contains(t)) {
            return Vec::new();
        }
        let pressed = !prev.use_tool || prev.tool != input.tool;
        match input.tool {
            tools::DIG => {
                vec![Command::Dig {
                    x: t.x,
                    y: t.y,
                    radius: i32::from(input.param.clamp(1, 24)).max(4),
                }]
            }
            tools::PAINT => vec![Command::Paint {
                x: t.x,
                y: t.y,
                radius: 3,
                material: MaterialId(input.param),
                replace: false,
            }],
            tools::EXPLODE if pressed => vec![Command::Explode {
                x: t.x,
                y: t.y,
                radius: if input.param == 0 { 10 } else { input.param.min(40) as u8 },
                power: 180,
            }],
            tools::HEAT => vec![Command::Heat { x: t.x, y: t.y, radius: 4, delta: 6 }],
            tools::FREEZE => vec![Command::Heat { x: t.x, y: t.y, radius: 4, delta: -6 }],
            _ => Vec::new(),
        }
    }

    /// The crate tool: fills the air in a 12x12 square with wood and carves it out
    /// as a body. It never overwrites other cells and never loads chunks, so it
    /// cannot cut indestructible terrain or grow a fixed world.
    fn drop_crate(&mut self, avatar: &Avatar, input: &PlayerInput, prev: &PlayerInput) {
        let pressed = !prev.use_tool || prev.tool != input.tool;
        if !input.use_tool || input.tool != tools::CRATE || !pressed {
            return;
        }
        let Some(wood) = self.material("wood") else { return };
        let t = avatar.aim_target(input);
        let rect = IRect::from_size(t.x - 6, t.y - 6, 12, 12);
        let (a, b) = rect.chunk_range();
        let loaded = (a.y..b.y)
            .all(|cy| (a.x..b.x).all(|cx| self.world.grid.contains(ChunkPos::new(cx, cy))));
        if !loaded {
            return;
        }
        for y in rect.min.y..rect.max.y {
            for x in rect.min.x..rect.max.x {
                let p = CellPos::new(x, y);
                if self.world.get(p).is_air() {
                    let c = self.world.make_cell(wood, p);
                    let _ = self.world.set(p, c);
                }
            }
        }
        let mut lost = Vec::new();
        self.physics.carve_body(&mut self.world, rect, &mut lost);
        for q in lost {
            self.seed_support(q);
        }
    }

    /// Runs one tick.
    ///
    /// `inputs` are per-player inputs for this tick; players without an entry act
    /// as if idle. `commands` run first, in order.
    pub fn step(&mut self, inputs: &BTreeMap<u8, PlayerInput>, commands: &[Command]) {
        let t_total = Instant::now();
        let mut times = PhaseTimes::default();
        let tick = self.world.tick();
        self.physics.begin_tick();

        let t = Instant::now();
        // Blasts from commands and tools push bodies along with the cell explosions below.
        let mut blasts = Vec::new();
        for c in commands {
            self.apply_with(c, &mut blasts);
        }
        let avatars = self.avatars.clone();
        for a in &avatars {
            let input = inputs.get(&a.id).copied().unwrap_or_default();
            let prev = self.last_inputs.get(&a.id).copied().unwrap_or_default();
            for c in self.tool_commands(a, &input, &prev) {
                self.apply_with(&c, &mut blasts);
            }
            self.drop_crate(a, &input, &prev);
        }
        times.commands = us(t);

        let t = Instant::now();
        let events = self.world.step_cells();
        let cell_stats = self.world.stats();
        times.cells = us(t);

        let t = Instant::now();
        let mut pending: Vec<Pending> = Vec::new();
        for e in &events {
            match *e {
                Event::Explosion { pos, radius, power } => {
                    merge_blast(&mut pending, pos, radius, power);
                }
                Event::SupportLost(p) => self.seed_support(p),
            }
        }
        settle_blasts(&mut pending);
        let explosions = pending.len() as u32;
        for b in pending {
            self.run_explosion(b.center(), b.radius, b.power, &mut blasts);
        }
        times.explosions = us(t);

        let t = Instant::now();
        self.particles.step(&mut self.world);
        times.particles = us(t);

        let t = Instant::now();
        self.physics.sync(&mut self.world, &mut self.particles);
        times.physics = us(t);

        let t = Instant::now();
        let mut seeds = std::mem::take(&mut self.support_seeds);
        seeds.sort();
        seeds.dedup_by_key(|s| s.pos);
        let (detached, retry) = fe_sim::check_support(&self.world, &seeds, SUPPORT_SEARCH_LIMIT);
        self.support_seeds = retry;
        let detached_count = detached.len() as u32;
        self.physics.detach(&mut self.world, &mut self.particles, &detached);
        times.support = us(t);

        let t = Instant::now();
        self.physics.unstamp(&mut self.world);
        self.physics.apply_blasts(&blasts);
        self.physics.step(&mut self.world, &mut self.particles);
        self.physics.stamp(&mut self.world);
        times.physics += us(t);

        let t = Instant::now();
        for a in &mut self.avatars {
            let input = inputs.get(&a.id).copied().unwrap_or_default();
            a.step(&self.world, &input);
        }
        for a in &self.avatars {
            self.last_inputs.insert(a.id, inputs.get(&a.id).copied().unwrap_or_default());
        }
        times.avatars = us(t);

        let t = Instant::now();
        self.stream_chunks();
        times.streaming = us(t);

        self.world.finish_tick();
        times.total = us(t_total);
        let ps = self.physics.stats();
        self.report = TickReport {
            tick,
            active_chunks: cell_stats.active_chunks,
            loaded_chunks: self.world.grid.len() as u32,
            visited_cells: cell_stats.visited_cells,
            explosions,
            particles: self.particles.len() as u32,
            bodies: ps.bodies,
            bodies_created: ps.created,
            detached: detached_count,
            joints_broken: ps.joints_broken,
            times,
        };
    }

    /// Runs `n` ticks with no input.
    pub fn run(&mut self, n: u64) {
        let empty = BTreeMap::new();
        for _ in 0..n {
            self.step(&empty, &[]);
        }
    }

    fn stream_chunks(&mut self) {
        let Some(cfg) = self.stream else { return };
        if self.avatars.is_empty() || !self.world.tick().is_multiple_of(4) {
            return;
        }
        let centers: Vec<ChunkPos> = self.avatars.iter().map(|a| a.center().chunk()).collect();
        for c in &centers {
            for dy in -cfg.radius..=cfg.radius {
                for dx in -cfg.radius..=cfg.radius {
                    self.load_chunk(ChunkPos::new(c.x + dx, c.y + dy));
                }
            }
        }
        let keep = cfg.radius + cfg.margin;
        let far: Vec<ChunkPos> = self
            .world
            .grid
            .positions()
            .filter(|p| {
                centers.iter().all(|c| (p.x - c.x).abs() > keep || (p.y - c.y).abs() > keep)
            })
            // A chunk under a rigid body stays loaded: storing its stamped cells
            // would destroy the body and leave cells no body owns.
            .filter(|p| {
                self.world.grid.chunk(*p).is_some_and(|c| {
                    (0..fe_core::CHUNK_AREA).all(|i| !c.get(i).has(Cell::FLAG_IN_BODY))
                })
            })
            .collect();
        for p in far {
            self.unload_chunk(p);
        }
    }

    /// Hash of the whole simulation state.
    pub fn hash(&self) -> u64 {
        let mut h = StableHasher::new();
        h.write_u64(self.world.hash());
        self.particles.hash_into(&mut h);
        self.physics.hash_into(&mut h);
        h.write_u64(self.avatars.len() as u64);
        for a in &self.avatars {
            a.hash_into(&mut h);
        }
        for s in &self.support_seeds {
            h.write_i32(s.pos.x);
            h.write_i32(s.pos.y);
            h.write_u32(u32::from(s.retries));
        }
        for (id, i) in &self.last_inputs {
            h.write_u32(u32::from(*id));
            h.write_u32(u32::from(i.use_tool) | u32::from(i.tool) << 8);
        }
        h.write_u64(self.stored.len() as u64);
        // Game state only enters the hash when a game set some, so scenes without a
        // script keep the hashes they had before game state existed.
        if self.has_game_state() {
            h.write_u64(self.game_hash());
        }
        h.finish()
    }

    fn has_game_state(&self) -> bool {
        !self.tool_masks.is_empty()
            || !self.tool_zones.is_empty()
            || !self.status.is_empty()
            || !self.script_state.is_empty()
    }

    fn game_hash(&self) -> u64 {
        let mut h = StableHasher::new();
        h.write_u64(self.tool_masks.len() as u64);
        for (p, m) in &self.tool_masks {
            h.write_u32(u32::from(*p) | u32::from(*m) << 8);
        }
        h.write_u64(self.tool_zones.len() as u64);
        for (p, z) in &self.tool_zones {
            h.write_u32(u32::from(*p));
            h.write_i32(z.min.x);
            h.write_i32(z.min.y);
            h.write_i32(z.max.x);
            h.write_i32(z.max.y);
        }
        h.write_bytes(self.status.as_bytes());
        h.write_bytes(&self.script_state);
        h.finish()
    }

    /// The state hash split by subsystem, so a mismatch says what diverged.
    pub fn hash_parts(&self) -> HashParts {
        let mut p = StableHasher::new();
        self.particles.hash_into(&mut p);
        let mut b = StableHasher::new();
        self.physics.hash_into(&mut b);
        let mut a = StableHasher::new();
        for av in &self.avatars {
            av.hash_into(&mut a);
        }
        HashParts {
            cells: self.world.hash(),
            particles: p.finish(),
            bodies: b.finish(),
            avatars: a.finish(),
            game: if self.has_game_state() { self.game_hash() } else { 0 },
            total: self.hash(),
        }
    }

    /// Replaces chunks with copies from another run of the same simulation, used by
    /// network repair. Restores each chunk's scheduling state along with its cells.
    pub fn replace_chunks(&mut self, chunks: &[ChunkState]) {
        for c in chunks {
            if c.cells.len() != fe_core::CHUNK_AREA {
                continue;
            }
            let cells: Vec<Cell> = c.cells.iter().copied().map(Cell::from_bits).collect();
            let mut chunk = fe_sim::Chunk::new(c.pos, Cell::AIR);
            chunk.load_cells(&cells);
            chunk.set_schedule(c.dirty, c.next);
            chunk.set_solid_changed(c.solid_changed);
            self.world.grid.insert(chunk);
        }
    }

    /// A loaded chunk's cells and scheduling state, for [`Simulation::replace_chunks`].
    pub fn chunk_state(&self, pos: ChunkPos) -> Option<ChunkState> {
        self.world.grid.chunk(pos).map(|c| ChunkState {
            pos,
            cells: c.to_vec().into_iter().map(Cell::to_bits).collect(),
            dirty: c.dirty(),
            next: c.pending(),
            solid_changed: c.solid_changed(),
        })
    }

    /// Packed cells of a loaded chunk.
    pub fn chunk_bits(&self, pos: ChunkPos) -> Option<Vec<u32>> {
        self.world.grid.chunk(pos).map(|c| c.to_vec().into_iter().map(Cell::to_bits).collect())
    }

    /// Per-chunk hashes in canonical order, for finding where two runs differ.
    pub fn chunk_hashes(&self) -> Vec<(ChunkPos, u64)> {
        self.world.grid.chunks().map(|c| (c.pos(), fe_sim::Grid::chunk_hash(c))).collect()
    }

    /// Distance in cells from a position to the nearest character, if any.
    pub fn nearest_avatar(&self, p: CellPos) -> Option<(u8, i32)> {
        self.avatars
            .iter()
            .map(|a| {
                let c = a.center();
                let (dx, dy) = (i64::from(c.x - p.x), i64::from(c.y - p.y));
                (a.id, isqrt((dx * dx + dy * dy) as u64) as i32)
            })
            .min_by_key(|&(id, d)| (d, id))
    }

    /// Renders a rectangle as text, one character per cell, using the first letter
    /// of each material name (`.` for air, `#` for the boundary, `@` for body cells).
    pub fn ascii(&self, rect: IRect) -> String {
        let mut s = String::new();
        for y in rect.min.y..rect.max.y {
            for x in rect.min.x..rect.max.x {
                let c = self.world.get(CellPos::new(x, y));
                let ch = if c.is_air() {
                    '.'
                } else if c.material() == MaterialId::BOUNDARY {
                    '#'
                } else if c.has(Cell::FLAG_IN_BODY) {
                    '@'
                } else {
                    let m = self.world.materials().get(c.material());
                    let first = m.name.chars().next().unwrap_or('?');
                    if m.state == State::Static { first.to_ascii_uppercase() } else { first }
                };
                s.push(ch);
            }
            s.push('\n');
        }
        s
    }
}
