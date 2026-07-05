//! Running scenes tick by tick, checking expectations, and recording replays.

use crate::build::{Resolved, apply_now, build, resolve};
use crate::parse::{Check, Expect, Scene, SceneError, When};
use fe_core::MaterialId;
use fe_engine::{Command, PlayerInput, Simulation};
use fe_materials::MaterialSet;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Inputs and commands for one tick.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Frame {
    /// Tick the frame applies to.
    pub tick: u64,
    /// Per-player input.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub inputs: BTreeMap<u8, PlayerInput>,
    /// Commands, applied before the tick in order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub commands: Vec<Command>,
    /// Commands applied between ticks with [`Runner::apply_command`], before
    /// anything else in this tick.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pre: Vec<Command>,
}

/// The outcome of one expectation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckResult {
    /// Scene line of the expectation.
    pub line: usize,
    /// Tick it was checked after.
    pub after: u64,
    /// Whether it held.
    pub passed: bool,
    /// What was checked and what was found.
    pub detail: String,
    /// For hash checks, the hash found.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub found_hash: Option<u64>,
}

/// Summary of a full scene run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunSummary {
    /// Every expectation held.
    pub passed: bool,
    /// Ticks run.
    pub ticks: u64,
    /// Final state hash.
    pub hash: u64,
    /// All expectation results in check order.
    pub checks: Vec<CheckResult>,
}

struct TimedResolved {
    when: When,
    action: Action,
}

enum Action {
    Commands(Vec<Command>),
    Input(u8, PlayerInput),
    Other(crate::parse::Op, usize),
}

/// Runs a scene one tick at a time.
pub struct Runner {
    /// The scene.
    pub scene: Scene,
    /// The simulation.
    pub sim: Simulation,
    mats: Arc<MaterialSet>,
    timeline: Vec<TimedResolved>,
    /// Frames to merge in on top of the scene timeline, keyed by tick.
    pub extra: BTreeMap<u64, Frame>,
    recorded: Vec<Frame>,
    pre: Vec<Command>,
    results: Vec<CheckResult>,
    /// Per expectation, whether it has been checked.
    checked: Vec<bool>,
    script: Option<fe_script::Script>,
}

impl Runner {
    /// Builds the scene and prepares its timeline.
    pub fn new(scene: Scene, mats: Arc<MaterialSet>, threads: usize) -> Result<Self, SceneError> {
        let mut sim = build(&scene, &mats, threads)?;
        let script = match (&scene.script, &scene.script_wasm) {
            (None, _) => None,
            (Some((path, line)), None) => {
                return Err(SceneError::new(
                    *line,
                    format!("script {path} is not loaded (use fe_scene::load or attach_script)"),
                ));
            }
            (Some((path, line)), Some(wasm)) => {
                let mut s = fe_script::Script::load(&wasm.0, &sim)
                    .map_err(|e| SceneError::new(*line, format!("{path}: {e}")))?;
                for c in s.init(&mut sim) {
                    sim.apply(&c);
                }
                Some(s)
            }
        };
        let mut timeline = Vec::new();
        for t in &scene.timeline {
            let action = match resolve(&t.op, &mats, t.line)? {
                Resolved::Commands(c) => Action::Commands(c),
                Resolved::Input(p, i) => Action::Input(p, i),
                _ => Action::Other(t.op.clone(), t.line),
            };
            timeline.push(TimedResolved { when: t.when, action });
        }
        for e in &scene.expects {
            if let Check::Count(m, ..) | Check::Temp(m, ..) = &e.check {
                mats.id(m)
                    .ok_or_else(|| SceneError::new(e.line, format!("unknown material '{m}'")))?;
            }
        }
        let checked = vec![false; scene.expects.len()];
        let mut r = Self {
            scene,
            sim,
            mats,
            timeline,
            extra: BTreeMap::new(),
            recorded: Vec::new(),
            pre: Vec::new(),
            results: Vec::new(),
            checked,
            script,
        };
        // Expectations at or before the start tick (`after 0`) hold for the built world.
        let start = r.check_due();
        r.results.extend(start);
        Ok(r)
    }

    /// Applies a command now, between ticks, and records it so a replay applies
    /// it at the same point.
    pub fn apply_command(&mut self, cmd: &Command) {
        self.sim.apply(cmd);
        self.pre.push(cmd.clone());
    }

    /// Checks every expectation not yet checked whose tick has been reached. A
    /// tick skipped by a `run` op is checked late, at the current tick.
    fn check_due(&mut self) -> Vec<CheckResult> {
        let done = self.sim.tick();
        let mut due: Vec<usize> = (0..self.scene.expects.len())
            .filter(|&i| !self.checked[i] && self.scene.expects[i].after <= done)
            .collect();
        due.sort_by_key(|&i| self.scene.expects[i].after);
        due.into_iter()
            .map(|i| {
                self.checked[i] = true;
                let e = &self.scene.expects[i];
                let mut c = self.check(e);
                if e.after != done {
                    c.detail = format!("{} (checked late, at tick {done})", c.detail);
                }
                c
            })
            .collect()
    }

    /// Replaces the simulation, for example with one restored from a network
    /// snapshot, and restores the game script from its saved state so it
    /// continues exactly where the snapshot's script was.
    pub fn replace_sim(&mut self, sim: Simulation) -> Result<(), SceneError> {
        self.sim = sim;
        if let Some(s) = self.script.as_mut() {
            s.restore(&self.sim).map_err(|e| SceneError::new(0, e.to_string()))?;
        }
        Ok(())
    }

    /// [`fe_script::mod_hash`] of the running script, or 0 if there is none.
    pub fn mods_hash(&self) -> u64 {
        self.script.as_ref().map_or(0, fe_script::Script::mod_hash)
    }

    /// Script log lines since the last call.
    pub fn take_script_logs(&mut self) -> Vec<String> {
        self.script.as_mut().map(fe_script::Script::take_logs).unwrap_or_default()
    }

    /// The frame the scene timeline (plus any extra frames) gives for a tick.
    pub fn frame_for(&self, tick: u64) -> Frame {
        let mut f = Frame { tick, ..Frame::default() };
        for t in &self.timeline {
            if !t.when.fires(tick) {
                continue;
            }
            match &t.action {
                Action::Commands(c) => f.commands.extend(c.iter().cloned()),
                Action::Input(p, i) => {
                    f.inputs.insert(*p, *i);
                }
                Action::Other(..) => {}
            }
        }
        if let Some(e) = self.extra.get(&tick) {
            f.commands.extend(e.commands.iter().cloned());
            for (p, i) in &e.inputs {
                f.inputs.insert(*p, *i);
            }
        }
        f
    }

    /// Runs one tick with extra inputs and commands on top of the timeline.
    ///
    /// Returns the expectations checked after this tick.
    pub fn step_with(
        &mut self,
        inputs: &BTreeMap<u8, PlayerInput>,
        commands: &[Command],
    ) -> Vec<CheckResult> {
        let tick = self.sim.tick();
        if let Some(e) = self.extra.get(&tick) {
            for c in e.pre.clone() {
                self.sim.apply(&c);
            }
        }
        let pre = std::mem::take(&mut self.pre);
        for t in &self.timeline {
            if let Action::Other(op, line) = &t.action
                && t.when.fires(tick)
                && let Ok(r) = resolve(op, &self.mats, *line)
            {
                apply_now(&mut self.sim, r, self.scene.seed ^ tick);
            }
        }
        // The game script runs first; its commands apply this tick, before the
        // timeline's. They are not recorded: a replay runs the script again.
        let mut script_cmds = match self.script.as_mut() {
            Some(s) => s.tick(&mut self.sim),
            None => Vec::new(),
        };
        let mut frame = self.frame_for(tick);
        script_cmds.append(&mut frame.commands);
        frame.commands = script_cmds;
        for (p, i) in inputs {
            frame.inputs.insert(*p, *i);
        }
        frame.commands.extend(commands.iter().cloned());
        if !inputs.is_empty() || !commands.is_empty() || !pre.is_empty() {
            self.recorded.push(Frame {
                tick,
                inputs: inputs.clone(),
                commands: commands.to_vec(),
                pre,
            });
        }
        self.sim.step(&frame.inputs, &frame.commands);
        let checked = self.check_due();
        self.results.extend(checked.iter().cloned());
        checked
    }

    /// Runs one tick from the timeline alone.
    pub fn step(&mut self) -> Vec<CheckResult> {
        self.step_with(&BTreeMap::new(), &[])
    }

    /// True once the scene's tick count has run.
    pub fn finished(&self) -> bool {
        self.sim.tick() >= self.scene.ticks
    }

    /// Runs to the end of the scene and summarizes. Expectations for ticks past
    /// the end are reported as failed.
    pub fn run_to_end(&mut self) -> RunSummary {
        while !self.finished() {
            self.step();
        }
        let end = self.sim.tick();
        for (i, e) in self.scene.expects.iter().enumerate() {
            if !self.checked[i] {
                self.checked[i] = true;
                self.results.push(CheckResult {
                    line: e.line,
                    after: e.after,
                    passed: false,
                    detail: format!("tick {} never reached; the run ended at tick {end}", e.after),
                    found_hash: None,
                });
            }
        }
        self.summary()
    }

    /// Summary of the results so far.
    pub fn summary(&self) -> RunSummary {
        RunSummary {
            passed: self.results.iter().all(|r| r.passed),
            ticks: self.sim.tick(),
            hash: self.sim.hash(),
            checks: self.results.clone(),
        }
    }

    /// Frames injected through [`Runner::step_with`] and [`Runner::apply_command`],
    /// for saving as a replay.
    pub fn recorded(&self) -> Vec<Frame> {
        let mut out = self.recorded.clone();
        if !self.pre.is_empty() {
            out.push(Frame { tick: self.sim.tick(), pre: self.pre.clone(), ..Frame::default() });
        }
        out
    }

    /// Evaluates one expectation against the current state.
    pub fn check(&self, e: &Expect) -> CheckResult {
        let sim = &self.sim;
        let mut found_hash = None;
        let (passed, detail) = match &e.check {
            Check::Count(m, r, cmp, n) => {
                let id = self.mats.id(m).unwrap_or(MaterialId::BOUNDARY);
                let got = sim.world.count(*r, id) as i64;
                (cmp.test(got, *n), format!("count {m} = {got}, expected {} {n}", cmp.symbol()))
            }
            Check::Hash(h) => {
                let got = sim.hash();
                found_hash = Some(got);
                (got == *h, format!("hash = {got:#018x}, expected {h:#018x}"))
            }
            Check::Bodies(cmp, n) => {
                let got = sim.physics.bodies().len() as i64;
                (cmp.test(got, *n), format!("bodies = {got}, expected {} {n}", cmp.symbol()))
            }
            Check::Particles(cmp, n) => {
                let got = sim.particles.len() as i64;
                (cmp.test(got, *n), format!("particles = {got}, expected {} {n}", cmp.symbol()))
            }
            Check::Active(cmp, n) => {
                let got = i64::from(sim.report().active_chunks);
                (cmp.test(got, *n), format!("active chunks = {got}, expected {} {n}", cmp.symbol()))
            }
            Check::Avatar(id, is_x, cmp, n) => match sim.avatars.iter().find(|a| a.id == *id) {
                Some(a) => {
                    let got = i64::from(if *is_x { a.x.floor() } else { a.y.floor() });
                    let axis = if *is_x { "x" } else { "y" };
                    (
                        cmp.test(got, *n),
                        format!("avatar {id} {axis} = {got}, expected {} {n}", cmp.symbol()),
                    )
                }
                None => (false, format!("avatar {id} does not exist")),
            },
            Check::Temp(m, r, cmp, n) => {
                let id = self.mats.id(m).unwrap_or(MaterialId::BOUNDARY);
                let (mut sum, mut cells) = (0i64, 0i64);
                for y in r.min.y..r.max.y {
                    for x in r.min.x..r.max.x {
                        let c = sim.world.get(fe_core::CellPos::new(x, y));
                        if c.material() == id {
                            sum += i64::from(fe_core::temp_to_celsius(c.temp));
                            cells += 1;
                        }
                    }
                }
                if cells == 0 {
                    (false, format!("no {m} cells in the rectangle"))
                } else {
                    let got = sum / cells;
                    (
                        cmp.test(got, *n),
                        format!("mean temp of {m} = {got} C, expected {} {n}", cmp.symbol()),
                    )
                }
            }
        };
        CheckResult { line: e.line, after: e.after, passed, detail, found_hash }
    }
}

/// A saved run: scene text plus the inputs and commands injected on top of it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Replay {
    /// Replay format version.
    pub format: u32,
    /// Full scene source.
    pub scene: String,
    /// Directory the scene's relative paths resolve against.
    pub scene_dir: String,
    /// Injected frames in tick order.
    pub frames: Vec<Frame>,
    /// `(tick, hash)` checkpoints recorded while the replay was made.
    pub hashes: Vec<(u64, u64)>,
}

impl Replay {
    /// Current replay format.
    pub const FORMAT: u32 = 1;

    /// Parses a replay from JSON.
    pub fn from_json(text: &str) -> Result<Self, SceneError> {
        serde_json::from_str(text)
            .map_err(|e| SceneError::new(0, format!("replay is not valid JSON: {e}")))
    }

    /// Serializes to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Builds a runner with the replay's frames loaded.
    pub fn runner(&self, mats: Arc<MaterialSet>, threads: usize) -> Result<Runner, SceneError> {
        let mut scene = crate::parse(&self.scene)?;
        crate::attach_script(&mut scene, std::path::Path::new(&self.scene_dir))?;
        let mut r = Runner::new(scene, mats, threads)?;
        for f in &self.frames {
            r.extra.insert(f.tick, f.clone());
        }
        Ok(r)
    }
}
