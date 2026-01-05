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
    results: Vec<CheckResult>,
}

impl Runner {
    /// Builds the scene and prepares its timeline.
    pub fn new(scene: Scene, mats: Arc<MaterialSet>, threads: usize) -> Result<Self, SceneError> {
        let sim = build(&scene, &mats, threads)?;
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
            if let Check::Count(m, ..) = &e.check {
                mats.id(m)
                    .ok_or_else(|| SceneError::new(e.line, format!("unknown material '{m}'")))?;
            }
        }
        Ok(Self {
            scene,
            sim,
            mats,
            timeline,
            extra: BTreeMap::new(),
            recorded: Vec::new(),
            results: Vec::new(),
        })
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
        for t in &self.timeline {
            if let Action::Other(op, line) = &t.action
                && t.when.fires(tick)
                && let Ok(r) = resolve(op, &self.mats, *line)
            {
                apply_now(&mut self.sim, r, self.scene.seed ^ tick);
            }
        }
        let mut frame = self.frame_for(tick);
        for (p, i) in inputs {
            frame.inputs.insert(*p, *i);
        }
        frame.commands.extend(commands.iter().cloned());
        if !inputs.is_empty() || !commands.is_empty() {
            self.recorded.push(Frame { tick, inputs: inputs.clone(), commands: commands.to_vec() });
        }
        self.sim.step(&frame.inputs, &frame.commands);
        let done = self.sim.tick();
        let due: Vec<Expect> =
            self.scene.expects.iter().filter(|e| e.after == done).cloned().collect();
        let checked: Vec<CheckResult> = due.iter().map(|e| self.check(e)).collect();
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

    /// Runs to the end of the scene and summarizes.
    pub fn run_to_end(&mut self) -> RunSummary {
        while !self.finished() {
            self.step();
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

    /// Frames injected through [`Runner::step_with`], for saving as a replay.
    pub fn recorded(&self) -> &[Frame] {
        &self.recorded
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
        let scene = crate::parse(&self.scene)?;
        let mut r = Runner::new(scene, mats, threads)?;
        for f in &self.frames {
            r.extra.insert(f.tick, f.clone());
        }
        Ok(r)
    }
}
