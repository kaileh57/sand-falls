//! Turns a parsed scene into a running simulation.

use crate::parse::{Op, ParamSpec, Scene, SceneError, Setup};
use fe_core::{CellPos, IRect, MaterialId, PosSet, SeqRng, celsius_to_temp};
use fe_engine::{Command, PlayerInput, SimConfig, Simulation, StreamConfig};
use fe_materials::MaterialSet;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

/// Loads the material set a scene names, relative to `base_dir`.
pub fn load_materials(scene: &Scene, base_dir: &Path) -> Result<MaterialSet, SceneError> {
    if scene.materials == "base" {
        return Ok(fe_materials::base_set());
    }
    let path = base_dir.join(&scene.materials);
    let text = std::fs::read_to_string(&path).map_err(|e| {
        SceneError::new(0, format!("cannot read materials {}: {e}", path.display()))
    })?;
    MaterialSet::from_toml(&text)
        .map_err(|e| SceneError::new(0, format!("{}: {e}", path.display())))
}

fn mat(mats: &MaterialSet, name: &str, line: usize) -> Result<MaterialId, SceneError> {
    mats.id(name).ok_or_else(|| SceneError::new(line, format!("unknown material '{name}'")))
}

/// What an op becomes: commands, an input, or something handled by the runner.
pub(crate) enum Resolved {
    Commands(Vec<Command>),
    Input(u8, PlayerInput),
    Run(u64),
    Terrain(Op),
    Scatter(IRect, MaterialId, u32),
    Temp(IRect, u8),
    Carve(IRect),
    Body(IRect, MaterialId),
}

pub(crate) fn resolve(op: &Op, mats: &MaterialSet, line: usize) -> Result<Resolved, SceneError> {
    Ok(Resolved::Commands(match op {
        Op::Fill(r, m) => vec![Command::FillRect { rect: *r, material: mat(mats, m, line)? }],
        Op::Circle(x, y, r, m) => {
            vec![Command::Paint {
                x: *x,
                y: *y,
                radius: *r,
                material: mat(mats, m, line)?,
                replace: true,
            }]
        }
        Op::Paint(x, y, r, m) => {
            vec![Command::Paint {
                x: *x,
                y: *y,
                radius: *r,
                material: mat(mats, m, line)?,
                replace: false,
            }]
        }
        Op::Body(r, m) => return Ok(Resolved::Body(*r, mat(mats, m, line)?)),
        Op::Carve(r) => return Ok(Resolved::Carve(*r)),
        Op::Terrain { material, .. } => {
            mat(mats, material, line)?;
            return Ok(Resolved::Terrain(op.clone()));
        }
        Op::Scatter(r, m, p) => return Ok(Resolved::Scatter(*r, mat(mats, m, line)?, *p)),
        Op::Temp(r, c) => return Ok(Resolved::Temp(*r, celsius_to_temp(*c))),
        Op::Player(id, x, y) => vec![Command::AddPlayer { id: *id, x: *x, y: *y }],
        Op::RemovePlayer(id) => vec![Command::RemovePlayer { id: *id }],
        Op::Pin(b, x, y, f) => vec![Command::Pin { body: *b, x: *x, y: *y, break_force: *f }],
        Op::Joint(a, b, x, y, kind, f) => {
            vec![Command::Joint { a: *a, b: *b, x: *x, y: *y, kind: *kind, break_force: *f }]
        }
        Op::Explode(x, y, r, p) => vec![Command::Explode { x: *x, y: *y, radius: *r, power: *p }],
        Op::Dig(x, y, r) => vec![Command::Dig { x: *x, y: *y, radius: *r }],
        Op::Heat(x, y, r, d) => vec![Command::Heat { x: *x, y: *y, radius: *r, delta: *d }],
        Op::Impulse(b, vx, vy) => vec![Command::Impulse { body: *b, vx: *vx, vy: *vy }],
        Op::Input(spec) => {
            let param = match &spec.param {
                ParamSpec::None => 0,
                ParamSpec::Num(n) => *n,
                ParamSpec::Material(m) => mat(mats, m, line)?.0,
            };
            return Ok(Resolved::Input(
                spec.player,
                PlayerInput {
                    move_x: spec.move_x,
                    jump: spec.jump,
                    aim_x: spec.aim.0,
                    aim_y: spec.aim.1,
                    tool: spec.tool,
                    param,
                    use_tool: spec.use_tool,
                },
            ));
        }
        Op::Run(n) => return Ok(Resolved::Run(*n)),
    }))
}

/// Applies a resolved op directly to a simulation (outside the tick loop).
pub(crate) fn apply_now(sim: &mut Simulation, r: Resolved, seed: u64) {
    match r {
        Resolved::Commands(cmds) => {
            for c in &cmds {
                sim.apply(c);
            }
        }
        Resolved::Run(n) => sim.run(n),
        Resolved::Terrain(Op::Terrain { x0, x1, base, amp, scale, bottom, material }) => {
            let Some(id) = sim.material(&material) else { return };
            let mut rng = SeqRng::new(seed ^ 0x7e44);
            // Random heights every `scale` columns, linearly interpolated.
            let knots: Vec<i32> =
                (0..=((x1 - x0) / scale + 1)).map(|_| rng.range_i32(-amp, amp)).collect();
            for x in x0..x1 {
                let k = ((x - x0) / scale) as usize;
                let f = (x - x0) % scale;
                let h = knots[k] + (knots[k + 1] - knots[k]) * f / scale;
                let top = base + h;
                let bot = bottom.min(
                    sim.world.grid.positions().map(|p| p.origin().y + 64).max().unwrap_or(top),
                );
                for y in top..bot {
                    let p = CellPos::new(x, y);
                    let c = sim.world.make_cell(id, p);
                    let _ = sim.world.set(p, c);
                }
            }
        }
        Resolved::Terrain(_) | Resolved::Input(..) => {}
        Resolved::Scatter(rect, id, pct) => {
            let mut rng = SeqRng::new(seed ^ 0x5ca7);
            for y in rect.min.y..rect.max.y {
                for x in rect.min.x..rect.max.x {
                    let p = CellPos::new(x, y);
                    if rng.below(100) < pct && sim.world.get(p).is_air() {
                        let c = sim.world.make_cell(id, p);
                        let _ = sim.world.set(p, c);
                    }
                }
            }
        }
        Resolved::Temp(rect, t) => {
            for y in rect.min.y..rect.max.y {
                for x in rect.min.x..rect.max.x {
                    let p = CellPos::new(x, y);
                    let mut c = sim.world.get(p);
                    if c.material() != MaterialId::BOUNDARY {
                        c.temp = t;
                        sim.world.grid.set(p, c);
                    }
                }
            }
        }
        Resolved::Carve(rect) => {
            sim.apply(&Command::CarveBody { rect });
        }
        Resolved::Body(rect, id) => {
            sim.apply(&Command::FillRect { rect, material: id });
            sim.apply(&Command::CarveBody { rect });
        }
    }
}

fn place_grid(
    sim: &mut Simulation,
    scene: &Scene,
    mats: &MaterialSet,
    (gx, gy, rows, line): (i32, i32, &[String], usize),
) -> Result<(), SceneError> {
    let mut body_cells: BTreeMap<char, Vec<CellPos>> = BTreeMap::new();
    for (dy, row) in rows.iter().enumerate() {
        for (dx, ch) in row.chars().enumerate() {
            if ch == ' ' {
                continue;
            }
            let entry = scene.legend.get(&ch).ok_or_else(|| {
                SceneError::new(line + dy, format!("grid character '{ch}' is not in the legend"))
            })?;
            let id = mat(mats, &entry.material, line + dy)?;
            let p = CellPos::new(gx + dx as i32, gy + dy as i32);
            let mut c = sim.world.make_cell(id, p);
            if let Some(t) = entry.temp {
                c.temp = celsius_to_temp(t);
            }
            let _ = sim.world.set(p, c);
            if entry.body {
                body_cells.entry(ch).or_default().push(p);
            }
        }
    }
    // Each 4-connected group of a body character becomes one body, in reading order.
    for cells in body_cells.values() {
        let mut members = PosSet::with_capacity(cells.len());
        for &p in cells {
            members.insert(p);
        }
        let mut done = PosSet::with_capacity(cells.len());
        for &start in cells {
            if done.contains(start) {
                continue;
            }
            let mut group = vec![start];
            done.insert(start);
            let mut i = 0;
            while i < group.len() {
                let p = group[i];
                i += 1;
                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let q = CellPos::new(p.x + dx, p.y + dy);
                    if members.contains(q) && done.insert(q) {
                        group.push(q);
                    }
                }
            }
            group.sort_by_key(|p| (p.y, p.x));
            sim.carve_cells(&group);
        }
    }
    Ok(())
}

/// Builds the starting simulation for a scene.
pub fn build(
    scene: &Scene,
    mats: &Arc<MaterialSet>,
    threads: usize,
) -> Result<Simulation, SceneError> {
    let cfg = SimConfig {
        seed: scene.seed,
        threads,
        worldgen: scene.worldgen.clone(),
        stream: scene.stream.map(|(radius, margin)| StreamConfig { radius, margin }),
    };
    let mut sim = Simulation::new(mats.clone(), cfg);
    sim.load_rect(scene.bounds);
    for step in &scene.setup {
        match step {
            Setup::Grid { x, y, rows, line } => {
                place_grid(&mut sim, scene, mats, (*x, *y, rows, *line))?;
            }
            Setup::Op(op, line) => {
                let r = resolve(op, mats, *line)?;
                apply_now(&mut sim, r, scene.seed ^ *line as u64);
            }
        }
    }
    Ok(sim)
}
