use super::*;
use fe_core::{CellPos, IRect};
use std::collections::BTreeMap;
use std::sync::Arc;

fn sim(threads: usize) -> Simulation {
    let mats = Arc::new(fe_materials::base_set());
    let mut s = Simulation::new(mats, SimConfig { seed: 11, threads, ..SimConfig::default() });
    s.load_rect(IRect::new(0, 0, 320, 256));
    s
}

fn id(s: &Simulation, n: &str) -> fe_core::MaterialId {
    s.material(n).unwrap_or_default()
}

/// A scene that touches every system: liquids, fire, explosions, a tower that
/// loses support, a crate, and a character.
fn busy(threads: usize) -> Simulation {
    let mut s = sim(threads);
    let cmds = vec![
        Command::FillRect { rect: IRect::new(0, 230, 320, 256), material: id(&s, "bedrock") },
        Command::FillRect { rect: IRect::new(20, 120, 80, 200), material: id(&s, "water") },
        Command::FillRect { rect: IRect::new(150, 130, 158, 230), material: id(&s, "brick") },
        Command::FillRect { rect: IRect::new(130, 122, 180, 130), material: id(&s, "brick") },
        Command::FillRect { rect: IRect::new(200, 200, 260, 230), material: id(&s, "wood") },
        Command::FillRect {
            rect: IRect::new(225, 196, 229, 200),
            material: id(&s, "burning_wood"),
        },
        Command::FillRect { rect: IRect::new(270, 150, 282, 162), material: id(&s, "wood") },
        Command::CarveBody { rect: IRect::new(270, 150, 282, 162) },
        Command::AddPlayer { id: 0, x: 100, y: 200 },
    ];
    s.step(&BTreeMap::new(), &cmds);
    s
}

fn drive(s: &mut Simulation, ticks: u64) {
    for t in 0..ticks {
        let mut inputs = BTreeMap::new();
        inputs.insert(
            0,
            PlayerInput {
                move_x: if t % 200 < 100 { 1 } else { -1 },
                jump: t % 37 == 0,
                ..Default::default()
            },
        );
        let cmds = if t == 30 {
            vec![Command::Explode { x: 154, y: 215, radius: 10, power: 220 }]
        } else {
            vec![]
        };
        s.step(&inputs, &cmds);
    }
}

#[test]
fn full_tick_is_independent_of_thread_count() {
    let mut a = busy(1);
    let mut b = busy(4);
    drive(&mut a, 240);
    drive(&mut b, 240);
    assert_eq!(a.hash(), b.hash());
    assert!(a.report().bodies > 0 || a.physics.stats().created > 0 || a.report().particles > 0);
}

#[test]
fn snapshot_restores_bit_exact_and_continues_identically() {
    let mut a = busy(1);
    drive(&mut a, 60);
    let bytes = a.snapshot();
    let mut b =
        Simulation::restore(&bytes, a.materials().clone(), 2).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(a.hash(), b.hash(), "restored state differs");
    drive(&mut a, 120);
    drive(&mut b, 120);
    assert_eq!(a.hash(), b.hash(), "runs diverged after restore");
}

#[test]
fn explosion_under_tower_detaches_it_into_a_body() {
    let mut s = busy(1);
    drive(&mut s, 40);
    assert!(s.physics.bodies().len() >= 2, "bodies: {}", s.physics.bodies().len());
}

#[test]
fn crate_falls_and_lands() {
    let mut s = sim(1);
    let cmds = vec![
        Command::FillRect { rect: IRect::new(0, 230, 320, 256), material: id(&s, "bedrock") },
        Command::FillRect { rect: IRect::new(100, 50, 112, 62), material: id(&s, "wood") },
        Command::CarveBody { rect: IRect::new(100, 50, 112, 62) },
    ];
    s.step(&BTreeMap::new(), &cmds);
    s.run(240);
    let info = s
        .physics
        .body_info(0)
        .or_else(|| s.physics.bodies().first().and_then(|b| s.physics.body_info(b.id)));
    match info {
        Some(b) => assert!(b.y > 210.0 && b.y < 230.0, "crate at y {}", b.y),
        None => {
            // Merged into terrain after resting: it must be on the floor.
            assert!(s.world.count(IRect::new(90, 200, 125, 230), id(&s, "wood")) > 100);
        }
    }
}

#[test]
fn wood_floats_and_stone_sinks() {
    let mut s = sim(1);
    let cmds = vec![
        Command::FillRect { rect: IRect::new(0, 230, 320, 256), material: id(&s, "bedrock") },
        Command::FillRect { rect: IRect::new(0, 100, 10, 230), material: id(&s, "bedrock") },
        Command::FillRect { rect: IRect::new(310, 100, 320, 230), material: id(&s, "bedrock") },
        Command::FillRect { rect: IRect::new(10, 150, 310, 230), material: id(&s, "water") },
        Command::FillRect { rect: IRect::new(60, 120, 80, 130), material: id(&s, "wood") },
        Command::CarveBody { rect: IRect::new(60, 120, 80, 130) },
        Command::FillRect { rect: IRect::new(200, 120, 212, 130), material: id(&s, "stone") },
        Command::CarveBody { rect: IRect::new(200, 120, 212, 130) },
    ];
    s.step(&BTreeMap::new(), &cmds);
    s.run(400);
    let wood = s.physics.body_info(0);
    let stone = s.physics.body_info(1);
    assert!(wood.is_some_and(|b| b.y < 172.0), "wood {wood:?}");
    assert!(stone.is_none_or(|b| b.y > 200.0), "stone {stone:?}");
}

#[test]
fn character_digs_with_a_tool() {
    let mut s = sim(1);
    let cmds = vec![
        Command::FillRect { rect: IRect::new(0, 150, 320, 256), material: id(&s, "soil") },
        Command::AddPlayer { id: 3, x: 100, y: 150 },
    ];
    s.step(&BTreeMap::new(), &cmds);
    s.run(10);
    let before = s.world.count(IRect::new(80, 150, 140, 190), id(&s, "soil"));
    let mut inputs = BTreeMap::new();
    inputs.insert(
        3,
        PlayerInput {
            tool: tools::DIG,
            param: 6,
            use_tool: true,
            aim_x: 0,
            aim_y: 14,
            ..Default::default()
        },
    );
    for _ in 0..5 {
        s.step(&inputs, &[]);
    }
    let after = s.world.count(IRect::new(80, 150, 140, 190), id(&s, "soil"));
    assert!(after < before - 50, "before {before} after {after}");
    assert!(s.nearest_avatar(CellPos::new(100, 140)).is_some());
}

#[test]
fn streaming_loads_around_characters_and_keeps_edits() {
    let mats = Arc::new(fe_materials::base_set());
    let mut s = Simulation::new(
        mats,
        SimConfig {
            seed: 5,
            threads: 1,
            worldgen: WorldGen::Caves { surface: 40 },
            stream: Some(StreamConfig { radius: 2, margin: 1 }),
        },
    );
    s.load_rect(IRect::new(-64, -64, 128, 128));
    s.apply(&Command::AddPlayer { id: 0, x: 0, y: 0 });
    s.apply(&Command::FillRect { rect: IRect::new(0, 0, 4, 4), material: id(&s, "brick") });
    s.run(8);
    assert!(s.world.grid.len() >= 25);
    // Move the character far away, then back.
    s.avatars[0].x = fe_core::Fx::from_int(5000);
    s.run(8);
    assert!(!s.world.grid.contains(fe_core::ChunkPos::new(0, 0)));
    s.avatars[0].x = fe_core::Fx::from_int(0);
    s.run(8);
    assert!(s.world.count(IRect::new(0, 0, 4, 4), id(&s, "brick")) >= 12);
}
