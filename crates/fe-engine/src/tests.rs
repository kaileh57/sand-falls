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

fn in_body_cells(s: &Simulation) -> usize {
    s.world
        .grid
        .chunks()
        .map(|c| {
            (0..fe_core::CHUNK_AREA).filter(|&i| c.get(i).has(fe_core::Cell::FLAG_IN_BODY)).count()
        })
        .sum()
}

fn stamped_cells(s: &Simulation) -> usize {
    s.physics.bodies().iter().map(|b| b.stamped.len()).sum()
}

#[test]
fn repairing_a_chunk_with_identical_state_changes_nothing() {
    let build = || {
        let mut s = sim(1);
        let cmds = vec![
            Command::FillRect { rect: IRect::new(0, 230, 320, 256), material: id(&s, "bedrock") },
            Command::FillRect { rect: IRect::new(60, 40, 110, 120), material: id(&s, "sand") },
        ];
        s.step(&BTreeMap::new(), &cmds);
        s.run(300);
        s
    };
    let (mut a, mut b) = (build(), build());
    let pos = fe_core::ChunkPos::new(1, 1);
    let state = b.chunk_state(pos).unwrap_or_else(|| panic!("chunk not loaded"));
    b.replace_chunks(&[state]);
    for _ in 0..10 {
        a.run(1);
        b.run(1);
        assert_eq!(a.hash(), b.hash(), "diverged at tick {}", a.tick());
    }
}

#[test]
fn loading_a_neighbor_wakes_material_resting_on_the_old_border() {
    let mats = Arc::new(fe_materials::base_set());
    let mut s = Simulation::new(mats, SimConfig { seed: 3, ..SimConfig::default() });
    s.load_rect(IRect::new(0, 0, 64, 64));
    let sand = id(&s, "sand");
    s.apply(&Command::FillRect { rect: IRect::new(30, 60, 34, 64), material: sand });
    s.run(100);
    assert_eq!(s.world.count(IRect::new(0, 56, 64, 64), sand), 16);
    s.load_rect(IRect::new(0, 64, 64, 128));
    s.run(200);
    assert_eq!(s.world.count(IRect::new(0, 0, 64, 64), sand), 0);
    assert_eq!(s.world.count(IRect::new(0, 64, 64, 128), sand), 16);
}

#[test]
fn streaming_keeps_chunks_under_bodies_loaded() {
    let mats = Arc::new(fe_materials::base_set());
    let mut s = Simulation::new(
        mats,
        SimConfig {
            seed: 5,
            threads: 1,
            worldgen: WorldGen::Empty,
            stream: Some(StreamConfig { radius: 2, margin: 1 }),
        },
    );
    s.load_rect(IRect::new(-64, -64, 128, 128));
    s.apply(&Command::AddPlayer { id: 0, x: 0, y: 60 });
    s.apply(&Command::FillRect { rect: IRect::new(0, 40, 64, 50), material: id(&s, "bedrock") });
    s.apply(&Command::FillRect { rect: IRect::new(20, 10, 32, 22), material: id(&s, "wood") });
    s.apply(&Command::CarveBody { rect: IRect::new(20, 10, 32, 22) });
    s.run(60);
    assert_eq!(s.physics.bodies().len(), 1);
    s.avatars[0].x = fe_core::Fx::from_int(5000);
    s.run(8);
    assert!(s.world.grid.contains(fe_core::ChunkPos::new(0, 0)));
    s.avatars[0].x = fe_core::Fx::from_int(0);
    s.run(8);
    assert_eq!(s.physics.bodies().len(), 1);
    assert_eq!(in_body_cells(&s), stamped_cells(&s));
    assert_eq!(in_body_cells(&s), 144);
}

#[test]
fn explosion_commands_push_bodies() {
    let mut s = sim(1);
    let cmds = vec![
        Command::FillRect { rect: IRect::new(0, 230, 320, 256), material: id(&s, "bedrock") },
        Command::FillRect { rect: IRect::new(100, 200, 120, 230), material: id(&s, "stone") },
        Command::CarveBody { rect: IRect::new(100, 200, 120, 230) },
    ];
    s.step(&BTreeMap::new(), &cmds);
    s.run(30);
    let explode = [Command::Explode { x: 141, y: 215, radius: 20, power: 220 }];
    s.step(&BTreeMap::new(), &explode);
    let b = s.physics.body_info(0).unwrap_or_else(|| panic!("body gone"));
    assert!(b.vx < -1.0, "body velocity {} {}", b.vx, b.vy);
}

#[test]
fn bodies_that_leave_the_world_are_removed() {
    let mut s = sim(1);
    let cmds = vec![
        Command::FillRect { rect: IRect::new(100, 50, 112, 62), material: id(&s, "wood") },
        Command::CarveBody { rect: IRect::new(100, 50, 112, 62) },
    ];
    s.step(&BTreeMap::new(), &cmds);
    assert_eq!(s.physics.bodies().len(), 1);
    s.run(300);
    assert!(s.physics.bodies().is_empty());
}

#[test]
fn crate_tool_spares_indestructible_cells_and_loads_nothing() {
    let mut s = sim(1);
    let bedrock = id(&s, "bedrock");
    let cmds = vec![
        Command::FillRect { rect: IRect::new(0, 200, 320, 256), material: bedrock },
        Command::AddPlayer { id: 0, x: 100, y: 200 },
    ];
    s.step(&BTreeMap::new(), &cmds);
    s.run(20);
    let before = s.world.count(IRect::new(0, 0, 320, 256), bedrock);
    let chunks = s.world.grid.len();
    let press = |aim_x, aim_y| PlayerInput {
        tool: tools::CRATE,
        use_tool: true,
        aim_x,
        aim_y,
        ..Default::default()
    };
    for (ax, ay) in [(0, 10), (30000, 0), (-30000, -30000)] {
        let mut inputs = BTreeMap::new();
        inputs.insert(0, press(ax, ay));
        s.step(&inputs, &[]);
        s.step(&BTreeMap::new(), &[]);
    }
    assert_eq!(s.world.count(IRect::new(0, 0, 320, 256), bedrock), before);
    assert_eq!(s.world.grid.len(), chunks);
}

#[test]
fn bodies_land_on_flat_ground_at_any_height() {
    let mats = Arc::new(fe_materials::base_set());
    for floor in [150, 600, 1000, 1500, 1800, 2200] {
        let mut s = Simulation::new(mats.clone(), SimConfig::default());
        s.load_rect(IRect::new(0, 0, 128, floor + 100));
        let stone = id(&s, "stone");
        let cmds = vec![
            Command::FillRect {
                rect: IRect::new(0, floor, 128, floor + 100),
                material: id(&s, "bedrock"),
            },
            Command::FillRect { rect: IRect::new(40, floor - 30, 44, floor - 26), material: stone },
            Command::CarveBody { rect: IRect::new(40, floor - 30, 44, floor - 26) },
        ];
        s.step(&BTreeMap::new(), &cmds);
        s.run(300);
        let landed = s.world.count(IRect::new(0, floor - 20, 128, floor), stone);
        assert_eq!(landed, 16, "floor at {floor}: {landed} stone cells on the surface");
    }
}

#[test]
fn a_body_sinking_through_water_conserves_the_water() {
    let mut s = sim(1);
    let water = id(&s, "water");
    let cmds = vec![
        Command::FillRect { rect: IRect::new(0, 230, 320, 256), material: id(&s, "bedrock") },
        Command::FillRect { rect: IRect::new(0, 60, 4, 230), material: id(&s, "bedrock") },
        Command::FillRect { rect: IRect::new(252, 60, 256, 230), material: id(&s, "bedrock") },
        Command::FillRect { rect: IRect::new(4, 100, 252, 230), material: water },
        Command::FillRect { rect: IRect::new(100, 20, 140, 50), material: id(&s, "stone") },
        Command::CarveBody { rect: IRect::new(100, 20, 140, 50) },
    ];
    let total = |s: &Simulation| {
        s.world.census()[water.index()]
            + s.particles.iter().filter(|p| p.cell.material() == water).count() as u64
    };
    s.step(&BTreeMap::new(), &cmds);
    let start = total(&s);
    for _ in 0..150 {
        s.run(1);
        assert_eq!(total(&s), start, "water changed at tick {}", s.tick());
    }
}

#[test]
fn a_scratch_does_not_drop_a_pinned_plank() {
    let mut s = sim(1);
    let cmds = vec![
        Command::FillRect { rect: IRect::new(100, 40, 140, 46), material: id(&s, "wood") },
        Command::CarveBody { rect: IRect::new(100, 40, 140, 46) },
        Command::Pin { body: 0, x: 101, y: 43, break_force: 0.0 },
        Command::Pin { body: 0, x: 139, y: 43, break_force: 0.0 },
    ];
    s.step(&BTreeMap::new(), &cmds);
    s.run(10);
    s.step(&BTreeMap::new(), &[Command::Dig { x: 120, y: 45, radius: 0 }]);
    s.run(120);
    assert_eq!(s.physics.bodies().len(), 1);
    assert_eq!(s.physics.joints().len(), 2);
    let b = s.physics.bodies()[0].id;
    let info = s.physics.body_info(b).unwrap_or_else(|| panic!("no body"));
    assert!(info.y < 50.0, "plank fell to y {}", info.y);
}

#[test]
fn digging_a_lopsided_piece_off_a_pinned_plank_keeps_it_in_place() {
    let mut s = sim(1);
    let wood = id(&s, "wood");
    let cmds = vec![
        Command::FillRect { rect: IRect::new(100, 40, 140, 46), material: wood },
        Command::CarveBody { rect: IRect::new(100, 40, 140, 46) },
        Command::Pin { body: 0, x: 101, y: 43, break_force: 0.0 },
        Command::Pin { body: 0, x: 139, y: 43, break_force: 0.0 },
    ];
    s.step(&BTreeMap::new(), &cmds);
    s.run(10);
    // The top half of the left side: 84 of 240 cells. The plank stays in one
    // piece but its center of mass moves about 2 cells right and 1 down.
    let dig: Vec<Command> =
        (102..130).flat_map(|x| (40..43).map(move |y| Command::Dig { x, y, radius: 0 })).collect();
    s.step(&BTreeMap::new(), &dig);
    s.run(120);
    assert_eq!(s.physics.bodies().len(), 1);
    assert_eq!(s.physics.joints().len(), 2);
    let kept = s.world.count(IRect::new(100, 40, 140, 46), wood);
    assert_eq!(kept, 156, "the plank moved: {kept} of 156 cells still in place");
}

#[test]
fn carving_out_structure_checks_what_it_held_up() {
    let mut s = sim(1);
    let brick = id(&s, "brick");
    let cmds = vec![
        Command::FillRect { rect: IRect::new(0, 120, 320, 256), material: id(&s, "bedrock") },
        Command::FillRect { rect: IRect::new(60, 40, 70, 120), material: brick },
    ];
    s.step(&BTreeMap::new(), &cmds);
    s.step(&BTreeMap::new(), &[Command::CarveBody { rect: IRect::new(60, 90, 70, 110) }]);
    let r = *s.report();
    assert_eq!(r.detached, 1, "{r:?}");
    assert_eq!(r.bodies_created, 2, "{r:?}");
    // Push the carved block out; the column above has nothing left to rest on.
    s.step(&BTreeMap::new(), &[Command::Impulse { body: 0, vx: 300.0, vy: 0.0 }]);
    s.run(120);
    assert!(s.world.count(IRect::new(60, 40, 70, 90), brick) < 500);
}

#[test]
fn carving_too_wide_a_region_keeps_the_cells() {
    let mats = Arc::new(fe_materials::base_set());
    let mut s = Simulation::new(mats, SimConfig::default());
    s.load_rect(IRect::new(0, 0, 4200, 64));
    let wood = id(&s, "wood");
    s.apply(&Command::FillRect { rect: IRect::new(0, 10, 4200, 12), material: wood });
    s.apply(&Command::CarveBody { rect: IRect::new(0, 10, 4200, 12) });
    s.run(1);
    assert_eq!(s.world.count(IRect::new(0, 0, 4200, 64), wood), 8400);
    assert!(s.physics.bodies().is_empty());
}

#[test]
fn tool_masks_and_zones_gate_tools_and_enter_hash_and_snapshot() {
    let mut s = sim(1);
    s.step(
        &BTreeMap::new(),
        &[
            Command::FillRect { rect: IRect::new(0, 230, 320, 256), material: id(&s, "bedrock") },
            Command::AddPlayer { id: 0, x: 100, y: 229 },
        ],
    );
    s.run(30);
    let plain = s.hash();
    s.set_tool_mask(0, tools::ALL);
    s.set_tool_zone(0, None);
    assert_eq!(s.hash(), plain, "default game state must not change the hash");

    let sand = id(&s, "sand");
    let paint = PlayerInput {
        tool: tools::PAINT,
        param: sand.0,
        aim_y: -40,
        use_tool: true,
        ..Default::default()
    };
    let mut inputs = BTreeMap::new();
    inputs.insert(0, paint);
    let area = IRect::new(0, 0, 320, 230);
    let before = s.world.count(area, sand);

    s.set_tool_mask(0, tools::bit(tools::DIG));
    assert_ne!(s.hash(), plain);
    s.step(&inputs, &[]);
    assert_eq!(s.world.count(area, sand), before, "masked tool must do nothing");

    s.set_tool_mask(0, tools::bit(tools::PAINT));
    s.set_tool_zone(0, Some(IRect::new(200, 0, 320, 230)));
    s.step(&inputs, &[]);
    assert_eq!(s.world.count(area, sand), before, "tool aimed outside its zone must do nothing");

    s.set_tool_zone(0, Some(area));
    s.step(&inputs, &[]);
    assert!(s.world.count(area, sand) > before);

    s.status = "ROUND 1".into();
    s.script_state = vec![1, 2, 3];
    let b = Simulation::restore(&s.snapshot(), s.materials().clone(), 1)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(b.hash(), s.hash());
    assert_eq!(b.hash_parts(), s.hash_parts());
    assert_eq!(b.status, "ROUND 1");
    assert_eq!(b.script_state, vec![1, 2, 3]);
    assert_eq!(b.tool_mask(0), tools::bit(tools::PAINT));
    assert_eq!(b.tool_zone(0), Some(area));
}

/// A brick floor on bedrock across the test world, with its top at y = 200.
fn floor(s: &mut Simulation) {
    let (bedrock, brick) = (id(s, "bedrock"), id(s, "brick"));
    s.apply(&Command::FillRect { rect: IRect::new(0, 230, 320, 256), material: bedrock });
    s.apply(&Command::FillRect { rect: IRect::new(0, 200, 320, 230), material: brick });
}

fn particles_of(s: &Simulation, name: &str) -> usize {
    let m = id(s, name);
    s.particles.iter().filter(|p| p.cell.material() == m).count()
}

#[test]
fn a_stack_of_tnt_goes_off_as_one_big_blast() {
    let mut s = sim(2);
    floor(&mut s);
    let (tnt, flash, brick) = (id(&s, "tnt"), id(&s, "flash"), id(&s, "brick"));
    s.apply(&Command::FillRect { rect: IRect::new(154, 188, 166, 200), material: tnt });
    let all = IRect::new(0, 0, 320, 256);
    let bricks = s.world.count(all, brick);
    // Light one corner: the rest goes up with it, not cell by cell.
    s.apply(&Command::Heat { x: 155, y: 198, radius: 1, delta: 30 });
    let (mut ticks, mut blasts, mut most_flash, mut most_particles, mut sparks) = (0, 0, 0, 0, 0);
    while s.world.count(all, tnt) > 0 {
        assert!(ticks < 6, "{} tnt left after {ticks} ticks", s.world.count(all, tnt));
        s.run(1);
        ticks += 1;
        blasts += s.report().explosions;
        most_flash = most_flash.max(s.world.count(all, flash));
        most_particles = most_particles.max(s.particles.len());
        sparks = sparks.max(particles_of(&s, "ember"));
    }
    assert!(blasts <= 4, "{blasts} separate blasts");
    assert!(most_flash >= 300, "flash {most_flash}");
    assert!(most_particles >= 600, "particles {most_particles}");
    assert!(sparks >= 10, "embers thrown {sparks}");
    let crater = bricks - s.world.count(all, brick);
    assert!(crater >= 1500, "crater of {crater} bricks");
}

#[test]
fn a_blast_sets_off_a_charge_it_reaches() {
    let mut s = sim(1);
    floor(&mut s);
    let tnt = id(&s, "tnt");
    s.apply(&Command::FillRect { rect: IRect::new(100, 196, 104, 200), material: tnt });
    s.apply(&Command::FillRect { rect: IRect::new(112, 196, 116, 200), material: tnt });
    s.apply(&Command::Heat { x: 101, y: 198, radius: 1, delta: 30 });
    s.run(4);
    // The second charge went off too; before, the first blast erased it unlit.
    assert_eq!(s.world.count(IRect::new(0, 0, 320, 256), tnt), 0);
    assert!(s.report().explosions > 0 || s.particles.len() > 100);
}

#[test]
fn a_shell_bursting_in_the_air_throws_its_stars_all_round() {
    let mut s = sim(1);
    floor(&mut s);
    let molten = id(&s, "molten_metal");
    s.apply(&Command::Paint { x: 160, y: 80, radius: 5, material: molten, replace: true });
    s.apply(&Command::Explode { x: 160, y: 80, radius: 8, power: 160 });
    s.run(12);
    let stars: Vec<_> = s.particles.iter().filter(|p| p.cell.material() == molten).collect();
    assert!(stars.len() >= 60, "{} stars", stars.len());
    let below = stars.iter().filter(|p| p.y.floor() > 82).count();
    let above = stars.iter().filter(|p| p.y.floor() < 78).count();
    // Grounded debris bounces up out of its crater; a burst in the air is round.
    assert!(below * 3 >= stars.len(), "{below} of {} below the center", stars.len());
    assert!(above * 3 >= stars.len(), "{above} of {} above the center", stars.len());
}

#[test]
fn a_ground_blast_still_throws_its_debris_up() {
    let mut s = sim(1);
    floor(&mut s);
    s.apply(&Command::Explode { x: 160, y: 205, radius: 12, power: 200 });
    s.run(3);
    let rubble = id(&s, "rubble");
    let thrown: Vec<_> = s.particles.iter().filter(|p| p.cell.material() == rubble).collect();
    assert!(thrown.len() > 50, "{} thrown", thrown.len());
    let rising = thrown.iter().filter(|p| p.vy.0 < 0).count();
    assert!(rising * 10 >= thrown.len() * 9, "{rising} of {} rising", thrown.len());
}

#[test]
fn an_oil_slick_on_water_burns_across() {
    let mut s = sim(1);
    let (bedrock, water, oil) = (id(&s, "bedrock"), id(&s, "water"), id(&s, "oil"));
    s.apply(&Command::FillRect { rect: IRect::new(0, 230, 320, 256), material: bedrock });
    s.apply(&Command::FillRect { rect: IRect::new(0, 100, 6, 230), material: bedrock });
    s.apply(&Command::FillRect { rect: IRect::new(314, 100, 320, 230), material: bedrock });
    s.apply(&Command::FillRect { rect: IRect::new(6, 180, 314, 230), material: water });
    s.apply(&Command::FillRect { rect: IRect::new(40, 177, 280, 180), material: oil });
    s.run(30);
    s.apply(&Command::Heat { x: 42, y: 178, radius: 2, delta: 30 });
    let all = IRect::new(0, 0, 320, 256);
    let start = s.world.count(all, oil);
    s.run(600);
    // It used to flash into a few flames and go out.
    let left = s.world.count(all, oil);
    assert!(left * 10 < start, "{left} of {start} oil unburned");
}

#[test]
fn fill_then_carve_on_one_tick_makes_a_body() {
    let mut s = sim(1);
    floor(&mut s);
    let wood = id(&s, "wood");
    let rect = IRect::new(100, 40, 112, 52);
    s.step(
        &BTreeMap::new(),
        &[Command::FillRect { rect, material: wood }, Command::CarveBody { rect }],
    );
    assert_eq!(s.physics.bodies().len(), 1);
    // It is in the world at once, so nothing flows into the space it holds.
    assert_eq!(in_body_cells(&s), 144);
    assert_eq!(stamped_cells(&s), 144);
}
