//! Bodies afloat: hollow hulls, righting, flooding and sinking.

use super::*;
use fe_core::{IRect, MaterialId};
use std::sync::Arc;

/// One tick of cells, particles and bodies, in the engine's order.
fn tick(w: &mut World, ps: &mut Particles, ph: &mut Physics) {
    w.step_cells();
    ps.step(w);
    ph.sync(w, ps);
    ph.unstamp(w);
    ph.step(w, ps);
    ph.stamp(w);
    w.finish_tick();
}

fn run(w: &mut World, ps: &mut Particles, ph: &mut Physics, ticks: u32) {
    for _ in 0..ticks {
        tick(w, ps, ph);
    }
}

fn mat(w: &World, name: &str) -> MaterialId {
    w.materials().id(name).unwrap_or_else(|| panic!("no {name}"))
}

/// Left edge and top of the hull in [`sea_and_hull`].
const HULL: (i32, i32) = (240, 80);

/// A sea 588 cells wide with its surface at y = 100, and a wooden hull 120
/// wide and 24 tall: walls 3 cells thick, a sealed hold split by bulkheads at
/// the given x offsets, and a row of iron ballast on the floor of the hold.
/// It weighs 1,386 kg and its outline holds 2,880 kg of water, so it should
/// float about 11.6 cells deep. Its 942 solid cells hold only 942 kg, so it
/// sinks once its hold floods.
fn sea_and_hull(bulkheads: &[i32]) -> (World, Particles, Physics, u32) {
    let mut w = World::new(Arc::new(fe_materials::base_set()), 5, 1);
    w.ensure_rect(IRect::new(0, 0, 600, 192), Cell::AIR);
    let (bedrock, water) = (mat(&w, "bedrock"), mat(&w, "water"));
    let (wood, metal) = (mat(&w, "wood"), mat(&w, "metal"));
    w.fill_rect(IRect::new(0, 180, 600, 192), bedrock);
    w.fill_rect(IRect::new(0, 40, 6, 180), bedrock);
    w.fill_rect(IRect::new(594, 40, 600, 180), bedrock);
    w.fill_rect(IRect::new(6, 100, 594, 180), water);
    let (x0, top) = HULL;
    w.fill_rect(IRect::new(x0, top, x0 + 120, top + 24), wood);
    w.fill_rect(IRect::new(x0 + 3, top + 3, x0 + 117, top + 21), MaterialId::AIR);
    w.fill_rect(IRect::new(x0 + 3, top + 20, x0 + 117, top + 21), metal);
    for &b in bulkheads {
        w.fill_rect(IRect::new(x0 + b, top + 3, x0 + b + 2, top + 20), wood);
    }
    let mut ph = Physics::new();
    let rect = IRect::new(x0, top, x0 + 120, top + 24);
    let id = ph.carve_body(&mut w, rect, &mut Vec::new()).unwrap_or_else(|| panic!("no hull"));
    (w, Particles::default(), ph, id)
}

fn body(ph: &Physics, id: u32) -> &PixelBody {
    ph.bodies.iter().find(|b| b.id == id).unwrap_or_else(|| panic!("body {id} is gone"))
}

/// The world rows the body's top and bottom cells are stamped in.
fn rows(ph: &Physics, id: u32) -> (i32, i32) {
    let b = body(ph, id);
    let top = b.stamped.iter().map(|s| s.0.y).min().unwrap_or(0);
    let bottom = b.stamped.iter().map(|s| s.0.y).max().unwrap_or(0);
    (top, bottom)
}

/// The topmost water cell of a column.
fn surface(w: &World, x: i32) -> i32 {
    let water = mat(w, "water");
    (0..192).find(|&y| w.get(CellPos::new(x, y)).material() == water).unwrap_or(192)
}

fn angle(ph: &Physics, id: u32) -> f32 {
    ph.rapier.bodies[body(ph, id).handle].rotation().angle()
}

/// Water inside the hold of a level hull.
fn hold_water(w: &World, ph: &Physics, id: u32) -> u64 {
    let (top, bottom) = rows(ph, id);
    let x0 = HULL.0;
    w.count(IRect::new(x0 + 4, top + 4, x0 + 116, bottom - 4), mat(w, "water"))
}

#[test]
fn a_hollow_hull_floats_with_its_deck_clear_and_its_hold_dry() {
    let (mut w, mut ps, mut ph, id) = sea_and_hull(&[]);
    run(&mut w, &mut ps, &mut ph, 600);
    let (top, bottom) = rows(&ph, id);
    let sea = surface(&w, 100);
    let draft = bottom + 1 - sea;
    assert!((9..=14).contains(&draft), "draft {draft} (deck {top}, keel {bottom}, sea {sea})");
    assert!(sea - top >= 8, "freeboard {} (deck {top}, sea {sea})", sea - top);
    assert!(angle(&ph, id).abs() < 0.01, "angle {}", angle(&ph, id));
    assert_eq!(hold_water(&w, &ph, id), 0);
    assert!(body(&ph, id).floating);
}

#[test]
fn rocking_a_floating_hull_pumps_no_water_into_its_hold() {
    let (mut w, mut ps, mut ph, id) = sea_and_hull(&[]);
    run(&mut w, &mut ps, &mut ph, 200);
    for k in 0..8 {
        ph.push_body(id, 0.0, if k % 2 == 0 { 40.0 } else { -40.0 });
        run(&mut w, &mut ps, &mut ph, 40);
    }
    run(&mut w, &mut ps, &mut ph, 300);
    assert_eq!(hold_water(&w, &ph, id), 0);
    assert!(angle(&ph, id).abs() < 0.02, "angle {}", angle(&ph, id));
}

#[test]
fn a_tilted_hull_rights_itself() {
    let (mut w, mut ps, mut ph, id) = sea_and_hull(&[]);
    run(&mut w, &mut ps, &mut ph, 300);
    let h = body(&ph, id).handle;
    let rb = &mut ph.rapier.bodies[h];
    let mut pose = *rb.position();
    pose.rotation = Rotation::new(0.3);
    rb.set_position(pose, true);
    // Turning the hull in place swings its sealed hold down over sea water,
    // which a real hull would have pushed away: take it out, so the hull is
    // dry as it rights itself. Left in, a few rows of water trapped on the
    // floor's steps hold the hull at a small list.
    tick(&mut w, &mut ps, &mut ph);
    let water = mat(&w, "water");
    for p in hold_cells(&ph, id) {
        if w.get(p).material() == water {
            w.grid.set(p, Cell::AIR);
        }
    }
    assert_eq!(water_in_hold(&w, &ph, id), 0);
    run(&mut w, &mut ps, &mut ph, 59);
    let early = angle(&ph, id);
    assert!(early.abs() < 0.25, "still at {early} after a second");
    run(&mut w, &mut ps, &mut ph, 540);
    let a = angle(&ph, id);
    assert!(a.abs() < 0.03, "settled at {a}");
}

#[test]
fn a_hull_holed_on_one_side_lists_that_way() {
    // Three compartments; the hole floods the left one.
    let (mut w, mut ps, mut ph, id) = sea_and_hull(&[38, 80]);
    run(&mut w, &mut ps, &mut ph, 300);
    let (_, bottom) = rows(&ph, id);
    let x0 = HULL.0;
    for y in bottom - 8..bottom - 3 {
        for x in x0..x0 + 3 {
            w.grid.set(CellPos::new(x, y), Cell::AIR);
        }
    }
    run(&mut w, &mut ps, &mut ph, 1200);
    let a = angle(&ph, id);
    assert!(a < -0.02, "angle {a}: the left side should be down");
    let water = mat(&w, "water");
    let flooded = w.count(IRect::new(x0 + 3, 60, x0 + 38, 120), water);
    assert!(flooded > 100, "left compartment holds {flooded} water");
    assert!(body(&ph, id).floating, "one flooded compartment of three must not sink it");
}

#[test]
fn a_flooded_hull_sinks() {
    let (mut w, mut ps, mut ph, id) = sea_and_hull(&[]);
    run(&mut w, &mut ps, &mut ph, 300);
    // Open the left side from the floor of the hold to just under the deck.
    let (top, bottom) = rows(&ph, id);
    let x0 = HULL.0;
    for y in top + 4..bottom - 3 {
        for x in x0..x0 + 3 {
            w.grid.set(CellPos::new(x, y), Cell::AIR);
        }
    }
    run(&mut w, &mut ps, &mut ph, 1500);
    let (_, bottom) = rows(&ph, id);
    assert!(bottom >= 176, "keel at {bottom}, the sea floor is at 180");
}

/// Water in the whole world: in cells and in flight.
fn all_water(w: &World, ps: &Particles) -> u64 {
    let water = mat(w, "water");
    let flying = ps.iter().filter(|p| p.cell.material() == water).count() as u64;
    w.count(IRect::new(0, 0, 600, 192), water) + flying
}

#[test]
fn a_hull_holed_in_its_bottom_floods_and_sinks() {
    let (mut w, mut ps, mut ph, id) = sea_and_hull(&[]);
    run(&mut w, &mut ps, &mut ph, 300);
    let before = all_water(&w, &ps);
    // A hole 7 cells wide through the floor and the ballast, about 8 cells
    // under the sea outside. No water rises through it on its own.
    let (_, bottom) = rows(&ph, id);
    assert!(surface(&w, 100) <= bottom - 8, "sea at {}, keel at {bottom}", surface(&w, 100));
    for y in bottom - 3..=bottom {
        for x in HULL.0 + 57..HULL.0 + 64 {
            w.grid.set(CellPos::new(x, y), Cell::AIR);
        }
    }
    run(&mut w, &mut ps, &mut ph, 120);
    let early = hold_water(&w, &ph, id);
    assert!(early > 150, "{early} water in the hold 2 s after the hole");
    let a = angle(&ph, id);
    assert!(a.abs() < 0.05, "the hull lists to {a} as it floods evenly");
    run(&mut w, &mut ps, &mut ph, 1380);
    let (_, bottom) = rows(&ph, id);
    assert!(bottom >= 176, "keel at {bottom}, the sea floor is at 180");
    let held = hold_water(&w, &ph, id);
    assert!(held > 1500, "{held} water in the sunken hold");
    assert_eq!(all_water(&w, &ps), before, "flooding made or lost water");
}

#[test]
fn a_hull_holed_low_in_its_side_floods_and_sinks() {
    // One compartment, holed in its side just over the ballast, below the
    // waterline. Water runs in to the top of the hole by itself; the rest of
    // the way to the level outside it is pushed in, and the hull goes down.
    let (mut w, mut ps, mut ph, id) = sea_and_hull(&[]);
    run(&mut w, &mut ps, &mut ph, 300);
    let (_, bottom) = rows(&ph, id);
    for y in bottom - 8..bottom - 3 {
        for x in HULL.0..HULL.0 + 3 {
            w.grid.set(CellPos::new(x, y), Cell::AIR);
        }
    }
    run(&mut w, &mut ps, &mut ph, 300);
    let held = hold_water(&w, &ph, id);
    assert!(held > 400, "{held} water in the hold: it stopped at the top of the hole");
    run(&mut w, &mut ps, &mut ph, 2100);
    let (_, bottom) = rows(&ph, id);
    assert!(bottom >= 176, "keel at {bottom}, the sea floor is at 180");
}

#[test]
fn water_a_hull_holds_over_the_sea_runs_out_through_a_hole_in_its_bottom() {
    // A hull resting on two piers with its keel under the sea, its hold
    // filled near to the deck and holed in its floor. Without pressure the
    // water would stand there; it runs out until it is level with the sea.
    // The piers stop short of the sea floor, so the water between them
    // reaches the open sea under them.
    let mut w = World::new(Arc::new(fe_materials::base_set()), 5, 1);
    w.ensure_rect(IRect::new(0, 0, 600, 192), Cell::AIR);
    let (bedrock, water) = (mat(&w, "bedrock"), mat(&w, "water"));
    let (wood, metal) = (mat(&w, "wood"), mat(&w, "metal"));
    w.fill_rect(IRect::new(0, 180, 600, 192), bedrock);
    w.fill_rect(IRect::new(0, 40, 6, 180), bedrock);
    w.fill_rect(IRect::new(594, 40, 600, 180), bedrock);
    w.fill_rect(IRect::new(6, 94, 594, 180), water);
    w.fill_rect(IRect::new(250, 100, 262, 124), bedrock);
    w.fill_rect(IRect::new(338, 100, 350, 124), bedrock);
    let (x0, top) = (240, 76);
    w.fill_rect(IRect::new(x0, top, x0 + 120, top + 24), wood);
    w.fill_rect(IRect::new(x0 + 3, top + 3, x0 + 117, top + 20), water);
    w.fill_rect(IRect::new(x0 + 3, top + 3, x0 + 117, top + 6), MaterialId::AIR);
    w.fill_rect(IRect::new(x0 + 3, top + 20, x0 + 117, top + 21), metal);
    let mut ph = Physics::new();
    let rect = IRect::new(x0, top, x0 + 120, top + 24);
    let id = ph.carve_body(&mut w, rect, &mut Vec::new()).unwrap_or_else(|| panic!("no hull"));
    let mut ps = Particles::default();
    run(&mut w, &mut ps, &mut ph, 60);
    let (_, bottom) = rows(&ph, id);
    assert!((98..=100).contains(&bottom), "keel at {bottom}: the hull should rest on the piers");
    for y in bottom - 3..=bottom {
        for x in x0 + 57..x0 + 64 {
            w.grid.set(CellPos::new(x, y), Cell::AIR);
        }
    }
    let before = all_water(&w, &ps);
    run(&mut w, &mut ps, &mut ph, 900);
    let inside = (top..=bottom)
        .find(|&y| w.get(CellPos::new(x0 + 30, y)).material() == water)
        .unwrap_or(bottom);
    let sea = surface(&w, 100);
    assert!((inside - sea).abs() <= 2, "water inside stands at {inside}, the sea at {sea}");
    assert_eq!(all_water(&w, &ps), before, "draining made or lost water");
}

/// Water in the hold of the hull of [`sea_and_hull`], at any angle: in the
/// world cells its local hold maps to.
fn water_in_hold(w: &World, ph: &Physics, id: u32) -> u64 {
    let water = mat(w, "water");
    hold_cells(ph, id).into_iter().filter(|&p| w.get(p).material() == water).count() as u64
}

/// The world cells the hold of the hull of [`sea_and_hull`] maps to.
fn hold_cells(ph: &Physics, id: u32) -> std::collections::BTreeSet<CellPos> {
    let b = body(ph, id);
    let pose = *ph.rapier.bodies[b.handle].position();
    let mut seen = std::collections::BTreeSet::new();
    for ly in 3..20 {
        for lx in 3..117 {
            seen.insert(b.local_to_world(&pose, lx, ly));
        }
    }
    seen
}

#[test]
fn a_stone_heeling_a_sealed_hull_lets_no_water_into_its_hold() {
    // A 14-cell stone dropped on the edge of the deck heels the hull hard,
    // and sits in the sea against the deck. The water it pushes aside must not
    // go through the deck into the hold under it.
    let (mut w, mut ps, mut ph, id) = sea_and_hull(&[]);
    run(&mut w, &mut ps, &mut ph, 240);
    let stone = mat(&w, "stone");
    let rect = IRect::new(HULL.0 + 108, 20, HULL.0 + 122, 34);
    w.fill_rect(rect, stone);
    ph.carve_body(&mut w, rect, &mut Vec::new()).unwrap_or_else(|| panic!("no stone"));
    let mut heel = 0.0f32;
    for _ in 0..30 {
        run(&mut w, &mut ps, &mut ph, 10);
        heel = heel.max(angle(&ph, id).abs());
        assert_eq!(water_in_hold(&w, &ph, id), 0, "water in the sealed hold at heel {heel}");
    }
    assert!(heel > 0.2, "the stone heeled the hull only to {heel}");
}

#[test]
fn a_hold_holed_in_its_floor_is_still_the_bodys_inside() {
    let (mut w, mut ps, mut ph, id) = sea_and_hull(&[]);
    run(&mut w, &mut ps, &mut ph, 200);
    let (top, bottom) = rows(&ph, id);
    for y in bottom - 3..=bottom {
        for x in HULL.0 + 57..HULL.0 + 64 {
            w.grid.set(CellPos::new(x, y), Cell::AIR);
        }
    }
    run(&mut w, &mut ps, &mut ph, 2);
    let (top2, _) = rows(&ph, id);
    let b = body(&ph, id);
    let dy = top2 - top;
    for y in top + 4..bottom - 4 {
        let p = CellPos::new(HULL.0 + 60, y + dy);
        assert!(b.hollow.contains(&p), "the hold over the hole at {p:?} is not inside");
    }
}

#[test]
fn the_air_under_an_arch_is_not_the_bodys_inside() {
    // Two legs and a span, 30 cells tall: the sky goes on under it.
    let mut w = World::new(Arc::new(fe_materials::base_set()), 5, 1);
    w.ensure_rect(IRect::new(0, 0, 128, 128), Cell::AIR);
    let brick = mat(&w, "brick");
    w.fill_rect(IRect::new(0, 120, 128, 128), mat(&w, "bedrock"));
    w.fill_rect(IRect::new(30, 86, 100, 90), brick);
    w.fill_rect(IRect::new(30, 90, 36, 120), brick);
    w.fill_rect(IRect::new(94, 90, 100, 120), brick);
    let mut ph = Physics::new();
    let rect = IRect::new(30, 86, 100, 120);
    let id = ph.carve_body(&mut w, rect, &mut Vec::new()).unwrap_or_else(|| panic!("no arch"));
    let mut ps = Particles::default();
    run(&mut w, &mut ps, &mut ph, 2);
    let hollow = &body(&ph, id).hollow;
    assert!(hollow.is_empty(), "{} cells under the arch draw as inside", hollow.len());
}

#[test]
fn a_floating_crate_stays_a_body() {
    let mut w = World::new(Arc::new(fe_materials::base_set()), 5, 1);
    w.ensure_rect(IRect::new(0, 0, 192, 128), Cell::AIR);
    w.fill_rect(IRect::new(0, 120, 192, 128), mat(&w, "bedrock"));
    w.fill_rect(IRect::new(0, 60, 192, 120), mat(&w, "water"));
    let rect = IRect::new(90, 50, 102, 62);
    w.fill_rect(rect, mat(&w, "wood"));
    let mut ph = Physics::new();
    let id = ph.carve_body(&mut w, rect, &mut Vec::new()).unwrap_or_else(|| panic!("no crate"));
    let mut ps = Particles::default();
    run(&mut w, &mut ps, &mut ph, MERGE_AFTER_TICKS * 3);
    assert_eq!(ph.bodies.len(), 1, "the crate merged into the water");
    assert!(body(&ph, id).floating);
    let (top, _) = rows(&ph, id);
    assert!(top < 60, "the crate rides with its top at {top}, the surface is at 60");
}

#[test]
fn powder_on_a_heavy_body_does_not_float_it() {
    // A brick slab under water with scrap piled on it. Lift used to come from
    // the powder above a body, which floated it at the scrap's density.
    let mut w = World::new(Arc::new(fe_materials::base_set()), 5, 1);
    w.ensure_rect(IRect::new(0, 0, 192, 192), Cell::AIR);
    w.fill_rect(IRect::new(0, 180, 192, 192), mat(&w, "bedrock"));
    w.fill_rect(IRect::new(0, 40, 192, 180), mat(&w, "water"));
    let rect = IRect::new(70, 60, 120, 66);
    w.fill_rect(rect, mat(&w, "brick"));
    let mut ph = Physics::new();
    let id = ph.carve_body(&mut w, rect, &mut Vec::new()).unwrap_or_else(|| panic!("no slab"));
    w.fill_rect(IRect::new(72, 55, 118, 60), mat(&w, "scrap"));
    let mut ps = Particles::default();
    run(&mut w, &mut ps, &mut ph, 400);
    // Brick sinks in water at about a meter a second.
    let (top, _) = rows(&ph, id);
    assert!(top >= 100, "slab top at {top}, it started at 60");
    // The water it pushes aside now fills the room it leaves behind it, under
    // the scrap, which then settles onto the slab a little later; it still
    // sinks at half a meter a second, where floating it would stand still.
    let vy = ph.rapier.bodies[body(&ph, id).handle].linvel().y;
    assert!(vy > 0.3, "slab sinking at {vy} m/s");
}

#[test]
fn a_hold_with_its_side_blown_in_is_still_the_bodys_inside() {
    let (mut w, mut ps, mut ph, id) = sea_and_hull(&[]);
    run(&mut w, &mut ps, &mut ph, 200);
    let mid = |ph: &Physics| {
        let (top, _) = rows(ph, id);
        CellPos::new(HULL.0 + 60, top + 10)
    };
    assert!(body(&ph, id).hollow.contains(&mid(&ph)), "the sealed hold");
    let (top, bottom) = rows(&ph, id);
    for y in top + 4..bottom - 3 {
        for x in HULL.0..HULL.0 + 3 {
            w.grid.set(CellPos::new(x, y), Cell::AIR);
        }
    }
    run(&mut w, &mut ps, &mut ph, 2);
    assert!(body(&ph, id).hollow.contains(&mid(&ph)), "the hold open to the sea");
}

#[test]
fn the_air_under_a_high_spar_is_not_the_bodys_inside() {
    // A deck with a mast and a spar across its top, 60 cells up.
    let mut w = World::new(Arc::new(fe_materials::base_set()), 5, 1);
    w.ensure_rect(IRect::new(0, 0, 128, 128), Cell::AIR);
    let wood = mat(&w, "wood");
    w.fill_rect(IRect::new(0, 120, 128, 128), mat(&w, "bedrock"));
    w.fill_rect(IRect::new(30, 116, 100, 120), wood);
    w.fill_rect(IRect::new(63, 56, 66, 116), wood);
    w.fill_rect(IRect::new(50, 53, 80, 56), wood);
    let mut ph = Physics::new();
    let rect = IRect::new(30, 53, 100, 120);
    let id = ph.carve_body(&mut w, rect, &mut Vec::new()).unwrap_or_else(|| panic!("no body"));
    let mut ps = Particles::default();
    run(&mut w, &mut ps, &mut ph, 2);
    let hollow = &body(&ph, id).hollow;
    assert!(hollow.is_empty(), "{} cells under the spar draw as inside", hollow.len());
}

#[test]
fn a_flooded_hull_with_tall_masts_still_sinks() {
    // The hull of [`sea_and_hull`] with two masts 80 cells tall, holed in its
    // floor. The air between the masts over the deck is no hold: once the
    // hull floods and goes under, it must not float on that air.
    let mut w = World::new(Arc::new(fe_materials::base_set()), 5, 1);
    w.ensure_rect(IRect::new(0, 0, 600, 192), Cell::AIR);
    let (wood, bedrock) = (mat(&w, "wood"), mat(&w, "bedrock"));
    w.fill_rect(IRect::new(0, 180, 600, 192), bedrock);
    w.fill_rect(IRect::new(0, 40, 6, 180), bedrock);
    w.fill_rect(IRect::new(594, 40, 600, 180), bedrock);
    w.fill_rect(IRect::new(6, 100, 594, 180), mat(&w, "water"));
    let (x0, deck) = (240, 80);
    w.fill_rect(IRect::new(x0, deck, x0 + 120, deck + 24), wood);
    w.fill_rect(IRect::new(x0 + 3, deck + 3, x0 + 117, deck + 21), MaterialId::AIR);
    w.fill_rect(IRect::new(x0 + 3, deck + 20, x0 + 117, deck + 21), mat(&w, "metal"));
    w.fill_rect(IRect::new(x0 + 57, deck + 20, x0 + 64, deck + 24), MaterialId::AIR);
    w.fill_rect(IRect::new(x0 + 20, 0, x0 + 23, deck), wood);
    w.fill_rect(IRect::new(x0 + 97, 0, x0 + 100, deck), wood);
    let mut ph = Physics::new();
    let rect = IRect::new(x0, 0, x0 + 120, deck + 24);
    let id = ph.carve_body(&mut w, rect, &mut Vec::new()).unwrap_or_else(|| panic!("no hull"));
    let mut ps = Particles::default();
    run(&mut w, &mut ps, &mut ph, 2400);
    let (_, bottom) = rows(&ph, id);
    assert!(bottom >= 176, "keel at {bottom}, the sea floor is at 180");
}

#[test]
fn an_open_boat_floats_on_the_air_it_holds() {
    // A wooden boat with no deck, 8 rows deep inside, and iron on its floor.
    // Its solid cells hold only 180 kg of water and it weighs 309 kg, so it
    // floats only on the air inside it.
    let mut w = World::new(Arc::new(fe_materials::base_set()), 5, 1);
    w.ensure_rect(IRect::new(0, 0, 320, 160), Cell::AIR);
    w.fill_rect(IRect::new(0, 150, 320, 160), mat(&w, "bedrock"));
    w.fill_rect(IRect::new(0, 80, 320, 150), mat(&w, "water"));
    let (x0, top) = (130, 60);
    w.fill_rect(IRect::new(x0, top, x0 + 60, top + 10), mat(&w, "wood"));
    w.fill_rect(IRect::new(x0 + 2, top, x0 + 58, top + 8), MaterialId::AIR);
    w.fill_rect(IRect::new(x0 + 16, top + 7, x0 + 44, top + 8), mat(&w, "metal"));
    let mut ph = Physics::new();
    let rect = IRect::new(x0, top, x0 + 60, top + 10);
    let id = ph.carve_body(&mut w, rect, &mut Vec::new()).unwrap_or_else(|| panic!("no boat"));
    let mut ps = Particles::default();
    run(&mut w, &mut ps, &mut ph, 600);
    let (gunwale, _) = rows(&ph, id);
    let sea = surface(&w, 40);
    assert!(sea - gunwale >= 2, "gunwale at {gunwale}, sea at {sea}");
    assert!(body(&ph, id).floating);
}
