//! Free particles: cells that left the grid and fly until they land.
//!
//! A particle's cell keeps living in flight: it trades heat with the air it
//! passes through, burns down if it has a lifetime, and melts, freezes or
//! catches fire like a cell in the grid. A molten speck cools from white
//! through orange to a dull red and freezes to scrap before it lands, and an
//! ember thrown by a blast burns out in the air.

use crate::rules::{heat_split, new_cell};
use crate::world::World;
use fe_core::{AMBIENT_TEMP, Cell, CellPos, Fx, MaterialId, cell_rand};
use fe_materials::{CHANCE_ONE, State, Transition};
use serde::{Deserialize, Serialize};

/// Gravity on particles, in cells per tick squared: the world's [`fe_core::GRAVITY`].
pub const PARTICLE_GRAVITY: Fx = fe_core::GRAVITY;
/// Fastest particle speed on each axis, in cells per tick.
pub const PARTICLE_MAX_SPEED: i32 = 40;

/// A cell in flight.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Particle {
    /// Spawn order. Particles update in this order.
    pub id: u32,
    /// Position in cells.
    pub x: Fx,
    /// Position in cells, growing downward.
    pub y: Fx,
    /// Velocity in cells per tick.
    pub vx: Fx,
    /// Velocity in cells per tick.
    pub vy: Fx,
    /// The cell that lands when the particle stops.
    pub cell: Cell,
    /// Ticks in flight.
    pub age: u16,
    /// Landings that found no room, after which the particle was thrown back up.
    pub retries: u8,
}

/// All particles in flight, in spawn order.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Particles {
    list: Vec<Particle>,
    next_id: u32,
    lost: u64,
}

/// A burning particle loses this fraction of its speed to the air each tick
/// (one part in 16), which caps its fall at about 2.6 m/s.
const FIRE_DRAG: i32 = 16;

/// Landings a particle may retry before it is pushed up its column.
const LAND_RETRIES: u8 = 30;

impl Particles {
    /// Number of particles in flight.
    pub fn len(&self) -> usize {
        self.list.len()
    }

    /// Returns true when nothing is in flight.
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// Particles in spawn order.
    pub fn iter(&self) -> impl Iterator<Item = &Particle> {
        self.list.iter()
    }

    /// Launches a cell from a position with a velocity in cells per tick.
    pub fn spawn(&mut self, x: Fx, y: Fx, vx: Fx, vy: Fx, cell: Cell) {
        let lim = Fx::from_int(PARTICLE_MAX_SPEED);
        self.list.push(Particle {
            id: self.next_id,
            x,
            y,
            vx: vx.clamp(-lim, lim),
            vy: vy.clamp(-lim, lim),
            cell: cell.with_flag(Cell::FLAG_IN_BODY, false),
            age: 0,
            retries: 0,
        });
        self.next_id = self.next_id.wrapping_add(1);
    }

    /// Moves every particle one tick and lands the ones that hit something.
    ///
    /// Returns the number that landed.
    pub fn step(&mut self, world: &mut World) -> usize {
        let mut landed = 0;
        let lim = Fx::from_int(PARTICLE_MAX_SPEED);
        let list = std::mem::take(&mut self.list);
        let mut keep = Vec::with_capacity(list.len());
        for mut p in list {
            p.vy = (p.vy + PARTICLE_GRAVITY).clamp(-lim, lim);
            if world.materials().get(p.cell.material()).state == State::Fire {
                // Embers and flames are light: the air slows them, so sparks
                // from a blast fly out, stall and drift down.
                p.vx = Fx(p.vx.0 - p.vx.0 / FIRE_DRAG);
                p.vy = Fx(p.vy.0 - p.vy.0 / FIRE_DRAG);
            }
            p.age = p.age.saturating_add(1);
            match fly(world, &mut p) {
                Flight::Airborne => match burn(world, &mut p) {
                    InFlight::Flying => keep.push(p),
                    InFlight::Released(cell) => {
                        // Smoke or steam joins the grid where the particle is.
                        let at = CellPos::new(p.x.floor(), p.y.floor());
                        if !cell.is_air() && world.get(at).is_air() {
                            let _ = world.set(at, cell);
                        }
                    }
                },
                Flight::Landed(at) => {
                    if land(world, at, p) {
                        landed += 1;
                    } else if p.retries < LAND_RETRIES {
                        // No room here: bounce up and try again further on.
                        p.retries += 1;
                        p.vy = Fx(-(p.vy.0.abs() / 2));
                        keep.push(p);
                    } else if land_up_column(world, at, p) {
                        landed += 1;
                    } else {
                        self.lost += 1;
                    }
                }
                Flight::Lost => {}
            }
        }
        self.list = keep;
        landed
    }

    /// Particles that found no room to land anywhere and were dropped, over the run.
    pub const fn lost(&self) -> u64 {
        self.lost
    }

    /// Hash of every particle in order.
    pub fn hash_into(&self, h: &mut fe_core::StableHasher) {
        h.write_u64(self.list.len() as u64);
        h.write_u32(self.next_id);
        for p in &self.list {
            h.write_u32(p.id);
            h.write_i32(p.x.0);
            h.write_i32(p.y.0);
            h.write_i32(p.vx.0);
            h.write_i32(p.vy.0);
            h.write_u32(p.cell.to_bits());
            h.write_u32(u32::from(p.age) | u32::from(p.retries) << 16);
        }
    }
}

enum Flight {
    Airborne,
    Landed(CellPos),
    Lost,
}

fn passable(world: &World, p: CellPos) -> Option<bool> {
    let c = world.get(p);
    if c.material() == MaterialId::BOUNDARY {
        return None;
    }
    if c.is_air() {
        return Some(true);
    }
    let m = world.materials().get(c.material());
    Some(!c.has(Cell::FLAG_IN_BODY) && matches!(m.state, State::Gas | State::Fire))
}

/// Traces the particle's path one cell at a time.
fn fly(world: &World, p: &mut Particle) -> Flight {
    let steps = (p.vx.abs().0.max(p.vy.abs().0) as u32).div_ceil(Fx::ONE.0 as u32).max(1) as i32;
    let (sx, sy) = (Fx(p.vx.0 / steps), Fx(p.vy.0 / steps));
    let mut last = CellPos::new(p.x.floor(), p.y.floor());
    for _ in 0..steps {
        let (nx, ny) = (p.x + sx, p.y + sy);
        let cell = CellPos::new(nx.floor(), ny.floor());
        if cell != last {
            match passable(world, cell) {
                None => return Flight::Lost,
                Some(false) => return Flight::Landed(last),
                Some(true) => {}
            }
        }
        p.x = nx;
        p.y = ny;
        last = cell;
    }
    // A particle that barely moves has come to rest, unless it is only at the top
    // of its arc with open space below it.
    if p.age > 4
        && p.vx.abs().0 < Fx::ONE.0 / 8
        && p.vy.abs().0 < Fx::ONE.0 / 8
        && passable(world, CellPos::new(last.x, last.y + 1)) != Some(true)
    {
        return Flight::Landed(last);
    }
    Flight::Airborne
}

/// What a tick in flight did to a particle's cell.
enum InFlight {
    /// It flies on, perhaps as another material.
    Flying,
    /// It became a gas, or nothing: it leaves the flight as this cell.
    Released(Cell),
}

/// One tick in flight for a particle's cell, following the grid's cell rules:
/// it trades heat with the cell it is passing through (air, gas or fire), the
/// material's own cooling and held temperature apply, a lifetime counts down,
/// and it melts, freezes or catches fire. A liquid that freezes in the air
/// becomes its solid's debris, as it would falling in the grid. Explosives do
/// not go off in flight; they do when they land.
fn burn(world: &World, p: &mut Particle) -> InFlight {
    let mats = world.materials();
    let id = p.id;
    let rnd = |salt: u32| cell_rand(world.seed(), world.tick(), id as i32, -0x5f1, salt);
    let mut cell = p.cell;
    let m = mats.get(cell.material());
    let around = world.get(CellPos::new(p.x.floor(), p.y.floor()));
    let am = mats.get(around.material());
    let diff = i32::from(around.temp) - i32::from(cell.temp);
    let k = i32::from(m.conductivity.min(am.conductivity).min(128));
    if diff.abs() >= 2 && k > 0 {
        let (ca, cb) = (i32::from(m.heat_capacity.max(1)), i32::from(am.heat_capacity.max(1)));
        let (q, _) = heat_split(diff, k, ca, cb, rnd(0), rnd(1));
        cell.temp = (i32::from(cell.temp) + q).clamp(0, 255) as u8;
    }
    if m.hold_temp > cell.temp {
        cell.temp = m.hold_temp;
    }
    if m.cooling > 0 && cell.temp != AMBIENT_TEMP && ((rnd(2) >> 24) as u8) < m.cooling {
        cell.temp = if cell.temp > AMBIENT_TEMP { cell.temp - 1 } else { cell.temp + 1 };
    }
    // The cell becomes another material: it flies on unless that is a gas.
    let mut change = |into: MaterialId, temp: u8| {
        let c = new_cell(mats, into, Some(temp), rnd(4));
        if c.is_air() || mats.get(into).state == State::Gas {
            return InFlight::Released(c);
        }
        p.cell = c;
        InFlight::Flying
    };
    if m.lifetime.is_some()
        && (m.lifetime_rate >= CHANCE_ONE || (rnd(3) & 0xffff) < m.lifetime_rate)
    {
        if cell.value <= 1 {
            let into = match m.residue {
                Some((res, chance)) if (rnd(5) & 0xffff) < chance => res,
                _ => m.decay,
            };
            return change(into, cell.temp);
        }
        cell.value -= 1;
    }
    let latent_done = |t: Transition, temp: u8| {
        t.latent == 0 || rnd(6) % u32::from(t.latent) < u32::from(temp.abs_diff(t.at))
    };
    if let Some(t) = m.melt
        && cell.temp >= t.at
    {
        if latent_done(t, cell.temp) {
            return change(t.into, cell.temp);
        }
        cell.temp = t.at;
    }
    if let Some(t) = m.freeze
        && cell.temp <= t.at
    {
        if latent_done(t, cell.temp) {
            let target = mats.get(t.into);
            let unsupported = m.state == State::Liquid && target.state == State::Static;
            let into = if unsupported && target.debris != MaterialId::AIR {
                target.debris
            } else {
                t.into
            };
            return change(into, cell.temp);
        }
        cell.temp = t.at;
    }
    if let Some(ig) = m.ignite
        && cell.temp >= ig.at
        && (rnd(7) & 0xffff) < ig.chance
    {
        return change(ig.into, cell.temp);
    }
    p.cell = cell;
    InFlight::Flying
}

/// The cell a particle turns into when it lands.
fn landing_cell(world: &World, p: &Particle) -> Cell {
    let m = world.materials().get(p.cell.material());
    let mut cell = p.cell;
    if m.state.falls() && !m.has_lifetime() {
        cell.value = (p.vy.0.max(0) >> 12).min(255) as u8;
    }
    cell
}

/// Puts the particle's cell back into the grid at or near `at`. Returns false if
/// there is no air within six cells.
fn land(world: &mut World, at: CellPos, p: Particle) -> bool {
    let cell = landing_cell(world, &p);
    // Look for air at the landing spot, then above it, then in a small ring.
    let candidates = [(0, 0), (0, -1), (-1, 0), (1, 0), (0, -2), (-1, -1), (1, -1), (0, -3)];
    for (dx, dy) in candidates {
        let q = CellPos::new(at.x + dx, at.y + dy);
        if world.get(q).is_air() {
            let _ = world.set(q, cell);
            return true;
        }
    }
    // Then in rings out to six cells, the upper half of each ring first.
    for r in 2..=6i32 {
        for upper in [true, false] {
            for dy in -r..=r {
                if (dy <= 0) != upper {
                    continue;
                }
                for dx in -r..=r {
                    if dx.abs() != r && dy.abs() != r {
                        continue;
                    }
                    let q = CellPos::new(at.x + dx, at.y + dy);
                    if world.get(q).is_air() {
                        let _ = world.set(q, cell);
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Last resort for a particle buried in a crowd: the first air up its column.
fn land_up_column(world: &mut World, at: CellPos, p: Particle) -> bool {
    let cell = landing_cell(world, &p);
    for k in 0..=64 {
        let q = CellPos::new(at.x, at.y - k);
        let c = world.get(q);
        if c.material() == MaterialId::BOUNDARY {
            return false;
        }
        if c.is_air() {
            let _ = world.set(q, cell);
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use fe_core::IRect;
    use std::sync::Arc;

    #[test]
    fn particle_lands_on_floor() {
        let mats = Arc::new(fe_materials::base_set());
        let sand = mats.id("sand").unwrap_or_default();
        let stone = mats.id("stone").unwrap_or_default();
        let mut w = World::new(mats, 1, 1);
        w.ensure_rect(IRect::new(0, 0, 128, 128), Cell::AIR);
        w.fill_rect(IRect::new(0, 100, 128, 128), stone);
        let mut ps = Particles::default();
        let c = w.make_cell(sand, CellPos::new(0, 0));
        ps.spawn(Fx::from_int(10), Fx::from_int(40), Fx(1 << 14), Fx::from_int(-1), c);
        for _ in 0..200 {
            ps.step(&mut w);
        }
        assert!(ps.is_empty());
        assert_eq!(w.count(IRect::new(0, 0, 128, 100), sand), 1);
        assert_eq!(w.count(IRect::new(0, 99, 128, 100), sand), 1);
    }

    /// A tall empty world and one particle launched straight up from near
    /// its bottom, rising about 650 cells before it falls back.
    fn launch(name: &str, celsius: i32) -> (World, Particles) {
        let mats = Arc::new(fe_materials::base_set());
        let id = mats.id(name).unwrap_or_else(|| panic!("no {name}"));
        let mut w = World::new(mats, 9, 1);
        w.ensure_rect(IRect::new(0, 0, 64, 1088), Cell::AIR);
        let mut c = w.make_cell(id, CellPos::new(0, 0));
        c.temp = fe_core::celsius_to_temp(celsius);
        let mut ps = Particles::default();
        ps.spawn(Fx::from_int(32), Fx::from_int(1000), Fx::ZERO, Fx::from_int(-6), c);
        (w, ps)
    }

    fn fly_for(w: &mut World, ps: &mut Particles, ticks: u32) -> Option<Particle> {
        for _ in 0..ticks {
            ps.step(w);
            w.finish_tick();
        }
        ps.iter().next().copied()
    }

    #[test]
    fn a_molten_speck_cools_and_freezes_to_scrap_in_flight() {
        let (mut w, mut ps) = launch("molten_metal", 1900);
        let (molten, scrap) = (w.materials().id("molten_metal"), w.materials().id("scrap"));
        let c = |p: Option<Particle>| p.unwrap_or_else(|| panic!("landed early")).cell;
        let early = c(fly_for(&mut w, &mut ps, 40));
        assert_eq!(Some(early.material()), molten);
        let t = fe_core::temp_to_celsius(early.temp);
        assert!((1400..1880).contains(&t), "{t} C after 40 ticks");
        let later = c(fly_for(&mut w, &mut ps, 260));
        assert_eq!(Some(later.material()), scrap, "a speck in the air freezes to scrap");
        let t = fe_core::temp_to_celsius(later.temp);
        assert!(t < 1100, "{t} C after 300 ticks");
    }

    #[test]
    fn an_ember_burns_out_in_flight() {
        let (mut w, mut ps) = launch("ember", 700);
        let ember = w.materials().id("ember");
        let p = fly_for(&mut w, &mut ps, 90);
        assert!(p.is_none_or(|p| Some(p.cell.material()) != ember), "{p:?}");
    }

    #[test]
    fn hot_splinters_catch_fire_in_flight() {
        let (mut w, mut ps) = launch("splinters", 600);
        let fire = w.materials().id("fire");
        let p = fly_for(&mut w, &mut ps, 30);
        // It burned in the air: it is fire now, or already smoke.
        assert!(p.is_none_or(|p| Some(p.cell.material()) == fire), "{p:?}");
    }

    #[test]
    fn particle_leaving_world_is_dropped() {
        let mats = Arc::new(fe_materials::base_set());
        let sand = mats.id("sand").unwrap_or_default();
        let mut w = World::new(mats, 1, 1);
        w.ensure_rect(IRect::new(0, 0, 64, 64), Cell::AIR);
        let mut ps = Particles::default();
        let c = w.make_cell(sand, CellPos::new(0, 0));
        ps.spawn(Fx::from_int(10), Fx::from_int(10), Fx::from_int(-8), Fx::ZERO, c);
        for _ in 0..10 {
            ps.step(&mut w);
        }
        assert!(ps.is_empty());
    }
}
