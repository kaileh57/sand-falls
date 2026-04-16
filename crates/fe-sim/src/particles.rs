//! Free particles: cells that left the grid and fly until they land.

use crate::world::World;
use fe_core::{Cell, CellPos, Fx, MaterialId};
use fe_materials::State;
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
}

/// All particles in flight, in spawn order.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Particles {
    list: Vec<Particle>,
    next_id: u32,
}

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
            p.age = p.age.saturating_add(1);
            match fly(world, &mut p) {
                Flight::Airborne => keep.push(p),
                Flight::Landed(at) => {
                    land(world, at, p);
                    landed += 1;
                }
                Flight::Lost => {}
            }
        }
        self.list = keep;
        landed
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
            h.write_u32(u32::from(p.age));
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

/// Puts the particle's cell back into the grid at or near `at`.
fn land(world: &mut World, at: CellPos, p: Particle) {
    let m = world.materials().get(p.cell.material());
    let mut cell = p.cell;
    if m.state.falls() && !m.has_lifetime() {
        cell.value = (p.vy.0.max(0) >> 12).min(255) as u8;
    }
    // Look for air at the landing spot, then above it, then in a small ring.
    let candidates = [(0, 0), (0, -1), (-1, 0), (1, 0), (0, -2), (-1, -1), (1, -1), (0, -3)];
    for (dx, dy) in candidates {
        let q = CellPos::new(at.x + dx, at.y + dy);
        if world.get(q).is_air() {
            let _ = world.set(q, cell);
            return;
        }
    }
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
