//! Explosions: damage, debris, flung cells, heat, and fire.

use crate::particles::Particles;
use crate::rules::Event;
use crate::world::World;
use fe_core::{Cell, CellPos, Fx, MaterialId, SeqRng, isqrt};
use fe_materials::State;

/// What an explosion did, for systems outside the cell grid.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Blast {
    /// Center of the blast.
    pub pos: CellPos,
    /// Radius in cells.
    pub radius: u8,
    /// Damage at the center.
    pub power: u8,
    /// Stamped body cells that were destroyed.
    pub body_cells_hit: u32,
}

/// Applies one explosion to the world. Returns support events for destroyed structure.
pub fn explode(
    world: &mut World,
    particles: &mut Particles,
    pos: CellPos,
    radius: u8,
    power: u8,
    blast: &mut Blast,
) -> Vec<Event> {
    let mut events = Vec::new();
    let r = i32::from(radius.max(1));
    let mut rng = SeqRng::new(
        world.seed()
            ^ world.tick().rotate_left(17)
            ^ (u64::from(pos.x as u32) << 32 | u64::from(pos.y as u32)),
    );
    let fire = world.materials().id("fire");
    blast.pos = pos;
    blast.radius = radius;
    blast.power = power;
    for dy in -r..=r {
        for dx in -r..=r {
            let d2 = dx * dx + dy * dy;
            if d2 > r * r {
                continue;
            }
            let d = isqrt(d2 as u64) as i32;
            let falloff = r - d;
            let dmg = (i32::from(power) * falloff / r).max(1);
            let p = CellPos::new(pos.x + dx, pos.y + dy);
            let c = world.get(p);
            if c.material() == MaterialId::BOUNDARY {
                continue;
            }
            let roll = rng.next_u32();
            if c.is_air() {
                let mut hot = c;
                hot.temp = hot.temp.saturating_add((dmg / 3) as u8);
                if let Some(f) = fire
                    && d * 2 < r
                    && roll.is_multiple_of(4)
                {
                    hot = world.make_cell(f, p);
                }
                let _ = world.set(p, hot);
                continue;
            }
            let m = world.materials().get(c.material()).clone();
            if m.indestructible {
                continue;
            }
            match m.state {
                State::Static => {
                    if i32::from(c.value) > dmg {
                        let mut hurt = c;
                        hurt.value = (i32::from(c.value) - dmg) as u8;
                        hurt.temp = hurt.temp.saturating_add((dmg / 4) as u8);
                        world.grid.set(p, hurt);
                        continue;
                    }
                    if c.has(Cell::FLAG_IN_BODY) {
                        blast.body_cells_hit += 1;
                    }
                    let mut debris = world.make_cell(m.debris, p);
                    debris.temp = c.temp.saturating_add((dmg / 4) as u8);
                    let dm = world.materials().get(m.debris);
                    // The inner part of the blast throws its debris; the rim crumbles in place.
                    if dm.state.falls() && (d * 3 < r * 2 || roll.is_multiple_of(3)) {
                        events.extend(world.set(p, Cell::AIR));
                        fling(particles, pos, p, falloff, power, debris, &mut rng);
                    } else {
                        events.extend(world.set(p, debris));
                    }
                }
                State::Powder | State::Liquid => {
                    if d * 4 < r * 3 {
                        let _ = world.set(p, Cell::AIR);
                        let mut flung = c;
                        flung.temp = c.temp.saturating_add((dmg / 6) as u8);
                        fling(particles, pos, p, falloff, power, flung, &mut rng);
                    } else {
                        let mut warm = c;
                        warm.temp = c.temp.saturating_add((dmg / 6) as u8);
                        let _ = world.set(p, warm);
                    }
                }
                State::Gas | State::Fire | State::Empty => {}
            }
        }
    }
    events
}

fn fling(
    particles: &mut Particles,
    center: CellPos,
    p: CellPos,
    falloff: i32,
    power: u8,
    cell: Cell,
    rng: &mut SeqRng,
) {
    let (dx, dy) = (p.x - center.x, p.y - center.y);
    let len = (isqrt((dx * dx + dy * dy) as u64) as i32).max(1);
    // Speed in cells per tick, scaled by distance from the edge of the blast.
    let speed = (i32::from(power) * falloff / 96).clamp(1, 24);
    let jitter = |rng: &mut SeqRng| Fx(rng.range_i32(-(1 << 15), 1 << 15));
    let vx = Fx::from_ratio(dx * speed, len) + jitter(rng);
    // Debris below the center bounces back up out of the crater instead of
    // burying itself, so most of it flies up and outward.
    let up = if dy > 0 { -dy * speed * 3 / 10 } else { dy * speed };
    let vy = Fx::from_ratio(up, len) - Fx::from_int(speed / 3 + 1) + jitter(rng);
    particles.spawn(Fx::from_int(p.x) + Fx(1 << 15), Fx::from_int(p.y) + Fx(1 << 15), vx, vy, cell);
}

#[cfg(test)]
mod tests {
    use super::*;
    use fe_core::IRect;
    use std::sync::Arc;

    #[test]
    fn explosion_digs_a_crater_and_flings_debris() {
        let mats = Arc::new(fe_materials::base_set());
        let stone = mats.id("stone").unwrap_or_default();
        let mut w = World::new(mats, 3, 1);
        w.ensure_rect(IRect::new(0, 0, 128, 128), Cell::AIR);
        w.fill_rect(IRect::new(0, 64, 128, 128), stone);
        let mut ps = Particles::default();
        let mut blast = Blast::default();
        let before = w.count(IRect::new(0, 0, 128, 128), stone);
        let events = explode(&mut w, &mut ps, CellPos::new(64, 64), 12, 200, &mut blast);
        let after = w.count(IRect::new(0, 0, 128, 128), stone);
        assert!(after < before - 100, "before {before} after {after}");
        assert!(!ps.is_empty());
        assert!(!events.is_empty());
    }
}
