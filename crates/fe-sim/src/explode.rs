//! Explosions: damage, debris, flung cells, heat, and fire.

use crate::particles::Particles;
use crate::rules::Event;
use crate::world::World;
use fe_core::{Cell, CellPos, Fx, MaterialId, SeqRng, celsius_to_temp, isqrt};
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

/// The hottest an explosion makes the cells it hits, unless they were hotter:
/// a crater wall glows for a while, but repeated small blasts in one place
/// do not heat the rock until it melts.
const BLAST_HEAT: u8 = celsius_to_temp(800);

/// `temp` raised by `by`, to at most [`BLAST_HEAT`].
fn heated(temp: u8, by: i32) -> u8 {
    if temp >= BLAST_HEAT { temp } else { (i32::from(temp) + by).min(i32::from(BLAST_HEAT)) as u8 }
}

/// Blasts at least this wide flash: their core fills with `flash` for a few
/// ticks, which lights everything around, and they throw burning sparks.
pub const FLASH_RADIUS: i32 = 12;

/// Applies one explosion to the world. Returns support events for destroyed structure.
///
/// Air in the blast heats up and, near the center, catches fire; a large
/// blast's core flashes white and throws embers. Solid cells lose hit points
/// and those destroyed become debris, thrown out of the inner part; powders
/// and liquids are thrown too. Explosive solids the blast reaches are set off
/// instead of destroyed, so a stack of charges goes up together on the next
/// tick and merges into one large blast. Debris whose column has ground under
/// it inside the blast bounces up out of the crater; with open space below,
/// as in a shell bursting in the air, it flies straight out, so an air burst
/// is round.
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
    let flash = world.materials().id("flash").filter(|_| r >= FLASH_RADIUS);
    let ember = world.materials().id("ember").filter(|_| r >= FLASH_RADIUS);
    let dust = world.materials().id("dust");
    blast.pos = pos;
    blast.radius = radius;
    blast.power = power;
    let ground = ground_below(world, pos, r, power);
    let throw = Throw { center: pos, radius: r, power };
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
            // Ground under this cell inside the blast, in its column.
            let grounded = dy < ground[(dx + r) as usize];
            if c.is_air() {
                let mut hot = c;
                hot.temp = hot.temp.saturating_add((dmg / 3) as u8);
                if let Some(f) = flash
                    && d * 3 < r
                {
                    hot = world.make_cell(f, p);
                    if let Some(e) = ember
                        && (roll >> 8).is_multiple_of(12)
                    {
                        let mut spark = world.make_cell(e, p);
                        spark.temp = celsius_to_temp(1500);
                        throw.fling(particles, p, falloff, spark, grounded, &mut rng);
                    }
                } else if let Some(f) = fire
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
                    if let Some(e) = m.explode {
                        // A charge the blast reaches goes off with the next tick's cells.
                        let mut armed = c;
                        armed.temp = armed.temp.max(e.at);
                        let _ = world.set(p, armed);
                        continue;
                    }
                    if i32::from(c.value) > dmg {
                        let mut hurt = c;
                        hurt.value = (i32::from(c.value) - dmg) as u8;
                        hurt.temp = heated(hurt.temp, dmg / 4);
                        world.grid.set(p, hurt);
                        continue;
                    }
                    if c.has(Cell::FLAG_IN_BODY) {
                        blast.body_cells_hit += 1;
                    }
                    let mut debris = world.make_cell(m.debris, p);
                    debris.temp = heated(c.temp, dmg / 4);
                    let dm = world.materials().get(m.debris);
                    // The inner part of the blast throws its debris, glowing hot;
                    // the rim crumbles in place.
                    if dm.state.falls() && (d * 3 < r * 2 || roll.is_multiple_of(3)) {
                        debris.temp = heated(c.temp, dmg / 3);
                        // A third of the thrown powder leaves a puff of dust behind.
                        let left = match dust {
                            Some(id)
                                if dm.state == State::Powder && (roll >> 8).is_multiple_of(3) =>
                            {
                                world.make_cell(id, p)
                            }
                            _ => Cell::AIR,
                        };
                        events.extend(world.set(p, left));
                        throw.fling(particles, p, falloff, debris, grounded, &mut rng);
                    } else {
                        events.extend(world.set(p, debris));
                    }
                }
                State::Powder | State::Liquid => {
                    if d * 4 < r * 3 {
                        let _ = world.set(p, Cell::AIR);
                        let mut flung = c;
                        flung.temp = heated(c.temp, dmg / 6);
                        throw.fling(particles, p, falloff, flung, grounded, &mut rng);
                    } else {
                        let mut warm = c;
                        warm.temp = heated(c.temp, dmg / 6);
                        let _ = world.set(p, warm);
                    }
                }
                State::Gas | State::Fire | State::Empty => {}
            }
        }
    }
    events
}

/// For each column of the blast, from `dx = -r`, the topmost row offset whose
/// cells have ground under them: a cell the blast leaves standing, in the
/// column below them inside the blast or just under its rim. Cells above that
/// row bounce their debris up; cells at or below it have open space under
/// them and throw it straight out.
fn ground_below(world: &World, pos: CellPos, r: i32, power: u8) -> Vec<i32> {
    let mats = world.materials();
    (-r..=r)
        .map(|dx| {
            let h = isqrt((r * r - dx * dx) as u64) as i32;
            for dy in (-h..=h + 1).rev() {
                let c = world.get(CellPos::new(pos.x + dx, pos.y + dy));
                if c.is_air() {
                    continue;
                }
                let m = mats.get(c.material());
                let falloff = r - isqrt((dx * dx + dy * dy) as u64) as i32;
                let dmg = (i32::from(power) * falloff / r).max(1);
                let stays = c.material() == MaterialId::BOUNDARY
                    || m.indestructible
                    || dy > h
                    || match m.state {
                        State::Static => m.explode.is_some() || i32::from(c.value) > dmg,
                        State::Powder | State::Liquid => (dx * dx + dy * dy) * 16 >= r * r * 9,
                        State::Gas | State::Fire | State::Empty => false,
                    };
                // Only solids and fluids that stay hold debris up; gas does not.
                let holds = !matches!(m.state, State::Gas | State::Fire | State::Empty);
                if stays && holds {
                    return dy;
                }
            }
            -h - 1
        })
        .collect()
}

/// How one blast throws cells.
struct Throw {
    center: CellPos,
    radius: i32,
    power: u8,
}

impl Throw {
    /// Throws a cell out of the blast as a particle.
    ///
    /// Speed is `power * falloff / (80 * radius)` cells per tick, at least 0.25,
    /// and at most 3 (18 m/s) for a blast of radius 18 or less, rising to 6
    /// for the largest blasts. A `grounded` cell, one with ground under it in
    /// the blast, flies up and outward with an upward bias of a quarter of that
    /// speed; debris below the center bounces back up out of the crater. At the
    /// world's gravity the highest debris of a small blast peaks about 260 cells
    /// above it. Other cells fly straight out from the center.
    fn fling(
        &self,
        particles: &mut Particles,
        p: CellPos,
        falloff: i32,
        cell: Cell,
        grounded: bool,
        rng: &mut SeqRng,
    ) {
        let (dx, dy) = (p.x - self.center.x, p.y - self.center.y);
        let len = i64::from((isqrt((dx * dx + dy * dy) as u64) as i32).max(1));
        let cap = Fx(3 << 16) + Fx::from_ratio(3 * (self.radius - 18).clamp(0, 30), 30);
        let speed = Fx::from_ratio(i32::from(self.power) * falloff, 80 * self.radius.max(1))
            .clamp(Fx(1 << 14), cap);
        let jitter = |rng: &mut SeqRng| Fx(rng.range_i32(-(1 << 14), 1 << 14));
        let along = |num: i64, den: i64| Fx((i64::from(speed.0) * num / (len * den)) as i32);
        let vx = along(i64::from(dx), 1) + jitter(rng);
        let vy = if grounded {
            let up = if dy > 0 { along(-3 * i64::from(dy), 10) } else { along(i64::from(dy), 1) };
            up - Fx(speed.0 / 4) + jitter(rng)
        } else {
            along(i64::from(dy), 1) + jitter(rng)
        };
        particles.spawn(
            Fx::from_int(p.x) + Fx(1 << 15),
            Fx::from_int(p.y) + Fx(1 << 15),
            vx,
            vy,
            cell,
        );
    }
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
