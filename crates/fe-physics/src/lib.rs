//! Pixel rigid bodies.
//!
//! A body is a small grid of cells that moves as one piece. Rapier moves it; the
//! cell world sees it as static cells flagged [`Cell::FLAG_IN_BODY`], stamped in
//! at the body's position every tick. Damage to stamped cells flows back into the
//! body, which splits when it is no longer connected.
//!
//! Rapier uses floats. It runs with `enhanced-determinism` on one thread, and
//! every float computed here uses plain IEEE arithmetic with no transcendental
//! functions, so results match across platforms that follow IEEE 754.
#![forbid(unsafe_code)]

mod body;
mod terrain;

pub use body::{BodyInfo, JointKind, MIN_BODY_CELLS, PixelBody};

use fe_core::{Cell, CellPos, ChunkPos, Fx, SeqRng, StableHasher};
use fe_materials::State;
use fe_sim::{Blast, Particles, World};
use rapier2d::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Cells per meter in physics space: [`fe_core::CELLS_PER_METER`].
pub const CELLS_PER_METER: f32 = fe_core::CELLS_PER_METER as f32;
/// Gravity in meters per second squared: [`fe_core::GRAVITY`] (7/256 cells per
/// tick squared, 98.4 cells/s^2) at [`CELLS_PER_METER`], so bodies fall exactly
/// as fast as cells, particles and characters.
pub const GRAVITY: f32 = 9.843_75;
/// Ticks a small body must rest before it merges into terrain.
pub const MERGE_AFTER_TICKS: u32 = 300;
/// Bodies at or below this size merge into terrain after resting.
pub const MERGE_MAX_CELLS: usize = 2048;

/// A joint between two bodies, with the force that breaks it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JointInfo {
    /// Rapier handle.
    pub handle: ImpulseJointHandle,
    /// First body id.
    pub a: u32,
    /// Second body id.
    pub b: u32,
    /// Break force in newtons. 0 never breaks.
    pub break_force: f32,
}

/// Counters from the last physics step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhysicsStats {
    /// Bodies alive.
    pub bodies: u32,
    /// Bodies awake.
    pub awake: u32,
    /// Terrain chunk colliders alive.
    pub terrain_colliders: u32,
    /// Bodies created this tick (detachment, splits, fracture).
    pub created: u32,
    /// Bodies removed this tick (merged, destroyed, split).
    pub removed: u32,
    /// Joints broken this tick.
    pub joints_broken: u32,
}

/// All bodies, joints and terrain colliders.
#[derive(Serialize, Deserialize)]
pub struct Physics {
    rapier: PhysicsWorld,
    bodies: Vec<PixelBody>,
    next_id: u32,
    terrain: BTreeMap<ChunkPos, terrain::TerrainCollider>,
    terrain_body: RigidBodyHandle,
    joints: Vec<JointInfo>,
    #[serde(skip)]
    stats: PhysicsStats,
}

impl Default for Physics {
    fn default() -> Self {
        Self::new()
    }
}

impl Physics {
    /// Creates an empty physics world.
    pub fn new() -> Self {
        let mut rapier = PhysicsWorld::new();
        rapier.gravity = Vector::new(0.0, GRAVITY);
        rapier.integration_parameters.dt = 1.0 / fe_core::TICKS_PER_SECOND as f32;
        let terrain_body = rapier.bodies.insert(RigidBodyBuilder::fixed());
        Self {
            rapier,
            bodies: Vec::new(),
            next_id: 0,
            terrain: BTreeMap::new(),
            terrain_body,
            joints: Vec::new(),
            stats: PhysicsStats::default(),
        }
    }

    /// Counters from the last step.
    pub const fn stats(&self) -> PhysicsStats {
        self.stats
    }

    /// Bodies in id order.
    pub fn bodies(&self) -> &[PixelBody] {
        &self.bodies
    }

    /// Joints in creation order.
    pub fn joints(&self) -> &[JointInfo] {
        &self.joints
    }

    /// Summary of a body's motion, or `None` if the id is not alive.
    pub fn body_info(&self, id: u32) -> Option<BodyInfo> {
        let b = self.bodies.iter().find(|b| b.id == id)?;
        Some(b.info(&self.rapier))
    }

    /// Creates a body from cells at world positions. Returns its id, or `None` if
    /// the cells are too few (they are then dropped as debris by the caller).
    ///
    /// The cells must already be removed from the world.
    pub fn create_body(
        &mut self,
        world: &World,
        cells: &[(CellPos, Cell)],
        linvel: (f32, f32),
        angvel: f32,
    ) -> Option<u32> {
        if cells.len() < MIN_BODY_CELLS {
            return None;
        }
        let id = self.next_id;
        let body = PixelBody::from_world_cells(id, world, &mut self.rapier, cells, linvel, angvel)?;
        self.next_id += 1;
        self.bodies.push(body);
        self.stats.created += 1;
        Some(id)
    }

    /// Cuts the cells inside a rectangle out of the world and turns them into a body.
    pub fn carve_body(&mut self, world: &mut World, rect: fe_core::IRect) -> Option<u32> {
        let mut cells = Vec::new();
        for y in rect.min.y..rect.max.y {
            for x in rect.min.x..rect.max.x {
                let p = CellPos::new(x, y);
                let c = world.get(p);
                if is_body_material(world, c) {
                    cells.push((p, c));
                }
            }
        }
        if cells.len() < MIN_BODY_CELLS {
            return None;
        }
        for &(p, _) in &cells {
            world.grid.set(p, Cell::AIR);
            if let Some(ch) = world.grid.chunk(p.chunk()) {
                ch.flag_solid_changed();
            }
        }
        self.create_body(world, &cells, (0.0, 0.0), 0.0)
    }

    /// Changes a body's velocity by `(vx, vy)` cells per second.
    pub fn push_body(&mut self, id: u32, vx: f32, vy: f32) {
        if let Some(b) = self.bodies.iter().find(|b| b.id == id)
            && let Some(rb) = self.rapier.bodies.get_mut(b.handle)
        {
            let v = rb.linvel() + Vector::new(vx, vy) / CELLS_PER_METER;
            rb.set_linvel(v, true);
        }
    }

    /// Connects two bodies with a joint at a world position.
    pub fn add_joint(
        &mut self,
        a: u32,
        b: u32,
        at: CellPos,
        kind: JointKind,
        break_force: f32,
    ) -> Option<ImpulseJointHandle> {
        let ba = self.bodies.iter().find(|x| x.id == a)?;
        let bb = self.bodies.iter().find(|x| x.id == b)?;
        let (ha, hb) = (ba.handle, bb.handle);
        let pa = self.rapier.bodies.get(ha)?.position().inverse_transform_point(cell_to_m(at));
        let pb = self.rapier.bodies.get(hb)?.position().inverse_transform_point(cell_to_m(at));
        let joint: GenericJoint = match kind {
            JointKind::Hinge => {
                RevoluteJointBuilder::new().local_anchor1(pa).local_anchor2(pb).into()
            }
            JointKind::Fixed => FixedJointBuilder::new().local_anchor1(pa).local_anchor2(pb).into(),
        };
        let handle = self.rapier.impulse_joints.insert(ha, hb, joint, true);
        self.joints.push(JointInfo { handle, a, b, break_force });
        Some(handle)
    }

    /// Pins a body to the world with a hinge at a world position.
    pub fn pin_body(
        &mut self,
        id: u32,
        at: CellPos,
        break_force: f32,
    ) -> Option<ImpulseJointHandle> {
        let b = self.bodies.iter().find(|x| x.id == id)?;
        let h = b.handle;
        let pb = self.rapier.bodies.get(h)?.position().inverse_transform_point(cell_to_m(at));
        let joint: GenericJoint =
            RevoluteJointBuilder::new().local_anchor1(cell_to_m(at)).local_anchor2(pb).into();
        let handle = self.rapier.impulse_joints.insert(self.terrain_body, h, joint, true);
        self.joints.push(JointInfo { handle, a: u32::MAX, b: id, break_force });
        Some(handle)
    }

    /// Reads damage from stamped cells back into bodies, splitting broken ones.
    ///
    /// Call after the cell passes and explosions, before [`Physics::unstamp`].
    pub fn sync(&mut self, world: &mut World, particles: &mut Particles) {
        let mut changed = Vec::new();
        for (i, b) in self.bodies.iter_mut().enumerate() {
            if b.sync_from_world(world) {
                changed.push(i);
            }
        }
        for i in changed.into_iter().rev() {
            self.resplit(world, particles, i, None);
        }
    }

    /// Computes buoyancy and drag from the current stamps, then removes every
    /// moving body's cells from the world.
    pub fn unstamp(&mut self, world: &mut World) {
        for b in &self.bodies {
            b.apply_fluid_forces(&mut self.rapier, world);
        }
        for b in &mut self.bodies {
            if b.needs_restamp(&self.rapier) {
                b.unstamp(world);
            }
        }
    }

    /// Applies blast impulses to bodies near each explosion.
    pub fn apply_blasts(&mut self, blasts: &[Blast]) {
        for blast in blasts {
            let center = cell_to_m(blast.pos);
            let r = f32::from(blast.radius) / CELLS_PER_METER * 2.5;
            for b in &self.bodies {
                let Some(rb) = self.rapier.bodies.get_mut(b.handle) else { continue };
                let delta = rb.translation() - center;
                let d = delta.length();
                if d > r || d <= 1e-4 {
                    continue;
                }
                let strength = f32::from(blast.power) * 0.6 * (1.0 - d / r) * rb.mass().sqrt();
                rb.apply_impulse(delta / d * strength, true);
                rb.apply_torque_impulse(
                    strength * 0.02 * if delta.x > 0.0 { 1.0 } else { -1.0 },
                    true,
                );
            }
        }
    }

    /// Steps the physics world: buoyancy, terrain colliders, Rapier, fracture,
    /// joint breaks and merging. Call between [`Physics::unstamp`] and [`Physics::stamp`].
    pub fn step(&mut self, world: &mut World, particles: &mut Particles) {
        self.stats.created = 0;
        self.stats.removed = 0;
        self.stats.joints_broken = 0;

        terrain::update(
            &mut self.terrain,
            &mut self.rapier,
            self.terrain_body,
            &self.bodies,
            world,
        );

        let (cs, _cr) = std::sync::mpsc::channel();
        let (fs, fr) = std::sync::mpsc::channel();
        let (ts, _tr) = std::sync::mpsc::channel();
        let events = ChannelEventCollector::new(cs, fs, ts);
        self.rapier.step_with_events(&(), &events);

        // Fracture: the strongest impact per body, in body order.
        let mut impacts: BTreeMap<u32, f32> = BTreeMap::new();
        while let Ok(e) = fr.try_recv() {
            for c in [e.collider1, e.collider2] {
                if let Some(b) = self.bodies.iter().find(|b| b.collider == c) {
                    let v = impacts.entry(b.id).or_insert(0.0);
                    *v = v.max(e.max_force_magnitude);
                }
            }
        }
        let mut to_shatter = Vec::new();
        for (&id, &force) in &impacts {
            if let Some(i) = self.bodies.iter().position(|b| b.id == id) {
                let b = &self.bodies[i];
                if b.fracture > 0
                    && let Some(rb) = self.rapier.bodies.get(b.handle)
                    && force > f32::from(b.fracture) * 4.0 * rb.mass()
                {
                    to_shatter.push(i);
                }
            }
        }
        for i in to_shatter.into_iter().rev() {
            let seed =
                world.seed() ^ world.tick().wrapping_mul(0x9e37) ^ u64::from(self.bodies[i].id);
            self.resplit(world, particles, i, Some(seed));
        }

        self.break_joints();
        self.merge_resting(world);
        self.stats.bodies = self.bodies.len() as u32;
        self.stats.awake = self
            .bodies
            .iter()
            .filter(|b| self.rapier.bodies.get(b.handle).is_some_and(|r| !r.is_sleeping()))
            .count() as u32;
        self.stats.terrain_colliders = self.terrain.len() as u32;
    }

    /// Writes every body's cells into the world at its new position.
    pub fn stamp(&mut self, world: &mut World) {
        for b in &mut self.bodies {
            if b.stamped.is_empty() || b.needs_restamp(&self.rapier) {
                b.stamp(world, &self.rapier);
            }
        }
    }

    /// Turns structure that lost support into bodies, or into debris if too small.
    pub fn detach(
        &mut self,
        world: &mut World,
        particles: &mut Particles,
        groups: &[fe_sim::Detached],
    ) {
        for g in groups {
            let cells: Vec<(CellPos, Cell)> = g.cells.iter().map(|&p| (p, world.get(p))).collect();
            for &(p, _) in &cells {
                world.grid.set(p, Cell::AIR);
                if let Some(ch) = world.grid.chunk(p.chunk()) {
                    ch.flag_solid_changed();
                }
            }
            if self.create_body(world, &cells, (0.0, 0.0), 0.0).is_none() {
                drop_as_debris(world, particles, &cells, (0.0, 0.0));
            }
        }
    }

    /// Replaces body `i` with its connected parts, or with Voronoi shards when `shatter` is set.
    fn resplit(
        &mut self,
        world: &mut World,
        particles: &mut Particles,
        i: usize,
        shatter: Option<u64>,
    ) {
        let body = self.bodies.remove(i);
        let Some(rb) = self.rapier.bodies.get(body.handle) else { return };
        let pose = *rb.position();
        let linvel = rb.linvel();
        let angvel = rb.angvel();
        let had_stamp = !body.stamped.is_empty();
        let mut body = body;
        if had_stamp {
            body.unstamp(world);
        }
        let parts = match shatter {
            Some(seed) => body.shards(seed),
            None => body.components(),
        };
        self.remove_rapier_body(&body);
        self.stats.removed += 1;
        let mut rng = SeqRng::new(shatter.unwrap_or(0) ^ u64::from(body.id));
        for part in parts {
            let cells: Vec<(CellPos, Cell)> =
                part.iter().map(|&(lx, ly, c)| (body.local_to_world(&pose, lx, ly), c)).collect();
            let mut v = (linvel.x, linvel.y);
            if shatter.is_some() {
                v.0 += (rng.range_i32(-100, 100) as f32) / 40.0;
                v.1 += (rng.range_i32(-100, 20) as f32) / 40.0;
            }
            if self.create_body(world, &cells, v, angvel).is_none() {
                drop_as_debris(world, particles, &cells, v);
            }
        }
    }

    fn remove_rapier_body(&mut self, body: &PixelBody) {
        self.joints.retain(|j| j.a != body.id && j.b != body.id);
        self.rapier.bodies.remove(
            body.handle,
            &mut self.rapier.islands,
            &mut self.rapier.colliders,
            &mut self.rapier.impulse_joints,
            &mut self.rapier.multibody_joints,
            &mut self.rapier.soft_bodies,
            true,
        );
    }

    fn break_joints(&mut self) {
        let dt = self.rapier.integration_parameters.dt;
        let mut broken = Vec::new();
        for (k, j) in self.joints.iter().enumerate() {
            if j.break_force <= 0.0 {
                continue;
            }
            if let Some(joint) = self.rapier.impulse_joints.get(j.handle) {
                let imp = joint.impulses;
                let force = (imp.x * imp.x + imp.y * imp.y).sqrt() / dt;
                if force > j.break_force {
                    broken.push(k);
                }
            }
        }
        for k in broken.into_iter().rev() {
            let j = self.joints.remove(k);
            self.rapier.impulse_joints.remove(j.handle, true);
            self.stats.joints_broken += 1;
        }
    }

    fn merge_resting(&mut self, world: &mut World) {
        let mut k = 0;
        while k < self.bodies.len() {
            let b = &mut self.bodies[k];
            let resting = self.rapier.bodies.get(b.handle).is_some_and(|rb| {
                rb.is_sleeping() || (rb.linvel().length() < 0.05 && rb.angvel().abs() < 0.05)
            });
            let jointed = self.joints.iter().any(|j| j.a == b.id || j.b == b.id);
            b.rest_ticks = if resting && !jointed { b.rest_ticks + 1 } else { 0 };
            if b.rest_ticks >= MERGE_AFTER_TICKS
                && b.cell_count() <= MERGE_MAX_CELLS
                && !b.stamped.is_empty()
            {
                let b = self.bodies.remove(k);
                b.merge_into(world);
                self.remove_rapier_body(&b);
                self.stats.removed += 1;
                continue;
            }
            k += 1;
        }
    }

    /// Hash of every body, joint and Rapier body state, in id order.
    pub fn hash_into(&self, h: &mut StableHasher) {
        h.write_u64(self.bodies.len() as u64);
        h.write_u32(self.next_id);
        for b in &self.bodies {
            b.hash_into(h, &self.rapier);
        }
        h.write_u64(self.joints.len() as u64);
        for j in &self.joints {
            h.write_u32(j.a);
            h.write_u32(j.b);
            h.write_u32(j.break_force.to_bits());
        }
    }
}

/// Converts a world cell position to its center in physics meters.
pub fn cell_to_m(p: CellPos) -> Vector {
    Vector::new((p.x as f32 + 0.5) / CELLS_PER_METER, (p.y as f32 + 0.5) / CELLS_PER_METER)
}

/// Returns true if a cell can be part of a rigid body.
pub fn is_body_material(world: &World, c: Cell) -> bool {
    if c.is_air() || c.has(Cell::FLAG_IN_BODY) || c.material() == fe_core::MaterialId::BOUNDARY {
        return false;
    }
    let m = world.materials().get(c.material());
    m.state == State::Static && !m.anchored && !m.indestructible
}

/// Puts cells too small to be a body back into the world as debris particles.
fn drop_as_debris(
    world: &mut World,
    particles: &mut Particles,
    cells: &[(CellPos, Cell)],
    v: (f32, f32),
) {
    let vx = Fx((v.0 * CELLS_PER_METER / fe_core::TICKS_PER_SECOND as f32 * 65536.0) as i32);
    let vy = Fx((v.1 * CELLS_PER_METER / fe_core::TICKS_PER_SECOND as f32 * 65536.0) as i32);
    for &(p, c) in cells {
        let m = world.materials().get(c.material());
        let debris_id = m.debris;
        let mut d = world.make_cell(debris_id, p);
        d.temp = c.temp;
        if d.is_air() {
            continue;
        }
        if world.materials().get(debris_id).state.falls() {
            particles.spawn(Fx::from_int(p.x), Fx::from_int(p.y), vx, vy, d);
        } else if world.get(p).is_air() {
            let _ = world.set(p, d);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gravity_matches_the_cell_world() {
        let cells_per_tick2 = fe_core::GRAVITY.0 as f32 / 65536.0;
        let tps = fe_core::TICKS_PER_SECOND as f32;
        let expected = cells_per_tick2 * tps * tps / fe_core::CELLS_PER_METER as f32;
        assert_eq!(GRAVITY.to_bits(), expected.to_bits());
    }
}
