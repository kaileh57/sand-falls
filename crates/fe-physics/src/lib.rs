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
#[cfg(test)]
mod float_tests;
mod terrain;

pub use body::{BodyInfo, JointKind, MAX_BODY_SIDE, MIN_BODY_CELLS, PixelBody};

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
/// Most a body's speed changes from one tick's blasts, in m/s.
pub const MAX_BLAST_DV: f32 = 14.0;
/// Velocity change in one step, in m/s, past which a landing raises dust.
const IMPACT_DUST_DV: f32 = 2.5;

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

/// A joint taken off a body that is splitting, to put back on one of its parts.
struct CarriedJoint {
    /// The body on the other end, or `u32::MAX` for the terrain.
    other: u32,
    /// The splitting body was the joint's first body.
    first: bool,
    /// The splitting body's joint frame in world space.
    world_frame: Pose,
    data: GenericJoint,
    break_force: f32,
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

    /// Resets the per-tick counters. Call at the start of each tick, before any
    /// body is created, split or removed.
    pub const fn begin_tick(&mut self) {
        self.stats.created = 0;
        self.stats.removed = 0;
        self.stats.joints_broken = 0;
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
    ///
    /// Structure removed this way is added to `support_lost`, so the caller can
    /// check what it held up.
    pub fn carve_body(
        &mut self,
        world: &mut World,
        rect: fe_core::IRect,
        support_lost: &mut Vec<CellPos>,
    ) -> Option<u32> {
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
        self.carve_cells(world, &cells, support_lost)
    }

    /// Removes cells from the world and turns them into one body. The cells must
    /// be body material (see [`is_body_material`]). Leaves the world unchanged and
    /// returns `None` if the cells are too few or span too large an area.
    ///
    /// Structure removed this way is added to `support_lost`.
    pub fn carve_cells(
        &mut self,
        world: &mut World,
        cells: &[(CellPos, Cell)],
        support_lost: &mut Vec<CellPos>,
    ) -> Option<u32> {
        if cells.len() < MIN_BODY_CELLS || !fits_in_body(cells) {
            return None;
        }
        for &(p, _) in cells {
            if let Some(fe_sim::Event::SupportLost(q)) = world.set(p, Cell::AIR) {
                support_lost.push(q);
            }
        }
        let id = self.create_body(world, cells, (0.0, 0.0), 0.0)?;
        // Stamp it now: a body carved before the cell passes would otherwise leave
        // a hole for one tick, and water would run into a hull's hold through it.
        let holders = body::Holders::of(&self.bodies, &self.rapier);
        self.stamp_body(world, self.bodies.len() - 1, &holders);
        Some(id)
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
            // A body that is still in one piece keeps its Rapier body and motion;
            // rebuilding it would reset its contacts and kick it.
            if self.bodies[i].components().len() == 1 {
                let shift = self.bodies[i].refresh_shape(world, &mut self.rapier);
                self.shift_joint_frames(self.bodies[i].id, shift);
            } else {
                self.resplit(world, particles, i, None);
            }
        }
    }

    /// Computes buoyancy and drag from the current stamps, then removes every
    /// moving body's cells from the world.
    pub fn unstamp(&mut self, world: &mut World) {
        for b in &mut self.bodies {
            b.apply_fluid_forces(&mut self.rapier, world);
        }
        for b in &mut self.bodies {
            if b.needs_restamp(&self.rapier) {
                b.unstamp(world);
            }
        }
    }

    /// Applies blast impulses to bodies near each explosion.
    ///
    /// The blast reaches out to 2.5 times its radius. Each body inside that circle
    /// gets one impulse at its cell nearest the blast, scaled by how close that
    /// cell is, so a charge at the foot of a tall tower pushes the foot and the
    /// tower tips instead of sliding as a whole.
    ///
    /// The blasts of one tick together change a body's speed by at most
    /// [`MAX_BLAST_DV`]: a pocket of a few hundred gunpowder cells going off at
    /// once throws a hut, not out of the world.
    pub fn apply_blasts(&mut self, blasts: &[Blast]) {
        let mut given = vec![0.0f32; self.bodies.len()];
        for blast in blasts {
            let center = (blast.pos.x as f32 + 0.5, blast.pos.y as f32 + 0.5);
            let r = f32::from(blast.radius) * 2.5;
            for (k, b) in self.bodies.iter().enumerate() {
                let Some(rb) = self.rapier.bodies.get_mut(b.handle) else { continue };
                let pose = *rb.position();
                let Some((px, py, d)) = b.nearest_cell(&pose, center, r) else { continue };
                let mass = rb.mass().max(1e-6);
                let mut strength = f32::from(blast.power) * 0.28 * (1.0 - d / r) * mass.sqrt();
                let room = (MAX_BLAST_DV - given[k]).max(0.0);
                if strength / mass > room {
                    strength = room * mass;
                }
                given[k] += strength / mass;
                if strength <= 0.0 {
                    continue;
                }
                let dir = if d > 1e-3 {
                    Vector::new((px - center.0) / d, (py - center.1) / d)
                } else {
                    Vector::new(0.0, -1.0)
                };
                let at = Vector::new(px / CELLS_PER_METER, py / CELLS_PER_METER);
                rb.apply_impulse_at_point(dir * strength, at, true);
            }
        }
    }

    /// Steps the physics world: buoyancy, terrain colliders, Rapier, fracture,
    /// joint breaks and merging. Call between [`Physics::unstamp`] and [`Physics::stamp`].
    pub fn step(&mut self, world: &mut World, particles: &mut Particles) {
        terrain::update(
            &mut self.terrain,
            &mut self.rapier,
            self.terrain_body,
            &self.bodies,
            world,
        );

        // Fracture by impact: a body shatters when the step changed its velocity,
        // apart from gravity, by more than its strength allows. A sudden stop is an
        // impact; resting under a heavy pile is not.
        let before: Vec<Vector> = self
            .bodies
            .iter()
            .map(|b| self.rapier.bodies.get(b.handle).map_or(Vector::ZERO, RigidBody::linvel))
            .collect();
        self.rapier.step();
        let fall = Vector::new(0.0, GRAVITY * self.rapier.integration_parameters.dt);
        let mut to_shatter = Vec::new();
        for (i, b) in self.bodies.iter().enumerate() {
            let Some(rb) = self.rapier.bodies.get(b.handle) else { continue };
            let dvv = rb.linvel() - before[i] - fall;
            let dv = dvv.length();
            if dv < IMPACT_DUST_DV {
                continue;
            }
            // The impact pushed the body along `dvv`, so it was struck on the
            // side facing the other way. Turn that direction into the body frame.
            let (c, s) = (rb.rotation().cos(), rb.rotation().sin());
            let (wx, wy) = (-dvv.x / dv, -dvv.y / dv);
            let dir = (c * wx + s * wy, -s * wx + c * wy);
            if b.fracture != 0 && dv > f32::from(b.fracture) / 32.0 {
                to_shatter.push((i, dv, dir));
            } else if b.stamped.len() >= 64 {
                // A hard landing that does not break the body still raises dust.
                let pose = *rb.position();
                let seed = world.seed() ^ world.tick().wrapping_mul(0x51d7) ^ u64::from(b.id);
                impact_dust(world, b, &pose, dir, dust_amount(b.cell_count(), dv), seed);
            }
        }
        for (i, dv, dir) in to_shatter.into_iter().rev() {
            let seed =
                world.seed() ^ world.tick().wrapping_mul(0x9e37) ^ u64::from(self.bodies[i].id);
            self.resplit(world, particles, i, Some((seed, dv, dir)));
        }

        self.break_joints();
        self.remove_lost(world);
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
        let holders = body::Holders::of(&self.bodies, &self.rapier);
        for i in 0..self.bodies.len() {
            let b = &self.bodies[i];
            if b.stamped.is_empty() || b.needs_restamp(&self.rapier) {
                self.stamp_body(world, i, &holders);
            }
        }
    }

    /// Stamps body `i`, keeping the fluid it pushes aside out of the air the
    /// other bodies hold.
    fn stamp_body(&mut self, world: &mut World, i: usize, holders: &body::Holders) {
        let (before, rest) = self.bodies.split_at_mut(i);
        let Some((b, after)) = rest.split_first_mut() else { return };
        b.stamp(world, &self.rapier, &body::Others::new(holders, i, before, after));
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

    /// Replaces body `i` with its connected parts, or with Voronoi shards when
    /// `shatter` is set: `(seed, impact speed change, direction to the contact
    /// in the body frame)`.
    ///
    /// Joints on the body move to the part nearest each joint's anchor. A joint
    /// whose anchor part became debris is dropped.
    fn resplit(
        &mut self,
        world: &mut World,
        particles: &mut Particles,
        i: usize,
        shatter: Option<(u64, f32, (f32, f32))>,
    ) {
        let body = self.bodies.remove(i);
        let Some(rb) = self.rapier.bodies.get(body.handle) else { return };
        let pose = *rb.position();
        let linvel = rb.linvel();
        let angvel = rb.angvel();
        let carried = self.joints_of(&body, &pose);
        let had_stamp = !body.stamped.is_empty();
        let mut body = body;
        if had_stamp {
            body.unstamp(world);
        }
        let parts = match shatter {
            Some((seed, _, dir)) => body.shards(seed, body.contact_cell(dir)),
            None => body.components(),
        };
        self.remove_rapier_body(&body);
        self.stats.removed += 1;
        if let Some((seed, dv, dir)) = shatter {
            let amount = dust_amount(body.cell_count(), dv);
            crack_dust(world, &body, &pose, &parts, seed, amount / 2);
            impact_dust(world, &body, &pose, dir, amount / 2, seed ^ 0x1d);
        }
        let mut rng = SeqRng::new(shatter.map_or(0, |s| s.0) ^ u64::from(body.id));
        let mut made: Vec<(u32, Vec<CellPos>)> = Vec::new();
        for part in parts {
            // Each piece moves with the parent's motion at its own center of mass
            // (see `from_local_cells`), plus a small kick, so the wreck does not
            // jump back up after it lands.
            let mut v = linvel;
            if shatter.is_some() {
                v.x += (rng.range_i32(-100, 100) as f32) / 400.0;
                v.y += (rng.range_i32(-100, 100) as f32) / 400.0;
            }
            if part.len() >= MIN_BODY_CELLS {
                let id = self.next_id;
                if let Some(child) = PixelBody::from_local_cells(
                    id,
                    world,
                    &mut self.rapier,
                    &body,
                    &pose,
                    &part,
                    v,
                    angvel,
                ) {
                    self.next_id += 1;
                    let at = part.iter().map(|&(lx, ly, _)| body.local_to_world(&pose, lx, ly));
                    made.push((id, at.collect()));
                    self.bodies.push(child);
                    self.stats.created += 1;
                    continue;
                }
            }
            let cells: Vec<(CellPos, Cell)> =
                part.iter().map(|&(lx, ly, c)| (body.local_to_world(&pose, lx, ly), c)).collect();
            drop_as_debris(world, particles, &cells, (v.x, v.y));
        }
        for c in &carried {
            self.reattach(c, &made);
        }
    }

    /// Moves the frames of a body's joints by `-shift` in its local space, after
    /// its origin moved by `shift`, so every anchor stays at the same world point.
    fn shift_joint_frames(&mut self, id: u32, shift: Vector) {
        if shift == Vector::ZERO {
            return;
        }
        for j in &self.joints {
            let (first, second) = (j.a == id, j.b == id);
            if !first && !second {
                continue;
            }
            let Some(ij) = self.rapier.impulse_joints.get_mut(j.handle, false) else { continue };
            if first {
                ij.data.local_frame1.translation -= shift;
            }
            if second {
                ij.data.local_frame2.translation -= shift;
            }
        }
    }

    /// The joints attached to a body, with the body's anchor frames in world space.
    fn joints_of(&self, body: &PixelBody, pose: &Pose) -> Vec<CarriedJoint> {
        let mut out = Vec::new();
        for j in &self.joints {
            let first = j.a == body.id;
            if !first && j.b != body.id {
                continue;
            }
            let Some(ij) = self.rapier.impulse_joints.get(j.handle) else { continue };
            let local = if first { ij.data.local_frame1 } else { ij.data.local_frame2 };
            out.push(CarriedJoint {
                other: if first { j.b } else { j.a },
                first,
                world_frame: *pose * local,
                data: ij.data,
                break_force: j.break_force,
            });
        }
        out
    }

    /// Puts a carried joint back on the new part closest to its anchor.
    fn reattach(&mut self, c: &CarriedJoint, parts: &[(u32, Vec<CellPos>)]) {
        let t = c.world_frame.translation * CELLS_PER_METER;
        let anchor = CellPos::new(t.x.floor() as i32, t.y.floor() as i32);
        let dist = |p: &CellPos| {
            let (dx, dy) =
                (i64::from(p.x) - i64::from(anchor.x), i64::from(p.y) - i64::from(anchor.y));
            dx * dx + dy * dy
        };
        let nearest = parts
            .iter()
            .filter_map(|(id, cells)| cells.iter().map(dist).min().map(|d| (d, *id)))
            .min();
        let Some((_, id)) = nearest else { return };
        let Some(new_handle) = self.bodies.iter().find(|b| b.id == id).map(|b| b.handle) else {
            return;
        };
        let other_handle = if c.other == u32::MAX {
            Some(self.terrain_body)
        } else {
            self.bodies.iter().find(|b| b.id == c.other).map(|b| b.handle)
        };
        let Some(other_handle) = other_handle else { return };
        let Some(new_pose) = self.rapier.bodies.get(new_handle).map(|rb| *rb.position()) else {
            return;
        };
        let mut data = c.data;
        let local = new_pose.inverse() * c.world_frame;
        let (h1, h2, a, b) = if c.first {
            data.set_local_frame1(local);
            (new_handle, other_handle, id, c.other)
        } else {
            data.set_local_frame2(local);
            (other_handle, new_handle, c.other, id)
        };
        let handle = self.rapier.impulse_joints.insert(h1, h2, data, true);
        self.joints.push(JointInfo { handle, a, b, break_force: c.break_force });
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

    /// Removes bodies that left the loaded chunks. Nothing collides with them or
    /// stamps them there, so they would fall forever.
    fn remove_lost(&mut self, world: &World) {
        let mut k = 0;
        while k < self.bodies.len() {
            let b = &self.bodies[k];
            let lost = b.stamped.is_empty()
                && self.rapier.bodies.get(b.handle).is_some_and(|rb| {
                    let t = rb.translation() * CELLS_PER_METER;
                    let (w, h) = b.size();
                    let r = f32::from(w.max(h));
                    let a = CellPos::new((t.x - r).floor() as i32, (t.y - r).floor() as i32);
                    let z = CellPos::new((t.x + r).floor() as i32, (t.y + r).floor() as i32);
                    let (a, z) = (a.chunk(), z.chunk());
                    !(a.y..=z.y)
                        .any(|cy| (a.x..=z.x).any(|cx| world.grid.contains(ChunkPos::new(cx, cy))))
                });
            if lost {
                let b = self.bodies.remove(k);
                self.remove_rapier_body(&b);
                self.stats.removed += 1;
                continue;
            }
            k += 1;
        }
    }

    /// Merges small bodies that have rested on the terrain for a while back into
    /// it. A body that floats, or rests on another body, never merges: a crate
    /// on the water would freeze into the surface, and a cask in a hold would
    /// turn into a block of terrain holding the ship up.
    fn merge_resting(&mut self, world: &mut World) {
        let mut k = 0;
        while k < self.bodies.len() {
            let b = &self.bodies[k];
            let resting = self.rapier.bodies.get(b.handle).is_some_and(|rb| {
                rb.is_sleeping() || (rb.linvel().length() < 0.05 && rb.angvel().abs() < 0.05)
            });
            let jointed = self.joints.iter().any(|j| j.a == b.id || j.b == b.id);
            // A body floating in, or resting on, a liquid or powder stays a body:
            // a raft would otherwise turn into terrain in the middle of a lake.
            // So does one resting on another body.
            let settled = resting && !jointed && !b.floating && !self.on_body(b);
            let b = &mut self.bodies[k];
            b.rest_ticks = if settled { b.rest_ticks + 1 } else { 0 };
            if b.rest_ticks >= MERGE_AFTER_TICKS
                && b.cell_count() <= MERGE_MAX_CELLS
                && !b.stamped.is_empty()
                && b.fully_stamped()
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

    /// True if the body touches another body.
    fn on_body(&self, b: &PixelBody) -> bool {
        self.rapier.narrow_phase.contact_pairs_with(b.collider).any(|pair| {
            let other = if pair.collider1 == b.collider { pair.collider2 } else { pair.collider1 };
            pair.has_any_active_contact()
                && self
                    .rapier
                    .colliders
                    .get(other)
                    .and_then(Collider::parent)
                    .is_some_and(|h| h != self.terrain_body)
        })
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

/// True if the cells span no more than [`MAX_BODY_SIDE`] in each direction.
fn fits_in_body(cells: &[(CellPos, Cell)]) -> bool {
    let mut r = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
    for &(p, _) in cells {
        r = (r.0.min(p.x.into()), r.1.min(p.y.into()), r.2.max(p.x.into()), r.3.max(p.y.into()));
    }
    r.2 - r.0 < i64::from(MAX_BODY_SIDE) && r.3 - r.1 < i64::from(MAX_BODY_SIDE)
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
        // A rotated part can map two cells onto one world cell; use the nearest air.
        let at = nearest_air(world, p, 3);
        if world.materials().get(debris_id).state.falls() {
            let q = at.unwrap_or(p);
            particles.spawn(Fx::from_int(q.x), Fx::from_int(q.y), vx, vy, d);
        } else if let Some(q) = at {
            let _ = world.set(q, d);
        }
    }
}

/// Dust cells an impact raises: in proportion to the body's kinetic energy
/// lost, from nothing for a crate set down gently to several thousand for a
/// tower hitting the ground.
fn dust_amount(cells: usize, dv: f32) -> u32 {
    let e = cells as f32 * dv * dv / 60.0;
    (e as u32).min(10_000)
}

/// Raises up to `amount` dust cells in the air around the face of a body that
/// took an impact, the face toward `dir` in the body frame: a puff that rolls out
/// along the ground where a tower lands.
fn impact_dust(
    world: &mut World,
    body: &PixelBody,
    pose: &Pose,
    dir: (f32, f32),
    amount: u32,
    seed: u64,
) {
    if amount < 8 {
        return;
    }
    let Some(dust) = world.materials().id("dust") else { return };
    let (w, _) = body.size();
    let w = usize::from(w);
    // The face: cells within four cells of the furthest one along `dir`.
    let com = body.com();
    let mut face: Vec<(f32, i32, i32)> = Vec::new();
    let mut top = f32::MIN;
    for (i, c) in body.cells().iter().enumerate() {
        if c.is_air() {
            continue;
        }
        let (x, y) = ((i % w) as i32, (i / w) as i32);
        let d = (x as f32 + 0.5 - com.0) * dir.0 + (y as f32 + 0.5 - com.1) * dir.1;
        top = top.max(d);
        face.push((d, x, y));
    }
    face.retain(|f| f.0 >= top - 4.0);
    if face.is_empty() {
        return;
    }
    let mut rng = SeqRng::new(seed ^ 0xd0_57);
    let spread = 3 + (amount / 200).min(14) as i32;
    // The face's outward normal in the world, and the direction along it. Dust
    // goes out along the face, beside the body and backed away from whatever
    // it hit, never into the ground or into the body's own footprint.
    let (c, s) = (pose.rotation.cos(), pose.rotation.sin());
    let n = (c * dir.0 - s * dir.1, s * dir.0 + c * dir.1);
    let along = (-n.1, n.0);
    let mut made = 0;
    for _ in 0..amount * 3 {
        let f = face[rng.below(face.len() as u32) as usize];
        let p = body.local_to_world(pose, f.1, f.2);
        let t = rng.range_i32(-2 * spread, 2 * spread) as f32;
        let u = rng.range_i32(1, spread) as f32;
        let q = CellPos::new(
            p.x + (along.0 * t - n.0 * u).round() as i32,
            p.y + (along.1 * t - n.1 * u).round() as i32,
        );
        if world.get(q).is_air() {
            let c = world.make_cell(dust, q);
            let _ = world.set(q, c);
            made += 1;
            if made >= amount {
                return;
            }
        }
    }
}

/// Releases up to `max` dust cells where the cracks between a shattered body's
/// shards reach its surface: a third of the open cells next to a shard-boundary
/// cell turn to dust.
fn crack_dust(
    world: &mut World,
    body: &PixelBody,
    pose: &Pose,
    parts: &[Vec<(i32, i32, Cell)>],
    seed: u64,
    max: u32,
) {
    let Some(dust) = world.materials().id("dust") else { return };
    let (w, h) = body.size();
    let (w, h) = (i32::from(w), i32::from(h));
    let mut label = vec![u32::MAX; (w * h) as usize];
    for (k, part) in parts.iter().enumerate() {
        for &(x, y, _) in part {
            label[(y * w + x) as usize] = k as u32;
        }
    }
    let at = |x: i32, y: i32| {
        if x < 0 || y < 0 || x >= w || y >= h { u32::MAX } else { label[(y * w + x) as usize] }
    };
    let mut rng = SeqRng::new(seed ^ 0xd057);
    let mut made = 0;
    for (k, part) in parts.iter().enumerate() {
        for &(x, y, _) in part {
            let dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)];
            let cracked = dirs.iter().any(|&(dx, dy)| {
                let l = at(x + dx, y + dy);
                l != u32::MAX && l != k as u32
            });
            if !cracked {
                continue;
            }
            for (dx, dy) in dirs {
                if at(x + dx, y + dy) != u32::MAX || rng.below(3) != 0 {
                    continue;
                }
                let p = body.local_to_world(pose, x + dx, y + dy);
                if world.get(p).is_air() {
                    let c = world.make_cell(dust, p);
                    let _ = world.set(p, c);
                    made += 1;
                    if made >= max {
                        return;
                    }
                }
            }
        }
    }
}

/// The air cell nearest to `p` within `r` cells (Chebyshev rings, in a fixed order).
fn nearest_air(world: &World, p: CellPos, r: i32) -> Option<CellPos> {
    if world.get(p).is_air() {
        return Some(p);
    }
    for k in 1..=r {
        for dy in -k..=k {
            for dx in -k..=k {
                if dx.abs() != k && dy.abs() != k {
                    continue;
                }
                let q = CellPos::new(p.x + dx, p.y + dy);
                if world.get(q).is_air() {
                    return Some(q);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use fe_core::IRect;
    use std::sync::Arc;

    fn world() -> World {
        let mats = Arc::new(fe_materials::base_set());
        let mut w = World::new(mats, 3, 1);
        w.ensure_rect(IRect::new(0, 0, 256, 256), Cell::AIR);
        w
    }

    fn brick_body(w: &mut World, ph: &mut Physics, rect: IRect) -> u32 {
        let brick = w.materials().id("brick").unwrap_or_default();
        w.fill_rect(rect, brick);
        ph.carve_body(w, rect, &mut Vec::new()).unwrap_or_else(|| panic!("no body"))
    }

    #[test]
    fn resplit_of_a_rotated_body_keeps_every_cell() {
        let mut w = world();
        let mut ps = Particles::default();
        let mut ph = Physics::new();
        let id = brick_body(&mut w, &mut ph, IRect::new(98, 98, 158, 158));
        let h = ph.bodies[0].handle;
        let rb = &mut ph.rapier.bodies[h];
        let mut pose = *rb.position();
        pose.rotation = Rotation::new(std::f32::consts::FRAC_PI_4);
        rb.set_position(pose, true);
        ph.stamp(&mut w);
        ph.resplit(&mut w, &mut ps, 0, None);
        assert_eq!(ph.bodies.len(), 1);
        assert_ne!(ph.bodies[0].id, id);
        assert_eq!(ph.bodies[0].cell_count(), 3600);
        let a = ph.rapier.bodies[ph.bodies[0].handle].rotation().angle();
        assert!((a - std::f32::consts::FRAC_PI_4).abs() < 1e-5, "angle {a}");
    }

    #[test]
    fn a_body_that_loses_one_cell_keeps_its_identity_and_motion() {
        let mut w = world();
        let mut ps = Particles::default();
        let mut ph = Physics::new();
        let id = brick_body(&mut w, &mut ph, IRect::new(100, 100, 130, 120));
        ph.push_body(id, 30.0, -12.0);
        ph.stamp(&mut w);
        let before = ph.rapier.bodies[ph.bodies[0].handle].linvel();
        let (p, _) = ph.bodies[0].stamped[37];
        w.grid.set(p, Cell::AIR);
        ph.sync(&mut w, &mut ps);
        assert_eq!(ph.bodies.len(), 1);
        assert_eq!(ph.bodies[0].id, id);
        assert_eq!(ph.bodies[0].cell_count(), 599);
        let after = ph.rapier.bodies[ph.bodies[0].handle].linvel();
        assert!((after - before).length() < 1e-3, "{before:?} -> {after:?}");
    }

    #[test]
    fn gravity_matches_the_cell_world() {
        let cells_per_tick2 = fe_core::GRAVITY.0 as f32 / 65536.0;
        let tps = fe_core::TICKS_PER_SECOND as f32;
        let expected = cells_per_tick2 * tps * tps / fe_core::CELLS_PER_METER as f32;
        assert_eq!(GRAVITY.to_bits(), expected.to_bits());
    }
}
