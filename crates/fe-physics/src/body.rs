use crate::{CELLS_PER_METER, is_body_material};
use fe_core::{Cell, CellPos, SeqRng, StableHasher};
use fe_materials::State;
use fe_sim::World;
use rapier2d::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Smallest group of cells that becomes a body. Smaller groups become debris.
pub const MIN_BODY_CELLS: usize = 16;

/// Physical size of one cell in meters.
const S: f32 = 1.0 / CELLS_PER_METER;

/// Joint types between bodies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum JointKind {
    /// Free rotation around the anchor.
    Hinge,
    /// No relative motion.
    Fixed,
}

/// A summary of a body's state, for tools and the control API.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodyInfo {
    /// Body id.
    pub id: u32,
    /// Center of mass in world cells.
    pub x: f32,
    /// Center of mass in world cells.
    pub y: f32,
    /// Rotation in radians.
    pub angle: f32,
    /// Velocity in cells per second.
    pub vx: f32,
    /// Velocity in cells per second.
    pub vy: f32,
    /// Number of cells.
    pub cells: u32,
    /// Rapier put the body to sleep.
    pub sleeping: bool,
}

/// One rigid body made of cells.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PixelBody {
    /// Stable id, assigned in creation order.
    pub id: u32,
    w: u16,
    h: u16,
    cells: Vec<Cell>,
    interior: Vec<bool>,
    com: (f32, f32),
    /// Rapier body.
    pub handle: RigidBodyHandle,
    /// Rapier collider.
    pub collider: ColliderHandle,
    /// World cells written by the last stamp, with the local index each came from.
    pub stamped: Vec<(CellPos, u32)>,
    cavity: Vec<CellPos>,
    last_pose: [u32; 4],
    /// Ticks spent at rest.
    pub rest_ticks: u32,
    /// Fracture strength: the body shatters when one tick's change in velocity,
    /// apart from gravity, exceeds `fracture / 32` m/s. 0 never shatters.
    pub fracture: u16,
    /// A liquid or powder held the body up on the last tick. Recomputed every
    /// tick before it is read, so it is not saved.
    #[serde(skip)]
    pub(crate) wet: bool,
}

fn cell_mass(world: &World, c: Cell) -> f32 {
    let d = world.materials().get(c.material()).density;
    let d = if d <= 0 { 200 } else { d };
    f32::from(d) / 100.0
}

fn pose_bits(p: &Pose) -> [u32; 4] {
    [
        p.translation.x.to_bits(),
        p.translation.y.to_bits(),
        p.rotation.cos().to_bits(),
        p.rotation.sin().to_bits(),
    ]
}

impl PixelBody {
    /// Builds a body from cells at world positions and inserts it into Rapier.
    pub(crate) fn from_world_cells(
        id: u32,
        world: &World,
        rapier: &mut PhysicsWorld,
        cells: &[(CellPos, Cell)],
        linvel: (f32, f32),
        angvel: f32,
    ) -> Option<Self> {
        let x0 = cells.iter().map(|c| c.0.x).min()?;
        let y0 = cells.iter().map(|c| c.0.y).min()?;
        let x1 = cells.iter().map(|c| c.0.x).max()?;
        let y1 = cells.iter().map(|c| c.0.y).max()?;
        let (w, h) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
        if w > 4096 || h > 4096 {
            return None;
        }
        let mut grid = vec![Cell::AIR; w * h];
        for &(p, c) in cells {
            grid[(p.y - y0) as usize * w + (p.x - x0) as usize] =
                c.with_flag(Cell::FLAG_IN_BODY, false);
        }
        let mut body = Self::from_grid(id, w, h, grid);
        let (mass, inertia) = body.refresh_props(world);
        let origin = Vector::new((x0 as f32 + body.com.0) * S, (y0 as f32 + body.com.1) * S);
        let pose = Pose::from_translation(origin);
        body.insert(rapier, pose, Vector::new(linvel.0, linvel.1), angvel, mass, inertia);
        Some(body)
    }

    /// Builds a body from part of another body's local grid, keeping the parent's
    /// rotation, so no cell is lost or doubled the way rasterizing a rotated part
    /// into world cells would. `linvel` and `angvel` are the parent's velocity; the
    /// child gets the velocity of the parent's motion at its own center of mass.
    #[expect(clippy::too_many_arguments, reason = "the parent's whole motion is needed")]
    pub(crate) fn from_local_cells(
        id: u32,
        world: &World,
        rapier: &mut PhysicsWorld,
        parent: &Self,
        pose: &Pose,
        part: &[(i32, i32, Cell)],
        linvel: Vector,
        angvel: f32,
    ) -> Option<Self> {
        let x0 = part.iter().map(|c| c.0).min()?;
        let y0 = part.iter().map(|c| c.1).min()?;
        let x1 = part.iter().map(|c| c.0).max()?;
        let y1 = part.iter().map(|c| c.1).max()?;
        let (w, h) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
        let mut grid = vec![Cell::AIR; w * h];
        for &(lx, ly, c) in part {
            grid[(ly - y0) as usize * w + (lx - x0) as usize] =
                c.with_flag(Cell::FLAG_IN_BODY, false);
        }
        let mut body = Self::from_grid(id, w, h, grid);
        let (mass, inertia) = body.refresh_props(world);
        // Offset of the child's center of mass from the parent's, in world meters.
        let ox = (x0 as f32 + body.com.0 - parent.com.0) * S;
        let oy = (y0 as f32 + body.com.1 - parent.com.1) * S;
        let (c, s) = (pose.rotation.cos(), pose.rotation.sin());
        let off = Vector::new(c * ox - s * oy, s * ox + c * oy);
        let mut child = *pose;
        child.translation = pose.translation + off;
        let v = linvel + Vector::new(-angvel * off.y, angvel * off.x);
        body.insert(rapier, child, v, angvel, mass, inertia);
        Some(body)
    }

    fn from_grid(id: u32, w: usize, h: usize, cells: Vec<Cell>) -> Self {
        Self {
            id,
            w: w as u16,
            h: h as u16,
            cells,
            interior: Vec::new(),
            com: (0.0, 0.0),
            handle: RigidBodyHandle::invalid(),
            collider: ColliderHandle::invalid(),
            stamped: Vec::new(),
            cavity: Vec::new(),
            last_pose: [0; 4],
            rest_ticks: 0,
            fracture: 0,
            wet: false,
        }
    }

    /// Recomputes the center of mass, hull interior and fracture strength from the
    /// cells. Returns the mass and angular inertia.
    fn refresh_props(&mut self, world: &World) -> (f32, f32) {
        let (mass, com, inertia) = self.mass_props(world);
        self.com = com;
        self.interior = self.find_interior();
        self.fracture = self.compute_fracture(world);
        (mass, inertia)
    }

    fn insert(
        &mut self,
        rapier: &mut PhysicsWorld,
        pose: Pose,
        linvel: Vector,
        angvel: f32,
        mass: f32,
        inertia: f32,
    ) {
        let rb = RigidBodyBuilder::dynamic()
            .pose(pose)
            .linvel(linvel)
            .angvel(angvel)
            .additional_mass_properties(MassProperties::new(Vector::ZERO, mass, inertia))
            .ccd_enabled(true)
            .build();
        self.handle = rapier.bodies.insert(rb);
        let collider = self.build_collider();
        self.collider =
            rapier.colliders.insert_with_parent(collider, self.handle, &mut rapier.bodies);
    }

    /// Updates a body that lost cells but is still in one piece, keeping its Rapier
    /// body, id, velocity, sleep state and rest time: new mass properties, hull,
    /// fracture strength and collider, and a translation shifted so the remaining
    /// cells stay where they are.
    ///
    /// Returns how far the center of mass moved in the body's own frame, in
    /// meters, so joint anchors on the body can be moved by the same amount.
    pub(crate) fn refresh_shape(&mut self, world: &World, rapier: &mut PhysicsWorld) -> Vector {
        let old_com = self.com;
        let (mass, inertia) = self.refresh_props(world);
        let shape = self.compound_shape();
        if let Some(col) = rapier.colliders.get_mut(self.collider) {
            col.set_shape(shape);
        }
        let shift = Vector::new((self.com.0 - old_com.0) * S, (self.com.1 - old_com.1) * S);
        let Some(rb) = rapier.bodies.get_mut(self.handle) else { return shift };
        let sleeping = rb.is_sleeping();
        rb.set_additional_mass_properties(MassProperties::new(Vector::ZERO, mass, inertia), false);
        let mut pose = *rb.position();
        let (dx, dy) = ((self.com.0 - old_com.0) * S, (self.com.1 - old_com.1) * S);
        let (c, s) = (pose.rotation.cos(), pose.rotation.sin());
        let off = Vector::new(c * dx - s * dy, s * dx + c * dy);
        pose.translation += off;
        rb.set_position(pose, false);
        // The new center of mass moves with the old body's motion at that point.
        let w = rb.angvel();
        let v = rb.linvel() + Vector::new(-w * off.y, w * off.x);
        rb.set_linvel(v, false);
        if sleeping {
            rb.sleep();
        }
        // Force a restamp so the lost cells' slots are written fresh.
        self.last_pose = [u32::MAX; 4];
        shift
    }

    /// The mass-weighted mean fracture strength of the cells that can shatter, or
    /// 0 (never shatters) when those cells are under half of the body's mass. One
    /// glass pane in a brick tower does not make the whole tower shatter like glass.
    fn compute_fracture(&self, world: &World) -> u16 {
        let (mut total, mut brittle, mut sum) = (0.0f32, 0.0f32, 0.0f32);
        for c in self.cells.iter().filter(|c| !c.is_air()) {
            let m = cell_mass(world, *c);
            let f = world.materials().get(c.material()).fracture;
            total += m;
            if f > 0 {
                brittle += m;
                sum += m * f32::from(f);
            }
        }
        if brittle <= 0.0 || brittle * 2.0 < total {
            return 0;
        }
        (sum / brittle).round().clamp(1.0, f32::from(u16::MAX)) as u16
    }

    /// Splits the cells into rectangles: horizontal runs merged with identical runs below.
    fn rectangles(&self) -> Vec<(i32, i32, i32, i32)> {
        let (w, h) = (i32::from(self.w), i32::from(self.h));
        let solid = |x: i32, y: i32| !self.cells[(y * w + x) as usize].is_air();
        let mut used = vec![false; self.cells.len()];
        let mut rects = Vec::new();
        for y in 0..h {
            let mut x = 0;
            while x < w {
                if !solid(x, y) || used[(y * w + x) as usize] {
                    x += 1;
                    continue;
                }
                let x0 = x;
                while x < w && solid(x, y) && !used[(y * w + x) as usize] {
                    x += 1;
                }
                let x1 = x;
                let mut y1 = y + 1;
                while y1 < h && (x0..x1).all(|xx| solid(xx, y1) && !used[(y1 * w + xx) as usize]) {
                    y1 += 1;
                }
                for yy in y..y1 {
                    for xx in x0..x1 {
                        used[(yy * w + xx) as usize] = true;
                    }
                }
                rects.push((x0, y, x1, y1));
            }
        }
        rects
    }

    /// A compound of cuboids covering the cells.
    ///
    /// Rapier collides voxel shapes with other shapes but not with each other, so
    /// bodies use cuboids and terrain uses voxels.
    fn build_collider(&self) -> Collider {
        ColliderBuilder::new(self.compound_shape())
            .density(0.0)
            .friction(0.7)
            .restitution(0.05)
            .build()
    }

    fn compound_shape(&self) -> SharedShape {
        let shapes: Vec<(Pose, SharedShape)> = self
            .rectangles()
            .into_iter()
            .map(|(x0, y0, x1, y1)| {
                let cx = (x0 + x1) as f32 * 0.5 - self.com.0;
                let cy = (y0 + y1) as f32 * 0.5 - self.com.1;
                let hx = (x1 - x0) as f32 * 0.5 * S;
                let hy = (y1 - y0) as f32 * 0.5 * S;
                (Pose::from_translation(Vector::new(cx * S, cy * S)), SharedShape::cuboid(hx, hy))
            })
            .collect();
        SharedShape::compound(shapes)
    }

    /// Mass in kilograms, center of mass in local cells, and angular inertia.
    fn mass_props(&self, world: &World) -> (f32, (f32, f32), f32) {
        let (mut m, mut sx, mut sy) = (0.0f32, 0.0f32, 0.0f32);
        for ly in 0..self.h as usize {
            for lx in 0..self.w as usize {
                let c = self.cells[ly * self.w as usize + lx];
                if !c.is_air() {
                    let cm = cell_mass(world, c);
                    m += cm;
                    sx += cm * (lx as f32 + 0.5);
                    sy += cm * (ly as f32 + 0.5);
                }
            }
        }
        let m = m.max(1e-3);
        let com = (sx / m, sy / m);
        let mut inertia = 0.0f32;
        for ly in 0..self.h as usize {
            for lx in 0..self.w as usize {
                let c = self.cells[ly * self.w as usize + lx];
                if !c.is_air() {
                    let cm = cell_mass(world, c);
                    let dx = (lx as f32 + 0.5 - com.0) * S;
                    let dy = (ly as f32 + 0.5 - com.1) * S;
                    inertia += cm * (dx * dx + dy * dy) + cm * S * S / 6.0;
                }
            }
        }
        (m, com, inertia.max(1e-6))
    }

    /// Local cells not connected to the grid's border through air: the inside of a hull.
    fn find_interior(&self) -> Vec<bool> {
        let (w, h) = (self.w as usize, self.h as usize);
        let mut outside = vec![false; w * h];
        let mut stack = Vec::new();
        for x in 0..w {
            stack.push((x, 0));
            stack.push((x, h - 1));
        }
        for y in 0..h {
            stack.push((0, y));
            stack.push((w - 1, y));
        }
        while let Some((x, y)) = stack.pop() {
            let i = y * w + x;
            if outside[i] || !self.cells[i].is_air() {
                continue;
            }
            outside[i] = true;
            if x > 0 {
                stack.push((x - 1, y));
            }
            if x + 1 < w {
                stack.push((x + 1, y));
            }
            if y > 0 {
                stack.push((x, y - 1));
            }
            if y + 1 < h {
                stack.push((x, y + 1));
            }
        }
        (0..w * h).map(|i| self.cells[i].is_air() && !outside[i]).collect()
    }

    /// Number of solid cells.
    pub fn cell_count(&self) -> usize {
        self.cells.iter().filter(|c| !c.is_air()).count()
    }

    /// Center of mass in local cell coordinates.
    pub const fn com(&self) -> (f32, f32) {
        self.com
    }

    /// Local grid size in cells.
    pub const fn size(&self) -> (u16, u16) {
        (self.w, self.h)
    }

    /// Local grid cells, row-major. Air marks an empty slot.
    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    pub(crate) fn info(&self, rapier: &PhysicsWorld) -> BodyInfo {
        let rb = &rapier.bodies[self.handle];
        let t = rb.translation();
        let v = rb.linvel();
        BodyInfo {
            id: self.id,
            x: t.x * CELLS_PER_METER,
            y: t.y * CELLS_PER_METER,
            angle: rb.rotation().angle(),
            vx: v.x * CELLS_PER_METER,
            vy: v.y * CELLS_PER_METER,
            cells: self.cell_count() as u32,
            sleeping: rb.is_sleeping(),
        }
    }

    /// World cell nearest to the center of local cell `(lx, ly)` under `pose`.
    pub(crate) fn local_to_world(&self, pose: &Pose, lx: i32, ly: i32) -> CellPos {
        let (c, s) = (pose.rotation.cos(), pose.rotation.sin());
        let ox = lx as f32 + 0.5 - self.com.0;
        let oy = ly as f32 + 0.5 - self.com.1;
        let wx = pose.translation.x * CELLS_PER_METER + c * ox - s * oy;
        let wy = pose.translation.y * CELLS_PER_METER + s * ox + c * oy;
        CellPos::new(wx.floor() as i32, wy.floor() as i32)
    }

    /// The body's solid cell nearest to `center` (world cells) within `r` cells,
    /// as its world-space center and distance.
    pub(crate) fn nearest_cell(
        &self,
        pose: &Pose,
        center: (f32, f32),
        r: f32,
    ) -> Option<(f32, f32, f32)> {
        let (c, s) = (pose.rotation.cos(), pose.rotation.sin());
        let tx = pose.translation.x * CELLS_PER_METER;
        let ty = pose.translation.y * CELLS_PER_METER;
        // Reject by the bounding circle of the local grid first.
        let (w, h) = (f32::from(self.w), f32::from(self.h));
        // (`hypot` is not guaranteed to round the same on every platform; `sqrt` is.)
        let (ex, ey) = (self.com.0.max(w - self.com.0), self.com.1.max(h - self.com.1));
        let reach = (ex * ex + ey * ey).sqrt();
        let (cx, cy) = (tx - center.0, ty - center.1);
        if (cx * cx + cy * cy).sqrt() > r + reach {
            return None;
        }
        let mut best: Option<(f32, f32, f32)> = None;
        let wi = self.w as usize;
        for (i, cell) in self.cells.iter().enumerate() {
            if cell.is_air() {
                continue;
            }
            let ox = (i % wi) as f32 + 0.5 - self.com.0;
            let oy = (i / wi) as f32 + 0.5 - self.com.1;
            let wx = tx + c * ox - s * oy;
            let wy = ty + s * ox + c * oy;
            let d2 = (wx - center.0) * (wx - center.0) + (wy - center.1) * (wy - center.1);
            if d2 <= r * r && best.is_none_or(|b| d2 < b.2) {
                best = Some((wx, wy, d2));
            }
        }
        best.map(|(x, y, d2)| (x, y, d2.sqrt()))
    }

    /// True if the body moved since its last stamp or was never stamped.
    pub(crate) fn needs_restamp(&self, rapier: &PhysicsWorld) -> bool {
        match rapier.bodies.get(self.handle) {
            Some(rb) => self.stamped.is_empty() || pose_bits(rb.position()) != self.last_pose,
            None => false,
        }
    }

    /// Copies damage from stamped world cells into the body. Returns true if cells were lost.
    pub(crate) fn sync_from_world(&mut self, world: &World) -> bool {
        let mut lost = false;
        let solid = |c: Cell| {
            c.has(Cell::FLAG_IN_BODY) && world.materials().get(c.material()).state == State::Static
        };
        // Two world cells can map to one local cell under rotation; a loss on either counts.
        let mut seen: BTreeMap<u32, bool> = BTreeMap::new();
        for &(p, idx) in &self.stamped {
            let wc = world.get(p);
            let ok = solid(wc);
            let e = seen.entry(idx).or_insert(true);
            *e &= ok;
            if ok && *e {
                self.cells[idx as usize] = wc.with_flag(Cell::FLAG_IN_BODY, false);
            }
        }
        for (idx, ok) in seen {
            if !ok && !self.cells[idx as usize].is_air() {
                self.cells[idx as usize] = Cell::AIR;
                lost = true;
            }
        }
        lost
    }

    /// Clears the body's stamped cells from the world.
    pub(crate) fn unstamp(&mut self, world: &mut World) {
        for &(p, _) in &self.stamped {
            let c = world.get(p);
            if c.has(Cell::FLAG_IN_BODY) {
                let mut air = Cell::AIR;
                air.temp = c.temp;
                world.grid.set(p, air);
            }
        }
        self.stamped.clear();
        self.cavity.clear();
    }

    /// Writes the body's cells into the world at its current pose, pushing fluids
    /// aside. Returns the number of displaced fluid cells that found no room.
    pub(crate) fn stamp(&mut self, world: &mut World, rapier: &PhysicsWorld) -> u32 {
        let Some(rb) = rapier.bodies.get(self.handle) else { return 0 };
        let mut lost = 0;
        let pose = *rb.position();
        if !self.stamped.is_empty() {
            self.unstamp(world);
        }
        self.last_pose = pose_bits(&pose);
        let (c, s) = (pose.rotation.cos(), pose.rotation.sin());
        let tx = pose.translation.x * CELLS_PER_METER;
        let ty = pose.translation.y * CELLS_PER_METER;
        let (w, h) = (f32::from(self.w), f32::from(self.h));
        let corners = [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)];
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for (lx, ly) in corners {
            let (ox, oy) = (lx - self.com.0, ly - self.com.1);
            let wx = tx + c * ox - s * oy;
            let wy = ty + s * ox + c * oy;
            x0 = x0.min(wx);
            y0 = y0.min(wy);
            x1 = x1.max(wx);
            y1 = y1.max(wy);
        }
        let (wi, hi) = (i32::from(self.w), i32::from(self.h));
        for wy in (y0.floor() as i32)..=(y1.floor() as i32) {
            for wx in (x0.floor() as i32)..=(x1.floor() as i32) {
                let px = wx as f32 + 0.5 - tx;
                let py = wy as f32 + 0.5 - ty;
                let lxf = c * px + s * py + self.com.0;
                let lyf = -s * px + c * py + self.com.1;
                let (lx, ly) = (lxf.floor() as i32, lyf.floor() as i32);
                if lx < 0 || ly < 0 || lx >= wi || ly >= hi {
                    continue;
                }
                let idx = (ly * wi + lx) as usize;
                let p = CellPos::new(wx, wy);
                if self.interior.get(idx).copied().unwrap_or(false) {
                    self.cavity.push(p);
                    continue;
                }
                let cell = self.cells[idx];
                if cell.is_air() {
                    continue;
                }
                match place(world, p, cell) {
                    Placed::Yes => self.stamped.push((p, idx as u32)),
                    Placed::Displaced(ok) => {
                        self.stamped.push((p, idx as u32));
                        lost += u32::from(!ok);
                    }
                    Placed::No => {}
                }
            }
        }
        lost
    }

    /// Applies buoyancy and drag from liquids and powders around the body.
    ///
    /// Only liquid or powder outside the body counts. Water in a tank the body
    /// carries, or an oil film on one of its floors, moves with it: it adds
    /// weight through the cells it rests on, not lift or drag. A few loose
    /// grains do not count either; the medium has to be a real pool or heap.
    pub(crate) fn apply_fluid_forces(&mut self, rapier: &mut PhysicsWorld, world: &World) {
        self.wet = false;
        let Some(rb) = rapier.bodies.get_mut(self.handle) else { return };
        rb.reset_forces(false);
        if self.stamped.is_empty() {
            return;
        }
        let mut cavity = self.cavity.clone();
        cavity.sort_unstable();
        // Group the body's footprint (cells and hull cavity) by world column.
        let mut cols: BTreeMap<i32, Vec<(i32, bool)>> = BTreeMap::new();
        for &(p, _) in &self.stamped {
            cols.entry(p.x).or_default().push((p.y, true));
        }
        for &p in &self.cavity {
            cols.entry(p.x).or_default().push((p.y, false));
        }
        let is_fluid = |p: CellPos| {
            let c = world.get(p);
            !c.is_air()
                && !c.has(Cell::FLAG_IN_BODY)
                && matches!(
                    world.materials().get(c.material()).state,
                    State::Liquid | State::Powder
                )
        };
        let fluid = |p: CellPos| -> Option<(i16, u8)> {
            if !is_fluid(p) || cavity.binary_search(&p).is_ok() {
                return None;
            }
            // A pool or heap, not a speck: at least four of the eight cells
            // around it are liquid or powder too.
            let mut around = 0;
            for (dx, dy) in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                around += u32::from(is_fluid(CellPos::new(p.x + dx, p.y + dy)));
            }
            if around < 4 {
                return None;
            }
            let m = world.materials().get(world.get(p).material());
            Some((m.density, m.viscosity.max(40)))
        };
        let (mut force, mut n, mut visc) = (0.0f32, 0u32, 0u32);
        for (x, mut entries) in cols {
            entries.sort_unstable();
            let mut medium: Option<(i16, u8)> = None;
            for (y, is_cell) in entries {
                if medium.is_none() {
                    // Look past the air a moving body leaves behind, up to three cells out.
                    'dirs: for (dx, dy) in [(0, -1), (-1, 0), (1, 0), (0, 1)] {
                        for k in 1..=3 {
                            let q = CellPos::new(x + dx * k, y + dy * k);
                            let c = world.get(q);
                            if c.is_air() {
                                continue;
                            }
                            if let Some(f) = fluid(q) {
                                medium = Some(f);
                                break 'dirs;
                            }
                            break;
                        }
                    }
                }
                let Some((density, v)) = medium else { continue };
                // A flooded cavity cell holds liquid, not air, so it adds no lift.
                if !is_cell && !world.get(CellPos::new(x, y)).is_air() {
                    continue;
                }
                let lift = f32::from(density.max(0)) / 100.0 * crate::GRAVITY;
                force += lift;
                n += 1;
                visc += u32::from(v);
            }
        }
        let total = (self.stamped.len() + self.cavity.len()).max(1) as f32;
        if n > 0 && force > 0.0 {
            // Lift acts at the center of mass. Applying it at the centroid of the wet
            // cells would let the air a spinning body leaves behind feed its spin.
            rb.add_force(Vector::new(0.0, -force), false);
            let frac = n as f32 / total;
            let avg_visc = visc as f32 / n as f32;
            rb.set_linear_damping(frac * avg_visc / 8.0);
            rb.set_angular_damping(frac * avg_visc / 6.0);
            self.wet = true;
        } else {
            rb.set_linear_damping(0.0);
            rb.set_angular_damping(0.05);
        }
    }

    /// Connected parts of the remaining cells, as `(lx, ly, cell)` lists.
    pub(crate) fn components(&self) -> Vec<Vec<(i32, i32, Cell)>> {
        let (w, h) = (i32::from(self.w), i32::from(self.h));
        let mut label = vec![u32::MAX; self.cells.len()];
        let mut parts = Vec::new();
        for start in 0..self.cells.len() {
            if self.cells[start].is_air() || label[start] != u32::MAX {
                continue;
            }
            let id = parts.len() as u32;
            let mut part = Vec::new();
            let mut stack = vec![start];
            label[start] = id;
            while let Some(i) = stack.pop() {
                let (x, y) = (i as i32 % w, i as i32 / w);
                part.push((x, y, self.cells[i]));
                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w || ny >= h {
                        continue;
                    }
                    let j = (ny * w + nx) as usize;
                    if !self.cells[j].is_air() && label[j] == u32::MAX {
                        label[j] = id;
                        stack.push(j);
                    }
                }
            }
            part.sort_unstable_by_key(|&(x, y, _)| (y, x));
            parts.push(part);
        }
        parts
    }

    /// The local cell on the side of the body facing `dir` (local frame): where
    /// an impact pushing the body along `-dir` struck it.
    pub(crate) fn contact_cell(&self, dir: (f32, f32)) -> Option<(i32, i32)> {
        let w = self.w as usize;
        let mut best: Option<(f32, usize)> = None;
        for (i, c) in self.cells.iter().enumerate() {
            if c.is_air() {
                continue;
            }
            let ox = (i % w) as f32 + 0.5 - self.com.0;
            let oy = (i / w) as f32 + 0.5 - self.com.1;
            let d = ox * dir.0 + oy * dir.1;
            if best.is_none_or(|b| d > b.0) {
                best = Some((d, i));
            }
        }
        best.map(|(_, i)| ((i % w) as i32, (i / w) as i32))
    }

    /// Splits the body into Voronoi shards seeded from `seed`, each split again
    /// into connected parts. With a `contact` cell the sites crowd around it, so
    /// the body breaks into small pieces where it was hit and large slabs far away.
    pub(crate) fn shards(
        &self,
        seed: u64,
        contact: Option<(i32, i32)>,
    ) -> Vec<Vec<(i32, i32, Cell)>> {
        let w = i32::from(self.w);
        let solid: Vec<(i32, i32)> = (0..self.cells.len())
            .filter(|&i| !self.cells[i].is_air())
            .map(|i| (i as i32 % w, i as i32 / w))
            .collect();
        let k = (2 + solid.len() / 150).min(40);
        let mut rng = SeqRng::new(seed);
        let mut sites: Vec<(i32, i32)> = Vec::with_capacity(k);
        for s in 0..k {
            let mut p = solid[rng.below(solid.len() as u32) as usize];
            // Three sites in four are the nearest to the contact of a few draws.
            if let Some((cx, cy)) = contact
                && s % 4 != 3
            {
                for _ in 0..3 {
                    let q = solid[rng.below(solid.len() as u32) as usize];
                    let d = |a: (i32, i32)| (a.0 - cx).pow(2) + (a.1 - cy).pow(2);
                    if d(q) < d(p) {
                        p = q;
                    }
                }
            }
            sites.push(p);
        }
        let mut groups: Vec<Vec<(i32, i32, Cell)>> = vec![Vec::new(); k];
        for &(x, y) in &solid {
            let best = (0..k)
                .min_by_key(|&s| {
                    let (dx, dy) = (x - sites[s].0, y - sites[s].1);
                    (dx * dx + dy * dy, s)
                })
                .unwrap_or(0);
            groups[best].push((x, y, self.cells[(y * w + x) as usize]));
        }
        let mut out = Vec::new();
        for g in groups {
            if g.is_empty() {
                continue;
            }
            let mut sub = self.clone();
            sub.cells.fill(Cell::AIR);
            for &(x, y, c) in &g {
                sub.cells[(y * w + x) as usize] = c;
            }
            out.extend(sub.components());
        }
        out
    }

    /// Writes the body's cells into the world as plain terrain.
    pub(crate) fn merge_into(&self, world: &mut World) {
        for &(p, _) in &self.stamped {
            let c = world.get(p);
            if c.has(Cell::FLAG_IN_BODY)
                && is_body_material(world, c.with_flag(Cell::FLAG_IN_BODY, false))
            {
                let _ = world.set(p, c.with_flag(Cell::FLAG_IN_BODY, false));
            }
        }
    }

    pub(crate) fn hash_into(&self, h: &mut StableHasher, rapier: &PhysicsWorld) {
        h.write_u32(self.id);
        h.write_u32(u32::from(self.w) | u32::from(self.h) << 16);
        for c in &self.cells {
            h.write_u32(c.to_bits());
        }
        if let Some(rb) = rapier.bodies.get(self.handle) {
            for b in pose_bits(rb.position()) {
                h.write_u32(b);
            }
            h.write_u32(rb.linvel().x.to_bits());
            h.write_u32(rb.linvel().y.to_bits());
            h.write_u32(rb.angvel().to_bits());
        }
        h.write_u32(self.rest_ticks);
    }
}

/// What [`place`] did.
enum Placed {
    /// The cell was written into air or a gas.
    Yes,
    /// The cell was written over a liquid or powder cell, which moved to air if
    /// the flag is true and was lost if it is false.
    Displaced(bool),
    /// Something solid is in the way.
    No,
}

/// Places a body cell into the world, pushing a fluid or powder cell to nearby air.
fn place(world: &mut World, p: CellPos, cell: Cell) -> Placed {
    let existing = world.get(p);
    if existing.material() == fe_core::MaterialId::BOUNDARY || existing.has(Cell::FLAG_IN_BODY) {
        return Placed::No;
    }
    let mut placed = Placed::Yes;
    if !existing.is_air() {
        let m = world.materials().get(existing.material());
        if m.state == State::Static {
            return Placed::No;
        }
        if matches!(m.state, State::Liquid | State::Powder) {
            placed = Placed::Displaced(push_aside(world, p, existing));
        }
    }
    world.grid.set(p, cell.with_flag(Cell::FLAG_IN_BODY, true));
    placed
}

/// Moves a displaced cell to the nearest air within a few cells, sideways first.
///
/// When there is none, as under a wide slab sinking into a lake, the cell goes up
/// its column through liquid, powder and body cells to the first air, which is
/// usually the space the body just left above itself. Returns false if that also
/// fails and the cell is lost.
fn push_aside(world: &mut World, from: CellPos, cell: Cell) -> bool {
    for r in 1..=4 {
        for (dx, dy) in [(-r, 0), (r, 0), (-r, -r), (r, -r), (-r, r), (r, r), (0, -r), (0, r)] {
            let q = CellPos::new(from.x + dx, from.y + dy);
            if world.get(q).is_air() {
                world.grid.set(q, cell);
                return true;
            }
        }
    }
    for k in 1..=256 {
        let q = CellPos::new(from.x, from.y - k);
        let c = world.get(q);
        if c.is_air() {
            world.grid.set(q, cell);
            return true;
        }
        let through = c.has(Cell::FLAG_IN_BODY)
            || matches!(world.materials().get(c.material()).state, State::Liquid | State::Powder);
        if !through || c.material() == fe_core::MaterialId::BOUNDARY {
            return false;
        }
    }
    false
}
