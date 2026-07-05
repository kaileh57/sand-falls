use crate::{CELLS_PER_METER, is_body_material};
use fe_core::{Cell, CellPos, MaterialId, SeqRng, StableHasher};
use fe_materials::State;
use fe_sim::World;
use rapier2d::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Smallest group of cells that becomes a body. Smaller groups become debris.
pub const MIN_BODY_CELLS: usize = 16;
/// Largest width or height of a body, in cells.
pub const MAX_BODY_SIDE: i32 = 4096;

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
    /// Air cells inside the body's outline: bit 0 marks a cell between solid
    /// cells of its local row, bit 1 one between solid cells of its local
    /// column. A hold, the inside of an open boat, or air under an upturned hull.
    envelope: Vec<u8>,
    com: (f32, f32),
    /// Rapier body.
    pub handle: RigidBodyHandle,
    /// Rapier collider.
    pub collider: ColliderHandle,
    /// World cells written by the last stamp, with the local index each came from.
    pub stamped: Vec<(CellPos, u32)>,
    /// World cells of the air the body closes in at its last stamp: a sealed
    /// hold, or air with the body close above and below it in its local column,
    /// such as a hold with its side blown in or a hull broken in two. Only the
    /// renderer reads it, to draw a body's inside; it is rebuilt by every stamp
    /// and not saved.
    #[serde(skip)]
    pub hollow: Vec<CellPos>,
    last_pose: [u32; 4],
    /// Ticks spent at rest.
    pub rest_ticks: u32,
    /// Fracture strength: the body shatters when one tick's change in velocity,
    /// apart from gravity, exceeds `fracture / 32` m/s. 0 never shatters.
    pub fracture: u16,
    /// Cells the last stamp left out because the liquid or powder there had
    /// nowhere to go.
    wet: Vec<CellPos>,
    /// The last fluid step found the body afloat or submerged in a liquid or powder.
    pub floating: bool,
    /// The net vertical force (N) and torque (N m) the fluid put on the body in
    /// the last step. A sleeping body wakes when they change: when its hold
    /// floods, or the water under it drains away.
    fluid_push: (f32, f32),
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
        if x1 - x0 >= MAX_BODY_SIDE || y1 - y0 >= MAX_BODY_SIDE {
            return None;
        }
        let (w, h) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
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
            envelope: Vec::new(),
            com: (0.0, 0.0),
            handle: RigidBodyHandle::invalid(),
            collider: ColliderHandle::invalid(),
            stamped: Vec::new(),
            hollow: Vec::new(),
            last_pose: [0; 4],
            rest_ticks: 0,
            fracture: 0,
            wet: Vec::new(),
            floating: false,
            fluid_push: (0.0, 0.0),
        }
    }

    /// Recomputes the center of mass, hull envelope and fracture strength from the
    /// cells. Returns the mass and angular inertia.
    fn refresh_props(&mut self, world: &World) -> (f32, f32) {
        let (mass, com, inertia) = self.mass_props(world);
        self.com = com;
        self.envelope = self.find_envelope();
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
    /// Returns how far the body's origin moved in its own frame, in meters. The
    /// caller must shift joint frames on this body back by that much, or the
    /// anchors move with the center of mass and drag the body after them.
    pub(crate) fn refresh_shape(&mut self, world: &World, rapier: &mut PhysicsWorld) -> Vector {
        let old_com = self.com;
        let (mass, inertia) = self.refresh_props(world);
        let shape = self.compound_shape();
        if let Some(col) = rapier.colliders.get_mut(self.collider) {
            col.set_shape(shape);
        }
        let (dx, dy) = ((self.com.0 - old_com.0) * S, (self.com.1 - old_com.1) * S);
        let Some(rb) = rapier.bodies.get_mut(self.handle) else { return Vector::ZERO };
        let sleeping = rb.is_sleeping();
        rb.set_additional_mass_properties(MassProperties::new(Vector::ZERO, mass, inertia), false);
        let mut pose = *rb.position();
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
        Vector::new(dx, dy)
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
        let w = i32::from(self.w);
        merge_rects(w, i32::from(self.h), |x, y| !self.cells[(y * w + x) as usize].is_air())
    }

    /// A compound of cuboids covering the cells.
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

    /// Marks the air inside the body's outline along each local axis (see
    /// [`PixelBody::envelope`]). Whatever fills those cells in the world, air or
    /// water, is held by the hull and moves its buoyancy: a dry hold floats it,
    /// a flooded one does not.
    fn find_envelope(&self) -> Vec<u8> {
        let (w, h) = (self.w as usize, self.h as usize);
        let solid = |x: usize, y: usize| !self.cells[y * w + x].is_air();
        let mut env = vec![0u8; w * h];
        for y in 0..h {
            let first = (0..w).find(|&x| solid(x, y));
            let last = (0..w).rev().find(|&x| solid(x, y));
            if let (Some(a), Some(b)) = (first, last) {
                for x in a + 1..b {
                    if !solid(x, y) {
                        env[y * w + x] |= ROW_INSIDE;
                    }
                }
            }
        }
        for x in 0..w {
            let first = (0..h).find(|&y| solid(x, y));
            let last = (0..h).rev().find(|&y| solid(x, y));
            if let (Some(a), Some(b)) = (first, last) {
                let mut roof = a;
                for y in a + 1..=b {
                    if !solid(x, y) {
                        env[y * w + x] |= COL_INSIDE;
                        continue;
                    }
                    // A short run of air with the body above and below it is
                    // under cover, like a hold under its deck; the air under a
                    // spar high over the deck is not.
                    if y - roof - 1 <= COVER_DEPTH {
                        for k in roof + 1..y {
                            env[k * w + x] |= COVERED;
                        }
                    }
                    roof = y;
                }
            }
        }
        // Air between solid cells of its row with nothing over it in its
        // column holds water only near a floor: the inside of an open boat,
        // or water on a deck. Higher up it is the air between two masts,
        // which holds nothing. Air with the body over it, such as a hold
        // holed in its floor, is still held.
        for x in 0..w {
            let mut below = None;
            let mut roofed = vec![false; h];
            let mut over = None;
            for (y, r) in roofed.iter_mut().enumerate() {
                if solid(x, y) {
                    over = Some(y);
                } else {
                    *r = over.is_some_and(|o| y - o <= COVER_DEPTH);
                }
            }
            for y in (0..h).rev() {
                if solid(x, y) {
                    below = Some(y);
                    continue;
                }
                let e = &mut env[y * w + x];
                if *e & ROW_INSIDE != 0
                    && *e & COL_INSIDE == 0
                    && !roofed[y]
                    && below.is_none_or(|b| b - y > OPEN_DEPTH)
                {
                    *e &= !ROW_INSIDE;
                }
            }
        }
        // Air no path through air leads out of: a sealed hold.
        let mut open = vec![false; w * h];
        let mut stack: Vec<(usize, usize)> = Vec::new();
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
            if open[i] || solid(x, y) {
                continue;
            }
            open[i] = true;
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
        for (i, e) in env.iter_mut().enumerate() {
            if !open[i] && self.cells[i].is_air() {
                *e |= ENCLOSED;
            }
        }
        // A hold holed in its floor: the air over the hole has the deck above
        // it but no floor under it, so the columns leave it uncovered, though
        // it lies between the hull's sides among covered air. A run of air
        // between two solid cells of a row that is mostly inside is inside
        // all along, and so is the air between the sides of the hole under
        // it. The air under an arch or a bridge, or over a deck under the
        // eaves of a deckhouse, is mostly not, so it stays what lies behind
        // the body.
        let inside = |e: u8| e & (COVERED | ENCLOSED) != 0;
        for y in 0..h {
            let row = &mut env[y * w..(y + 1) * w];
            let mut x = 0;
            while x < w {
                if solid(x, y) {
                    x += 1;
                    continue;
                }
                let a = x;
                while x < w && !solid(x, y) {
                    x += 1;
                }
                let covered = row[a..x].iter().filter(|&&e| inside(e)).count();
                if a > 0 && x < w && covered * 2 > x - a {
                    for e in &mut row[a..x] {
                        *e |= COVERED;
                    }
                }
            }
        }
        for y in 1..h {
            for x in 0..w {
                let i = y * w + x;
                if !solid(x, y) && env[i] & ROW_INSIDE != 0 && inside(env[i - w]) {
                    env[i] |= COVERED;
                }
            }
        }
        env
    }

    /// True if the last stamp placed every cell it could reach.
    pub fn fully_stamped(&self) -> bool {
        self.wet.is_empty()
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

    /// True if the body moved since its last stamp, was never stamped, or its
    /// last stamp left cells out.
    pub(crate) fn needs_restamp(&self, rapier: &PhysicsWorld) -> bool {
        match rapier.bodies.get(self.handle) {
            Some(rb) => {
                self.stamped.is_empty()
                    || !self.wet.is_empty()
                    || pose_bits(rb.position()) != self.last_pose
            }
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
        self.wet.clear();
        self.hollow.clear();
    }

    /// Writes the body's cells into the world at its current pose, pushing fluids
    /// aside. A cell whose fluid finds no room stays out (see `wet`). `others`
    /// are the other bodies, whose holds the fluid is kept out of.
    pub(crate) fn stamp(&mut self, world: &mut World, rapier: &PhysicsWorld, others: &Others<'_>) {
        let Some(rb) = rapier.bodies.get(self.handle) else { return };
        let pose = *rb.position();
        if !self.stamped.is_empty() {
            self.unstamp(world);
        }
        // Where the body was stamped last, to tell what it holds from the sea.
        let prev = Frame::from_bits(self, self.last_pose);
        self.last_pose = pose_bits(&pose);
        let frame = Frame::new(self, &pose);
        let (x0, y0, x1, y1) = frame.world_bounds();
        // The world cells the body covers, and the outline of each world row,
        // which tells displaced fluid where the surface beside the body is.
        let mut targets = Vec::new();
        let mut hollow = Vec::new();
        let mut rows = vec![(i32::MAX, i32::MIN); (y1 - y0 + 1) as usize];
        for wy in y0..=y1 {
            for wx in x0..=x1 {
                let p = CellPos::new(wx, wy);
                let Some(idx) = frame.local_of(p) else { continue };
                if self.cells[idx].is_air() {
                    let e = self.envelope[idx];
                    if e & (ENCLOSED | COVERED) != 0 {
                        hollow.push(p);
                    }
                    continue;
                }
                let r = &mut rows[(wy - y0) as usize];
                *r = (r.0.min(wx), r.1.max(wx));
                targets.push((p, idx));
            }
        }
        let mut shore = Shore { frame, prev, body: self, others, y0, rows, cursor: [None, None] };
        let mut placed = Vec::with_capacity(targets.len());
        let mut wet = Vec::new();
        for (p, idx) in targets {
            match place(world, p, self.cells[idx], &mut shore) {
                Place::Placed => placed.push((p, idx as u32)),
                Place::Blocked => {}
                Place::NoRoom => wet.push(p),
            }
        }
        self.stamped = placed;
        self.wet = wet;
        self.hollow = hollow;
    }

    /// Floods a hold that is open to the sea below the waterline, then applies
    /// buoyancy, the weight of what the hull holds, and drag from the liquids
    /// and powders around the body.
    ///
    /// The medium's level is read just outside the body on the left and on the
    /// right (see [`Shoreline::measure`]) and taken as the mean of the two.
    /// Every solid cell and every cell of the envelope (a hold, the inside of
    /// an open hull) below that line displaces the medium and is lifted by its
    /// weight, the cell at the line in proportion to how far it is under, so a
    /// wide hull with a dry hold floats with its deck clear of the water, and a
    /// flooded hold adds no lift. Liquid and powder inside the envelope weigh
    /// on the hull where they lie. Each force acts where it is centered, so a
    /// floating body that tilts finds the lift moving to its lower side and
    /// rights itself, and a hull flooding on one side lists that way.
    pub(crate) fn apply_fluid_forces(&mut self, rapier: &mut PhysicsWorld, world: &mut World) {
        let Some(rb) = rapier.bodies.get_mut(self.handle) else { return };
        rb.reset_forces(false);
        rb.reset_torques(false);
        self.floating = false;
        let in_air = |rb: &mut RigidBody| {
            rb.set_linear_damping(0.0);
            rb.set_angular_damping(0.05);
        };
        if self.stamped.is_empty() && self.wet.is_empty() {
            in_air(rb);
            return;
        }
        // The rows and columns the body covers in the world.
        let footprint = || self.stamped.iter().map(|s| s.0).chain(self.wet.iter().copied());
        let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        for p in footprint() {
            (x0, y0, x1, y1) = (x0.min(p.x), y0.min(p.y), x1.max(p.x), y1.max(p.y));
        }
        let (ymin, n) = (y0, (y1 - y0 + 1) as usize);
        let l = Shoreline::measure(world, x0 - 1, ymin, n, -1);
        let r = Shoreline::measure(world, x1 + 1, ymin, n, 1);
        if l.level.is_none() && r.level.is_none() {
            in_air(rb);
            self.wake_on_change(rb, (0.0, 0.0));
            return;
        }
        // Under a wide hull the cell world does not carry pressure from one
        // side to the other, so the two levels can differ for a long time; the
        // water under the hull is taken as level, at their mean.
        let level = match (l.level, r.level) {
            (Some(a), Some(b)) => (a + b) * 0.5,
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => f32::MAX,
        };
        let mid = (x0 + x1) as f32 * 0.5 + 0.5;
        let medium_at = |x: f32, row: usize| {
            let (a, b) = if x <= mid { (&l.media, &r.media) } else { (&r.media, &l.media) };
            a[row].or(b[row])
        };
        let frame = Frame::new(self, rb.position());
        // A body lies in a sea, whose pressure the flooding stands in for,
        // when the medium touches it on both sides. A spill running down one
        // side of a cask against a wall is no sea.
        if l.level.is_some() && r.level.is_some() && self.flood(world, &frame, level) > 0 {
            rb.wake_up(true);
        }
        let world: &World = world;
        // The envelope is taken along the local axis nearest to the world's
        // horizontal, so an open hull holds air only while its opening faces up.
        let bit = if frame.c.abs() >= frame.s.abs() { ROW_INSIDE } else { COL_INSIDE };
        let (w, h) = (i32::from(self.w), i32::from(self.h));
        let mut lift = Moment::default();
        let mut load = Moment::default();
        let (mut cells, mut sub, mut visc) = (0u32, 0.0f32, 0.0f32);
        for ly in 0..h {
            for lx in 0..w {
                let i = (ly * w + lx) as usize;
                let inside = self.cells[i].is_air();
                if inside && self.envelope[i] & bit == 0 {
                    continue;
                }
                let (wx, wy) = frame.world_of(lx, ly);
                if inside {
                    let c = world.get(CellPos::new(wx.floor() as i32, wy.floor() as i32));
                    if !c.is_air() && !c.has(Cell::FLAG_IN_BODY) {
                        let m = world.materials().get(c.material());
                        match m.state {
                            // What the hull holds weighs on it where it lies.
                            State::Liquid | State::Powder => {
                                load.add(f32::from(m.density.max(0)), wx, wy);
                            }
                            // Terrain poking into the hull displaces nothing.
                            State::Static => continue,
                            _ => {}
                        }
                    }
                }
                cells += 1;
                let f = (wy + 0.5 - level).clamp(0.0, 1.0);
                if f <= 0.0 {
                    continue;
                }
                let row = (wy.floor() as i32 - ymin).clamp(0, n as i32 - 1) as usize;
                let Some((density, v)) = medium_at(wx, row) else { continue };
                lift.add(f * f32::from(density.max(0)), wx, wy);
                sub += f;
                visc += f * f32::from(v);
            }
        }
        // A cell of density 100 holds 1 kg.
        let g = crate::GRAVITY / 100.0;
        let com = rb.translation();
        let mut push = (0.0f32, 0.0f32);
        for (m, sign) in [(&lift, -1.0f32), (&load, 1.0)] {
            if let Some(at) = m.center() {
                let f = sign * m.weight * g;
                rb.add_force_at_point(Vector::new(0.0, f), at, false);
                push.0 += f;
                push.1 += (at.x - com.x) * f;
            }
        }
        self.wake_on_change(rb, push);
        if sub > 0.0 {
            self.floating = true;
            let frac = sub / cells.max(1) as f32;
            let avg_visc = visc / sub;
            rb.set_linear_damping(frac * avg_visc / 8.0);
            rb.set_angular_damping(frac * avg_visc / 6.0);
        } else {
            in_air(rb);
        }
    }

    /// Wakes a sleeping body whose fluid force or torque changed by more than
    /// a fiftieth of its weight (the torque by that much at one meter) since
    /// it fell asleep, so a change that builds up slowly wakes it as well.
    fn wake_on_change(&mut self, rb: &mut RigidBody, push: (f32, f32)) {
        let limit = rb.mass() * crate::GRAVITY / 50.0;
        let (df, dt) = (push.0 - self.fluid_push.0, push.1 - self.fluid_push.1);
        if rb.is_sleeping() {
            if df.abs() <= limit && dt.abs() <= limit {
                return;
            }
            rb.wake_up(true);
        }
        self.fluid_push = push;
    }

    /// Lets the sea into a hold that is open to it below the waterline, and
    /// out again while the water inside stands higher than the sea.
    ///
    /// The cell world carries no pressure, so water does not rise through a
    /// hole in a hull's bottom, nor above the top of a hole in its side, and
    /// water a hull lifts does not run back out through a hole in its bottom.
    /// This stands in for the pressure. An opening is a cell of the hold
    /// below `level` with the liquid outside beside it. The air the openings
    /// reach through the hold below `level` fills with that liquid, lowest
    /// cells first. Once none is left, the liquid they reach through the
    /// liquid in the hold that stands more than a row over `level` runs out,
    /// highest cells first. Each opening cell passes sqrt(2 g h) cells a
    /// tick, water running out of an orifice, where h is how far the level
    /// outside and the level inside or the opening differ, so a hold floods
    /// in seconds through a shell hole and in minutes through a seam. The
    /// liquid comes from, and goes back to, the surface of the liquid outside
    /// the openings, found through that liquid, the highest cells first and
    /// the lowest air first, so none is made or lost and none moves between
    /// pools. A hold whose openings all lie above the water, such as an open
    /// boat or a holed deck, takes on none this way, and neither does a body
    /// that holds less than [`MIN_HOLD`] cells of air, such as a cracked plank.
    ///
    /// `level` is the world y of the top of the medium outside. Returns the
    /// number of cells moved.
    fn flood(&self, world: &mut World, frame: &Frame, level: f32) -> usize {
        // A crack or a notch in a plank is no hold, and a sealed hold has no
        // opening.
        let (mut held, mut open) = (0, false);
        for (c, &e) in self.cells.iter().zip(&self.envelope) {
            if c.is_air() && e & HOLD != 0 {
                held += 1;
                open |= e & ENCLOSED == 0;
            }
        }
        if held < MIN_HOLD || !open {
            return 0;
        }
        // The first row under the surface outside.
        let top = level.ceil() as i32;
        let (bx0, by0, bx1, by1) = frame.world_bounds();
        if top > by1 {
            return 0;
        }
        let w = (bx1 - bx0 + 1) as usize;
        let slot = |p: CellPos| (p.y - by0) as usize * w + (p.x - bx0) as usize;
        let boxed = |p: CellPos| p.x >= bx0 && p.x <= bx1 && p.y >= by0 && p.y <= by1;
        let in_hold = |p: CellPos| {
            frame
                .local_of(p)
                .is_some_and(|i| self.cells[i].is_air() && self.envelope[i] & HOLD != 0)
        };
        let outside = |p: CellPos| {
            frame.local_of(p).is_none_or(|i| self.cells[i].is_air() && self.envelope[i] & HOLD == 0)
        };
        let state = |c: Cell| world.materials().get(c.material()).state;
        let liquid = |c: Cell| !c.has(Cell::FLAG_IN_BODY) && state(c) == State::Liquid;
        // What water fills or runs through: air, gas, fire and liquid.
        let passable = |c: Cell| {
            c.is_air()
                || (!c.has(Cell::FLAG_IN_BODY)
                    && matches!(state(c), State::Liquid | State::Gas | State::Fire))
        };
        let around = |p: CellPos| {
            [(0, 1), (-1, 0), (1, 0), (0, -1)].map(|(dx, dy)| CellPos::new(p.x + dx, p.y + dy))
        };
        // Each opening, with the liquid outside it.
        let mut openings: Vec<(CellPos, CellPos)> = Vec::new();
        let mut fluid: Option<(i32, MaterialId)> = None;
        for y in by0.max(top)..=by1 {
            for x in bx0..=bx1 {
                let p = CellPos::new(x, y);
                if !in_hold(p) || !passable(world.get(p)) {
                    continue;
                }
                let beside = around(p)
                    .into_iter()
                    .find(|&o| o.y >= top && outside(o) && liquid(world.get(o)));
                if let Some(o) = beside {
                    openings.push((p, o));
                    // The hold takes the liquid at its deepest opening.
                    if fluid.is_none_or(|(row, _)| y > row) {
                        fluid = Some((y, world.get(o).material()));
                    }
                }
            }
        }
        let Some((_, fluid)) = fluid else { return 0 };
        // Searches the hold from the openings through the cells `pass` lets
        // through, and returns the cells `take` picks.
        let search = |pass: &dyn Fn(CellPos) -> bool, take: &dyn Fn(CellPos) -> bool| {
            let mut seen = vec![false; w * (by1 - by0 + 1) as usize];
            let mut queue: Vec<CellPos> =
                openings.iter().map(|&(p, _)| p).filter(|&p| pass(p)).collect();
            for &p in &queue {
                seen[slot(p)] = true;
            }
            let mut found = Vec::new();
            let mut k = 0;
            while k < queue.len() {
                let p = queue[k];
                k += 1;
                if take(p) {
                    found.push(p);
                }
                for q in around(p) {
                    if boxed(q) && !seen[slot(q)] && in_hold(q) && pass(q) {
                        seen[slot(q)] = true;
                        queue.push(q);
                    }
                }
            }
            found
        };
        let is_fluid = |p: CellPos| {
            let c = world.get(p);
            c.material() == fluid && !c.has(Cell::FLAG_IN_BODY)
        };
        let mut air = search(&|p| p.y >= top && passable(world.get(p)), &|p| world.get(p).is_air());
        let filling = !air.is_empty();
        // The cells that change, the first first, and the head over each opening.
        let (cells, head): (Vec<CellPos>, Box<dyn Fn(CellPos) -> f32>) = if filling {
            air.sort_unstable_by_key(|p| (std::cmp::Reverse(p.y), p.x));
            // The top of the liquid inside, under the lowest air.
            let inside = air[0].y as f32 + 1.0;
            (air, Box::new(move |o: CellPos| (o.y as f32 + 0.5).min(inside) - level))
        } else {
            let mut high = search(&is_fluid, &|p| p.y < top - 1);
            high.sort_unstable_by_key(|p| (p.y, p.x));
            let Some(first) = high.first() else { return 0 };
            let inside = first.y as f32;
            (high, Box::new(move |o: CellPos| if is_fluid(o) { level - inside } else { 0.0 }))
        };
        let tps = fe_core::TICKS_PER_SECOND as f32;
        let g = crate::GRAVITY * CELLS_PER_METER / (tps * tps);
        let mut rate = 0.0f32;
        for &(o, _) in &openings {
            let h = head(o);
            if h > 0.0 {
                rate += (2.0 * g * h).sqrt();
            }
        }
        // Whole cells, with the fraction rounded up by chance.
        let chance = fe_core::cell_rand(world.seed(), world.tick(), self.id as i32, 0, 0xf100);
        let mut n = rate as usize;
        if (chance >> 8) as f32 / 16_777_216.0 < rate.fract() {
            n += 1;
        }
        let n = n.min(cells.len()).min(FLOOD_MAX);
        if n == 0 {
            return 0;
        }
        // The top of the liquid outside that reaches the openings: liquid
        // comes from the highest of it, and goes onto the lowest.
        let starts: Vec<CellPos> =
            openings.iter().map(|&(_, o)| o).filter(|&o| is_fluid(o)).collect();
        let sea = |p: CellPos| is_fluid(p) && !in_hold(p);
        let mut tops = surface_of(world, &starts, &sea, n * 2 + 8);
        let outer: Vec<CellPos> = if filling {
            tops.sort_unstable_by_key(|p| (p.y, p.x));
            tops.into_iter().take(n).collect()
        } else {
            tops.sort_unstable_by_key(|p| (std::cmp::Reverse(p.y), p.x));
            tops.iter()
                .map(|p| CellPos::new(p.x, p.y - 1))
                .filter(|&p| world.get(p).is_air())
                .take(n)
                .collect()
        };
        drop(head);
        for (&a, &b) in cells.iter().zip(&outer) {
            let (ca, cb) = (world.get(a), world.get(b));
            world.grid.set(a, cb);
            world.grid.set(b, ca);
        }
        outer.len()
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
        h.write_u64(self.wet.len() as u64);
    }
}

/// Covers the cells where `solid` is true with rectangles: horizontal runs
/// merged with identical runs below. Returns `(x0, y0, x1, y1)`, ends exclusive.
pub(crate) fn merge_rects(
    w: i32,
    h: i32,
    solid: impl Fn(i32, i32) -> bool,
) -> Vec<(i32, i32, i32, i32)> {
    let mut used = vec![false; (w * h) as usize];
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

/// A body that holds air: its index in the body list, its frame at its pose
/// in the physics world, and a box round the world cells its grid can cover.
struct Holder {
    index: usize,
    frame: Frame,
    bounds: (i32, i32, i32, i32),
}

/// The bodies that hold air, for keeping fluid out of their holds.
pub(crate) struct Holders(Vec<Holder>);

impl Holders {
    /// The bodies of `bodies` that hold any air, at their poses in `rapier`,
    /// with a box round the world cells their local grids can cover.
    pub(crate) fn of(bodies: &[PixelBody], rapier: &PhysicsWorld) -> Self {
        Self(
            bodies
                .iter()
                .enumerate()
                .filter(|(_, b)| b.envelope.iter().any(|&e| e != 0))
                .filter_map(|(index, b)| {
                    let frame = Frame::new(b, rapier.bodies.get(b.handle)?.position());
                    // A cell wider, so rounding at the edge never rejects a
                    // cell that maps into the grid.
                    let (x0, y0, x1, y1) = frame.world_bounds();
                    Some(Holder { index, frame, bounds: (x0 - 1, y0 - 1, x1 + 1, y1 + 1) })
                })
                .collect(),
        )
    }
}

/// The bodies other than the one being stamped, at the poses they are
/// stamped at this tick: `before` and `after` are the bodies before and after
/// it, number `me`, in the list [`Holders`] was made from.
pub(crate) struct Others<'a> {
    holders: &'a Holders,
    me: usize,
    before: &'a [PixelBody],
    after: &'a [PixelBody],
}

impl<'a> Others<'a> {
    pub(crate) const fn new(
        holders: &'a Holders,
        me: usize,
        before: &'a [PixelBody],
        after: &'a [PixelBody],
    ) -> Self {
        Self { holders, me, before, after }
    }

    /// True if `q` is air one of the other bodies holds.
    fn holds(&self, q: CellPos) -> bool {
        self.holders.0.iter().any(|h| {
            let (x0, y0, x1, y1) = h.bounds;
            if q.x < x0 || q.y < y0 || q.x > x1 || q.y > y1 {
                return false;
            }
            let b = match h.index.cmp(&self.me) {
                std::cmp::Ordering::Less => &self.before[h.index],
                std::cmp::Ordering::Greater => &self.after[h.index - self.me - 1],
                std::cmp::Ordering::Equal => return false,
            };
            h.frame.local_of(q).is_some_and(|i| b.cells[i].is_air() && b.envelope[i] != 0)
        })
    }
}

/// What happened when a body cell was placed.
enum Place {
    /// The cell is in the world.
    Placed,
    /// Something solid is there; the cell stays out.
    Blocked,
    /// A liquid or powder is there with no air to move it to; the cell
    /// stays out so the displaced cell is not destroyed.
    NoRoom,
}

/// [`PixelBody::envelope`] bit: between solid cells of the local row.
const ROW_INSIDE: u8 = 1;
/// [`PixelBody::envelope`] bit: between solid cells of the local column.
const COL_INSIDE: u8 = 2;
/// [`PixelBody::envelope`] bit: closed in on every side, like a sealed hold.
const ENCLOSED: u8 = 4;
/// [`PixelBody::envelope`] bit: in a run of air no more than [`COVER_DEPTH`]
/// cells tall between solid cells of the local column, or in a row of a hold
/// with such air in it (see [`PixelBody::find_envelope`]). Only drawing reads it.
const COVERED: u8 = 8;
/// [`PixelBody::envelope`] bits of the air a hull holds: between its solid
/// cells along a local row or column.
const HOLD: u8 = ROW_INSIDE | COL_INSIDE;
/// The tallest gap between a body's cells, along its local column, that still
/// draws as the body's inside rather than as what lies behind it, and the
/// farthest air with the body over it may lie from it and still be held.
const COVER_DEPTH: usize = 48;
/// The deepest air open above that a body holds over a floor of its own: the
/// inside of an open boat, or water on a deck between its bulwarks.
const OPEN_DEPTH: usize = 8;

/// A body's pose as plain numbers, for mapping between its local grid and
/// world cells. Positions are in cells.
#[derive(Clone, Copy)]
struct Frame {
    tx: f32,
    ty: f32,
    c: f32,
    s: f32,
    com: (f32, f32),
    w: i32,
    h: i32,
}

impl Frame {
    fn new(body: &PixelBody, pose: &Pose) -> Self {
        Self {
            tx: pose.translation.x * CELLS_PER_METER,
            ty: pose.translation.y * CELLS_PER_METER,
            c: pose.rotation.cos(),
            s: pose.rotation.sin(),
            com: body.com,
            w: i32::from(body.w),
            h: i32::from(body.h),
        }
    }

    /// The frame of a stamp made with [`pose_bits`], or `None` if there was
    /// none: the body was never stamped, or its shape changed since.
    fn from_bits(body: &PixelBody, b: [u32; 4]) -> Option<Self> {
        if b == [0; 4] || b == [u32::MAX; 4] {
            return None;
        }
        Some(Self {
            tx: f32::from_bits(b[0]) * CELLS_PER_METER,
            ty: f32::from_bits(b[1]) * CELLS_PER_METER,
            c: f32::from_bits(b[2]),
            s: f32::from_bits(b[3]),
            com: body.com,
            w: i32::from(body.w),
            h: i32::from(body.h),
        })
    }

    /// World position of the center of local cell `(lx, ly)`.
    fn world_of(&self, lx: i32, ly: i32) -> (f32, f32) {
        let ox = lx as f32 + 0.5 - self.com.0;
        let oy = ly as f32 + 0.5 - self.com.1;
        (self.tx + self.c * ox - self.s * oy, self.ty + self.s * ox + self.c * oy)
    }

    /// Index of the local cell under the center of world cell `p`, if any.
    fn local_of(&self, p: CellPos) -> Option<usize> {
        let px = p.x as f32 + 0.5 - self.tx;
        let py = p.y as f32 + 0.5 - self.ty;
        let lx = (self.c * px + self.s * py + self.com.0).floor() as i32;
        let ly = (-self.s * px + self.c * py + self.com.1).floor() as i32;
        if lx < 0 || ly < 0 || lx >= self.w || ly >= self.h {
            return None;
        }
        Some((ly * self.w + lx) as usize)
    }

    /// The world cells the local grid can cover: `(x0, y0, x1, y1)`, inclusive.
    fn world_bounds(&self) -> (i32, i32, i32, i32) {
        let (w, h) = (self.w as f32, self.h as f32);
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for (lx, ly) in [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)] {
            let (ox, oy) = (lx - self.com.0, ly - self.com.1);
            let wx = self.tx + self.c * ox - self.s * oy;
            let wy = self.ty + self.s * ox + self.c * oy;
            x0 = x0.min(wx);
            y0 = y0.min(wy);
            x1 = x1.max(wx);
            y1 = y1.max(wy);
        }
        (x0.floor() as i32, y0.floor() as i32, x1.floor() as i32, y1.floor() as i32)
    }
}

/// A sum of weights at positions, for the point a force acts at.
#[derive(Default)]
struct Moment {
    weight: f32,
    x: f32,
    y: f32,
}

impl Moment {
    fn add(&mut self, k: f32, x: f32, y: f32) {
        self.weight += k;
        self.x += k * x;
        self.y += k * y;
    }

    /// The weighted center in physics meters, or `None` if nothing was added.
    fn center(&self) -> Option<Vector> {
        (self.weight > 0.0).then(|| {
            Vector::new(
                self.x / self.weight / CELLS_PER_METER,
                self.y / self.weight / CELLS_PER_METER,
            )
        })
    }
}

/// A liquid or powder a body can float in: its density and viscosity.
fn medium(world: &World, c: Cell) -> Option<(i16, u8)> {
    if c.is_air() || c.has(Cell::FLAG_IN_BODY) {
        return None;
    }
    let m = world.materials().get(c.material());
    matches!(m.state, State::Liquid | State::Powder).then_some((m.density, m.viscosity.max(40)))
}

/// Air, gas or fire: room a moving body can leave behind it.
fn open(world: &World, c: Cell) -> bool {
    c.is_air()
        || (!c.has(Cell::FLAG_IN_BODY)
            && matches!(world.materials().get(c.material()).state, State::Gas | State::Fire))
}

/// The level of the liquid or powder against one side of a body.
struct Shoreline {
    /// World y of the top of the medium beside the body, or `None` if no
    /// medium touches this side.
    level: Option<f32>,
    /// The medium beside each row, from the level down.
    media: Vec<Option<(i16, u8)>>,
}

impl Shoreline {
    /// Reads the medium in the `n` rows from `ymin` beside a body, in the
    /// column `x` just outside it and up to two cells further out on the side
    /// `dir` points to, past the air a moving body leaves behind. The column is
    /// outside the whole body, so a hole in its side does not show its hold.
    ///
    /// The level is the top of the medium that reaches up from below: rows are
    /// read from the bottom up, rows against something solid neither count nor
    /// break the run, and more than [`LEVEL_GAP`] dry rows in a row end it.
    /// Drops running down the side of a body that leaves the water, or a
    /// splash beside it, then do not hold it up.
    fn measure(world: &World, x: i32, ymin: i32, n: usize, dir: i32) -> Self {
        /// What lies beside a row.
        #[derive(Clone, Copy, PartialEq)]
        enum Beside {
            Medium((i16, u8)),
            Open,
            Solid,
        }
        let beside: Vec<Beside> = (0..n)
            .map(|i| {
                let y = ymin + i as i32;
                match (0..3)
                    .map(|k| world.get(CellPos::new(x + dir * k, y)))
                    .find(|&c| !open(world, c))
                {
                    None => Beside::Open,
                    Some(c) => medium(world, c).map_or(Beside::Solid, Beside::Medium),
                }
            })
            .collect();
        let mut out = Self { level: None, media: vec![None; n] };
        let (mut top, mut gap) = (None, 0);
        for i in (0..n).rev() {
            match beside[i] {
                Beside::Medium(_) => {
                    top = Some(i);
                    gap = 0;
                }
                Beside::Open => {
                    gap += 1;
                    if gap > LEVEL_GAP {
                        break;
                    }
                }
                Beside::Solid => {}
            }
        }
        let Some(top) = top else { return out };
        // A body wet up to its top row is under: the level is above it.
        let level = if top == 0 { ymin - 1 } else { ymin + top as i32 };
        out.level = Some(level as f32);
        let mut carry = None;
        for i in top..n {
            if let Beside::Medium(m) = beside[i] {
                carry = Some(m);
            }
            out.media[i] = carry;
        }
        out
    }
}

/// Up to `want` cells at the top of the liquid that `sea` lets through and
/// `starts` lie in: cells of it with open air or gas above them. They are
/// searched best-first by height through the liquid, so an opening deep
/// under a hull finds the surface beside the hull, and liquid in another
/// pool is never taken. The search stops after [`SURFACE_SEARCH`] cells.
fn surface_of(
    world: &World,
    starts: &[CellPos],
    sea: &dyn Fn(CellPos) -> bool,
    want: usize,
) -> Vec<CellPos> {
    let mut seen: BTreeSet<(i32, i32)> = BTreeSet::new();
    let mut queue: BTreeSet<(i32, i32)> = BTreeSet::new();
    for s in starts {
        if seen.insert((s.y, s.x)) {
            queue.insert((s.y, s.x));
        }
    }
    let mut tops = Vec::new();
    let mut visited = 0;
    while let Some((y, x)) = queue.pop_first() {
        visited += 1;
        if visited > SURFACE_SEARCH || tops.len() >= want {
            break;
        }
        if open(world, world.get(CellPos::new(x, y - 1))) {
            tops.push(CellPos::new(x, y));
        }
        for (dx, dy) in [(0, -1), (-1, 0), (1, 0), (0, 1)] {
            let q = (y + dy, x + dx);
            if seen.insert(q) && sea(CellPos::new(q.1, q.0)) {
                queue.insert(q);
            }
        }
    }
    tops
}

/// Most cells one hold takes on in a tick.
const FLOOD_MAX: usize = 128;
/// Fewest cells of air a body must hold for its hold to flood.
const MIN_HOLD: usize = 64;
/// Most cells of liquid searched for the surface a flooding hold draws from.
const SURFACE_SEARCH: u32 = 4096;

/// Dry rows beside a body that end the medium's run up its side.
const LEVEL_GAP: u32 = 2;
/// How far along the surface fluid displaced by a body is spread in one tick.
const SHORE_REACH: i32 = 256;

/// Where fluid pushed out of a body's way goes while the body is stamped.
struct Shore<'a> {
    frame: Frame,
    /// The body's frame at its last stamp.
    prev: Option<Frame>,
    body: &'a PixelBody,
    others: &'a Others<'a>,
    /// First world row of `rows`.
    y0: i32,
    /// Leftmost and rightmost world x the body covers in each row.
    rows: Vec<(i32, i32)>,
    /// The last cell filled at the surface on the left and on the right side;
    /// the next one goes beside it, so displaced water spreads over the surface
    /// instead of piling up against the hull.
    cursor: [Option<CellPos>; 2],
}

impl Shore<'_> {
    /// True if `p` is under the body: one of its cells or the air it holds.
    fn inside(&self, p: CellPos) -> bool {
        self.frame
            .local_of(p)
            .is_some_and(|i| !self.body.cells[i].is_air() || self.body.envelope[i] != 0)
    }

    /// True if `p` is air the body holds at its new pose: a cell of its hold.
    fn hold(&self, p: CellPos) -> bool {
        self.frame
            .local_of(p)
            .is_some_and(|i| self.body.cells[i].is_air() && self.body.envelope[i] != 0)
    }

    /// True if `p` was in the body's hold at its last stamp: a liquid there is
    /// the body's cargo, not the sea it floats in.
    fn held(&self, p: CellPos) -> bool {
        self.prev
            .and_then(|f| f.local_of(p))
            .is_some_and(|i| self.body.cells[i].is_air() && self.body.envelope[i] != 0)
    }

    /// Air in the hold that cargo pushed from `from` can reach without going
    /// through the body: through the liquid and powder around it, a few dozen
    /// cells at most, then up its column to the air over the liquid.
    fn hold_air(&self, world: &World, from: CellPos) -> Option<CellPos> {
        let wall = |q: CellPos| {
            let c = world.get(q);
            c.has(Cell::FLAG_IN_BODY)
                || c.material() == MaterialId::BOUNDARY
                || self.frame.local_of(q).is_some_and(|i| !self.body.cells[i].is_air())
                || (!c.is_air() && medium(world, c).is_none())
        };
        let mut seen = vec![from];
        let mut i = 0;
        while i < seen.len() && seen.len() < 96 {
            let p = seen[i];
            i += 1;
            for (dx, dy) in [(0, -1), (-1, 0), (1, 0), (0, 1)] {
                let q = CellPos::new(p.x + dx, p.y + dy);
                if seen.contains(&q) || wall(q) {
                    continue;
                }
                if world.get(q).is_air() && self.hold(q) {
                    return Some(q);
                }
                seen.push(q);
            }
        }
        (1..=256)
            .map(|k| CellPos::new(from.x, from.y - k))
            .take_while(|&q| !wall(q))
            .find(|&q| world.get(q).is_air() && self.hold(q))
    }

    fn row(&self, y: i32) -> Option<(i32, i32)> {
        let i = usize::try_from(y - self.y0).ok()?;
        self.rows.get(i).copied().filter(|r| r.0 <= r.1)
    }

    /// Air at the surface of the medium beside the body, on the side nearer to
    /// `from` first.
    fn surface(&mut self, world: &World, from: CellPos) -> Option<CellPos> {
        let (a, b) = self.row(from.y)?;
        let first = if from.x - a <= b - from.x { -1 } else { 1 };
        self.surface_on(world, from.y, first).or_else(|| self.surface_on(world, from.y, -first))
    }

    fn surface_on(&mut self, world: &World, y: i32, dir: i32) -> Option<CellPos> {
        let side = usize::from(dir > 0);
        if let Some(mut q) = self.cursor[side] {
            for _ in 0..SHORE_REACH {
                q.x += dir;
                let c = world.get(q);
                if self.free(world, q) {
                    self.cursor[side] = Some(q);
                    return Some(q);
                }
                if medium(world, c).is_none() {
                    break;
                }
            }
            self.cursor[side] = None;
        }
        // Up the body's outline while a medium touches it, then straight up.
        // A few rows whose outer cell lies in the body's hold (a hole in its
        // side), or in another body's, are passed over; more than that and
        // this side gives up, so water is never put beside a mast high over
        // the sea.
        let mut x = 0;
        let mut held = 0;
        for yy in (y - 64..=y).rev() {
            if let Some((a, b)) = self.row(yy) {
                x = if dir < 0 { a - 1 } else { b + 1 };
            } else if yy >= self.y0 {
                continue;
            }
            let q = CellPos::new(x, yy);
            if self.inside(q) || self.others.holds(q) {
                held += 1;
                if held > 12 {
                    return None;
                }
                continue;
            }
            held = 0;
            let c = world.get(q);
            if c.is_air() {
                self.cursor[side] = Some(q);
                return Some(q);
            }
            medium(world, c)?;
        }
        None
    }

    /// True if displaced fluid may go to `q`: air that neither this body nor
    /// another holds.
    fn free(&self, world: &World, q: CellPos) -> bool {
        world.get(q).is_air() && !self.inside(q) && !self.others.holds(q)
    }

    /// Moves a displaced liquid or powder cell to air outside the body: close by
    /// if there is any, else to the surface beside the body, else up its column
    /// through the body to the first air above it. It never goes into the air
    /// the body holds, so moving does not pump water into a sealed hold, nor
    /// into the air another body holds, so a stone on a deck does not push the
    /// sea through the deck into the hold under it. A cell that was in the
    /// body's hold, such as the water in a flooded hold or the oil in a tank,
    /// is its cargo: it moves to air in the hold it can reach without passing
    /// through a wall, so the hold neither leaks nor drains as the body moves;
    /// only a full hold spills over. Returns false if none of that works; the
    /// body cell then stays out of the world so nothing is destroyed.
    fn push_aside(&mut self, world: &mut World, from: CellPos, cell: Cell) -> bool {
        if self.held(from)
            && let Some(q) = self.hold_air(world, from)
        {
            world.grid.set(q, cell);
            return true;
        }
        for r in 1..=4 {
            for (dx, dy) in [(-r, 0), (r, 0), (-r, -r), (r, -r), (-r, r), (r, r), (0, -r), (0, r)] {
                let q = CellPos::new(from.x + dx, from.y + dy);
                if self.free(world, q) {
                    world.grid.set(q, cell);
                    return true;
                }
            }
        }
        if let Some(q) = self.surface(world, from) {
            world.grid.set(q, cell);
            return true;
        }
        for k in 1..=256 {
            let q = CellPos::new(from.x, from.y - k);
            let c = world.get(q);
            if self.free(world, q) {
                world.grid.set(q, cell);
                return true;
            }
            let through = c.is_air()
                || c.has(Cell::FLAG_IN_BODY)
                || matches!(
                    world.materials().get(c.material()).state,
                    State::Liquid | State::Powder
                );
            if !through || c.material() == MaterialId::BOUNDARY {
                return false;
            }
        }
        false
    }
}

/// Places a body cell into the world, pushing a fluid or powder cell aside.
fn place(world: &mut World, p: CellPos, cell: Cell, shore: &mut Shore<'_>) -> Place {
    let existing = world.get(p);
    if existing.material() == MaterialId::BOUNDARY || existing.has(Cell::FLAG_IN_BODY) {
        return Place::Blocked;
    }
    if !existing.is_air() {
        let m = world.materials().get(existing.material());
        if m.state == State::Static {
            return Place::Blocked;
        }
        if matches!(m.state, State::Liquid | State::Powder) && !shore.push_aside(world, p, existing)
        {
            return Place::NoRoom;
        }
    }
    world.grid.set(p, cell.with_flag(Cell::FLAG_IN_BODY, true));
    Place::Placed
}
