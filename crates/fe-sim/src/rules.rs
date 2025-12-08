//! Per-cell update rules, run inside scheduler jobs.
//!
//! Every rule reads and writes through a [`Neighborhood`] and draws randomness
//! from [`RuleCtx::rand`], so its result depends only on the world state, the
//! seed and the tick.

use crate::sched::{Neighborhood, REACH};
use fe_core::{AMBIENT_TEMP, Cell, CellPos, LocalRect, MaterialId, mix64};
use fe_materials::{Material, MaterialSet, SpawnDir, State};

/// Fall-speed units per cell: the value byte stores speed in 1/16 cells per tick.
pub const SPEED_UNITS: u32 = 16;
/// Fall speed gained per tick, in 1/16 cells per tick.
pub const GRAVITY: u32 = 2;
/// Fastest fall speed, in 1/16 cells per tick (15.9 cells per tick).
pub const MAX_SPEED: u32 = 255;

/// How far a resting liquid cell looks along a surface for a drop.
const SURFACE_SCAN: i32 = REACH - 1;

const DIRS4: [(i32, i32); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];

/// Something a cell rule asks the world to do after the pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    /// A cell detonated.
    Explosion {
        /// Center of the blast.
        pos: CellPos,
        /// Radius in cells.
        radius: u8,
        /// Damage at the center.
        power: u8,
    },
    /// A structural cell stopped being structural; its neighbors may have lost support.
    SupportLost(CellPos),
}

/// Per-job output, merged in chunk order after each pass.
#[derive(Default, Debug)]
pub struct JobOut {
    /// Events in the order the job produced them.
    pub events: Vec<Event>,
    /// Cells the job visited.
    pub visited: u64,
}

/// Read-only state shared by every job in a tick.
pub struct RuleCtx<'a> {
    /// Material tables.
    pub mats: &'a MaterialSet,
    /// Per-material flag: true if the material has any reaction.
    pub reactive: &'a [bool],
    /// Current tick.
    pub tick: u64,
    base: u64,
}

impl<'a> RuleCtx<'a> {
    /// Creates the context for one tick.
    pub fn new(mats: &'a MaterialSet, reactive: &'a [bool], seed: u64, tick: u64) -> Self {
        Self { mats, reactive, tick, base: mix64(seed ^ mix64(tick.wrapping_add(0x51ed))) }
    }

    /// Random bits for one decision at one world cell on this tick.
    #[inline]
    pub fn rand(&self, x: i32, y: i32, salt: u32) -> u32 {
        let pos = ((x as u32 as u64) << 32) | y as u32 as u64;
        (mix64(self.base ^ pos ^ u64::from(salt).wrapping_mul(0x9e37_79b9_7f4a_7c15)) >> 32) as u32
    }

    #[inline]
    fn parity(&self) -> u32 {
        (self.tick & 1) as u32
    }
}

/// Creates a fresh cell of a material: lifetime, hit points, or zero speed in the value byte.
pub fn new_cell(mats: &MaterialSet, id: MaterialId, temp: Option<u8>, rnd: u32) -> Cell {
    let m = mats.get(id);
    let value = match (m.lifetime, m.state) {
        (Some((a, b)), _) => a + (rnd % (u32::from(b - a) + 1)) as u8,
        (None, State::Static) => m.hp,
        _ => 0,
    };
    Cell::new(id, temp.unwrap_or(m.temp), value)
}

/// Builds the per-material reaction flags used by [`RuleCtx`].
pub fn reactive_table(mats: &MaterialSet) -> Vec<bool> {
    mats.iter().map(|(id, _)| mats.has_reactions(id)).collect()
}

/// Updates every cell in a chunk's dirty rect, bottom row first.
pub fn update_chunk(nb: &Neighborhood<'_>, rect: LocalRect, ctx: &RuleCtx<'_>, out: &mut JobOut) {
    if rect.is_empty() {
        return;
    }
    let origin = nb.pos().origin();
    let forward = ctx.tick & 1 == 0;
    for ly in (i32::from(rect.y0)..=i32::from(rect.y1)).rev() {
        if forward {
            for lx in i32::from(rect.x0)..=i32::from(rect.x1) {
                update_cell(nb, ctx, out, origin, lx, ly);
            }
        } else {
            for lx in (i32::from(rect.x0)..=i32::from(rect.x1)).rev() {
                update_cell(nb, ctx, out, origin, lx, ly);
            }
        }
    }
}

#[inline]
fn update_cell(
    nb: &Neighborhood<'_>,
    ctx: &RuleCtx<'_>,
    out: &mut JobOut,
    origin: CellPos,
    lx: i32,
    ly: i32,
) {
    let c = nb.get(lx, ly);
    out.visited += 1;
    let (wx, wy) = (origin.x + lx, origin.y + ly);
    if c.is_air() {
        if c.temp != AMBIENT_TEMP {
            update_air(nb, ctx, lx, ly, c, wx, wy);
        }
        return;
    }
    let id = c.material();
    if id == MaterialId::BOUNDARY {
        return;
    }
    if c.parity() == ctx.parity() {
        // Either the cell moved here earlier this tick, or it has slept since a
        // tick of the same parity. Look again next tick in case it is the latter.
        nb.keep(lx, ly);
        return;
    }
    let m = ctx.mats.get(id);
    let r = ctx.rand(wx, wy, 0);
    let mut cell = c.with_parity(ctx.tick);
    let mut changed = false;
    let mut awake = false;

    if m.lifetime.is_some() {
        if cell.value <= 1 {
            let nc = new_cell(ctx.mats, m.decay, Some(cell.temp), r);
            transform(nb, ctx, out, lx, ly, wx, wy, c, nc);
            return;
        }
        cell.value -= 1;
        changed = true;
    }

    if m.conductivity > 0 && exchange_heat(nb, ctx, lx, ly, &mut cell, m, r) {
        changed = true;
    }
    if m.hold_temp > cell.temp {
        cell.temp = m.hold_temp;
        changed = true;
    }
    if m.cooling > 0 && cell.temp != AMBIENT_TEMP && ((r >> 24) as u8) < m.cooling {
        cell.temp = if cell.temp > AMBIENT_TEMP { cell.temp - 1 } else { cell.temp + 1 };
        changed = true;
    }

    if let Some(t) = m.melt
        && cell.temp >= t.at
    {
        let nc = new_cell(ctx.mats, t.into, Some(cell.temp), r);
        transform(nb, ctx, out, lx, ly, wx, wy, c, nc);
        return;
    }
    if let Some(t) = m.freeze
        && cell.temp <= t.at
    {
        let nc = new_cell(ctx.mats, t.into, Some(cell.temp), r);
        transform(nb, ctx, out, lx, ly, wx, wy, c, nc);
        return;
    }
    if let Some(e) = m.explode
        && cell.temp >= e.at
    {
        out.events.push(Event::Explosion {
            pos: CellPos::new(wx, wy),
            radius: e.radius,
            power: e.power,
        });
        let nc = Cell::new(MaterialId::AIR, cell.temp, 0);
        transform(nb, ctx, out, lx, ly, wx, wy, c, nc);
        return;
    }
    if let Some(ig) = m.ignite
        && cell.temp >= ig.at
    {
        if (r & 0xffff) < ig.chance {
            let nc = new_cell(ctx.mats, ig.into, Some(cell.temp), r);
            transform(nb, ctx, out, lx, ly, wx, wy, c, nc);
            return;
        }
        awake = true;
    }

    if let Some(s) = m.spawn {
        awake = true;
        let r1 = ctx.rand(wx, wy, 1);
        if (r1 & 0xffff) < s.chance {
            let (dx, dy) = match s.dir {
                SpawnDir::Up => (0, -1),
                SpawnDir::Down => (0, 1),
                SpawnDir::Any => DIRS4[(r1 >> 16) as usize & 3],
            };
            if nb.get(lx + dx, ly + dy).is_air() {
                let spawned = new_cell(ctx.mats, s.into, None, r1 >> 8).with_parity(ctx.tick);
                nb.set(lx + dx, ly + dy, spawned);
                nb.mark(lx + dx, ly + dy);
                if ctx.mats.get(s.into).state == State::Static {
                    nb.solid_changed(lx + dx, ly + dy);
                }
            }
        }
    }

    if ctx.reactive[id.index()] {
        let r2 = ctx.rand(wx, wy, 2);
        let (dx, dy) = DIRS4[(r2 >> 16) as usize & 3];
        let n = nb.get(lx + dx, ly + dy);
        if let Some(rx) = ctx.mats.reaction(id, n.material()) {
            if (r2 & 0xffff) < rx.chance {
                let me = new_cell(ctx.mats, rx.into_self, None, r2);
                let other = new_cell(ctx.mats, rx.into_other, None, r2 >> 7);
                transform(nb, ctx, out, lx + dx, ly + dy, wx + dx, wy + dy, n, other);
                transform(nb, ctx, out, lx, ly, wx, wy, c, me);
                return;
            }
            awake = true;
        }
    }

    if !cell.has(Cell::FLAG_IN_BODY) {
        let moved = match m.state {
            State::Powder => move_powder(nb, ctx, lx, ly, wx, wy, &mut cell, m, r, &mut changed),
            State::Liquid => {
                match move_liquid_inner(nb, ctx, lx, ly, &mut cell, m, r, &mut changed) {
                    LiquidMove::None => false,
                    LiquidMove::Side => true,
                    LiquidMove::Down => {
                        // A falling cell can open a drop for surface cells up to
                        // SURFACE_SCAN away; wake them so they look again.
                        nb.mark_band(lx - SURFACE_SCAN, lx + SURFACE_SCAN, ly);
                        true
                    }
                }
            }
            State::Gas => move_gas(nb, ctx, lx, ly, wx, wy, cell, m),
            State::Fire => move_fire(nb, ctx, lx, ly, wx, wy, cell),
            State::Static | State::Empty => false,
        };
        if moved {
            return;
        }
    }

    nb.set(lx, ly, cell);
    if changed {
        nb.mark(lx, ly);
    } else if awake {
        nb.keep(lx, ly);
    }
}

fn update_air(
    nb: &Neighborhood<'_>,
    ctx: &RuleCtx<'_>,
    lx: i32,
    ly: i32,
    c: Cell,
    wx: i32,
    wy: i32,
) {
    let m = ctx.mats.get(MaterialId::AIR);
    let r = ctx.rand(wx, wy, 0);
    let mut cell = c;
    exchange_heat(nb, ctx, lx, ly, &mut cell, m, r);
    if ((r >> 24) as u8) < m.cooling && cell.temp != AMBIENT_TEMP {
        cell.temp = if cell.temp > AMBIENT_TEMP { cell.temp - 1 } else { cell.temp + 1 };
    }
    nb.set(lx, ly, cell);
    nb.mark(lx, ly);
}

/// Replaces a cell with a different material and records the side effects.
#[expect(clippy::too_many_arguments, reason = "hot path; a struct would add copies")]
fn transform(
    nb: &Neighborhood<'_>,
    ctx: &RuleCtx<'_>,
    out: &mut JobOut,
    lx: i32,
    ly: i32,
    wx: i32,
    wy: i32,
    old: Cell,
    new: Cell,
) {
    let om = ctx.mats.get(old.material());
    let nm = ctx.mats.get(new.material());
    let in_body = old.has(Cell::FLAG_IN_BODY);
    let mut new = new.with_parity(ctx.tick);
    if in_body && nm.state == State::Static {
        new = new.with_flag(Cell::FLAG_IN_BODY, true);
    }
    nb.set(lx, ly, new);
    nb.mark(lx, ly);
    if in_body {
        return;
    }
    if om.state == State::Static || nm.state == State::Static {
        nb.solid_changed(lx, ly);
    }
    let was_structural = om.state == State::Static && om.structural;
    let is_structural = nm.state == State::Static && nm.structural;
    if was_structural && !is_structural {
        out.events.push(Event::SupportLost(CellPos::new(wx, wy)));
    }
}

#[inline]
fn exchange_heat(
    nb: &Neighborhood<'_>,
    ctx: &RuleCtx<'_>,
    lx: i32,
    ly: i32,
    cell: &mut Cell,
    m: &Material,
    r: u32,
) -> bool {
    let (dx, dy) = DIRS4[(r >> 16) as usize & 3];
    let n = nb.get(lx + dx, ly + dy);
    if n.material() == MaterialId::BOUNDARY {
        return false;
    }
    let diff = i32::from(n.temp) - i32::from(cell.temp);
    if diff.abs() < 2 {
        return false;
    }
    let k = i32::from(m.conductivity.min(ctx.mats.get(n.material()).conductivity));
    if k == 0 {
        return false;
    }
    let total = diff * k;
    let mut q = total / 256;
    if (((r >> 8) & 0xff) as i32) < (total % 256).abs() {
        q += total.signum();
    }
    if q == 0 {
        return false;
    }
    cell.temp = (i32::from(cell.temp) + q) as u8;
    let mut n2 = n;
    n2.temp = (i32::from(n.temp) - q) as u8;
    nb.set(lx + dx, ly + dy, n2);
    nb.mark(lx + dx, ly + dy);
    true
}

/// Can a moving cell enter `target`? `sinking` is true when moving down or sideways
/// into lighter material, false when rising into heavier material.
#[inline]
fn can_enter(ctx: &RuleCtx<'_>, m: &Material, target: Cell, sinking: bool, r: u32) -> bool {
    if target.is_air() {
        return true;
    }
    if target.has(Cell::FLAG_IN_BODY) {
        return false;
    }
    let tm = ctx.mats.get(target.material());
    if !tm.state.is_fluid() {
        return false;
    }
    let ok = if sinking { tm.density < m.density } else { tm.density > m.density };
    // Swapping through a liquid is slower than falling through air or gas.
    ok && (tm.state != State::Liquid || r & 1 == 0)
}

/// Moves `cell` from `(lx, ly)` to `(tx, ty)`, putting the displaced cell back where it came from.
#[inline]
fn swap_to(
    nb: &Neighborhood<'_>,
    ctx: &RuleCtx<'_>,
    lx: i32,
    ly: i32,
    tx: i32,
    ty: i32,
    cell: Cell,
) {
    let displaced = nb.get(tx, ty);
    nb.set(tx, ty, cell);
    nb.set(lx, ly, if displaced.is_air() { displaced } else { displaced.with_parity(ctx.tick) });
    nb.mark(lx, ly);
    nb.mark(tx, ty);
}

/// Falls straight down with acceleration. Returns true if the cell moved.
#[expect(clippy::too_many_arguments, reason = "hot path; a struct would add copies")]
fn fall(
    nb: &Neighborhood<'_>,
    ctx: &RuleCtx<'_>,
    lx: i32,
    ly: i32,
    cell: &mut Cell,
    m: &Material,
    r: u32,
    changed: &mut bool,
) -> bool {
    let has_speed = !m.has_lifetime();
    if !can_enter(ctx, m, nb.get(lx, ly + 1), true, r) {
        if has_speed && cell.value != 0 {
            cell.value = 0;
            *changed = true;
        }
        return false;
    }
    let mut v = if has_speed { u32::from(cell.value) } else { SPEED_UNITS };
    v = (v + GRAVITY).min(MAX_SPEED);
    let mut steps = v / SPEED_UNITS;
    if (r >> 4) & (SPEED_UNITS - 1) < v % SPEED_UNITS {
        steps += 1;
    }
    let steps = steps.max(1) as i32;
    let mut y = ly;
    let mut blocked = false;
    for i in 0..steps {
        let t = nb.get(lx, y + 1);
        if !can_enter(ctx, m, t, true, r >> (8 + i as u32 % 16)) {
            blocked = true;
            break;
        }
        nb.set(lx, y, if t.is_air() { t } else { t.with_parity(ctx.tick) });
        nb.mark(lx, y);
        y += 1;
    }
    if has_speed {
        cell.value = if blocked { (v / 2) as u8 } else { v as u8 };
    }
    nb.set(lx, y, *cell);
    nb.mark(lx, y);
    true
}

#[expect(clippy::too_many_arguments, reason = "hot path; a struct would add copies")]
fn move_powder(
    nb: &Neighborhood<'_>,
    ctx: &RuleCtx<'_>,
    lx: i32,
    ly: i32,
    _wx: i32,
    _wy: i32,
    cell: &mut Cell,
    m: &Material,
    r: u32,
    changed: &mut bool,
) -> bool {
    if fall(nb, ctx, lx, ly, cell, m, r, changed) {
        return true;
    }
    if (r >> 8) & 0xffff >= m.slide {
        return false;
    }
    let d = if r & (1 << 30) == 0 { 1 } else { -1 };
    for dx in [d, -d] {
        if can_enter(ctx, m, nb.get(lx + dx, ly + 1), true, r >> 3) {
            swap_to(nb, ctx, lx, ly, lx + dx, ly + 1, *cell);
            return true;
        }
    }
    false
}

/// How a liquid cell moved.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LiquidMove {
    None,
    Down,
    Side,
}

#[expect(clippy::too_many_arguments, reason = "hot path; a struct would add copies")]
fn move_liquid_inner(
    nb: &Neighborhood<'_>,
    ctx: &RuleCtx<'_>,
    lx: i32,
    ly: i32,
    cell: &mut Cell,
    m: &Material,
    r: u32,
    changed: &mut bool,
) -> LiquidMove {
    if fall(nb, ctx, lx, ly, cell, m, r, changed) {
        return LiquidMove::Down;
    }
    let d = if r & (1 << 30) == 0 { 1 } else { -1 };
    for dx in [d, -d] {
        if can_enter(ctx, m, nb.get(lx + dx, ly + 1), true, r >> 2) {
            swap_to(nb, ctx, lx, ly, lx + dx, ly + 1, *cell);
            return LiquidMove::Down;
        }
    }
    let above = nb.get(lx, ly - 1);
    let pressured = !above.is_air() && ctx.mats.get(above.material()).state == State::Liquid;
    let spread = i32::from(m.spread.max(1));
    for dx in [d, -d] {
        let mut best = None;
        for k in 1..=spread {
            let tx = lx + dx * k;
            let t = nb.get(tx, ly);
            let open = t.is_air()
                || (!t.has(Cell::FLAG_IN_BODY)
                    && matches!(ctx.mats.get(t.material()).state, State::Gas | State::Fire));
            if !open {
                break;
            }
            if can_enter(ctx, m, nb.get(tx, ly + 1), true, 0) {
                best = Some(tx);
                break;
            }
            if pressured {
                best = Some(tx);
            }
        }
        if let Some(tx) = best {
            swap_to(nb, ctx, lx, ly, tx, ly, *cell);
            return LiquidMove::Side;
        }
    }

    // A cell resting on top of liquid with an open side is a bump on the surface.
    // It walks along the surface until it drops off an edge. It cannot climb onto
    // a higher step, so on a staircase the walk always ends one step lower, and
    // the surface levels no matter how wide the steps are. On a level surface the
    // walk ends with a small chance each tick and the cell sleeps.
    let below = nb.get(lx, ly + 1);
    if below.is_air() || ctx.mats.get(below.material()).state != State::Liquid {
        return LiquidMove::None;
    }
    let left = nb.get(lx - 1, ly).is_air();
    let right = nb.get(lx + 1, ly).is_air();
    if !(left || right) || (r >> 26) == 0 {
        return LiquidMove::None;
    }
    // Head for a drop if one is in sight, otherwise toward an open side.
    let mut target_dir = match (left, right) {
        (true, true) => d,
        (true, false) => -1,
        _ => 1,
    };
    'scan: for dx in [d, -d] {
        let mut k = 1;
        while k <= SURFACE_SCAN && nb.get(lx + dx * k, ly).is_air() {
            if nb.get(lx + dx * k, ly + 1).is_air() {
                target_dir = dx;
                break 'scan;
            }
            k += 1;
        }
    }
    let mut tx = lx;
    while tx - lx != target_dir * spread && nb.get(tx + target_dir, ly).is_air() {
        tx += target_dir;
        if nb.get(tx, ly + 1).is_air() {
            break;
        }
    }
    if tx == lx {
        return LiquidMove::None;
    }
    swap_to(nb, ctx, lx, ly, tx, ly, *cell);
    LiquidMove::Side
}

#[expect(clippy::too_many_arguments, reason = "hot path; a struct would add copies")]
fn move_gas(
    nb: &Neighborhood<'_>,
    ctx: &RuleCtx<'_>,
    lx: i32,
    ly: i32,
    wx: i32,
    wy: i32,
    cell: Cell,
    m: &Material,
) -> bool {
    let r = ctx.rand(wx, wy, 3);
    let side = if r & 1 == 0 { 1 } else { -1 };
    let first = match (r >> 1) % 8 {
        0..=3 => (0, -1),
        4 | 5 => (side, -1),
        _ => (side, 0),
    };
    for (dx, dy) in [first, (0, -1), (-side, -1)] {
        if can_enter(ctx, m, nb.get(lx + dx, ly + dy), false, r >> 5) {
            swap_to(nb, ctx, lx, ly, lx + dx, ly + dy, cell);
            return true;
        }
    }
    let spread = i32::from(m.spread.max(1));
    let k = 1 + ((r >> 8) % spread as u32) as i32;
    for dx in [side * k, -side * k] {
        if nb.get(lx + dx, ly).is_air() && nb.get(lx + dx.signum(), ly).is_air() {
            swap_to(nb, ctx, lx, ly, lx + dx, ly, cell);
            return true;
        }
    }
    false
}

fn move_fire(
    nb: &Neighborhood<'_>,
    ctx: &RuleCtx<'_>,
    lx: i32,
    ly: i32,
    wx: i32,
    wy: i32,
    cell: Cell,
) -> bool {
    let r = ctx.rand(wx, wy, 4);
    if r.is_multiple_of(2) {
        return false;
    }
    let side = if r & 8 == 0 { 1 } else { -1 };
    for (dx, dy) in [(0, -1), (side, -1), (-side, -1)] {
        if nb.get(lx + dx, ly + dy).is_air() {
            swap_to(nb, ctx, lx, ly, lx + dx, ly + dy, cell);
            return true;
        }
    }
    false
}
