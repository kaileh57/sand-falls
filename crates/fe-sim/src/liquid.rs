//! Liquid cells: their velocity codes and the rule the checkerboard passes run
//! on them.
//!
//! Resting liquid does not move cell by cell. It moves in the flow phase
//! ([`crate::World::step_cells`] runs it after the passes), which treats each
//! column of resting liquid as a column of shallow water: the top cell of a
//! column carries the column's flow speed, and the phase moves whole columns
//! of cells at once. See `flow.rs`.
//!
//! The per-cell rule here handles what a column model cannot: liquid in the
//! air (a falling stream, a jet, a cell thrown off an edge) moves as single
//! cells along a straight line and lands on the column below it, a void inside
//! a column closes from above, and liquid pressed against a hole in a wall
//! jets out of it.
//!
//! A cell carries one of two codes, told apart by [`Cell::FLAG_LIQ_AIR`]:
//!
//! * Resting liquid: the value byte is `m`, and the flow speed is `m * m`
//!   in 1/4096 cells per tick (up to 15.9 cells per tick), pointing left if
//!   [`Cell::FLAG_LIQ_LEFT`] is set. Only a column's top cell carries a speed;
//!   the cells under it are at rest, with value 0 and both flags clear, which
//!   is what a new cell is.
//! * Liquid in the air: the high nibble is a horizontal speed code and the
//!   low nibble a fall speed code, both in 1/16 cells per tick, with the sign
//!   of the horizontal speed in [`Cell::FLAG_LIQ_LEFT`].

use crate::rules::{Event, JobOut, RuleCtx, gravity_gain};
use crate::sched::{HEAD_MAX, Neighborhood};
use fe_core::{CHUNK_SIZE, Cell, CellPos, Fx, MaterialId};
use fe_materials::{MaterialSet, State};

/// Horizontal speed codes for liquid in the air, in 1/16 cells per tick.
pub const VX_TABLE: [u8; 16] = [0, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 112, 160];
/// Fall speed codes for liquid in the air, in 1/16 cells per tick.
pub const VY_TABLE: [u8; 16] = [0, 2, 4, 7, 11, 16, 22, 30, 40, 52, 67, 85, 108, 140, 190, 255];
/// Gravity's gain in fall speed per tick, in 1/256 of 1/16 cells per tick:
/// 7/256 cells per tick squared.
const GAIN16: i32 = 7 * 16;
/// Change in squared speed (in 1/16 cell per tick, squared) per cell of height:
/// `2 * g * 1 cell` with `g` = 7/256 cells per tick squared.
pub const E1: i32 = 14;
/// Largest flow speed code: 255 * 255 / 4096 = 15.9 cells per tick.
pub const FLOW_MAX: i32 = 255 * 255;

const VX_MAX: i32 = 160;
const VY_MAX: i32 = 255;
/// Cells a void bubbles up a column in one step.
const BUBBLE_SCAN: i32 = 32;
/// Least fall speed of a cell under liquid with open space below it.
const HOLE_FALL: i32 = 16;
/// Fall speed a drip leaves with: that of a cell that has fallen one row.
const DRIP_FALL: i32 = 4;
/// Most cells a jet covers in one tick.
const MAX_WALK: i32 = 11;
/// How far up a side column a pressed cell looks for the roof of a hole.
const ROOF_SCAN: i32 = 16;
/// Speed kept when bouncing off a solid, out of 256.
const WALL_BOUNCE: i32 = 128;
/// Least impact speed that can throw a splash.
const SPLASH_MIN: i32 = 80;
/// Most splashes one job throws per tick.
const MAX_SPLASHES: u32 = 24;
/// How deep a landing cell looks to share its speed with the column below.
const BLEND_DEPTH: i32 = 24;

const SALT_IMPACT: u32 = 17;
const SALT_SPLASH: u32 = 20;
pub(crate) const SALT_ENCODE: u32 = 21;
const SALT_SIDE: u32 = 23;
const SALT_PATH: u32 = 24;

/// For each speed, the code of the largest table entry not above it.
const fn floor_codes<const N: usize>(t: &[u8]) -> [u8; N] {
    let mut out = [0u8; N];
    let mut v = 0;
    let mut code = 0;
    while v < N {
        while code + 1 < t.len() && t[code + 1] as usize <= v {
            code += 1;
        }
        out[v] = code as u8;
        v += 1;
    }
    out
}

const VX_FLOOR: [u8; VX_MAX as usize + 1] = floor_codes::<{ VX_MAX as usize + 1 }>(&VX_TABLE);
const VY_FLOOR: [u8; VY_MAX as usize + 1] = floor_codes::<{ VY_MAX as usize + 1 }>(&VY_TABLE);

/// True for liquid in the air: falling, flying or thrown.
#[inline]
pub const fn is_airborne(c: Cell) -> bool {
    c.has(Cell::FLAG_LIQ_AIR)
}

/// True for liquid falling straight down as part of a stream that left a
/// hole: liquid in the air with no sideways speed and the left flag, which
/// such liquid does not otherwise use. Such liquid gains speed by the rows it
/// crosses ([`fall_row`]), not by the tick, so every cell of the stream
/// speeds up at the same places and the stream stays steady. Liquid falling
/// as a mass, which left its support all at once, gains by the tick and
/// falls as one; by the rows, the rows of a falling slab would cross each
/// place one after another and pull apart.
#[inline]
pub const fn is_stream(c: Cell) -> bool {
    is_airborne(c) && c.value >> 4 == 0 && c.has(Cell::FLAG_LIQ_LEFT)
}

/// Liquid in the air with no sideways speed, marked as part of a stream
/// ([`is_stream`]).
#[inline]
#[must_use]
pub const fn as_stream(c: Cell) -> Cell {
    if is_airborne(c) && c.value >> 4 == 0 { c.with_flag(Cell::FLAG_LIQ_LEFT, true) } else { c }
}

/// True if a liquid cell is falling or flying.
#[inline]
pub const fn falling(c: Cell) -> bool {
    is_airborne(c)
}

/// The flow speed of a resting liquid cell, in 1/4096 cells per tick, negative
/// to the left. Zero for liquid in the air.
#[inline]
pub const fn flow_speed(c: Cell) -> i32 {
    if is_airborne(c) {
        return 0;
    }
    let m = c.value as i32;
    if c.has(Cell::FLAG_LIQ_LEFT) { -m * m } else { m * m }
}

/// A resting liquid cell carrying a flow speed (1/4096 cells per tick). A speed
/// between two codes rounds up with a chance equal to how far it is from the
/// lower one, using 16 bits of `r`, so speeds are kept in expectation.
#[inline]
#[must_use]
pub fn with_flow_speed(c: Cell, u: i32, r: u32) -> Cell {
    let a = u.unsigned_abs().min(FLOW_MAX as u32);
    let mut m = isqrt32(a);
    let lo = m * m;
    if m < 255 && a > lo && (((r & 0xffff) * (2 * m + 1)) >> 16) < a - lo {
        m += 1;
    }
    let mut out =
        c.with_flag(Cell::FLAG_LIQ_AIR, false).with_flag(Cell::FLAG_LIQ_LEFT, m != 0 && u < 0);
    out.value = m as u8;
    out
}

/// The velocity of liquid in the air: horizontal (negative is left) and fall
/// speed, in 1/16 cells per tick.
#[inline]
pub const fn air_velocity(c: Cell) -> (i32, i32) {
    let m = VX_TABLE[(c.value >> 4) as usize] as i32;
    let vx = if c.has(Cell::FLAG_LIQ_LEFT) { -m } else { m };
    (vx, VY_TABLE[(c.value & 15) as usize] as i32)
}

/// A liquid cell's velocity in 1/16 cells per tick: its flight velocity in the
/// air, or its column's flow speed and no fall when resting.
#[inline]
pub const fn decode(c: Cell) -> (i32, i32) {
    if is_airborne(c) { air_velocity(c) } else { (flow_speed(c) / 256, 0) }
}

#[inline]
fn round_code(t: &[u8], floor: &[u8], v: i32, r16: u32) -> usize {
    let top = t.len() - 1;
    if v >= i32::from(t[top]) {
        return top;
    }
    if v <= 0 {
        return 0;
    }
    let i = floor[v as usize] as usize;
    let lo = i32::from(t[i]);
    let span = (i32::from(t[i + 1]) - lo) as u32;
    if ((r16 & 0xffff) * span) >> 16 < (v - lo) as u32 { i + 1 } else { i }
}

/// Liquid in the air with a velocity (1/16 cells per tick), keeping material,
/// temperature and parity. Speeds between codes round stochastically with 16
/// bits of `r` for each axis.
#[inline]
#[must_use]
pub fn airborne(c: Cell, vx: i32, vy: i32, r: u32) -> Cell {
    let xc = round_code(&VX_TABLE, &VX_FLOOR, vx.abs(), r);
    let yc = round_code(&VY_TABLE, &VY_FLOOR, vy, r >> 16);
    let mut out =
        c.with_flag(Cell::FLAG_LIQ_AIR, true).with_flag(Cell::FLAG_LIQ_LEFT, xc != 0 && vx < 0);
    out.value = ((xc as u8) << 4) | yc as u8;
    out
}

/// The horizontal speed (a table value, 1/16 cells per tick) liquid poured
/// over a brink or out of a pipe at speed `v` leaves with: the nearest code,
/// or `prev`, the speed of the liquid poured just before it on the same path,
/// while that is within one code of it. Cells poured on successive ticks then
/// keep one speed, and the sheet or jet they make stays whole. A speed that
/// hovered between two codes was poured at one code on one tick and the other
/// on the next, a quarter faster, and the sheet tore into a comb of bars as
/// the faster slices caught up with the slower ones ahead.
pub(crate) fn pour_speed(v: i32, prev: Option<i32>) -> i32 {
    let code = round_code(&VX_TABLE, &VX_FLOOR, v, 0x8000);
    if let Some(p) = prev {
        let pc = usize::from(VX_FLOOR[p.clamp(0, VX_MAX) as usize]);
        if code.abs_diff(pc) <= 1 {
            return i32::from(VX_TABLE[pc]);
        }
    }
    i32::from(VX_TABLE[code])
}

/// The horizontal speed (1/16 cells per tick) of a sheet poured over a brink
/// at speed `v`: a whole number of cells per tick, at least one, so the sheet
/// moves the same number of columns on every tick and every tick's slice of
/// it is as tall as the last. It keeps `prev`, the speed of the sheet poured
/// just before, while `v` is within three quarters of a cell per tick of it.
/// At a speed between whole cells, a sheet moved one column on one tick and
/// two on the next, so the outflow filled one column to the top and then two
/// to half their height, and the sheet wore a comb of bars. A sheet slower
/// than a cell per tick filled the columns past the lip over several ticks
/// and fell in clumps.
pub(crate) fn sheet_speed(v: i32, prev: Option<i32>) -> i32 {
    const WHOLE: [i32; 7] = [16, 32, 48, 64, 80, 112, 160];
    if let Some(p) = prev
        && WHOLE.contains(&p)
        && (v - p).abs() < 12
    {
        return p;
    }
    WHOLE.iter().copied().min_by_key(|&w| (w - v).abs()).unwrap_or(16)
}

/// Writes a velocity in 1/16 cells per tick: liquid in the air if `vy` is not
/// zero or `air` is set, resting liquid with that flow speed otherwise.
#[inline]
#[must_use]
pub fn encode(c: Cell, vx: i32, vy: i32, r: u32) -> Cell {
    if vy > 0 { airborne(c, vx, vy, r) } else { with_flow_speed(c, vx * 256, r) }
}

/// A liquid cell at rest: no speed and both liquid flags clear.
#[inline]
#[must_use]
pub const fn at_rest(c: Cell) -> Cell {
    let mut out = c.with_flag(Cell::FLAG_LIQ_AIR, false).with_flag(Cell::FLAG_LIQ_LEFT, false);
    out.value = 0;
    out
}

/// A liquid cell at rest that the pressure pass checked and found nothing to
/// do for. It asks again only once something moves it.
#[inline]
pub(crate) const fn settled_cell(c: Cell) -> Cell {
    at_rest(c).with_flag(Cell::FLAG_LIQ_LEFT, true)
}

/// True for a resting liquid cell carrying the settled mark.
#[inline]
pub const fn is_settled(c: Cell) -> bool {
    c.value == 0 && c.has(Cell::FLAG_LIQ_LEFT) && !c.has(Cell::FLAG_LIQ_AIR)
}

/// Integer square root, rounded down.
#[inline]
pub(crate) fn isqrt32(v: u32) -> u32 {
    if v < 2 {
        return v;
    }
    let bits = 32 - v.leading_zeros();
    // 2^ceil(bits / 2) is at least sqrt(v), and Newton's method converges from above.
    let mut x = 1u32 << bits.div_ceil(2);
    loop {
        let y = u32::midpoint(x, v / x);
        if y >= x {
            return x;
        }
        x = y;
    }
}

#[inline]
fn isqrt_i(v: i32) -> i32 {
    isqrt32(v.max(0) as u32) as i32
}

/// Whole cells to move this tick at speed `v` (1/16 cells per tick) with phase
/// `p`: over 16 consecutive phases the steps add up to exactly `v`.
#[inline]
pub(crate) const fn dither(v: i32, p: u64) -> i32 {
    let v = v as u64;
    ((((p + 1) * v) >> 4) - ((p * v) >> 4)) as i32
}

/// The fall speed `vy` (a table value, 1/16 cells per tick) of liquid in the
/// air after a step at place `key` along its path that took `dt` 1/256 ticks.
///
/// Liquid flying sideways gains speed by its steps along its path rather than
/// by the tick. A cell keeps only a speed code, so it moves up to the next
/// code at places along the path picked by a dither with an offset of the
/// code's own, at half the rate that gravity carries it across the code's
/// share of speeds, from halfway between the code and the one below to
/// halfway to the one above. From wherever a cell starts in a code it then
/// stays there on average as long as gravity takes to cross that share,
/// never more than twice as long, and its code is its speed rounded to the
/// nearest code on average. The places are
/// columns where it moves more sideways than down and rows otherwise, so the
/// cells across a jet or a sheet decide alike, and so do the cells that
/// follow one another along its path: every cell of a jet poured with one
/// velocity takes one path, and the stream stays whole. Rounding with random
/// bits of the tick instead put cells of every age at one speed, and a jet
/// fell as a train of cells that broke off whenever the train's speed went
/// up.
pub(crate) fn fall_gain(ctx: &RuleCtx<'_>, vy: i32, key: i32, dt: i64) -> i32 {
    let top = VY_TABLE.len() - 1;
    let i = VY_FLOOR[vy.clamp(0, VY_MAX) as usize] as usize;
    if i >= top {
        return vy;
    }
    // Twice the code's share of speeds, in 1/16 cells per tick.
    let below = if i == 0 { -i64::from(VY_TABLE[1]) } else { i64::from(VY_TABLE[i - 1]) };
    let share2 = i64::from(VY_TABLE[i + 1]) - below;
    // Gravity's gain over the step (1/65536 of 1/16 cells per tick) over
    // twice the share: half the share of it the step crosses, out of 65536.
    let rate = dt * i64::from(GAIN16) / share2.max(1);
    let off = i64::from(ctx.rand_still(i as i32, 0, SALT_PATH) & 0xffff);
    let k = i64::from(key);
    let up = ((k + 1) * rate + off) >> 16 > (k * rate + off) >> 16;
    i32::from(VY_TABLE[if up { i + 1 } else { i }])
}

/// The fall speed code nearest a speed `v` (1/16 cells per tick).
pub(crate) fn nearest_fall(v: i32) -> i32 {
    i32::from(VY_TABLE[round_code(&VY_TABLE, &VY_FLOOR, v, 0x8000)])
}

/// The fall speed of liquid falling straight down at `v` (a speed code)
/// after it crosses into row `y`, gaining gravity's speed by the place, as
/// [`fall_gain`] does along a path.
pub(crate) fn fall_row(ctx: &RuleCtx<'_>, v: i32, y: i32) -> i32 {
    fall_gain(ctx, v, y, 4096 / i64::from(v.max(1)))
}

/// One column of the path of liquid flying sideways at `ax` (1/16 cells per
/// tick, more than its fall speed `v`): the rows it steps down as it crosses
/// into the column with path key `key` (its column, negated if it flies
/// left), where the line of its path through whole cells does, and its fall
/// speed after the column ([`fall_gain`]). Both depend only on the place and
/// the speed it has there, so every cell of a stream that crosses the column
/// with one speed takes the same step.
#[inline]
pub(crate) fn path_column(ctx: &RuleCtx<'_>, v: i32, ax: i32, key: i64) -> (i32, i32) {
    let (a, v64) = (i64::from(ax.max(1)), i64::from(v));
    let dy = (key * v64).div_euclid(a) - ((key - 1) * v64).div_euclid(a);
    (dy as i32, fall_gain(ctx, v, key as i32, 4096 / a))
}

/// Dither phase of horizontal steps on a tick. Every liquid cell in the air
/// shares it (and the tick itself for vertical steps), so cells moving at one
/// speed step together and a falling mass keeps its shape.
#[inline]
const fn xphase(tick: u64) -> u64 {
    tick.wrapping_add(7)
}

/// What kind of liquid a material is, for the liquid rule.
const KIND_NONE: u8 = 0;
/// A liquid with its velocity in the value byte.
const KIND_VEL: u8 = 1;
/// A liquid whose value byte holds a lifetime: it moves, but keeps no speed between ticks.
const KIND_PLAIN: u8 = 2;

/// Liquid parameters per material, derived from its viscosity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LiqFast {
    kind: u8,
    /// Whether hard impacts throw splashes.
    pub can_splash: bool,
    /// Share of the impact energy kept when landing on a solid, out of 256.
    pub e2_solid: u16,
    /// Bed friction of a flowing column, in 1/65536: the speed lost per tick is
    /// `cf * u * u / h`.
    pub cf: u32,
    /// Viscous drag of a flowing column, in 1/65536 cells squared per tick:
    /// the share of its speed lost per tick is `nu / h^2`.
    pub nu: u32,
    /// Share of its speed a flowing column loses per tick, in 1/65536.
    pub damp: u32,
}

impl LiqFast {
    /// True for liquid materials.
    pub const fn is_liquid(self) -> bool {
        self.kind != KIND_NONE
    }

    /// True for liquids that keep a velocity in their cells.
    pub const fn keeps_speed(self) -> bool {
        self.kind == KIND_VEL
    }
}

/// Builds the per-material liquid table from viscosity `v`.
///
/// Every liquid flows by the same rules; viscosity sets its drag. Bed
/// friction is `120 + 4 v` (water, at 20, loses 0.3% of its speed per cell it
/// flows over a cell-deep bed), viscous drag `v^3 / 64` grows as the layer
/// thins, and a steady loss of `24 + (v - 20)^2 / 4` of its speed each tick
/// lets slow sloshing die out and makes thick liquids sluggish: lava slumps at
/// under half the pace of water and stops in a heap. Impacts on solids keep
/// `(256 - v)^2 / 256` of their energy (within 8 and 220), and only liquids
/// thinner than 100 splash.
pub fn liquid_table(mats: &MaterialSet) -> Vec<LiqFast> {
    mats.iter()
        .map(|(_, m)| {
            if m.state != State::Liquid {
                return LiqFast::default();
            }
            let v = u32::from(m.viscosity);
            LiqFast {
                kind: if m.lifetime.is_some() { KIND_PLAIN } else { KIND_VEL },
                can_splash: v < 100,
                e2_solid: ((256 - v).pow(2) / 256).clamp(8, 220) as u16,
                cf: 120 + 4 * v,
                nu: v * v * v / 64,
                damp: 24 + v.saturating_sub(20).pow(2) / 4,
            }
        })
        .collect()
}

/// True if `c` counts toward the head over the cell below it: resting liquid.
/// Liquid in the air presses on nothing.
#[inline]
pub(crate) fn is_head(liq: &[LiqFast], c: Cell) -> bool {
    match liq.get(c.material().index()).map_or(KIND_NONE, |q| q.kind) {
        KIND_VEL => !is_airborne(c),
        KIND_PLAIN => true,
        _ => false,
    }
}

/// Most cells to either side of a cell of liquid in a hole in a floor that
/// the hole's edge may lie: liquid over a hole up to 15 cells wide is held.
pub(crate) const HOLE_W: i32 = 8;
/// Rows above liquid in a hole that a roof over it is looked for.
const HOLE_ROOF: i32 = 24;
/// Rows under liquid over open space that resting liquid of its kind is
/// looked for: open space no deeper than this over liquid is a void.
const HOLE_VOID: i32 = 8;

/// True if `b`, under liquid of density `density`, holds nothing up: open
/// space, a lighter fluid, or falling liquid or powder.
#[inline]
pub(crate) fn unsupported(ctx: &RuleCtx<'_>, density: i16, b: Cell) -> bool {
    if b.is_air() {
        return true;
    }
    if b.has(Cell::FLAG_IN_BODY) {
        return false;
    }
    let f = ctx.f(b);
    match f.state {
        State::Gas | State::Fire | State::Empty => true,
        State::Liquid => is_airborne(b) || f.density < density,
        State::Powder => b.value > 0 && !f.has_lifetime(),
        State::Static => false,
    }
}

/// True if `t` is something solid that a hole's edge can be: a static cell
/// of the terrain or resting powder. A body is no floor with a hole in it: it
/// moves, and liquid held between a mast and a deck edge, or on a spar,
/// would pile up there and never level with the sea around it.
#[inline]
fn solid(ctx: &RuleCtx<'_>, t: Cell) -> bool {
    if t.has(Cell::FLAG_IN_BODY) {
        return false;
    }
    match ctx.f(t).state {
        State::Static => true,
        State::Powder => !unsupported(ctx, 0, t),
        _ => false,
    }
}

/// True if resting liquid `c` at `(x, y)`, with nothing under it, lies over a
/// hole in a floor or in a narrow shaft through one, with liquid of its kind
/// on top of it pressing it in. A hole is a gap in a row of solid cells no
/// wider than [`HOLE_W`] cells each side: the row under the liquid, which
/// holds open space or falling liquid between its solid edges, or the
/// liquid's own row, where liquid of its kind hangs between them. Such
/// liquid is held, and the flow phase drains it through the hole at the
/// speed its head gives (see `flow.rs`). Left to fall, the column over a
/// hole fell as one rod as deep as the pool, the pool refilled it, and a
/// drain ran as rods 30 cells long with gaps of 60 between them. Liquid in a
/// full pipe over a shaft (a roof over the liquid above within a few rows)
/// is pressed down the shaft instead, and liquid over a void inside a body
/// (resting liquid of its kind a few rows under the gap) falls in and closes
/// it. `get` reads a cell by its coordinates.
pub(crate) fn held_in_hole(
    ctx: &RuleCtx<'_>,
    c: Cell,
    x: i32,
    y: i32,
    mut get: impl FnMut(i32, i32) -> Cell,
) -> bool {
    let density = ctx.f(c).density;
    let rest =
        |t: Cell| t.material() == c.material() && !is_airborne(t) && !t.has(Cell::FLAG_IN_BODY);
    if !rest(get(x, y - 1)) {
        return false;
    }
    for k in 2..=HOLE_ROOF {
        let t = get(x, y - k);
        if rest(t) {
            continue;
        }
        if !unsupported(ctx, density, t) && ctx.f(t).state != State::Liquid {
            return false;
        }
        break;
    }
    for k in 1..=HOLE_VOID {
        let t = get(x, y + k);
        if rest(t) {
            return false;
        }
        if !unsupported(ctx, density, t) {
            break;
        }
    }
    // The gap along row `row` around column `x`: every cell up to a solid
    // edge on each side passes `inside`.
    let mut gap = |row: i32, inside: &dyn Fn(Cell, Cell) -> bool| {
        [-1, 1].into_iter().all(|s| {
            for k in 1..=HOLE_W {
                let t = get(x + s * k, row);
                if solid(ctx, t) {
                    return true;
                }
                if !inside(t, get(x + s * k, row + 1)) {
                    return false;
                }
            }
            false
        })
    };
    gap(y + 1, &|t, _| unsupported(ctx, density, t))
        || gap(y, &|t, under| rest(t) && unsupported(ctx, density, under))
}

/// True if resting liquid `c` at `(lx, ly)` with nothing under it is held
/// over a hole ([`held_in_hole`]). A void one cell deep under a column is
/// not a hole: the column's top cell drops into it.
#[inline]
fn held(nb: &Neighborhood<'_>, ctx: &RuleCtx<'_>, lx: i32, ly: i32, c: Cell) -> bool {
    let density = ctx.f(c).density;
    let below = nb.get(lx, ly + 1);
    if !is_airborne(below)
        && !unsupported(ctx, density, nb.get(lx, ly + 2))
        && nb.get(lx, ly - 1).material() == c.material()
    {
        return false;
    }
    held_in_hole(ctx, c, lx, ly, |x, y| nb.get(x, y))
}

/// True if `c` is liquid whose next move is through the air: it is falling or
/// flying, or rests on open space or on falling liquid or powder and is not
/// held over a hole in a floor. The fall sweep moves these.
#[inline]
pub(crate) fn in_air(nb: &Neighborhood<'_>, ctx: &RuleCtx<'_>, lx: i32, ly: i32, c: Cell) -> bool {
    if c.has(Cell::FLAG_IN_BODY) {
        return false;
    }
    // The air flag is set on nothing but liquid.
    if is_airborne(c) {
        return true;
    }
    if !ctx.lq(c).is_liquid() {
        return false;
    }
    unsupported(ctx, ctx.f(c).density, nb.get(lx, ly + 1)) && !held(nb, ctx, lx, ly, c)
}

/// Moving direction of liquid in the air: 1 right, -1 left, 0 still, resting
/// or not a liquid. The row sweeps update these front first.
#[inline]
pub(crate) fn mover_dir(ctx: &RuleCtx<'_>, c: Cell) -> i32 {
    if c.value < 16 || !is_airborne(c) || ctx.lq(c).kind != KIND_VEL {
        return 0;
    }
    if c.has(Cell::FLAG_LIQ_LEFT) { -1 } else { 1 }
}

/// True if liquid in the air at `(lx, ly)` is about to fly into liquid of its
/// kind in the air that has not moved yet this tick: just ahead of it, below
/// it, or ahead and below. In the fall sweep's first round such a cell waits
/// for a second round, by which time the liquid in its way has moved.
///
/// The sweep runs a row of chunks in two halves, so liquid flying across a
/// chunk border meets the liquid ahead, in the chunk that runs later, still
/// unmoved on half of the ticks, and a dense jet or sheet came out past every
/// border thinned into bars. Within a chunk the liquid ahead and below moves
/// first, and the chunks below ran before, so until some cell of the job has
/// waited (`any`) only a cell flying across the border can meet unmoved
/// liquid. After that the cells behind and above a cell that waited wait in
/// turn and keep their order.
#[inline]
pub(crate) fn waits(
    nb: &Neighborhood<'_>,
    ctx: &RuleCtx<'_>,
    lx: i32,
    ly: i32,
    c: Cell,
    any: bool,
) -> bool {
    if !is_airborne(c) {
        return false;
    }
    let s = mover_dir(ctx, c);
    let across = s != 0 && !(0..CHUNK_SIZE).contains(&(lx + s));
    if !any && !across {
        return false;
    }
    let parity = (ctx.tick & 1) as u32;
    let unmoved = |t: Cell| {
        t.material() == c.material()
            && is_airborne(t)
            && !t.has(Cell::FLAG_IN_BODY)
            && t.parity() != parity
    };
    unmoved(nb.get(lx, ly + 1))
        || (s != 0 && (unmoved(nb.get(lx + s, ly)) || unmoved(nb.get(lx + s, ly + 1))))
}

/// What the liquid rule did with a cell.
pub(crate) enum Outcome {
    /// The rule wrote the cell and its dirty marks.
    Done,
    /// The cell stays where it is; the caller writes it. `keep` asks for a visit on the next tick.
    Stay {
        /// Visit the cell again on the next tick.
        keep: bool,
    },
}

#[inline]
fn back(d: Cell, tick: u64) -> Cell {
    if d.is_air() { d } else { d.with_parity(tick) }
}

/// The cell a moving cell is in, and what the grid holds there right now. The
/// moving cell itself is written once, at the end: each step moves the
/// displaced cell back into the previous position, and skips the write when it
/// would not change anything (air through air), so a long fall costs two writes.
struct Path {
    x: i32,
    y: i32,
    stored: Cell,
}

impl Path {
    #[inline]
    fn step(&mut self, nb: &Neighborhood<'_>, tick: u64, tx: i32, ty: i32) {
        let d = nb.get(tx, ty);
        let b = back(d, tick);
        if b != self.stored {
            nb.set(self.x, self.y, b);
        }
        self.stored = d;
        self.x = tx;
        self.y = ty;
    }
}

/// One liquid cell's update.
struct Liq<'n, 'a, 'c, 'm> {
    nb: &'n Neighborhood<'a>,
    ctx: &'c RuleCtx<'m>,
    lx: i32,
    ly: i32,
    wx: i32,
    wy: i32,
    mat: MaterialId,
    density: i16,
    q: LiqFast,
}

/// Runs the liquid rule on a cell. `start` is the cell as it was before this
/// tick's update; `cell` is the working copy with heat and parity already
/// applied. Heat, reactions and transitions run before this.
#[expect(clippy::too_many_arguments, reason = "hot path; a struct would add copies")]
pub(crate) fn update(
    nb: &Neighborhood<'_>,
    ctx: &RuleCtx<'_>,
    out: &mut JobOut,
    lx: i32,
    ly: i32,
    wx: i32,
    wy: i32,
    start: Cell,
    cell: &mut Cell,
) -> Outcome {
    let q = ctx.lq(start);
    let l =
        Liq { nb, ctx, lx, ly, wx, wy, mat: start.material(), density: ctx.f(start).density, q };
    let below = nb.get(lx, ly + 1);
    if is_airborne(start)
        || ((l.enterable(below) || l.falling_under(below)) && !held(nb, ctx, lx, ly, start))
    {
        l.in_air(out, start, cell, below)
    } else {
        l.resting(out, start, cell, below)
    }
}

impl Liq<'_, '_, '_, '_> {
    #[inline]
    fn rand(&self, salt: u32) -> u32 {
        self.ctx.rand(self.wx, self.wy, salt)
    }

    #[inline]
    fn coin(&self, salt: u32) -> i32 {
        if self.rand(salt) & 1 == 0 { 1 } else { -1 }
    }

    /// Air, gas, fire, or a lighter free fluid: space this cell can move into.
    #[inline]
    fn enterable(&self, t: Cell) -> bool {
        if t.is_air() {
            return true;
        }
        if t.has(Cell::FLAG_IN_BODY) {
            return false;
        }
        let f = self.ctx.f(t);
        f.state.is_fluid() && f.density < self.density
    }

    /// Air, gas or fire: open space over a free surface.
    #[inline]
    fn open(&self, t: Cell) -> bool {
        t.is_air()
            || (!t.has(Cell::FLAG_IN_BODY)
                && matches!(self.ctx.f(t).state, State::Gas | State::Fire))
    }

    #[inline]
    fn liquid(&self, t: Cell) -> bool {
        !t.has(Cell::FLAG_IN_BODY) && self.ctx.f(t).state == State::Liquid
    }

    #[inline]
    fn same(&self, t: Cell) -> bool {
        t.material() == self.mat && !t.has(Cell::FLAG_IN_BODY)
    }

    /// The same liquid, resting.
    #[inline]
    fn same_rest(&self, t: Cell) -> bool {
        self.same(t) && !is_airborne(t)
    }

    /// True if no liquid of this kind lies beside the cell or on top of it.
    #[inline]
    fn lone(&self) -> bool {
        let (nb, lx, ly) = (self.nb, self.lx, self.ly);
        [(-1, 0), (1, 0), (0, -1)].into_iter().all(|(dx, dy)| !self.same(nb.get(lx + dx, ly + dy)))
    }

    /// Head above `(x, y)`, capped.
    #[inline]
    fn head(&self, x: i32, y: i32, cap: u32) -> u32 {
        let liq = self.ctx.liq;
        self.nb.head(x, y, cap, |c| is_head(liq, c))
    }

    /// True if resting liquid beside this cell stands on a floor: this cell is
    /// over a hole in the floor of a body that is held up. Only a floor that
    /// cannot move this tick counts. Liquid beside that stands on more liquid
    /// may be about to fall itself (its chunk may simply come later in the
    /// tick), so it holds nothing up.
    fn held_beside(&self) -> bool {
        let (nb, lx, ly) = (self.nb, self.lx, self.ly);
        [-1, 1].into_iter().any(|s| {
            let c = nb.get(lx + s, ly);
            self.same_rest(c) && {
                let b = nb.get(lx + s, ly + 1);
                !self.enterable(b)
                    && !self.falling_under(b)
                    && (b.has(Cell::FLAG_IN_BODY) || !self.ctx.f(b).state.is_fluid())
            }
        })
    }

    /// Writes the cell's final code: resting with a flow speed, or in the air.
    /// Liquid in the air rounds its speeds with random bits shared by every
    /// cell on the tick, so cells moving together stay together.
    #[inline]
    fn enc_air(&self, c: Cell, vx: i32, vy: i32) -> Cell {
        if self.q.kind == KIND_VEL { airborne(c, vx, vy, self.ctx.air_round) } else { c }
    }

    #[inline]
    fn enc_rest(&self, c: Cell, u: i32) -> Cell {
        if self.q.kind == KIND_VEL { with_flow_speed(c, u, self.rand(SALT_ENCODE)) } else { c }
    }

    /// The fall speed of `t` if it is falling liquid or powder, which a cell on
    /// top of it rides instead of resting on.
    #[inline]
    fn fall_speed(&self, t: Cell) -> Option<i32> {
        if t.has(Cell::FLAG_IN_BODY) {
            return None;
        }
        match self.ctx.f(t).state {
            State::Liquid if is_airborne(t) => Some(air_velocity(t).1),
            State::Powder if t.value > 0 && !self.ctx.f(t).has_lifetime() => {
                Some(i32::from(t.value))
            }
            _ => None,
        }
    }

    /// True if `t` is falling, so a resting cell on top of it falls with it.
    #[inline]
    fn falling_under(&self, t: Cell) -> bool {
        self.fall_speed(t).is_some()
    }

    /// The cell lands on `t`, below `(cx, cy)`. Returns the impact speed, or
    /// `None` when `t` is falling liquid or powder that the cell rides instead,
    /// so a falling blob stays together.
    ///
    /// A cell that comes down on slower liquid of its kind in the air shares
    /// its fall speed with it, as the two would by viscosity, so the lanes of
    /// a jet or a sheet that met keep one speed and one path. Taking the
    /// slower speed, a cell of a jet lost the speed it had gained at the
    /// places already behind it and fell on a flatter path from then on, and
    /// the jet split into two or three trails.
    #[inline]
    fn vertical_block(&self, t: Cell, cx: i32, cy: i32, vx: &mut i32, vy: &mut i32) -> Option<i32> {
        if let Some(tvy) = self.fall_speed(t) {
            if tvy < *vy && self.same(t) && is_airborne(t) && self.q.kind == KIND_VEL {
                let mean = i32::midpoint(*vy, tvy);
                self.nb.set(cx, cy + 1, self.enc_air(t, air_velocity(t).0, mean));
                *vy = mean;
            }
            return None;
        }
        let impact = *vy;
        if t.has(Cell::FLAG_IN_BODY)
            || !matches!(self.ctx.f(t).state, State::Liquid | State::Powder)
        {
            // On a solid the fall turns into sideways speed, so a stream
            // hitting a floor spreads out. On liquid or powder it soaks in.
            let e2 = i32::from(self.q.e2_solid);
            let s = isqrt_i(*vx * *vx + *vy * *vy * e2 / 256);
            let dir = if *vx != 0 {
                vx.signum()
            } else {
                let l = self.enterable(self.nb.get(cx - 1, cy));
                let r = self.enterable(self.nb.get(cx + 1, cy));
                match (l, r) {
                    (true, false) => -1,
                    (false, true) => 1,
                    (true, true) => self.coin(SALT_IMPACT),
                    (false, false) => 0,
                }
            };
            *vx = dir * s;
        }
        // On liquid the column below takes the cell's sideways speed and the
        // fall is absorbed.
        *vy = 0;
        Some(impact)
    }

    /// Liquid in the air, or resting liquid with open space under it.
    fn in_air(&self, out: &mut JobOut, start: Cell, cell: &mut Cell, below: Cell) -> Outcome {
        let (nb, lx, ly, tick) = (self.nb, self.lx, self.ly, self.ctx.tick);
        let was_air = is_airborne(start);
        if was_air
            && !self.enterable(below)
            && dither(air_velocity(start).0.abs(), xphase(tick)) == 0
        {
            // Already on something and not moving sideways this tick: it
            // lands here, or rides a falling cell. A cell falling straight
            // down that rides gains the tick's gravity like the cell under
            // it, so a falling column speeds up as one; without the gain, the
            // cells of a stack that had stopped moving sideways sank at the
            // speed they had, a cell at a time from the bottom, and stood at
            // the lip of a weir as a wall of falling cells for many ticks. A
            // cell moving sideways gains by its path, and takes the path
            // below, which rides or lands as it goes.
            let (mut vx, mut vy) = air_velocity(start);
            let old = vy;
            *cell = match self.vertical_block(below, lx, ly, &mut vx, &mut vy) {
                // A stream gains by the rows it crosses, and this cell crossed none.
                None if is_stream(start) => as_stream(self.enc_air(*cell, vx, vy)),
                None if vx == 0 && self.q.kind == KIND_VEL => {
                    self.enc_air(*cell, vx, (vy + gravity_gain(tick) as i32).min(VY_MAX))
                }
                None if vy == old => *cell,
                None => self.enc_air(*cell, vx, vy),
                Some(_) => self.land(*cell, lx, ly, below, vx),
            };
            return Outcome::Stay { keep: true };
        }
        // A one-cell void under a resting column: the column's top cell drops
        // into it, so the void surfaces at once instead of foaming the body.
        if !was_air
            && self.open(below)
            && self.same_rest(nb.get(lx, ly - 1))
            && !self.enterable(nb.get(lx, ly + 2))
        {
            let mut k = 1;
            while k < BUBBLE_SCAN && self.same_rest(nb.get(lx, ly - 1 - k)) {
                k += 1;
            }
            let ty = ly - k;
            let top = nb.get(lx, ty);
            nb.set(lx, ly + 1, at_rest(top).with_parity(tick));
            nb.set(lx, ty, back(below, tick));
            // The column's speed moves to its new top cell.
            if k > 1 {
                nb.set(lx, ly, *cell);
                let nt = nb.get(lx, ty + 1);
                nb.set(lx, ty + 1, self.enc_rest(nt, flow_speed(top)));
            } else {
                nb.set(lx, ly, self.enc_rest(*cell, flow_speed(top)));
            }
            nb.mark_rect(lx - 1, ty - 1, lx + 1, ly + 2);
            return Outcome::Done;
        }
        let above = nb.get(lx, ly - 1);
        let (mut vx, mut vy) =
            if was_air { air_velocity(start) } else { (flow_speed(start) / 256, 0) };
        // Liquid flying sideways gains speed by its steps along its path:
        // see `fall_gain`. Everything else gains it by the tick.
        let path = was_air && vx != 0 && vy > 0 && self.q.kind == KIND_VEL;
        // A stream falling straight down out of a hole gains speed by the
        // rows it crosses, so its cells speed up at the same places one
        // after another and the stream thins out smoothly as it falls (see
        // [`is_stream`]). Gaining by the tick with bits shared by every
        // cell, cells of every age in a stream moved up a code together,
        // pulled away from those that had just left, and the stream fell as
        // trains of cells with growing gaps between them, a dashed line. A
        // falling mass gains by the tick, so it falls as one.
        // A lone cell dropping from rest, with none of its kind beside or
        // above it, is a drip off a tap or a ledge: it leaves at the speed of
        // a cell that has fallen one row and falls as a stream, so drip after
        // drip falls along one profile and the drips stay evenly spaced.
        // Gaining by the tick, drips that left a tap a few ticks apart met at
        // different speeds, ran into one another, and fell as clumps with
        // gaps of 40 to 60 rows between them.
        let drip = !was_air && vx == 0 && self.q.kind == KIND_VEL && self.lone();
        if drip {
            vy = vy.max(DRIP_FALL);
        }
        let place = (was_air && is_stream(start) && vy > 0) || drip;
        if !path && !place {
            vy = (vy + gravity_gain(tick) as i32).min(VY_MAX);
        }
        // Pressured fall: liquid over a hole in the floor of a body that is
        // held up beside it leaves at the Torricelli speed for its head. A
        // body with nothing under it falls as one, at the speed of gravity.
        if !was_air && self.held_beside() {
            if self.liquid(above) && !is_airborne(above) {
                vy = vy.max(HOLE_FALL);
            }
            let h = self.head(lx, ly, HEAD_MAX + 64) as i32;
            if h >= 1 {
                vy = vy.max(isqrt_i(E1 * (h + 1)).min(VY_MAX));
            }
        }
        if self.q.kind != KIND_VEL {
            vy = vy.max(16);
        }
        vx = vx.clamp(-VX_MAX, VX_MAX);
        let (p, impact) = if path {
            let Some(w) = self.walk_path(start, &mut vx, &mut vy) else {
                *cell = self.enc_air(*cell, vx, vy);
                return Outcome::Stay { keep: true };
            };
            w
        } else {
            let (nx, ny) = (dither(vx.abs(), xphase(tick)), dither(vy, tick));
            if nx == 0 && ny == 0 {
                let c = self.enc_air(*cell, vx, vy);
                *cell = if place && vx == 0 { as_stream(c) } else { c };
                return Outcome::Stay { keep: true };
            }
            let (p, impact) = self.walk_line(start, nx, ny, &mut vx, &mut vy);
            if place && impact.is_none() {
                for k in 1..=p.y - ly {
                    vy = fall_row(self.ctx, vy, self.wy + k);
                }
            }
            (p, impact)
        };
        self.finish(out, cell, p, impact, vx, vy, place)
    }

    /// Moves liquid in the air that is not flying sideways along the line
    /// toward `(nx, ny)` cells away, one cell at a time, until something
    /// stops it. Returns where it got to and the speed it hit something
    /// below at, if it did.
    fn walk_line(
        &self,
        start: Cell,
        nx: i32,
        ny: i32,
        vx: &mut i32,
        vy: &mut i32,
    ) -> (Path, Option<i32>) {
        let (nb, lx, ly, tick) = (self.nb, self.lx, self.ly, self.ctx.tick);
        let mut p = Path { x: lx, y: ly, stored: start };
        let mut impact = None;
        if nx == 0 {
            // Straight down: skip over plain air at once.
            let run = nb.run_below(lx, ly, ny, Cell::AIR);
            if run > 0 {
                nb.set(lx, ly, Cell::AIR);
                p.stored = Cell::AIR;
                p.y += run;
            }
            for _ in run..ny {
                let t = nb.get(lx, p.y + 1);
                if !self.enterable(t) {
                    impact = self.vertical_block(t, p.x, p.y, vx, vy);
                    break;
                }
                p.step(nb, tick, lx, p.y + 1);
            }
        } else {
            // A line toward (lx + nx, ly + ny), one cell at a time.
            let s = vx.signum();
            let n = nx.max(ny);
            let (mut hb, mut vb) = (false, false);
            for i in 1..=n {
                if !hb && nx * i / n > (p.x - lx).abs() {
                    let t = nb.get(p.x + s, p.y);
                    if self.enterable(t) {
                        p.step(nb, tick, p.x + s, p.y);
                    } else {
                        hb = true;
                        self.held_ahead(t, p.x + s, p.y, vx);
                    }
                }
                if !vb && ny * i / n > p.y - ly {
                    let t = nb.get(p.x, p.y + 1);
                    if self.enterable(t) {
                        p.step(nb, tick, p.x, p.y + 1);
                    } else {
                        vb = true;
                        impact = self.vertical_block(t, p.x, p.y, vx, vy);
                    }
                }
            }
        }
        (p, impact)
    }

    /// Something that is not room for it lies ahead of liquid moving
    /// sideways, at `(tx, ty)`: it holds the cell up for the tick. Faster
    /// than liquid in the air ahead moving the same way, it pushes it on: the
    /// two share their speed, so a stream evens out its speed instead of
    /// piling up. Taking the slower speed, a lane of a jet that met one slow
    /// cell went on at its speed for good and fell away as a second trail.
    /// Behind liquid that does not move its way, it keeps its speed and goes
    /// on once that has fallen out of the way; stopped by that too, a cell
    /// behind one that had stopped stopped as well, the sheet over a weir
    /// jammed at the lip, and the jammed cells fell straight down as a comb of
    /// bars. Resting liquid stops it, a wall bounces it.
    ///
    /// Returns true if the cell waits behind liquid of its kind in the air
    /// moving its way, on its path; false if it is free to go on falling.
    fn held_ahead(&self, t: Cell, tx: i32, ty: i32, vx: &mut i32) -> bool {
        if !self.liquid(t) {
            *vx = -*vx * WALL_BOUNCE / 256;
            return false;
        }
        if !is_airborne(t) {
            *vx = 0;
            return false;
        }
        let (ox, oy) = air_velocity(t);
        if ox * *vx <= 0 {
            return false;
        }
        if ox.abs() < vx.abs() {
            *vx = i32::midpoint(*vx, ox);
            if self.same(t) && self.q.kind == KIND_VEL {
                self.nb.set(tx, ty, self.enc_air(t, *vx, oy));
            }
        }
        true
    }

    /// Walks liquid flying sideways along its path for the tick, and gains
    /// gravity's speed by the places it crosses: see [`fall_gain`]. Returns
    /// where it got to and the speed it hit something below at, if it did,
    /// or `None` if it does not move this tick.
    ///
    /// Where it moves more sideways than down it takes the tick's dither of
    /// columns, like every moving cell, and at each column it crosses it
    /// steps down where the line of its path through whole cells does, then
    /// gains for that column; on a steep path it takes the tick's dither of
    /// rows, and at each row it steps aside where the line does. The steps
    /// and the gains are worked out one place at a time, with the speed the
    /// cell has there, so the cells of a stream take one path through the
    /// same cells however the ticks split their moves, and a cell held up on
    /// its way waits on its path, keeping the gains of the places it crossed.
    /// Worked out over a tick's whole move with the speed it started the
    /// tick with, and with no gain at all for a cell held up on its way, two
    /// cells that crossed one column on different ticks, or one of them
    /// held up for a tick, came out a row or a code apart, and the lanes of
    /// a jet drifted apart into two or three trails.
    fn walk_path(&self, start: Cell, vx: &mut i32, vy: &mut i32) -> Option<(Path, Option<i32>)> {
        let (nb, tick) = (self.nb, self.ctx.tick);
        let s = vx.signum();
        let ax = i64::from(vx.abs());
        let mut v = *vy;
        let mut p = Path { x: self.lx, y: self.ly, stored: start };
        let mut impact = None;
        // At a cell a tick sideways or faster it crosses a column on every
        // tick, so it takes the tick's columns and the rows each column's
        // line gives, however steep its path. A slice of a sheet or a jet,
        // cells poured into one column on one tick, then moves as one.
        // Keyed by rows on a steep path, each cell of such a slice stepped
        // aside at its own rows, the slice sheared across the path of the
        // slice ahead, and a sheet off a ledge jammed into a block of
        // liquid standing in the air.
        if ax >= i64::from(v) || ax >= 16 {
            let nx = dither(vx.abs(), xphase(tick));
            if nx == 0 {
                return None;
            }
            let x0 = i64::from(if s > 0 { self.wx } else { -self.wx });
            'cols: for k in 1..=i64::from(nx) {
                let t = nb.get(p.x + s, p.y);
                if !self.enterable(t) {
                    self.held_ahead(t, p.x + s, p.y, vx);
                    impact = self.drop_down(&mut p, vx, &mut v);
                    break;
                }
                p.step(nb, tick, p.x + s, p.y);
                let dy;
                (dy, v) = path_column(self.ctx, v, ax as i32, x0 + k);
                for _ in 0..dy {
                    let t = nb.get(p.x, p.y + 1);
                    if !self.enterable(t) {
                        impact = self.vertical_block(t, p.x, p.y, vx, &mut v);
                        break 'cols;
                    }
                    p.step(nb, tick, p.x, p.y + 1);
                }
            }
        } else {
            let ny = dither(v, tick);
            if ny == 0 {
                return None;
            }
            let y0 = i64::from(self.wy);
            let mut key = i64::from(if s > 0 { self.wx } else { -self.wx });
            'rows: for k in 1..=i64::from(ny) {
                let t = nb.get(p.x, p.y + 1);
                if !self.enterable(t) {
                    impact = self.vertical_block(t, p.x, p.y, vx, &mut v);
                    break;
                }
                p.step(nb, tick, p.x, p.y + 1);
                let v64 = i64::from(v.max(1));
                let dx = ((y0 + k) * ax).div_euclid(v64) - ((y0 + k - 1) * ax).div_euclid(v64);
                for _ in 0..dx {
                    let t = nb.get(p.x + s, p.y);
                    if !self.enterable(t) {
                        self.held_ahead(t, p.x + s, p.y, vx);
                        impact = self.drop_down(&mut p, vx, &mut v);
                        break 'rows;
                    }
                    p.step(nb, tick, p.x + s, p.y);
                    key += 1;
                    v = fall_gain(self.ctx, v, key as i32, 4096 / ax);
                }
            }
        }
        *vy = v;
        Some((p, impact))
    }

    /// Liquid flying sideways that something other than liquid of its
    /// stream holds up goes on falling straight down for the rest of the
    /// tick, as far as its fall speed takes it, so two cells flying at each
    /// other do not hold each other up in the air for good. Returns the speed
    /// it hit something below at, if it did.
    fn drop_down(&self, p: &mut Path, vx: &mut i32, v: &mut i32) -> Option<i32> {
        let (nb, tick) = (self.nb, self.ctx.tick);
        let fall = dither(*v, tick) - (p.y - self.ly);
        for _ in 0..fall {
            let t = nb.get(p.x, p.y + 1);
            if !self.enterable(t) {
                return self.vertical_block(t, p.x, p.y, vx, v);
            }
            p.step(nb, tick, p.x, p.y + 1);
        }
        None
    }

    /// Writes liquid in the air where its move took it: thrown back up as a
    /// splash after a hard landing, at rest where it landed, or in the air
    /// with its new speed.
    #[expect(clippy::too_many_arguments, reason = "the state of one move")]
    fn finish(
        &self,
        out: &mut JobOut,
        cell: &mut Cell,
        p: Path,
        impact: Option<i32>,
        vx: i32,
        vy: i32,
        stream: bool,
    ) -> Outcome {
        let (nb, lx, ly) = (self.nb, self.lx, self.ly);
        let moved = p.x != lx || p.y != ly;
        if let Some(v) = impact
            && self.q.can_splash
            && v >= SPLASH_MIN
            && out.splashes < MAX_SPLASHES
            && nb.get(p.x, p.y - 1).is_air()
            && self.rand(SALT_SPLASH).trailing_zeros() >= 3
        {
            // A hard landing throws the cell back up as a free particle.
            let s = if vx != 0 { vx.signum() } else { self.coin(SALT_IMPACT) };
            let origin = nb.pos().origin();
            out.events.push(Event::Splash {
                pos: CellPos::new(origin.x + p.x, origin.y + p.y),
                cell: at_rest(*cell),
                vx: Fx(s * v * 1024),
                vy: Fx(-v * 1024),
            });
            out.splashes += 1;
            if !moved {
                nb.set(lx, ly, Cell::AIR);
            }
            nb.mark_rect(lx.min(p.x) - 1, ly.min(p.y) - 1, lx.max(p.x) + 1, ly.max(p.y) + 1);
            return Outcome::Done;
        }
        // After landing it may have slid on along the row and over an edge.
        let on = nb.get(p.x, p.y + 1);
        let nc = if impact.is_some() && !self.enterable(on) {
            self.land(*cell, p.x, p.y, on, vx)
        } else if stream && vx == 0 {
            as_stream(self.enc_air(*cell, vx, vy))
        } else {
            self.enc_air(*cell, vx, vy)
        };
        if !moved {
            *cell = nc;
            return Outcome::Stay { keep: true };
        }
        nb.set(p.x, p.y, nc);
        nb.mark_rect(lx.min(p.x) - 1, ly.min(p.y) - 1, lx.max(p.x) + 1, ly.max(p.y) + 1);
        Outcome::Done
    }

    /// The cell comes to rest at `(x, y)` on top of `on`, with sideways speed
    /// `vx` (1/16 cells per tick). Landing on a column of the same liquid it
    /// becomes the column's top cell and shares its speed with the column, in
    /// proportion to depth; the cell below stops being the top.
    fn land(&self, cell: Cell, x: i32, y: i32, on: Cell, vx: i32) -> Cell {
        let nb = self.nb;
        let mut u = vx * 256;
        if self.same_rest(on) {
            let mut d = 1;
            while d < BLEND_DEPTH && self.same_rest(nb.get(x, y + 1 + d)) {
                d += 1;
            }
            u = (flow_speed(on) * d + u) / (d + 1);
            if on.value != 0 || on.has(Cell::FLAG_LIQ_LEFT) {
                nb.set(x, y + 1, at_rest(on));
            }
        }
        self.enc_rest(at_rest(cell), u)
    }

    /// Resting liquid. It moves with its column in the flow phase; here it only
    /// jets out of a hole it is pressed against, and a column's top cell offers
    /// itself to the flow phase and the pressure pass.
    fn resting(&self, out: &mut JobOut, start: Cell, cell: &mut Cell, below: Cell) -> Outcome {
        let (nb, lx, ly) = (self.nb, self.lx, self.ly);
        let above = nb.get(lx, ly - 1);
        let top = !self.same_rest(above);
        if !top && self.liquid(above) {
            let pref = self.coin(SALT_SIDE);
            for s in [pref, -pref] {
                if self.enterable(nb.get(lx + s, ly)) && self.roofed(lx + s) {
                    return self.jet(start, cell, s);
                }
            }
        }
        if top {
            out.tops.push((self.wy, self.wx));
            if self.liquid(below) && start.value == 0 && !start.has(Cell::FLAG_LIQ_LEFT) {
                out.pressure.push((self.wy, self.wx));
            }
        }
        Outcome::Stay { keep: false }
    }

    /// True if the open cell beside this one at column `x` is under a roof: a
    /// hole in a wall rather than open space beside a free face, which the flow
    /// phase handles.
    fn roofed(&self, x: i32) -> bool {
        for k in 1..=ROOF_SCAN {
            let c = self.nb.get(x, self.ly - k);
            if self.enterable(c) {
                continue;
            }
            return !self.liquid(c);
        }
        false
    }

    /// A pressed cell jets sideways through a hole at the Torricelli speed for
    /// its head, and flies on once it is past the floor's edge.
    fn jet(&self, start: Cell, cell: &mut Cell, s: i32) -> Outcome {
        let (nb, lx, ly, tick) = (self.nb, self.lx, self.ly, self.ctx.tick);
        let h = self.head(lx, ly, HEAD_MAX + 64) as i32;
        let t = isqrt_i(E1 * h).clamp(16, VX_MAX);
        let n = dither(t, xphase(tick)).clamp(1, MAX_WALK);
        let mut p = Path { x: lx, y: ly, stored: start };
        let mut flying = false;
        for _ in 0..n {
            if !self.enterable(nb.get(p.x + s, p.y)) {
                break;
            }
            p.step(nb, tick, p.x + s, p.y);
            if self.enterable(nb.get(p.x, p.y + 1)) {
                flying = true;
                break;
            }
        }
        if p.x == lx {
            *cell = self.enc_rest(*cell, s * t * 256);
            return Outcome::Stay { keep: true };
        }
        let nc =
            if flying { self.enc_air(*cell, s * t, 1) } else { self.enc_rest(*cell, s * t * 256) };
        nb.set(p.x, p.y, nc);
        nb.mark_rect(lx.min(p.x) - 1, ly - 1, lx.max(p.x) + 1, ly + 1);
        Outcome::Done
    }
}

/// Ticks of travel over which a stream of liquid in the air is drawn joined
/// up: a stream that drips at least this often reads as one line.
pub const STREAK_TICKS: i32 = 16;
/// Longest gap, in cells, that a stream is drawn across.
pub const STREAK_MAX: i32 = 48;
/// Gaps up to this many cells are joined behind any cell in the air; longer
/// ones only along a stream that leads back to its source.
const SHORT_JOIN: i32 = 6;

// Whether a cell's chain of joins leads back to a source, in `drawn`.
const UNKNOWN: u8 = 0;
const BUSY: u8 = 1;
const YES: u8 = 2;
const NO: u8 = 3;

/// What a cell of a stream in the air is joined to behind it.
#[derive(Clone, Copy)]
enum Link {
    /// Liquid that is not in the air, or the edge of the window.
    Source,
    /// Another cell in the air, by index.
    Cell(usize),
}

/// The cells as they are drawn: liquid in the air with the path it swept,
/// and small tears in streams and sheets closed. `cells` is a `w` by `h`
/// window of the world, row by row; `liquid` says whether a cell is a free
/// liquid. The renderer draws this, and the scene check `drawn` counts it.
///
/// Each cell of a stream is joined to the liquid of its kind behind it on
/// its path, the arc gravity bent its velocity along, as a long exposure
/// blurs a stream, where that liquid lies within [`STREAK_TICKS`] ticks of
/// travel. A jet or a thin falling stream is whole cells spaced as far apart
/// as they move in a tick, or further once the stream thins below a cell a
/// tick, and a still frame showed it as a dotted line; a trickle off a ledge
/// that left a cell every few ticks was drawn as drops. The cells of a
/// stream follow one path, so the liquid behind a cell is where its own arc
/// back says; traced back along a straight line, a long gap missed it.
/// Nothing is drawn behind the last cell of a stream, so a falling mass
/// keeps its shape and a lone drop stays a drop. Then air between two cells
/// of one liquid, one of them in the air, is drawn as that liquid where the
/// two lie along the way the liquid in the air moves (a row, a column or a
/// diagonal, whichever is nearest): one cell of it, or two where both are in
/// the air, so a stream the grid tore open reads whole. Closed along every
/// row, column and diagonal, a scatter of drops at the foot of a jet was
/// drawn as a lattice.
pub fn drawn(
    cells: &[Cell],
    w: usize,
    h: usize,
    liquid: impl Fn(Cell) -> bool,
    firm: impl Fn(Cell) -> bool,
) -> Vec<Cell> {
    let mut out = cells.to_vec();
    let at = |x: i32, y: i32| {
        (x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h)
            .then(|| cells[y as usize * w + x as usize])
    };
    // Rounds a / b to the nearest whole number, for b > 0.
    let round = |a: i64, b: i64| (2 * a + b).div_euclid(2 * b) as i32;
    // Each cell's join: the cells of travel to fill behind it and what it
    // joins to, so a join is drawn only when its chain leads back to a source.
    let mut links: Vec<(usize, i32, Link)> = Vec::new();
    for (i, &c) in cells.iter().enumerate() {
        if !is_airborne(c) || c.has(Cell::FLAG_IN_BODY) {
            continue;
        }
        let (vx, vy) = air_velocity(c);
        let m = vx.abs().max(vy);
        let (x0, y0) = ((i % w) as i32, (i / w) as i32);
        // A cell close to a ledge or a pool has just left it: it heads a
        // stream. A drop out in the open (rain condensed from steam, spray
        // from a blast) heads nothing.
        let near = || {
            (-3..=3).any(|dy| {
                (-3..=3).any(|dx| {
                    at(x0 + dx, y0 + dy).is_some_and(|t| firm(t) || (liquid(t) && !is_airborne(t)))
                })
            })
        };
        if m < 4 {
            if near() {
                links.push((i, 0, Link::Source));
            }
            continue;
        }
        let reach = ((m * STREAK_TICKS + 15) / 16).clamp(2, STREAK_MAX);
        // Where the cell was `k` cells of travel along its faster axis ago:
        // 16 k / m ticks back, when it fell slower by gravity's gain.
        let (mx, vx64, vy64) = (i64::from(m), i64::from(vx), i64::from(vy));
        let back = |k: i32| {
            let k = i64::from(k);
            (
                x0 - round(vx64 * k, mx),
                y0 - round(2 * vy64 * k * mx - 7 * k * k, 2 * mx * mx).max(0),
            )
        };
        // Liquid in the air a cell to either side of the path joins too, so
        // a stream that wanders by a cell is joined.
        let (sx, sy) = if vx.abs() >= vy { (0, 1) } else { (1, 0) };
        let joins = |t: Cell| t.material() == c.material() && !t.has(Cell::FLAG_IN_BODY);
        let side = |t: Cell| joins(t) && is_airborne(t);
        let mut found = 0;
        let mut link = Link::Source;
        let mut edge = false;
        let mut last = (x0, y0);
        let idx = |x: i32, y: i32| y as usize * w + x as usize;
        for k in 1..=reach {
            let (x, y) = back(k);
            if (x, y) == last {
                continue;
            }
            last = (x, y);
            let Some(t) = at(x, y) else {
                // The path leaves the window: what lies beyond is unknown, so
                // take it for the stream it looks like, though nothing is
                // drawn behind the cell.
                edge = true;
                break;
            };
            if joins(t) {
                found = k;
                if is_airborne(t) {
                    link = Link::Cell(idx(x, y));
                }
                break;
            }
            if !t.is_air() {
                // A cell that has just left a ledge or a wall heads a stream.
                edge = k <= SHORT_JOIN;
                break;
            }
            for (px, py) in [(x + sx, y + sy), (x - sx, y - sy)] {
                if at(px, py).is_some_and(side) {
                    found = k + 1;
                    link = Link::Cell(idx(px, py));
                    break;
                }
            }
            if found > 0 {
                break;
            }
        }
        if found > 0 || edge {
            links.push((i, found, link));
        } else if near() {
            links.push((i, 0, Link::Source));
        }
    }
    // A join is drawn when it is short, or when its chain of joins leads back
    // to liquid that is not in the air (the pool, hole or jet mouth the
    // stream leaves) or out of the window. Lone drops in the air, such as
    // rain condensed from steam or spray from a blast, are drawn as drops:
    // joined to one another they drew as long thin lines and branches.
    let mut link_of = vec![None; cells.len()];
    for &(i, _, l) in &links {
        link_of[i] = Some(l);
    }
    let mut anchored = vec![UNKNOWN; cells.len()];
    let mut stack = Vec::new();
    for &(i, _, _) in &links {
        let mut j = i;
        // Walk up the chain to a known answer, then mark the cells on the way.
        let answer = loop {
            match anchored[j] {
                YES => break true,
                NO | BUSY => break false,
                _ => {}
            }
            match link_of[j] {
                None => break false,
                Some(Link::Source) => break true,
                Some(Link::Cell(t)) => {
                    anchored[j] = BUSY;
                    stack.push(j);
                    j = t;
                }
            }
        };
        for j in stack.drain(..) {
            anchored[j] = if answer { YES } else { NO };
        }
    }
    for &(i, found, _) in &links {
        let ok = found <= SHORT_JOIN || anchored[i] == YES;
        if !ok {
            continue;
        }
        let c = cells[i];
        let (x0, y0) = ((i % w) as i32, (i / w) as i32);
        let (vx, vy) = air_velocity(c);
        let m = vx.abs().max(vy);
        let (mx, vx64, vy64) = (i64::from(m), i64::from(vx), i64::from(vy));
        for k in 1..found {
            let k64 = i64::from(k);
            let x = x0 - round(vx64 * k64, mx);
            let y = y0 - round(2 * vy64 * k64 * mx - 7 * k64 * k64, 2 * mx * mx).max(0);
            if at(x, y).is_some_and(Cell::is_air) {
                out[y as usize * w + x as usize] = c;
            }
        }
    }
    let joined = out.clone();
    let near = |x: i32, y: i32| {
        (x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h)
            .then(|| joined[y as usize * w + x as usize])
    };
    let wet = |t: Cell| !t.is_air() && !t.has(Cell::FLAG_IN_BODY) && liquid(t);
    // True if liquid in the air `t` moves along the axis `(ax, ay)`: within 45
    // degrees of it, either way.
    let along = |t: Cell, ax: i32, ay: i32| {
        let (vx, vy) = air_velocity(t);
        (vx * ay - vy * ax).abs() <= (vx * ax + vy * ay).abs()
    };
    for (i, o) in out.iter_mut().enumerate() {
        if !o.is_air() {
            continue;
        }
        let (x, y) = ((i % w) as i32, (i / w) as i32);
        'axes: for (ax, ay) in [(1, 0), (0, 1), (1, 1), (1, -1)] {
            for (b, f) in [(1, 1), (1, 2), (2, 1)] {
                let (Some(p), Some(q)) =
                    (near(x - ax * b, y - ay * b), near(x + ax * f, y + ay * f))
                else {
                    continue;
                };
                let both = is_airborne(p) && is_airborne(q);
                let moving =
                    (is_airborne(p) && along(p, ax, ay)) || (is_airborne(q) && along(q, ax, ay));
                if wet(p)
                    && wet(q)
                    && p.material() == q.material()
                    && moving
                    && (b + f == 2 || both)
                {
                    *o = if is_airborne(p) { p } else { q };
                    break 'axes;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use fe_core::mix64;

    fn water() -> (MaterialSet, MaterialId) {
        let mats = fe_materials::base_set();
        let id = mats.id("water").unwrap_or_default();
        (mats, id)
    }

    #[test]
    fn air_codes_encode_exactly() {
        let (_, w) = water();
        let c = Cell::new(w, 30, 0);
        for (i, &vx) in VX_TABLE.iter().enumerate() {
            for (j, &vy) in VY_TABLE.iter().enumerate() {
                for sign in [1, -1] {
                    let vx = i32::from(vx) * sign;
                    for r in [0, 0xffff_ffff, 0x1234_5678] {
                        let e = airborne(c, vx, i32::from(vy), r);
                        assert!(is_airborne(e));
                        assert_eq!(air_velocity(e), (vx, i32::from(vy)), "codes {i} {j}");
                        assert_eq!(e.material(), w);
                        assert_eq!(e.temp, 30);
                    }
                }
            }
        }
    }

    #[test]
    fn flow_codes_encode_exactly() {
        let (_, w) = water();
        let c = Cell::new(w, 30, 0);
        for m in 0..=255i32 {
            for sign in [1, -1] {
                for r in [0, 0xffff_ffff, 0x1234_5678] {
                    let e = with_flow_speed(c, sign * m * m, r);
                    assert!(!is_airborne(e));
                    assert_eq!(flow_speed(e), sign * m * m, "m {m}");
                }
            }
        }
    }

    /// Most a mean of 4096 stochastic roundings of `v` may miss it by: a unit,
    /// or five standard deviations of the rounding between the two codes
    /// around `v` (at most a 128th of their gap each).
    fn mean_tolerance(t: &[u8], v: i32) -> i64 {
        let i = t.iter().rposition(|&c| i32::from(c) <= v).unwrap_or(0);
        let gap = t.get(i + 1).map_or(1, |&c| i64::from(c) - i64::from(t[i]));
        (gap * 5 / 128).max(1)
    }

    #[test]
    fn speeds_between_codes_are_kept_on_average() {
        let (_, w) = water();
        let c = Cell::new(w, 30, 0);
        let n = 4096i64;
        for v in 0..=VX_MAX {
            let mut sum = 0i64;
            for i in 0..n {
                let r = mix64(i as u64 * 7 + v as u64) as u32;
                sum += i64::from(air_velocity(airborne(c, v, 0, r)).0);
            }
            let tol = mean_tolerance(&VX_TABLE, v);
            assert!((sum - i64::from(v) * n).abs() < n * tol, "vx {v}: sum {sum}");
        }
        for v in 0..=VY_MAX {
            let mut sum = 0i64;
            for i in 0..n {
                let r = mix64(i as u64 * 13 + v as u64) as u32;
                sum += i64::from(air_velocity(airborne(c, 0, v, r)).1);
            }
            let tol = mean_tolerance(&VY_TABLE, v);
            assert!((sum - i64::from(v) * n).abs() < n * tol, "vy {v}: sum {sum}");
        }
        for u in (0..FLOW_MAX).step_by(97) {
            let mut sum = 0i64;
            for i in 0..n {
                let r = mix64(i as u64 * 31 + u as u64) as u32;
                sum += i64::from(flow_speed(with_flow_speed(c, -u, r)));
            }
            // Within half a percent, or one code step near zero.
            let err = (sum + i64::from(u) * n).abs();
            assert!(err < n * (1 + i64::from(u) / 200), "u {u}: sum {sum}");
        }
    }

    #[test]
    fn out_of_range_speeds_clamp() {
        let (_, w) = water();
        let c = Cell::new(w, 30, 0);
        assert_eq!(air_velocity(airborne(c, 1000, 1000, 0)), (160, 255));
        assert_eq!(air_velocity(airborne(c, -1000, -5, 0)), (-160, 0));
        assert_eq!(flow_speed(with_flow_speed(c, -1 << 30, 0)), -FLOW_MAX);
    }

    #[test]
    fn a_resting_cell_is_a_new_cell() {
        let (mats, w) = water();
        let fresh = crate::new_cell(&mats, w, None, 0);
        assert_eq!(decode(fresh), (0, 0));
        assert_eq!(flow_speed(fresh), 0);
        assert_eq!(at_rest(airborne(fresh, -40, 100, 3)), fresh);
        assert_eq!(at_rest(with_flow_speed(fresh, -4000, 3)), fresh);
        assert_eq!(with_flow_speed(fresh, 0, 77), fresh);
        let s = settled_cell(fresh);
        assert!(is_settled(s) && !is_settled(fresh));
        assert_eq!(flow_speed(s), 0);
        assert_eq!(at_rest(s), fresh);
    }

    #[test]
    fn isqrt32_matches_the_reference() {
        for v in 0..(1u32 << 20) {
            assert_eq!(u64::from(isqrt32(v)), fe_core::isqrt(u64::from(v)), "{v}");
        }
        let mut s = 1u64;
        for _ in 0..100_000 {
            s = mix64(s);
            let v = s as u32;
            assert_eq!(u64::from(isqrt32(v)), fe_core::isqrt(u64::from(v)), "{v}");
        }
        assert_eq!(isqrt32(u32::MAX), 65535);
    }

    #[test]
    fn dither_adds_up_over_sixteen_ticks() {
        for v in [0, 1, 7, 16, 37, 160, 255] {
            for start in [0u64, 3, 1000] {
                let sum: i32 = (start..start + 16).map(|p| dither(v, p)).sum();
                assert_eq!(sum, v);
            }
        }
    }

    #[test]
    fn a_stream_is_drawn_whole_and_a_drop_alone() {
        let (_, w) = water();
        let (width, height) = (16usize, 24usize);
        let mut cells = vec![Cell::AIR; width * height];
        // Three cells a tick straight down, a cell every three rows.
        let fall = airborne(Cell::new(w, 30, 0), 0, 48, 0);
        for y in [4, 7, 10, 13, 16] {
            cells[y * width + 2] = fall;
        }
        // A lone drop, and a block falling as one.
        cells[14 * width + 7] = fall;
        for y in 18..21 {
            for x in 11..14 {
                cells[y * width + x] = fall;
            }
        }
        let out = drawn(&cells, width, height, |c| c.material() == w, |_| false);
        let wet = |x: usize, y: usize| out[y * width + x].material() == w;
        assert!((4..=16).all(|y| wet(2, y)), "the stream is one line");
        assert!((0..4).all(|y| !wet(2, y)), "nothing above its last cell");
        assert!((0..14).all(|y| !wet(7, y)), "a lone drop has no tail");
        assert!((0..18).all(|y| (11..14).all(|x| !wet(x, y))), "the block keeps its shape");
        assert_eq!(out.iter().filter(|c| c.material() == w).count(), 13 + 1 + 9);
    }

    #[test]
    fn lone_fast_drops_are_not_joined_into_lines() {
        let (_, w) = water();
        let (width, height) = (16usize, 200usize);
        let fast = airborne(Cell::new(w, 30, 0), 0, 255, 0);
        let mut cells = vec![Cell::AIR; width * height];
        // Rain condensed from steam: fast drops 20 rows apart in open sky.
        for y in [100, 120, 140, 160, 180] {
            cells[y * width + 5] = fast;
        }
        let count = |cells: &[Cell]| {
            drawn(cells, width, height, |c| c.material() == w, |_| false)
                .iter()
                .filter(|c| c.material() == w)
                .count()
        };
        assert_eq!(count(&cells), 5, "drops in the open stay drops");
        // The same cells falling from a pool above them are one stream.
        for x in 0..width {
            cells[80 * width + x] = Cell::new(w, 30, 0);
        }
        assert!(count(&cells) > 16 + 5 + 60, "a stream from a pool is drawn whole");
    }

    #[test]
    fn viscosity_sets_the_liquid_table() {
        let mats = fe_materials::base_set();
        let t = liquid_table(&mats);
        let q = |n: &str| t[mats.id(n).unwrap_or_default().index()];
        let w = q("water");
        assert!(w.can_splash && w.keeps_speed());
        let lava = q("lava");
        assert!(!lava.can_splash);
        for (thin, thick) in [("water", "oil"), ("oil", "molten_metal"), ("molten_metal", "lava")] {
            let (a, b) = (q(thin), q(thick));
            assert!(a.cf < b.cf && a.nu < b.nu && a.damp < b.damp, "{thin} against {thick}");
        }
        assert!(!q("sand").is_liquid() && !q("air").is_liquid());
    }
}
