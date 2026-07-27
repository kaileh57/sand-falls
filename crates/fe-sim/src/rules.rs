//! Per-cell update rules, run inside scheduler jobs.
//!
//! Every rule reads and writes through a [`Neighborhood`] and draws randomness
//! from [`RuleCtx::rand`], so its result depends only on the world state, the
//! seed and the tick.

use crate::liquid::{self, LiqFast, Outcome};
use crate::sched::Neighborhood;
use fe_core::{AMBIENT_TEMP, Cell, CellPos, Fx, GRAVITY, LocalRect, MaterialId, mix64};
use fe_materials::{MaterialSet, SpawnDir, State, Transition};

/// Fall-speed units per cell: the value byte stores speed in 1/16 cells per tick.
pub const SPEED_UNITS: u32 = 16;
/// Fastest fall speed, in 1/16 cells per tick (15.9 cells per tick).
pub const MAX_SPEED: u32 = 255;

/// Fall speed gained on `tick`, in 1/16 cells per tick.
///
/// [`GRAVITY`] is 7/256 cells per tick squared, which is 7/16 of a speed unit per
/// tick. The fraction is dithered over ticks with no randomness: the gains over
/// any 16 consecutive ticks add up to exactly 7.
#[inline]
pub const fn gravity_gain(tick: u64) -> u32 {
    let g = GRAVITY.0 as u64 * SPEED_UNITS as u64;
    ((((tick + 1) * g) >> 16) - ((tick * g) >> 16)) as u32
}

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
    /// A liquid cell landed hard and was thrown back up. The cell is already
    /// gone from the grid; the caller launches it as a free particle from the
    /// center of `pos`.
    Splash {
        /// Where the cell landed.
        pos: CellPos,
        /// The cell, at rest.
        cell: Cell,
        /// Launch velocity in cells per tick.
        vx: Fx,
        /// Launch velocity in cells per tick, negative upward.
        vy: Fx,
    },
}

/// Per-job output, merged in chunk order after each pass.
#[derive(Default, Debug)]
pub struct JobOut {
    /// Events in the order the job produced them.
    pub events: Vec<Event>,
    /// Cells the job visited.
    pub visited: u64,
    /// Top cells of resting liquid columns the job visited, for the flow
    /// phase: world `(y, x)`.
    pub tops: Vec<(i32, i32)>,
    /// Resting liquid surface cells on top of liquid, for the pressure pass: world `(y, x)`.
    pub pressure: Vec<(i32, i32)>,
    /// Splashes thrown by this job this tick.
    pub splashes: u32,
    /// The main sweep left liquid in the air for the fall sweep.
    pub falling: bool,
    /// Columns of the chunk's top row where the fall sweep found liquid in
    /// the air, one bit per column: liquid resting on it in the chunk above
    /// may have lost its support.
    pub fall_up: u64,
    /// Set before the job runs: in the fall sweep's first round, liquid in
    /// the air that would fly into liquid ahead that has not moved yet waits
    /// for the second round (see `liquid::waits`).
    pub(crate) may_wait: bool,
    /// Some liquid in the air waited for the second round.
    pub(crate) waited: bool,
}

/// The material fields the inner loop reads, packed into 12 bytes.
#[derive(Clone, Copy, Debug)]
pub struct Fast {
    pub(crate) state: State,
    flags: u8,
    pub(crate) density: i16,
    conductivity: u8,
    cooling: u8,
    hold_temp: u8,
    spread: u8,
    heat_cap: u8,
    /// Chance to slide out of 65,536; [`SLIDE_ALWAYS`] means every time.
    slide: u16,
}

impl Fast {
    /// True if the material's value byte holds a lifetime rather than a speed.
    #[inline]
    pub(crate) const fn has_lifetime(&self) -> bool {
        self.flags & F_LIFETIME != 0
    }
}

/// [`Fast::slide`] value for powders that always slide.
const SLIDE_ALWAYS: u16 = u16::MAX;

const F_LIFETIME: u8 = 1;
const F_TRANSITIONS: u8 = 2;
const F_SPAWN: u8 = 4;
const F_REACTIVE: u8 = 8;

/// Builds the packed per-material table used by [`RuleCtx`].
pub fn fast_table(mats: &MaterialSet) -> Vec<Fast> {
    mats.iter()
        .map(|(id, m)| {
            let mut flags = 0;
            if m.lifetime.is_some() {
                flags |= F_LIFETIME;
            }
            if m.melt.is_some() || m.freeze.is_some() || m.explode.is_some() || m.ignite.is_some() {
                flags |= F_TRANSITIONS;
            }
            if m.spawn.is_some() {
                flags |= F_SPAWN;
            }
            if mats.has_reactions(id) {
                flags |= F_REACTIVE;
            }
            Fast {
                state: m.state,
                flags,
                density: m.density,
                conductivity: m.conductivity,
                cooling: m.cooling,
                hold_temp: m.hold_temp,
                spread: m.spread,
                heat_cap: m.heat_capacity.max(1),
                slide: u16::try_from(m.slide).unwrap_or(SLIDE_ALWAYS),
            }
        })
        .collect()
}

/// Read-only state shared by every job in a tick.
pub struct RuleCtx<'a> {
    /// Material tables.
    pub mats: &'a MaterialSet,
    /// Packed per-material fields.
    pub fast: &'a [Fast],
    /// Per-material liquid parameters.
    pub liq: &'a [LiqFast],
    /// Current tick.
    pub tick: u64,
    base: u64,
    wind: i32,
    /// Random bits liquid in the air rounds its speeds with on this tick,
    /// shared by every cell.
    pub(crate) air_round: u32,
    /// Seed of the random bits that do not depend on the tick.
    still: u64,
}

impl<'a> RuleCtx<'a> {
    /// Creates the context for one tick.
    pub fn new(
        mats: &'a MaterialSet,
        fast: &'a [Fast],
        liq: &'a [LiqFast],
        seed: u64,
        tick: u64,
    ) -> Self {
        let mut ctx = Self {
            mats,
            fast,
            liq,
            tick,
            base: mix64(seed ^ mix64(tick.wrapping_add(0x51ed))),
            wind: 0,
            air_round: 0,
            still: mix64(seed ^ 0x7a11_5eed),
        };
        ctx.air_round = ctx.rand(0, 0, liquid::SALT_ENCODE);
        ctx
    }

    /// Sets the wind (-4..=4) gases and fire drift with.
    #[must_use]
    pub const fn with_wind(mut self, wind: i8) -> Self {
        self.wind = wind as i32;
        self
    }

    /// Picks a sideways direction, biased by the wind. `r` supplies the randomness.
    #[inline]
    fn side(&self, r: u32) -> i32 {
        // Even the strongest wind leaves an eighth of the moves upwind.
        if self.wind != 0 && ((r >> 20) & 3) < self.wind.unsigned_abs().min(3) {
            self.wind.signum()
        } else if r & 1 == 0 {
            1
        } else {
            -1
        }
    }

    /// Packed fields for a cell's material. Unknown ids read as the boundary.
    #[inline]
    pub fn f(&self, c: Cell) -> Fast {
        match self.fast.get(c.material().index()) {
            Some(f) => *f,
            None => self.fast[1],
        }
    }

    /// Liquid parameters for a cell's material. Non-liquids read as not liquid.
    #[inline]
    pub fn lq(&self, c: Cell) -> LiqFast {
        self.liq.get(c.material().index()).copied().unwrap_or_default()
    }

    /// Random bits for one decision at one world cell on this tick.
    #[inline]
    pub fn rand(&self, x: i32, y: i32, salt: u32) -> u32 {
        let pos = ((x as u32 as u64) << 32) | y as u32 as u64;
        (mix64(self.base ^ pos ^ u64::from(salt).wrapping_mul(0x9e37_79b9_7f4a_7c15)) >> 32) as u32
    }

    /// Random bits for one decision at a place, the same on every tick.
    #[inline]
    pub(crate) fn rand_still(&self, x: i32, y: i32, salt: u32) -> u32 {
        let pos = ((x as u32 as u64) << 32) | y as u32 as u64;
        (mix64(self.still ^ pos ^ u64::from(salt).wrapping_mul(0x9e37_79b9_7f4a_7c15)) >> 32) as u32
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

/// Which cells a sweep over a chunk updates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sweep {
    /// Every cell except liquid in the air, which the fall sweep moves.
    Main,
    /// Only liquid in the air. It runs after the main passes, one chunk row at
    /// a time from the bottom up, so liquid falling across a chunk border
    /// always finds the liquid below it already moved and a falling stream or
    /// mass keeps together.
    Fall,
}

/// Updates the cells of a chunk's dirty rect that `sweep` covers, bottom row
/// first.
///
/// Each row takes two sweeps. The first goes in the tick's direction (left to
/// right on even ticks) and updates every cell except liquid moving the same
/// way; the second goes back the other way and updates only those. Moving
/// liquid is therefore always updated front first, so a stream moving several
/// cells per tick does not run into its own tail.
pub fn update_chunk(
    nb: &Neighborhood<'_>,
    rect: LocalRect,
    ctx: &RuleCtx<'_>,
    out: &mut JobOut,
    sweep: Sweep,
) {
    if rect.is_empty() {
        return;
    }
    let origin = nb.pos().origin();
    let forward = ctx.tick & 1 == 0;
    // Liquid moving in the first sweep's direction is left for the second.
    let late = if forward { 1 } else { -1 };
    let (x0, x1) = (i32::from(rect.x0), i32::from(rect.x1));
    for ly in (i32::from(rect.y0)..=i32::from(rect.y1)).rev() {
        let mut skipped = 0u64;
        let mut first = |lx: i32| {
            let c = nb.get(lx, ly);
            let air = liquid::in_air(nb, ctx, lx, ly, c);
            if air != (sweep == Sweep::Fall) {
                out.falling |= air;
                return;
            }
            if air && ly == 0 {
                out.fall_up |= 1 << lx;
            }
            if air && c.parity() == ctx.parity() && !liquid::is_airborne(c) {
                fall_late(nb, ctx, out, origin, lx, ly, c);
            } else if liquid::mover_dir(ctx, c) == late {
                skipped |= 1 << (lx - x0);
            } else {
                update_cell(nb, ctx, out, origin, lx, ly, c);
            }
        };
        if forward {
            (x0..=x1).for_each(&mut first);
        } else {
            (x0..=x1).rev().for_each(&mut first);
        }
        if skipped == 0 {
            continue;
        }
        // The second sweep revisits only the cells the first one skipped. One
        // that has already moved this tick marked itself and gets no keep.
        let mut second = |lx: i32| {
            if skipped & (1 << (lx - x0)) != 0 {
                let c = nb.get(lx, ly);
                if c.parity() != ctx.parity() {
                    update_cell(nb, ctx, out, origin, lx, ly, c);
                }
            }
        };
        if forward {
            (x0..=x1).rev().for_each(&mut second);
        } else {
            (x0..=x1).for_each(&mut second);
        }
    }
}

/// Moves liquid that rested when the main sweep updated it and is in the air
/// now, because the liquid under it started falling in the fall sweep. Its
/// heat and reactions already ran this tick; it only moves, so a falling body
/// starts to fall as one instead of one cell per tick from the bottom up.
fn fall_late(
    nb: &Neighborhood<'_>,
    ctx: &RuleCtx<'_>,
    out: &mut JobOut,
    origin: CellPos,
    lx: i32,
    ly: i32,
    c: Cell,
) {
    let mut cell = c;
    if let Outcome::Stay { keep } =
        liquid::update(nb, ctx, out, lx, ly, origin.x + lx, origin.y + ly, c, &mut cell)
    {
        nb.set(lx, ly, cell);
        if keep {
            nb.keep(lx, ly);
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
    c: Cell,
) {
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
    if out.may_wait && liquid::waits(nb, ctx, lx, ly, c, out.waited) {
        out.waited = true;
        return;
    }
    let f = ctx.f(c);
    let r = ctx.rand(wx, wy, 0);
    let mut cell = c.with_parity(ctx.tick);
    let mut changed = false;
    let mut awake = false;

    if f.flags & F_LIFETIME != 0 {
        let m = ctx.mats.get(id);
        // A lifetime rate under 100% stretches a u8 lifetime past 255 ticks.
        if m.lifetime_rate >= fe_materials::CHANCE_ONE
            || (ctx.rand(wx, wy, 6) & 0xffff) < m.lifetime_rate
        {
            if cell.value <= 1 {
                // Most of a burnt-out cell goes up as smoke; only some leaves residue.
                let into = match m.residue {
                    Some((res, chance)) if (ctx.rand(wx, wy, 5) & 0xffff) < chance => res,
                    _ => m.decay,
                };
                let nc = new_cell(ctx.mats, into, Some(cell.temp), r);
                transform(nb, ctx, out, lx, ly, wx, wy, c, nc);
                return;
            }
            cell.value -= 1;
            changed = true;
        } else {
            awake = true;
        }
    }

    if f.conductivity > 0 && exchange_heat(nb, ctx, lx, ly, wx, wy, &mut cell, f, r) {
        changed = true;
    }
    if f.hold_temp > cell.temp {
        cell.temp = f.hold_temp;
        changed = true;
    }
    if f.cooling > 0 && cell.temp != AMBIENT_TEMP && ((r >> 24) as u8) < f.cooling {
        cell.temp = if cell.temp > AMBIENT_TEMP { cell.temp - 1 } else { cell.temp + 1 };
        changed = true;
    }

    if f.flags & F_TRANSITIONS != 0 {
        let m = ctx.mats.get(id);
        if let Some(t) = m.melt
            && cell.temp >= t.at
        {
            if latent_done(ctx, wx, wy, t, cell.temp) {
                let nc = new_cell(ctx.mats, t.into, Some(cell.temp), r);
                transform(nb, ctx, out, lx, ly, wx, wy, c, nc);
                return;
            }
            // Still taking in latent heat: hold at the melting point.
            if cell.temp != t.at {
                cell.temp = t.at;
                changed = true;
            }
            awake = true;
        }
        if let Some(t) = m.freeze
            && cell.temp <= t.at
        {
            if latent_done(ctx, wx, wy, t, cell.temp) {
                let into = freeze_into(nb, ctx, lx, ly, cell, f, t.into);
                let nc = new_cell(ctx.mats, into, Some(cell.temp), r);
                transform(nb, ctx, out, lx, ly, wx, wy, c, nc);
                return;
            }
            // Still giving off latent heat: hold at the freezing point.
            if cell.temp != t.at {
                cell.temp = t.at;
                changed = true;
            }
            awake = true;
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
    }

    if f.flags & F_SPAWN != 0
        && let Some(s) = ctx.mats.get(id).spawn
    {
        awake = true;
        let r1 = ctx.rand(wx, wy, 1);
        if (r1 & 0xffff) < s.chance {
            let (dx, dy) = match s.dir {
                SpawnDir::Up => (0, -1),
                SpawnDir::Down => (0, 1),
                SpawnDir::Any => DIRS4[(r1 >> 16) as usize & 3],
            };
            // Emitters push through gas, so a sprinkler or a chimney keeps
            // going in the steam or smoke it made.
            let t = nb.get(lx + dx, ly + dy);
            if t.is_air()
                || (!t.has(Cell::FLAG_IN_BODY)
                    && ctx.f(t).state == State::Gas
                    && ctx.mats.get(s.into).state != State::Gas)
            {
                let spawned = new_cell(ctx.mats, s.into, None, r1 >> 8).with_parity(ctx.tick);
                nb.set(lx + dx, ly + dy, spawned);
                nb.mark(lx + dx, ly + dy);
                if ctx.mats.get(s.into).state == State::Static {
                    nb.solid_changed(lx + dx, ly + dy);
                }
            }
        }
    }

    if f.flags & F_REACTIVE != 0 {
        let r2 = ctx.rand(wx, wy, 2);
        let (dx, dy) = DIRS4[(r2 >> 16) as usize & 3];
        let n = nb.get(lx + dx, ly + dy);
        if let Some(rx) = ctx.mats.reaction(id, n.material()) {
            if (r2 & 0xffff) < rx.chance {
                // Both products start at the reactants' mixed temperature, so
                // water on lava gives hot steam and glowing obsidian, and water on
                // a burning plank leaves wood too cool to catch again.
                // A reaction can also set a floor, so water quenching an ember
                // flashes into steam hot enough to rise.
                let fnb = ctx.f(n);
                let (ca, cb) = (u32::from(f.heat_cap), u32::from(fnb.heat_cap));
                let mix = (ca * u32::from(cell.temp) + cb * u32::from(n.temp)) / (ca + cb);
                let mix = Some((mix as u8).max(rx.min_temp));
                // A liquid that reacts into a solid follows the freezing rule: the
                // obsidian skin where water meets lava crumbles to gravel and sinks
                // unless it rests on lava or rock, so the contact keeps renewing.
                let into_self = freeze_into(nb, ctx, lx, ly, cell, f, rx.into_self);
                let into_other = freeze_into(nb, ctx, lx + dx, ly + dy, n, fnb, rx.into_other);
                let me = new_cell(ctx.mats, into_self, mix, r2);
                let other = new_cell(ctx.mats, into_other, mix, r2 >> 7);
                transform(nb, ctx, out, lx + dx, ly + dy, wx + dx, wy + dy, n, other);
                transform(nb, ctx, out, lx, ly, wx, wy, c, me);
                return;
            }
            awake = true;
        }
    }

    if !cell.has(Cell::FLAG_IN_BODY) {
        let moved = match f.state {
            State::Powder => move_powder(nb, ctx, lx, ly, wx, wy, &mut cell, f, r, &mut changed),
            State::Liquid => match liquid::update(nb, ctx, out, lx, ly, wx, wy, c, &mut cell) {
                Outcome::Done => true,
                Outcome::Stay { keep } => {
                    awake |= keep;
                    false
                }
            },
            State::Gas => move_gas(nb, ctx, lx, ly, wx, wy, cell, f),
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
    let f = ctx.fast[0];
    let r = ctx.rand(wx, wy, 0);
    let mut cell = c;
    exchange_heat(nb, ctx, lx, ly, wx, wy, &mut cell, f, r);
    if ((r >> 24) as u8) < f.cooling && cell.temp != AMBIENT_TEMP {
        cell.temp = if cell.temp > AMBIENT_TEMP { cell.temp - 1 } else { cell.temp + 1 };
    }
    nb.set(lx, ly, cell);
    nb.mark(lx, ly);
}

/// Latent heat: a cell past a transition threshold changes with a chance equal to
/// its overshoot over the latent heat, and is held at the threshold otherwise, so
/// on average it takes in (or gives off) `latent` steps of heat before it changes.
#[inline]
fn latent_done(ctx: &RuleCtx<'_>, wx: i32, wy: i32, t: Transition, temp: u8) -> bool {
    if t.latent == 0 {
        return true;
    }
    let over = u32::from(temp.abs_diff(t.at));
    ctx.rand(wx, wy, 5) % u32::from(t.latent) < over
}

/// What a freezing cell becomes. A liquid that freezes while falling, over open
/// space, or on top of a different fluid or powder crumbles into its solid's
/// debris instead of hanging in the air as a fixed solid: falling lava lands as
/// rubble, and molten metal poured into water sinks as scrap. A crust on its own
/// melt (basalt on lava, metal on molten metal, ice on water) stays solid if
/// the melt under it stands on something solid or is [`CRUST_DEPTH`] cells
/// deep: on a few cells of melt that are falling, or that lie on powder or a
/// different fluid, the melt freezes to debris or runs off in turn and would
/// leave the crust hanging in the air.
fn freeze_into(
    nb: &Neighborhood<'_>,
    ctx: &RuleCtx<'_>,
    lx: i32,
    ly: i32,
    cell: Cell,
    f: Fast,
    target: MaterialId,
) -> MaterialId {
    if f.state != State::Liquid || ctx.mats.get(target).state != State::Static {
        return target;
    }
    let below = nb.get(lx, ly + 1);
    let falling = f.flags & F_LIFETIME == 0 && liquid::falling(cell);
    let unsupported = if falling || below.is_air() {
        true
    } else if below.has(Cell::FLAG_IN_BODY) {
        false
    } else {
        match ctx.f(below).state {
            State::Gas | State::Fire | State::Empty => true,
            // Its own melt holds a crust only while that melt rests on something
            // solid or is deep: a drop freezing on top of another falling drop is
            // not resting on anything.
            State::Liquid | State::Powder => {
                below.material() != cell.material() || !on_solid(nb, ctx, lx, ly, cell)
            }
            State::Static => false,
        }
    };
    let debris = ctx.mats.get(target).debris;
    if unsupported && debris != MaterialId::AIR { debris } else { target }
}

/// Melt this deep under a freezing cell holds its crust up.
const CRUST_DEPTH: i32 = 8;

/// True if the melt of `cell` under `(lx, ly)` rests on something solid, or
/// is [`CRUST_DEPTH`] cells deep: a pool a crust can float on.
fn on_solid(nb: &Neighborhood<'_>, ctx: &RuleCtx<'_>, lx: i32, ly: i32, cell: Cell) -> bool {
    for k in 1..=CRUST_DEPTH {
        let c = nb.get(lx, ly + k);
        if c.material() == cell.material() && !c.has(Cell::FLAG_IN_BODY) {
            if liquid::is_airborne(c) {
                return false;
            }
            continue;
        }
        return c.has(Cell::FLAG_IN_BODY) || ctx.f(c).state == State::Static;
    }
    true
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

/// Temperature changes for one heat exchange between a cell (capacity `ca`) and
/// a neighbor (capacity `cb`) that is `diff` steps hotter, with conductivity `k`
/// (at most 128). Returns the change for the cell and for the neighbor.
///
/// Each side changes in inverse proportion to its heat capacity, so `ca * qa`
/// equals `-cb * qb` in expectation and the total heat is conserved. Fractions
/// are rounded stochastically: with equal capacities both sides use the same
/// draw from `r_same`, which is exactly the original equal-capacity rule and
/// conserves heat exactly; otherwise each side rounds with its own 16 bits of
/// `r_split`.
pub(crate) fn heat_split(
    diff: i32,
    k: i32,
    ca: i32,
    cb: i32,
    r_same: u32,
    r_split: u32,
) -> (i32, i32) {
    if ca == cb {
        let total = diff * k;
        let mut q = total / 256;
        if (((r_same >> 8) & 0xff) as i32) < (total % 256).abs() {
            q += total.signum();
        }
        return (q, -q);
    }
    let den = 128 * (ca + cb);
    let round = |num: i32, bits: u32| {
        let mut q = num / den;
        let rem = (num % den).unsigned_abs();
        if ((u64::from(bits & 0xffff) * den as u64) >> 16) < u64::from(rem) {
            q += num.signum();
        }
        q
    };
    (round(diff * k * cb, r_split), -round(diff * k * ca, r_split >> 16))
}

#[inline(always)]
#[expect(clippy::too_many_arguments, reason = "hot path; a struct would add copies")]
#[expect(clippy::inline_always, reason = "runs for most visited cells; the call cost showed")]
fn exchange_heat(
    nb: &Neighborhood<'_>,
    ctx: &RuleCtx<'_>,
    lx: i32,
    ly: i32,
    wx: i32,
    wy: i32,
    cell: &mut Cell,
    f: Fast,
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
    let fnb = ctx.f(n);
    let k = i32::from(f.conductivity.min(fnb.conductivity).min(128));
    if k == 0 {
        return false;
    }
    let (ca, cb) = (i32::from(f.heat_cap), i32::from(fnb.heat_cap));
    let r_split = if ca == cb { 0 } else { ctx.rand(wx, wy, 7) };
    let (qa, qb) = heat_split(diff, k, ca, cb, r, r_split);
    if qa == 0 && qb == 0 {
        return false;
    }
    cell.temp = (i32::from(cell.temp) + qa).clamp(0, 255) as u8;
    if qb != 0 {
        let mut n2 = n;
        n2.temp = (i32::from(n.temp) + qb).clamp(0, 255) as u8;
        nb.set(lx + dx, ly + dy, n2);
        nb.mark(lx + dx, ly + dy);
    }
    true
}

/// Can a moving cell enter `target`? `sinking` is true when moving down or sideways
/// into lighter material, false when rising into heavier material.
#[inline]
fn can_enter(ctx: &RuleCtx<'_>, m: Fast, target: Cell, sinking: bool, r: u32) -> bool {
    if target.is_air() {
        return true;
    }
    if target.has(Cell::FLAG_IN_BODY) {
        return false;
    }
    let tm = ctx.f(target);
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
    nb.mark_rect(lx.min(tx) - 1, ly.min(ty) - 1, lx.max(tx) + 1, ly.max(ty) + 1);
}

/// What [`fall`] did with a cell.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FallResult {
    /// Something is in the way below; the cell may slide or spread instead.
    Blocked,
    /// The cell moved down at least one cell. Its new position is written.
    Moved,
    /// The way down is open but the cell is still too slow to move a whole cell
    /// this tick. It stays put with its speed stored; it must not slide or spread.
    Hover,
}

/// Fastest a cell sinks through a liquid, in 1/16 cells per tick (1.5 cells
/// per tick before the 50% chance of each swap).
const SINK_SPEED: u32 = 24;

/// True if `t` is a free liquid lighter than `m`, which `m` sinks through.
#[inline]
fn sinks_through(ctx: &RuleCtx<'_>, m: Fast, t: Cell) -> bool {
    if t.is_air() || t.has(Cell::FLAG_IN_BODY) {
        return false;
    }
    let tm = ctx.f(t);
    tm.state == State::Liquid && tm.density < m.density
}

/// Fall speed a cell keeps when `below` stops it: a falling powder or liquid
/// cell carries the stream along at its own speed, anything else stops it dead.
#[inline]
fn blocked_speed(ctx: &RuleCtx<'_>, v: u32, below: Cell) -> u8 {
    if below.is_air() || below.has(Cell::FLAG_IN_BODY) {
        return 0;
    }
    let bf = ctx.f(below);
    if bf.flags & F_LIFETIME != 0 {
        0
    } else if bf.state == State::Liquid {
        v.min(liquid::decode(below).1 as u32) as u8
    } else if bf.state == State::Powder {
        v.min(u32::from(below.value)) as u8
    } else {
        0
    }
}

/// Falls straight down with acceleration. Powders only: liquids move in [`liquid`].
#[expect(clippy::too_many_arguments, reason = "hot path; a struct would add copies")]
fn fall(
    nb: &Neighborhood<'_>,
    ctx: &RuleCtx<'_>,
    lx: i32,
    ly: i32,
    cell: &mut Cell,
    m: Fast,
    r: u32,
    changed: &mut bool,
) -> FallResult {
    let has_speed = m.flags & F_LIFETIME == 0;
    let below = nb.get(lx, ly + 1);
    let mut v = if has_speed {
        (u32::from(cell.value) + gravity_gain(ctx.tick)).min(MAX_SPEED)
    } else {
        SPEED_UNITS
    };
    if !can_enter(ctx, m, below, true, r) {
        if has_speed && sinks_through(ctx, m, below) {
            // A lighter liquid below only slows the cell down: it keeps its speed,
            // up to the speed it sinks at, and tries again next tick instead of
            // spreading out over the surface. Molten metal poured into water
            // sinks while it is still liquid instead of skating across the top.
            cell.value = v.min(SINK_SPEED) as u8;
            nb.set(lx, ly, *cell);
            nb.mark(lx, ly);
            return FallResult::Hover;
        }
        if has_speed {
            // Riding on a falling cell: accelerate with it, never past it.
            let v = blocked_speed(ctx, v, below);
            if cell.value != v {
                cell.value = v;
                *changed = true;
            }
        }
        return FallResult::Blocked;
    }
    if !has_speed {
        v = SPEED_UNITS;
    }
    // Whole cells this tick. The fraction is dithered over ticks, with a phase
    // per column: cells in one column at one speed move together, so a falling
    // slab or stream keeps its shape, while neighboring columns do not move in
    // lockstep.
    let wx = nb.pos().origin().x + lx;
    let phase = ctx.tick + u64::from(mix64(wx as u32 as u64) as u32 & 15);
    let steps = (((phase + 1) * u64::from(v)) >> 4) - ((phase * u64::from(v)) >> 4);
    if steps == 0 {
        // Still accelerating from rest: keep the speed and look again next tick.
        cell.value = v as u8;
        nb.set(lx, ly, *cell);
        nb.mark(lx, ly);
        return FallResult::Hover;
    }
    let steps = steps as i32;
    let mut y = ly;
    let mut stop = 0u8;
    let mut blocked = false;
    // `stored` is what (lx, y) holds right now. Each step moves the displaced cell
    // up into (lx, y); when that is the value already there (air falling through
    // air), the write is skipped, so a long fall costs two writes, not one per cell.
    let mut stored = nb.get(lx, ly);
    // Fast path: plain ambient air shifts up unchanged, so skip over it at once.
    let run = nb.run_below(lx, ly, steps, Cell::AIR);
    if run > 0 {
        nb.set(lx, ly, Cell::AIR);
        stored = Cell::AIR;
        y += run;
    }
    for i in run..steps {
        let t = nb.get(lx, y + 1);
        if !can_enter(ctx, m, t, true, r >> (8 + i as u32 % 16)) {
            blocked = true;
            stop = if sinks_through(ctx, m, t) {
                v.min(SINK_SPEED) as u8
            } else {
                blocked_speed(ctx, v, t)
            };
            break;
        }
        if !t.is_air() && ctx.f(t).state == State::Liquid {
            // Drag: nothing sinks through a liquid faster than SINK_SPEED.
            v = v.min(SINK_SPEED);
        }
        let up = if t.is_air() { t } else { t.with_parity(ctx.tick) };
        if up != stored {
            nb.set(lx, y, up);
        }
        stored = t;
        y += 1;
    }
    if has_speed {
        cell.value = if blocked { stop } else { v as u8 };
    }
    nb.set(lx, y, *cell);
    // One mark covers the whole column the cell passed through.
    nb.mark_rect(lx - 1, ly - 1, lx + 1, y + 1);
    FallResult::Moved
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
    m: Fast,
    r: u32,
    changed: &mut bool,
) -> bool {
    match fall(nb, ctx, lx, ly, cell, m, r, changed) {
        FallResult::Moved | FallResult::Hover => return true,
        FallResult::Blocked => {}
    }
    if m.slide != SLIDE_ALWAYS && (r >> 8) & 0xffff >= u32::from(m.slide) {
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

#[expect(clippy::too_many_arguments, reason = "hot path; a struct would add copies")]
fn move_gas(
    nb: &Neighborhood<'_>,
    ctx: &RuleCtx<'_>,
    lx: i32,
    ly: i32,
    wx: i32,
    wy: i32,
    cell: Cell,
    m: Fast,
) -> bool {
    let r = ctx.rand(wx, wy, 3);
    let side = ctx.side(r);
    // Gas caught in a liquid bubbles up through it, whatever its density, so
    // dust from a blast under water washes out instead of hanging in the water.
    let above = nb.get(lx, ly - 1);
    if !above.is_air()
        && !above.has(Cell::FLAG_IN_BODY)
        && ctx.f(above).state == State::Liquid
        && (r >> 5) & 1 == 0
    {
        swap_to(nb, ctx, lx, ly, lx, ly - 1, cell);
        return true;
    }
    // Buoyancy: the further a gas's density is from air's, the more often it
    // moves vertically, up if lighter and down if heavier. Steam (-10) always
    // rises, smoke (-5) on 62% of ticks, mist (-1) on 12%, so mist lingers and
    // drifts while steam boils straight up.
    let lift = (u32::from(m.density.unsigned_abs()) * 32).min(255);
    if m.density != 0 && (r >> 24) < lift {
        let dy = if m.density < 0 { -1 } else { 1 };
        let first = if (r >> 1).is_multiple_of(4) { (side, dy) } else { (0, dy) };
        for (dx, dy) in [first, (0, dy), (-side, dy)] {
            if can_enter(ctx, m, nb.get(lx + dx, ly + dy), dy > 0, r >> 5) {
                swap_to(nb, ctx, lx, ly, lx + dx, ly + dy, cell);
                return true;
            }
        }
    }
    let spread = i32::from(m.spread.max(1));
    let k = 1 + ((r >> 8) % spread as u32) as i32;
    for s in [side, -side] {
        // Every cell on the way must be air, or gas would pass through thin walls.
        if (1..=k).all(|j| nb.get(lx + s * j, ly).is_air()) {
            swap_to(nb, ctx, lx, ly, lx + s * k, ly, cell);
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
    let side = ctx.side(r >> 3);
    if ctx.f(cell).density > 0 {
        // Heavy sparks such as embers fall instead, drifting downwind, so they
        // can land on fuel, or in water, which puts them out.
        for (dx, dy) in [(0, 1), (side, 1), (side, 0)] {
            if nb.get(lx + dx, ly + dy).is_air() {
                swap_to(nb, ctx, lx, ly, lx + dx, ly + dy, cell);
                return true;
            }
        }
        return false;
    }
    // Wind tilts some flames, more the stronger it blows; the rest rise straight.
    let gust = ctx.wind != 0 && ((r >> 12) & 3) < ctx.wind.unsigned_abs();
    let order =
        if gust { [(side, -1), (0, -1), (side, 0)] } else { [(0, -1), (side, -1), (-side, -1)] };
    for (dx, dy) in order {
        if nb.get(lx + dx, ly + dy).is_air() {
            swap_to(nb, ctx, lx, ly, lx + dx, ly + dy, cell);
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rule before heat capacities existed: one amount, moved from one cell
    /// to the other.
    fn old_rule(diff: i32, k: i32, r: u32) -> i32 {
        let total = diff * k;
        let mut q = total / 256;
        if (((r >> 8) & 0xff) as i32) < (total % 256).abs() {
            q += total.signum();
        }
        q
    }

    #[test]
    fn equal_capacities_match_the_old_rule() {
        let mut seed = 1u64;
        for _ in 0..20_000 {
            seed = mix64(seed);
            let diff = (seed % 511) as i32 - 255;
            let k = ((seed >> 16) % 129) as i32;
            let c = ((seed >> 24) % 255 + 1) as i32;
            let r = (seed >> 32) as u32;
            let q = old_rule(diff, k, r);
            assert_eq!(heat_split(diff, k, c, c, r, (seed >> 7) as u32), (q, -q));
        }
    }

    #[test]
    fn heat_is_conserved_in_expectation() {
        let mut seed = 7u64;
        let mut net = 0i64;
        let mut moved = 0i64;
        for _ in 0..10_000 {
            seed = mix64(seed);
            let diff = (seed % 400) as i32 - 200;
            let k = ((seed >> 12) % 128 + 1) as i32;
            let ca = [1, 4, 14, 20, 30, 42, 50][(seed >> 20) as usize % 7];
            let cb = [1, 6, 16, 22, 40, 60][(seed >> 28) as usize % 6];
            let (qa, qb) = heat_split(diff, k, ca, cb, (seed >> 36) as u32, (seed >> 3) as u32);
            // The colder side warms, the hotter side cools, neither overshoots.
            assert!(qa * diff.signum() >= 0 && qb * diff.signum() <= 0);
            assert!((qa - qb).abs() <= diff.abs() + 1);
            net += i64::from(ca * qa + cb * qb);
            moved += i64::from((ca * qa).abs());
        }
        assert!(moved > 100_000, "moved {moved}");
        assert!(net.abs() * 100 < moved, "net {net} of {moved}");
    }

    #[test]
    fn fast_fields_stay_packed() {
        assert_eq!(size_of::<Fast>(), 12);
    }

    #[test]
    fn gravity_gain_adds_seven_units_every_sixteen_ticks() {
        for start in [0u64, 5, 1000, 123_456] {
            let sum: u32 = (start..start + 16).map(gravity_gain).sum();
            assert_eq!(sum, 7);
        }
    }
}
