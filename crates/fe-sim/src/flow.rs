//! The flow phase: resting liquid moves as shallow water.
//!
//! A naive falling-sand liquid moves one cell at a time, so a column of water
//! drains a few cells per tick and a collapsing face turns to spray. Here a
//! body of resting liquid moves as a whole. Each column of resting liquid is
//! a column of shallow water with a depth and a flow speed, and once a tick,
//! after the cell passes, every body that is not at rest is stepped with the
//! shallow water equations: a finite volume scheme with HLL fluxes and
//! hydrostatic reconstruction over the bed, in fixed point, with as many
//! substeps as the fastest wave needs. The cells then follow the columns'
//! new depths: the top cells of columns that lost liquid move to the tops of
//! columns that gained it, each move a swap with the air or gas it replaces.
//! So a body never tears, a collapse runs at the speed of a real one, and a
//! surge carries its momentum up slopes and back. Where columns close on each
//! other, as in a bore, momentum mixes between them over a length of a
//! quarter of the depth (a turbulent roller), so a bore's front slopes like a
//! breaking wave instead of standing as a wall of cells.
//!
//! A body that is slow everywhere and level to within a row or two is at
//! rest. It is not stepped; instead settle moves carry single top cells from
//! high columns to columns two or more rows lower along the surface, which
//! levels what whole cells cannot show to the solver, and once nothing moves
//! its chunks sleep.
//!
//! A body is a row of columns linked side by side: spans of resting liquid of
//! one material, plus the dry columns next to them that the liquid can reach
//! this tick. A column is linked to its neighbor when the neighbor's surface
//! or floor is within reach of its own liquid; a neighbor too far below is a
//! brink, and liquid that flows over it leaves as cells in the air, which the
//! cell rule carries down until they land on another column. Liquid that
//! lands joins the column it lands on.
//!
//! A hole in a wall below the surface, a pipe, a tap or a gate is part of
//! the body too: a column with a wall beside its surface looks down that side
//! for a hole, and the columns in the hole and beyond it join the body. Edges
//! where liquid is pressed through an opening under a roof form pipes, whose
//! liquid moves as one slug driven by the heads at their ends, so a hole
//! passes about what Torricelli's law gives however thick the wall.
//!
//! Everything here depends only on the cells: the top cell of each column
//! carries its flow speed, and the phase keeps nothing else between ticks.

use crate::grid::Grid;
use crate::liquid::{self, LiqFast};
use crate::rules::RuleCtx;
use crate::sched::Chunk;
use fe_core::{CHUNK_SIZE, Cell, CellPos, ChunkPos, GRAVITY, MaterialId, PosSet};

/// One cell (or one cell per tick) in the solver's fixed point.
const Q: i64 = 1 << 16;
/// Gravity, cells per tick squared, in fixed point.
const GQ: i64 = GRAVITY.0 as i64;
/// Most dry columns a body reaches past its last wet column.
const FRONTIER: usize = 32;
/// Deepest step down to a neighboring column that still flows as one body.
/// Liquid pours over a deeper drop as cells in the air.
const STEP_DOWN: i32 = 2;
/// Highest step up a dry column ahead of a front may have.
const STEP_UP: i32 = 6;
/// Most substeps in one tick.
const MAX_SUB: i64 = 48;
/// Courant number of the substeps.
const CFL: i64 = Q * 3 / 4;
/// A column shallower than this has no speed.
const H_EPS: i64 = Q / 64;
/// Deepest column the solver models, in cells.
const H_CAP: i64 = 4096;
/// Fastest flow, cells per tick in fixed point.
const U_CAP: i64 = 15 * Q;
/// A body whose every column is slower than this (1/4096 cells per tick) and
/// whose surface is level to within [`rest_step`] is at rest.
const U_REST: i32 = 320;
/// Tallest step between neighboring surfaces of shallow liquid at rest.
const REST_STEP: i32 = 2;
/// A film this many cells deep or less sticks: slower flow in it stops.
const THIN: i32 = 2;
/// Speed under which a thin film sticks, 1/4096 cells per tick.
const U_STICK: i32 = 1024;
/// Columns on each side of an edge whose mean surface heights are compared to
/// find a gentle slope that single-row steps hide.
const TILT_W: usize = 24;
/// An edge is flat when the fitted surface falls less than a row over this
/// many half windows: a lean of a row over 96 columns still drives the flow.
const FLAT_LEAN: i128 = 4;
/// How far along a resting surface a settle move looks.
const SETTLE_REACH: usize = 2048;
/// Most settle moves in one body per tick: this many, and one more for
/// every eight columns.
const SETTLE_MOVES: usize = 64;
/// Most rows above a brink's lip that liquid pours into.
const BRINK_ROWS: i32 = 48;
/// Most columns past a brink's lip that liquid pours into in one tick.
const MAX_LIP: i32 = 16;
/// Fall speed of liquid poured over a lip or out of a pipe, 1/16 cells per
/// tick: a speed code, so the cells of a stream start alike.
const POUR_FALL: i32 = 4;
/// Liquid poured over a lip with liquid or a floor at most this many rows
/// below it starts down there, hugging the face.
const CLING: i32 = 8;
/// A jet out of an opening contracts to this share of it (the vena
/// contracta), in fixed point. With the head loss of [`PIPE_K0`] the jet then
/// leaves at the Torricelli speed `sqrt(2 g h)`.
const JET_CONTRACT: i64 = Q * 62 / 100;
/// The turbulent roller of a bore mixes momentum over a length of a quarter
/// of the depth: `depth >> ROLLER_SHIFT`.
const ROLLER_SHIFT: u32 = 2;
/// Slowest closing speed between two columns that the roller acts on.
const ROLLER_MIN: i64 = Q / 256;
/// Strongest coupling between two columns in the roller, 1/256 cells.
const ROLLER_CAP: i64 = 1 << 36;
/// The roller acts where the closing speed over a mixing length is at least
/// the wave speed divided by this.
const ROLLER_FROUDE: i64 = 8;
/// Most columns in one body.
const MAX_NODES: usize = 1 << 16;
/// Head loss of liquid pressed through an opening under a roof, as `1 + K`
/// in fixed point. A short hole passes `1 / sqrt(2.6)`, 0.62, of what a
/// frictionless one would, as a sharp-edged orifice does.
const PIPE_K0: i64 = Q * 26 / 10;
/// Extra head loss for every cell of a pipe's length per cell of its height.
const PIPE_KF: i64 = Q / 10;
/// Rows above a column that a roof over an opening is looked for.
const ROOF_SCAN: i32 = 64;
/// Deepest shaft down from a full pipe that is part of the pipe.
const SHAFT: i32 = 64;
/// Widest shaft down from a full pipe that is part of the pipe.
const SHAFT_W: i32 = 8;
/// An edge in no pipe.
const NO_PIPE: u32 = u32::MAX;
/// A height with nothing above it.
const OPEN: i32 = i32::MAX;

/// Random bits that round a speed to the nearest code.
const NEAREST: u32 = 0x8000_8000;
const SALT_ROUND: u32 = 40;
const SALT_CODE: u32 = 41;
const SALT_DRAIN: u32 = 42;
/// 2^64 over the golden ratio: a tick times this, in its top 16 bits, steps
/// through fractions that spread out evenly.
const GOLDEN: u64 = 0x9e37_79b9_7f4a_7c15;

/// Counters from one flow phase.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FlowStats {
    /// Bodies stepped with the shallow water solver.
    pub bodies: u32,
    /// Columns in those bodies.
    pub columns: u32,
    /// Substeps summed over those bodies.
    pub substeps: u32,
    /// Liquid cells moved from one column to another.
    pub moves: u32,
    /// Liquid cells poured over brinks.
    pub poured: u32,
    /// Liquid cells moved by settle moves.
    pub settled: u32,
}

/// One column of a body: a span of resting liquid, or a dry column next to one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Node {
    x: i32,
    /// Row of the first cell under the liquid: the floor.
    bed: i32,
    /// Row of the top liquid cell; `bed` for a dry column.
    top: i32,
    /// No room above: the column cannot rise, so it takes in no more than it
    /// gives out. Liquid of the body's kind falling onto the column is no
    /// cap: flow into the column makes it land.
    capped: bool,
    /// Flow speed at the start of the tick, 1/4096 cells per tick.
    u: i32,
    /// The column hangs over a hole in a floor, held there (see
    /// [`liquid::held_in_hole`]): `bed` is the open cell under its liquid,
    /// and liquid leaves through it.
    drain: bool,
}

impl Node {
    const fn depth(&self) -> i32 {
        self.bed - self.top
    }

    const fn wet(&self) -> bool {
        self.top < self.bed
    }

    /// Surface height, up from row 0.
    const fn eta(&self) -> i32 {
        -self.top
    }
}

/// What lies beside a column.
enum Link {
    Node(Node),
    Wall,
    /// A drop. The height of the roof over the opening that leads to it, if
    /// it is a hole in a wall below the surface; [`OPEN`] otherwise.
    Brink(i32),
}

/// Where a pipe ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PipeEnd {
    /// A column of the body that is not full, by index.
    Node(usize),
    /// The end of the body, at a wall: nothing flows.
    Wall,
    /// The end of the body, at a brink: liquid leaves into the air.
    Brink,
}

/// A pipe: a run of edges where liquid is pressed through openings under a
/// roof (a hole in a wall, a pipe, a tap, a gate), joined through columns
/// that are full and cannot rise. The liquid in it moves as one slug: the
/// same flow passes every edge, driven by the difference between the heads at
/// its two ends and slowed by the head loss of the openings, so a hole below
/// the surface of a tank passes what pressure drives through it however thick
/// the wall and however the pipe turns.
#[derive(Clone, Copy, Debug)]
struct Pipe {
    /// First and last edge.
    e0: usize,
    e1: usize,
    ends: [PipeEnd; 2],
    /// Flow, cells squared per tick in fixed point, positive to the right.
    q: i64,
    /// Inertance: the sum of 1 / height over its full columns and one opening.
    inert: i64,
    /// Height of the narrowest opening, cells in fixed point.
    amin: i64,
    /// Head loss `1 + K`, fixed point.
    loss: i64,
    /// Height of the opening at each end, cells in fixed point.
    open: [i64; 2],
    /// Middle of the opening at each end, a height in fixed point. Liquid
    /// leaving into the air there is under no pressure.
    mid: [i64; 2],
}

/// Scratch space for the flow phase, kept between ticks.
#[derive(Default)]
pub(crate) struct Flow {
    claimed: PosSet,
    tops: PosSet,
    nodes: Vec<Node>,
    left: Vec<Node>,
    h: Vec<i64>,
    hu: Vec<i64>,
    z: Vec<i64>,
    flux: Vec<i64>,
    ml: Vec<i64>,
    mr: Vec<i64>,
    moved: Vec<i64>,
    tr: Vec<i64>,
    cur: Vec<i32>,
    stack: Vec<CellPos>,
    flat: Vec<bool>,
    sum: Vec<i64>,
    ends: Vec<usize>,
    u: Vec<i64>,
    c: Vec<i64>,
    /// The roller's coupling through each edge, and its elimination pivots
    /// and sums per column.
    rc: Vec<i64>,
    ra: Vec<i64>,
    rm: Vec<i64>,
    rr: Vec<i64>,
    /// The pipe each edge belongs to, or [`NO_PIPE`].
    pipe_of: Vec<u32>,
    pipes: Vec<Pipe>,
    /// Floor and roof heights of each edge's opening.
    gate: Vec<(i32, i32)>,
    /// Full columns of runs that no end presses liquid through: they drain
    /// as open channel flow this tick.
    loose: Vec<usize>,
    /// Columns of the bodies stepped this tick, as `(x, bed)`.
    stepped: PosSet,
    /// For each edge, the way liquid crossing it falls over a step (see
    /// [`spills`]): 1 to the right, -1 to the left, 0 if it does not.
    spill: Vec<i8>,
    /// Liquid each column drained through a hole under it over the tick,
    /// cells in fixed point.
    dr: Vec<i64>,
    /// Cells written this phase, marked dirty in one go at its end.
    writes: Vec<CellPos>,
    blocks: Vec<(ChunkPos, [i32; 4])>,
}

/// Reads cells and column tables, remembering the last chunk it looked in.
struct View<'g> {
    grid: &'g Grid,
    pos: ChunkPos,
    chunk: Option<&'g Chunk>,
}

impl<'g> View<'g> {
    fn new(grid: &'g Grid) -> Self {
        Self { grid, pos: ChunkPos::new(i32::MIN, i32::MIN), chunk: None }
    }

    #[inline]
    fn chunk(&mut self, x: i32, y: i32) -> Option<&'g Chunk> {
        let cp = CellPos::new(x, y).chunk();
        if cp != self.pos {
            self.pos = cp;
            self.chunk = self.grid.chunk(cp);
        }
        self.chunk
    }

    #[inline]
    fn get(&mut self, x: i32, y: i32) -> Cell {
        match self.chunk(x, y) {
            Some(c) => c.get(CellPos::new(x, y).local_index()),
            None => Cell::BOUNDARY,
        }
    }
}

/// The liquid a body is made of, and how cells relate to it.
#[derive(Clone, Copy)]
struct Kind<'c, 'm> {
    ctx: &'c RuleCtx<'m>,
    mat: MaterialId,
    density: i16,
    q: LiqFast,
}

impl Kind<'_, '_> {
    /// The body's liquid, resting.
    #[inline]
    fn rest(&self, c: Cell) -> bool {
        c.material() == self.mat && !c.has(Cell::FLAG_IN_BODY) && !liquid::is_airborne(c)
    }

    /// Something a cell of this liquid can be swapped into: air, gas, fire or
    /// a lighter liquid.
    #[inline]
    fn room(&self, c: Cell) -> bool {
        if c.is_air() {
            return true;
        }
        if c.has(Cell::FLAG_IN_BODY) {
            return false;
        }
        let f = self.ctx.f(c);
        f.state.is_fluid() && f.density < self.density && !liquid::is_airborne(c)
    }

    /// This liquid in the air.
    #[inline]
    fn falling(&self, c: Cell) -> bool {
        c.material() == self.mat && !c.has(Cell::FLAG_IN_BODY) && liquid::is_airborne(c)
    }

    /// Something a column looks through for its neighbor's floor: room, or
    /// this liquid in the air.
    #[inline]
    fn clear(&self, c: Cell) -> bool {
        self.room(c) || self.falling(c)
    }

    /// The top of the run of resting liquid holding `(x, y)`.
    fn run_top(&self, v: &mut View<'_>, x: i32, y: i32) -> i32 {
        let mut t = y;
        loop {
            let row = t & (CHUNK_SIZE - 1);
            // Up the rest of this chunk cell by cell.
            for _ in 0..row {
                if !self.rest(v.get(x, t - 1)) {
                    return t;
                }
                t -= 1;
            }
            if !self.rest(v.get(x, t - 1)) {
                return t;
            }
            // At the chunk's top row with more liquid above: the tables say how
            // much head continues; trust them if the cells agree.
            if let Some(c) = v.chunk(x, t) {
                let h = i32::from(c.head_in()[(x & (CHUNK_SIZE - 1)) as usize]);
                if h > 1 {
                    let g = t - h;
                    if self.rest(v.get(x, g)) && !self.rest(v.get(x, g - 1)) {
                        return g;
                    }
                }
            }
            t -= 1;
        }
    }

    /// The floor under a run of resting liquid whose top is `top`, or `None`
    /// if the run hangs over open space and is about to fall. A run held
    /// over a hole in a floor drains through it: its floor is the open cell
    /// under it.
    fn run_bed(&self, v: &mut View<'_>, x: i32, top: i32) -> Option<i32> {
        let mut b = top + 1;
        loop {
            let row = b & (CHUNK_SIZE - 1);
            for _ in row..CHUNK_SIZE {
                let c = v.get(x, b);
                if !self.rest(c) {
                    return self.floor_at(v, x, b, c);
                }
                b += 1;
            }
            // Past a chunk's bottom row with more liquid below: the tables say
            // how much head continues under it.
            if let Some(c) = v.chunk(x, b - 1) {
                let f = i32::from(c.foot_in()[(x & (CHUNK_SIZE - 1)) as usize]);
                if f > 1 {
                    let g = b + f;
                    let below = v.get(x, g);
                    if self.rest(v.get(x, g - 1)) && !self.rest(below) {
                        return self.floor_at(v, x, g, below);
                    }
                }
            }
        }
    }

    /// The floor of a run of resting liquid whose first cell below is `c`,
    /// at row `b`: `b` if `c` holds it up or the run is held over a hole in a
    /// floor (as the cell rule holds it), `None` if it is about to fall. A
    /// void one cell deep under the run is no hole: the cell rule closes it.
    fn floor_at(&self, v: &mut View<'_>, x: i32, b: i32, c: Cell) -> Option<i32> {
        if !self.clear(c) {
            return Some(b);
        }
        let bottom = v.get(x, b - 1);
        let density = self.density;
        if !liquid::is_airborne(c)
            && !liquid::unsupported(self.ctx, density, v.get(x, b + 1))
            && self.rest(v.get(x, b - 2))
        {
            return None;
        }
        liquid::held_in_hole(self.ctx, bottom, x, b - 1, |px, py| v.get(px, py)).then_some(b)
    }

    fn node(&self, v: &mut View<'_>, x: i32, top: i32, bed: i32) -> Node {
        let t = v.get(x, top);
        let u = if self.q.keeps_speed() && top < bed { liquid::flow_speed(t) } else { 0 };
        let drain = top < bed && self.clear(v.get(x, bed));
        Node { x, bed, top, capped: !self.clear(v.get(x, top - 1)), u, drain }
    }

    /// What lies beside node `a` in direction `dir`.
    fn probe(&self, v: &mut View<'_>, a: &Node, dir: i32) -> Link {
        let x = a.x + dir;
        let level = if a.wet() { a.top } else { a.bed - 1 };
        let c = v.get(x, level);
        if self.rest(c) {
            let top = self.run_top(v, x, level);
            return match self.run_bed(v, x, top) {
                Some(bed) => Link::Node(self.node(v, x, top, bed)),
                None => Link::Wall,
            };
        }
        if self.clear(c) {
            // A full pipe that turns down a shaft goes on down it: the shaft
            // is part of the pipe, and fills.
            let reach = if a.capped && self.shaft(v, x, a, dir) { SHAFT } else { STEP_DOWN };
            return self.open_at(v, x, level, a.bed + reach, OPEN);
        }
        if a.wet() {
            // A wall beside the surface: a hole lower down it, below the
            // surface, joins the column to what lies beyond, so pressure can
            // drive liquid through it.
            for r in level + 1..a.bed {
                let c = v.get(x, r);
                if self.rest(c) {
                    // The hole holds liquid; the wall above it is its roof.
                    return match self.run_bed(v, x, r) {
                        Some(bed) => Link::Node(self.node(v, x, r, bed)),
                        None => Link::Wall,
                    };
                }
                if self.clear(c) {
                    return self.open_at(v, x, r, a.bed + STEP_DOWN, -r);
                }
            }
        }
        {
            // A step up onto a higher floor. Liquid cannot cross it until it
            // rises over it, but the body goes on over the step, so a front can
            // climb it and liquid left on the ledges beyond still belongs to it.
            for k in 1..=STEP_UP {
                let c = v.get(x, level - k);
                if self.rest(c) {
                    let top = self.run_top(v, x, level - k);
                    return match self.run_bed(v, x, top) {
                        Some(bed) => Link::Node(self.node(v, x, top, bed)),
                        None => Link::Wall,
                    };
                }
                if self.clear(c) {
                    let bed = level - k + 1;
                    return Link::Node(self.node(v, x, bed, bed));
                }
            }
        }
        Link::Wall
    }

    /// Column `x` as seen through the open cell at row `r`: the liquid or
    /// floor below the cell, or a brink if there is none down to row `limit`.
    /// `roof` is the height of the roof over the opening, [`OPEN`] if there is
    /// none.
    fn open_at(&self, v: &mut View<'_>, x: i32, r: i32, limit: i32, roof: i32) -> Link {
        let mut r = r + 1;
        loop {
            while r <= limit && self.clear(v.get(x, r)) {
                r += 1;
            }
            if r > limit {
                return Link::Brink(roof);
            }
            if !self.rest(v.get(x, r)) {
                return Link::Node(self.node(v, x, r, r));
            }
            if let Some(bed) = self.run_bed(v, x, r) {
                return Link::Node(self.node(v, x, r, bed));
            }
            // Liquid hanging over open space falls this tick: look past it.
            // Seen as a wall, a few cells poured past a lip that had landed
            // on falling liquid closed the brink for a tick, and the flow
            // piled up against the lip into a spike.
            while r <= limit && self.rest(v.get(x, r)) {
                r += 1;
            }
        }
    }

    /// True if column `x`, past the end of the full column `a` in direction
    /// `dir`, is a shaft down from a pipe: the pipe's roof goes on over it, and
    /// its far wall is near, so liquid pressed out of the pipe fills it rather
    /// than pouring out into open space.
    fn shaft(&self, v: &mut View<'_>, x: i32, a: &Node, dir: i32) -> bool {
        if self.clear(v.get(x, a.top - 1)) {
            return false;
        }
        (1..=SHAFT_W).any(|k| !self.clear(v.get(x + dir * k, a.bed)))
    }

    /// The height of the first thing above column `d` that liquid cannot
    /// pass, looking no higher than height `limit` and no more than
    /// [`ROOF_SCAN`] rows up; [`OPEN`] if there is none.
    fn roof_over(&self, v: &mut View<'_>, d: &Node, limit: i32) -> i32 {
        let stop = (-limit).max(d.top - ROOF_SCAN);
        let mut r = d.top - 1;
        while r >= stop {
            if !self.clear(v.get(d.x, r)) {
                return -(r + 1);
            }
            r -= 1;
        }
        OPEN
    }

    /// Where liquid or a floor lies fewer than [`CLING`] rows below the lip
    /// of a brink at floor `bed`, in column `x` past it, the lowest free cell
    /// down there: liquid poured over the lip goes there first, at rest, so a
    /// sheet hugs the face and lands at once instead of shooting out over a
    /// thin lens of air, which read as a crack in the water. Over a deeper
    /// drop it leaves from the lip.
    fn cling_cell(&self, grid: &Grid, x: i32, bed: i32) -> Option<CellPos> {
        let mut gap = 0;
        while gap < CLING && self.clear(grid.get(CellPos::new(x, bed + gap))) {
            gap += 1;
        }
        if gap >= CLING {
            return None;
        }
        (bed..bed + gap).rev().map(|y| CellPos::new(x, y)).find(|&p| self.room(grid.get(p)))
    }

    /// The cell past the lip of a brink beside column `x`, on floor `bed`,
    /// in direction `dir`, that liquid pours into: the lowest cell with room
    /// for it up to `rows` rows above the lip, in the column past the lip or,
    /// up to `reach` columns out, further along the row where the cells
    /// nearer hold liquid poured earlier. It looks past the body's own liquid
    /// but not past anything else, so nothing pours over the wall above a hole.
    ///
    /// `drop[j - 1]` is how many rows the liquid's path has stepped down by
    /// the `j`th column past the lip: a cell poured further out is where the
    /// liquid that left the lip earlier in the tick has got to. Returns the
    /// cell and how many columns past the lip it is.
    fn lip_cell(
        &self,
        grid: &Grid,
        x: i32,
        dir: i32,
        bed: i32,
        rows: i32,
        drop: &[i32],
    ) -> Option<(CellPos, usize)> {
        for k in 1..=rows {
            let p = CellPos::new(x + dir, bed - k);
            let c = grid.get(p);
            if self.room(c) {
                return Some((p, 1));
            }
            if c.material() != self.mat || c.has(Cell::FLAG_IN_BODY) {
                return None;
            }
            for (j, &dy) in drop.iter().enumerate().skip(1) {
                let q = CellPos::new(x + (j as i32 + 1) * dir, bed - k + dy);
                let c = grid.get(q);
                if self.room(c) {
                    return Some((q, j + 1));
                }
                if c.material() != self.mat || c.has(Cell::FLAG_IN_BODY) {
                    break;
                }
            }
        }
        None
    }
}

#[inline]
fn isqrt64(v: u64) -> u64 {
    if v < 2 {
        return v;
    }
    let bits = 64 - v.leading_zeros();
    let mut x = 1u64 << bits.div_ceil(2);
    loop {
        let y = u64::midpoint(x, v / x);
        if y >= x {
            return x;
        }
        x = y;
    }
}

/// Wave speed `sqrt(g h)` for a depth, both in fixed point.
#[inline]
fn wave(h: i64) -> i64 {
    if h <= 0 {
        return 0;
    }
    isqrt64(((GQ * h) >> 16) as u64 * Q as u64) as i64
}

/// Hydrostatic pressure force `g h^2 / 2` of a column, in fixed point.
#[inline]
fn press(h: i64) -> i64 {
    (((GQ * h) >> 16) * h) >> 17
}

/// HLL flux of mass and momentum between two states, with their wave speeds
/// `c = sqrt(g h)` given.
#[inline]
fn hll(hl: i64, ul: i64, cl: i64, hr: i64, ur: i64, cr: i64) -> (i64, i64) {
    if hl <= 0 && hr <= 0 {
        return (0, 0);
    }
    let (sl, sr) = if hl <= 0 {
        (ur - 2 * cr, ur + cr)
    } else if hr <= 0 {
        (ul - cl, ul + 2 * cl)
    } else {
        ((ul - cl).min(ur - cr), (ul + cl).max(ur + cr))
    };
    let fhl = (hl * ul) >> 16;
    let fhr = (hr * ur) >> 16;
    let fml = ((fhl * ul) >> 16) + press(hl);
    let fmr = ((fhr * ur) >> 16) + press(hr);
    if sl >= 0 {
        return (fhl, fml);
    }
    if sr <= 0 {
        return (fhr, fmr);
    }
    let den = sr - sl;
    let ss = (sl * sr) >> 16;
    let nh = sr * fhl - sl * fhr + ss * (hr - hl);
    let nm = sr * fml - sl * fmr + ss * (fhr - fhl);
    (nh / den, nm / den)
}

/// Runs the flow phase over every body that holds one of `tops`, the top
/// cells of resting columns the cell passes visited. Returns its counters.
pub(crate) fn run(
    grid: &mut Grid,
    flow: &mut Flow,
    ctx: &RuleCtx<'_>,
    mut tops: Vec<(i32, i32)>,
) -> FlowStats {
    let mut stats = FlowStats::default();
    if tops.is_empty() {
        return stats;
    }
    tops.sort_unstable();
    tops.dedup();
    flow.claimed.clear();
    flow.tops.clear();
    flow.stepped.clear();
    for &(y, x) in &tops {
        if flow.tops.contains(CellPos::new(x, y)) {
            continue;
        }
        let c = grid.get(CellPos::new(x, y));
        let q = ctx.lq(c);
        if !q.is_liquid() || c.has(Cell::FLAG_IN_BODY) || liquid::is_airborne(c) {
            continue;
        }
        let kind = Kind { ctx, mat: c.material(), density: ctx.f(c).density, q };
        let Some(seed) = seed_node(grid, &kind, x, y) else { continue };
        if flow.claimed.contains(CellPos::new(seed.x, seed.bed)) {
            continue;
        }
        let ends = build(grid, flow, &kind, seed);
        step_body(grid, flow, &kind, ends, &mut stats);
    }
    flush_marks(grid, &mut flow.writes, &mut flow.blocks);
    stats
}

/// True if the resting liquid cell at `(x, y)` is in a column of a body the
/// flow phase stepped this tick. Such liquid is flowing, and the pressure
/// pass leaves it to the flow: moving its surface cells to a lower opening
/// in the same body would only pile them up there.
pub(crate) fn flowing(grid: &Grid, flow: &Flow, ctx: &RuleCtx<'_>, x: i32, y: i32) -> bool {
    if flow.stepped.is_empty() {
        return false;
    }
    let c = grid.get(CellPos::new(x, y));
    let q = ctx.lq(c);
    if !q.is_liquid() || c.has(Cell::FLAG_IN_BODY) || liquid::is_airborne(c) {
        return false;
    }
    let kind = Kind { ctx, mat: c.material(), density: ctx.f(c).density, q };
    let mut v = View::new(grid);
    kind.run_bed(&mut v, x, y).is_some_and(|bed| flow.stepped.contains(CellPos::new(x, bed)))
}

/// The column holding the top cell at `(x, y)`.
fn seed_node(grid: &Grid, kind: &Kind<'_, '_>, x: i32, y: i32) -> Option<Node> {
    let mut v = View::new(grid);
    let top = kind.run_top(&mut v, x, y);
    let bed = kind.run_bed(&mut v, x, top)?;
    Some(kind.node(&mut v, x, top, bed))
}

/// How a body ends on each side.
#[derive(Clone, Copy)]
struct Ends {
    /// The end is a brink: liquid that flows over it leaves into the air.
    brink: [bool; 2],
    /// The height of the roof over a brink reached through a hole below the
    /// surface, [`OPEN`] otherwise.
    roof: [i32; 2],
}

/// Builds the body around `seed` into `flow.nodes`, left to right, claiming
/// its columns. Returns how it ends.
fn build(grid: &Grid, flow: &mut Flow, kind: &Kind<'_, '_>, seed: Node) -> Ends {
    let mut v = View::new(grid);
    flow.nodes.clear();
    flow.left.clear();
    claim(flow, &seed);
    let mut ends = [false; 2];
    let mut roofs = [OPEN; 2];
    for (side, dir) in [(0usize, -1i32), (1, 1)] {
        let mut cur = seed;
        let mut dry = 0usize;
        loop {
            if flow.nodes.len() + flow.left.len() >= MAX_NODES {
                break;
            }
            let next = match kind.probe(&mut v, &cur, dir) {
                Link::Wall => break,
                Link::Brink(roof) => {
                    ends[side] = true;
                    roofs[side] = roof;
                    break;
                }
                Link::Node(n) => n,
            };
            if flow.claimed.contains(CellPos::new(next.x, next.bed)) {
                break;
            }
            if next.wet() {
                // A wet neighbor must see this column too, or see a wall where
                // this column's floor stands above its surface; if it sees
                // another column instead, the two would not agree on how they
                // meet, and liquid spills onto the lower one.
                let back = kind.probe(&mut v, &next, -dir);
                let mutual = match back {
                    Link::Node(b) => b.x == cur.x && b.bed == cur.bed,
                    Link::Wall => true,
                    Link::Brink(_) => false,
                };
                if !mutual {
                    ends[side] = cur.wet() && next.top >= cur.bed;
                    break;
                }
                dry = 0;
            } else {
                if dry >= FRONTIER {
                    break;
                }
                dry += 1;
            }
            claim(flow, &next);
            if dir < 0 {
                flow.left.push(next);
            } else {
                flow.nodes.push(next);
            }
            cur = next;
        }
        if dir < 0 {
            flow.left.reverse();
            flow.left.push(seed);
        }
    }
    let right = std::mem::take(&mut flow.nodes);
    flow.nodes = std::mem::take(&mut flow.left);
    flow.nodes.extend(right);
    Ends { brink: ends, roof: roofs }
}

fn claim(flow: &mut Flow, n: &Node) {
    flow.claimed.insert(CellPos::new(n.x, n.bed));
    if n.wet() {
        flow.tops.insert(CellPos::new(n.x, n.top));
    }
}

/// Tallest step between two neighboring surfaces in a body at rest: two rows,
/// and a row more for every 256 rows of the shallower column's depth. In deep
/// liquid a step of a few rows is a ripple on a surface that whole cells can
/// only show to within a row or two, which the solver would turn into noise.
fn rest_step(a: &Node, b: &Node) -> i32 {
    REST_STEP + a.depth().min(b.depth()).max(0) / 256
}

/// True if liquid flowing from column `a` into the wet column `b` beside it
/// falls over a step: `b`'s floor lies more than [`STEP_DOWN`] rows below
/// `a`'s, and its surface more than [`STEP_DOWN`] rows and more than a third
/// of `a`'s depth below `a`'s, as over a weir whose tail water reaches less
/// than two thirds of the way up the head. The two stay one body, but the
/// upper column runs out as over a brink and its outflow is poured into the
/// air past the lip, to fall onto `b`. Moved straight onto `b`'s top as a
/// step in the bed of one body, it left no trace: the columns passed on what
/// they took in, and water on top of a block in a dam break stood 28 rows
/// over the lee with a vertical face for 140 ticks while the solver passed 38
/// cells a tick over it. A step deep under the surface is no fall.
fn spills(a: &Node, b: &Node) -> bool {
    a.wet()
        && !a.capped
        && b.wet()
        && b.bed - a.bed > STEP_DOWN
        && b.top - a.top > STEP_DOWN.max(a.depth() / 3)
}

/// True if no liquid can cross between two columns: both are dry, the higher
/// surface does not reach over the step between them, or the lower column has
/// no room to rise.
fn blocked(a: &Node, b: &Node) -> bool {
    let (hi, lo) = if a.eta() > b.eta() { (a, b) } else { (b, a) };
    let zf = (-hi.bed).max(-lo.bed);
    !hi.wet() || hi.eta() <= zf || lo.capped
}

/// Marks each edge (1..n, between columns k - 1 and k) as flat or not into
/// `flat`. An edge is flat when the two surfaces are within a row and the
/// surface around it does not lean: within the run of wet columns holding the
/// edge, a line fitted to the surface over a window around the edge falls less
/// than a row over the window's half width. Whole cells make a level surface
/// bumpy by a row; a flat edge carries the flow that is already moving but no
/// push from such bumps, so a level body comes to rest. The fit uses what the
/// run has on each side, so a gentle slope next to a wall, as in a collapsing
/// column, still drives the flow there.
fn flat_edges(nodes: &[Node], flat: &mut Vec<bool>, sum: &mut Vec<i64>, ends: &mut Vec<usize>) {
    let n = nodes.len();
    flat.clear();
    flat.resize(n + 1, false);
    // Prefix sums of the surface and of index times surface, relative to the
    // first column so the products stay small.
    let base = nodes.first().map_or(0, |d| i64::from(d.eta()));
    sum.clear();
    sum.resize(2 * (n + 1), 0);
    for (i, nd) in nodes.iter().enumerate() {
        let e = i64::from(nd.eta()) - base;
        sum[2 * i + 2] = sum[2 * i] + e;
        sum[2 * i + 3] = sum[2 * i + 1] + e * i as i64;
    }
    // End of the wet run starting at or after each column.
    ends.clear();
    ends.resize(n + 1, n);
    for i in (0..n).rev() {
        ends[i] = if nodes[i].wet() { ends[i + 1] } else { i };
    }
    let w = TILT_W as i64;
    // Start of the wet run each column belongs to.
    let mut start = 0;
    for k in 1..n {
        if !nodes[k - 1].wet() {
            start = k;
        }
        if (nodes[k - 1].eta() - nodes[k].eta()).abs() > 1 {
            continue;
        }
        if !(nodes[k - 1].wet() && nodes[k].wet()) {
            flat[k] = true;
            continue;
        }
        let a = k.saturating_sub(TILT_W).max(start);
        let b = (k + TILT_W).min(ends[k]);
        // Least squares slope over columns a..b: num / den rows per column.
        let m = (b - a) as i64;
        let (a64, b64) = (a as i64, b as i64);
        let sx = (b64 * (b64 - 1) - a64 * (a64 - 1)) / 2;
        let sxx = (b64 * (b64 - 1) * (2 * b64 - 1) - a64 * (a64 - 1) * (2 * a64 - 1)) / 6;
        let sy = sum[2 * b] - sum[2 * a];
        let sxy = sum[2 * b + 1] - sum[2 * a + 1];
        let num = i128::from(m * sxy) - i128::from(sx) * i128::from(sy);
        let den = i128::from(m) * i128::from(sxx) - i128::from(sx) * i128::from(sx);
        flat[k] = num.abs() * i128::from(w) * FLAT_LEAN < den;
    }
}

/// Steps one body: at rest it may level by settle moves; otherwise it runs the
/// shallow water solver and moves the cells to match.
fn step_body(
    grid: &mut Grid,
    flow: &mut Flow,
    kind: &Kind<'_, '_>,
    ends: Ends,
    stats: &mut FlowStats,
) {
    let n = flow.nodes.len();
    let [left_brink, right_brink] = ends.brink;
    flat_edges(&flow.nodes, &mut flow.flat, &mut flow.sum, &mut flow.ends);
    find_pipes(grid, flow, kind, ends);
    let nodes = &flow.nodes;
    let brink_live = (left_brink && nodes[0].wet())
        || (right_brink && nodes[n - 1].wet())
        || nodes.iter().any(|d| d.drain);
    // At rest: slow everywhere, no step taller than `rest_step` across an
    // edge that liquid can cross, and no pipe pressed by a difference in
    // head. What is left, single cells standing a row or two proud and a
    // gentle lean that only shows over a window, settle moves level:
    // rounding the solver's flows to whole cells would only stir them up
    // again, and a large body would never be still at once.
    let rest = !brink_live
        && nodes.iter().all(|d| d.u.abs() <= U_REST)
        && nodes.windows(2).all(|w| {
            (w[0].eta() - w[1].eta()).abs() <= rest_step(&w[0], &w[1]) || blocked(&w[0], &w[1])
        })
        && flow.pipes.iter().all(|p| pipe_head(nodes, p).abs() <= i64::from(REST_STEP) * Q);
    if rest {
        settle_body(grid, flow, kind, ends.brink, stats);
        return;
    }
    stats.bodies += 1;
    stats.columns += n as u32;
    for d in &flow.nodes {
        if d.wet() {
            flow.stepped.insert(CellPos::new(d.x, d.bed));
        }
    }
    flow.spill.clear();
    flow.spill.resize(n + 1, 0);
    for k in 1..n {
        let (a, b) = (&flow.nodes[k - 1], &flow.nodes[k]);
        if flow.pipe_of[k] != NO_PIPE {
            continue;
        }
        if spills(a, b) {
            flow.spill[k] = 1;
        } else if spills(b, a) {
            flow.spill[k] = -1;
        }
    }
    solve(flow, kind, left_brink, right_brink, stats);
    realize(grid, flow, kind, ends, stats);
    settle_moves(grid, flow, kind, ends.brink, true, stats);
}

/// The difference in head between a pipe's two ends at the start of the
/// tick, fixed point; zero for a pipe closed by a wall.
fn pipe_head(nodes: &[Node], p: &Pipe) -> i64 {
    if p.ends.contains(&PipeEnd::Wall) {
        return 0;
    }
    let head = |side: usize| match p.ends[side] {
        PipeEnd::Node(i) => (i64::from(nodes[i].eta()) * Q).max(p.mid[side]),
        _ => p.mid[side],
    };
    head(0) - head(1)
}

/// Finds the body's pipes. An edge is part of one when liquid crossing it is
/// pressed through an opening under a roof: a column beside it is full and
/// cannot rise, or the surface on one side stands above the roof of the
/// opening, as at a hole in a wall below the surface of a tank, a gate, or a
/// tap. A run of such edges joined through full columns is one pipe.
fn find_pipes(grid: &Grid, flow: &mut Flow, kind: &Kind<'_, '_>, ends: Ends) {
    let n = flow.nodes.len();
    let mut v = View::new(grid);
    flow.pipes.clear();
    flow.pipe_of.clear();
    flow.pipe_of.resize(n + 1, NO_PIPE);
    flow.gate.clear();
    flow.loose.clear();
    let nodes = &flow.nodes;
    let mut any = false;
    for k in 0..=n {
        let gate = if k == 0 || k == n {
            let side = usize::from(k == n);
            let e = &nodes[if k == 0 { 0 } else { n - 1 }];
            let roof = ends.roof[side];
            if e.capped {
                Some((-e.bed, e.eta().min(roof)))
            } else if ends.brink[side] && e.wet() && roof < e.eta() {
                Some((-e.bed, roof))
            } else {
                None
            }
        } else {
            let (a, b) = (&nodes[k - 1], &nodes[k]);
            let zf = (-a.bed).max(-b.bed);
            if a.capped || b.capped {
                let ra = if a.capped { a.eta() } else { kind.roof_over(&mut v, a, b.eta()) };
                let rb = if b.capped { b.eta() } else { kind.roof_over(&mut v, b, a.eta()) };
                Some((zf, ra.min(rb)))
            } else {
                let (hi, lo) = if a.eta() >= b.eta() { (a, b) } else { (b, a) };
                let roof = if hi.wet() { kind.roof_over(&mut v, lo, hi.eta()) } else { OPEN };
                (roof < hi.eta()).then_some((zf, roof))
            }
        };
        any |= gate.is_some();
        flow.gate.push(gate.unwrap_or((0, OPEN)));
    }
    if !any {
        return;
    }
    let opening = |k: usize| {
        let (zf, zr) = flow.gate[k];
        (i64::from(zr) - i64::from(zf)).max(1) * Q
    };
    let mut k = 0;
    while k <= n {
        if flow.gate[k].1 == OPEN {
            k += 1;
            continue;
        }
        let e0 = k;
        let mut e1 = k;
        while e1 < n && nodes[e1].capped {
            e1 += 1;
        }
        let end = |brink: bool| if brink { PipeEnd::Brink } else { PipeEnd::Wall };
        let ends = [
            if e0 == 0 { end(ends.brink[0]) } else { PipeEnd::Node(e0 - 1) },
            if e1 == n { end(ends.brink[1]) } else { PipeEnd::Node(e1) },
        ];
        // Liquid is pressed through only while the surface at an end stands
        // above the roof of the opening there. Once the ends are lower, air
        // reaches the full columns from an end and their liquid runs out as
        // open channel flow: they are not full columns of a pipe any more.
        // That takes an end open to the air: a brink, or a column with nothing
        // over it. A wall lets no air in, and under a roof (liquid shut in
        // under a floating crate, say) the full columns stay full.
        let mut pressed = |side: usize, e: usize| match ends[side] {
            PipeEnd::Node(i) => {
                nodes[i].eta() > flow.gate[e].1 || kind.roof_over(&mut v, &nodes[i], OPEN) != OPEN
            }
            PipeEnd::Brink => false,
            PipeEnd::Wall => true,
        };
        if e1 > e0 && !pressed(0, e0) && !pressed(1, e1) {
            flow.loose.extend(e0..e1);
            k = e1 + 1;
            continue;
        }
        let mut amin = (e0..=e1).map(opening).min().unwrap_or(Q);
        let (mut inert, mut qsum) = (0i64, 0i64);
        for d in &nodes[e0..e1] {
            let h = i64::from(d.depth()).clamp(1, H_CAP) * Q;
            amin = amin.min(h);
            inert += Q * Q / h;
            qsum += (h * i64::from(d.u) * 16) >> 16;
        }
        inert += Q * Q / amin;
        let len = (e1 - e0 + 1) as i64;
        let mid = |k: usize| i64::from(flow.gate[k].0) * Q + opening(k) / 2;
        flow.pipes.push(Pipe {
            e0,
            e1,
            ends,
            q: if e1 > e0 { qsum / (e1 - e0) as i64 } else { 0 },
            inert,
            amin,
            loss: PIPE_K0 + PIPE_KF * len * Q / amin,
            open: [opening(e0), opening(e1)],
            mid: [mid(e0), mid(e1)],
        });
        let id = (flow.pipes.len() - 1) as u32;
        for e in e0..=e1 {
            flow.pipe_of[e] = id;
        }
        k = e1 + 1;
    }
    for &i in &flow.loose {
        flow.nodes[i].capped = false;
    }
}

/// A body at rest: zeroes leftover speeds and moves top cells down gentle
/// staircases that single-row steps hide from the solver, and over a brink at
/// the end of a surface that falls toward it. Every move lowers a cell, so the
/// body comes to rest.
fn settle_body(
    grid: &mut Grid,
    flow: &mut Flow,
    kind: &Kind<'_, '_>,
    brinks: [bool; 2],
    stats: &mut FlowStats,
) {
    let n = flow.nodes.len();
    for i in 0..n {
        let d = flow.nodes[i];
        if d.wet() && d.u != 0 {
            let p = CellPos::new(d.x, d.top);
            let c = grid.get(p);
            if kind.rest(c) {
                put(grid, &mut flow.writes, p, liquid::at_rest(c));
            }
            flow.nodes[i].u = 0;
        }
    }
    settle_moves(grid, flow, kind, brinks, false, stats);
}

/// Settle moves: each wet column that is still looks along the surface for a
/// column at least two rows lower, over columns lower than itself, and moves
/// its top cell there, or pours it over a brink at the end of such a surface.
/// With `thin_only`, only still films one or two cells deep move: a moving
/// body leaves them behind on slopes, where single-row steps hold them.
fn settle_moves(
    grid: &mut Grid,
    flow: &mut Flow,
    kind: &Kind<'_, '_>,
    brinks: [bool; 2],
    thin_only: bool,
    stats: &mut FlowStats,
) {
    let tick = kind.ctx.tick;
    let n = flow.nodes.len();
    let mut moves = 0usize;
    let fwd = tick & 1 == 0;
    for k in 0..n {
        if moves >= SETTLE_MOVES + n / 8 {
            break;
        }
        let i = if fwd { k } else { n - 1 - k };
        let d = flow.nodes[i];
        if !d.wet() || (thin_only && (d.depth() > THIN || d.u != 0)) {
            continue;
        }
        let e = flow.nodes[i].eta();
        // A lower column to move to, or a brink to pour over (`Err(dir)`).
        let mut target = None;
        for dir in if fwd { [1i32, -1] } else { [-1, 1] } {
            let mut j = i as i64;
            for _ in 0..SETTLE_REACH {
                j += i64::from(dir);
                if j < 0 || j >= n as i64 {
                    // A moving body pours its still films over a brink only
                    // from beside it. From further in, the cell would leap
                    // over the film between, and the front of a flow running
                    // onto a film would never get deeper and move on.
                    if brinks[usize::from(dir > 0)] && (!thin_only || (j - i as i64).abs() <= 2) {
                        target = Some(Err(dir));
                    }
                    break;
                }
                let o = &flow.nodes[j as usize];
                if o.eta() >= e {
                    break;
                }
                if o.eta() <= e - 2 && !o.capped {
                    target = Some(Ok(j as usize));
                    break;
                }
            }
            if target.is_some() {
                break;
            }
        }
        let a = flow.nodes[i];
        let from = CellPos::new(a.x, a.top);
        let moved = match target {
            Some(Ok(j)) => {
                let b = flow.nodes[j];
                let to = CellPos::new(b.x, b.top - 1);
                let done = swap(grid, &mut flow.writes, kind, from, to);
                if done != Swap::Blocked {
                    flow.nodes[j].top -= 1;
                    flow.nodes[j].capped = !kind.clear(grid.get(CellPos::new(b.x, b.top - 2)));
                }
                done
            }
            Some(Err(dir)) => {
                let end = flow.nodes[if dir > 0 { n - 1 } else { 0 }];
                let x = end.x + dir;
                let c = grid.get(from);
                // The lowest free cell over the lip, past cells poured earlier.
                let rows = end.depth().clamp(1, BRINK_ROWS);
                let to = kind
                    .cling_cell(grid, x, end.bed)
                    .or_else(|| kind.lip_cell(grid, end.x, dir, end.bed, rows, &[0]).map(|q| q.0));
                match to {
                    Some(to) if kind.rest(c) => {
                        let d = grid.get(to);
                        // It leaves as the slowest sheet does, a cell a tick,
                        // so a film trickling off a ledge falls along the
                        // path of the stream poured over it before; at half
                        // that it fell along a path of its own, and the
                        // stream off the ledge was two trails of drops.
                        let fly = if to.y >= end.bed {
                            liquid::at_rest(c)
                        } else {
                            liquid::airborne(liquid::at_rest(c), 16 * dir, POUR_FALL, NEAREST)
                        };
                        put(grid, &mut flow.writes, to, fly.with_parity(tick));
                        put(
                            grid,
                            &mut flow.writes,
                            from,
                            if d.is_air() { d } else { d.with_parity(tick) },
                        );
                        Swap::Moved
                    }
                    _ => Swap::Blocked,
                }
            }
            None => continue,
        };
        match moved {
            Swap::Moved => {
                flow.nodes[i].top += 1;
                moves += 1;
            }
            Swap::Landed => moves += 1,
            // The way is blocked for now, by liquid poured or landing there:
            // look again next tick.
            Swap::Blocked => grid.keep(from),
        }
    }
    stats.settled += moves as u32;
}

/// Writes a cell, leaving its dirty mark for [`flush_marks`].
#[inline]
fn put(grid: &mut Grid, writes: &mut Vec<CellPos>, p: CellPos, c: Cell) {
    if grid.set_raw(p, c) {
        writes.push(p);
    }
}

/// Marks the cells written this phase and their neighbors for a visit on the
/// next tick: the same cells as marking each write's 3x3 block, gathered into
/// one rectangle per chunk so each chunk is marked once.
fn flush_marks(grid: &Grid, writes: &mut Vec<CellPos>, blocks: &mut Vec<(ChunkPos, [i32; 4])>) {
    blocks.clear();
    for &p in writes.iter() {
        let (a, b) =
            (CellPos::new(p.x - 1, p.y - 1).chunk(), CellPos::new(p.x + 1, p.y + 1).chunk());
        for cy in a.y..=b.y {
            for cx in a.x..=b.x {
                let cp = ChunkPos::new(cx, cy);
                let o = cp.origin();
                let r = [
                    (p.x - 1).max(o.x),
                    (p.y - 1).max(o.y),
                    (p.x + 1).min(o.x + CHUNK_SIZE - 1),
                    (p.y + 1).min(o.y + CHUNK_SIZE - 1),
                ];
                match blocks.last_mut() {
                    Some((last, q)) if *last == cp => {
                        *q = [q[0].min(r[0]), q[1].min(r[1]), q[2].max(r[2]), q[3].max(r[3])];
                    }
                    _ => blocks.push((cp, r)),
                }
            }
        }
    }
    blocks.sort_unstable_by_key(|(c, _)| (c.y, c.x));
    let mut i = 0;
    while i < blocks.len() {
        let (cp, mut r) = blocks[i];
        i += 1;
        while i < blocks.len() && blocks[i].0 == cp {
            let q = blocks[i].1;
            r = [r[0].min(q[0]), r[1].min(q[1]), r[2].max(q[2]), r[3].max(q[3])];
            i += 1;
        }
        grid.mark_rect(r[0], r[1], r[2], r[3]);
    }
    writes.clear();
}

/// What a swap did.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Swap {
    /// Nothing: the cell at `from` is not resting liquid of the body's kind,
    /// or `to` has no room.
    Blocked,
    /// The cell at `from` moved to `to`, and what was at `to` back to `from`.
    Moved,
    /// `to` held liquid of the body's kind falling onto the column, which
    /// landed there instead; the cell at `from` did not move.
    Landed,
}

/// Moves the liquid cell at `from` into the cell `to`, and what was at `to`
/// back to `from`. The moved cell rests with no speed. Where liquid of the
/// body's kind is falling onto the column at `to`, the flow meets it: the
/// falling cell lands on the column there, and the moving cell stays.
fn swap(
    grid: &mut Grid,
    writes: &mut Vec<CellPos>,
    kind: &Kind<'_, '_>,
    from: CellPos,
    to: CellPos,
) -> Swap {
    let tick = kind.ctx.tick;
    let c = grid.get(from);
    let d = grid.get(to);
    if !kind.rest(c) || !kind.clear(d) {
        return Swap::Blocked;
    }
    if kind.falling(d) {
        put(grid, writes, to, liquid::at_rest(d).with_parity(tick));
        return Swap::Landed;
    }
    put(grid, writes, to, liquid::at_rest(c).with_parity(tick));
    put(grid, writes, from, if d.is_air() { d } else { d.with_parity(tick) });
    Swap::Moved
}

/// Steps the body's columns with the shallow water equations, leaving the
/// liquid moved through each edge over the tick in `flow.moved` and the new
/// momentum in `flow.hu`, `flow.h`. Pipes are stepped as slugs alongside.
fn solve(
    flow: &mut Flow,
    kind: &Kind<'_, '_>,
    left_brink: bool,
    right_brink: bool,
    stats: &mut FlowStats,
) {
    let n = flow.nodes.len();
    let q = kind.q;
    for v in [&mut flow.h, &mut flow.hu, &mut flow.z, &mut flow.u, &mut flow.c] {
        v.clear();
    }
    for d in &flow.nodes {
        let h = i64::from(d.depth()).min(H_CAP) * Q;
        let u = i64::from(d.u) * 16;
        flow.h.push(h);
        flow.hu.push((h * u) >> 16);
        flow.z.push(-i64::from(d.bed) * Q);
    }
    // The fastest wave: |u| + c, or |u| + 2c where liquid meets a dry column
    // or a brink and runs out at the front's speed. A full column has no
    // surface for a wave to run on.
    let mut amax = 0i64;
    for i in 0..n {
        let h = flow.h[i];
        if h <= 0 || flow.nodes[i].capped {
            continue;
        }
        let c = wave(h);
        let dry = |j: Option<usize>, brink: bool| j.map_or(brink, |j| flow.h[j] <= 0);
        let edge =
            dry(i.checked_sub(1), left_brink) || dry((i + 1 < n).then_some(i + 1), right_brink);
        let u = (i64::from(flow.nodes[i].u) * 16).abs();
        amax = amax.max(u + if edge { 2 * c } else { c });
    }
    let subs = ((amax + CFL - 1) / CFL).clamp(1, MAX_SUB);
    stats.substeps += subs as u32;
    for v in [&mut flow.flux, &mut flow.ml, &mut flow.mr, &mut flow.moved] {
        v.clear();
        v.resize(n + 1, 0);
    }
    flow.dr.clear();
    flow.dr.resize(n, 0);
    let drains = flow.nodes.iter().any(|d| d.drain);
    flow.u.resize(n, 0);
    flow.c.resize(n, 0);
    let sq = subs * Q;
    for _ in 0..subs {
        for i in 0..n {
            let h = flow.h[i];
            flow.u[i] = if h > H_EPS { flow.hu[i] * Q / h } else { 0 };
            flow.c[i] = wave(h);
        }
        let (h, hu, z, u, c) = (&flow.h, &flow.hu, &flow.z, &flow.u, &flow.c);
        // Edges: 0 is the left end, n the right end, k between k - 1 and k.
        for k in 0..=n {
            if flow.pipe_of[k] != NO_PIPE {
                continue;
            }
            let (f, ml, mr) = if k == 0 {
                if left_brink {
                    let (f, m) = hll(0, 0, 0, h[0], u[0], c[0]);
                    (f.min(0), 0, m)
                } else {
                    (0, 0, press(h[0]))
                }
            } else if k == n {
                if right_brink {
                    let (f, m) = hll(h[n - 1], u[n - 1], c[n - 1], 0, 0, 0);
                    (f.max(0), m, 0)
                } else {
                    (0, press(h[n - 1]), 0)
                }
            } else if flow.spill[k] != 0 {
                // Liquid falls over the step: the upper column runs out as
                // over a brink, and the lower one, which the liquid falls
                // into, takes it with none of its momentum but sees a wall.
                // The liquid is poured into the air past the lip and brings
                // its momentum when it lands.
                let (a, b) = (k - 1, k);
                if flow.spill[k] > 0 {
                    let (f, m) = hll(h[a], u[a], c[a], 0, 0, 0);
                    (f.max(0), m, press(h[b]))
                } else {
                    let (f, m) = hll(0, 0, 0, h[b], u[b], c[b]);
                    (f.min(0), press(h[a]), m)
                }
            } else if flow.flat[k] {
                // A flat edge carries what already moves through it, with no
                // push from the one-row step and no spreading across it.
                let (a, b) = (k - 1, k);
                let (ua, ub) = (u[a].max(0), u[b].min(0));
                let (fa, fb) = ((h[a] * ua) >> 16, (h[b] * ub) >> 16);
                let g = ((fa * ua) >> 16) + ((fb * ub) >> 16);
                (fa + fb, g + press(h[a]), g + press(h[b]))
            } else {
                let (a, b) = (k - 1, k);
                let zf = z[a].max(z[b]);
                let hl = (z[a] + h[a] - zf).max(0);
                let hr = (z[b] + h[b] - zf).max(0);
                // Over a step the depths that meet differ from the columns'.
                let cl = if hl == h[a] { c[a] } else { wave(hl) };
                let cr = if hr == h[b] { c[b] } else { wave(hr) };
                // The deeper column's flow crosses the step in the part above
                // it, so it carries its whole discharge there, up to the speed
                // of a wave over the step. With the column's own speed instead,
                // flow onto or off a cliff deep under the surface could not
                // pass what reaches it and the surface stood a step high.
                let ul = step_speed(hl, h[a], hu[a], u[a], cl);
                let ur = step_speed(hr, h[b], hu[b], u[b], cr);
                let (f, m) = hll(hl, ul, cl, hr, ur, cr);
                (f, m + press(h[a]) - press(hl), m + press(h[b]) - press(hr))
            };
            flow.flux[k] = f / subs;
            flow.ml[k] = ml;
            flow.mr[k] = mr;
        }
        if !flow.pipes.is_empty() {
            step_pipes(flow, q, subs);
        }
        // A column held over a hole in a floor drains through it as through
        // an orifice: at the Torricelli speed sqrt(2 g h) for its depth,
        // contracted to 0.62 of the hole (the vena contracta).
        let mut sink = [0i64; 0].to_vec();
        if drains {
            sink =
                flow.nodes
                    .iter()
                    .zip(&flow.h)
                    .map(|(d, &h)| {
                        if d.drain && h > H_EPS {
                            (wave(2 * h) * JET_CONTRACT / Q) / subs
                        } else {
                            0
                        }
                    })
                    .collect();
        }
        // No column sends more than it holds. A full column in a pipe passes
        // on what it takes in.
        for i in 0..n {
            if flow.nodes[i].capped {
                continue;
            }
            let d = sink.get(i).copied().unwrap_or(0);
            let out = flow.flux[i + 1].max(0) + (-flow.flux[i]).max(0) + d;
            if out > flow.h[i] && out > 0 {
                let hi = flow.h[i];
                if flow.flux[i + 1] > 0 {
                    flow.flux[i + 1] = flow.flux[i + 1] * hi / out;
                }
                if flow.flux[i] < 0 {
                    flow.flux[i] = flow.flux[i] * hi / out;
                }
                if d > 0 {
                    sink[i] = d * hi / out;
                }
            }
        }
        // A pipe passes only what its upstream end could send.
        for p in &mut flow.pipes {
            let f = flow.flux[if p.q >= 0 { p.e0 } else { p.e1 }];
            flow.flux[p.e0..=p.e1].fill(f);
            p.q = f * subs;
        }
        for k in 0..=n {
            flow.moved[k] += flow.flux[k];
        }
        // Mass and momentum, then drag: bed friction, viscosity, and a steady
        // loss for thick liquids. A full column in a pipe moves with its slug.
        for i in 0..n {
            // Liquid falling over a step leaves the upper column into the
            // air, poured past the lip; the lower column takes it in only as
            // its cells land. Counted in the lower column at once, it was
            // passed on downstream before it arrived: a column at the foot of
            // the step sent on liquid that was still in the air, sank into a
            // well a cell wide below its neighbors, and showed as a slit of
            // air between the step's face and the water past it.
            let fin = if flow.spill[i] > 0 { 0 } else { flow.flux[i] };
            let fout = if flow.spill[i + 1] < 0 { 0 } else { flow.flux[i + 1] };
            let mut h = (flow.h[i] + fin - fout).max(0);
            if let Some(&d) = sink.get(i)
                && d > 0
            {
                // What drains takes its share of the column's momentum.
                let d = d.min(h);
                flow.dr[i] += d;
                flow.hu[i] = if h > 0 { flow.hu[i] * (h - d) / h } else { 0 };
                h -= d;
            }
            flow.h[i] = h;
            if h <= H_EPS {
                flow.hu[i] = 0;
                continue;
            }
            if flow.nodes[i].capped {
                let p = flow.pipe_of[i];
                let qp = if p == NO_PIPE { 0 } else { flow.pipes[p as usize].q };
                let u = (qp * Q / h).clamp(-U_CAP, U_CAP);
                flow.u[i] = u;
                flow.hu[i] = (h * u) >> 16;
                continue;
            }
            let hu = flow.hu[i] + (flow.mr[i] - flow.ml[i + 1]) / subs;
            let u = (hu * Q / h).clamp(-U_CAP, U_CAP);
            let hc = h.max(Q);
            let k = ((i64::from(q.cf) * u.abs()) >> 16) * Q / hc
                + (i64::from(q.nu) * Q / hc) * Q / hc
                + i64::from(q.damp);
            let u = u * sq / (sq + k);
            flow.u[i] = u;
            flow.hu[i] = (h * u) >> 16;
        }
        roller(flow, subs);
    }
}

/// The speed a column's flow has where it crosses an edge over a step in the
/// bed, when only depth `he` of its depth `h` reaches over the step: its whole
/// discharge `hu` passes there, at most as fast as a wave over the step `ce`
/// (or as the column itself, if that is faster).
#[inline]
fn step_speed(he: i64, h: i64, hu: i64, u: i64, ce: i64) -> i64 {
    if he >= h || he <= H_EPS {
        return u;
    }
    let cap = u.abs().max(ce);
    (hu * Q / he).clamp(-cap, cap)
}

/// Steps every pipe over one substep: its flow changes with the difference
/// between the heads at its ends, less the head loss of its openings and the
/// drag of its walls (taken implicitly, so it is stable however short the
/// pipe), and the same flow passes every edge of it. The columns at its ends
/// see a wall where the pipe meets them, and the one it flows into takes the
/// momentum of the jet.
fn step_pipes(flow: &mut Flow, lq: LiqFast, subs: i64) {
    for p in &mut flow.pipes {
        if p.ends.contains(&PipeEnd::Wall) {
            p.q = 0;
        } else {
            let head = |side: usize| match p.ends[side] {
                PipeEnd::Node(i) => (flow.z[i] + flow.h[i]).max(p.mid[side]),
                _ => p.mid[side],
            };
            let dh = head(0) - head(1);
            let drive = ((GQ * dh) >> 16) * Q / p.inert;
            let (a, qa) = (p.amin, p.q.abs());
            // Head loss (1 + K) q |q| / (2 a^2 inertance), as a rate on q.
            let loss = (i128::from(p.loss) * i128::from(qa) * i128::from(Q * Q)
                / (2 * i128::from(a) * i128::from(a) * i128::from(p.inert)).max(1))
                as i64;
            // The drag of a column as deep as the pipe is high.
            let ua = qa * Q / a;
            let drag = ((i64::from(lq.cf) * ua) >> 16) * Q / a
                + (i64::from(lq.nu) * Q / a) * Q / a
                + i64::from(lq.damp);
            let mut qn = (p.q * subs + drive) * Q / (subs * Q + loss + drag);
            // Nothing flows in over a brink.
            if p.ends[0] == PipeEnd::Brink {
                qn = qn.min(0);
            }
            if p.ends[1] == PipeEnd::Brink {
                qn = qn.max(0);
            }
            p.q = qn.clamp(-U_CAP * a / Q * 4, U_CAP * a / Q * 4);
        }
        let qp = p.q;
        flow.flux[p.e0..=p.e1].fill(qp / subs);
        let jet = |open: i64| (qp * Q / open).clamp(-U_CAP, U_CAP);
        if let PipeEnd::Node(l) = p.ends[0] {
            let v = if qp > 0 { flow.u[l] } else { jet(p.open[0]) };
            flow.ml[p.e0] = press(flow.h[l]) + ((qp * v) >> 16);
        }
        if let PipeEnd::Node(r) = p.ends[1] {
            let v = if qp > 0 { jet(p.open[1]) } else { flow.u[r] };
            flow.mr[p.e1] = press(flow.h[r]) + ((qp * v) >> 16);
        }
    }
}

/// The turbulent roller of a bore. The shallow water equations steepen a
/// bore into a step, which in cells is a wall of liquid hundreds of rows
/// tall; a real bore's front is a breaking roller as long as the liquid is
/// deep. So where columns close on each other, momentum mixes between them
/// with an eddy viscosity `(depth / 4)^2 * closing speed` (a mixing length a
/// quarter of the depth), the closing speed taken as the fastest within two
/// edges so the roller covers the toe of the bore as well. The mixing is
/// implicit, so it is stable however strong, and moves no liquid: a bore
/// keeps its height and speed and only its front slopes. Smooth flow, where
/// columns draw apart, is untouched.
fn roller(flow: &mut Flow, subs: i64) {
    let n = flow.nodes.len();
    let (h, z, u) = (&flow.h, &flow.z, &flow.u);
    // Closing speed through each edge liquid crosses; -1 where none does.
    flow.rc.clear();
    flow.rc.resize(n + 1, -1);
    let mut any = false;
    for k in 1..n {
        let (a, b) = (k - 1, k);
        if flow.pipe_of[k] != NO_PIPE {
            continue;
        }
        let zf = z[a].max(z[b]);
        if (z[a] + h[a] - zf).min(z[b] + h[b] - zf) <= H_EPS {
            continue;
        }
        let close = u[a] - u[b];
        flow.rc[k] = close.max(0);
        any |= close > ROLLER_MIN;
    }
    if !any {
        return;
    }
    flow.ra.clear();
    flow.ra.resize(n + 1, 0);
    for k in 1..n {
        if flow.rc[k] < 0 {
            continue;
        }
        let close =
            flow.rc[k.saturating_sub(2)..(k + 3).min(n + 1)].iter().copied().max().unwrap_or(0);
        if close <= ROLLER_MIN {
            continue;
        }
        let (a, b) = (k - 1, k);
        let zf = z[a].max(z[b]);
        let he = (z[a] + h[a] - zf).min(z[b] + h[b] - zf);
        let l = he >> ROLLER_SHIFT;
        // Only where the closing speed over a mixing length is a fair part of
        // the wave speed: a bore. Gentle sloshing is left alone. Nor does the
        // roller act across a step in the bed taller than its mixing length:
        // flow onto a cliff deep under the surface speeds up over it, and
        // mixing it with the slow deep flow behind would hold it back until
        // the liquid piled up at the cliff.
        if ((l * close) >> 16) * ROLLER_FROUDE < wave(he) || (z[a] - z[b]).abs() > l {
            continue;
        }
        let nu = (((l * l) >> 16) * close) >> 16;
        let a8 = (((nu >> 8) * (he >> 8)) >> 8) / subs;
        flow.ra[k] = a8.clamp(0, ROLLER_CAP);
    }
    flow.rm.resize(n, 0);
    flow.rr.resize(n, 0);
    // Each run of coupled columns is a tridiagonal system in the new speeds,
    // solved by elimination left to right and substitution back. Masses are
    // in 1/256 cells, momenta in 1/2^24.
    let mut s = 0;
    while s < n {
        if s + 1 >= n || flow.ra[s + 1] == 0 {
            s += 1;
            continue;
        }
        let mut e = s + 1;
        while e + 1 < n && flow.ra[e + 1] > 0 {
            e += 1;
        }
        let mut sigma = 0i64;
        let mut r = (flow.h[s] >> 8) * flow.u[s];
        for i in s..=e {
            let hq = flow.h[i] >> 8;
            let next = if i < e { flow.ra[i + 1] } else { 0 };
            let m = hq + sigma + next;
            flow.rm[i] = m;
            flow.rr[i] = r;
            if i < e {
                let w = (next << 16) / m;
                sigma = (w * (hq + sigma)) >> 16;
                let carry = ((i128::from(w) * i128::from(r)) >> 16) as i64;
                r = (flow.h[i + 1] >> 8) * flow.u[i + 1] + carry;
            }
        }
        let mut x = 0;
        for i in (s..=e).rev() {
            let carry = if i < e { flow.ra[i + 1] * x } else { 0 };
            x = ((flow.rr[i] + carry) / flow.rm[i]).clamp(-U_CAP, U_CAP);
            flow.u[i] = x;
            flow.hu[i] = (flow.h[i] * x) >> 16;
        }
        s = e + 1;
    }
}

/// Moves the cells to match the solver: rounds the liquid moved through each
/// edge to whole cells, then carries top cells from the columns that lost
/// liquid to the tops of the columns that gained it, and over brinks.
fn realize(
    grid: &mut Grid,
    flow: &mut Flow,
    kind: &Kind<'_, '_>,
    ends: Ends,
    stats: &mut FlowStats,
) {
    let n = flow.nodes.len();
    let ctx = kind.ctx;
    let [left_brink, right_brink] = ends.brink;
    // Whole cells through each edge, rounded stochastically and kept within
    // what each column holds. Every edge of a pipe rounds the same way, so
    // its full columns pass on exactly what they take in.
    flow.tr.clear();
    flow.cur.clear();
    for d in &flow.nodes {
        flow.cur.push(d.depth());
    }
    for k in 0..=n {
        let p = flow.pipe_of[k];
        let key = if p == NO_PIPE { k } else { flow.pipes[p as usize].e0 };
        let anchor = if key < n { flow.nodes[key].x } else { flow.nodes[n - 1].x + 1 };
        let bed = flow.nodes[key.min(n - 1)].bed;
        // Liquid leaving over a brink or a step into the air is rounded
        // along an even sequence, so a trickle leaves at a steady rhythm and
        // falls as a line; rounded at random, it left in bursts with gaps of
        // up to ten ticks between them, and fell as dashes and drops.
        let pours = p == NO_PIPE && (k == 0 || k == n || flow.spill[k] != 0);
        let r = i64::from(if pours {
            ((ctx.tick.wrapping_mul(GOLDEN) >> 48) as u32)
                .wrapping_add(ctx.rand_still(anchor, bed, SALT_ROUND))
                & 0xffff
        } else {
            ctx.rand(anchor, bed, SALT_ROUND) & 0xffff
        });
        let mut t = (flow.moved[k] + r).div_euclid(Q);
        if (k == 0 && !left_brink) || (k == n && !right_brink) {
            t = 0;
        }
        if k == 0 {
            t = t.min(0);
        }
        if k == n {
            t = t.max(0);
        }
        flow.tr.push(t);
    }
    // Rightward transfers left to right and leftward ones right to left, so a
    // column passes on what it takes in on the same tick.
    for k in 0..=n {
        let t = flow.tr[k];
        if t > 0 && k > 0 {
            let t = t.min(i64::from(flow.cur[k - 1]));
            flow.cur[k - 1] -= t as i32;
            if k < n {
                flow.cur[k] += t as i32;
            }
            flow.tr[k] = t;
        }
    }
    for k in (0..=n).rev() {
        let t = flow.tr[k];
        if t < 0 && k < n {
            let t = (-t).min(i64::from(flow.cur[k]));
            flow.cur[k] -= t as i32;
            if k > 0 {
                flow.cur[k - 1] += t as i32;
            }
            flow.tr[k] = -t;
        }
    }
    // Current top row of each column as cells move.
    flow.cur.clear();
    for d in &flow.nodes {
        flow.cur.push(d.top);
    }
    let mut moves = 0u32;
    let mut poured = 0u32;
    // Rightward transfers, left to right, then leftward ones, right to left.
    for dir in [1i32, -1] {
        flow.stack.clear();
        for step in 0..n {
            let i = if dir > 0 { step } else { n - 1 - step };
            let out_edge = if dir > 0 { i + 1 } else { i };
            let leave = (flow.tr[out_edge] * i64::from(dir)).max(0) as usize;
            let arrive = flow.stack.len();
            if arrive > leave {
                let mut landed = 0usize;
                for _ in 0..arrive - leave {
                    let Some(&from) = flow.stack.last() else { break };
                    let d = flow.nodes[i];
                    let to = CellPos::new(d.x, flow.cur[i] - 1);
                    match swap(grid, &mut flow.writes, kind, from, to) {
                        Swap::Blocked => break,
                        Swap::Moved => {
                            flow.stack.pop();
                            moves += 1;
                        }
                        Swap::Landed => landed += 1,
                    }
                    flow.cur[i] -= 1;
                }
                // Cells the column found no room for go back where they came
                // from. Passed on, they were stacked on the next column with
                // room however high: a column under a floating crate with one
                // free cell took in what the solver sent it, and the rest
                // stood beside the crate as a tower ten cells high. Cells
                // that liquid landing on the column stood in for pass on, as
                // the flow carries what lands: sent back, they held the foot
                // of a waterfall up and its sheet broke into strands.
                while flow.stack.len() > leave + landed {
                    let Some(p) = flow.stack.pop() else { break };
                    restore_top(grid, flow, kind, p);
                }
            } else {
                for _ in 0..leave - arrive {
                    let d = flow.nodes[i];
                    if flow.cur[i] >= d.bed {
                        break;
                    }
                    flow.stack.push(CellPos::new(d.x, flow.cur[i]));
                    flow.cur[i] += 1;
                }
            }
            // Over a step the cells crossing the edge are poured into the
            // air past the lip; what finds no room there flows on onto the
            // column past the step.
            if (1..n).contains(&out_edge)
                && i32::from(flow.spill[out_edge]) == dir
                && !flow.stack.is_empty()
            {
                poured += pour(grid, flow, kind, i, dir, out_edge);
            }
        }
        let brink = if dir > 0 { right_brink } else { left_brink };
        if brink && !flow.stack.is_empty() {
            let i = if dir > 0 { n - 1 } else { 0 };
            poured += pour(grid, flow, kind, i, dir, if dir > 0 { n } else { 0 });
        }
        // Cells that found no place stay where they are.
        while let Some(p) = flow.stack.pop() {
            restore_top(grid, flow, kind, p);
        }
    }
    // Liquid drained through a hole under a column leaves below the hole.
    for i in 0..n {
        if flow.nodes[i].drain && flow.dr[i] > 0 {
            poured += drain(grid, flow, kind, i);
        }
    }
    stats.moves += moves;
    stats.poured += poured;
    // New flow speeds on the new top cells; old top cells become plain.
    for i in 0..n {
        let d = flow.nodes[i];
        let top = flow.cur[i];
        if d.wet() && top != d.top {
            let p = CellPos::new(d.x, d.top);
            let c = grid.get(p);
            if kind.rest(c) && (c.value != 0 || c.has(Cell::FLAG_LIQ_LEFT)) {
                put(grid, &mut flow.writes, p, liquid::at_rest(c));
            }
        }
        if top >= d.bed {
            continue;
        }
        let h = flow.h[i];
        let u = if h > H_EPS { (flow.hu[i] * Q / h).clamp(-U_CAP, U_CAP) } else { 0 };
        let mut u12 = (u / 16) as i32;
        // Slow flow between flat edges stops, and so does slow flow in a film
        // one or two cells thin. Flow through a full pipe goes on.
        let lf = if i == 0 { !left_brink } else { flow.flat[i] };
        let rf = if i == n - 1 { !right_brink } else { flow.flat[i + 1] };
        if !d.capped
            && ((lf && rf && u12.abs() <= U_REST) || (d.bed - top <= THIN && u12.abs() < U_STICK))
        {
            u12 = 0;
        }
        let p = CellPos::new(d.x, top);
        let c = grid.get(p);
        if !kind.rest(c) {
            continue;
        }
        let nc = if kind.q.keeps_speed() {
            liquid::with_flow_speed(c, u12, ctx.rand(d.x, top, SALT_CODE))
        } else {
            c
        };
        if nc != c {
            put(grid, &mut flow.writes, p, nc);
        }
        if liquid::flow_speed(nc) != 0 || top != d.top {
            grid.keep(p);
        }
        flow.nodes[i].u = liquid::flow_speed(nc);
    }
    // The columns as they are now, for the settle moves that follow.
    for i in 0..n {
        let x = flow.nodes[i].x;
        let top = flow.cur[i];
        flow.nodes[i].top = top;
        if top >= flow.nodes[i].bed {
            flow.nodes[i].u = 0;
        }
        flow.nodes[i].capped = !kind.clear(grid.get(CellPos::new(x, top - 1)));
    }
    // Keep a body that is still out of balance under watch even if no cell
    // moved and no speed survived rounding. Every column is kept, so a puddle
    // that the body stops reaching becomes a body of its own and settles.
    for i in 0..n {
        if flow.cur[i] < flow.nodes[i].bed {
            grid.keep(CellPos::new(flow.nodes[i].x, flow.cur[i]));
        }
    }
}

/// A cell taken from a column for a transfer that found no place goes back on
/// its column: the column's top row rises again. If cells taken from under it
/// have moved on, it drops onto what is left, so the column has no gap.
fn restore_top(grid: &mut Grid, flow: &mut Flow, kind: &Kind<'_, '_>, p: CellPos) {
    // A body's columns are side by side, left to right.
    let i = flow.nodes.first().map_or(usize::MAX, |d| p.x.wrapping_sub(d.x) as usize);
    if flow.nodes.get(i).is_none_or(|d| d.x != p.x) {
        return;
    }
    let top = flow.cur[i];
    if top == p.y + 1 {
        flow.cur[i] = p.y;
    } else if top > p.y + 1
        && swap(grid, &mut flow.writes, kind, p, CellPos::new(p.x, top - 1)) == Swap::Moved
    {
        flow.cur[i] = top - 1;
    }
}

/// Pours the cells on the stack over the brink or step beside column `i`,
/// in direction `dir`, past edge `edge`: each becomes liquid in the air past
/// the lip. A sheet off a brink or a step leaves at the speed that carries
/// what the edge passed this tick through the column's depth, rounded to
/// whole cells per tick, and fills the block of cells it crossed in the tick
/// from the lip up, as many columns out as it moves; the block's top is level
/// with the column's surface, and blocks poured on successive ticks lie side
/// by side as one sheet. Where liquid or a floor lies at most [`CLING`] rows
/// below the lip, the cells first fill the space down there, at rest: the
/// sheet hugs the face and lands at once. The end of a pipe pours a jet as
/// thick as the opening contracts to and as fast as the flow through that,
/// which is the Torricelli speed for the head: [`JET_CONTRACT`]. Cells poured
/// on successive ticks keep the speed of those poured before while it is
/// close, so they fly along one path. Returns how many it poured.
fn pour(
    grid: &mut Grid,
    flow: &mut Flow,
    kind: &Kind<'_, '_>,
    i: usize,
    dir: i32,
    edge: usize,
) -> u32 {
    let d = flow.nodes[i];
    let pipe = flow.pipe_of[edge];
    let jet = pipe != NO_PIPE;
    let (u, rows) = if jet {
        let p = &flow.pipes[pipe as usize];
        let open = p.open[usize::from(dir > 0)];
        let jet = (open * JET_CONTRACT / Q).max(Q);
        let rows = (open / Q).clamp(1, i64::from(BRINK_ROWS)) as i32;
        ((p.q.abs() * Q / jet).min(U_CAP), rows)
    } else {
        // The sheet is as deep as the column and leaves at the speed that
        // carries what the edge passed this tick through that depth, so its
        // top is level with the column's surface.
        let rows = d.depth().clamp(1, BRINK_ROWS);
        (flow.moved[edge].abs() / i64::from(rows), rows)
    };
    let raw = ((u >> 12) as i32).clamp(4, 160);
    // The speed of the liquid poured just before, if it is still beside the lip.
    let near = ((raw + 15) / 16).clamp(1, MAX_LIP) + 1;
    let prev = (1..=rows).find_map(|r| {
        (1..=near).find_map(|k| {
            let c = grid.get(CellPos::new(d.x + dir * k, d.bed - r));
            let vx = liquid::air_velocity(c).0;
            (kind.falling(c) && vx * dir > 0).then_some(vx.abs())
        })
    });
    let vx = if jet { liquid::pour_speed(raw, prev) } else { liquid::sheet_speed(raw, prev) };
    let mut poured = 0;
    if !jet {
        while let Some(to) = kind.cling_cell(grid, d.x + dir, d.bed) {
            if !pour_one(grid, flow, kind, to, liquid::at_rest) {
                break;
            }
            poured += 1;
        }
        poured += top_up(grid, flow, kind, i, dir, edge);
    }
    // A jet fills the block as far out as it reaches in a tick; a sheet,
    // moving whole cells per tick, as many columns as it moves on every tick.
    // Liquid poured into the first column past the lip leaves with a slow
    // fall; further out it has the fall speed and the drop that its path
    // from the first column gives there, so it is where the liquid that left
    // the lip earlier in the tick would be, and every cell of the stream
    // follows one path. Poured further out with the speed of the first
    // column, it skipped the places in between where the path steps down a
    // row or gains speed, and the lanes of a jet set out a row or a code
    // apart.
    let reach = ((vx + 15) / 16).clamp(1, MAX_LIP) as usize;
    let mut drop = [0i32; MAX_LIP as usize];
    let mut fall = [POUR_FALL; MAX_LIP as usize];
    let key0 = i64::from((d.x + dir) * dir);
    for j in 1..reach {
        let (dy, v) = liquid::path_column(kind.ctx, fall[j - 1], vx, key0 + j as i64);
        drop[j] = drop[j - 1] + dy;
        fall[j] = v;
    }
    while let Some((to, j)) = kind.lip_cell(grid, d.x, dir, d.bed, rows, &drop[..reach]) {
        let vy = fall[j - 1];
        let fly = |c: Cell| liquid::airborne(liquid::at_rest(c), vx * dir, vy, NEAREST);
        if !pour_one(grid, flow, kind, to, fly) {
            break;
        }
        poured += 1;
    }
    poured
}

/// Liquid falling over the step at `edge`, past column `i` in direction
/// `dir`, into tail water that lies near the lip or over it, first fills the
/// column at the foot of the step at rest up to the level of the tail water
/// beyond it; only what is left is poured into the air as a sheet. The
/// liquid crossing such a step plunges into the tail water rather than
/// falling through air. Poured over the column at the foot as a sheet that
/// flew on past it, it left that column fed by nothing, and the column sank
/// into a notch a few cells wide between the step's face and the tail water,
/// or a slit one cell wide. Returns how many cells it placed.
fn top_up(
    grid: &mut Grid,
    flow: &mut Flow,
    kind: &Kind<'_, '_>,
    i: usize,
    dir: i32,
    edge: usize,
) -> u32 {
    let n = flow.nodes.len();
    if edge == 0 || edge >= n {
        return 0;
    }
    let lip = flow.nodes[i].bed;
    let (Some(b), Some(t)) =
        (i.checked_add_signed(dir as isize), i.checked_add_signed(2 * dir as isize))
    else {
        return 0;
    };
    if t >= n || !flow.nodes[t].wet() {
        return 0;
    }
    let tail = flow.cur[t];
    if tail >= lip + CLING {
        return 0;
    }
    let x = flow.nodes[b].x;
    let mut placed = 0;
    while flow.cur[b] > tail && flow.cur[b] > flow.nodes[i].top {
        let to = CellPos::new(x, flow.cur[b] - 1);
        if !kind.room(grid.get(to)) || !pour_one(grid, flow, kind, to, liquid::at_rest) {
            break;
        }
        flow.cur[b] -= 1;
        placed += 1;
    }
    placed
}

/// Pours what column `i` drained through the hole under it this tick into
/// the open space below the hole, as a stream falling at the Torricelli speed
/// for the column's depth: into the cells the stream crosses in a tick, from
/// the hole down, each cell with the fall speed that its path from the hole
/// gives there, so the cells of the stream fall alike. The cells come off
/// the top of the column, which is the same as the column moving down a cell
/// and the bottom one leaving. The liquid drained is rounded to whole cells
/// along a sequence that spreads the rounding up evenly over the ticks, so a
/// thin stream leaves the hole at a steady rhythm rather than in bursts.
/// Returns how many cells it poured.
fn drain(grid: &mut Grid, flow: &mut Flow, kind: &Kind<'_, '_>, i: usize) -> u32 {
    let d = flow.nodes[i];
    let tick = kind.ctx.tick;
    let r = ((tick.wrapping_mul(GOLDEN) >> 48) as u32)
        .wrapping_add(kind.ctx.rand_still(d.x, d.bed, SALT_DRAIN))
        & 0xffff;
    let mut t = (flow.dr[i] + i64::from(r)).div_euclid(Q);
    if t <= 0 {
        return 0;
    }
    let h = i64::from(d.depth().max(1)) * Q;
    let v16 = (wave(2 * h) * 16 / Q).clamp(1, 255) as i32;
    let mut vy = liquid::nearest_fall(v16);
    let reach = ((v16 + 15) / 16).clamp(1, 16);
    let mut poured = 0;
    for k in 0..reach {
        if t <= 0 || flow.cur[i] >= d.bed {
            break;
        }
        let to = CellPos::new(d.x, d.bed + k);
        let dst = grid.get(to);
        if k > 0 {
            vy = liquid::fall_row(kind.ctx, vy, d.bed + k);
        }
        if !kind.room(dst) {
            break;
        }
        let from = CellPos::new(d.x, flow.cur[i]);
        let c = grid.get(from);
        if !kind.rest(c) {
            break;
        }
        let fly = liquid::as_stream(liquid::airborne(liquid::at_rest(c), 0, vy, NEAREST));
        put(grid, &mut flow.writes, to, fly.with_parity(tick));
        put(grid, &mut flow.writes, from, if dst.is_air() { dst } else { dst.with_parity(tick) });
        flow.cur[i] += 1;
        t -= 1;
        poured += 1;
    }
    poured
}

/// Moves the cell on top of the stack to `to` as `make` makes it, and what
/// was at `to` back to where the cell came from. False if the stack is empty
/// or the cell is no longer resting liquid of the body's kind.
fn pour_one(
    grid: &mut Grid,
    flow: &mut Flow,
    kind: &Kind<'_, '_>,
    to: CellPos,
    make: impl Fn(Cell) -> Cell,
) -> bool {
    let Some(&from) = flow.stack.last() else { return false };
    let c = grid.get(from);
    if !kind.rest(c) {
        return false;
    }
    let dst = grid.get(to);
    put(grid, &mut flow.writes, to, make(c).with_parity(kind.ctx.tick));
    put(
        grid,
        &mut flow.writes,
        from,
        if dst.is_air() { dst } else { dst.with_parity(kind.ctx.tick) },
    );
    flow.stack.pop();
    true
}
