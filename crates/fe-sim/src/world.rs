use crate::flow::{self, Flow};
use crate::grid::Grid;
use crate::liquid::{self, LiqFast};
use crate::rules::{self, Event, JobOut, RuleCtx, Sweep};
use crate::sched::{self, Chunk, HEAD_MAX};
use fe_core::{CHUNK_SIZE, Cell, CellPos, ChunkPos, IRect, LocalRect, MaterialId, StableHasher};
use fe_materials::{MaterialSet, State};
use rayon::prelude::*;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

/// Counters from the most recent cell step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CellStats {
    /// Chunks that had a non-empty dirty rect.
    pub active_chunks: u32,
    /// Loaded chunks.
    pub loaded_chunks: u32,
    /// Cells visited by the rules.
    pub visited_cells: u64,
    /// Liquid cells settle moves carried down gentle steps in bodies at rest.
    pub settle_moves: u32,
    /// Liquid cells the pressure pass moved through their body to a lower opening.
    pub pressure_moves: u32,
    /// Liquid cells thrown up as splashes.
    pub splashes: u32,
    /// Bodies of liquid the flow phase stepped.
    pub flow_bodies: u32,
    /// Columns in those bodies.
    pub flow_columns: u32,
    /// Liquid cells the flow phase moved from one column to another.
    pub flow_moves: u32,
    /// Liquid cells the flow phase poured over brinks.
    pub poured: u32,
}

/// Pressure candidates handled per tick.
const PRESSURE_BUDGET: usize = 16;
/// Cells the pressure search may take out of its queue.
const BFS_LIMIT: usize = 4096;
/// Half size of the pressure search window: it spans 257x257 cells.
const BFS_HALF: i32 = 128;
const BFS_SIDE: usize = (2 * BFS_HALF + 1) as usize;

/// Scratch space for the pressure search, kept between ticks.
#[derive(Default)]
struct Bfs {
    seen: Vec<u64>,
    queue: VecDeque<(i32, i32)>,
}

/// Reads cells outside the passes, remembering the last chunk it looked in.
struct Reader<'g> {
    grid: &'g Grid,
    pos: ChunkPos,
    chunk: Option<&'g Chunk>,
}

impl<'g> Reader<'g> {
    fn new(grid: &'g Grid) -> Self {
        Self { grid, pos: ChunkPos::new(i32::MIN, i32::MIN), chunk: None }
    }

    #[inline]
    fn get(&mut self, x: i32, y: i32) -> Cell {
        let p = CellPos::new(x, y);
        let cp = p.chunk();
        if cp != self.pos {
            self.pos = cp;
            self.chunk = self.grid.chunk(cp);
        }
        self.chunk.map_or(Cell::BOUNDARY, |c| c.get(p.local_index()))
    }
}

/// The cell world: material set, chunks, seed, tick, and the thread pool.
pub struct World {
    mats: Arc<MaterialSet>,
    fast: Vec<rules::Fast>,
    liq: Vec<LiqFast>,
    /// Loaded chunks.
    pub grid: Grid,
    seed: u64,
    tick: u64,
    threads: usize,
    pool: Option<rayon::ThreadPool>,
    stats: CellStats,
    wind: i8,
    /// Grid generation the head tables were last built for.
    heads_gen: Option<u64>,
    order: Vec<ChunkPos>,
    bfs: Bfs,
    flow: Flow,
}

impl World {
    /// Creates an empty world. `threads` of 0 or 1 runs passes on the calling thread.
    pub fn new(mats: Arc<MaterialSet>, seed: u64, threads: usize) -> Self {
        let fast = rules::fast_table(&mats);
        let liq = liquid::liquid_table(&mats);
        let mut w = Self {
            mats,
            fast,
            liq,
            grid: Grid::default(),
            seed,
            tick: 0,
            threads: 1,
            pool: None,
            stats: CellStats::default(),
            wind: 0,
            heads_gen: None,
            order: Vec::new(),
            bfs: Bfs::default(),
            flow: Flow::default(),
        };
        w.set_threads(threads);
        w
    }

    /// Changes the number of worker threads. Results do not depend on this value.
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.max(1);
        self.pool = if self.threads > 1 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.threads)
                .thread_name(|i| format!("fe-sim-{i}"))
                .build()
                .ok()
        } else {
            None
        };
    }

    /// Worker thread count.
    pub const fn threads(&self) -> usize {
        self.threads
    }

    /// The material set.
    pub fn materials(&self) -> &Arc<MaterialSet> {
        &self.mats
    }

    /// The world seed.
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// The current tick: the number of steps run so far.
    pub const fn tick(&self) -> u64 {
        self.tick
    }

    /// Sets the tick counter, used when loading a snapshot.
    pub const fn set_tick(&mut self, tick: u64) {
        self.tick = tick;
    }

    /// Wind from -4 (strong, to the left) to 4 (strong, to the right). Gases and fire drift with it.
    pub const fn wind(&self) -> i8 {
        self.wind
    }

    /// Sets the wind, clamped to -4..=4.
    pub fn set_wind(&mut self, w: i8) {
        self.wind = w.clamp(-4, 4);
    }

    /// Counters from the last [`World::step_cells`].
    pub const fn stats(&self) -> CellStats {
        self.stats
    }

    /// Creates every chunk overlapping `rect` that is not already loaded, filled with `fill`.
    pub fn ensure_rect(&mut self, rect: IRect, fill: Cell) {
        if rect.is_empty() {
            return;
        }
        let (a, b) = rect.chunk_range();
        for cy in a.y..b.y {
            for cx in a.x..b.x {
                let p = ChunkPos::new(cx, cy);
                if !self.grid.contains(p) {
                    self.grid.insert(Chunk::new(p, fill));
                }
            }
        }
    }

    /// Reads a cell. Unloaded cells read as the boundary.
    #[inline]
    pub fn get(&self, p: CellPos) -> Cell {
        self.grid.get(p)
    }

    /// Creates a fresh cell of a material at a position, with the material's defaults.
    pub fn make_cell(&self, id: MaterialId, p: CellPos) -> Cell {
        let r = fe_core::cell_rand(self.seed, self.tick, p.x, p.y, 0x5e7);
        rules::new_cell(&self.mats, id, None, r)
    }

    /// Writes a cell, marks it dirty, and returns any support event the change causes.
    ///
    /// Use this for edits made outside the cell rules: tools, explosions, scene loading.
    pub fn set(&mut self, p: CellPos, cell: Cell) -> Option<Event> {
        let old = self.grid.get(p);
        if old.material() == MaterialId::BOUNDARY
            && old == Cell::BOUNDARY
            && !self.grid.contains(p.chunk())
        {
            return None;
        }
        self.grid.set(p, cell);
        let om = self.mats.get(old.material());
        let nm = self.mats.get(cell.material());
        if (om.state == State::Static || nm.state == State::Static)
            && let Some(c) = self.grid.chunk(p.chunk())
        {
            c.flag_solid_changed();
        }
        let was = om.state == State::Static && om.structural && !old.has(Cell::FLAG_IN_BODY);
        let is = nm.state == State::Static && nm.structural;
        (was && !is).then_some(Event::SupportLost(p))
    }

    /// Fills a rectangle with fresh cells of one material. Returns support events.
    pub fn fill_rect(&mut self, rect: IRect, id: MaterialId) -> Vec<Event> {
        let mut events = Vec::new();
        for y in rect.min.y..rect.max.y {
            for x in rect.min.x..rect.max.x {
                let p = CellPos::new(x, y);
                let c = self.make_cell(id, p);
                events.extend(self.set(p, c));
            }
        }
        events
    }

    /// Runs one tick of the cell rules: the head tables, four checkerboard
    /// passes over dirty chunks, the fall sweep, the flow phase, which moves
    /// resting liquid as shallow water, then the pressure pass.
    ///
    /// Returns the events the rules produced, in canonical order. Does not advance
    /// the tick counter; the caller does that once the whole tick is done.
    pub fn step_cells(&mut self) -> Vec<Event> {
        self.grid.advance();
        self.refresh_heads();
        let mut groups: [Vec<(ChunkPos, LocalRect)>; 4] = Default::default();
        let mut active = 0u32;
        for c in self.grid.chunks() {
            let d = c.dirty();
            if !d.is_empty() {
                groups[c.pos().group()].push((c.pos(), d));
                active += 1;
            }
        }
        let ctx = RuleCtx::new(&self.mats, &self.fast, &self.liq, self.seed, self.tick)
            .with_wind(self.wind);
        let mut events = Vec::new();
        let mut visited = 0u64;
        let mut tops = Vec::new();
        let mut pressure = Vec::new();
        let mut splashes = 0u32;
        // Which neighbor goes first flips every tick across vertical chunk
        // borders and every two ticks across horizontal ones, so a flow that
        // crosses a border is updated front first on half of the ticks
        // whichever way it runs.
        let (fx, fy) = ((self.tick & 1) as usize, ((self.tick >> 1) & 1) as usize);
        let mut fall: Vec<(ChunkPos, LocalRect)> = Vec::new();
        let mut merge = |o: JobOut, visited: &mut u64, events: &mut Vec<Event>| {
            *visited += o.visited;
            events.extend(o.events);
            tops.extend(o.tops);
            pressure.extend(o.pressure);
            splashes += o.splashes;
        };
        for k in 0..4 {
            let jobs = &groups[((k & 1) ^ fx) | ((((k >> 1) & 1) ^ fy) << 1)];
            if jobs.is_empty() {
                continue;
            }
            let mut outs: Vec<JobOut> = (0..jobs.len()).map(|_| JobOut::default()).collect();
            sched::run_pass(
                &mut self.grid,
                jobs,
                &mut outs,
                self.pool.as_ref(),
                |nb, rect, out| {
                    rules::update_chunk(nb, rect, &ctx, out, Sweep::Main);
                },
            );
            for (o, &job) in outs.into_iter().zip(jobs) {
                if o.falling {
                    fall.push(job);
                }
                merge(o, &mut visited, &mut events);
            }
        }
        // The fall sweep: liquid in the air, one chunk row at a time from the
        // bottom up, each row in two halves by chunk x parity so jobs that run
        // together stay two chunks apart, the half that goes first flipping
        // every tick like the passes. Liquid flying across a chunk border
        // into the half that runs later would meet the liquid ahead of it
        // there not yet moved, so it waits (`liquid::waits`) and the chunks
        // where any liquid waited run again, in the same order, after both
        // halves. With one round, a dense jet or sheet piled up against every
        // other border and came out past it thinned into bars. Where liquid
        // in a chunk's top row is in the air, the columns above it in the
        // chunk above join the sweep too, even if that chunk slept, so a mass
        // whose support falls away falls as one piece instead of tearing at
        // chunk borders.
        let mut rows: BTreeMap<i32, Vec<(ChunkPos, LocalRect)>> = BTreeMap::new();
        for job in fall {
            rows.entry(job.0.y).or_default().push(job);
        }
        while let Some((y, mut row)) = rows.pop_last() {
            row.sort_unstable_by_key(|(p, _)| (p.x & 1, p.x));
            row.dedup_by(|b, a| {
                let same = a.0 == b.0;
                if same {
                    a.1 = a.1.union(b.1);
                }
                same
            });
            let mut up = Vec::new();
            let first = fx as i32;
            let mut again: Vec<(ChunkPos, LocalRect)> = Vec::new();
            for round in [true, false] {
                for par in [first, 1 - first] {
                    let jobs: Vec<(ChunkPos, LocalRect)> = if round { &row } else { &again }
                        .iter()
                        .copied()
                        .filter(|(p, _)| p.x & 1 == par)
                        .collect();
                    if jobs.is_empty() {
                        continue;
                    }
                    let mut outs: Vec<JobOut> = (0..jobs.len())
                        .map(|_| JobOut { may_wait: round, ..JobOut::default() })
                        .collect();
                    sched::run_pass(
                        &mut self.grid,
                        &jobs,
                        &mut outs,
                        self.pool.as_ref(),
                        |nb, rect, out| {
                            rules::update_chunk(nb, rect, &ctx, out, Sweep::Fall);
                        },
                    );
                    for (o, &job) in outs.into_iter().zip(&jobs) {
                        let p = job.0;
                        if o.fall_up != 0 {
                            let x0 = o.fall_up.trailing_zeros() as u8;
                            let x1 = 63 - o.fall_up.leading_zeros() as u8;
                            let r = LocalRect { x0, y0: 0, x1, y1: LocalRect::FULL.y1 };
                            up.push((ChunkPos::new(p.x, y - 1), r));
                        }
                        if o.waited {
                            again.push(job);
                        }
                        merge(o, &mut visited, &mut events);
                    }
                }
                if again.is_empty() {
                    break;
                }
            }
            for job in up {
                if self.grid.chunk(job.0).is_some() {
                    rows.entry(y - 1).or_default().push(job);
                }
            }
        }
        let ctx = RuleCtx::new(&self.mats, &self.fast, &self.liq, self.seed, self.tick);
        let fs = flow::run(&mut self.grid, &mut self.flow, &ctx, tops);
        let pressure_moves = self.pressure_pass(pressure);
        self.stats = CellStats {
            active_chunks: active,
            loaded_chunks: self.grid.len() as u32,
            visited_cells: visited,
            settle_moves: fs.settled,
            pressure_moves,
            splashes,
            flow_bodies: fs.bodies,
            flow_columns: fs.columns,
            flow_moves: fs.moves,
            poured: fs.poured,
        };
        events
    }

    /// Rebuilds the column head tables from the cells as they are now.
    ///
    /// A chunk's bottom run is rebuilt when it has a dirty rect (every rule
    /// write that can change it marks the cell) or its cells were replaced.
    /// The head coming in from above is then summed down each column in
    /// canonical order, so the tables depend only on the cells.
    fn refresh_heads(&mut self) {
        let liq = &self.liq;
        let rebuild = |c: &mut Chunk| -> bool {
            let stale = c.take_tables_stale();
            if !stale && c.dirty().is_empty() {
                return false;
            }
            let (bottom, top) = column_runs(c, liq);
            c.set_runs(bottom, top)
        };
        let chunks = self.grid.chunk_slice_mut();
        let changed = match self.pool.as_ref() {
            Some(pool) if chunks.len() > 16 => {
                pool.install(|| chunks.par_iter_mut().map(rebuild).reduce(|| false, |a, b| a | b))
            }
            _ => chunks.iter_mut().map(rebuild).fold(false, |a, b| a | b),
        };
        let generation = self.grid.generation();
        if !changed && self.heads_gen == Some(generation) {
            return;
        }
        self.heads_gen = Some(generation);
        self.order.clear();
        self.order.extend(self.grid.positions());
        // Head comes down each column from the chunks above; canonical order
        // visits a chunk after the one above it.
        for &pos in &self.order {
            let mut h = [0u16; CHUNK_SIZE as usize];
            if let Some(a) = self.grid.chunk(ChunkPos::new(pos.x, pos.y - 1)) {
                for (x, h) in h.iter_mut().enumerate() {
                    let run = a.bottom_run()[x];
                    *h = if i32::from(run) == CHUNK_SIZE {
                        (CHUNK_SIZE as u32 + u32::from(a.head_in()[x])).min(HEAD_MAX) as u16
                    } else {
                        u16::from(run)
                    };
                }
            }
            if let Some(c) = self.grid.chunk_mut(pos) {
                c.set_head_in(h);
            }
        }
        // Depth comes up each column from the chunks below, in reverse order.
        for &pos in self.order.iter().rev() {
            let mut f = [0u16; CHUNK_SIZE as usize];
            if let Some(b) = self.grid.chunk(ChunkPos::new(pos.x, pos.y + 1)) {
                for (x, f) in f.iter_mut().enumerate() {
                    let run = b.top_run()[x];
                    *f = if i32::from(run) == CHUNK_SIZE {
                        (CHUNK_SIZE as u32 + u32::from(b.foot_in()[x])).min(HEAD_MAX) as u16
                    } else {
                        u16::from(run)
                    };
                }
            }
            if let Some(c) = self.grid.chunk_mut(pos) {
                c.set_foot_in(f);
            }
        }
    }

    /// The pressure pass, after the flow phase: levels connected vessels that
    /// the flow phase cannot see, such as a pool joined to another through a
    /// hole in its floor. It leaves alone liquid in bodies the flow phase
    /// stepped this tick, which is flowing already.
    ///
    /// The candidate list is sorted and deduplicated, and a window of its
    /// budget is handled, starting at an offset that turns with the tick so
    /// that every candidate gets its turn. Candidates outside the window are
    /// marked, so they are visited and offered again next tick, and a handled
    /// candidate that finds no lower opening gets the settled mark, so it stops
    /// asking until something moves it. That is what lets a level pool sleep.
    /// Every move lowers a cell.
    fn pressure_pass(&mut self, mut pressure: Vec<(i32, i32)>) -> u32 {
        pressure.sort_unstable();
        pressure.dedup();
        let mut pressure_moves = 0;
        let (start, n) = window(pressure.len(), PRESSURE_BUDGET, self.tick);
        // Liquid the flow phase is moving is left to it.
        let ctx = RuleCtx::new(&self.mats, &self.fast, &self.liq, self.seed, self.tick);
        let flowing: Vec<bool> = (0..n)
            .map(|j| {
                let (y, x) = pressure[(start + j) % pressure.len()];
                flow::flowing(&self.grid, &self.flow, &ctx, x, y)
            })
            .collect();
        for j in 0..pressure.len() {
            let (y, x) = pressure[(start + j) % pressure.len()];
            let p = CellPos::new(x, y);
            if j >= n {
                self.grid.keep(p);
            } else if flowing[j] {
                // Left to the flow phase.
            } else if self.pressure_one(x, y) {
                pressure_moves += 1;
            } else {
                let c = self.grid.get(p);
                if self.liq_kind(c) && c.value == 0 && !c.has(Cell::FLAG_LIQ_AIR) {
                    self.grid.set_quiet(p, liquid::settled_cell(c));
                }
            }
        }
        pressure_moves
    }

    /// True for a liquid that keeps its velocity in its cells.
    fn liq_kind(&self, c: Cell) -> bool {
        self.liq.get(c.material().index()).is_some_and(|q| q.is_liquid())
            && self.mats.get(c.material()).lifetime.is_none()
            && !c.has(Cell::FLAG_IN_BODY)
    }

    fn state(&self, c: Cell) -> State {
        if c.has(Cell::FLAG_IN_BODY) {
            return State::Static;
        }
        self.fast.get(c.material().index()).map_or(State::Static, |f| f.state)
    }

    fn open(&self, c: Cell) -> bool {
        matches!(self.state(c), State::Empty | State::Gas | State::Fire)
    }

    /// Can a cell of density `d` move into `c`?
    fn enterable(&self, c: Cell, d: i16) -> bool {
        if c.is_air() {
            return true;
        }
        if c.has(Cell::FLAG_IN_BODY) {
            return false;
        }
        self.fast.get(c.material().index()).is_some_and(|f| f.state.is_fluid() && f.density < d)
    }

    /// Pressure: a resting surface cell searches the body of resting liquid
    /// under it (breadth first, bounded) for an opening at least two rows
    /// lower that it would rest in, such as the surface of the other arm of a
    /// U-tube, and moves there. The search goes through no jet or stream, so
    /// liquid that a falling stream lands on is not part of the body above
    /// the stream. Returns true if it moved.
    fn pressure_one(&mut self, x: i32, y: i32) -> bool {
        let mut r = Reader::new(&self.grid);
        let c = r.get(x, y);
        if !self.liq_kind(c) {
            return false;
        }
        if !self.open(r.get(x, y - 1)) {
            return false;
        }
        // Resting liquid of this kind: the search goes through no stream or jet.
        let same = |b: Cell| {
            b.material() == c.material() && !b.has(Cell::FLAG_IN_BODY) && !liquid::is_airborne(b)
        };
        if !same(r.get(x, y + 1)) {
            return false;
        }
        let density = self.fast.get(c.material().index()).map_or(0, |f| f.density);
        let words = (BFS_SIDE * BFS_SIDE).div_ceil(64);
        let mut seen = std::mem::take(&mut self.bfs.seen);
        let mut queue = std::mem::take(&mut self.bfs.queue);
        seen.clear();
        seen.resize(words, 0);
        queue.clear();
        let bit = |px: i32, py: i32| -> Option<usize> {
            let (dx, dy) = (px - x, py - y);
            (dx.abs() <= BFS_HALF && dy.abs() <= BFS_HALF)
                .then(|| (dy + BFS_HALF) as usize * BFS_SIDE + (dx + BFS_HALF) as usize)
        };
        let visit = |seen: &mut Vec<u64>, px: i32, py: i32| -> bool {
            match bit(px, py) {
                Some(i) if seen[i / 64] & (1 << (i % 64)) == 0 => {
                    seen[i / 64] |= 1 << (i % 64);
                    true
                }
                _ => false,
            }
        };
        visit(&mut seen, x, y);
        visit(&mut seen, x, y + 1);
        queue.push_back((x, y + 1));
        let mut taken = 0;
        let mut found = None;
        'search: while let Some((qx, qy)) = queue.pop_front() {
            taken += 1;
            if taken > BFS_LIMIT {
                break;
            }
            for (dx, dy) in [(0, 1), (-1, 0), (1, 0), (0, -1)] {
                let (nx, ny) = (qx + dx, qy + dy);
                if !visit(&mut seen, nx, ny) {
                    continue;
                }
                let n = r.get(nx, ny);
                if self.open(n) {
                    if ny >= y + 2 && !self.enterable(r.get(nx, ny + 1), density) {
                        found = Some(CellPos::new(nx, ny));
                        break 'search;
                    }
                } else if same(n) {
                    queue.push_back((nx, ny));
                }
            }
        }
        self.bfs.seen = seen;
        self.bfs.queue = queue;
        let Some(t) = found else { return false };
        self.move_down(CellPos::new(x, y), t, c);
        true
    }

    /// Moves the resting cell `c` from `from` down to the open cell `to`, and
    /// what was at `to` (air or gas) up to `from`.
    fn move_down(&mut self, from: CellPos, to: CellPos, c: Cell) {
        let d = self.grid.get(to);
        let d = if d.is_air() { d } else { d.with_parity(self.tick) };
        self.grid.set(to, liquid::at_rest(c).with_parity(self.tick));
        self.grid.set(from, d);
    }

    /// Advances the tick counter by one.
    pub const fn finish_tick(&mut self) {
        self.tick += 1;
    }

    /// Hash of the tick, seed and every loaded cell.
    pub fn hash(&self) -> u64 {
        let mut h = StableHasher::new();
        h.write_u64(self.tick);
        h.write_u64(self.seed);
        h.write_i32(i32::from(self.wind));
        h.write_u64(self.mats.hash());
        h.write_u64(self.grid.hash());
        h.finish()
    }

    /// Counts cells of a material inside a rectangle.
    pub fn count(&self, rect: IRect, id: MaterialId) -> u64 {
        let mut n = 0;
        for y in rect.min.y..rect.max.y {
            for x in rect.min.x..rect.max.x {
                if self.grid.get(CellPos::new(x, y)).material() == id {
                    n += 1;
                }
            }
        }
        n
    }

    /// Potential energy of the liquid: the sum of `-y` over liquid cells, in
    /// cells. Every settle and pressure move lowers it.
    pub fn liquid_potential(&self) -> i64 {
        let mut e = 0i64;
        for c in self.grid.chunks() {
            let o = c.pos().origin();
            for i in 0..fe_core::CHUNK_AREA {
                if self.liq.get(c.get(i).material().index()).is_some_and(|q| q.is_liquid()) {
                    e -= i64::from(o.y + (i as i32 >> fe_core::CHUNK_SHIFT));
                }
            }
        }
        e
    }

    /// Counts every cell of each material across loaded chunks, indexed by material id.
    pub fn census(&self) -> Vec<u64> {
        let mut counts = vec![0u64; self.mats.len()];
        for c in self.grid.chunks() {
            for i in 0..fe_core::CHUNK_AREA {
                let id = c.get(i).material().index();
                if let Some(n) = counts.get_mut(id) {
                    *n += 1;
                }
            }
        }
        counts
    }
}

/// The window of a candidate list handled this tick: its start and length.
fn window(len: usize, budget: usize, tick: u64) -> (usize, usize) {
    if len == 0 {
        return (0, 0);
    }
    (((tick % len as u64) * budget as u64 % len as u64) as usize, budget.min(len))
}

/// Per column of a chunk, the runs of head liquid going up from the bottom
/// row and going down from the top row.
fn column_runs(
    c: &Chunk,
    liq: &[LiqFast],
) -> ([u8; CHUNK_SIZE as usize], [u8; CHUNK_SIZE as usize]) {
    let size = CHUNK_SIZE as usize;
    let run = |rows: &mut dyn Iterator<Item = usize>| {
        let mut run = [0u8; CHUNK_SIZE as usize];
        let mut alive = u64::MAX;
        for y in rows {
            let mut m = alive;
            while m != 0 {
                let x = m.trailing_zeros() as usize;
                m &= m - 1;
                if liquid::is_head(liq, c.get(y * size + x)) {
                    run[x] += 1;
                } else {
                    alive &= !(1u64 << x);
                }
            }
            if alive == 0 {
                break;
            }
        }
        run
    };
    (run(&mut (0..size).rev()), run(&mut (0..size)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world(threads: usize) -> World {
        let mats = Arc::new(fe_materials::base_set());
        let mut w = World::new(mats, 7, threads);
        w.ensure_rect(IRect::new(0, 0, 256, 192), Cell::AIR);
        w
    }

    fn id(w: &World, name: &str) -> MaterialId {
        w.materials().id(name).unwrap_or_default()
    }

    fn run(w: &mut World, ticks: u32) {
        for _ in 0..ticks {
            w.step_cells();
            w.finish_tick();
        }
    }

    #[test]
    fn sand_falls_and_piles() {
        let mut w = world(1);
        let sand = id(&w, "sand");
        let stone = id(&w, "stone");
        w.fill_rect(IRect::new(0, 180, 256, 192), stone);
        w.fill_rect(IRect::new(100, 10, 120, 30), sand);
        run(&mut w, 300);
        assert_eq!(w.count(IRect::new(0, 0, 256, 192), sand), 400);
        assert_eq!(w.count(IRect::new(0, 0, 256, 150), sand), 0, "sand should settle on the floor");
        assert!(w.count(IRect::new(80, 170, 140, 180), sand) > 200);
    }

    #[test]
    fn water_conserves_mass_and_levels() {
        let mut w = world(1);
        let water = id(&w, "water");
        let stone = id(&w, "stone");
        w.fill_rect(IRect::new(0, 180, 256, 192), stone);
        w.fill_rect(IRect::new(0, 100, 4, 180), stone);
        w.fill_rect(IRect::new(252, 100, 256, 180), stone);
        w.fill_rect(IRect::new(10, 20, 60, 120), water);
        run(&mut w, 4000);
        assert_eq!(w.count(IRect::new(0, 0, 256, 192), water), 5000);
        // 5000 cells over a 248-wide floor is about 20 rows deep (surface at y 160).
        assert_eq!(w.count(IRect::new(0, 0, 256, 155), water), 0);
        assert_eq!(w.count(IRect::new(4, 165, 252, 180), water), 248 * 15);
        // A level pool sleeps, partly filled top row and all.
        assert_eq!(w.stats().active_chunks, 0, "visited {}", w.stats().visited_cells);
    }

    #[test]
    fn world_sleeps_after_settling() {
        let mut w = world(1);
        let sand = id(&w, "sand");
        let stone = id(&w, "stone");
        w.fill_rect(IRect::new(0, 180, 256, 192), stone);
        w.fill_rect(IRect::new(100, 10, 120, 30), sand);
        run(&mut w, 600);
        assert_eq!(w.stats().active_chunks, 0);
    }

    #[test]
    fn thread_count_does_not_change_results() {
        let build = |threads| {
            let mut w = world(threads);
            let (sand, water, lava, wood) =
                (id(&w, "sand"), id(&w, "water"), id(&w, "lava"), id(&w, "wood"));
            w.fill_rect(IRect::new(0, 170, 256, 192), id(&w, "stone"));
            w.fill_rect(IRect::new(20, 10, 90, 60), sand);
            w.fill_rect(IRect::new(100, 20, 180, 90), water);
            w.fill_rect(IRect::new(190, 30, 230, 50), lava);
            w.fill_rect(IRect::new(140, 150, 200, 170), wood);
            run(&mut w, 400);
            w.hash()
        };
        let one = build(1);
        assert_eq!(one, build(2));
        assert_eq!(one, build(4));
    }

    #[test]
    fn water_on_lava_makes_obsidian_and_steam() {
        let mut w = world(1);
        w.fill_rect(IRect::new(0, 180, 256, 192), id(&w, "stone"));
        w.fill_rect(IRect::new(100, 160, 160, 180), id(&w, "lava"));
        w.fill_rect(IRect::new(100, 120, 160, 140), id(&w, "water"));
        run(&mut w, 200);
        let census = w.census();
        assert!(census[id(&w, "obsidian").index()] > 0);
        assert!(census[id(&w, "steam").index()] + census[id(&w, "water").index()] > 0);
    }

    #[test]
    fn wood_burns_to_smoke_and_a_little_ash() {
        let mut w = world(1);
        let wood = id(&w, "wood");
        w.fill_rect(IRect::new(0, 180, 256, 192), id(&w, "stone"));
        w.fill_rect(IRect::new(100, 150, 140, 180), wood);
        w.fill_rect(IRect::new(118, 148, 122, 150), id(&w, "burning_wood"));
        let mut smoke_seen = 0;
        for _ in 0..15 {
            run(&mut w, 100);
            smoke_seen = smoke_seen.max(w.census()[id(&w, "smoke").index()]);
        }
        let census = w.census();
        let left = census[wood.index()] + census[id(&w, "burning_wood").index()];
        let burned = 1200 + 8 - left;
        let ash = census[id(&w, "ash").index()];
        assert!(left < 1000, "wood left {left}");
        assert!(smoke_seen > 50, "smoke {smoke_seen}");
        assert!(ash > 0 && ash * 4 < burned, "ash {ash} of {burned} burned");
    }

    #[test]
    fn head_tables_follow_columns_across_chunks() {
        let mut w = world(1);
        w.fill_rect(IRect::new(0, 180, 256, 192), id(&w, "stone"));
        // A one-cell column of water from y 20 down to the floor at 180.
        w.fill_rect(IRect::new(10, 20, 11, 180), id(&w, "water"));
        // Nothing moves on the first tick (fresh cells carry its parity), so
        // the tables describe the column as placed.
        w.step_cells();
        let at =
            |cy| w.grid.chunk(ChunkPos::new(0, cy)).map(|c| (c.bottom_run()[10], c.head_in()[10]));
        assert_eq!(at(0), Some((44, 0)));
        assert_eq!(at(1), Some((64, 44)));
        assert_eq!(at(2), Some((0, 108)));
    }

    #[test]
    fn a_water_column_collapses_fast_and_comes_to_rest() {
        // A column 100 tall: a real collapse halves its height at the wall in
        // about two sqrt(h / g), 120 ticks, and the front runs at up to
        // 2 sqrt(g h), 3.3 cells per tick.
        let mats = Arc::new(fe_materials::base_set());
        let mut w = World::new(mats, 3, 2);
        w.ensure_rect(IRect::new(0, 0, 512, 192), Cell::AIR);
        let water = id(&w, "water");
        let bed = id(&w, "bedrock");
        w.fill_rect(IRect::new(0, 150, 512, 192), bed);
        w.fill_rect(IRect::new(0, 0, 4, 150), bed);
        w.fill_rect(IRect::new(508, 0, 512, 150), bed);
        w.fill_rect(IRect::new(4, 50, 54, 150), water);
        run(&mut w, 120);
        assert_eq!(w.count(IRect::new(4, 0, 8, 100), water), 0, "still over half height");
        assert!(w.count(IRect::new(250, 0, 508, 150), water) > 50, "front too slow");
        run(&mut w, 2880);
        assert_eq!(w.stats().active_chunks, 0, "not asleep");
        assert_eq!(w.count(IRect::new(0, 0, 512, 192), water), 5000);
        // 5,000 cells over a 504-wide floor: 9 or 10 rows everywhere.
        assert_eq!(w.count(IRect::new(0, 0, 512, 140), water), 0);
        assert_eq!(w.count(IRect::new(4, 141, 508, 150), water), 504 * 9);
    }

    /// The top liquid row of each column in `x0..x1`, or `y1` where there is none.
    fn tops(w: &World, liquid: MaterialId, x0: i32, x1: i32, y1: i32) -> Vec<i32> {
        (x0..x1)
            .map(|x| {
                (0..y1).find(|&y| w.get(CellPos::new(x, y)).material() == liquid).unwrap_or(y1)
            })
            .collect()
    }

    #[test]
    fn a_bore_slopes_like_a_breaking_front() {
        // Deep water released into a shallow pool drives a bore 70 rows tall
        // down the tank. The shallow water equations make its front a step one
        // column wide; the roller spreads it over many columns, so the front
        // of the bore never stands as a wall.
        let mats = Arc::new(fe_materials::base_set());
        let mut w = World::new(mats, 5, 2);
        w.ensure_rect(IRect::new(0, 0, 1024, 256), Cell::AIR);
        let water = id(&w, "water");
        let bed = id(&w, "bedrock");
        w.fill_rect(IRect::new(0, 240, 1024, 256), bed);
        w.fill_rect(IRect::new(0, 0, 4, 240), bed);
        w.fill_rect(IRect::new(1020, 0, 1024, 240), bed);
        w.fill_rect(IRect::new(4, 40, 300, 240), water);
        w.fill_rect(IRect::new(300, 190, 1020, 240), water);
        let mass = w.count(IRect::new(0, 0, 1024, 256), water);
        let mut steepest = 0;
        for _ in 0..6 {
            run(&mut w, 25);
            let t = tops(&w, water, 300, 1020, 240);
            steepest = steepest.max(t.windows(2).map(|p| (p[0] - p[1]).abs()).max().unwrap_or(0));
        }
        let t = tops(&w, water, 300, 1020, 240);
        let front = 300 + t.iter().rposition(|&y| y < 180).unwrap_or(0) as i32;
        // A bore from 50 to about 110 rows deep runs at about 2.1 cells a tick.
        assert!((560..=720).contains(&front), "bore front at {front}");
        assert!(steepest <= 8, "a step of {steepest} rows between neighboring columns");
        assert_eq!(w.count(IRect::new(0, 0, 1024, 256), water), mass);
    }

    #[test]
    fn a_falling_slab_stays_whole_across_chunk_borders() {
        // A slab of water over open space, its underside rising to the right
        // across four chunk borders. It falls as one piece: no column, at a
        // chunk border or anywhere else, runs ahead of its neighbors.
        let mut w = world(4);
        let water = id(&w, "water");
        w.fill_rect(IRect::new(0, 180, 256, 192), id(&w, "stone"));
        for x in 20..240 {
            w.fill_rect(IRect::new(x, 4, x + 1, 130 - (x - 20) / 2), water);
        }
        run(&mut w, 30);
        // Its underside starts from row 129 up to row 20 and falls about 12
        // rows in 30 ticks.
        let lows: Vec<i32> = (20..240)
            .map(|x| {
                (0..180).rev().find(|&y| w.get(CellPos::new(x, y)).material() == water).unwrap_or(0)
            })
            .collect();
        let worst = lows.windows(2).map(|p| (p[0] - p[1]).abs()).max().unwrap_or(0);
        assert!(worst <= 3, "lowest cells of neighboring columns differ by {worst} rows: {lows:?}");
        let fell = lows.iter().zip(20..).all(|(&y, x)| y >= 139 - (x - 20) / 2);
        assert!(fell, "the slab did not fall: {lows:?}");
    }

    #[test]
    fn a_deep_lake_sloshes_then_levels_and_sleeps() {
        // 300 rows of water released into a 248-wide basin slosh for a while,
        // then the lake levels to within a row or two and the world sleeps.
        let mats = Arc::new(fe_materials::base_set());
        let mut w = World::new(mats, 9, 2);
        w.ensure_rect(IRect::new(0, 0, 256, 448), Cell::AIR);
        let water = id(&w, "water");
        let bed = id(&w, "bedrock");
        w.fill_rect(IRect::new(0, 420, 256, 448), bed);
        w.fill_rect(IRect::new(0, 0, 4, 420), bed);
        w.fill_rect(IRect::new(252, 0, 256, 420), bed);
        w.fill_rect(IRect::new(4, 120, 100, 420), water);
        let mass = w.count(IRect::new(0, 0, 256, 448), water);
        run(&mut w, 4000);
        assert_eq!(w.stats().active_chunks, 0, "not asleep");
        assert_eq!(w.count(IRect::new(0, 0, 256, 448), water), mass);
        // 28,800 cells over 248 columns: 116 rows and a part row.
        let t = tops(&w, water, 4, 252, 420);
        let (lo, hi) = (t.iter().min().copied().unwrap_or(0), t.iter().max().copied().unwrap_or(0));
        assert!(hi - lo <= 2 && (302..=305).contains(&lo), "surface from {lo} to {hi}");
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(24))]
        #[test]
        fn pressure_pass_only_lowers_liquid(
            blobs in proptest::collection::vec((0i32..220, 0i32..140, 1i32..36, 1i32..30), 1..5),
            steps in proptest::collection::vec((0i32..240, 100i32..180, 4i32..40), 0..4),
            ticks in 0u32..60,
        ) {
            let mut w = world(1);
            let water = id(&w, "water");
            let stone = id(&w, "stone");
            w.fill_rect(IRect::new(0, 180, 256, 192), stone);
            for (x, y, wd) in steps {
                w.fill_rect(IRect::new(x, y, x + wd, 180), stone);
            }
            for (x, y, wd, h) in blobs {
                w.fill_rect(IRect::new(x, y, x + wd, y + h), water);
            }
            run(&mut w, ticks);
            // Offer every liquid cell to the pass.
            let mut pressure = Vec::new();
            for y in 0..192 {
                for x in 0..256 {
                    if w.get(CellPos::new(x, y)).material() == water {
                        pressure.push((y, x));
                    }
                }
            }
            let mass = w.count(IRect::new(0, 0, 256, 192), water);
            let before = w.liquid_potential();
            w.pressure_pass(pressure);
            proptest::prop_assert!(w.liquid_potential() <= before);
            proptest::prop_assert_eq!(w.count(IRect::new(0, 0, 256, 192), water), mass);
        }
    }

    #[test]
    fn wind_tilts_flames_but_keeps_some_jitter() {
        for wind in [3, 4] {
            flames_in_wind(wind);
        }
    }

    fn flames_in_wind(wind: i8) {
        let mut w = world(1);
        w.set_wind(wind);
        w.fill_rect(IRect::new(0, 180, 256, 192), id(&w, "stone"));
        w.fill_rect(IRect::new(126, 176, 130, 180), id(&w, "heater"));
        w.fill_rect(IRect::new(126, 172, 130, 176), id(&w, "burning_coal"));
        let fire = id(&w, "fire");
        let (mut upwind, mut downwind) = (0, 0);
        for _ in 0..120 {
            run(&mut w, 1);
            upwind += w.count(IRect::new(100, 140, 126, 172), fire);
            downwind += w.count(IRect::new(130, 140, 156, 172), fire);
        }
        assert!(downwind > upwind, "wind {wind}: down {downwind} up {upwind}");
        assert!(upwind > 0, "wind {wind}: no flame ever moved upwind");
    }

    #[test]
    fn steam_does_not_pass_through_thin_walls() {
        let mut w = world(1);
        let stone = id(&w, "stone");
        let steam = id(&w, "steam");
        // A sealed box with 1-cell walls; the inside is 41..59 on both axes.
        w.fill_rect(IRect::new(40, 40, 60, 41), stone);
        w.fill_rect(IRect::new(40, 59, 60, 60), stone);
        w.fill_rect(IRect::new(40, 40, 41, 60), stone);
        w.fill_rect(IRect::new(59, 40, 60, 60), stone);
        w.fill_rect(IRect::new(45, 50, 55, 59), steam);
        for _ in 0..200 {
            w.step_cells();
            w.finish_tick();
            let outside = w.count(IRect::new(0, 0, 256, 192), steam)
                - w.count(IRect::new(41, 41, 59, 59), steam);
            assert_eq!(outside, 0, "steam left the box at tick {}", w.tick());
        }
    }
}
