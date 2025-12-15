//! Chunk storage and the checkerboard scheduler.
//!
//! This is the only module in the engine that uses `unsafe`. It lets several
//! threads write cells in neighboring chunks at the same time, which safe Rust
//! cannot express.
//!
//! # Why the concurrent writes do not race
//!
//! Chunks are split into four groups by the parity of their chunk coordinates
//! (see [`ChunkPos::group`]). A pass updates one group. Two chunks in the same
//! group are at least two chunks (128 cells) apart on some axis, with one full
//! chunk between them.
//!
//! A job may read and write cells only within [`REACH`] cells of its own chunk.
//! [`Neighborhood`] enforces that limit on every access, in release builds too:
//! an access outside it reads as the world boundary and writes nothing. Since
//! `REACH` is half a chunk, the cells two same-group jobs can touch never
//! overlap, so no cell is accessed by two threads during a pass.
//!
//! Dirty-rect updates may land on any chunk in the 3x3 neighborhood, including
//! ones another job is also marking, so they go through atomics. Min/max merges
//! commute, which keeps the result independent of thread timing.
//!
//! Outside a pass, cells are read through `&Chunk` and written through
//! `&mut Chunk`. A pass needs `&mut Grid`, so no `&Chunk` can exist anywhere
//! else while one is running.

use crate::grid::Grid;
use fe_core::{CHUNK_AREA, CHUNK_SIZE, Cell, ChunkPos, LocalRect};
use rayon::prelude::*;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

/// Largest distance, in cells, a job may reach past its own chunk's edge.
pub const REACH: i32 = CHUNK_SIZE / 2;

/// A cell that may be written from a scheduler job.
#[repr(transparent)]
pub(crate) struct SyncCell(UnsafeCell<Cell>);

// SAFETY: every shared access to the inner cell happens either outside a pass,
// where `&mut Grid` rules out concurrent writers, or inside a pass through a
// `Neighborhood`, whose reach check keeps concurrently running jobs on disjoint
// cells (see the module docs).
unsafe impl Sync for SyncCell {}

/// A rectangle that several threads can grow at once.
pub(crate) struct AtomicRect {
    x0: AtomicU8,
    y0: AtomicU8,
    x1: AtomicU8,
    y1: AtomicU8,
}

impl AtomicRect {
    fn new(r: LocalRect) -> Self {
        Self {
            x0: AtomicU8::new(r.x0),
            y0: AtomicU8::new(r.y0),
            x1: AtomicU8::new(r.x1),
            y1: AtomicU8::new(r.y1),
        }
    }

    #[inline]
    fn include(&self, x: u8, y: u8) {
        self.x0.fetch_min(x, Ordering::Relaxed);
        self.y0.fetch_min(y, Ordering::Relaxed);
        self.x1.fetch_max(x, Ordering::Relaxed);
        self.y1.fetch_max(y, Ordering::Relaxed);
    }

    fn load(&self) -> LocalRect {
        LocalRect {
            x0: self.x0.load(Ordering::Relaxed),
            y0: self.y0.load(Ordering::Relaxed),
            x1: self.x1.load(Ordering::Relaxed),
            y1: self.y1.load(Ordering::Relaxed),
        }
    }

    fn take(&mut self) -> LocalRect {
        let r = self.load();
        *self = Self::new(LocalRect::EMPTY);
        r
    }
}

/// One 64x64 chunk of cells plus its scheduling state.
pub struct Chunk {
    pos: ChunkPos,
    cells: Box<[SyncCell]>,
    /// Cells to visit on the current tick.
    dirty: LocalRect,
    /// Cells that changed this tick, visited on the next one.
    next: AtomicRect,
    /// Set when a static cell changed, so physics can rebuild terrain colliders.
    solid_changed: AtomicBool,
}

impl Chunk {
    /// Creates a chunk filled with one cell value. The whole chunk starts dirty.
    pub fn new(pos: ChunkPos, fill: Cell) -> Self {
        let cells = (0..CHUNK_AREA).map(|_| SyncCell(UnsafeCell::new(fill))).collect();
        Self {
            pos,
            cells,
            dirty: LocalRect::EMPTY,
            next: AtomicRect::new(LocalRect::FULL),
            solid_changed: AtomicBool::new(true),
        }
    }

    /// The chunk's position.
    pub const fn pos(&self) -> ChunkPos {
        self.pos
    }

    /// Reads the cell at a local index (row-major, 0..4096).
    #[inline]
    pub fn get(&self, i: usize) -> Cell {
        // SAFETY: a pass requires `&mut Grid`, so while this `&Chunk` exists no
        // pass is running and nothing writes the cell concurrently.
        unsafe { *self.cells[i].0.get() }
    }

    /// Writes the cell at a local index without touching dirty state.
    #[inline]
    pub fn set_raw(&mut self, i: usize, c: Cell) {
        *self.cells[i].0.get_mut() = c;
    }

    /// Copies all cells out, row-major.
    pub fn to_vec(&self) -> Vec<Cell> {
        (0..CHUNK_AREA).map(|i| self.get(i)).collect()
    }

    /// Replaces all cells from a row-major slice of exactly 4,096 cells.
    pub fn load_cells(&mut self, cells: &[Cell]) {
        for (dst, src) in self.cells.iter_mut().zip(cells) {
            *dst.0.get_mut() = *src;
        }
    }

    /// Cells to visit on the current tick.
    pub const fn dirty(&self) -> LocalRect {
        self.dirty
    }

    /// Cells already marked for the next tick.
    pub fn pending(&self) -> LocalRect {
        self.next.load()
    }

    /// Marks a local cell for a visit on the next tick.
    #[inline]
    pub fn mark(&self, x: u8, y: u8) {
        self.next.include(x, y);
    }

    /// Marks the whole chunk for a visit on the next tick.
    pub fn mark_all(&self) {
        self.next.include(0, 0);
        self.next.include((CHUNK_SIZE - 1) as u8, (CHUNK_SIZE - 1) as u8);
    }

    /// Records that a static cell changed.
    pub fn flag_solid_changed(&self) {
        self.solid_changed.store(true, Ordering::Relaxed);
    }

    /// Reads the solid-changed flag without clearing it.
    pub fn solid_changed(&self) -> bool {
        self.solid_changed.load(Ordering::Relaxed)
    }

    /// Sets the solid-changed flag, used when loading a snapshot.
    pub fn set_solid_changed(&mut self, v: bool) {
        *self.solid_changed.get_mut() = v;
    }

    /// Returns and clears the solid-changed flag.
    pub fn take_solid_changed(&mut self) -> bool {
        std::mem::replace(self.solid_changed.get_mut(), false)
    }

    /// Restores scheduling state, used when loading a snapshot.
    pub fn set_schedule(&mut self, dirty: LocalRect, next: LocalRect) {
        self.dirty = dirty;
        self.next = AtomicRect::new(next);
    }

    /// Moves the pending rect into the current one. Called once per tick.
    pub(crate) fn advance(&mut self) {
        self.dirty = self.next.take();
    }
}

/// Access to one chunk and the cells within [`REACH`] of it, for one job.
///
/// Coordinates are local to the center chunk: `(0, 0)` is its top-left cell,
/// and valid coordinates run from `-REACH` to `CHUNK_SIZE + REACH - 1`.
pub struct Neighborhood<'a> {
    chunks: [Option<&'a Chunk>; 9],
    pos: ChunkPos,
}

impl<'a> Neighborhood<'a> {
    fn new(grid: &'a Grid, pos: ChunkPos) -> Self {
        let mut chunks = [None; 9];
        for dy in -1..=1 {
            for dx in -1..=1 {
                let i = ((dy + 1) * 3 + (dx + 1)) as usize;
                chunks[i] = grid.chunk(ChunkPos::new(pos.x + dx, pos.y + dy));
            }
        }
        Self { chunks, pos }
    }

    /// The center chunk's position.
    pub const fn pos(&self) -> ChunkPos {
        self.pos
    }

    #[inline]
    fn locate(&self, lx: i32, ly: i32, reach: i32) -> Option<(&'a Chunk, usize)> {
        if lx < -reach || ly < -reach || lx >= CHUNK_SIZE + reach || ly >= CHUNK_SIZE + reach {
            debug_assert!(reach > REACH, "cell rule reached ({lx}, {ly}) outside its neighborhood");
            return None;
        }
        let cx = (lx >> fe_core::CHUNK_SHIFT) + 1;
        let cy = (ly >> fe_core::CHUNK_SHIFT) + 1;
        let chunk = self.chunks[(cy * 3 + cx) as usize]?;
        let i = ((ly & (CHUNK_SIZE - 1)) * CHUNK_SIZE + (lx & (CHUNK_SIZE - 1))) as usize;
        Some((chunk, i))
    }

    /// Reads a cell. Missing chunks and out-of-reach cells read as the boundary.
    #[inline]
    pub fn get(&self, lx: i32, ly: i32) -> Cell {
        match self.locate(lx, ly, REACH) {
            // SAFETY: `locate` limits the cell to within REACH of this job's
            // chunk, which no other job in the same pass can reach.
            Some((c, i)) => unsafe { *c.cells[i].0.get() },
            None => Cell::BOUNDARY,
        }
    }

    /// Writes a cell. Writes to missing chunks or out-of-reach cells are dropped.
    #[inline]
    pub fn set(&self, lx: i32, ly: i32, cell: Cell) {
        if let Some((c, i)) = self.locate(lx, ly, REACH) {
            // SAFETY: as in `get`, the cell is reachable only by this job.
            unsafe { *c.cells[i].0.get() = cell };
        }
    }

    /// Marks a cell and its eight neighbors for a visit on the next tick.
    #[inline]
    pub fn mark(&self, lx: i32, ly: i32) {
        for y in ly - 1..=ly + 1 {
            for x in lx - 1..=lx + 1 {
                if let Some((c, i)) = self.locate(x, y, REACH + 1) {
                    c.mark((i as i32 & (CHUNK_SIZE - 1)) as u8, (i as i32 >> 6) as u8);
                }
            }
        }
    }

    /// Marks a horizontal band of rows `ly - 1 ..= ly + 1` from `x0` to `x1` for the next tick.
    ///
    /// The band is clamped to the reach limit. Cost is two rect updates per chunk touched.
    pub fn mark_band(&self, x0: i32, x1: i32, ly: i32) {
        let x0 = x0.max(-REACH);
        let x1 = x1.min(CHUNK_SIZE + REACH - 1);
        for y in [ly - 1, ly + 1] {
            let mut x = x0;
            while x <= x1 {
                // End of the current chunk column, or x1.
                let chunk_end = ((x >> fe_core::CHUNK_SHIFT) + 1) * CHUNK_SIZE - 1;
                let end = chunk_end.min(x1);
                for xx in [x, end] {
                    if let Some((c, i)) = self.locate(xx, y, REACH + 1) {
                        c.mark((i as i32 & (CHUNK_SIZE - 1)) as u8, (i as i32 >> 6) as u8);
                    }
                }
                x = end + 1;
            }
        }
    }

    /// Marks only the cell itself for a visit on the next tick.
    #[inline]
    pub fn keep(&self, lx: i32, ly: i32) {
        if let Some((c, i)) = self.locate(lx, ly, REACH) {
            c.mark((i as i32 & (CHUNK_SIZE - 1)) as u8, (i as i32 >> 6) as u8);
        }
    }

    /// Records that a static cell at this position changed.
    #[inline]
    pub fn solid_changed(&self, lx: i32, ly: i32) {
        if let Some((c, _)) = self.locate(lx, ly, REACH) {
            c.flag_solid_changed();
        }
    }
}

/// Runs one checkerboard pass: `f` once per job, in parallel when a pool is given.
///
/// Every job must belong to the same checkerboard group.
pub(crate) fn run_pass<T, F>(
    grid: &mut Grid,
    jobs: &[(ChunkPos, LocalRect)],
    outs: &mut [T],
    pool: Option<&rayon::ThreadPool>,
    f: F,
) where
    T: Send,
    F: Fn(&Neighborhood<'_>, LocalRect, &mut T) + Sync,
{
    debug_assert_eq!(jobs.len(), outs.len());
    debug_assert!(jobs.windows(2).all(|w| w[0].0.group() == w[1].0.group()));
    let grid: &Grid = grid;
    let work = |(&(pos, rect), out): (&(ChunkPos, LocalRect), &mut T)| {
        let nb = Neighborhood::new(grid, pos);
        f(&nb, rect, out);
    };
    match pool {
        Some(pool) if jobs.len() > 1 => {
            pool.install(|| jobs.par_iter().zip(outs.par_iter_mut()).for_each(work));
        }
        _ => jobs.iter().zip(outs.iter_mut()).for_each(work),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fe_core::MaterialId;

    fn grid_3x3() -> Grid {
        let mut g = Grid::default();
        for y in 0..3 {
            for x in 0..3 {
                g.insert(Chunk::new(ChunkPos::new(x, y), Cell::AIR));
            }
        }
        g
    }

    #[test]
    fn neighborhood_reads_and_writes_neighbors() {
        let mut g = grid_3x3();
        let stone = Cell::new(MaterialId(5), 7, 0);
        let jobs = [(ChunkPos::new(1, 1), LocalRect::FULL)];
        let mut outs = [()];
        run_pass(&mut g, &jobs, &mut outs, None, |nb, _, ()| {
            nb.set(-1, -1, stone);
            nb.set(64 + REACH - 1, 10, stone);
            nb.mark(-1, -1);
        });
        assert_eq!(g.get(fe_core::CellPos::new(63, 63)), stone);
        assert_eq!(g.get(fe_core::CellPos::new(128 + REACH - 1, 74)), stone);
        let corner = g.chunk(ChunkPos::new(0, 0)).map(Chunk::pending);
        assert!(corner.is_some_and(|r| r.x1 == 63 && r.y1 == 63));
    }

    #[test]
    fn missing_chunks_read_as_boundary() {
        let mut g = Grid::default();
        g.insert(Chunk::new(ChunkPos::new(0, 0), Cell::AIR));
        let jobs = [(ChunkPos::new(0, 0), LocalRect::FULL)];
        let mut outs = [Cell::AIR];
        run_pass(&mut g, &jobs, &mut outs, None, |nb, _, out| {
            *out = nb.get(-1, 0);
            nb.set(-1, 0, Cell::AIR);
        });
        assert_eq!(outs[0], Cell::BOUNDARY);
    }

    #[test]
    fn parallel_pass_matches_sequential() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().ok();
        let run = |pool: Option<&rayon::ThreadPool>| {
            let mut g = Grid::default();
            for y in 0..6 {
                for x in 0..6 {
                    g.insert(Chunk::new(ChunkPos::new(x, y), Cell::AIR));
                }
            }
            let jobs: Vec<_> =
                g.positions().filter(|p| p.group() == 0).map(|p| (p, LocalRect::FULL)).collect();
            let mut outs = vec![0u32; jobs.len()];
            run_pass(&mut g, &jobs, &mut outs, pool, |nb, _, out| {
                for y in -REACH..CHUNK_SIZE + REACH {
                    for x in -REACH..CHUNK_SIZE + REACH {
                        let c = nb.get(x, y);
                        nb.set(x, y, Cell::new(MaterialId(c.material().0 + 1), 7, 0));
                        *out += 1;
                    }
                }
            });
            g.hash()
        };
        assert_eq!(run(None), run(pool.as_ref()));
    }
}
