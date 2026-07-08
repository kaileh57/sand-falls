use crate::sched::Chunk;
use fe_core::{CHUNK_SIZE, Cell, CellPos, ChunkPos, StableHasher, mix64};
use std::collections::BTreeMap;

/// Fast lookup from chunk position to storage slot.
///
/// Open addressing with a fixed hash, so lookups are deterministic and cheap.
/// Iteration never goes through this table; ordered iteration uses `Grid::order`.
#[derive(Clone, Debug)]
struct ChunkIndex {
    keys: Vec<u64>,
    vals: Vec<u32>,
    mask: usize,
    len: usize,
}

/// Marks an unused slot in `vals`. Slot indices never reach this value.
const EMPTY: u32 = u32::MAX;

impl ChunkIndex {
    fn new() -> Self {
        Self { keys: vec![0; 64], vals: vec![EMPTY; 64], mask: 63, len: 0 }
    }

    #[inline]
    const fn key(p: ChunkPos) -> u64 {
        ((p.x as u32 as u64) << 32) | p.y as u32 as u64
    }

    #[inline]
    fn slot(&self, k: u64) -> usize {
        mix64(k) as usize & self.mask
    }

    #[inline]
    fn get(&self, p: ChunkPos) -> Option<u32> {
        let k = Self::key(p);
        let mut i = self.slot(k);
        loop {
            if self.vals[i] == EMPTY {
                return None;
            }
            if self.keys[i] == k {
                return Some(self.vals[i]);
            }
            i = (i + 1) & self.mask;
        }
    }

    fn insert(&mut self, p: ChunkPos, v: u32) {
        if (self.len + 1) * 2 > self.keys.len() {
            self.grow();
        }
        let k = Self::key(p);
        let mut i = self.slot(k);
        loop {
            if self.vals[i] == EMPTY {
                self.keys[i] = k;
                self.vals[i] = v;
                self.len += 1;
                return;
            }
            if self.keys[i] == k {
                self.vals[i] = v;
                return;
            }
            i = (i + 1) & self.mask;
        }
    }

    /// Removes with backward-shift deletion, so no tombstones accumulate.
    fn remove(&mut self, p: ChunkPos) {
        let k = Self::key(p);
        let mut i = self.slot(k);
        loop {
            if self.vals[i] == EMPTY {
                return;
            }
            if self.keys[i] == k {
                break;
            }
            i = (i + 1) & self.mask;
        }
        self.len -= 1;
        let mut hole = i;
        let mut j = i;
        loop {
            j = (j + 1) & self.mask;
            if self.vals[j] == EMPTY {
                break;
            }
            let kj = self.keys[j];
            let home = self.slot(kj);
            // Move j into the hole if its home slot is not cyclically in (hole, j].
            let dist_hole = hole.wrapping_sub(home) & self.mask;
            let dist_j = j.wrapping_sub(home) & self.mask;
            if dist_hole < dist_j {
                self.keys[hole] = kj;
                self.vals[hole] = self.vals[j];
                hole = j;
            }
        }
        self.vals[hole] = EMPTY;
    }

    fn grow(&mut self) {
        let keys = std::mem::take(&mut self.keys);
        let vals = std::mem::take(&mut self.vals);
        let n = keys.len() * 2;
        self.keys = vec![0; n];
        self.vals = vec![EMPTY; n];
        self.mask = n - 1;
        self.len = 0;
        for (k, v) in keys.into_iter().zip(vals) {
            if v != EMPTY {
                self.insert(ChunkPos::new((k >> 32) as u32 as i32, k as u32 as i32), v);
            }
        }
    }
}

/// All loaded chunks.
pub struct Grid {
    chunks: Vec<Chunk>,
    index: ChunkIndex,
    order: BTreeMap<ChunkPos, usize>,
    /// Bumped whenever a chunk is inserted or removed.
    generation: u64,
}

impl Default for Grid {
    fn default() -> Self {
        Self { chunks: Vec::new(), index: ChunkIndex::new(), order: BTreeMap::new(), generation: 0 }
    }
}

impl Grid {
    /// Number of loaded chunks.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Returns true when no chunks are loaded.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Adds a chunk, replacing any chunk already at its position.
    pub fn insert(&mut self, chunk: Chunk) {
        self.generation += 1;
        let pos = chunk.pos();
        if let Some(i) = self.index.get(pos) {
            self.chunks[i as usize] = chunk;
            return;
        }
        let i = self.chunks.len();
        self.chunks.push(chunk);
        self.index.insert(pos, i as u32);
        self.order.insert(pos, i);
    }

    /// Removes and returns the chunk at a position.
    pub fn remove(&mut self, pos: ChunkPos) -> Option<Chunk> {
        let i = self.index.get(pos)? as usize;
        self.generation += 1;
        self.index.remove(pos);
        self.order.remove(&pos);
        let chunk = self.chunks.swap_remove(i);
        if i < self.chunks.len() {
            let moved = self.chunks[i].pos();
            self.index.insert(moved, i as u32);
            self.order.insert(moved, i);
        }
        Some(chunk)
    }

    /// Returns the chunk at a position.
    #[inline]
    pub fn chunk(&self, pos: ChunkPos) -> Option<&Chunk> {
        self.index.get(pos).map(|i| &self.chunks[i as usize])
    }

    /// Returns the chunk at a position, mutably.
    #[inline]
    pub fn chunk_mut(&mut self, pos: ChunkPos) -> Option<&mut Chunk> {
        self.index.get(pos).map(|i| &mut self.chunks[i as usize])
    }

    /// Returns true if the chunk is loaded.
    pub fn contains(&self, pos: ChunkPos) -> bool {
        self.index.get(pos).is_some()
    }

    /// Chunk positions in canonical order.
    pub fn positions(&self) -> impl Iterator<Item = ChunkPos> + '_ {
        self.order.keys().copied()
    }

    /// Chunks in canonical order.
    pub fn chunks(&self) -> impl Iterator<Item = &Chunk> + '_ {
        self.order.values().map(|&i| &self.chunks[i])
    }

    /// Chunks in storage order, mutably. The order is not canonical.
    pub fn chunks_mut_unordered(&mut self) -> impl Iterator<Item = &mut Chunk> + '_ {
        self.chunks.iter_mut()
    }

    /// Chunks in storage order as a slice, for per-chunk work that does not
    /// depend on order.
    pub(crate) fn chunk_slice_mut(&mut self) -> &mut [Chunk] {
        &mut self.chunks
    }

    /// A counter that changes whenever a chunk is inserted or removed.
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Reads a cell. Unloaded cells read as the boundary.
    #[inline]
    pub fn get(&self, p: CellPos) -> Cell {
        match self.chunk(p.chunk()) {
            Some(c) => c.get(p.local_index()),
            None => Cell::BOUNDARY,
        }
    }

    /// Writes a cell without marking anything dirty. Returns false if unloaded.
    #[inline]
    pub fn set_raw(&mut self, p: CellPos, cell: Cell) -> bool {
        match self.chunk_mut(p.chunk()) {
            Some(c) => {
                c.set_raw(p.local_index(), cell);
                true
            }
            None => false,
        }
    }

    /// Writes a cell without marking it or touching the column tables. Only for
    /// changes that cannot affect them, such as a liquid cell's settled mark.
    pub(crate) fn set_quiet(&mut self, p: CellPos, cell: Cell) {
        if let Some(c) = self.chunk_mut(p.chunk()) {
            c.set_quiet(p.local_index(), cell);
        }
    }

    /// Marks a cell and its eight neighbors for a visit on the next tick.
    pub fn mark(&self, p: CellPos) {
        for y in p.y - 1..=p.y + 1 {
            for x in p.x - 1..=p.x + 1 {
                let q = CellPos::new(x, y);
                if let Some(c) = self.chunk(q.chunk()) {
                    c.mark((q.x & (CHUNK_SIZE - 1)) as u8, (q.y & (CHUNK_SIZE - 1)) as u8);
                }
            }
        }
    }

    /// Marks every cell of the world rectangle `x0..=x1, y0..=y1` for a visit on
    /// the next tick.
    pub fn mark_rect(&self, x0: i32, y0: i32, x1: i32, y1: i32) {
        let (a, b) = (CellPos::new(x0, y0).chunk(), CellPos::new(x1, y1).chunk());
        for cy in a.y..=b.y {
            for cx in a.x..=b.x {
                let Some(c) = self.chunk(ChunkPos::new(cx, cy)) else { continue };
                let o = c.pos().origin();
                let (lx0, ly0) = ((x0.max(o.x) - o.x) as u8, (y0.max(o.y) - o.y) as u8);
                let (lx1, ly1) = (
                    (x1.min(o.x + CHUNK_SIZE - 1) - o.x) as u8,
                    (y1.min(o.y + CHUNK_SIZE - 1) - o.y) as u8,
                );
                c.mark(lx0, ly0);
                c.mark(lx1, ly1);
            }
        }
    }

    /// Marks only the cell itself for a visit on the next tick.
    pub fn keep(&self, p: CellPos) {
        if let Some(c) = self.chunk(p.chunk()) {
            c.mark((p.x & (CHUNK_SIZE - 1)) as u8, (p.y & (CHUNK_SIZE - 1)) as u8);
        }
    }

    /// Writes a cell and marks it and its neighbors dirty. Returns false if unloaded.
    pub fn set(&mut self, p: CellPos, cell: Cell) -> bool {
        if self.set_raw(p, cell) {
            self.mark(p);
            true
        } else {
            false
        }
    }

    /// Moves every chunk's pending dirty rect into its current one.
    pub(crate) fn advance(&mut self) {
        for c in &mut self.chunks {
            c.advance();
        }
    }

    /// Hash of one chunk's cells and position.
    pub fn chunk_hash(chunk: &Chunk) -> u64 {
        let mut h = StableHasher::new();
        h.write_i32(chunk.pos().x);
        h.write_i32(chunk.pos().y);
        for i in (0..fe_core::CHUNK_AREA).step_by(2) {
            h.write_u64(
                u64::from(chunk.get(i).to_bits()) | u64::from(chunk.get(i + 1).to_bits()) << 32,
            );
        }
        h.finish()
    }

    /// Hash of every loaded chunk in canonical order.
    pub fn hash(&self) -> u64 {
        let mut h = StableHasher::new();
        for c in self.chunks() {
            h.write_u64(Self::chunk_hash(c));
        }
        h.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn index_matches_btreemap(ops in proptest::collection::vec((-20i32..20, -20i32..20, any::<bool>()), 0..400)) {
            let mut idx = ChunkIndex::new();
            let mut reference = BTreeMap::new();
            for (n, (x, y, ins)) in ops.into_iter().enumerate() {
                let p = ChunkPos::new(x, y);
                if ins {
                    idx.insert(p, n as u32);
                    reference.insert(p, n as u32);
                } else {
                    idx.remove(p);
                    reference.remove(&p);
                }
            }
            for x in -20..20 {
                for y in -20..20 {
                    let p = ChunkPos::new(x, y);
                    prop_assert_eq!(idx.get(p), reference.get(&p).copied());
                }
            }
        }
    }

    #[test]
    fn remove_keeps_other_chunks_reachable() {
        let mut g = Grid::default();
        for i in 0..10 {
            g.insert(Chunk::new(ChunkPos::new(i, -i), Cell::AIR));
        }
        assert!(g.remove(ChunkPos::new(3, -3)).is_some());
        assert_eq!(g.len(), 9);
        for i in 0..10 {
            assert_eq!(g.contains(ChunkPos::new(i, -i)), i != 3);
            if i != 3 {
                assert_eq!(
                    g.chunk(ChunkPos::new(i, -i)).map(Chunk::pos),
                    Some(ChunkPos::new(i, -i))
                );
            }
        }
    }

    #[test]
    fn set_marks_across_chunk_borders() {
        let mut g = Grid::default();
        g.insert(Chunk::new(ChunkPos::new(0, 0), Cell::AIR));
        g.insert(Chunk::new(ChunkPos::new(1, 0), Cell::AIR));
        g.advance();
        g.advance();
        assert!(g.chunk(ChunkPos::new(0, 0)).is_some_and(|c| c.dirty().is_empty()));
        g.set(CellPos::new(64, 5), Cell::BOUNDARY);
        let left = g.chunk(ChunkPos::new(0, 0)).map(Chunk::pending);
        assert!(left.is_some_and(|r| r.x0 == 63 && r.y0 == 4 && r.y1 == 6));
    }
}
