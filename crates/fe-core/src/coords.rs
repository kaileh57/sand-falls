use serde::{Deserialize, Serialize};

/// Chunk edge length in cells.
pub const CHUNK_SIZE: i32 = 64;
/// `log2(CHUNK_SIZE)`.
pub const CHUNK_SHIFT: u32 = 6;
/// Cells per chunk.
pub const CHUNK_AREA: usize = (CHUNK_SIZE * CHUNK_SIZE) as usize;

/// A cell position in world coordinates. `y` grows downward.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct CellPos {
    /// Column.
    pub x: i32,
    /// Row, growing downward.
    pub y: i32,
}

impl CellPos {
    /// Creates a position.
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// Returns the chunk that contains this cell.
    pub const fn chunk(self) -> ChunkPos {
        ChunkPos { x: self.x >> CHUNK_SHIFT, y: self.y >> CHUNK_SHIFT }
    }

    /// Returns the cell's index inside its chunk, row-major.
    pub const fn local_index(self) -> usize {
        let lx = self.x & (CHUNK_SIZE - 1);
        let ly = self.y & (CHUNK_SIZE - 1);
        (ly * CHUNK_SIZE + lx) as usize
    }
}

/// A chunk position: world cell coordinates divided by [`CHUNK_SIZE`], rounded down.
///
/// The derived ordering (by `y`, then `x`) is the canonical chunk order used for
/// hashing and for merging per-chunk results.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkPos {
    /// Chunk column.
    pub x: i32,
    /// Chunk row.
    pub y: i32,
}

impl PartialOrd for ChunkPos {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ChunkPos {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.y, self.x).cmp(&(other.y, other.x))
    }
}

impl ChunkPos {
    /// Creates a chunk position.
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// World position of the chunk's top-left cell.
    pub const fn origin(self) -> CellPos {
        CellPos { x: self.x << CHUNK_SHIFT, y: self.y << CHUNK_SHIFT }
    }

    /// Checkerboard group (0 to 3) used by the parallel scheduler.
    pub const fn group(self) -> usize {
        ((self.x & 1) | ((self.y & 1) << 1)) as usize
    }
}

/// An axis-aligned rectangle in world cell coordinates, `min` inclusive and `max` exclusive.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IRect {
    /// Inclusive minimum corner.
    pub min: CellPos,
    /// Exclusive maximum corner.
    pub max: CellPos,
}

impl IRect {
    /// Creates a rectangle from an inclusive minimum and exclusive maximum.
    pub const fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        Self { min: CellPos::new(x0, y0), max: CellPos::new(x1, y1) }
    }

    /// Creates a rectangle from a position and size.
    pub const fn from_size(x: i32, y: i32, w: i32, h: i32) -> Self {
        Self::new(x, y, x + w, y + h)
    }

    /// Width in cells (0 if empty).
    pub const fn width(&self) -> i32 {
        if self.max.x > self.min.x { self.max.x - self.min.x } else { 0 }
    }

    /// Height in cells (0 if empty).
    pub const fn height(&self) -> i32 {
        if self.max.y > self.min.y { self.max.y - self.min.y } else { 0 }
    }

    /// Returns true if the rectangle covers no cells.
    pub const fn is_empty(&self) -> bool {
        self.width() == 0 || self.height() == 0
    }

    /// Returns true if the cell lies inside the rectangle.
    pub const fn contains(&self, p: CellPos) -> bool {
        p.x >= self.min.x && p.x < self.max.x && p.y >= self.min.y && p.y < self.max.y
    }

    /// Returns the overlap of two rectangles.
    #[must_use]
    pub fn intersect(&self, o: &Self) -> Self {
        Self::new(
            self.min.x.max(o.min.x),
            self.min.y.max(o.min.y),
            self.max.x.min(o.max.x),
            self.max.y.min(o.max.y),
        )
    }

    /// Returns the range of chunks the rectangle touches, as an inclusive-exclusive chunk rect.
    pub const fn chunk_range(&self) -> (ChunkPos, ChunkPos) {
        let a = self.min.chunk();
        let b = CellPos::new(self.max.x - 1, self.max.y - 1).chunk();
        (a, ChunkPos::new(b.x + 1, b.y + 1))
    }
}

/// A rectangle inside one chunk in local coordinates, both corners inclusive.
///
/// The empty rect is represented by `min > max`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LocalRect {
    /// Inclusive minimum x.
    pub x0: u8,
    /// Inclusive minimum y.
    pub y0: u8,
    /// Inclusive maximum x.
    pub x1: u8,
    /// Inclusive maximum y.
    pub y1: u8,
}

impl Default for LocalRect {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl LocalRect {
    /// The empty rect.
    pub const EMPTY: Self = Self { x0: 255, y0: 255, x1: 0, y1: 0 };
    /// The whole chunk.
    pub const FULL: Self =
        Self { x0: 0, y0: 0, x1: (CHUNK_SIZE - 1) as u8, y1: (CHUNK_SIZE - 1) as u8 };

    /// Returns true if the rect covers no cells.
    pub const fn is_empty(&self) -> bool {
        self.x0 > self.x1 || self.y0 > self.y1
    }

    /// Grows the rect to include a local cell.
    #[inline]
    pub fn include(&mut self, x: u8, y: u8) {
        self.x0 = self.x0.min(x);
        self.y0 = self.y0.min(y);
        self.x1 = self.x1.max(x);
        self.y1 = self.y1.max(y);
    }

    /// Returns the union of two rects.
    #[must_use]
    #[inline]
    pub fn union(self, o: Self) -> Self {
        Self {
            x0: self.x0.min(o.x0),
            y0: self.y0.min(o.y0),
            x1: self.x1.max(o.x1),
            y1: self.y1.max(o.y1),
        }
    }

    /// Number of cells covered.
    pub const fn area(&self) -> u32 {
        if self.is_empty() {
            0
        } else {
            (self.x1 - self.x0 + 1) as u32 * (self.y1 - self.y0 + 1) as u32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_coordinates_map_to_correct_chunks() {
        assert_eq!(CellPos::new(-1, -1).chunk(), ChunkPos::new(-1, -1));
        assert_eq!(CellPos::new(-64, 63).chunk(), ChunkPos::new(-1, 0));
        assert_eq!(CellPos::new(-65, 64).chunk(), ChunkPos::new(-2, 1));
        assert_eq!(CellPos::new(-1, 0).local_index(), 63);
    }

    #[test]
    fn groups_never_neighbor_each_other() {
        let a = ChunkPos::new(4, 6);
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                assert_ne!(a.group(), ChunkPos::new(a.x + dx, a.y + dy).group());
            }
        }
    }

    #[test]
    fn chunk_order_is_row_major() {
        let mut v = vec![ChunkPos::new(1, 0), ChunkPos::new(0, 1), ChunkPos::new(0, 0)];
        v.sort();
        assert_eq!(v, vec![ChunkPos::new(0, 0), ChunkPos::new(1, 0), ChunkPos::new(0, 1)]);
    }

    #[test]
    fn local_rect_tracks_bounds() {
        let mut r = LocalRect::EMPTY;
        assert!(r.is_empty());
        r.include(3, 4);
        r.include(10, 2);
        assert_eq!(r, LocalRect { x0: 3, y0: 2, x1: 10, y1: 4 });
        assert_eq!(r.area(), 8 * 3);
    }
}
