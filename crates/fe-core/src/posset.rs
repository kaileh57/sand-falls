use crate::{CellPos, rng::mix64};

/// A set of cell positions with deterministic behavior.
///
/// Open addressing with linear probing and a fixed hash. It supports insert and
/// membership only, so no result can depend on iteration order.
#[derive(Clone, Debug)]
pub struct PosSet {
    slots: Vec<u64>,
    used: Vec<bool>,
    len: usize,
    mask: usize,
}

impl Default for PosSet {
    fn default() -> Self {
        Self::with_capacity(16)
    }
}

impl PosSet {
    /// Creates a set sized for at least `cap` entries before growing.
    pub fn with_capacity(cap: usize) -> Self {
        let n = (cap * 2).next_power_of_two().max(16);
        Self { slots: vec![0; n], used: vec![false; n], len: 0, mask: n - 1 }
    }

    #[inline]
    const fn key(p: CellPos) -> u64 {
        ((p.x as u32 as u64) << 32) | p.y as u32 as u64
    }

    /// Number of entries.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the set has no entries.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Removes every entry, keeping the allocation.
    pub fn clear(&mut self) {
        self.used.fill(false);
        self.len = 0;
    }

    /// Returns true if the position is in the set.
    pub fn contains(&self, p: CellPos) -> bool {
        let k = Self::key(p);
        let mut i = mix64(k) as usize & self.mask;
        loop {
            if !self.used[i] {
                return false;
            }
            if self.slots[i] == k {
                return true;
            }
            i = (i + 1) & self.mask;
        }
    }

    /// Inserts a position. Returns true if it was not already present.
    pub fn insert(&mut self, p: CellPos) -> bool {
        if (self.len + 1) * 2 > self.slots.len() {
            self.grow();
        }
        let k = Self::key(p);
        let mut i = mix64(k) as usize & self.mask;
        loop {
            if !self.used[i] {
                self.slots[i] = k;
                self.used[i] = true;
                self.len += 1;
                return true;
            }
            if self.slots[i] == k {
                return false;
            }
            i = (i + 1) & self.mask;
        }
    }

    fn grow(&mut self) {
        let old = std::mem::take(&mut self.slots);
        let used = std::mem::take(&mut self.used);
        let n = old.len() * 2;
        self.slots = vec![0; n];
        self.used = vec![false; n];
        self.mask = n - 1;
        self.len = 0;
        for (k, u) in old.into_iter().zip(used) {
            if u {
                let p = CellPos::new((k >> 32) as u32 as i32, k as u32 as i32);
                self.insert(p);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_contains() {
        let mut s = PosSet::with_capacity(4);
        for i in -500..500 {
            assert!(s.insert(CellPos::new(i, i * 3)));
        }
        assert!(!s.insert(CellPos::new(0, 0)));
        assert_eq!(s.len(), 1000);
        assert!(s.contains(CellPos::new(-7, -21)));
        assert!(!s.contains(CellPos::new(-7, -20)));
        assert!(!s.contains(CellPos::new(-1, -1)));
        assert!(s.insert(CellPos::new(-1, -1)));
        assert!(s.contains(CellPos::new(-1, -1)));
        s.clear();
        assert!(s.is_empty());
    }
}
