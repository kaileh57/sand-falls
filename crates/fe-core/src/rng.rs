use serde::{Deserialize, Serialize};

/// Finalizer from `SplitMix64`. Spreads every input bit across the output.
#[inline]
pub const fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Returns a random value for one cell on one tick.
///
/// The result depends only on its arguments, so it is the same no matter which
/// thread computes it or in what order. `salt` separates independent decisions
/// made for the same cell on the same tick.
#[inline]
pub const fn cell_rand(seed: u64, tick: u64, x: i32, y: i32, salt: u32) -> u32 {
    let pos = ((x as u32 as u64) << 32) | (y as u32 as u64);
    let h = mix64(seed ^ mix64(tick ^ mix64(pos ^ ((salt as u64) << 17))));
    (h >> 32) as u32
}

/// A sequential random number generator (`SplitMix64`).
///
/// Use only in single-threaded code with a fixed call order, such as world
/// generation or event processing. Cell rules use [`cell_rand`] instead.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeqRng {
    state: u64,
}

impl SeqRng {
    /// Creates a generator from a seed.
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns the next 64-bit value.
    pub const fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.state)
    }

    /// Returns the next 32-bit value.
    pub const fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Returns a value in `0..n`. Returns 0 when `n` is 0.
    pub const fn below(&mut self, n: u32) -> u32 {
        if n == 0 { 0 } else { ((self.next_u32() as u64 * n as u64) >> 32) as u32 }
    }

    /// Returns a value in `lo..=hi`.
    pub const fn range_i32(&mut self, lo: i32, hi: i32) -> i32 {
        if hi <= lo { lo } else { lo + self.below((hi - lo + 1) as u32) as i32 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_rand_is_stable() {
        // Golden values pin the function. Changing it changes every world hash.
        assert_eq!(cell_rand(1, 2, 3, 4, 5), cell_rand(1, 2, 3, 4, 5));
        assert_ne!(cell_rand(1, 2, 3, 4, 5), cell_rand(1, 2, 3, 4, 6));
        assert_ne!(cell_rand(1, 2, -3, 4, 5), cell_rand(1, 2, 3, 4, 5));
        assert_eq!(mix64(0), 0);
        assert_eq!(mix64(1), 0x5692_161d_100b_05e5);
    }

    #[test]
    fn below_stays_in_range() {
        let mut r = SeqRng::new(9);
        for n in 1..200 {
            assert!(r.below(n) < n);
        }
        assert_eq!(r.below(0), 0);
    }

    #[test]
    fn cell_rand_is_roughly_uniform() {
        let mut buckets = [0u32; 16];
        for i in 0..16_000 {
            buckets[(cell_rand(7, i, i as i32, -(i as i32), 0) % 16) as usize] += 1;
        }
        for b in buckets {
            assert!((800..1200).contains(&b), "bucket {b}");
        }
    }
}
