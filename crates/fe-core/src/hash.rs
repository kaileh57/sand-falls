use crate::rng::mix64;

/// A 64-bit hasher whose output is identical on every platform and every run.
///
/// It is not cryptographic. It exists to compare world states between machines
/// and between runs, so it must never depend on pointer values, endianness, or
/// random seeds.
#[derive(Clone, Debug)]
pub struct StableHasher {
    state: u64,
    len: u64,
}

impl Default for StableHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl StableHasher {
    /// Creates a hasher with the fixed initial state.
    pub const fn new() -> Self {
        Self { state: 0x243f_6a88_85a3_08d3, len: 0 }
    }

    /// Feeds one 64-bit word.
    #[inline]
    pub fn write_u64(&mut self, v: u64) {
        self.state = mix64(self.state ^ v).wrapping_add(0x9e37_79b9_7f4a_7c15);
        self.len = self.len.wrapping_add(1);
    }

    /// Feeds one 32-bit word.
    #[inline]
    pub fn write_u32(&mut self, v: u32) {
        self.write_u64(u64::from(v));
    }

    /// Feeds a signed 32-bit word.
    #[inline]
    pub fn write_i32(&mut self, v: i32) {
        self.write_u64(u64::from(v as u32));
    }

    /// Feeds a byte slice, padding the final word with zeros.
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            let mut w = [0u8; 8];
            w.copy_from_slice(c);
            self.write_u64(u64::from_le_bytes(w));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut w = [0u8; 8];
            w[..rest.len()].copy_from_slice(rest);
            self.write_u64(u64::from_le_bytes(w));
        }
        self.write_u64(bytes.len() as u64);
    }

    /// Returns the hash of everything written so far.
    pub const fn finish(&self) -> u64 {
        mix64(self.state ^ self.len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_matters() {
        let mut a = StableHasher::new();
        a.write_u32(1);
        a.write_u32(2);
        let mut b = StableHasher::new();
        b.write_u32(2);
        b.write_u32(1);
        assert_ne!(a.finish(), b.finish());
    }

    #[test]
    fn padding_is_length_aware() {
        let mut a = StableHasher::new();
        a.write_bytes(&[1, 0]);
        let mut b = StableHasher::new();
        b.write_bytes(&[1]);
        assert_ne!(a.finish(), b.finish());
    }
}
