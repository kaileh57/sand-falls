use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

/// A signed 16.16 fixed-point number.
///
/// Used for particle and character positions and velocities so that motion is
/// bit-identical on every platform.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct Fx(pub i32);

impl Fx {
    /// Number of fractional bits.
    pub const FRAC_BITS: u32 = 16;
    /// The value 0.
    pub const ZERO: Self = Self(0);
    /// The value 1.
    pub const ONE: Self = Self(1 << Self::FRAC_BITS);

    /// Converts an integer.
    #[inline]
    pub const fn from_int(v: i32) -> Self {
        Self(v << Self::FRAC_BITS)
    }

    /// Builds `num / den` without floats. `den` must not be zero.
    #[inline]
    pub const fn from_ratio(num: i32, den: i32) -> Self {
        Self(((num as i64) << Self::FRAC_BITS) as i32 / den)
    }

    /// Rounds toward negative infinity.
    #[inline]
    pub const fn floor(self) -> i32 {
        self.0 >> Self::FRAC_BITS
    }

    /// Rounds to the nearest integer; halves round toward positive infinity.
    #[inline]
    pub const fn round(self) -> i32 {
        (self.0 + (1 << (Self::FRAC_BITS - 1))) >> Self::FRAC_BITS
    }

    /// Absolute value.
    #[inline]
    #[must_use]
    pub const fn abs(self) -> Self {
        Self(self.0.wrapping_abs())
    }

    /// Fixed-point multiply.
    #[inline]
    #[must_use]
    pub const fn mul_fx(self, o: Self) -> Self {
        Self(((self.0 as i64 * o.0 as i64) >> Self::FRAC_BITS) as i32)
    }

    /// Fixed-point divide. Returns zero when dividing by zero.
    #[inline]
    #[must_use]
    pub const fn div_fx(self, o: Self) -> Self {
        if o.0 == 0 {
            Self::ZERO
        } else {
            Self((((self.0 as i64) << Self::FRAC_BITS) / o.0 as i64) as i32)
        }
    }

    /// Clamps between two values.
    #[inline]
    #[must_use]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        Self(self.0.clamp(lo.0, hi.0))
    }
}

impl Add for Fx {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self(self.0.wrapping_add(o.0))
    }
}

impl Sub for Fx {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self(self.0.wrapping_sub(o.0))
    }
}

impl Neg for Fx {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(self.0.wrapping_neg())
    }
}

impl Mul<i32> for Fx {
    type Output = Self;
    #[inline]
    fn mul(self, o: i32) -> Self {
        Self(self.0.wrapping_mul(o))
    }
}

impl AddAssign for Fx {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}

impl SubAssign for Fx {
    #[inline]
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}

/// Integer square root, rounded down.
pub const fn isqrt(v: u64) -> u64 {
    if v < 2 {
        return v;
    }
    // Newton's method on integers converges from above.
    let mut x = v;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = u64::midpoint(x, v / x);
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn basic_arithmetic() {
        let a = Fx::from_ratio(3, 2);
        assert_eq!((a + a).floor(), 3);
        assert_eq!(a.mul_fx(Fx::from_int(4)).floor(), 6);
        assert_eq!(Fx::from_int(-1).floor(), -1);
        assert_eq!(Fx::from_ratio(-1, 2).floor(), -1);
        assert_eq!(Fx::from_int(7).div_fx(Fx::from_int(2)), Fx::from_ratio(7, 2));
        assert_eq!(Fx::ONE.div_fx(Fx::ZERO), Fx::ZERO);
    }

    proptest! {
        #[test]
        fn isqrt_is_floor_sqrt(v in 0u64..(1u64 << 52)) {
            let r = isqrt(v);
            prop_assert!(r * r <= v);
            prop_assert!((r + 1) * (r + 1) > v);
        }
    }
}
