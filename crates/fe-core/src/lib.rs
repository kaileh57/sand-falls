//! Core types shared by every falling-engine crate.
//!
//! Everything in this crate is deterministic: integer and fixed-point math only,
//! no wall-clock time, and no hash maps with random state.
#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic)]

mod cell;
mod coords;
mod fixed;
mod hash;
mod posset;
mod rng;

pub use cell::{Cell, MaterialId};
pub use coords::{CHUNK_AREA, CHUNK_SHIFT, CHUNK_SIZE, CellPos, ChunkPos, IRect, LocalRect};
pub use fixed::{Fx, isqrt};
pub use hash::StableHasher;
pub use posset::PosSet;
pub use rng::{SeqRng, cell_rand, mix64};

/// Fixed simulation rate in ticks per second.
pub const TICKS_PER_SECOND: u32 = 60;

/// Ambient temperature in engine units (20 degrees C).
pub const AMBIENT_TEMP: u8 = 7;

/// Converts degrees Celsius to the engine's 8-bit temperature scale.
///
/// The scale covers -50 C (0) to 2500 C (255) in steps of 10 degrees.
pub const fn celsius_to_temp(celsius: i32) -> u8 {
    let t = (celsius + 50) / 10;
    if t < 0 {
        0
    } else if t > 255 {
        255
    } else {
        t as u8
    }
}

/// Converts the engine's 8-bit temperature to degrees Celsius.
pub const fn temp_to_celsius(temp: u8) -> i32 {
    temp as i32 * 10 - 50
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_scale_round_trips_on_ten_degree_steps() {
        assert_eq!(celsius_to_temp(20), AMBIENT_TEMP);
        assert_eq!(temp_to_celsius(AMBIENT_TEMP), 20);
        assert_eq!(celsius_to_temp(-1000), 0);
        assert_eq!(celsius_to_temp(100_000), 255);
        for t in 0..=255u8 {
            assert_eq!(celsius_to_temp(temp_to_celsius(t)), t);
        }
    }
}
