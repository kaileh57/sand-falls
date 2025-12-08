//! Deterministic cell simulation.
//!
//! The world is a grid of 64x64 chunks. Each tick, chunks with changes are
//! updated in four checkerboard passes, so neighboring chunks never update at
//! the same time and the result is the same for any number of threads.
#![deny(clippy::float_arithmetic)]

mod grid;
mod rules;
#[allow(unsafe_code, reason = "the scheduler shares chunk cells between threads; see module docs")]
mod sched;
mod world;

pub use grid::Grid;
pub use rules::{Event, GRAVITY, MAX_SPEED, RuleCtx, SPEED_UNITS, new_cell};
pub use sched::{Chunk, Neighborhood, REACH};
pub use world::{CellStats, World};
