//! Deterministic cell simulation.
//!
//! The world is a grid of 64x64 chunks. Each tick, chunks with changes are
//! updated in four checkerboard passes, so neighboring chunks never update at
//! the same time and the result is the same for any number of threads.
#![deny(clippy::float_arithmetic)]

mod avatar;
mod explode;
mod grid;
mod particles;
mod rules;
#[allow(unsafe_code, reason = "the scheduler shares chunk cells between threads; see module docs")]
mod sched;
mod support;
mod world;

pub use avatar::{AVATAR_H, AVATAR_W, Avatar, PlayerInput};
pub use explode::{Blast, explode};
pub use grid::Grid;
pub use particles::{PARTICLE_GRAVITY, Particle, Particles};
pub use rules::{Event, GRAVITY, MAX_SPEED, RuleCtx, SPEED_UNITS, new_cell};
pub use sched::{Chunk, Neighborhood, REACH};
pub use support::{Detached, SUPPORT_RETRIES, SUPPORT_SEARCH_LIMIT, SupportSeed, check_support};
pub use world::{CellStats, World};
