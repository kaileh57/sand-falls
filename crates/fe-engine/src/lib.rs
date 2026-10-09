//! The full falling-engine tick.
//!
//! [`Simulation`] owns the cell world, particles, rigid bodies and characters,
//! and runs them in a fixed order each tick:
//!
//! 1. commands and character tools
//! 2. cell passes (four checkerboard passes)
//! 3. explosions raised by the cell rules
//! 4. free particles
//! 5. body damage sync, structural support checks, detachment
//! 6. unstamp bodies, physics step, restamp
//! 7. characters
//! 8. chunk streaming
//!
//! Given the same starting state and the same inputs and commands, every step
//! produces the same [`Simulation::hash`] on every platform and thread count.
#![forbid(unsafe_code)]

mod command;
mod sim;
mod snapshot;
#[deny(clippy::float_arithmetic)]
mod worldgen;

pub use command::Command;
pub use fe_physics::JointKind;
pub use fe_sim::{Avatar, PlayerInput};
pub use sim::{
    ChunkState, HashParts, PhaseTimes, STREAM_FADE_TICKS, SimConfig, Simulation, StreamConfig,
    TickReport, tools,
};
pub use snapshot::SnapshotError;
pub use worldgen::WorldGen;

#[cfg(test)]
mod tests;
