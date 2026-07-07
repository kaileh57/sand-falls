//! Saving and restoring the whole simulation, bit for bit.

use crate::sim::{SimConfig, Simulation, StreamConfig};
use crate::worldgen::WorldGen;
use fe_core::{Cell, ChunkPos, IRect, LocalRect};
use fe_materials::MaterialSet;
use fe_physics::Physics;
use fe_sim::{Avatar, Particles, PlayerInput, SupportSeed};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Bumped to 6 when bodies began saving their hull envelope and fluid state,
/// and to 7 when liquid cells began storing their velocity in the two top flag
/// bits and their value.
const FORMAT_VERSION: u32 = 7;

#[derive(Serialize, Deserialize)]
struct ChunkData {
    pos: ChunkPos,
    cells: Vec<u32>,
    dirty: LocalRect,
    next: LocalRect,
    solid_changed: bool,
}

#[derive(Serialize, Deserialize)]
struct SnapshotData {
    version: u32,
    materials_hash: u64,
    seed: u64,
    tick: u64,
    wind: i8,
    chunks: Vec<ChunkData>,
    particles: Particles,
    physics: Physics,
    avatars: Vec<Avatar>,
    support_seeds: Vec<SupportSeed>,
    last_inputs: Vec<(u8, PlayerInput)>,
    worldgen: WorldGen,
    stream: Option<StreamConfig>,
    stored: Vec<(ChunkPos, Vec<u32>)>,
    tool_masks: Vec<(u8, u16)>,
    tool_zones: Vec<(u8, IRect)>,
    status: String,
    script_state: Vec<u8>,
}

/// An error from loading a snapshot.
#[derive(Debug)]
pub enum SnapshotError {
    /// The bytes are not a snapshot this build can read.
    Corrupt(String),
    /// The snapshot was saved with a different material set.
    MaterialMismatch {
        /// Hash stored in the snapshot.
        expected: u64,
        /// Hash of the material set provided.
        found: u64,
    },
    /// The snapshot was written by a different format version.
    Version(u32),
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Corrupt(e) => write!(f, "snapshot is corrupt: {e}"),
            Self::MaterialMismatch { expected, found } => write!(
                f,
                "snapshot needs material set {expected:016x}, but {found:016x} was provided"
            ),
            Self::Version(v) => write!(
                f,
                "snapshot format {v} is not supported (this build reads {FORMAT_VERSION})"
            ),
        }
    }
}

impl std::error::Error for SnapshotError {}

impl Simulation {
    /// Serializes the whole simulation into a compressed byte string.
    pub fn snapshot(&self) -> Vec<u8> {
        let chunks = self
            .world
            .grid
            .chunks()
            .map(|c| ChunkData {
                pos: c.pos(),
                cells: c.to_vec().into_iter().map(Cell::to_bits).collect(),
                dirty: c.dirty(),
                next: c.pending(),
                solid_changed: c.solid_changed(),
            })
            .collect();
        let data = SnapshotDataRef {
            version: FORMAT_VERSION,
            materials_hash: self.materials().hash(),
            seed: self.world.seed(),
            tick: self.world.tick(),
            wind: self.world.wind(),
            chunks,
            particles: &self.particles,
            physics: &self.physics,
            avatars: &self.avatars,
            support_seeds: &self.support_seeds,
            last_inputs: self.last_inputs.iter().map(|(k, v)| (*k, *v)).collect(),
            worldgen: &self.worldgen,
            stream: self.stream,
            stored: self.stored.iter().map(|(k, v)| (*k, v.clone())).collect(),
            tool_masks: self.tool_masks.iter().map(|(k, v)| (*k, *v)).collect(),
            tool_zones: self.tool_zones.iter().map(|(k, v)| (*k, *v)).collect(),
            status: &self.status,
            script_state: &self.script_state,
        };
        let raw = postcard::to_allocvec(&data).unwrap_or_default();
        zstd::encode_all(raw.as_slice(), 3).unwrap_or(raw)
    }

    /// Rebuilds a simulation from [`Simulation::snapshot`] output.
    pub fn restore(
        bytes: &[u8],
        mats: Arc<MaterialSet>,
        threads: usize,
    ) -> Result<Self, SnapshotError> {
        let raw = zstd::decode_all(bytes).map_err(|e| SnapshotError::Corrupt(e.to_string()))?;
        let d: SnapshotData =
            postcard::from_bytes(&raw).map_err(|e| SnapshotError::Corrupt(e.to_string()))?;
        if d.version != FORMAT_VERSION {
            return Err(SnapshotError::Version(d.version));
        }
        if d.materials_hash != mats.hash() {
            return Err(SnapshotError::MaterialMismatch {
                expected: d.materials_hash,
                found: mats.hash(),
            });
        }
        let cfg = SimConfig { seed: d.seed, threads, worldgen: d.worldgen, stream: d.stream };
        let mut sim = Simulation::new(mats, cfg);
        sim.world.set_tick(d.tick);
        sim.world.set_wind(d.wind);
        for c in d.chunks {
            if c.cells.len() != fe_core::CHUNK_AREA {
                return Err(SnapshotError::Corrupt(format!(
                    "chunk {:?} has {} cells",
                    c.pos,
                    c.cells.len()
                )));
            }
            let mut chunk = fe_sim::Chunk::new(c.pos, Cell::AIR);
            let cells: Vec<Cell> = c.cells.into_iter().map(Cell::from_bits).collect();
            chunk.load_cells(&cells);
            chunk.set_schedule(c.dirty, c.next);
            chunk.set_solid_changed(c.solid_changed);
            sim.world.grid.insert(chunk);
        }
        sim.particles = d.particles;
        sim.physics = d.physics;
        sim.avatars = d.avatars;
        sim.support_seeds = d.support_seeds;
        sim.last_inputs = d.last_inputs.into_iter().collect();
        sim.stored = d.stored.into_iter().collect();
        sim.tool_masks = d.tool_masks.into_iter().collect();
        sim.tool_zones = d.tool_zones.into_iter().collect();
        sim.status = d.status;
        sim.script_state = d.script_state;
        Ok(sim)
    }
}

/// Borrowing twin of [`SnapshotData`], so saving does not clone the physics world.
#[derive(Serialize)]
struct SnapshotDataRef<'a> {
    version: u32,
    materials_hash: u64,
    seed: u64,
    tick: u64,
    wind: i8,
    chunks: Vec<ChunkData>,
    particles: &'a Particles,
    physics: &'a Physics,
    avatars: &'a Vec<Avatar>,
    support_seeds: &'a Vec<SupportSeed>,
    last_inputs: Vec<(u8, PlayerInput)>,
    worldgen: &'a WorldGen,
    stream: Option<StreamConfig>,
    stored: Vec<(ChunkPos, Vec<u32>)>,
    tool_masks: Vec<(u8, u16)>,
    tool_zones: Vec<(u8, IRect)>,
    status: &'a str,
    script_state: &'a [u8],
}
