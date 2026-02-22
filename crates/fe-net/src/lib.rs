//! Lockstep multiplayer.
//!
//! Every peer runs the same deterministic simulation; only inputs travel. The
//! server orders inputs into confirmed ticks and is the final authority. Clients
//! predict their own character, compare state hashes every
//! [`HASH_INTERVAL`] ticks, and get damaged chunks resent (or a full snapshot)
//! when a hash disagrees.
//!
//! The cores in [`server`] and [`client`] never touch sockets or clocks; they
//! take the current time and a [`Transport`]. That lets the same code run over
//! UDP and over [`SimNet`], a simulated network with latency and loss used by
//! tests and the soak demo.
#![forbid(unsafe_code)]

pub mod client;
pub mod server;
pub mod soak;
mod transport;

pub use transport::{Peer, SimNet, SimNetConfig, Transport, UdpTransport};

use fe_core::ChunkPos;
use fe_engine::{HashParts, PlayerInput};
use serde::{Deserialize, Serialize};

/// Wire protocol version. Peers with a different version are rejected.
pub const PROTOCOL: u16 = 1;
/// Ticks between hash checks.
pub const HASH_INTERVAL: u64 = 30;
/// Bytes of snapshot data per packet.
pub const SNAPSHOT_PART: usize = 1000;
/// Simulation tick length in microseconds (60 Hz).
pub const TICK_US: u64 = 16_667;

const MAGIC: [u8; 2] = *b"FE";

/// One wire message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Msg {
    /// Client asks to join.
    Hello {
        /// Must equal [`PROTOCOL`].
        protocol: u16,
        /// Display name.
        name: String,
        /// Hash of the client's material set.
        materials: u64,
        /// Hash of the client's scene text.
        scene: u64,
    },
    /// Client input as runs: each entry is the input from its tick until the next entry.
    Inputs {
        /// Last tick these runs cover.
        upto: u64,
        /// `(first tick, input)` runs in tick order.
        runs: Vec<(u64, PlayerInput)>,
        /// Highest confirmed tick the client has received with no gaps.
        ack: u64,
    },
    /// Client state hash after a tick.
    HashReport {
        /// Tick the hash is for.
        tick: u64,
        /// Hash by subsystem.
        parts: HashParts,
    },
    /// Client per-chunk hashes for a tick, sent when the server asks.
    ChunkHashes {
        /// Tick the hashes are for.
        tick: u64,
        /// Chunk hashes in canonical order.
        hashes: Vec<(ChunkPos, u64)>,
    },
    /// Client reports which snapshot parts it still needs.
    SnapshotAck {
        /// Snapshot id.
        id: u32,
        /// Part numbers not yet received.
        missing: Vec<u32>,
    },
    /// Round-trip probe, client time in microseconds.
    Ping(u64),
    /// Leaving.
    Bye,
    /// Server accepts a client.
    Welcome {
        /// Assigned player id.
        player: u8,
    },
    /// Server refuses a client.
    Reject(String),
    /// One piece of a compressed snapshot.
    SnapshotPart {
        /// Snapshot id.
        id: u32,
        /// Part number.
        part: u32,
        /// Number of parts.
        total: u32,
        /// Bytes.
        data: Vec<u8>,
    },
    /// Confirmed inputs and commands for ticks `from..upto`.
    Frames {
        /// First tick covered.
        from: u64,
        /// One past the last tick covered.
        upto: u64,
        /// `(player, first tick, input)` runs.
        runs: Vec<(u8, u64, PlayerInput)>,
        /// `(tick, command as JSON)` in apply order.
        commands: Vec<(u64, String)>,
    },
    /// Server asks for chunk hashes at a checked tick.
    NeedChunkHashes(u64),
    /// Server's chunks to overwrite once the client's simulation reaches `tick`.
    Repair {
        /// Apply after this many ticks have run.
        tick: u64,
        /// `(chunk, zstd-compressed little-endian cell bits)`.
        chunks: Vec<(ChunkPos, Vec<u8>)>,
    },
    /// Echo of a [`Msg::Ping`].
    Pong(u64),
}

/// A message that could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeError(pub String);

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bad packet: {}", self.0)
    }
}

impl std::error::Error for DecodeError {}

/// Encodes a message for the wire.
pub fn encode(m: &Msg) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    out.extend(postcard::to_allocvec(m).unwrap_or_default());
    out
}

/// Decodes a wire message. Never panics on any input.
pub fn decode(bytes: &[u8]) -> Result<Msg, DecodeError> {
    let body = bytes.strip_prefix(&MAGIC).ok_or_else(|| DecodeError("missing magic".into()))?;
    postcard::from_bytes(body).map_err(|e| DecodeError(e.to_string()))
}

/// Compresses a chunk's cell bits for a repair message.
pub fn pack_chunk(bits: &[u32]) -> Vec<u8> {
    let raw: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    zstd::encode_all(raw.as_slice(), 3).unwrap_or(raw)
}

/// Reverses [`pack_chunk`].
pub fn unpack_chunk(data: &[u8]) -> Option<Vec<u32>> {
    let raw = zstd::decode_all(data).ok()?;
    if raw.len() != fe_core::CHUNK_AREA * 4 {
        return None;
    }
    Some(raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Hash of scene source text, so peers can check they loaded the same scene.
pub fn scene_hash(text: &str) -> u64 {
    let mut h = fe_core::StableHasher::new();
    h.write_bytes(text.as_bytes());
    h.finish()
}

/// Byte and packet counters for one peer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Traffic {
    /// Bytes sent.
    pub sent: u64,
    /// Bytes received.
    pub received: u64,
    /// Packets sent.
    pub packets_sent: u64,
    /// Packets received.
    pub packets_received: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_round_trip() {
        let msgs = vec![
            Msg::Hello { protocol: PROTOCOL, name: "a".into(), materials: 1, scene: 2 },
            Msg::Inputs {
                upto: 9,
                runs: vec![(3, PlayerInput { move_x: -1, jump: true, ..Default::default() })],
                ack: 2,
            },
            Msg::Frames {
                from: 1,
                upto: 5,
                runs: vec![(0, 1, PlayerInput::default())],
                commands: vec![(4, "{}".into())],
            },
            Msg::Repair {
                tick: 7,
                chunks: vec![(ChunkPos::new(-1, 2), pack_chunk(&vec![5; fe_core::CHUNK_AREA]))],
            },
            Msg::Pong(77),
        ];
        for m in msgs {
            assert_eq!(decode(&encode(&m)), Ok(m));
        }
    }

    #[test]
    fn garbage_is_rejected() {
        assert!(decode(b"").is_err());
        assert!(decode(b"FE\xff\xff\xff").is_err());
        assert!(decode(b"XX").is_err());
    }

    #[test]
    fn chunk_packing_round_trips() {
        let bits: Vec<u32> = (0..fe_core::CHUNK_AREA as u32).collect();
        assert_eq!(unpack_chunk(&pack_chunk(&bits)), Some(bits));
        assert_eq!(unpack_chunk(b"nope"), None);
    }
}
