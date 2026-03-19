//! The authoritative server core.
//!
//! The server runs the same simulation as every client. Each tick it takes the
//! latest input each player has sent for that tick (or earlier), steps, records
//! the confirmed inputs, and sends every client the confirmed ticks it has not
//! acknowledged yet. It never waits for a slow client.

use crate::{
    HASH_INTERVAL, Msg, PROTOCOL, Peer, SNAPSHOT_PART, Traffic, Transport, decode, encode,
    pack_chunk,
};
use fe_core::ChunkPos;
use fe_engine::{Command, HashParts, PlayerInput};
use fe_scene::Runner;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Confirmed history kept for clients that fall behind, in ticks.
const HISTORY: u64 = 1200;
/// Hash checkpoints kept for comparing late reports.
const CHECKPOINTS: usize = 40;
/// A client silent this long is dropped, in microseconds.
const TIMEOUT_US: u64 = 5_000_000;
/// Most confirmed ticks sent in one frames packet.
const MAX_FRAMES: u64 = 120;
/// Mismatched hash reports in a row before a full resync.
const RESYNC_AFTER: u32 = 3;

/// Server counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerStats {
    /// Players that joined.
    pub joins: u32,
    /// Players that left or timed out.
    pub leaves: u32,
    /// Hash reports received.
    pub reports: u32,
    /// Hash reports that did not match.
    pub desyncs: u32,
    /// Chunk repairs sent.
    pub repairs: u32,
    /// Full snapshot resyncs sent (joins not counted).
    pub resyncs: u32,
    /// Snapshot bytes sent, including joins.
    pub snapshot_bytes: u64,
}

struct Outgoing {
    id: u32,
    parts: Vec<Vec<u8>>,
    pending: BTreeSet<u32>,
    cursor: u32,
}

struct ClientState {
    player: u8,
    name: String,
    runs: BTreeMap<u64, PlayerInput>,
    ack: u64,
    synced: bool,
    want_snapshot: bool,
    snapshot: Option<Outgoing>,
    mismatches: u32,
    last_seen: u64,
    traffic: Traffic,
}

/// One connected client, as seen by the server.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientInfo {
    /// Player id.
    pub player: u8,
    /// Display name.
    pub name: String,
    /// Has the full state and is following confirmed ticks.
    pub synced: bool,
    /// Highest confirmed tick acknowledged.
    pub ack: u64,
    /// Bytes and packets.
    pub traffic: Traffic,
}

/// The server.
pub struct ServerCore {
    /// The authoritative simulation and scene.
    pub runner: Runner,
    clients: BTreeMap<Peer, ClientState>,
    history: BTreeMap<u64, (BTreeMap<u8, PlayerInput>, Vec<Command>)>,
    checkpoints: BTreeMap<u64, (HashParts, Vec<(ChunkPos, u64)>)>,
    pending: Vec<Command>,
    materials: u64,
    scene: u64,
    spawns: Vec<(i32, i32)>,
    max_players: u8,
    next_snapshot: u32,
    snapshot_rate: usize,
    /// Counters.
    pub stats: ServerStats,
}

impl ServerCore {
    /// Creates a server for a running scene. `scene_hash` is [`crate::scene_hash`] of its text.
    pub fn new(runner: Runner, scene_hash: u64, max_players: u8) -> Self {
        let b = runner.scene.bounds;
        let mut spawns = runner.scene.spawns.clone();
        if spawns.is_empty() {
            spawns.push((i32::midpoint(b.min.x, b.max.x), b.min.y + 16));
        }
        let materials = runner.sim.materials().hash();
        Self {
            runner,
            clients: BTreeMap::new(),
            history: BTreeMap::new(),
            checkpoints: BTreeMap::new(),
            pending: Vec::new(),
            materials,
            scene: scene_hash,
            spawns,
            max_players: max_players.clamp(1, 64),
            next_snapshot: 1,
            snapshot_rate: 96,
            stats: ServerStats::default(),
        }
    }

    /// Current tick.
    pub fn tick_count(&self) -> u64 {
        self.runner.sim.tick()
    }

    /// Hash checkpoints by tick.
    pub fn checkpoint(&self, tick: u64) -> Option<HashParts> {
        self.checkpoints.get(&tick).map(|c| c.0)
    }

    /// Connected clients.
    pub fn clients(&self) -> Vec<ClientInfo> {
        self.clients
            .values()
            .map(|c| ClientInfo {
                player: c.player,
                name: c.name.clone(),
                synced: c.synced,
                ack: c.ack,
                traffic: c.traffic,
            })
            .collect()
    }

    fn send(&mut self, net: &mut impl Transport, to: Peer, m: &Msg) {
        let bytes = encode(m);
        if let Some(c) = self.clients.get_mut(&to) {
            c.traffic.sent += bytes.len() as u64;
            c.traffic.packets_sent += 1;
        }
        net.send(to, &bytes);
    }

    /// Reads and handles every waiting packet.
    pub fn poll(&mut self, now_us: u64, net: &mut impl Transport) {
        while let Some((from, bytes)) = net.recv() {
            if let Some(c) = self.clients.get_mut(&from) {
                c.traffic.received += bytes.len() as u64;
                c.traffic.packets_received += 1;
                c.last_seen = now_us;
            }
            if let Ok(m) = decode(&bytes) {
                self.handle(now_us, from, m, net);
            }
        }
    }

    fn handle(&mut self, now_us: u64, from: Peer, m: Msg, net: &mut impl Transport) {
        match m {
            Msg::Hello { protocol, name, materials, scene } => {
                if self.clients.contains_key(&from) {
                    return;
                }
                let reason = if protocol != PROTOCOL {
                    Some(format!("protocol {protocol} does not match server protocol {PROTOCOL}"))
                } else if materials != self.materials {
                    Some("material set differs from the server's".to_string())
                } else if scene != self.scene {
                    Some("scene differs from the server's".to_string())
                } else if self.clients.len() >= self.max_players as usize {
                    Some("server is full".to_string())
                } else {
                    None
                };
                if let Some(r) = reason {
                    net.send(from, &encode(&Msg::Reject(r)));
                    return;
                }
                let used: BTreeSet<u8> = self.clients.values().map(|c| c.player).collect();
                let player = (0..=255u8).find(|p| !used.contains(p)).unwrap_or(0);
                self.clients.insert(
                    from,
                    ClientState {
                        player,
                        name,
                        runs: BTreeMap::new(),
                        ack: 0,
                        synced: false,
                        want_snapshot: true,
                        snapshot: None,
                        mismatches: 0,
                        last_seen: now_us,
                        traffic: Traffic::default(),
                    },
                );
                self.stats.joins += 1;
                let n = self.spawns.len();
                let (sx, sy) = self.spawns[player as usize % n];
                let x = sx + (player as usize / n) as i32 * 14;
                self.pending.push(Command::AddPlayer { id: player, x, y: sy });
                self.send(net, from, &Msg::Welcome { player });
            }
            Msg::Inputs { upto: _, runs, ack } => {
                let Some(c) = self.clients.get_mut(&from) else { return };
                for (t, i) in runs {
                    c.runs.insert(t, i);
                }
                if c.synced {
                    c.ack = c.ack.max(ack);
                }
            }
            Msg::SnapshotAck { id, missing } => {
                let Some(c) = self.clients.get_mut(&from) else { return };
                let Some(out) = c.snapshot.as_mut() else { return };
                if out.id != id {
                    return;
                }
                if missing.is_empty() {
                    c.snapshot = None;
                    c.synced = true;
                } else {
                    out.pending =
                        missing.into_iter().filter(|p| (*p as usize) < out.parts.len()).collect();
                    out.cursor = 0;
                }
            }
            Msg::HashReport { tick, parts } => {
                self.stats.reports += 1;
                let Some(&(mine, _)) = self.checkpoints.get(&tick) else { return };
                let Some(c) = self.clients.get_mut(&from) else { return };
                if mine == parts {
                    c.mismatches = 0;
                    return;
                }
                self.stats.desyncs += 1;
                c.mismatches += 1;
                let only_cells = mine.particles == parts.particles
                    && mine.bodies == parts.bodies
                    && mine.avatars == parts.avatars;
                if c.mismatches >= RESYNC_AFTER || !only_cells {
                    c.want_snapshot = true;
                    c.synced = false;
                    c.mismatches = 0;
                    self.stats.resyncs += 1;
                } else {
                    self.send(net, from, &Msg::NeedChunkHashes(tick));
                }
            }
            Msg::ChunkHashes { tick, hashes } => {
                let Some((_, mine)) = self.checkpoints.get(&tick) else { return };
                let theirs: BTreeMap<ChunkPos, u64> = hashes.into_iter().collect();
                let mut bad = BTreeSet::new();
                for (p, h) in mine {
                    if theirs.get(p) != Some(h) {
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                bad.insert(ChunkPos::new(p.x + dx, p.y + dy));
                            }
                        }
                    }
                }
                let now = self.runner.sim.tick();
                let chunks: Vec<(ChunkPos, Vec<u8>)> = bad
                    .into_iter()
                    .filter_map(|p| self.runner.sim.chunk_bits(p).map(|b| (p, pack_chunk(&b))))
                    .collect();
                for batch in chunks.chunks(6) {
                    self.send(net, from, &Msg::Repair { tick: now, chunks: batch.to_vec() });
                }
                self.stats.repairs += 1;
            }
            Msg::Ping(t) => self.send(net, from, &Msg::Pong(t)),
            Msg::Bye => self.drop_client(from),
            _ => {}
        }
    }

    fn drop_client(&mut self, peer: Peer) {
        if let Some(c) = self.clients.remove(&peer) {
            self.pending.push(Command::RemovePlayer { id: c.player });
            self.stats.leaves += 1;
        }
    }

    /// Adds a command to apply on the next tick, for host tools and scripted events.
    pub fn queue_command(&mut self, cmd: Command) {
        self.pending.push(cmd);
    }

    /// Runs one tick and sends updates. Call at 60 Hz.
    pub fn tick(&mut self, now_us: u64, net: &mut impl Transport) {
        let stale: Vec<Peer> = self
            .clients
            .iter()
            .filter(|(_, c)| now_us.saturating_sub(c.last_seen) > TIMEOUT_US)
            .map(|(p, _)| *p)
            .collect();
        for p in stale {
            self.drop_client(p);
        }

        let t = self.runner.sim.tick();
        let mut inputs = BTreeMap::new();
        for c in self.clients.values_mut() {
            if let Some((_, i)) = c.runs.range(..=t).next_back() {
                inputs.insert(c.player, *i);
            }
            // Drop runs that can never matter again, keeping the latest one.
            let cutoff = t.saturating_sub(HISTORY);
            let keep_from = c.runs.range(..=cutoff).next_back().map(|(k, _)| *k);
            if let Some(k) = keep_from {
                c.runs = c.runs.split_off(&k);
            }
        }
        let commands = std::mem::take(&mut self.pending);
        self.runner.step_with(&inputs, &commands);
        self.history.insert(t, (inputs, commands));
        while self.history.len() as u64 > HISTORY {
            self.history.pop_first();
        }
        let now = self.runner.sim.tick();
        if now.is_multiple_of(HASH_INTERVAL) {
            self.checkpoints
                .insert(now, (self.runner.sim.hash_parts(), self.runner.sim.chunk_hashes()));
            while self.checkpoints.len() > CHECKPOINTS {
                self.checkpoints.pop_first();
            }
        }

        let peers: Vec<Peer> = self.clients.keys().copied().collect();
        for peer in peers {
            self.send_updates(peer, now, false, net);
        }
    }

    /// Sends pending updates without running a tick, for shutting down or pausing.
    pub fn flush(&mut self, net: &mut impl Transport) {
        let now = self.runner.sim.tick();
        let peers: Vec<Peer> = self.clients.keys().copied().collect();
        for peer in peers {
            self.send_updates(peer, now, true, net);
        }
    }

    fn send_updates(&mut self, peer: Peer, now: u64, force: bool, net: &mut impl Transport) {
        let Some(c) = self.clients.get_mut(&peer) else { return };
        if c.want_snapshot {
            let bytes = self.runner.sim.snapshot();
            self.stats.snapshot_bytes += bytes.len() as u64;
            let parts: Vec<Vec<u8>> = bytes.chunks(SNAPSHOT_PART).map(<[u8]>::to_vec).collect();
            let id = self.next_snapshot;
            self.next_snapshot += 1;
            c.snapshot =
                Some(Outgoing { id, pending: (0..parts.len() as u32).collect(), parts, cursor: 0 });
            c.want_snapshot = false;
            c.synced = false;
            c.ack = now;
        }
        if let Some(out) = c.snapshot.as_mut() {
            let total = out.parts.len() as u32;
            let id = out.id;
            let batch: Vec<u32> =
                out.pending.range(out.cursor..).take(self.snapshot_rate).copied().collect();
            out.cursor = batch.last().map_or(0, |l| l + 1);
            if batch.is_empty() {
                out.cursor = 0;
            }
            let msgs: Vec<Msg> = batch
                .into_iter()
                .map(|part| Msg::SnapshotPart {
                    id,
                    part,
                    total,
                    data: out.parts[part as usize].clone(),
                })
                .collect();
            for m in msgs {
                self.send(net, peer, &m);
            }
            return;
        }
        if !c.synced || (!force && !now.is_multiple_of(2) && now > c.ack + 1) {
            return;
        }
        let from = c.ack.max(self.history.keys().next().copied().unwrap_or(0));
        if from >= now {
            return;
        }
        let upto = now.min(from + MAX_FRAMES);
        let mut runs = Vec::new();
        let mut last: BTreeMap<u8, PlayerInput> = BTreeMap::new();
        let mut commands = Vec::new();
        for (t, (inputs, cmds)) in self.history.range(from..upto) {
            for (p, i) in inputs {
                if last.get(p) != Some(i) {
                    runs.push((*p, *t, *i));
                    last.insert(*p, *i);
                }
            }
            // A player with no input this tick is idle; record that as a run too.
            let gone: Vec<u8> = last.keys().filter(|p| !inputs.contains_key(p)).copied().collect();
            for p in gone {
                if last.get(&p) != Some(&PlayerInput::default()) {
                    runs.push((p, *t, PlayerInput::default()));
                    last.insert(p, PlayerInput::default());
                }
            }
            for cmd in cmds {
                commands.push((*t, serde_json_string(cmd)));
            }
        }
        self.send(net, peer, &Msg::Frames { from, upto, runs, commands });
    }
}

fn serde_json_string(c: &Command) -> String {
    serde_json::to_string(c).unwrap_or_default()
}
