//! The authoritative server core.
//!
//! The server runs the same simulation as every client. Each tick it takes the
//! latest input each player has sent for that tick (or earlier), steps, records
//! the confirmed inputs, and sends every client the confirmed ticks it has not
//! acknowledged yet. It never waits for a slow client.

use crate::{
    HASH_INTERVAL, Msg, PROTOCOL, Peer, RepairChunk, SNAPSHOT_PART, Traffic, Transport, decode,
    encode,
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
/// Ticks a client's frames stop short of a repair's tick after the repair is
/// sent, so the repair arrives before the frames that reach that tick.
const REPAIR_HOLD: u64 = 6;
/// Furthest ahead of the server an input run may be, in ticks.
const MAX_LEAD: u64 = 2 * HISTORY;
/// Most input runs read from one packet.
const MAX_RUNS_PER_PACKET: usize = 512;
/// Most input runs kept per client.
const MAX_RUNS_STORED: usize = 4096;
/// Frames packets stop adding ticks past this many bytes of runs and commands.
const FRAMES_BUDGET: usize = 1200;

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
    /// Checkpoint tick whose chunk hashes the server asked this client for.
    asked_chunks: Option<u64>,
    /// Chunk hashes received for a checkpoint, repaired at the end of the next tick.
    repair_due: Option<(u64, Vec<(ChunkPos, u64)>)>,
    /// `(repair tick, server tick)`: until the server tick, frames stop short of
    /// the repair tick.
    hold: Option<(u64, u64)>,
    /// Server tick of the last repair sent.
    last_repair: Option<u64>,
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
    mods: u64,
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
        let mods = runner.mods_hash();
        Self {
            runner,
            clients: BTreeMap::new(),
            history: BTreeMap::new(),
            checkpoints: BTreeMap::new(),
            pending: Vec::new(),
            materials,
            scene: scene_hash,
            mods,
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
            Msg::Hello { protocol, name, materials, scene, mods } => {
                if let Some(c) = self.clients.get(&from) {
                    // The client did not get its welcome: send it again.
                    let player = c.player;
                    self.send(net, from, &Msg::Welcome { player });
                    return;
                }
                let reason = if protocol != PROTOCOL {
                    Some(format!("protocol {protocol} does not match server protocol {PROTOCOL}"))
                } else if materials != self.materials {
                    Some("material set differs from the server's".to_string())
                } else if scene != self.scene {
                    Some("scene differs from the server's".to_string())
                } else if mods != self.mods {
                    Some(format!(
                        "game script differs from the server's (client mods hash {mods:016x}, \
                         server {:016x})",
                        self.mods
                    ))
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
                        asked_chunks: None,
                        repair_due: None,
                        hold: None,
                        last_repair: None,
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
                let now = self.runner.sim.tick();
                let Some(c) = self.clients.get_mut(&from) else {
                    Self::reject_unknown(from, net);
                    return;
                };
                for (t, i) in runs.into_iter().take(MAX_RUNS_PER_PACKET) {
                    // Runs far in the future would never be pruned.
                    if t > now.saturating_add(MAX_LEAD)
                        || (c.runs.len() >= MAX_RUNS_STORED && !c.runs.contains_key(&t))
                    {
                        continue;
                    }
                    c.runs.insert(t, i);
                }
                if c.synced {
                    c.ack = c.ack.max(ack.min(now));
                }
            }
            Msg::SnapshotAck { id, missing } => {
                let Some(c) = self.clients.get_mut(&from) else {
                    Self::reject_unknown(from, net);
                    return;
                };
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
                let now = self.runner.sim.tick();
                let Some(c) = self.clients.get_mut(&from) else {
                    Self::reject_unknown(from, net);
                    return;
                };
                self.stats.reports += 1;
                let Some(&(mine, _)) = self.checkpoints.get(&tick) else { return };
                if mine == parts {
                    c.mismatches = 0;
                    return;
                }
                self.stats.desyncs += 1;
                c.mismatches += 1;
                let only_cells = mine.particles == parts.particles
                    && mine.bodies == parts.bodies
                    && mine.avatars == parts.avatars
                    && mine.game == parts.game;
                if c.mismatches >= RESYNC_AFTER || !only_cells {
                    c.want_snapshot = true;
                    c.synced = false;
                    c.mismatches = 0;
                    self.stats.resyncs += 1;
                } else if c.last_repair.is_none_or(|t| now >= t + HASH_INTERVAL) {
                    c.asked_chunks = Some(tick);
                    self.send(net, from, &Msg::NeedChunkHashes(tick));
                }
            }
            Msg::ChunkHashes { tick, hashes } => {
                let Some(c) = self.clients.get_mut(&from) else {
                    Self::reject_unknown(from, net);
                    return;
                };
                // Only answer the request this client was sent, once.
                if c.asked_chunks != Some(tick) || !self.checkpoints.contains_key(&tick) {
                    return;
                }
                c.asked_chunks = None;
                c.repair_due = Some((tick, hashes));
            }
            Msg::Ping(t) => self.send(net, from, &Msg::Pong(t)),
            Msg::Bye => self.drop_client(from),
            _ => {}
        }
    }

    /// Tells a peer that is not connected (it timed out, or never joined) so it
    /// stops waiting for frames.
    fn reject_unknown(peer: Peer, net: &mut impl Transport) {
        net.send(peer, &encode(&Msg::Reject("not connected".into())));
    }

    /// Sends a client the chunks that differed at the checkpoint it reported,
    /// as they are now, before any frames that reach this tick.
    fn send_repair(&mut self, peer: Peer, now: u64, net: &mut impl Transport) {
        let Some(c) = self.clients.get_mut(&peer) else { return };
        let Some((tick, hashes)) = c.repair_due.take() else { return };
        if !c.synced || c.want_snapshot {
            return;
        }
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
        let chunks: Vec<RepairChunk> = bad
            .into_iter()
            .filter_map(|p| self.runner.sim.chunk_state(p).map(|s| RepairChunk::pack(&s)))
            .collect();
        if chunks.is_empty() {
            return;
        }
        c.hold = Some((now, now + REPAIR_HOLD));
        c.last_repair = Some(now);
        for batch in chunks.chunks(6) {
            self.send(net, peer, &Msg::Repair { tick: now, chunks: batch.to_vec() });
        }
        self.stats.repairs += 1;
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
            self.send_repair(peer, now, net);
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
        let first_kept = self.history.keys().next().copied();
        if c.synced && first_kept.is_some_and(|k| k > c.ack) {
            // The client fell behind the confirmed history: frames can never catch it up.
            c.want_snapshot = true;
            self.stats.resyncs += 1;
        }
        if c.want_snapshot {
            c.asked_chunks = None;
            c.repair_due = None;
            c.hold = None;
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
        if !c.synced || (!force && !now.is_multiple_of(2) && now > c.ack.saturating_add(1)) {
            return;
        }
        let mut cap = now;
        if let Some((repair_tick, until)) = c.hold {
            if now < until {
                cap = cap.min(repair_tick.saturating_sub(1));
            } else {
                c.hold = None;
            }
        }
        let from = c.ack.max(first_kept.unwrap_or(0));
        if from >= cap {
            return;
        }
        let end = cap.min(from + MAX_FRAMES);
        let mut upto = from;
        let mut runs = Vec::new();
        let mut last: BTreeMap<u8, Option<PlayerInput>> = BTreeMap::new();
        let mut commands = Vec::new();
        let mut bytes = 0;
        for (t, (inputs, cmds)) in self.history.range(from..end) {
            let mut tick_runs = Vec::new();
            for (p, i) in inputs {
                if last.get(p) != Some(&Some(*i)) {
                    tick_runs.push((*p, *t, Some(*i)));
                }
            }
            // A player with no input from this tick on gets a run that says so.
            for (p, i) in &last {
                if i.is_some() && !inputs.contains_key(p) {
                    tick_runs.push((*p, *t, None));
                }
            }
            let tick_cmds: Vec<(u64, String)> =
                cmds.iter().map(|c| (*t, serde_json_string(c))).collect();
            let size = tick_runs
                .iter()
                .map(|r| postcard::to_allocvec(r).map_or(24, |v| v.len()))
                .chain(tick_cmds.iter().map(|(_, j)| j.len() + 12))
                .sum::<usize>();
            // Keep the datagram small; a client far behind gets the rest next time.
            if upto > from && bytes + size > FRAMES_BUDGET {
                break;
            }
            bytes += size;
            for r in tick_runs {
                last.insert(r.0, r.2);
                runs.push(r);
            }
            commands.extend(tick_cmds);
            upto = t + 1;
        }
        if upto == from {
            return;
        }
        self.send(net, peer, &Msg::Frames { from, upto, runs, commands });
    }
}

fn serde_json_string(c: &Command) -> String {
    serde_json::to_string(c).unwrap_or_default()
}

#[cfg(test)]
#[path = "server_tests.rs"]
mod tests;
