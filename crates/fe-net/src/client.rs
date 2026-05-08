//! The client core: follows confirmed ticks, predicts its own character, and
//! checks its state against the server's.

use crate::{
    HASH_INTERVAL, Msg, PROTOCOL, Peer, TICK_US, Traffic, Transport, decode, encode, unpack_chunk,
};
use fe_core::ChunkPos;
use fe_engine::{Avatar, Command, HashParts, PlayerInput, Simulation};
use fe_materials::MaterialSet;
use fe_scene::{Runner, Scene};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Most confirmed ticks simulated in one call, so a client catching up stays responsive.
const MAX_CATCH_UP: u32 = 12;
/// Checkpoints kept for answering chunk-hash requests.
const KEEP_CHECKPOINTS: usize = 12;

/// Client counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientStats {
    /// Bytes and packets.
    pub traffic: Traffic,
    /// Chunk repairs applied.
    pub repairs: u32,
    /// Full snapshots loaded (including the first).
    pub snapshots: u32,
    /// Time from hello to the first loaded snapshot, in microseconds.
    pub join_us: u64,
    /// Latest round-trip estimate in microseconds.
    pub rtt_us: u64,
    /// Ticks between sampling a local input and the server applying it.
    pub lead_ticks: u64,
    /// Snapshot data received, in bytes.
    pub snapshot_bytes: u64,
}

/// How long a client waits for a welcome before asking again, in microseconds.
const HELLO_RETRY_US: u64 = 400_000;

/// Connection state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClientPhase {
    /// Waiting for a welcome.
    Connecting,
    /// Receiving the snapshot.
    Loading,
    /// Following confirmed ticks.
    Playing,
    /// Refused by the server.
    Rejected(String),
}

struct Incoming {
    id: u32,
    total: u32,
    parts: BTreeMap<u32, Vec<u8>>,
}

/// The client.
pub struct ClientCore {
    scene: Scene,
    scene_hash: u64,
    mats: Arc<MaterialSet>,
    threads: usize,
    server: Peer,
    name: String,
    /// The confirmed simulation, once loaded.
    pub runner: Option<Runner>,
    /// Assigned player id.
    pub player: Option<u8>,
    /// Connection state.
    pub phase: ClientPhase,
    incoming: Option<Incoming>,
    frames: BTreeMap<u64, BTreeMap<u8, PlayerInput>>,
    commands: BTreeMap<u64, Vec<Command>>,
    known_upto: u64,
    local: BTreeMap<u64, PlayerInput>,
    last_local: PlayerInput,
    repairs: BTreeMap<u64, Vec<(ChunkPos, Vec<u32>)>>,
    chunk_hashes: BTreeMap<u64, Vec<(ChunkPos, u64)>>,
    /// Hashes at each checkpoint this client reached.
    pub hashes: BTreeMap<u64, HashParts>,
    predicted: Option<Avatar>,
    started_us: u64,
    last_ping: u64,
    /// When the join request was last sent.
    last_hello: u64,
    /// Id of the last snapshot loaded.
    loaded: Option<u32>,
    ticks_run: u64,
    /// Counters.
    pub stats: ClientStats,
}

impl ClientCore {
    /// Creates a client for a scene; `scene_text` must match the server's exactly.
    pub fn new(
        scene_text: &str,
        scene: Scene,
        mats: Arc<MaterialSet>,
        server: Peer,
        name: &str,
        threads: usize,
    ) -> Self {
        Self {
            scene,
            scene_hash: crate::scene_hash(scene_text),
            mats,
            threads,
            server,
            name: name.to_string(),
            runner: None,
            player: None,
            phase: ClientPhase::Connecting,
            incoming: None,
            frames: BTreeMap::new(),
            commands: BTreeMap::new(),
            known_upto: 0,
            local: BTreeMap::new(),
            last_local: PlayerInput::default(),
            repairs: BTreeMap::new(),
            chunk_hashes: BTreeMap::new(),
            hashes: BTreeMap::new(),
            predicted: None,
            started_us: 0,
            last_ping: 0,
            last_hello: 0,
            loaded: None,
            ticks_run: 0,
            stats: ClientStats::default(),
        }
    }

    /// The confirmed simulation, once loaded.
    pub fn sim(&self) -> Option<&Simulation> {
        self.runner.as_ref().map(|r| &r.sim)
    }

    /// The local character as predicted from unconfirmed inputs.
    pub fn predicted(&self) -> Option<&Avatar> {
        self.predicted.as_ref()
    }

    fn send(&mut self, net: &mut impl Transport, m: &Msg) {
        let bytes = encode(m);
        self.stats.traffic.sent += bytes.len() as u64;
        self.stats.traffic.packets_sent += 1;
        net.send(self.server, &bytes);
    }

    /// Sends the join request.
    pub fn connect(&mut self, now_us: u64, net: &mut impl Transport) {
        self.started_us = now_us;
        self.send_hello(now_us, net);
    }

    fn send_hello(&mut self, now_us: u64, net: &mut impl Transport) {
        self.last_hello = now_us;
        let hello = Msg::Hello {
            protocol: PROTOCOL,
            name: self.name.clone(),
            materials: self.mats.hash(),
            scene: self.scene_hash,
        };
        self.send(net, &hello);
    }

    /// Says goodbye.
    pub fn disconnect(&mut self, net: &mut impl Transport) {
        self.send(net, &Msg::Bye);
    }

    /// Reads and handles every waiting packet.
    pub fn poll(&mut self, now_us: u64, net: &mut impl Transport) {
        while let Some((from, bytes)) = net.recv() {
            if from != self.server {
                continue;
            }
            self.stats.traffic.received += bytes.len() as u64;
            self.stats.traffic.packets_received += 1;
            if let Ok(m) = decode(&bytes) {
                self.handle(now_us, m, net);
            }
        }
    }

    fn handle(&mut self, now_us: u64, m: Msg, net: &mut impl Transport) {
        match m {
            Msg::Welcome { player } => {
                self.player = Some(player);
                if self.phase == ClientPhase::Connecting {
                    self.phase = ClientPhase::Loading;
                }
            }
            Msg::Reject(r) => self.phase = ClientPhase::Rejected(r),
            Msg::SnapshotPart { id, part, total, data } => {
                self.stats.snapshot_bytes += data.len() as u64;
                if self.loaded.is_some_and(|l| id <= l) {
                    // Our acknowledgement of a snapshot we already loaded was
                    // lost and the server is still sending it: acknowledge again.
                    if part == 0 {
                        self.send(net, &Msg::SnapshotAck { id, missing: Vec::new() });
                    }
                    return;
                }
                if self.incoming.as_ref().is_none_or(|i| i.id != id) {
                    if self.incoming.as_ref().is_some_and(|i| i.id > id) {
                        return;
                    }
                    self.incoming = Some(Incoming { id, total, parts: BTreeMap::new() });
                }
                let Some(inc) = self.incoming.as_mut() else { return };
                inc.parts.insert(part, data);
                if inc.parts.len() as u32 == inc.total {
                    let bytes: Vec<u8> = inc.parts.values().flatten().copied().collect();
                    let id = inc.id;
                    self.incoming = None;
                    self.loaded = Some(id);
                    self.load_snapshot(now_us, &bytes);
                    self.send(net, &Msg::SnapshotAck { id, missing: Vec::new() });
                    self.send_inputs(net);
                } else if part + 1 == total {
                    // The last part arrived but some are missing: ask for them.
                    let missing: Vec<u32> =
                        (0..inc.total).filter(|p| !inc.parts.contains_key(p)).collect();
                    self.send(net, &Msg::SnapshotAck { id, missing });
                }
            }
            Msg::Frames { from, upto, runs, commands } => {
                let Some(r) = self.runner.as_ref() else { return };
                let have = self.known_upto.max(r.sim.tick());
                if from > have || upto <= have {
                    return;
                }
                let mut per_player: BTreeMap<u8, Vec<(u64, PlayerInput)>> = BTreeMap::new();
                for (p, t, i) in runs {
                    per_player.entry(p).or_default().push((t, i));
                }
                for t in have..upto {
                    let mut tick_inputs = BTreeMap::new();
                    for (p, list) in &per_player {
                        if let Some((_, i)) = list.iter().rev().find(|(s, _)| *s <= t)
                            && *i != PlayerInput::default()
                        {
                            tick_inputs.insert(*p, *i);
                        }
                    }
                    self.frames.insert(t, tick_inputs);
                }
                for (t, json) in commands {
                    if t >= have
                        && let Ok(c) = serde_json::from_str::<Command>(&json)
                    {
                        self.commands.entry(t).or_default().push(c);
                    }
                }
                self.known_upto = upto;
            }
            Msg::NeedChunkHashes(tick) => {
                if let Some(h) = self.chunk_hashes.get(&tick).cloned() {
                    self.send(net, &Msg::ChunkHashes { tick, hashes: h });
                }
            }
            Msg::Repair { tick, chunks } => {
                let unpacked: Vec<(ChunkPos, Vec<u32>)> = chunks
                    .into_iter()
                    .filter_map(|(p, d)| unpack_chunk(&d).map(|b| (p, b)))
                    .collect();
                self.repairs.entry(tick).or_default().extend(unpacked);
            }
            Msg::Pong(t) => {
                let rtt = now_us.saturating_sub(t);
                self.stats.rtt_us =
                    if self.stats.rtt_us == 0 { rtt } else { (self.stats.rtt_us * 7 + rtt) / 8 };
            }
            _ => {}
        }
    }

    fn load_snapshot(&mut self, now_us: u64, bytes: &[u8]) {
        let Ok(sim) = Simulation::restore(bytes, self.mats.clone(), self.threads) else { return };
        let runner = match self.runner.take() {
            Some(mut r) => {
                r.sim = sim;
                r
            }
            None => match Runner::new(self.scene.clone(), self.mats.clone(), self.threads) {
                Ok(mut r) => {
                    r.sim = sim;
                    r
                }
                Err(_) => return,
            },
        };
        let t = runner.sim.tick();
        self.runner = Some(runner);
        self.frames = self.frames.split_off(&t);
        self.commands = self.commands.split_off(&t);
        self.known_upto = t;
        self.repairs.clear();
        self.stats.snapshots += 1;
        if self.stats.join_us == 0 {
            self.stats.join_us = now_us.saturating_sub(self.started_us);
        }
        self.phase = ClientPhase::Playing;
    }

    /// Sets this tick's local input. Call once per tick before [`ClientCore::tick`].
    pub fn set_input(&mut self, input: PlayerInput) {
        self.last_local = input;
    }

    fn lead(&self) -> u64 {
        let rtt_ticks = self.stats.rtt_us.div_ceil(TICK_US);
        (rtt_ticks + 3).clamp(3, 40)
    }

    fn send_inputs(&mut self, net: &mut impl Transport) {
        let Some(r) = self.runner.as_ref() else { return };
        let ack = r.sim.tick().max(self.known_upto);
        let upto = self.local.keys().next_back().copied().unwrap_or(ack);
        // Every run from the one in force at the acknowledged tick onward.
        let first = self.local.range(..=ack).next_back().map_or(ack, |(k, _)| *k);
        let runs: Vec<(u64, PlayerInput)> =
            self.local.range(first..).map(|(k, v)| (*k, *v)).collect();
        self.send(net, &Msg::Inputs { upto, runs, ack });
    }

    /// Advances: runs confirmed ticks, records the local input, predicts, and sends.
    pub fn tick(&mut self, now_us: u64, net: &mut impl Transport) {
        self.ticks_run += 1;
        // The join request or the welcome can be lost: ask again until welcomed.
        if (self.phase == ClientPhase::Connecting
            || (self.phase != ClientPhase::Playing && self.player.is_none()))
            && !matches!(self.phase, ClientPhase::Rejected(_))
            && now_us.saturating_sub(self.last_hello) > HELLO_RETRY_US
        {
            self.send_hello(now_us, net);
        }
        if self.phase != ClientPhase::Playing {
            return;
        }
        let mut steps = 0;
        let mut reports = Vec::new();
        if let Some(r) = self.runner.as_mut() {
            while r.sim.tick() < self.known_upto && steps < MAX_CATCH_UP {
                let t = r.sim.tick();
                let inputs = self.frames.remove(&t).unwrap_or_default();
                let cmds = self.commands.remove(&t).unwrap_or_default();
                r.step_with(&inputs, &cmds);
                steps += 1;
                let now = r.sim.tick();
                if let Some(chunks) = self.repairs.remove(&now) {
                    r.sim.replace_chunks(&chunks);
                    self.stats.repairs += 1;
                }
                if now.is_multiple_of(HASH_INTERVAL) {
                    let parts = r.sim.hash_parts();
                    self.hashes.insert(now, parts);
                    self.chunk_hashes.insert(now, r.sim.chunk_hashes());
                    while self.chunk_hashes.len() > KEEP_CHECKPOINTS {
                        self.chunk_hashes.pop_first();
                    }
                    reports.push(Msg::HashReport { tick: now, parts });
                }
            }
        }
        for m in reports {
            self.send(net, &m);
        }
        // Repairs aimed at ticks already passed can never apply cleanly; drop them.
        if let Some(r) = self.runner.as_ref() {
            let now = r.sim.tick();
            self.repairs = self.repairs.split_off(&now);
        }

        // Record the local input for the tick the server will apply it on.
        let Some(r) = self.runner.as_ref() else { return };
        let base = r.sim.tick().max(self.known_upto);
        let lead = self.lead();
        self.stats.lead_ticks = lead;
        let target = base + lead;
        let current = self.local.range(..=target).next_back().map(|(_, v)| *v);
        if current != Some(self.last_local) {
            self.local.insert(target, self.last_local);
        }
        let keep = r.sim.tick().saturating_sub(120);
        if let Some(k) = self.local.range(..=keep).next_back().map(|(k, _)| *k) {
            self.local = self.local.split_off(&k);
        }

        // Predict the local character: replay unconfirmed inputs on the confirmed state.
        self.predicted = None;
        if let Some(p) = self.player
            && let Some(a) = r.sim.avatars.iter().find(|a| a.id == p)
        {
            let mut a = *a;
            let from = r.sim.tick();
            for t in from..target {
                let i = self.local.range(..=t).next_back().map(|(_, v)| *v).unwrap_or_default();
                a.step(&r.sim.world, &i);
            }
            self.predicted = Some(a);
        }

        if self.ticks_run.is_multiple_of(2) {
            self.send_inputs(net);
        }
        if now_us.saturating_sub(self.last_ping) > 500_000 {
            self.last_ping = now_us;
            self.send(net, &Msg::Ping(now_us));
        }
    }
}
