//! Protocol tests: a server and one client over a simulated network that can
//! drop chosen messages.

use super::*;
use crate::client::{ClientCore, ClientPhase};
use crate::transport::SimEndpoint;
use crate::{SimNet, SimNetConfig, TICK_US};
use fe_materials::MaterialSet;
use std::sync::Arc;

const ARENA: &str = "[scene]\nname = arena\nsize = 384 256\nseed = 4\nspawn = 120 150\n[build]\nfill 0 200 384 256 soil\nfill 0 240 384 256 bedrock\nfill 250 150 290 200 water\nfill 300 120 340 200 brick\n";

const SERVER: Peer = Peer(0);
const CLIENT: Peer = Peer(1);

/// A transport that drops the messages a filter picks.
struct Filtered<'a, 'b> {
    inner: SimEndpoint<'a>,
    drop: &'b mut dyn FnMut(&Msg) -> bool,
}

impl Transport for Filtered<'_, '_> {
    fn send(&mut self, to: Peer, data: &[u8]) {
        if let Ok(m) = decode(data)
            && (self.drop)(&m)
        {
            return;
        }
        self.inner.send(to, data);
    }

    fn recv(&mut self) -> Option<(Peer, Vec<u8>)> {
        self.inner.recv()
    }
}

struct Pair {
    server: ServerCore,
    client: ClientCore,
    net: SimNet,
    step: u64,
}

fn mats() -> Arc<MaterialSet> {
    Arc::new(fe_materials::base_set())
}

fn pair(text: &str) -> Pair {
    let scene = fe_scene::parse(text).unwrap_or_else(|e| panic!("{e}"));
    let runner = Runner::new(scene.clone(), mats(), 1).unwrap_or_else(|e| panic!("{e}"));
    let server = ServerCore::new(runner, crate::scene_hash(text), 8);
    let client = ClientCore::new(text, scene, mats(), SERVER, "bot", 1);
    let net = SimNet::new(SimNetConfig { rtt_ms: 10, jitter_ms: 0, loss_pct: 0, seed: 1 });
    Pair { server, client, net, step: 0 }
}

impl Pair {
    /// Runs `n` ticks. `down` picks server-to-client messages to drop, `up` the
    /// other way. The client runs only while `client_on` is true.
    fn run(
        &mut self,
        n: u64,
        down: &mut dyn FnMut(&Msg) -> bool,
        up: &mut dyn FnMut(&Msg) -> bool,
        client_on: bool,
    ) {
        for _ in 0..n {
            let now = self.step * TICK_US;
            self.net.advance_to(now);
            let mut s = Filtered { inner: self.net.endpoint(SERVER), drop: &mut *down };
            self.server.poll(now, &mut s);
            self.server.tick(now, &mut s);
            if client_on {
                let mut c = Filtered { inner: self.net.endpoint(CLIENT), drop: &mut *up };
                if self.step == 0 {
                    self.client.connect(now, &mut c);
                }
                self.client.poll(now, &mut c);
                self.client.set_input(PlayerInput::default());
                self.client.tick(now, &mut c);
            }
            self.step += 1;
        }
    }

    fn run_clean(&mut self, n: u64) {
        self.run(n, &mut |_| false, &mut |_| false, true);
    }

    fn client_tick(&self) -> u64 {
        self.client.sim().map_or(0, fe_engine::Simulation::tick)
    }
}

/// Cave terrain, so the snapshot takes several parts.
const CAVES: &str =
    "[scene]\nname = caves\nsize = 256 256\nseed = 4\nworldgen = caves 40\nspawn = 120 20\n";

#[test]
fn lost_final_snapshot_ack_does_not_stall_the_join() {
    let mut p = pair(CAVES);
    let mut dropped_part = false;
    let mut dropped_ack = false;
    p.run(
        300,
        &mut |m| {
            if !dropped_part && matches!(m, Msg::SnapshotPart { part: 0, total, .. } if *total > 1)
            {
                dropped_part = true;
                return true;
            }
            false
        },
        &mut |m| {
            if !dropped_ack && matches!(m, Msg::SnapshotAck { missing, .. } if missing.is_empty()) {
                dropped_ack = true;
                return true;
            }
            false
        },
        true,
    );
    assert!(dropped_part && dropped_ack);
    assert!(p.server.clients().iter().all(|c| c.synced));
    assert!(p.client_tick() + 20 > p.server.tick_count(), "client at {}", p.client_tick());
}

#[test]
fn lost_hello_and_welcome_are_retried() {
    let mut p = pair(ARENA);
    let mut hellos = 0;
    let mut welcomes = 0;
    p.run(
        200,
        &mut |m| {
            if matches!(m, Msg::Welcome { .. }) {
                welcomes += 1;
                return welcomes == 1;
            }
            false
        },
        &mut |m| {
            if matches!(m, Msg::Hello { .. }) {
                hellos += 1;
                return hellos == 1;
            }
            false
        },
        true,
    );
    assert!(hellos >= 2 && welcomes >= 2, "hellos {hellos}, welcomes {welcomes}");
    assert_eq!(p.client.player, Some(0));
    assert_eq!(p.client.phase, ClientPhase::Playing);
    assert!(p.client_tick() + 20 > p.server.tick_count());
}

#[test]
fn idle_network_input_overrides_scene_input_on_clients_too() {
    let text = format!("{ARENA}[events]\nat 60..300: input 0 move=1\n");
    let mut p = pair(&text);
    p.run_clean(360);
    assert!(p.server.stats.reports > 3);
    assert_eq!(p.server.stats.desyncs, 0, "{:?}", p.server.stats);
}

#[test]
fn chunk_hashes_are_answered_only_when_asked() {
    let mut p = pair(ARENA);
    p.run_clean(90);
    let tick = *p.server.checkpoints.keys().next_back().unwrap_or(&0);
    let stranger = Peer(99);
    for from in [stranger, CLIENT] {
        p.net.send(from, SERVER, &encode(&Msg::ChunkHashes { tick, hashes: Vec::new() }));
    }
    p.run_clean(10);
    assert_eq!(p.server.stats.repairs, 0);
    let mut to_stranger = Vec::new();
    while let Some((_, b)) = p.net.recv(stranger) {
        to_stranger.push(decode(&b));
    }
    assert_eq!(to_stranger, vec![Ok(Msg::Reject("not connected".into()))]);
}

#[test]
fn far_future_runs_and_huge_acks_are_bounded() {
    let mut p = pair(ARENA);
    p.run_clean(60);
    let runs: Vec<(u64, PlayerInput)> =
        (0..5000).map(|k| (1_000_000_000 + k, PlayerInput::default())).collect();
    p.net.send(CLIENT, SERVER, &encode(&Msg::Inputs { upto: 0, runs, ack: u64::MAX }));
    p.run_clean(4);
    let c = p.server.clients.get(&CLIENT).map(|c| (c.runs.len(), c.ack));
    let (runs, ack) = c.unwrap_or((usize::MAX, u64::MAX));
    assert!(runs < 100, "{runs} runs stored");
    assert!(ack <= p.server.tick_count());
}

#[test]
fn timed_out_client_is_told() {
    let mut p = pair(ARENA);
    p.run_clean(100);
    assert_eq!(p.client.phase, ClientPhase::Playing);
    // The client freezes for 6 s; the server drops it.
    p.run(360, &mut |_| false, &mut |_| false, false);
    assert!(p.server.clients().is_empty());
    p.run_clean(20);
    assert!(matches!(p.client.phase, ClientPhase::Rejected(_)), "{:?}", p.client.phase);
}

#[test]
fn client_behind_the_history_is_resynced() {
    let mut p = pair(ARENA);
    p.run_clean(100);
    // Frames stop arriving while the client keeps talking.
    p.run(HISTORY + 120, &mut |m| matches!(m, Msg::Frames { .. }), &mut |_| false, true);
    assert!(p.server.stats.resyncs >= 1, "{:?}", p.server.stats);
    p.run_clean(200);
    assert!(p.client_tick() + 20 > p.server.tick_count(), "client at {}", p.client_tick());
}

#[test]
fn frames_fit_in_a_datagram_with_many_players() {
    let mut p = pair(ARENA);
    p.run_clean(130);
    let now = p.server.tick_count();
    let from = now - MAX_FRAMES;
    for (k, (t, entry)) in p.server.history.range_mut(from..now).enumerate() {
        for player in 0..64u8 {
            let aim = (k as i16) * 7 + i16::from(player);
            entry
                .0
                .insert(player, PlayerInput { aim_x: aim, aim_y: -aim, ..PlayerInput::default() });
        }
        assert!(*t >= from);
    }
    if let Some(c) = p.server.clients.get_mut(&CLIENT) {
        c.ack = from;
    }
    // Clear out frames already in flight.
    p.net.advance_to((p.step + 10) * TICK_US);
    while p.net.recv(CLIENT).is_some() {}
    let mut ep = p.net.endpoint(SERVER);
    p.server.send_updates(CLIENT, now, true, &mut ep);
    p.net.advance_to((p.step + 20) * TICK_US);
    let mut sizes = Vec::new();
    while let Some((_, b)) = p.net.recv(CLIENT) {
        if let Ok(Msg::Frames { from: f, upto, .. }) = decode(&b) {
            assert_eq!(f, from);
            assert!(upto > from && upto < now);
            sizes.push(b.len());
        }
    }
    assert!(!sizes.is_empty() && sizes.iter().all(|&n| n < 2000), "{sizes:?}");
}
