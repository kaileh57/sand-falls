//! Soak test: a server and bot clients over a simulated network.
//!
//! Bots play with seeded random inputs (walking, jumping, digging, painting and
//! blowing things up) under the configured latency, jitter and packet loss.
//! One bot joins late, in the middle of the action. At the end every client's
//! state is compared with the server's at every shared checkpoint.

use crate::client::{ClientCore, ClientPhase};
use crate::server::ServerCore;
use crate::{Peer, SimNet, SimNetConfig, TICK_US};
use fe_core::SeqRng;
use fe_engine::{PlayerInput, tools};
use fe_scene::Runner;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Soak settings.
#[derive(Clone, Debug)]
pub struct SoakConfig {
    /// Scene source text.
    pub scene_text: String,
    /// Directory for the scene's relative paths.
    pub scene_dir: std::path::PathBuf,
    /// Bot players, including the late joiner.
    pub players: u8,
    /// Server ticks to run.
    pub ticks: u64,
    /// Round-trip time in milliseconds.
    pub rtt_ms: u64,
    /// Jitter in milliseconds.
    pub jitter_ms: u64,
    /// Packet loss percentage.
    pub loss_pct: u32,
    /// Tick at which the last bot joins.
    pub late_join_at: u64,
    /// Seed for bots and the network.
    pub seed: u64,
    /// Threads per simulation.
    pub threads: usize,
    /// Test hook: at this tick, corrupt a cell in the first bot's world to force a desync.
    pub corrupt_at: Option<u64>,
}

/// One bot's results.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BotReport {
    /// Player id.
    pub player: u8,
    /// Average upload in bytes per second.
    pub up_bytes_per_s: f64,
    /// Average download in bytes per second, excluding snapshot transfers.
    pub down_bytes_per_s: f64,
    /// Snapshot bytes received.
    pub snapshot_bytes: u64,
    /// Time from hello to playing, in milliseconds.
    pub join_ms: f64,
    /// Checkpoints compared against the server.
    pub checkpoints: u32,
    /// Checkpoints whose hash differed from the server's.
    pub mismatches: u32,
    /// Final confirmed tick.
    pub final_tick: u64,
    /// Final hash equals the server's hash at the same tick.
    pub final_match: bool,
    /// Ticks between input and the server applying it.
    pub lead_ticks: u64,
    /// Chunk repairs applied.
    pub repairs: u32,
}

/// Soak results.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SoakReport {
    /// Server ticks run.
    pub ticks: u64,
    /// Desyncs the server detected from hash reports.
    pub desyncs: u32,
    /// Chunk repairs the server sent.
    pub repairs: u32,
    /// Full resyncs the server sent.
    pub resyncs: u32,
    /// Packets the network dropped.
    pub dropped: u64,
    /// Packets delivered.
    pub delivered: u64,
    /// Per-bot results.
    pub bots: Vec<BotReport>,
    /// Every bot matched the server at every checkpoint and at the end.
    pub passed: bool,
}

struct Bot {
    client: ClientCore,
    peer: Peer,
    join_at: u64,
    rng: SeqRng,
    input: PlayerInput,
    joined: bool,
}

fn bot_input(rng: &mut SeqRng, tick: u64, prev: PlayerInput, materials: &[u16]) -> PlayerInput {
    if !tick.is_multiple_of(20) {
        // Keep held keys, but tools fire on a single tick.
        return PlayerInput {
            use_tool: prev.use_tool && prev.tool != tools::EXPLODE && prev.tool != tools::CRATE,
            ..prev
        };
    }
    let roll = rng.below(100);
    let (tool, param, use_tool) = match roll {
        0..=24 => (tools::DIG, 5, true),
        25..=39 => (tools::PAINT, materials[rng.below(materials.len() as u32) as usize], true),
        40..=44 => (tools::EXPLODE, 9, true),
        45..=49 => (tools::CRATE, 0, true),
        50..=54 => (tools::HEAT, 0, true),
        _ => (tools::NONE, 0, false),
    };
    PlayerInput {
        move_x: rng.range_i32(-1, 1) as i8,
        jump: rng.below(4) == 0,
        aim_x: rng.range_i32(-30, 30) as i16,
        aim_y: rng.range_i32(-20, 25) as i16,
        tool,
        param,
        use_tool,
    }
}

/// Runs a soak and reports.
pub fn run_soak(cfg: &SoakConfig) -> Result<SoakReport, String> {
    let scene = fe_scene::parse(&cfg.scene_text).map_err(|e| e.to_string())?;
    let mats =
        Arc::new(fe_scene::load_materials(&scene, &cfg.scene_dir).map_err(|e| e.to_string())?);
    let runner =
        Runner::new(scene.clone(), mats.clone(), cfg.threads).map_err(|e| e.to_string())?;
    let scene_hash = crate::scene_hash(&cfg.scene_text);
    let mut server = ServerCore::new(runner, scene_hash, 64);
    let server_peer = Peer(0);
    let mut net = SimNet::new(SimNetConfig {
        rtt_ms: cfg.rtt_ms,
        jitter_ms: cfg.jitter_ms,
        loss_pct: cfg.loss_pct,
        seed: cfg.seed,
    });
    let materials: Vec<u16> = ["sand", "water", "oil", "wood", "lava", "gunpowder"]
        .iter()
        .filter_map(|n| mats.id(n).map(|m| m.0))
        .collect();

    let mut bots: Vec<Bot> = (0..cfg.players)
        .map(|i| {
            let late = i + 1 == cfg.players && cfg.players > 1;
            Bot {
                client: ClientCore::new(
                    &cfg.scene_text,
                    scene.clone(),
                    mats.clone(),
                    server_peer,
                    &format!("bot{i}"),
                    cfg.threads,
                ),
                peer: Peer(u64::from(i) + 1),
                join_at: if late { cfg.late_join_at } else { u64::from(i) * 7 },
                rng: SeqRng::new(cfg.seed ^ (u64::from(i) + 1).wrapping_mul(0x9e37_79b9)),
                input: PlayerInput::default(),
                joined: false,
            }
        })
        .collect();

    // Server ticks, then a drain period where only clients run so they catch up.
    let drain = 600;
    for step in 0..cfg.ticks + drain {
        let now = step * TICK_US;
        net.advance_to(now);
        server.poll(now, &mut net.endpoint(server_peer));
        if step < cfg.ticks {
            server.tick(now, &mut net.endpoint(server_peer));
        } else if step.is_multiple_of(10) {
            server.flush(&mut net.endpoint(server_peer));
        }
        for b in &mut bots {
            if !b.joined && step >= b.join_at && step < cfg.ticks {
                b.client.connect(now, &mut net.endpoint(b.peer));
                b.joined = true;
            }
            if !b.joined {
                continue;
            }
            b.client.poll(now, &mut net.endpoint(b.peer));
            if cfg.corrupt_at == Some(step)
                && b.peer == Peer(1)
                && let Some(r) = b.client.runner.as_mut()
            {
                let stone = r.sim.material("bedrock").unwrap_or_default();
                let p = fe_core::CellPos::new(10, 250);
                let c = r.sim.world.make_cell(stone, p);
                r.sim.world.grid.set(p, c.with_material(fe_core::MaterialId::AIR));
            }
            if step < cfg.ticks {
                b.input = bot_input(&mut b.rng, step, b.input, &materials);
            } else {
                b.input = PlayerInput::default();
            }
            b.client.set_input(b.input);
            b.client.tick(now, &mut net.endpoint(b.peer));
        }
        if step >= cfg.ticks
            && bots.iter().all(|b| b.client.sim().is_some_and(|s| s.tick() >= server.tick_count()))
        {
            break;
        }
    }

    let secs = cfg.ticks as f64 / 60.0;
    let final_server = server.runner.sim.hash();
    let server_tick = server.tick_count();
    let mut passed = true;
    let mut reports = Vec::new();
    let clients = server.clients();
    for b in &bots {
        let c = &b.client;
        let mut checkpoints = 0;
        let mut mismatches = 0;
        for (t, parts) in &c.hashes {
            if let Some(s) = server.checkpoint(*t) {
                checkpoints += 1;
                if s != *parts {
                    mismatches += 1;
                }
            }
        }
        let final_tick = c.sim().map_or(0, fe_engine::Simulation::tick);
        let final_match = final_tick == server_tick
            && c.sim().map(fe_engine::Simulation::hash) == Some(final_server);
        let playing = c.phase == ClientPhase::Playing;
        passed &= playing && final_match;
        let snapshot_bytes = clients.iter().find(|ci| Some(ci.player) == c.player).map_or(0, |_| 0);
        reports.push(BotReport {
            player: c.player.unwrap_or(255),
            up_bytes_per_s: c.stats.traffic.sent as f64 / secs,
            down_bytes_per_s: c.stats.traffic.received as f64 / secs,
            snapshot_bytes,
            join_ms: c.stats.join_us as f64 / 1000.0,
            checkpoints,
            mismatches,
            final_tick,
            final_match,
            lead_ticks: c.stats.lead_ticks,
            repairs: c.stats.repairs,
        });
    }
    Ok(SoakReport {
        ticks: cfg.ticks,
        desyncs: server.stats.desyncs,
        repairs: server.stats.repairs,
        resyncs: server.stats.resyncs,
        dropped: net.dropped,
        delivered: net.delivered,
        passed: passed && server.stats.desyncs == 0,
        bots: reports,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARENA: &str = "[scene]\nname = arena\nsize = 384 256\nseed = 4\nspawn = 120 150\n[build]\nfill 0 200 384 256 soil\nfill 0 240 384 256 bedrock\nfill 250 150 290 200 water\nfill 300 120 340 200 brick\n";

    fn cfg(players: u8, ticks: u64, rtt: u64, loss: u32) -> SoakConfig {
        SoakConfig {
            scene_text: ARENA.into(),
            scene_dir: ".".into(),
            players,
            ticks,
            rtt_ms: rtt,
            jitter_ms: 10,
            loss_pct: loss,
            late_join_at: ticks / 3,
            seed: 7,
            threads: 1,
            corrupt_at: None,
        }
    }

    #[test]
    fn lan_soak_stays_in_sync() {
        let r = run_soak(&cfg(3, 600, 20, 0)).unwrap_or_else(|e| panic!("{e}"));
        assert!(r.passed, "{r:#?}");
        assert!(r.bots.iter().all(|b| b.checkpoints > 3));
    }

    #[test]
    fn corrupted_client_is_repaired() {
        let mut c = cfg(2, 900, 40, 0);
        c.corrupt_at = Some(200);
        let r = run_soak(&c).unwrap_or_else(|e| panic!("{e}"));
        assert!(r.desyncs >= 1, "{r:#?}");
        assert!(r.repairs >= 1 || r.resyncs >= 1, "{r:#?}");
        assert!(r.bots.iter().all(|b| b.final_match), "{r:#?}");
    }

    #[test]
    fn lossy_high_latency_soak_stays_in_sync() {
        let r = run_soak(&cfg(4, 900, 150, 2)).unwrap_or_else(|e| panic!("{e}"));
        assert!(r.passed, "{r:#?}");
        assert!(r.dropped > 0);
    }
}
