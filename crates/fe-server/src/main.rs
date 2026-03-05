//! `fe-server`: host a scene over UDP, or soak-test multiplayer over a simulated network.
#![forbid(unsafe_code)]
#![allow(clippy::print_stdout, clippy::print_stderr, reason = "this is a command-line tool")]

use fe_net::server::ServerCore;
use fe_net::soak::{SoakConfig, run_soak};
use fe_net::{TICK_US, UdpTransport};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

const USAGE: &str = "\
fe-server: falling-engine multiplayer server

USAGE:
    fe-server host <scene> [--port 7777] [--threads N] [--max-players 8]
    fe-server soak [--scene FILE] [--ticks 6000] [--players 4] [--latency MS] [--jitter MS]
                   [--loss PCT] [--late-join TICK] [--seed N] [--json]

host runs the scene at 60 ticks per second and accepts fe-client connections.
soak runs a server and bot clients in one process over a simulated network and
fails if any client's state ever differs from the server's.
";

const DEFAULT_ARENA: &str = include_str!("../../../scenes/multiplayer/chaos_soak.scene");

fn opt<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn num<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> Result<T, String> {
    match opt(args, name) {
        Some(v) => v.parse().map_err(|_| format!("{name} '{v}' is not a number")),
        None => Ok(default),
    }
}

fn threads(args: &[String]) -> usize {
    num(args, "--threads", std::thread::available_parallelism().map_or(1, std::num::NonZero::get))
        .unwrap_or(1)
}

fn soak(args: &[String]) -> Result<bool, String> {
    let (scene_text, scene_dir) = match opt(args, "--scene") {
        Some(p) => (
            std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?,
            Path::new(p).parent().map_or_else(|| PathBuf::from("."), Path::to_path_buf),
        ),
        None => (DEFAULT_ARENA.to_string(), PathBuf::from(".")),
    };
    let ticks = num(args, "--ticks", 6000u64)?;
    let cfg = SoakConfig {
        scene_text,
        scene_dir,
        players: num(args, "--players", 4u8)?.max(1),
        ticks,
        rtt_ms: num(args, "--latency", 150u64)?,
        jitter_ms: num(args, "--jitter", 15u64)?,
        loss_pct: num(args, "--loss", 2u32)?,
        late_join_at: num(args, "--late-join", ticks / 3)?,
        seed: num(args, "--seed", 1u64)?,
        threads: threads(args).min(2),
        corrupt_at: None,
    };
    let start = Instant::now();
    let r = run_soak(&cfg)?;
    if args.iter().any(|a| a == "--json") {
        println!("{}", serde_json::to_string_pretty(&r).unwrap_or_default());
    } else {
        println!(
            "soak: {} ticks, {} bots, {} ms round trip, {}% loss, late join at tick {}",
            r.ticks, cfg.players, cfg.rtt_ms, cfg.loss_pct, cfg.late_join_at
        );
        println!("  network: {} packets delivered, {} dropped", r.delivered, r.dropped);
        println!("  server: {} desyncs, {} repairs, {} resyncs", r.desyncs, r.repairs, r.resyncs);
        for b in &r.bots {
            println!(
                "  player {}: up {:.0} B/s, down {:.0} B/s, joined in {:.0} ms, lead {} ticks, {}/{} checkpoints matched, final {}",
                b.player,
                b.up_bytes_per_s,
                b.down_bytes_per_s,
                b.join_ms,
                b.lead_ticks,
                b.checkpoints - b.mismatches,
                b.checkpoints,
                if b.final_match { "matches" } else { "DIFFERS" }
            );
        }
        println!(
            "{} in {:.1}s",
            if r.passed { "PASS" } else { "FAIL" },
            start.elapsed().as_secs_f64()
        );
    }
    Ok(r.passed)
}

fn host(args: &[String]) -> Result<bool, String> {
    let path = args.first().ok_or("missing scene path")?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let scene = fe_scene::parse(&text).map_err(|e| format!("{path}: {e}"))?;
    let dir = Path::new(path).parent().map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    let mats = Arc::new(fe_scene::load_materials(&scene, &dir).map_err(|e| e.to_string())?);
    let runner = fe_scene::Runner::new(scene, mats, threads(args)).map_err(|e| e.to_string())?;
    let port = num(args, "--port", 7777u16)?;
    let mut net = UdpTransport::bind(&format!("0.0.0.0:{port}")).map_err(|e| e.to_string())?;
    let mut server =
        ServerCore::new(runner, fe_net::scene_hash(&text), num(args, "--max-players", 8u8)?);
    eprintln!("hosting {path} on UDP port {port}");
    let start = Instant::now();
    let mut next = Duration::ZERO;
    let mut last_report = 0u64;
    loop {
        let now_us = start.elapsed().as_micros() as u64;
        server.poll(now_us, &mut net);
        if start.elapsed() >= next {
            server.tick(now_us, &mut net);
            next += Duration::from_micros(TICK_US);
            let t = server.tick_count();
            if t >= last_report + 600 {
                last_report = t;
                let c = server.clients();
                eprintln!(
                    "tick {t}: {} players, {} desyncs, {} repairs, last tick {:.2} ms",
                    c.len(),
                    server.stats.desyncs,
                    server.stats.repairs,
                    server.runner.sim.report().times.total as f64 / 1000.0
                );
            }
        } else {
            std::thread::sleep(Duration::from_micros(500));
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("soak") => soak(&args[1..]),
        Some("host") => host(&args[1..]),
        _ => {
            print!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}
