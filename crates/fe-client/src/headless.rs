//! A headless bot client for testing real UDP connections.

use crate::opt;
use fe_core::SeqRng;
use fe_engine::{PlayerInput, tools};
use fe_net::client::{ClientCore, ClientPhase};
use fe_net::{TICK_US, UdpTransport};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// `fe-client bot <addr> --scene <file> [--ticks N]`: joins, plays random inputs, reports.
pub(crate) fn bot(args: &[String]) -> Result<(), String> {
    let addr = args.first().ok_or("missing host:port")?;
    let scene_path = opt(args, "--scene").ok_or("--scene FILE is required")?;
    let ticks: u64 = opt(args, "--ticks").and_then(|v| v.parse().ok()).unwrap_or(1200);
    let name = opt(args, "--name").unwrap_or("bot");
    let text = std::fs::read_to_string(scene_path).map_err(|e| format!("{scene_path}: {e}"))?;
    let mut scene = fe_scene::parse(&text).map_err(|e| e.to_string())?;
    let dir = Path::new(scene_path).parent().map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    fe_scene::attach_script(&mut scene, &dir).map_err(|e| e.to_string())?;
    let mats = Arc::new(fe_scene::load_materials(&scene, &dir).map_err(|e| e.to_string())?);
    let mut net = UdpTransport::bind("0.0.0.0:0").map_err(|e| e.to_string())?;
    let server_addr = std::net::ToSocketAddrs::to_socket_addrs(addr.as_str())
        .map_err(|e| format!("{addr}: {e}"))?
        .next()
        .ok_or_else(|| format!("{addr}: no address"))?;
    let server = net.peer_for(server_addr);
    let mut client = ClientCore::new(&text, scene, mats, server, name, 1);
    let start = Instant::now();
    client.connect(0, &mut net);
    let mut rng = SeqRng::new(u64::from(std::process::id()));
    let mut input = PlayerInput::default();
    let mut next = Duration::ZERO;
    let mut n = 0u64;
    while n < ticks {
        let now = start.elapsed().as_micros() as u64;
        client.poll(now, &mut net);
        if start.elapsed() < next {
            std::thread::sleep(Duration::from_micros(500));
            continue;
        }
        next += Duration::from_micros(TICK_US);
        if let ClientPhase::Rejected(r) = &client.phase {
            return Err(format!("rejected: {r}"));
        }
        if n.is_multiple_of(30) {
            input = PlayerInput {
                move_x: rng.range_i32(-1, 1) as i8,
                jump: rng.below(3) == 0,
                tool: if rng.below(4) == 0 { tools::DIG } else { tools::NONE },
                param: 5,
                use_tool: true,
                aim_x: rng.range_i32(-20, 20) as i16,
                aim_y: rng.range_i32(0, 20) as i16,
            };
        }
        client.set_input(input);
        client.tick(now, &mut net);
        if client.phase == ClientPhase::Playing {
            n += 1;
        }
        if start.elapsed() > Duration::from_secs(ticks / 60 + 30) {
            return Err("timed out".into());
        }
    }
    client.disconnect(&mut net);
    let st = client.stats;
    let secs = start.elapsed().as_secs_f64();
    println!(
        "bot {name}: player {:?}, tick {}, hash {:#018x}, rtt {} ms, up {:.0} B/s, down {:.0} B/s, repairs {}, snapshots {}",
        client.player,
        client.sim().map_or(0, fe_engine::Simulation::tick),
        client.sim().map_or(0, fe_engine::Simulation::hash),
        st.rtt_us / 1000,
        st.traffic.sent as f64 / secs,
        st.traffic.received as f64 / secs,
        st.repairs,
        st.snapshots
    );
    Ok(())
}
