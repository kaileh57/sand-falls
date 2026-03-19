//! `fe-server demo`: runs bots over the simulated network and records the
//! server's world with a network overlay.

use crate::{num, opt, soak_config};
use fe_net::soak::{SoakFrame, run_soak_observed};
use fe_render::{Image, Renderer, View};
use std::collections::VecDeque;
use std::io::Write;

fn hud(
    img: &mut Image,
    f: &SoakFrame<'_>,
    history: &VecDeque<f64>,
    rates: &[(Option<u8>, f64, f64, u64, u64)],
) {
    let sim = &f.server.runner.sim;
    let mut lines = vec![
        format!("TICK {}   HASH {:016X}", sim.tick(), sim.hash()),
        format!(
            "DESYNCS {}   REPAIRS {}   RESYNCS {}   PACKETS LOST {}",
            f.server.stats.desyncs, f.server.stats.repairs, f.server.stats.resyncs, f.dropped
        ),
    ];
    for (p, up, down, rtt, lead) in rates {
        lines.push(match p {
            Some(p) => format!(
                "P{p}  UP {up:5.0} B/S  DOWN {down:6.0} B/S  RTT {} MS  LEAD {lead}",
                rtt / 1000
            ),
            None => "JOINING...".to_string(),
        });
    }
    let h = 14 + 12 * lines.len() as i32;
    img.shade(0, 0, 470, h, 150);
    for (i, l) in lines.iter().enumerate() {
        let col = if i == 1 && f.server.stats.desyncs == 0 {
            [140, 255, 160, 255]
        } else {
            [235, 235, 235, 255]
        };
        img.text(8, 8 + 12 * i as i32, 1, l, col);
    }
    // Total download per second over time: it should stay flat.
    let (gx, gy, gw, gh) = (img.width as i32 - 260, 8, 250, 70);
    img.shade(gx - 4, gy - 4, gx + gw + 4, gy + gh + 18, 150);
    img.text(gx, gy + gh + 4, 1, "TOTAL DOWNLOAD B/S", [200, 200, 200, 255]);
    // Scale to the median so join snapshots do not flatten the steady state.
    let mut sorted: Vec<f64> = history.iter().copied().collect();
    sorted.sort_by(f64::total_cmp);
    let max = (sorted.get(sorted.len() / 2).copied().unwrap_or(1.0) * 3.0).max(2000.0);
    for (i, v) in history.iter().enumerate() {
        let x = gx + (i as i32 * gw / 120.max(history.len() as i32));
        let bar = (v.min(max) / max * f64::from(gh)) as i32;
        img.fill(x, gy + gh - bar, x + 2, gy + gh, [120, 200, 255, 255]);
    }
    img.text(gx, gy, 1, &format!("{max:.0}"), [160, 160, 160, 255]);
}

/// `fe-server demo <scene> --out FILE.mp4`.
pub(crate) fn demo(args: &[String]) -> Result<bool, String> {
    let scene_path = args.first().ok_or("missing scene path")?;
    let out = opt(args, "--out").ok_or("--out FILE.mp4 is required")?.to_string();
    let every = num(args, "--every", 2u64)?.max(1);
    let cfg = soak_config(args, Some(scene_path))?;
    let scene = fe_scene::parse(&cfg.scene_text).map_err(|e| e.to_string())?;
    let mats = fe_scene::load_materials(&scene, &cfg.scene_dir).map_err(|e| e.to_string())?;
    let renderer = Renderer::new(&mats);
    let settings = fe_control::render_settings(&scene.camera);
    let (w, h) = (settings.view_w * settings.scale, settings.view_h * settings.scale);
    let fps = (60 / every).max(1);
    let mut ff = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            &format!("{w}x{h}"),
            "-r",
            &fps.to_string(),
            "-i",
            "-",
        ])
        .args(["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", &out])
        .stdin(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot start ffmpeg: {e}"))?;
    let mut stdin = ff.stdin.take().ok_or("ffmpeg stdin")?;
    let mut history: VecDeque<f64> = VecDeque::new();
    let mut last: Vec<(u64, u64)> = Vec::new();
    let mut rates: Vec<(Option<u8>, f64, f64, u64, u64)> = Vec::new();
    let mut err = None;
    let report = run_soak_observed(&cfg, &mut |f| {
        if f.step >= cfg.ticks || err.is_some() {
            return;
        }
        if f.step.is_multiple_of(60) {
            let now: Vec<(u64, u64)> = f.bots.iter().map(|b| (b.1.sent, b.1.received)).collect();
            rates = f
                .bots
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    let (s0, r0) = last.get(i).copied().unwrap_or((b.1.sent, b.1.received));
                    (b.0, (b.1.sent - s0) as f64, (b.1.received - r0) as f64, b.2, b.3)
                })
                .collect();
            history.push_back(rates.iter().map(|r| r.2).sum());
            if history.len() > 120 {
                history.pop_front();
            }
            last = now;
        }
        if !f.step.is_multiple_of(every) {
            return;
        }
        let sim = &f.server.runner.sim;
        let (cx, cy, zoom) = fe_control::camera_at(&scene.camera, sim.tick(), scene.bounds);
        let mut img = renderer.render(sim, View { cx, cy, zoom }, &settings, sim.tick());
        hud(&mut img, f, &history, &rates);
        if let Err(e) = stdin.write_all(&img.rgba) {
            err = Some(e.to_string());
        }
    })?;
    drop(stdin);
    let status = ff.wait().map_err(|e| e.to_string())?;
    if let Some(e) = err {
        return Err(format!("ffmpeg: {e}"));
    }
    if !status.success() {
        return Err(format!("ffmpeg failed: {status}"));
    }
    println!(
        "wrote {out}: {} ticks, {} desyncs, {} bots, final states {}",
        report.ticks,
        report.desyncs,
        report.bots.len(),
        if report.passed { "match the server" } else { "DIFFER" }
    );
    Ok(report.passed)
}
