//! Subcommand implementations.

use crate::args::Args;
use fe_core::IRect;
use fe_materials::MaterialSet;
use fe_render::{Renderer, View};
use fe_scene::{Replay, Runner, Scene};
use serde_json::json;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

fn load(path: &str) -> Result<(Scene, Arc<MaterialSet>, PathBuf), String> {
    let p = Path::new(path);
    let scene = fe_scene::load(p).map_err(|e| format!("{path}: {e}"))?;
    let dir = p.parent().map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    let mats = fe_scene::load_materials(&scene, &dir).map_err(|e| format!("{path}: {e}"))?;
    Ok((scene, Arc::new(mats), dir))
}

fn runner(path: &str, threads: usize) -> Result<Runner, String> {
    let (scene, mats, _) = load(path)?;
    Runner::new(scene, mats, threads).map_err(|e| format!("{path}: {e}"))
}

fn run_to(r: &mut Runner, tick: u64) {
    while r.sim.tick() < tick {
        r.step();
    }
}

pub fn run(a: &Args) -> Result<bool, String> {
    let path = a.pos(0, "scene path")?;
    let mut r = runner(path, a.threads())?;
    if let Some(t) = a.num::<u64>("ticks")? {
        r.scene.ticks = t;
    }
    let start = Instant::now();
    let sum = r.run_to_end();
    let secs = start.elapsed().as_secs_f64();
    if a.flag("json") {
        println!("{}", serde_json::to_string_pretty(&sum).unwrap_or_default());
    } else {
        for c in &sum.checks {
            println!(
                "  {} line {:>4} after {:>5}: {}",
                if c.passed { "ok  " } else { "FAIL" },
                c.line,
                c.after,
                c.detail
            );
        }
        println!(
            "{} {}: {} ticks in {:.2}s ({:.0} ticks/s), hash {:#018x}",
            if sum.passed { "PASS" } else { "FAIL" },
            r.scene.name,
            sum.ticks,
            secs,
            sum.ticks as f64 / secs.max(1e-9),
            sum.hash
        );
        for l in r.take_script_logs() {
            println!("  script: {l}");
        }
        if !r.sim.status.is_empty() {
            println!("  status: {}", r.sim.status);
        }
    }
    Ok(sum.passed)
}

fn collect_scenes(paths: &[String]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack: Vec<PathBuf> = if paths.is_empty() {
        vec![PathBuf::from("scenes/tests")]
    } else {
        paths.iter().map(PathBuf::from).collect()
    };
    while let Some(p) = stack.pop() {
        if p.is_dir() {
            if let Ok(rd) = std::fs::read_dir(&p) {
                stack.extend(rd.filter_map(Result::ok).map(|e| e.path()));
            }
        } else if p.extension().is_some_and(|e| e == "scene") {
            out.push(p);
        }
    }
    out.sort();
    out
}

/// Rewrites the expected hash on one line of a scene file.
fn bless(path: &Path, line: usize, new_hash: u64) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    let l = lines.get_mut(line - 1).ok_or("line out of range")?;
    *l = bless_line(l, new_hash).ok_or("no hash on that line")?;
    let mut out = lines.join("\n");
    out.push('\n');
    std::fs::write(path, out).map_err(|e| e.to_string())
}

/// Replaces the value in a `hash == VALUE` check, splitting on whitespace the
/// way the scene parser does. Returns `None` if the line has no such check.
fn bless_line(l: &str, new_hash: u64) -> Option<String> {
    let mut tokens = Vec::new();
    let mut start = None;
    for (i, ch) in l.char_indices().chain(std::iter::once((l.len(), ' '))) {
        match (ch.is_whitespace(), start) {
            (false, None) => start = Some(i),
            (true, Some(s)) => {
                tokens.push((s, i));
                start = None;
            }
            _ => {}
        }
    }
    let tok = |k: usize| tokens.get(k).map(|&(a, b)| &l[a..b]);
    let k = (0..tokens.len()).find(|&k| tok(k) == Some("hash") && tok(k + 1) == Some("=="))?;
    let (a, b) = *tokens.get(k + 2)?;
    Some(format!("{}{new_hash:#018x}{}", &l[..a], &l[b..]))
}

pub fn test(a: &Args) -> Result<bool, String> {
    let scenes = collect_scenes(&a.positional);
    if scenes.is_empty() {
        return Err("no .scene files found".into());
    }
    let threads = a.threads();
    let mut all_ok = true;
    let mut results = Vec::new();
    for path in &scenes {
        let p = path.display().to_string();
        let start = Instant::now();
        let mut r = match runner(&p, threads) {
            Ok(r) => r,
            Err(e) => {
                println!("ERROR {p}: {e}");
                all_ok = false;
                continue;
            }
        };
        let sum = r.run_to_end();
        let mut ok = true;
        let mut notes = Vec::new();
        if a.flag("determinism") && threads > 1 {
            let single = runner(&p, 1).map(|mut r1| r1.run_to_end().hash)?;
            if single != sum.hash {
                ok = false;
                notes.push(format!(
                    "1 thread gave {single:#018x}, {threads} threads gave {:#018x}",
                    sum.hash
                ));
            }
        }
        for c in sum.checks.iter().filter(|c| !c.passed) {
            match c.found_hash {
                Some(h) if a.flag("bless") => match bless(path, c.line, h) {
                    Ok(()) => notes.push(format!("blessed line {}: {}", c.line, c.detail)),
                    Err(e) => {
                        ok = false;
                        notes.push(format!("line {}: cannot bless: {e}", c.line));
                    }
                },
                _ => {
                    ok = false;
                    notes.push(format!("line {}: {}", c.line, c.detail));
                }
            }
        }
        all_ok &= ok;
        let ms = start.elapsed().as_millis();
        if !a.flag("json") {
            println!("{} {p} ({} ticks, {ms} ms)", if ok { "PASS " } else { "FAIL " }, sum.ticks);
            for n in &notes {
                println!("      {n}");
            }
        }
        results.push(json!({"scene": p, "passed": ok, "ticks": sum.ticks, "ms": ms, "hash": format!("{:#018x}", sum.hash), "notes": notes}));
    }
    if a.flag("json") {
        println!("{}", serde_json::to_string_pretty(&results).unwrap_or_default());
    } else {
        let failed = results.iter().filter(|r| r["passed"] == false).count();
        println!("{} scenes, {} failed", results.len(), failed);
    }
    Ok(all_ok)
}

pub fn bench(a: &Args) -> Result<bool, String> {
    let path = a.pos(0, "scene path")?;
    let mut r = runner(path, a.threads())?;
    let warmup = a.num::<u64>("warmup")?.unwrap_or(0);
    let ticks = a.num::<u64>("ticks")?.unwrap_or(r.scene.ticks.max(1));
    run_to(&mut r, warmup);
    let mut sum = fe_engine::PhaseTimes::default();
    let mut max_total = 0u64;
    let mut visited = 0u64;
    let mut active = 0u64;
    for _ in 0..ticks {
        r.step();
        let t = r.sim.report().times;
        sum.commands += t.commands;
        sum.cells += t.cells;
        sum.explosions += t.explosions;
        sum.particles += t.particles;
        sum.support += t.support;
        sum.physics += t.physics;
        sum.avatars += t.avatars;
        sum.streaming += t.streaming;
        sum.total += t.total;
        max_total = max_total.max(t.total);
        visited += r.sim.report().visited_cells;
        active += u64::from(r.sim.report().active_chunks);
    }
    let n = ticks.max(1) as f64;
    let avg = |v: u64| v as f64 / n / 1000.0;
    let ns_per_cell = sum.cells as f64 * 1000.0 / visited.max(1) as f64;
    let out = json!({
        "scene": path, "threads": a.threads(), "ticks": ticks,
        "ms_per_tick": {"total": avg(sum.total), "max_total": max_total as f64 / 1000.0, "cells": avg(sum.cells),
            "explosions": avg(sum.explosions), "particles": avg(sum.particles), "support": avg(sum.support),
            "physics": avg(sum.physics), "avatars": avg(sum.avatars), "streaming": avg(sum.streaming), "commands": avg(sum.commands)},
        "visited_cells_per_tick": visited as f64 / n,
        "active_chunks_per_tick": active as f64 / n,
        "ns_per_visited_cell": ns_per_cell,
        "visited_cells_per_second": visited as f64 / (sum.cells.max(1) as f64 / 1e6),
        "bodies": r.sim.physics.bodies().len(), "particles": r.sim.particles.len(),
        "hash": format!("{:#018x}", r.sim.hash()),
    });
    if a.flag("json") {
        println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
    } else {
        println!("{path}: {ticks} ticks on {} threads", a.threads());
        println!(
            "  total     {:8.3} ms/tick (max {:.3})",
            avg(sum.total),
            max_total as f64 / 1000.0
        );
        for (k, v) in [
            ("cells", sum.cells),
            ("explode", sum.explosions),
            ("particles", sum.particles),
            ("support", sum.support),
            ("physics", sum.physics),
            ("avatars", sum.avatars),
            ("stream", sum.streaming),
        ] {
            println!("  {k:<9} {:8.3} ms/tick", avg(v));
        }
        println!(
            "  {:.0} cells visited per tick in {:.1} active chunks; {:.2} ns per visited cell",
            visited as f64 / n,
            active as f64 / n,
            ns_per_cell
        );
    }
    Ok(true)
}

fn renderer_for(r: &Runner) -> Renderer {
    Renderer::new(r.sim.materials())
}

pub fn render(a: &Args) -> Result<bool, String> {
    let path = a.pos(0, "scene path")?;
    let out = a.get("out").ok_or("--out FILE.png is required")?;
    let mut r = runner(path, a.threads())?;
    run_to(&mut r, a.num::<u64>("tick")?.unwrap_or(0));
    let ren = renderer_for(&r);
    let (cx, cy, zoom) = fe_control::camera_at(&r.scene.camera, r.sim.tick(), r.scene.bounds);
    let mut s = fe_control::render_settings(&r.scene.camera);
    s.overlay = a.flag("overlay");
    let img = ren.render(&r.sim, View { cx, cy, zoom }, &s, r.sim.tick());
    img.save_png(Path::new(out)).map_err(|e| format!("{out}: {e}"))?;
    println!("wrote {out} ({}x{}) at tick {}", img.width, img.height, r.sim.tick());
    Ok(true)
}

pub fn video(a: &Args) -> Result<bool, String> {
    let path = a.pos(0, "scene path")?;
    let out = a.get("out").ok_or("--out FILE.mp4 is required")?;
    let every = a.num::<u64>("every")?.unwrap_or(1).max(1);
    let mut r = runner(path, a.threads())?;
    let from = a.num::<u64>("from")?.unwrap_or(0);
    let end = a.num::<u64>("ticks")?.unwrap_or(r.scene.ticks);
    run_to(&mut r, from);
    let ren = renderer_for(&r);
    let s = fe_control::render_settings(&r.scene.camera);
    let (w, h) = (s.view_w * s.scale, s.view_h * s.scale);
    let fps = (60 / every).max(1);
    let frames_dir = a.get("frames-dir").map(PathBuf::from);
    let mut ffmpeg = if frames_dir.is_none() {
        Some(
            std::process::Command::new("ffmpeg")
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
                .args([
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                    out,
                ])
                .stdin(std::process::Stdio::piped())
                .spawn()
                .map_err(|e| {
                    format!(
                        "cannot start ffmpeg ({e}); use --frames-dir to write PNG frames instead"
                    )
                })?,
        )
    } else {
        None
    };
    if let Some(d) = &frames_dir {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    let start = Instant::now();
    let mut n = 0u64;
    while r.sim.tick() < end {
        if (r.sim.tick() - from).is_multiple_of(every) {
            let (cx, cy, zoom) =
                fe_control::camera_at(&r.scene.camera, r.sim.tick(), r.scene.bounds);
            let img = ren.render(&r.sim, View { cx, cy, zoom }, &s, r.sim.tick());
            if let Some(ff) = ffmpeg.as_mut() {
                ff.stdin
                    .as_mut()
                    .ok_or("ffmpeg stdin closed")?
                    .write_all(&img.rgba)
                    .map_err(|e| e.to_string())?;
            } else if let Some(d) = &frames_dir {
                img.save_png(&d.join(format!("frame_{n:05}.png"))).map_err(|e| e.to_string())?;
            }
            n += 1;
        }
        r.step();
    }
    if let Some(mut ff) = ffmpeg {
        drop(ff.stdin.take());
        let status = ff.wait().map_err(|e| e.to_string())?;
        if !status.success() {
            return Err(format!("ffmpeg failed: {status}"));
        }
    }
    println!(
        "wrote {n} frames ({w}x{h} at {fps} fps) to {} in {:.1}s",
        frames_dir.map_or(out.to_string(), |d| d.display().to_string()),
        start.elapsed().as_secs_f64()
    );
    Ok(true)
}

pub fn ascii(a: &Args) -> Result<bool, String> {
    let path = a.pos(0, "scene path")?;
    let mut r = runner(path, a.threads())?;
    run_to(&mut r, a.num::<u64>("tick")?.unwrap_or(0));
    let rect = match a.values("rect") {
        Some(v) => {
            let n: Result<Vec<i32>, _> = v.iter().map(|s| s.parse::<i32>()).collect();
            let n = n.map_err(|_| "--rect needs four integers")?;
            IRect::new(n[0], n[1], n[2], n[3])
        }
        None => r.scene.bounds,
    };
    print!("{}", r.sim.ascii(rect));
    Ok(true)
}

pub fn snapshot(a: &Args) -> Result<bool, String> {
    let path = a.pos(0, "scene path")?;
    let out = a.get("out").ok_or("--out FILE is required")?;
    let mut r = runner(path, a.threads())?;
    run_to(&mut r, a.num::<u64>("tick")?.unwrap_or(0));
    let bytes = r.sim.snapshot();
    std::fs::write(out, &bytes).map_err(|e| format!("{out}: {e}"))?;
    println!(
        "wrote {out}: {} bytes at tick {}, hash {:#018x}",
        bytes.len(),
        r.sim.tick(),
        r.sim.hash()
    );
    Ok(true)
}

fn replay_runner(path: &str, threads: usize) -> Result<(Runner, Vec<(u64, u64)>), String> {
    if Path::new(path).extension().is_some_and(|e| e.eq_ignore_ascii_case("scene")) {
        return runner(path, threads).map(|r| (r, Vec::new()));
    }
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let replay = Replay::from_json(&text).map_err(|e| format!("{path}: {e}"))?;
    let scene = fe_scene::parse(&replay.scene).map_err(|e| format!("{path}: {e}"))?;
    let mats = Arc::new(
        fe_scene::load_materials(&scene, Path::new(&replay.scene_dir))
            .map_err(|e| e.to_string())?,
    );
    let r = replay.runner(mats, threads).map_err(|e| format!("{path}: {e}"))?;
    Ok((r, replay.hashes))
}

pub fn replay(a: &Args) -> Result<bool, String> {
    let path = a.pos(0, "replay path")?;
    let (mut r, hashes) = replay_runner(path, a.threads())?;
    let mass = a.get("until").and_then(|u| u.strip_prefix("mass:")).map(str::to_string);
    let mass_id = match &mass {
        Some(m) => Some(r.sim.material(m).ok_or_else(|| format!("unknown material '{m}'"))?),
        None => None,
    };
    let count = |r: &Runner| {
        mass_id.map(|id| {
            r.sim.world.census()[id.index()]
                + r.sim.particles.iter().filter(|p| p.cell.material() == id).count() as u64
        })
    };
    let start_mass = count(&r);
    let end = hashes.last().map_or(r.scene.ticks, |h| h.0).max(r.scene.ticks);
    let mut hi = hashes.iter().peekable();
    while r.sim.tick() < end {
        r.step();
        let t = r.sim.tick();
        while let Some(&&(ht, hv)) = hi.peek() {
            if ht > t {
                break;
            }
            hi.next();
            if ht == t && hv != r.sim.hash() {
                println!(
                    "hash mismatch at tick {t}: recorded {hv:#018x}, got {:#018x}",
                    r.sim.hash()
                );
                return Ok(false);
            }
        }
        if let (Some(m), Some(s)) = (count(&r), start_mass)
            && m != s
        {
            println!("invariant mass:{} broke at tick {t}: {s} -> {m}", mass.unwrap_or_default());
            return Ok(false);
        }
    }
    println!(
        "replay ok: {} ticks, {} recorded hashes matched, final hash {:#018x}",
        r.sim.tick(),
        hashes.len(),
        r.sim.hash()
    );
    Ok(true)
}

pub fn diff(a: &Args) -> Result<bool, String> {
    let pa = a.pos(0, "first scene or replay")?;
    let pb = a.pos(1, "second scene or replay")?;
    let (mut ra, _) = replay_runner(pa, a.threads())?;
    let (mut rb, _) = replay_runner(pb, a.threads())?;
    let end = a.num::<u64>("ticks")?.unwrap_or(ra.scene.ticks.max(rb.scene.ticks));
    if ra.sim.hash() != rb.sim.hash() {
        println!("states differ before tick 0");
        report_diff(&ra, &rb);
        return Ok(false);
    }
    while ra.sim.tick() < end {
        ra.step();
        rb.step();
        if ra.sim.hash() != rb.sim.hash() {
            println!("first difference after tick {}", ra.sim.tick());
            report_diff(&ra, &rb);
            return Ok(false);
        }
    }
    println!("identical for {end} ticks, hash {:#018x}", ra.sim.hash());
    Ok(true)
}

fn report_diff(a: &Runner, b: &Runner) {
    let ha: std::collections::BTreeMap<_, _> = a.sim.chunk_hashes().into_iter().collect();
    let hb: std::collections::BTreeMap<_, _> = b.sim.chunk_hashes().into_iter().collect();
    let mut n = 0;
    for (p, h) in &ha {
        if hb.get(p) != Some(h) {
            if n < 20 {
                println!(
                    "  chunk ({}, {}) differs (cells {}..{}, {}..{})",
                    p.x,
                    p.y,
                    p.origin().x,
                    p.origin().x + 64,
                    p.origin().y,
                    p.origin().y + 64
                );
            }
            n += 1;
        }
    }
    for p in hb.keys() {
        if !ha.contains_key(p) {
            println!("  chunk ({}, {}) only in the second run", p.x, p.y);
            n += 1;
        }
    }
    println!(
        "  {n} chunks differ; bodies {} vs {}; particles {} vs {}; avatars {} vs {}",
        a.sim.physics.bodies().len(),
        b.sim.physics.bodies().len(),
        a.sim.particles.len(),
        b.sim.particles.len(),
        a.sim.avatars.len(),
        b.sim.avatars.len()
    );
}

pub fn serve(a: &Args) -> Result<bool, String> {
    let mut session = fe_control::Session::new(a.threads());
    if let Some(addr) = a.get("tcp") {
        let listener = std::net::TcpListener::bind(addr).map_err(|e| format!("{addr}: {e}"))?;
        eprintln!("control API listening on {addr}");
        for stream in listener.incoming() {
            let stream = stream.map_err(|e| e.to_string())?;
            let reader = std::io::BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
            if let Err(e) = fe_control::serve(reader, stream, &mut session) {
                eprintln!("connection closed: {e}");
            }
        }
        Ok(true)
    } else {
        let stdin = std::io::stdin();
        fe_control::serve(stdin.lock(), std::io::stdout().lock(), &mut session)
            .map_err(|e| e.to_string())?;
        Ok(true)
    }
}

pub fn materials(a: &Args) -> Result<bool, String> {
    let set = match a.positional.first() {
        Some(p) => {
            let text = std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?;
            MaterialSet::from_toml(&text).map_err(|e| format!("{p}: {e}"))?
        }
        None => fe_materials::base_set(),
    };
    for (id, m) in set.iter() {
        println!("{:>4} {:<16} {:?}", id.0, m.name, m.state);
    }
    println!("material set hash {:#018x}", set.hash());
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::bless_line;

    #[test]
    fn bless_finds_the_hash_with_any_spacing() {
        assert_eq!(
            bless_line("after 5: hash  ==   0x1  // note", 2).as_deref(),
            Some("after 5: hash  ==   0x0000000000000002  // note")
        );
        assert_eq!(
            bless_line("after 5:\thash == 0x1", 0xab).as_deref(),
            Some("after 5:\thash == 0x00000000000000ab")
        );
        assert_eq!(bless_line("after 5: bodies == 3", 1), None);
        assert_eq!(bless_line("after 5: hash ==", 1), None);
    }
}
