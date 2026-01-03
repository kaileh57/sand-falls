//! JSON-RPC 2.0 control API.
//!
//! A [`Session`] holds one running scene. Requests arrive one JSON object per
//! line on stdio or TCP; [`serve`] answers each with one line. Every method is
//! listed by [`methods`] with a JSON schema, which the MCP server reuses.
#![forbid(unsafe_code)]

use fe_core::{CellPos, IRect, MaterialId, temp_to_celsius};
use fe_engine::{Command, PlayerInput, tools};
use fe_materials::MaterialSet;
use fe_render::{RenderSettings, Renderer, Sky, View};
use fe_scene::{Replay, Runner};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A JSON-RPC error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RpcError {
    /// JSON-RPC error code.
    pub code: i64,
    /// Message for the caller.
    pub message: String,
}

impl RpcError {
    fn invalid(msg: impl Into<String>) -> Self {
        Self { code: -32602, message: msg.into() }
    }
    fn state(msg: impl Into<String>) -> Self {
        Self { code: -32000, message: msg.into() }
    }
}

/// Description of one control method.
#[derive(Debug, Clone)]
pub struct MethodInfo {
    /// Method name.
    pub name: &'static str,
    /// What it does.
    pub description: &'static str,
    /// JSON schema for the params object.
    pub schema: Value,
}

fn obj(props: Value, required: &[&str]) -> Value {
    json!({ "type": "object", "properties": props, "required": required })
}

/// Every method the control API supports.
pub fn methods() -> Vec<MethodInfo> {
    let rect = json!({"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4, "description": "[x0, y0, x1, y1], x1 and y1 exclusive"});
    vec![
        MethodInfo {
            name: "load_scene",
            description: "Load a scene from a .scene file path or from scene text. Replaces the current session.",
            schema: obj(
                json!({"path": {"type": "string"}, "text": {"type": "string"}, "threads": {"type": "integer"}}),
                &[],
            ),
        },
        MethodInfo {
            name: "step",
            description: "Run ticks. Optional per-player inputs and commands apply to the first tick only. Returns the tick, hash, last tick report and any expectation results.",
            schema: obj(
                json!({
                    "ticks": {"type": "integer", "default": 1},
                    "inputs": {"type": "object", "description": "player id -> {move_x, jump, aim_x, aim_y, tool, param, use_tool}; tool may be a name: dig, paint, explode, heat, freeze, crate"},
                    "commands": {"type": "array", "items": {"type": "object"}, "description": "Command objects, e.g. {\"op\":\"explode\",\"x\":10,\"y\":20,\"radius\":8,\"power\":150}; material may be a name"}
                }),
                &[],
            ),
        },
        MethodInfo {
            name: "command",
            description: "Apply one command immediately, outside a tick. Ops: paint, fill_rect, dig, explode, heat, carve_body, impulse, joint, pin, add_player, remove_player.",
            schema: obj(json!({"command": {"type": "object"}}), &["command"]),
        },
        MethodInfo {
            name: "read_region",
            description: "Read a rectangle as text: one character per cell, '.' air, '#' boundary, '@' body cell, uppercase first letter for static materials, lowercase for others. Includes a legend.",
            schema: obj(json!({"rect": rect}), &["rect"]),
        },
        MethodInfo {
            name: "cell",
            description: "Read one cell: material, temperature in C, value byte, flags.",
            schema: obj(json!({"x": {"type": "integer"}, "y": {"type": "integer"}}), &["x", "y"]),
        },
        MethodInfo {
            name: "count",
            description: "Count cells of a material inside a rectangle.",
            schema: obj(
                json!({"material": {"type": "string"}, "rect": rect}),
                &["material", "rect"],
            ),
        },
        MethodInfo {
            name: "hash",
            description: "Whole-state hash and tick.",
            schema: obj(json!({}), &[]),
        },
        MethodInfo {
            name: "chunk_hashes",
            description: "Per-chunk hashes in canonical order, for locating where two runs differ.",
            schema: obj(json!({}), &[]),
        },
        MethodInfo {
            name: "stats",
            description: "Counters and per-phase timings from the last tick.",
            schema: obj(json!({}), &[]),
        },
        MethodInfo {
            name: "bodies",
            description: "Every rigid body: id, position, angle, velocity, cell count.",
            schema: obj(json!({}), &[]),
        },
        MethodInfo {
            name: "avatars",
            description: "Every player character: id, feet position, velocity, flags.",
            schema: obj(json!({}), &[]),
        },
        MethodInfo {
            name: "materials",
            description: "Material names in id order.",
            schema: obj(json!({}), &[]),
        },
        MethodInfo {
            name: "screenshot",
            description: "Render a PNG. Defaults to the scene camera at the current tick. Returns the path written.",
            schema: obj(
                json!({
                    "path": {"type": "string"}, "cx": {"type": "number"}, "cy": {"type": "number"}, "zoom": {"type": "number"},
                    "view_w": {"type": "integer"}, "view_h": {"type": "integer"}, "scale": {"type": "integer"},
                    "sky": {"type": "string", "enum": ["day", "dusk", "night", "cave"]}, "overlay": {"type": "boolean"}
                }),
                &["path"],
            ),
        },
        MethodInfo {
            name: "snapshot_save",
            description: "Save the whole simulation to a file.",
            schema: obj(json!({"path": {"type": "string"}}), &["path"]),
        },
        MethodInfo {
            name: "snapshot_load",
            description: "Replace the simulation with a saved snapshot.",
            schema: obj(json!({"path": {"type": "string"}}), &["path"]),
        },
        MethodInfo {
            name: "replay_save",
            description: "Save the scene plus every injected input and command as a replay file that reproduces this run exactly.",
            schema: obj(json!({"path": {"type": "string"}}), &["path"]),
        },
        MethodInfo {
            name: "expectations",
            description: "Results of the scene's [expect] checks so far.",
            schema: obj(json!({}), &[]),
        },
    ]
}

/// One control session.
pub struct Session {
    runner: Option<Runner>,
    renderer: Option<Renderer>,
    scene_text: String,
    scene_dir: PathBuf,
    hashes: Vec<(u64, u64)>,
    default_threads: usize,
}

impl Session {
    /// Creates an empty session. Scenes run with `threads` workers unless a request says otherwise.
    pub fn new(threads: usize) -> Self {
        Self {
            runner: None,
            renderer: None,
            scene_text: String::new(),
            scene_dir: PathBuf::from("."),
            hashes: Vec::new(),
            default_threads: threads.max(1),
        }
    }

    fn runner(&mut self) -> Result<&mut Runner, RpcError> {
        self.runner
            .as_mut()
            .ok_or_else(|| RpcError::state("no scene loaded; call load_scene first"))
    }

    fn mats(&mut self) -> Result<Arc<MaterialSet>, RpcError> {
        Ok(self.runner()?.sim.materials().clone())
    }

    /// Handles one method call.
    pub fn handle(&mut self, method: &str, params: &Value) -> Result<Value, RpcError> {
        match method {
            "load_scene" => self.load_scene(params),
            "step" => self.step(params),
            "command" => {
                let mats = self.mats()?;
                let cmd = parse_command(params.get("command").unwrap_or(&Value::Null), &mats)?;
                let r = self.runner()?;
                r.sim.apply(&cmd);
                Ok(json!({"applied": true, "hash": format!("{:#018x}", r.sim.hash())}))
            }
            "read_region" => {
                let rect = parse_rect(params.get("rect"))?;
                if rect.width() * rect.height() > 400_000 {
                    return Err(RpcError::invalid("rect too large; keep it under 400,000 cells"));
                }
                let r = self.runner()?;
                let text = r.sim.ascii(rect);
                let mut legend = BTreeMap::new();
                for ch in text.chars().filter(|c| c.is_alphabetic()) {
                    if legend.contains_key(&ch) {
                        continue;
                    }
                    let names: Vec<&str> = r
                        .sim
                        .materials()
                        .iter()
                        .filter(|(_, m)| {
                            m.name.chars().next().is_some_and(|c| c.eq_ignore_ascii_case(&ch))
                        })
                        .map(|(_, m)| m.name.as_str())
                        .collect();
                    legend.insert(ch, names.join("|"));
                }
                Ok(json!({"text": text, "legend": legend}))
            }
            "cell" => {
                let x = int(params, "x")?;
                let y = int(params, "y")?;
                let r = self.runner()?;
                let c = r.sim.world.get(CellPos::new(x, y));
                let m = r.sim.materials().get(c.material());
                Ok(json!({
                    "material": m.name, "temp_c": temp_to_celsius(c.temp), "value": c.value,
                    "in_body": c.has(fe_core::Cell::FLAG_IN_BODY), "state": format!("{:?}", m.state)
                }))
            }
            "count" => {
                let rect = parse_rect(params.get("rect"))?;
                let mats = self.mats()?;
                let name = params.get("material").and_then(Value::as_str).unwrap_or("");
                let id = mats
                    .id(name)
                    .ok_or_else(|| RpcError::invalid(format!("unknown material '{name}'")))?;
                Ok(json!({"count": self.runner()?.sim.world.count(rect, id)}))
            }
            "hash" => {
                let r = self.runner()?;
                Ok(json!({"tick": r.sim.tick(), "hash": format!("{:#018x}", r.sim.hash())}))
            }
            "chunk_hashes" => {
                let r = self.runner()?;
                let v: Vec<Value> = r
                    .sim
                    .chunk_hashes()
                    .into_iter()
                    .map(|(p, h)| json!({"x": p.x, "y": p.y, "hash": format!("{h:#018x}")}))
                    .collect();
                Ok(json!({"tick": r.sim.tick(), "chunks": v}))
            }
            "stats" => {
                let r = self.runner()?;
                Ok(report_json(r))
            }
            "bodies" => {
                let r = self.runner()?;
                let v: Vec<Value> = r
                    .sim
                    .physics
                    .bodies()
                    .iter()
                    .filter_map(|b| r.sim.physics.body_info(b.id))
                    .map(|b| json!({"id": b.id, "x": b.x, "y": b.y, "angle": b.angle, "vx": b.vx, "vy": b.vy, "cells": b.cells, "sleeping": b.sleeping}))
                    .collect();
                Ok(json!({"bodies": v}))
            }
            "avatars" => {
                let r = self.runner()?;
                let v: Vec<Value> = r
                    .sim
                    .avatars
                    .iter()
                    .map(|a| json!({"id": a.id, "x": a.x.floor(), "y": a.y.floor(), "vx": a.vx.0 as f64 / 65536.0, "vy": a.vy.0 as f64 / 65536.0, "on_ground": a.on_ground, "in_liquid": a.in_liquid, "hp": a.hp}))
                    .collect();
                Ok(json!({"avatars": v}))
            }
            "materials" => {
                let mats = self.mats()?;
                let names: Vec<&str> = mats.iter().map(|(_, m)| m.name.as_str()).collect();
                Ok(json!({"materials": names}))
            }
            "screenshot" => self.screenshot(params),
            "snapshot_save" => {
                let path = string(params, "path")?;
                let bytes = self.runner()?.sim.snapshot();
                std::fs::write(&path, &bytes)
                    .map_err(|e| RpcError::state(format!("cannot write {path}: {e}")))?;
                Ok(json!({"path": path, "bytes": bytes.len()}))
            }
            "snapshot_load" => {
                let path = string(params, "path")?;
                let bytes = std::fs::read(&path)
                    .map_err(|e| RpcError::state(format!("cannot read {path}: {e}")))?;
                let mats = self.mats()?;
                let threads = self.default_threads;
                let sim = fe_engine::Simulation::restore(&bytes, mats, threads)
                    .map_err(|e| RpcError::state(e.to_string()))?;
                let r = self.runner()?;
                r.sim = sim;
                Ok(json!({"tick": r.sim.tick(), "hash": format!("{:#018x}", r.sim.hash())}))
            }
            "replay_save" => {
                let path = string(params, "path")?;
                let frames = self.runner()?.recorded().to_vec();
                let replay = Replay {
                    format: Replay::FORMAT,
                    scene: self.scene_text.clone(),
                    scene_dir: self.scene_dir.display().to_string(),
                    frames,
                    hashes: self.hashes.clone(),
                };
                std::fs::write(&path, replay.to_json())
                    .map_err(|e| RpcError::state(format!("cannot write {path}: {e}")))?;
                Ok(json!({"path": path}))
            }
            "expectations" => {
                let r = self.runner()?;
                Ok(serde_json::to_value(r.summary()).unwrap_or(Value::Null))
            }
            other => Err(RpcError { code: -32601, message: format!("unknown method '{other}'") }),
        }
    }

    fn load_scene(&mut self, params: &Value) -> Result<Value, RpcError> {
        let threads = params
            .get("threads")
            .and_then(Value::as_u64)
            .map_or(self.default_threads, |t| t as usize);
        let (text, dir) = if let Some(p) = params.get("path").and_then(Value::as_str) {
            let path = Path::new(p);
            let text = std::fs::read_to_string(path)
                .map_err(|e| RpcError::state(format!("cannot read {p}: {e}")))?;
            (text, path.parent().map_or_else(|| PathBuf::from("."), Path::to_path_buf))
        } else if let Some(t) = params.get("text").and_then(Value::as_str) {
            (t.to_string(), PathBuf::from("."))
        } else {
            return Err(RpcError::invalid("give 'path' or 'text'"));
        };
        let scene = fe_scene::parse(&text).map_err(|e| RpcError::invalid(e.to_string()))?;
        let mats = Arc::new(
            fe_scene::load_materials(&scene, &dir).map_err(|e| RpcError::invalid(e.to_string()))?,
        );
        let runner = Runner::new(scene, mats.clone(), threads)
            .map_err(|e| RpcError::invalid(e.to_string()))?;
        let info = json!({
            "name": runner.scene.name, "description": runner.scene.description,
            "bounds": [runner.scene.bounds.min.x, runner.scene.bounds.min.y, runner.scene.bounds.max.x, runner.scene.bounds.max.y],
            "ticks": runner.scene.ticks, "hash": format!("{:#018x}", runner.sim.hash()),
            "materials_hash": format!("{:#018x}", mats.hash())
        });
        self.renderer = Some(Renderer::new(&mats));
        self.runner = Some(runner);
        self.scene_text = text;
        self.scene_dir = dir;
        self.hashes.clear();
        Ok(info)
    }

    fn step(&mut self, params: &Value) -> Result<Value, RpcError> {
        let ticks = params.get("ticks").and_then(Value::as_u64).unwrap_or(1).min(1_000_000);
        let mats = self.mats()?;
        let mut inputs = BTreeMap::new();
        if let Some(m) = params.get("inputs").and_then(Value::as_object) {
            for (k, v) in m {
                let id: u8 = k
                    .parse()
                    .map_err(|_| RpcError::invalid(format!("player id '{k}' is not a number")))?;
                inputs.insert(id, parse_input(v, &mats)?);
            }
        }
        let mut commands = Vec::new();
        if let Some(list) = params.get("commands").and_then(Value::as_array) {
            for c in list {
                commands.push(parse_command(c, &mats)?);
            }
        }
        let r = self.runner()?;
        let mut checks = Vec::new();
        for i in 0..ticks {
            if i == 0 {
                checks.extend(r.step_with(&inputs, &commands));
            } else {
                checks.extend(r.step());
            }
        }
        let hash = r.sim.hash();
        let tick = r.sim.tick();
        let mut out = report_json(r);
        out["hash"] = json!(format!("{hash:#018x}"));
        out["checks"] = serde_json::to_value(&checks).unwrap_or(Value::Null);
        self.hashes.push((tick, hash));
        Ok(out)
    }

    fn screenshot(&mut self, params: &Value) -> Result<Value, RpcError> {
        let path = string(params, "path")?;
        let r = self
            .runner
            .as_ref()
            .ok_or_else(|| RpcError::state("no scene loaded; call load_scene first"))?;
        let renderer = self.renderer.as_ref().ok_or_else(|| RpcError::state("no renderer"))?;
        let cam = &r.scene.camera;
        let key = camera_at(cam, r.sim.tick(), r.scene.bounds);
        let f = |k: &str| params.get(k).and_then(Value::as_f64).map(|v| v as f32);
        let u = |k: &str| params.get(k).and_then(Value::as_u64).map(|v| v as u32);
        let view = View {
            cx: f("cx").unwrap_or(key.0),
            cy: f("cy").unwrap_or(key.1),
            zoom: f("zoom").unwrap_or(key.2),
        };
        let mut s = render_settings(cam);
        s.view_w = u("view_w").unwrap_or(s.view_w).clamp(16, 4096);
        s.view_h = u("view_h").unwrap_or(s.view_h).clamp(16, 4096);
        s.scale = u("scale").unwrap_or(s.scale).clamp(1, 8);
        if let Some(sky) = params.get("sky").and_then(Value::as_str).and_then(Sky::parse) {
            s.sky = sky;
        }
        s.overlay = params.get("overlay").and_then(Value::as_bool).unwrap_or(false);
        let img = renderer.render(&r.sim, view, &s, r.sim.tick());
        img.save_png(Path::new(&path))
            .map_err(|e| RpcError::state(format!("cannot write {path}: {e}")))?;
        Ok(json!({"path": path, "width": img.width, "height": img.height}))
    }
}

/// Render settings from a scene camera.
pub fn render_settings(cam: &fe_scene::Camera) -> RenderSettings {
    RenderSettings {
        view_w: cam.view.0,
        view_h: cam.view.1,
        scale: cam.scale,
        sky: Sky::parse(&cam.sky).unwrap_or(Sky::Cave),
        ambient: cam.ambient,
        exposure: cam.exposure,
        ..RenderSettings::default()
    }
}

/// Camera center and zoom at a tick, interpolated between keyframes.
pub fn camera_at(cam: &fe_scene::Camera, tick: u64, bounds: IRect) -> (f32, f32, f32) {
    let center =
        ((bounds.min.x + bounds.max.x) as f32 / 2.0, (bounds.min.y + bounds.max.y) as f32 / 2.0);
    let Some(first) = cam.keys.first() else { return (center.0, center.1, 1.0) };
    if tick <= first.tick {
        return (first.x, first.y, first.zoom);
    }
    for w in cam.keys.windows(2) {
        let (a, b) = (w[0], w[1]);
        if tick >= a.tick && tick <= b.tick {
            let t = (tick - a.tick) as f32 / (b.tick - a.tick).max(1) as f32;
            // Ease in and out between keys.
            let t = t * t * (3.0 - 2.0 * t);
            return (a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.zoom + (b.zoom - a.zoom) * t);
        }
    }
    cam.keys.last().map_or((center.0, center.1, 1.0), |k| (k.x, k.y, k.zoom))
}

fn report_json(r: &Runner) -> Value {
    let rep = r.sim.report();
    json!({
        "tick": r.sim.tick(),
        "report": serde_json::to_value(rep).unwrap_or(Value::Null),
        "bodies": r.sim.physics.bodies().len(),
        "particles": r.sim.particles.len(),
        "finished": r.finished(),
    })
}

fn int(p: &Value, k: &str) -> Result<i32, RpcError> {
    p.get(k)
        .and_then(Value::as_i64)
        .map(|v| v as i32)
        .ok_or_else(|| RpcError::invalid(format!("'{k}' must be an integer")))
}

fn string(p: &Value, k: &str) -> Result<String, RpcError> {
    p.get(k)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| RpcError::invalid(format!("'{k}' must be a string")))
}

fn parse_rect(v: Option<&Value>) -> Result<IRect, RpcError> {
    let a = v
        .and_then(Value::as_array)
        .ok_or_else(|| RpcError::invalid("'rect' must be [x0, y0, x1, y1]"))?;
    let n: Vec<i32> = a.iter().filter_map(Value::as_i64).map(|v| v as i32).collect();
    if n.len() != 4 {
        return Err(RpcError::invalid("'rect' must be [x0, y0, x1, y1]"));
    }
    Ok(IRect::new(n[0], n[1], n[2], n[3]))
}

fn material_value(v: &mut Value, mats: &MaterialSet) -> Result<(), RpcError> {
    if let Some(name) = v.get("material").and_then(Value::as_str) {
        let id =
            mats.id(name).ok_or_else(|| RpcError::invalid(format!("unknown material '{name}'")))?;
        v["material"] = json!(id.0);
    }
    Ok(())
}

/// Parses a command object, accepting material names.
pub fn parse_command(v: &Value, mats: &MaterialSet) -> Result<Command, RpcError> {
    let mut v = v.clone();
    material_value(&mut v, mats)?;
    if let Some(r) = v.get("rect").and_then(Value::as_array) {
        let n: Vec<i64> = r.iter().filter_map(Value::as_i64).collect();
        if n.len() == 4 {
            v["rect"] = json!({"min": {"x": n[0], "y": n[1]}, "max": {"x": n[2], "y": n[3]}});
        }
    }
    serde_json::from_value(v).map_err(|e| RpcError::invalid(format!("bad command: {e}")))
}

/// Parses a player input object, accepting tool names and material names as `param`.
pub fn parse_input(v: &Value, mats: &MaterialSet) -> Result<PlayerInput, RpcError> {
    let mut i = PlayerInput::default();
    let o = v.as_object().ok_or_else(|| RpcError::invalid("an input must be an object"))?;
    for (k, val) in o {
        match k.as_str() {
            "move_x" => i.move_x = val.as_i64().unwrap_or(0).clamp(-1, 1) as i8,
            "jump" => i.jump = val.as_bool().unwrap_or(false),
            "use_tool" => i.use_tool = val.as_bool().unwrap_or(false),
            "aim_x" => i.aim_x = val.as_i64().unwrap_or(0) as i16,
            "aim_y" => i.aim_y = val.as_i64().unwrap_or(0) as i16,
            "tool" => {
                i.tool = match val {
                    Value::String(s) => match s.as_str() {
                        "none" => tools::NONE,
                        "dig" => tools::DIG,
                        "paint" => tools::PAINT,
                        "explode" => tools::EXPLODE,
                        "heat" => tools::HEAT,
                        "freeze" => tools::FREEZE,
                        "crate" => tools::CRATE,
                        other => return Err(RpcError::invalid(format!("unknown tool '{other}'"))),
                    },
                    other => other.as_u64().unwrap_or(0) as u8,
                }
            }
            "param" => {
                i.param = match val {
                    Value::String(s) => mats.id(s).map_or(0, |m: MaterialId| m.0),
                    other => other.as_u64().unwrap_or(0) as u16,
                }
            }
            other => return Err(RpcError::invalid(format!("unknown input field '{other}'"))),
        }
    }
    Ok(i)
}

/// Serves newline-delimited JSON-RPC 2.0 until the reader closes.
pub fn serve(
    reader: impl BufRead,
    mut writer: impl Write,
    session: &mut Session,
) -> std::io::Result<()> {
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = handle_line(session, &line);
        if let Some(reply) = reply {
            writeln!(writer, "{reply}")?;
            writer.flush()?;
        }
    }
    Ok(())
}

/// Handles one JSON-RPC request line. Returns `None` for notifications.
pub fn handle_line(session: &mut Session, line: &str) -> Option<String> {
    let req: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => {
            return Some(json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": format!("parse error: {e}")}}).to_string());
        }
    };
    let id = req.get("id").cloned();
    let method = req.get("method").and_then(Value::as_str).unwrap_or("");
    let params = req.get("params").cloned().unwrap_or_else(|| json!({}));
    let result = session.handle(method, &params);
    let id = id?;
    Some(match result {
        Ok(v) => json!({"jsonrpc": "2.0", "id": id, "result": v}).to_string(),
        Err(e) => {
            json!({"jsonrpc": "2.0", "id": id, "error": {"code": e.code, "message": e.message}})
                .to_string()
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCENE: &str = "[scene]\nsize = 128 96\n[build]\nfill 0 80 128 96 bedrock\nfill 10 10 30 30 sand\n[expect]\nafter 100: count sand in 0 0 128 60 == 0\n";

    fn call(s: &mut Session, method: &str, params: Value) -> Value {
        let line =
            json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).to_string();
        let reply: Value = handle_line(s, &line)
            .and_then(|r| serde_json::from_str(&r).ok())
            .unwrap_or(Value::Null);
        assert!(reply.get("error").is_none(), "{method}: {reply}");
        reply["result"].clone()
    }

    #[test]
    fn drive_a_scene_end_to_end() {
        let mut s = Session::new(1);
        let info = call(&mut s, "load_scene", json!({"text": SCENE}));
        assert_eq!(info["ticks"], 100);
        call(
            &mut s,
            "command",
            json!({"command": {"op": "paint", "x": 64, "y": 20, "radius": 3, "material": "water"}}),
        );
        let out = call(
            &mut s,
            "step",
            json!({"ticks": 100, "commands": [{"op": "explode", "x": 64, "y": 79, "radius": 6, "power": 100}]}),
        );
        assert_eq!(out["tick"], 100);
        assert_eq!(out["checks"][0]["passed"], true, "{out}");
        let region = call(&mut s, "read_region", json!({"rect": [0, 70, 128, 82]}));
        assert!(region["text"].as_str().is_some_and(|t| t.contains('s')));
        let c = call(&mut s, "count", json!({"material": "sand", "rect": [0, 0, 128, 96]}));
        assert!(c["count"].as_u64().is_some_and(|n| n > 300));
        let dir = std::env::temp_dir().join(format!("fe-control-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let png = dir.join("shot.png");
        call(
            &mut s,
            "screenshot",
            json!({"path": png.display().to_string(), "view_w": 64, "view_h": 48, "scale": 1}),
        );
        assert!(png.exists());
        let snap = dir.join("s.snap");
        let h1 = call(&mut s, "hash", json!({}))["hash"].clone();
        call(&mut s, "snapshot_save", json!({"path": snap.display().to_string()}));
        call(&mut s, "step", json!({"ticks": 10}));
        call(&mut s, "snapshot_load", json!({"path": snap.display().to_string()}));
        assert_eq!(call(&mut s, "hash", json!({}))["hash"], h1);
    }

    #[test]
    fn errors_are_reported_not_panicked() {
        let mut s = Session::new(1);
        let r = handle_line(&mut s, r#"{"jsonrpc":"2.0","id":2,"method":"step","params":{}}"#)
            .unwrap_or_default();
        assert!(r.contains("no scene loaded"));
        let r = handle_line(&mut s, "not json").unwrap_or_default();
        assert!(r.contains("-32700"));
        assert!(handle_line(&mut s, r#"{"jsonrpc":"2.0","method":"hash"}"#).is_none());
    }
}
