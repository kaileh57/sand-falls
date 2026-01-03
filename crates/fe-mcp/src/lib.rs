//! Model Context Protocol server.
//!
//! Speaks MCP over stdio (newline-delimited JSON-RPC 2.0) and exposes every
//! control method from [`fe_control::methods`] as a tool named `fe_<method>`.
//! Tool results are the control method's JSON result as text; screenshots also
//! return the PNG as image content.
#![forbid(unsafe_code)]

use fe_control::Session;
use serde_json::{Value, json};
use std::io::{BufRead, Write};

/// Protocol revisions this server accepts, newest first.
pub const PROTOCOL_VERSIONS: &[&str] =
    &["2026-07-28", "2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

const INSTRUCTIONS: &str = "falling-engine is a deterministic falling-sand engine. Load a scene with \
fe_load_scene (a .scene path or scene text), advance it with fe_step, inspect it with fe_read_region, \
fe_cell, fe_count, fe_bodies and fe_stats, and look at it with fe_screenshot. The same scene and the \
same inputs always give the same fe_hash, so a hash mismatch means behavior changed.";

/// Encodes bytes as standard base64.
pub fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = u32::from(b[0]) << 16 | u32::from(b[1]) << 8 | u32::from(b[2]);
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

/// The MCP server state.
pub struct Server {
    session: Session,
}

impl Server {
    /// Creates a server whose scenes run with `threads` workers by default.
    pub fn new(threads: usize) -> Self {
        Self { session: Session::new(threads) }
    }

    fn tools() -> Value {
        let tools: Vec<Value> = fe_control::methods()
            .into_iter()
            .map(|m| json!({"name": format!("fe_{}", m.name), "description": m.description, "inputSchema": m.schema}))
            .collect();
        json!({ "tools": tools })
    }

    fn call_tool(&mut self, params: &Value) -> Value {
        let name = params.get("name").and_then(Value::as_str).unwrap_or("");
        let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
        let Some(method) = name.strip_prefix("fe_") else {
            return json!({"content": [{"type": "text", "text": format!("unknown tool '{name}'")}], "isError": true});
        };
        match self.session.handle(method, &args) {
            Ok(result) => {
                let mut content = vec![
                    json!({"type": "text", "text": serde_json::to_string_pretty(&result).unwrap_or_default()}),
                ];
                if method == "screenshot"
                    && let Some(path) = result.get("path").and_then(Value::as_str)
                    && let Ok(bytes) = std::fs::read(path)
                {
                    content.push(
                        json!({"type": "image", "data": base64(&bytes), "mimeType": "image/png"}),
                    );
                }
                json!({"content": content, "isError": false})
            }
            Err(e) => json!({"content": [{"type": "text", "text": e.message}], "isError": true}),
        }
    }

    /// Handles one JSON-RPC message. Returns the reply, or `None` for notifications.
    pub fn handle(&mut self, msg: &Value) -> Option<Value> {
        let id = msg.get("id").cloned();
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
        let params = msg.get("params").cloned().unwrap_or_else(|| json!({}));
        let result = match method {
            "initialize" => {
                let asked = params.get("protocolVersion").and_then(Value::as_str).unwrap_or("");
                let version =
                    if PROTOCOL_VERSIONS.contains(&asked) { asked } else { PROTOCOL_VERSIONS[0] };
                Ok(json!({
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": false}},
                    "serverInfo": {"name": "falling-engine", "version": env!("CARGO_PKG_VERSION")},
                    "instructions": INSTRUCTIONS,
                }))
            }
            "ping" => Ok(json!({})),
            "tools/list" => Ok(Self::tools()),
            "tools/call" => Ok(self.call_tool(&params)),
            m if m.starts_with("notifications/") => return None,
            other => Err(json!({"code": -32601, "message": format!("method '{other}' not found")})),
        };
        let id = id?;
        Some(match result {
            Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
            Err(e) => json!({"jsonrpc": "2.0", "id": id, "error": e}),
        })
    }
}

/// Runs the server on stdin and stdout until stdin closes.
pub fn run_stdio(threads: usize) -> std::io::Result<()> {
    let mut server = Server::new(threads);
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(msg) => server.handle(&msg),
            Err(e) => Some(
                json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": e.to_string()}}),
            ),
        };
        if let Some(r) = reply {
            writeln!(out, "{r}")?;
            out.flush()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_known_values() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn handshake_list_and_call() {
        let mut s = Server::new(1);
        let init = s.handle(&json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}));
        assert_eq!(
            init.as_ref().map(|r| r["result"]["protocolVersion"].clone()),
            Some(json!("2025-06-18"))
        );
        assert!(
            s.handle(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"})).is_none()
        );
        let list = s
            .handle(&json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))
            .unwrap_or_default();
        let names: Vec<&str> = list["result"]["tools"]
            .as_array()
            .map(|a| a.iter().filter_map(|t| t["name"].as_str()).collect())
            .unwrap_or_default();
        assert!(names.contains(&"fe_step") && names.contains(&"fe_screenshot"));
        let scene = "[scene]\nsize = 64 64\n[build]\nfill 0 50 64 64 bedrock\n";
        let r = s.handle(&json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "fe_load_scene", "arguments": {"text": scene}}})).unwrap_or_default();
        assert_eq!(r["result"]["isError"], false, "{r}");
        let r = s.handle(&json!({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "fe_step", "arguments": {"ticks": 5}}})).unwrap_or_default();
        assert!(
            r["result"]["content"][0]["text"].as_str().is_some_and(|t| t.contains("\"tick\": 5"))
        );
    }
}
