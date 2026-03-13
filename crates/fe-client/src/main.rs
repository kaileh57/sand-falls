//! `fe-client`: play a scene in a window, alone or connected to a server.
//!
//! Frames are rendered on the CPU by `fe-render` and copied into the window's
//! surface texture each frame.
#![forbid(unsafe_code)]
#![allow(clippy::print_stdout, clippy::print_stderr, reason = "this is a command-line tool")]

mod app;
mod gpu;
mod headless;

use std::process::ExitCode;

const USAGE: &str = "\
fe-client: falling-engine client

USAGE:
    fe-client play <scene> [--threads N]
    fe-client connect <host:port> --scene <scene> [--name NAME] [--threads N]
    fe-client bot <host:port> --scene <scene> [--ticks N] [--name NAME]

CONTROLS:
    A / D           walk            W / Space   jump
    mouse           aim             left click  use tool
    1 dig  2 paint  3 explode  4 heat  5 freeze  6 crate
    Q / E or wheel  change paint material
    + / -           zoom            F1          chunk overlay
    F2              lighting        P           pause (local only)
    F12             screenshot      Esc         quit
";

pub(crate) fn opt<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
}

pub(crate) fn threads(args: &[String]) -> usize {
    opt(args, "--threads")
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, std::num::NonZero::get))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("play") => app::play(&args[1..]),
        Some("connect") => app::connect(&args[1..]),
        Some("bot") => headless::bot(&args[1..]),
        _ => {
            print!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}
