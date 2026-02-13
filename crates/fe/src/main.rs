//! `fe`: the falling-engine command-line tool.
#![forbid(unsafe_code)]
#![allow(clippy::print_stdout, clippy::print_stderr, reason = "this is a command-line tool")]

mod args;
mod cmds;

use std::process::ExitCode;

const USAGE: &str = "\
fe: falling-engine command-line tool

USAGE:
    fe <command> [options]

COMMANDS:
    run <scene>                 Run a scene and check its expectations
    test [paths...]             Run every .scene under the paths (default: scenes/tests)
        --bless                 Rewrite failing 'hash ==' expectations with the new hash
        --determinism           Also run each scene with 1 and --threads threads and compare hashes
    bench <scene>               Time each phase of the tick
        --ticks N --warmup N
    render <scene> --tick N --out FILE.png [--overlay]
    video <scene> --out FILE.mp4 [--every K] [--from T] [--ticks N]
    ascii <scene> --tick N --rect X0 Y0 X1 Y1
    snapshot <scene> --tick N --out FILE
    replay <replay.json> [--until mass:MATERIAL]
                                Re-run a replay, checking recorded hashes, and find the
                                first tick where an invariant breaks
    diff <a> <b> [--ticks N]    Run two scenes or replays side by side and report the first
                                tick and chunks where they differ
    serve [--tcp ADDR]          Control API (JSON-RPC 2.0, one request per line) on stdio or TCP
    mcp                         MCP server on stdio
    materials [FILE.toml]       List materials

COMMON OPTIONS:
    --threads N                 Worker threads (default: available cores). Results never depend on it.
    --json                      Print machine-readable JSON
";

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some((cmd, rest)) = argv.split_first() else {
        print!("{USAGE}");
        return ExitCode::from(2);
    };
    let a = match args::Args::parse(rest) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let result = match cmd.as_str() {
        "run" => cmds::run(&a),
        "test" => cmds::test(&a),
        "bench" => cmds::bench(&a),
        "render" => cmds::render(&a),
        "video" => cmds::video(&a),
        "ascii" => cmds::ascii(&a),
        "snapshot" => cmds::snapshot(&a),
        "replay" => cmds::replay(&a),
        "diff" => cmds::diff(&a),
        "serve" => cmds::serve(&a),
        "mcp" => fe_mcp::run_stdio(a.threads()).map(|()| true).map_err(|e| e.to_string()),
        "materials" => cmds::materials(&a),
        "help" | "--help" | "-h" => {
            print!("{USAGE}");
            Ok(true)
        }
        other => Err(format!("unknown command '{other}'; run 'fe help'")),
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
