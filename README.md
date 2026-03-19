# falling-engine

falling-engine is a 2D engine in which every pixel is a simulated cell. Sand piles, water levels out, wood burns to ash, lava cools to basalt, and structures that lose their support fall as rigid bodies that can shatter, float or burn.

Three things set it apart from other falling-sand engines:

- **Deterministic:** the same starting state and inputs give the same world, bit for bit, on every platform and with any number of threads. CI checks this with stored hashes on Linux, Windows and macOS.
- **Lockstep multiplayer:** only player inputs cross the network, so bandwidth stays flat however chaotic the world gets. Desyncs are detected with hashes and repaired by resending chunks.
- **Built to be driven by agents:** a headless control API (JSON-RPC and MCP), text scene files with expectations, replays and diff tools let an agent such as Claude Code build, test and debug the engine without a person in the loop.

The design, its rules and the measured performance are in [docs/DESIGN.md](docs/DESIGN.md).

## Requirements

- Rust stable (1.90 or later).
- `ffmpeg` on the `PATH` to render videos. PNG output needs nothing extra.
- A GPU with Vulkan, Metal or DirectX 12 for the windowed client. Everything else runs headless.

## Build and test

To build everything, run:

```bash
cargo build --release
```

To run the unit tests and the scene tests, including the golden hashes and the thread-count determinism check, run:

```bash
cargo test --release --workspace
```

```bash
./target/release/fe test scenes/tests --determinism --threads 4
```

To soak-test multiplayer over a simulated network with 150 ms of round-trip time and 2% packet loss, run:

```bash
./target/release/fe-server soak --ticks 6000 --players 4 --latency 150 --loss 2
```

## Play

To play a scene in a window, run:

```bash
./target/release/fe-client play scenes/multiplayer/shared_sandbox.scene
```

Controls: A and D walk, W or Space jumps, the mouse aims, left click uses the tool. Keys 1 to 6 pick dig, paint, explode, heat, freeze and crate. Q and E (or the wheel) change the paint material.

To host a scene and join it from another machine, run the server on one machine and the client on the other. Both must load the same scene file.

```bash
./target/release/fe-server host scenes/multiplayer/chaos_soak.scene --port 7777
```

```bash
./target/release/fe-client connect SERVER_ADDRESS:7777 --scene scenes/multiplayer/chaos_soak.scene
```

## Render the demos

The showcase scenes are in `scenes/demos`: `glowcave`, `firestorm`, `foundry`, `demolition` and `dam_break`. To render one to video, run:

```bash
./target/release/fe video scenes/demos/firestorm.scene --out firestorm.mp4 --every 2
```

The multiplayer demos run bot players over a simulated network and draw a network overlay (tick, hash, desync counter, per-player bandwidth):

```bash
./target/release/fe-server demo scenes/multiplayer/chaos_soak.scene --out chaos.mp4 --ticks 2400 --players 4 --latency 150 --loss 2
```

## Drive the engine from an agent

To expose the engine as MCP tools over stdio, run `fe mcp`. To add it to Claude Code, run:

```bash
claude mcp add falling-engine -- ./target/release/fe mcp
```

The tools load scenes, step the simulation with inputs and commands, read regions as text, count materials, list bodies and characters, take screenshots, save snapshots and replays, and report hashes. The same methods are available as JSON-RPC over stdio (`fe serve`) or TCP (`fe serve --tcp 127.0.0.1:9000`).

Other commands for agents and for debugging:

| Command | What it does |
| --- | --- |
| `fe run SCENE` | Runs a scene and checks its `[expect]` lines |
| `fe ascii SCENE --tick N --rect X0 Y0 X1 Y1` | Prints a region as text |
| `fe render SCENE --tick N --out FILE.png` | Renders one frame; `--overlay` shows chunks and dirty rects |
| `fe bench SCENE` | Times each phase of the tick |
| `fe replay FILE --until mass:water` | Replays a recorded run and finds the first tick an invariant breaks |
| `fe diff A B` | Runs two scenes or replays side by side and reports the first differing tick and chunks |
| `fe test --bless` | Rewrites failing `hash ==` expectations after an intended change |

## Write a scene

Scenes are plain text. A small one:

```text
[scene]
name = sand pile
size = 192 128

[build]
fill 0 120 192 128 bedrock
fill 80 10 112 40 sand

[expect]
after 400: count sand in 0 0 192 90 == 0
after 600: asleep
```

The format is documented in [crates/fe-scene/src/lib.rs](crates/fe-scene/src/lib.rs), and every file in `scenes/` is an example. Materials and reactions are data, defined in [crates/fe-materials/data/base.toml](crates/fe-materials/data/base.toml).

## Crates

| Crate | Purpose |
| --- | --- |
| `fe-core` | Cells, coordinates, fixed-point math, deterministic hashing and randomness |
| `fe-materials` | Material and reaction format, compiled to integer tables |
| `fe-sim` | Chunks, the checkerboard scheduler (the only `unsafe` code), cell rules, particles, explosions, structural support, characters |
| `fe-physics` | Pixel rigid bodies on Rapier |
| `fe-engine` | The full tick, commands, streaming, world generation, hashing, snapshots |
| `fe-scene` | Scene format, the scene runner, replays |
| `fe-render` | CPU renderer with propagated light, bloom and heat haze |
| `fe-control` | JSON-RPC control API |
| `fe-mcp` | MCP server over the control API |
| `fe-net` | Lockstep protocol, server and client cores, simulated network, soak harness |
| `fe`, `fe-client`, `fe-server` | Command-line tool, windowed client, dedicated server |

## License

MIT or Apache-2.0, at your option.
