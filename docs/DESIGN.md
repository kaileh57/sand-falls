# falling-engine design

Status: draft, 2026-11-04

falling-engine is a custom 2D engine in which every pixel is a simulated cell. It takes the core ideas of Noita's Falling Everything engine and adds three things that engine does not have: bit-exact determinism, first-class multiplayer, and a control surface built for automated agents such as Claude Code to develop and test against. This document covers the architecture, the rules the code must follow, the performance targets, the demos, and the build order.

## Why build a new one

Nolla never released the Falling Everything engine. What they published is Petri Purho's GDC 2019 talk, [Exploring the Tech and Design of Noita](https://gdcvault.com/play/1025695/Exploring-the-Tech-and-Design), and nearly every open source clone copies its design: 64x64 chunks, dirty rects, a checkerboard update order for threads, and rigid bodies traced out of pixels. As of this draft, none of those clones is maintained. [FallingSandEngine](https://github.com/PieKing1215/FallingSandEngine), the most complete, had its last code commit in June 2023. Most stall at the same point, after sand and chunks work and before rigid bodies, performance, and networking are solved together.

Noita itself is not deterministic, which is why the co-op mod [Entangled Worlds](https://github.com/IntQuant/noita_entangled_worlds) has to assign each chunk an owner and send changed pixels over the network. That approach is limited by bandwidth during exactly the moments that make the genre fun, such as a flood or a chain of explosions. falling-engine is designed around determinism from the first commit so that multiplayer sends player inputs instead of pixels.

## Goals and non-goals

The engine targets four outcomes. A large world where millions of cells can be active at once, running at a fixed 60 Hz tick on a desktop CPU. Multiplayer for up to 8 players where bandwidth does not grow with how chaotic the world gets. A headless mode and control API good enough that an agent can write a scenario, run it, inspect the result, and bisect a regression without a human in the loop. And a data-driven material system general enough to build games that look nothing like Noita.

It does not target 3D, mobile in the first version, physically accurate fluid dynamics, or compatibility with Noita content or mods. It is not a general-purpose game engine either. Editors, asset pipelines, and UI frameworks stay minimal until a game needs them.

## Language and dependencies

The engine is written in Rust (stable toolchain, 2024 edition). The multithreaded chunk scheduler is where the hardest bugs live, and Rust's borrow checker rejects most data races at compile time. The tooling also gives the code-quality gates described later without extra infrastructure.

External crates are kept to a short, reviewed list: `winit` for windows and input, `wgpu` for rendering, `rapier2d` for rigid bodies, `rayon` or a small custom work-stealing pool for threads, `serde` for the scene and control formats, and `zstd` for snapshot compression. Godot was considered and dropped. It is MIT licensed, so its code could be reused, but it has nothing for per-cell simulation, and running the sim as a Godot extension would put a binding layer between the engine and every test.

## The determinism contract

Given the same starting world, the same material set, and the same sequence of inputs, every build of the engine on every supported platform produces the same world, bit for bit, regardless of thread count. Multiplayer, replays, regression tests, and agent tooling all rest on this contract, and it cannot be added later.

The contract is enforced by these rules:

1. The cell simulation and particle system use integer or fixed-point math only. Clippy's `float_arithmetic` lint is denied in the simulation crates.
2. There is no shared random number generator in the simulation. Randomness for a cell comes from a hash of (world seed, tick, x, y, purpose), so the value does not depend on which thread runs first.
3. Iteration order is fixed everywhere it can affect results. `HashMap` and `HashSet` with random state are banned in simulation crates through `clippy.toml` `disallowed-types`, and so are wall-clock time and thread-local RNGs.
4. Thread scheduling cannot change results. The checkerboard scheduler (below) is built so that no two chunks updated in parallel write to the same cell.
5. Rigid bodies run in Rapier with its `enhanced-determinism` feature, single-threaded, with bodies inserted and removed in a fixed order. [Rapier's determinism guide](https://rapier.rs/docs/user_guides/rust/determinism/) states that cross-platform results require IEEE 754-2008 compliant hardware and rule out SIMD lane width 8.
6. Rendering, audio, and lighting read from the simulation and never write to it. They are free to use floats and the GPU.

CI checks the contract directly. The same scenario runs with 1 thread and with 32 threads and must produce the same world hash. The same scenario runs on Linux x86_64, Windows x86_64, and macOS ARM64 and must produce the same hash. Any difference fails the build.

## World model

The world is an unbounded grid of cells split into 64x64 chunks. Each cell is 4 bytes: a 16-bit material id, an 8-bit flags field (including the tick-parity bit that stops a cell from being processed twice in one tick), and an 8-bit material-specific value such as temperature or remaining lifetime. A chunk is 16 KiB of cell data, small enough to stay in a core's L2 cache while it is updated.

Chunks are stored in a fixed-order map keyed by chunk coordinate. Each chunk tracks a dirty rect, the bounding box of cells that changed last tick, and a chunk whose dirty rect is empty sleeps until a neighbor writes into it. Chunks outside every player's active region are serialized to disk and unloaded, and world generation fills new chunks from the seed so that generation is part of the determinism contract.

Within a chunk, cells update bottom to top. The horizontal direction alternates by tick parity so that liquids do not drift to one side.

## Parallel scheduling

The scheduler follows the checkerboard scheme from the Noita talk. Chunks are split into four groups by the parity of their x and y chunk coordinates, and each tick runs four passes, one per group. Chunks in the same group are two chunks apart, so if every update writes at most 32 cells past its own chunk border, two chunks in the same pass can never touch the same cell. That limit means a cell moves at most 32 cells per tick (1,920 cells per second at 60 Hz). Anything faster is a free particle, described below.

Two chunks in the same pass can both write into the gap chunk between them, so dirty-rect updates to neighbors go through atomic min/max operations. The merged result is the same in any order, which keeps it deterministic.

The scheduler needs mutable access to a chunk and parts of its eight neighbors at once, which safe Rust cannot express. This is the only place in the engine where `unsafe` is allowed. It lives in one module, every block has a `// SAFETY:` comment that states the invariant it relies on, and the module's tests run under Miri in CI.

## Materials and reactions

Materials are data, loaded from files in the scene or game directory. A material definition includes its state (static, powder, liquid, gas, or fire), density, how far it spreads per tick, flammability, how heat moves through it, its melting and boiling behavior, and its render palette. Reactions are rules of the form A touching B becomes C and D, with a probability given as an integer out of 65,536 so that it stays inside the integer-only rule.

At load time the material set is compiled into lookup tables indexed by material id, so the inner loop never branches on strings or walks a list. The compiled set has a hash, and a multiplayer client cannot join a server whose material hash differs from its own.

The engine knows nothing about wands, spells, or any other game concept. Everything game-specific sits above the material layer.

## Free particles

Cells that are thrown, splashed, or blasted leave the grid and become free particles with fixed-point position and velocity. Particles are stored per chunk, updated in spawn-id order, and turned back into cells when they land. This is how the engine handles movement faster than 32 cells per tick, and it keeps explosions from breaking the checkerboard rule.

## Rigid bodies

A rigid body is a connected group of cells that moves as one piece, such as a falling slab of rock or a wooden crate. When terrain breaks free, a flood fill finds the connected component, marching squares traces its outline, Douglas-Peucker simplifies the outline, and the shape is split into convex pieces to make Rapier colliders. While a body moves, its cells are removed from the grid. At the end of each tick they are stamped back at the body's new position so that sand and water collide with them. When cells in a body are destroyed, the body re-runs the flood fill and splits if it is no longer connected.

Rapier uses floats. Cross-platform determinism holds only if Rapier's guarantees hold on every target, and this is the largest technical risk in the project (see Risks).

## Rendering

The renderer uses `wgpu`. Each loaded chunk maps to a slot in a texture array, and each frame uploads only dirty rects. Cell color comes from the material palette plus a per-cell variant so that sand reads as grains instead of a flat color.

Lighting is where the demos get their look. Emissive materials such as lava, fire, and glowing fungus light the scene through 2D global illumination (radiance cascades or a similar screen-space method). Bloom, heat haze over hot cells, and smoke that scatters light go on top. None of this feeds back into the simulation, so it does not need to be deterministic.

## Multiplayer

### Model

Multiplayer uses server-ordered lockstep. Every client and the server run the full simulation. Only inputs travel over the network: a player's movement, aim, and actions for a tick, usually under 16 bytes.

The server collects inputs, stamps each with the tick it applies to, and broadcasts the confirmed input set for each tick. The server does not wait for slow clients. An input that arrives after its tick's deadline is applied on the next open tick. A client simulates tick T only after it has the confirmed inputs for T.

### Latency

Waiting for confirmed inputs puts each client's world about half a round trip behind the server, plus a small buffer. To hide that, the player's own character is predicted locally and moves the moment a key is pressed. If the confirmed result differs, only the character is rewound and replayed, which is cheap because a character is a few hundred bytes of state. The world is never rolled back. A world change the player causes, such as digging, appears once it is confirmed, about one round trip later. A cosmetic preview can cover the gap.

### Desync detection and repair

Every 30 ticks, each client hashes each active chunk and sends a single combined hash to the server. On a mismatch, the server asks for the per-chunk hashes, finds the chunks that differ, and resends them along with any rigid bodies that overlap them. A desync is treated as a bug: every one is logged with the tick, chunk, and replay needed to reproduce it, and CI fails if the soak test records any.

### Joining

A late joiner receives a zstd-compressed snapshot of the loaded chunks, the rigid bodies, the particles, and the input log since the snapshot tick, then catches up by simulating forward faster than real time.

### Known limit

Every client simulates the active region around every player. If players spread far apart, each machine simulates several separate regions, and the slowest client sets the ceiling. The first version caps the total active area. A later version lets regions far from a client switch to state streamed from the server.

## Agent and developer hooks

The engine is meant to be developed by agents as much as by people. Determinism makes most of this cheap, because any run can be reproduced exactly from a seed and an input log.

The headless server binary exposes a JSON-RPC 2.0 control API over stdio and TCP. An MCP server wraps the same API so that Claude Code and similar tools can call it directly. The API covers loading a scene, stepping N ticks, painting and erasing cells, reading back a region as text or raw data, getting world and per-chunk hashes, reading per-phase performance counters, rendering a screenshot to PNG, and saving or loading a replay.

Scenes are plain text files. A scene has a header (size, seed, material set), a legend that maps characters to materials, an ASCII grid, and an optional expectations block, for example that after 120 ticks the region (10, 40)-(30, 50) holds at least 300 water cells, or that the world hash equals a stored value. An agent writes a test by writing one of these files.

The `fe` command-line tool runs scenes, tests, benchmarks, and replays. Three commands matter most for debugging. `fe replay --until <invariant>` finds the first tick where an invariant breaks. `fe diff <run-a> <run-b>` reports the first tick and chunk where two runs differ, which is the main tool for desync bugs. `fe test --bless` updates stored hashes on purpose and prints what changed.

Invariant checks run in debug builds and in CI: mass is conserved for materials with no active reactions, no cell is processed twice in one tick, every cell owned by a rigid body is accounted for, and dirty rects cover every cell that changed. Logging uses `tracing` with JSON output, so agents can parse it instead of scraping text.

## Code quality rules

These rules are enforced in CI, and a pull request that breaks one does not merge.

1. `cargo fmt --check` passes.
2. `cargo clippy --all-targets -- -D warnings` passes with the `pedantic` group on, plus `unwrap_used`, `expect_used`, `undocumented_unsafe_blocks`, and `missing_docs` for public items. Exceptions are allowed only through `#[expect(...)]` with a reason string.
3. Every crate has `#![forbid(unsafe_code)]` except the scheduler crate, whose unsafe code is tested under Miri.
4. Simulation crates deny `float_arithmetic` and the disallowed types listed in the determinism contract.
5. `cargo deny` passes for licenses, advisories, and duplicate dependencies.
6. Every material rule has a scene test. Property tests (`proptest`) cover conservation and order-independence. The scene loader and the network message decoder are fuzzed with `cargo-fuzz`.
7. The determinism tests (thread-count and cross-platform) pass.
8. Benchmarks run in CI with `criterion`, and a slowdown of more than 5% on any tracked benchmark fails the build unless the commit message explains it.

## Performance targets

The reference machine is a desktop with 8 physical cores from 2023 or later and a mid-range discrete GPU. The tick rate is fixed at 60 Hz, so a tick has 16.6 ms in total, of which the cell simulation gets 8 ms.

These numbers are starting targets, not measurements. Published per-cell numbers for falling-sand engines are rough: the Rust engine [sandengine](https://github.com/Ijtihed/sandengine) reports about 5.6 ns per cell on a saturated grid and about 0.56 ns per cell on a settled one, without naming its hardware. Phase 2 measures the real numbers, and this table is updated with them instead of being quietly lowered.

| Area | Target |
| --- | --- |
| Cost of one moving cell, single thread | 5 ns or less |
| Cost of a sleeping chunk | under 0.5 ns per cell, amortized |
| Fully active cells per tick within the 8 ms sim budget | 4 million (2048x2048); stretch goal 8 million |
| Parallel speedup on 8 cores, fully active scene | 6x or better |
| Loaded world in memory | 16 million cells (4096x4096) at 4 bytes per cell, 64 MiB |
| Free particles | 100,000 at 1 ms or less per tick |
| Rigid bodies | 1,000 dynamic bodies stepped in 3 ms or less |
| Building a body from a 64x64 cell region | 0.2 ms or less |
| Frame rate | 60 fps or higher at 2560x1440 |
| Chunk texture upload | 1 ms or less per frame with 256 dirty chunks |
| Upload per player | 1 KB/s or less average |
| Download per client, 8 players | 8 KB/s or less average, excluding joins and repairs |
| Input delay | 3 ticks (50 ms) on LAN; playable up to 150 ms round trip |
| Late-join snapshot | 4 MB or less for a 16-million-cell world; joined in under 3 s on a 25 Mbit/s link |
| Determinism soak | zero hash mismatches over 1,000,000 ticks on all three CI platforms |
| Headless speed | 1,000 ticks per second or more for a fully active 512x512 scene on 8 cores |
| Control API overhead | 50 microseconds or less per call, excluding simulation time |
| Full CI run | 10 minutes or less |

## Demos

Every demo is a deterministic scene file with a scripted camera and scripted events. The same file renders to video headless, runs as a benchmark, and is checked as a regression test by its world hash. A demo that drifts is a bug.

### Showcase scenes

The showcase scenes are built to look good in a trailer and to push a different part of the engine each.

Foundry is a white-hot metal pour into stone molds. The metal cools through yellow and orange to a dark solid, and where it meets a water channel it throws up steam. Nearly all the light in the scene comes from the metal itself, so it tests emissive lighting, heat transfer, and phase changes together.

Glowcave is a huge cavern under a rainstorm. Water seeps through soil, gathers into waterfalls, and fills pools, while mist drifts in shafts of light from cracks in the ceiling and blue fungus glows along the walls. It is the calm scene, and it tests liquids at scale and soft lighting.

Demolition is a tall tower of brick, timber, and glass. Charges go off floor by floor, the structure breaks into hundreds of rigid bodies that tumble and shatter, dust rolls outward, and the timber catches fire. It tests rigid body creation, splitting, and free particles under load.

Firestorm is a forest on a hillside that catches fire in a crosswind. Leaves flash first, trunks burn slowly, smoke rises and lights orange from below, and embers carry the fire across a river. It tests fire spread, gases, and wind across a wide active area.

Dam break is the stress test: a reservoir of 4 million water cells released at once into a valley. It doubles as the main benchmark for the 4-million-cell target.

### Multiplayer demos

Each multiplayer demo shows an on-screen overlay with the tick, the world hash, bandwidth per player, round-trip time, and a desync counter.

Shared sandbox puts 8 players in one world, painting, digging, and blowing things up. Its job is the bandwidth graph, which should stay flat while the screen fills with chaos.

Siege is 2v2. Each team builds a fort from a material budget, then the teams bombard each other until the forts collapse into rigid bodies. It shows that rigid body physics stays in sync across machines.

Flood rescue is co-op. Water rises through a cave system and players dig channels to steer it away from villagers. Large active liquid regions make it the stress test for lockstep.

Chaos soak runs bot players under 150 ms of simulated latency and 2% packet loss, with a player joining mid-explosion. The desync counter has to stay at zero for the whole run. It is a test that also looks good on screen.

## Game ideas

The engine is meant to support games well outside Noita's genre. These ideas show the range:

- Siege builder: teams build forts from real materials, then bombard each other. Wood burns, stone cracks, sand fills the breaches. Lockstep keeps a 1v1 with huge destruction cheap on bandwidth.
- Wildfire crew: co-op firefighting where fire spreads by wind, fuel, and moisture. Players dig firebreaks, drop water, and backburn.
- Flooded hull: a co-op submarine where hull breaches flood cell by cell. The crew welds, pumps, and reroutes water through compartments.
- Alchemist's run: a roguelike where the reaction table is shuffled for each seed, so players work out the chemistry by experiment on every run.
- Dig race: competitive mining where you can flood or collapse your opponent's tunnels.
- Rising lava: a last-player-standing game in fully destructible terrain while lava climbs from below.
- Terrarium: a slow god game with rain, erosion, and plants and fungi that grow by cell rules.
- Verified puzzles: physics puzzles where a solution is an input log. The server replays it to check it, so leaderboards cannot be faked.
- Agent arena: AI agents play through the control API against each other or against people, with humans watching.
- Orbital salvage: wrecked ships built from cells that you cut apart with lasers. The pieces drift as rigid bodies in zero gravity.

## Crate layout

The workspace starts with these crates:

| Crate | Purpose |
| --- | --- |
| `fe-core` | Cells, chunks, the world map, fixed-point math, hashing |
| `fe-materials` | Material and reaction file format, compiled lookup tables |
| `fe-sim` | Cell rules, free particles, the checkerboard scheduler (the only crate with `unsafe`) |
| `fe-physics` | Bridge between pixel bodies and Rapier |
| `fe-scene` | Text scene format and expectations |
| `fe-net` | Lockstep protocol, snapshots, desync detection |
| `fe-render` | `wgpu` renderer and lighting |
| `fe-control` | JSON-RPC control API |
| `fe-mcp` | MCP server over the control API |
| `fe` | Command-line tool: run, test, bench, replay, diff |
| `fe-client` | Windowed client |
| `fe-server` | Headless server |

## Build order

Each phase ends with a demo that has to run before the next phase starts.

1. Phase 0, foundation. Workspace, CI with every quality gate, the scene format, and the `fe` tool skeleton. Exit: an empty scene loads, steps, and hashes in CI on all three platforms.
2. Phase 1, deterministic core. Single-threaded simulation of powder, liquid, gas, fire, temperature, and reactions; the control API and MCP server; a plain windowed viewer with no lighting. Exit: scene tests for every material, the cross-platform hash test, and an agent driving the engine end to end through MCP.
3. Phase 2, parallel and large worlds. The checkerboard scheduler, dirty rects, sleeping chunks, chunk streaming, world generation, and benchmarks. Exit: the thread-count test passes, the per-cell targets are measured and the table above is updated, and Dam break runs at target.
4. Phase 3, rendering. Texture streaming, emissive lighting, bloom, heat haze, smoke scattering, and headless video capture. Exit: Glowcave and Firestorm render to video.
5. Phase 4, bodies and particles. Free particles, pixel rigid bodies, splitting, and the cross-platform Rapier test. Exit: Demolition and Foundry run, and their hashes match on all platforms.
6. Phase 5, multiplayer. Lockstep protocol, character prediction, desync detection and repair, late join. Exit: all four multiplayer demos, with Chaos soak at zero desyncs.
7. Phase 6, scripting. Game logic and mods in WebAssembly with NaN canonicalization on so that float results stay deterministic, plus mod hashes in the join handshake. Exit: a playable prototype of one game idea, likely Siege builder because it uses every system.

## Risks

Cross-platform float determinism in Rapier is the largest risk. The Rapier docs promise it under conditions, but this project needs it to hold for 1,000,000-tick soaks with hundreds of bodies. Phase 4 tests it first. If it fails, the fallback is a fixed-point rigid body solver written for this engine, which is several weeks of work and would delay multiplayer.

The cost of simulating every player's region on every client sets the real player limit, and it is unknown until Phase 5. Eight players spread across a large world may be too much for the reference machine without the region-streaming fallback.

Desync repair is harder than resending a chunk. Rigid bodies and particles cross chunk borders, so a repair has to move them consistently too. The repair path gets its own fuzz tests.

Lockstep adds input delay to world interaction. Character prediction hides it for movement, but digging and building still lag by one round trip. Whether 150 ms feels acceptable is a playtest question, not an engineering one.

The 32-cell reach limit per tick is set by the checkerboard pattern. Anything faster has to go through the particle system, and materials that need long-range effects in one tick, such as lightning or lasers, need a separate deterministic pass that runs after the four checkerboard passes.

Headless screenshots in CI need a GPU or a software Vulkan driver such as lavapipe. Rendering is not deterministic across drivers, so screenshots are for people and agents to look at, and world hashes remain the only pass/fail check.
