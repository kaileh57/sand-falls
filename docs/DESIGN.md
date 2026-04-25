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

The world is an unbounded grid of cells split into 64x64 chunks. Each cell is 4 bytes: a 12-bit material id, 4 flag bits (including the tick-parity bit that stops a cell from being processed twice in one tick, and the flag that marks a cell as part of a rigid body), an 8-bit temperature in 10-degree steps from -50 C to 2500 C, and an 8-bit material-specific value that holds remaining lifetime, hit points, or fall speed. A chunk is 16 KiB of cell data, small enough to stay in a core's L2 cache while it is updated.

Chunks are stored in a fixed-order map keyed by chunk coordinate. Each chunk tracks a dirty rect, the bounding box of cells that changed last tick, and a chunk whose dirty rect is empty sleeps until a neighbor writes into it. Chunks outside every player's active region are serialized to disk and unloaded, and world generation fills new chunks from the seed so that generation is part of the determinism contract.

Within a chunk, cells update bottom to top. The horizontal direction alternates by tick parity so that liquids do not drift to one side.

Liquids level in two ways. A cell that can find a drop within its spread distance moves toward it. A cell resting on top of liquid with an open side walks along the surface until it drops off an edge; it cannot climb onto a higher step, so the walk always ends lower and staircases flatten however wide their steps are. A level pool sleeps, except for a partly filled top row, whose cells keep shimmering. That keeps only the surface rows of a pool awake.

Powder and liquid cells fall with a speed stored in their 8-bit value, in sixteenths of a cell per tick, capped at 255 (about 16 cells per tick). Gravity adds 7 speed units every 16 ticks, spread over the ticks by a fixed dither on the tick number rather than by randomness. A cell moving slower than one cell per tick hovers in place and keeps its speed until it has built up a whole step. A falling cell blocked by another falling cell takes the slower of the two speeds instead of stopping, so a stream stays together instead of breaking into spray. Gases and fire, which have lifetimes in their value, move one cell per tick.

Heat moves between neighboring cells. Each material has a heat capacity (by default 20 for static solids, 14 for powders, 30 for liquids and 1 for gases and fire), and an exchange changes each side in inverse proportion to its capacity: with a rate k from the two conductivities, the cell gains diff * k * c_other / (128 * (c_self + c_other)) and its neighbor loses the matching share, each rounded stochastically from separate random bits. The total of capacity times temperature is conserved in expectation, and with equal capacities the rule reduces to the old one exactly. Air holds 1/50 of what a metal cell does, so it conducts freely without draining hot metal: it warms up, rises and shows as heat haze.

Melting, boiling and freezing can have a latent heat. A cell past its threshold changes with a chance of overshoot / latent per tick and is otherwise held at the threshold, so it sits on a plateau and takes in or gives off about the latent heat first. Water boils with 540 C of latent heat, so a quench boils little water, and molten metal and lava freeze with 300 C and 250 C, so a pour stays liquid long enough to fill a mold. A liquid that freezes while it is falling, hanging over open space, or resting on a different fluid or powder becomes its solid's debris (lava turns to rubble, molten metal to scrap) instead of a floating static crust. A crust on its own melt stays static.

Reaction products start at the reactants' capacity-weighted mean temperature. Water on lava gives hot steam and glowing obsidian, and water on burning wood leaves wood below its ignition point, which is how water puts fires out.

## World scale

Every system uses one scale: 10 cells per meter, and 60 ticks per second. Gravity is one constant in `fe-core`, `GRAVITY = Fx(1792)`, which is 7/256 cells per tick squared, 98.4 cells per second squared, or 9.84 m/s^2. Cells, free particles, rigid bodies and the avatar all fall with it; the rigid body world converts it to Rapier units with `CELLS_PER_METER`, and a unit test checks that the two agree. A water cell weighs 1 kg. Scenes are built to the same scale: a 4 m floor is 40 cells, and the 12-cell avatar is a small 1.2 m figure. Before this, cells, particles and bodies fell at 450 cells per second squared and the avatar at 900, so every fall played more than twice as fast as it should and a collapsing building looked like a time lapse.

## Parallel scheduling

The scheduler follows the checkerboard scheme from the Noita talk. Chunks are split into four groups by the parity of their x and y chunk coordinates, and each tick runs four passes, one per group. Chunks in the same group are two chunks apart, so if every update writes at most 32 cells past its own chunk border, two chunks in the same pass can never touch the same cell. That limit means a cell moves at most 32 cells per tick (1,920 cells per second at 60 Hz). Anything faster is a free particle, described below.

Two chunks in the same pass can both write into the gap chunk between them, so dirty-rect updates to neighbors go through atomic min/max operations. The merged result is the same in any order, which keeps it deterministic.

The scheduler needs mutable access to a chunk and parts of its eight neighbors at once, which safe Rust cannot express. This is the only place in the engine where `unsafe` is allowed. It lives in one module, every block has a `// SAFETY:` comment that states the invariant it relies on, and the module's tests run under Miri in CI.

## Materials and reactions

Materials are data, loaded from files in the scene or game directory. A material definition includes its state (static, powder, liquid, gas, or fire), density, how far it spreads per tick, flammability, how heat moves through it, its melting and boiling behavior, and its render palette. Reactions are rules of the form A touching B becomes C and D, with a probability given as an integer out of 65,536 so that it stays inside the integer-only rule.

At load time the material set is compiled into lookup tables indexed by material id, so the inner loop never branches on strings or walks a list. The compiled set has a hash, and a multiplayer client cannot join a server whose material hash differs from its own.

Cells with a lifetime (fire, smoke, embers) count it down each tick with a chance set by `lifetime_rate`, so an 8-bit lifetime can last longer than 255 ticks. When it runs out, the cell becomes its `residue` with the residue's chance and its `decay` otherwise. Burning wood, leaves and coal decay mostly to smoke and leave ash or embers only sometimes, so a burned forest leaves a thin layer of ash instead of one ash cell per plank. Embers glow at 400 C for a second or so and water puts them out.

Gases rise or sink by buoyancy. Each tick a gas tries a vertical move first with a chance of |density| * 32 / 256, up when it is lighter than air and down when it is heavier, and drifts sideways otherwise. Steam always rises, smoke rises on 62% of ticks and mist on 12%, so mist lingers in a cave while steam boils off. Dust is a gas slightly heavier than air that hangs and settles for about eight seconds; explosions and shattering bodies release it, always into air cells, so no mass is lost. Fire in wind tilts downwind with a chance of |wind| / 4 and keeps some sideways jitter even in the strongest wind.

The engine knows nothing about wands, spells, or any other game concept. Everything game-specific sits above the material layer.

## Free particles

Cells that are thrown, splashed, or blasted leave the grid and become free particles with fixed-point position and velocity. Particles are stored per chunk, updated in spawn-id order, and turned back into cells when they land. They fall with the same gravity as everything else. A particle that lands where no neighboring cell is free searches further out, then bounces up for a while, and finally takes the first free cell up its column, so particles do not delete what they land in. This is how the engine handles movement faster than 32 cells per tick, and it keeps explosions from breaking the checkerboard rule.

## Rigid and destructible bodies

A rigid body is a connected group of cells that moves as one piece, such as a falling slab of rock, a wooden crate, or a ship. Bodies come from three places: authored shapes placed by a scene or a game (crates, vehicles, doors), terrain that loses its support, and debris from a body that breaks apart. All three end up as the same thing in the engine, so every body can be damaged and split.

### Building a body

A flood fill finds the connected cells, and the cells are merged into rectangles (horizontal runs joined with identical runs below) that become a compound of Rapier cuboids. Terrain near bodies is one voxel collider per chunk. Rapier collides voxel shapes with other shapes but not with each other, which is why bodies use cuboids. While a body moves, its cells are removed from the grid. At the end of each tick they are stamped back at the body's new position so that sand, water, and other bodies collide with them.

### Structural support

Solid materials are either anchored (bedrock, and anything a scene marks as fixed) or structural (brick, stone, wood, metal, glass). Structural cells stay static as long as they connect to an anchor through other solid cells. When structural cells are destroyed in a tick, the engine runs a connectivity check over the affected area, and any component that no longer reaches an anchor becomes a rigid body and falls. Undermine a cliff and it drops. Knock out the base of a tower and the tower comes down.

The check is bounded so that one explosion cannot stall a tick. It searches up to 65,536 cells per component, and a component that hits the limit is treated as supported for that tick and checked again on the next. The first version uses connectivity only. A stress model, where a long beam of weak material can snap under its own weight, comes later if a game needs it.

### Damage and fracture

Every solid cell has hit points, stored in its 8-bit material value. Explosions, impacts, heat, and acid reduce them. A cell at zero becomes the debris its material defines: brick turns to rubble, wood to splinters, glass to shards, and metal to scrap. Depending on the material, debris becomes powder or free particles.

When a body is destroyed in pieces, it re-runs the flood fill and splits into one body per connected part. When one physics step changes a body's velocity, apart from gravity, by more than its fracture strength divided by 32 (in m/s), the body shatters along a Voronoi pattern whose seed points come from the deterministic hash, so the same hit cracks the same way on every machine. Glass breaks at a 1.9 m/s jolt, wood at 8.1, brick at 10 (about a 5 m drop) and concrete at 16. The test looks at impacts only, so the weight of a resting pile never shatters the bodies at its bottom. A body's fracture strength is the mass-weighted mean of its brittle cells, so one glass pane does not make a brick tower shatter like glass. Steel bends nothing and only loses cells.

A body split into parts builds each part from the parent's own local grid and keeps the parent's rotation, so no cells are lost to rasterizing a rotated part, and each part moves with the parent's velocity at its own center of mass. A body that loses cells but stays in one piece keeps its Rapier body, velocity and contacts and only gets a new collider and mass. Explosions push a body at the cell of it nearest the blast, with an impulse that scales with the square root of its mass, so a charge at the foot of a tall tower tips it over.

### Bodies in liquids

Bodies and liquids push on each other. When a body is stamped into the grid, liquid and powder cells in its way are moved to the nearest free cell, sideways first, and if there is no air within 4 cells they go up the column to the first air, usually the space the body just left. A cell that still finds no room is counted in the physics stats as lost. Each tick, the engine walks the body's footprint and its hull cavity column by column; cells below the first one touching a liquid or powder count as submerged. Lift from the medium's density acts at the center of mass, plus drag from its viscosity. Wood floats, stone sinks, and a hull goes down once water fills its cavity. Lift at the center of mass means bodies do not right themselves; applying it at the wet cells' centroid made spinning bodies feed their own spin through the air they leave behind.

### Joints

Authored bodies can be connected with Rapier joints: hinges for doors, fixed joints for multi-part structures, and chains of small bodies for ropes and bridges. Each joint has a break force, so a bridge snaps when it is overloaded or its supports are blown off.

### Keeping body counts bounded

Destruction produces debris fast, and every body costs physics time. Pieces smaller than 16 cells never become bodies and go straight to particles or powder. A small body that has been at rest for 5 seconds merges back into static terrain. These limits keep a large explosion from producing thousands of tiny bodies.

### Determinism

Rapier uses floats. Cross-platform determinism holds only if Rapier's guarantees hold on every target, and this is the largest technical risk in the project (see Risks). Everything around Rapier, including the connectivity check, the fracture pattern, the displacement search, and the order in which bodies are created and removed, uses integer math and fixed ordering.

## Rendering

The renderer works on the CPU at cell resolution and scales up to pixels, so frames can be rendered headless on any machine, including CI runners without a GPU. The windowed client copies each frame into the window's surface texture through `wgpu`. Cell color comes from the material palette plus a per-cell variant so that sand reads as grains instead of a flat color.

Lighting is where the demos get their look. Emissive materials and hot cells (which glow with their temperature) seed a light field that spreads through open space by repeated sweeps and is then softened with a blur. Sky light falls down each column until it meets a structure, which gives light shafts through cracks; foliage and smoke only dim it. Air underground shows a dim rock back wall that catches nearby light, and rock far from open space sinks into ambient occlusion. Gas is drawn as a soft volume: its density and color are blurred over neighboring cells, so smoke, steam, mist and dust read as clouds rather than specks. Hot cells glow by temperature with a brightness that keeps rising up to about 2000 C, so a white-hot pour and metal about to freeze look different, and lava dims toward red as it nears its freezing point. Bloom, heat haze over hot air, and smoke that scatters light go on top. None of this feeds back into the simulation, so it does not need to be deterministic.

A 1920x1080 frame (960x540 cells) takes about 0.2 s on four cores. That is enough for recording video and for a playable window at smaller sizes, but not the 60 fps at 1440p the original target asked for; moving the light pass to a GPU compute shader is the obvious next step.

## Multiplayer

### Model

Multiplayer uses server-ordered lockstep. Every client and the server run the full simulation. Only inputs travel over the network: a player's movement, aim, and actions for a tick, usually under 16 bytes.

The server collects inputs, stamps each with the tick it applies to, and broadcasts the confirmed input set for each tick. The server does not wait for slow clients. An input that arrives after its tick's deadline is applied on the next open tick. A client simulates tick T only after it has the confirmed inputs for T.

### Latency

Waiting for confirmed inputs puts each client's world about half a round trip behind the server, plus a small buffer. To hide that, the player's own character is predicted locally and moves the moment a key is pressed. If the confirmed result differs, only the character is rewound and replayed, which is cheap because a character is a few hundred bytes of state. The world is never rolled back. A world change the player causes, such as digging, appears once it is confirmed, about one round trip later. A cosmetic preview can cover the gap.

### Desync detection and repair

Every 30 ticks, each client sends the server its state hash split by subsystem: cells, particles, bodies, and characters. When only the cells differ, the server asks for the client's per-chunk hashes at that tick, finds the chunks that differ, and sends its own copies of them and their neighbors, to be applied when the client reaches the server's current tick. When anything else differs, or three reports in a row mismatch, the server sends a full snapshot instead. A desync is treated as a bug: CI runs a soak at 150 ms round trip and 2% loss and fails if the server records any.

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

The targets were set before any code existed, from rough published numbers: the Rust engine [sandengine](https://github.com/Ijtihed/sandengine) reports about 5.6 ns per cell on a saturated grid. The measured column comes from the 4-core cloud container this engine was built in, not the reference machine, so multi-core numbers there understate an 8-core desktop. Several targets are missed, and the table says so rather than moving the targets.

The worst case for the cell passes is fast-falling sand: a cell falling at up to 16 cells per tick does up to 16 steps of work each tick, so "cost per visited cell" includes that. Profiling shows the cost spread across many small steps (about 450 instructions per visited cell) with no single hot spot left after the fixes in phase 2: buffering dirty marks per job instead of using atomics per cell took it from 253 ns to 25 ns.

| Area | Target | Measured (4-core container) |
| --- | --- | --- |
| Cost of one moving cell, single thread | 5 ns or less | 25 ns per visited cell, fully active falling sand. Missed. |
| Cost of a sleeping chunk | under 0.5 ns per cell, amortized | 219 ns per tick for a settled 512x512 world, about 0.001 ns per cell. Met. |
| Fully active cells per tick within the 8 ms sim budget | 4 million (2048x2048); stretch goal 8 million | 726,000 visited cells per tick in 6.5 ms on 4 threads. Missed by about 2.5x after scaling to 8 cores. |
| Parallel speedup on 8 cores, fully active scene | 6x or better | 2.8x on 4 threads (sand storm). Not measured on 8 cores. |
| Loaded world in memory | 16 million cells (4096x4096) at 4 bytes per cell, 64 MiB | 4 bytes per cell as planned; the 8.4-million-cell dam break loads in about 34 MiB of cells. |
| Free particles | 100,000 at 1 ms or less per tick | Not measured yet. |
| Rigid bodies | 1,000 dynamic bodies stepped in 3 ms or less | Not measured yet. |
| Building a body from a 64x64 cell region | 0.2 ms or less | Not measured yet. |
| Support check after an explosion, 65,536-cell component | 1 ms or less | Not measured yet. |
| Shattering a 64x64 cell body | 0.5 ms or less | Not measured yet. |
| Frame rate | 60 fps or higher at 2560x1440 | About 5 fps at 1920x1080 on the CPU renderer. Missed; see Rendering. |
| Chunk texture upload | 1 ms or less per frame with 256 dirty chunks | Replaced by a full-frame copy from the CPU renderer. |
| Upload per player | 1 KB/s or less average | 0.55 to 0.84 KB/s at 150 ms round trip and 2% loss. Met. |
| Download per client, 8 players | 8 KB/s or less average, excluding joins and repairs | About 2.4 KB/s with 4 players. 8 players not measured yet. |
| Input delay | 3 ticks (50 ms) on LAN; playable up to 150 ms round trip | 4 ticks on localhost; 14 to 15 ticks at 150 ms round trip, with the local character predicted. |
| Late-join snapshot | 4 MB or less for a 16-million-cell world; joined in under 3 s on a 25 Mbit/s link | 98 KB for the 8.4-million-cell dam break; joined in about 200 ms at 150 ms round trip. Met, though textured terrain compresses less than that world. |
| Determinism soak | zero hash mismatches over 1,000,000 ticks on all three CI platforms | Zero mismatches in 3,000-tick soaks at 150 ms and 2% loss, golden hashes and 1-versus-4-thread checks in CI. The 1,000,000-tick soak has not been run. |
| Headless speed | 1,000 ticks per second or more for a fully active 512x512 scene on 8 cores | Not measured yet. |
| Control API overhead | 50 microseconds or less per call, excluding simulation time | Not measured yet. |
| Full CI run | 10 minutes or less | Not measured yet. |

## Demos

Every demo is a deterministic scene file with a scripted camera and scripted events. The same file renders to video headless, runs as a benchmark, and is checked as a regression test by its world hash. A demo that drifts is a bug.

### Showcase scenes

The showcase scenes are built to look good in a trailer and to push a different part of the engine each.

Foundry is a white-hot metal pour into stone molds. The metal cools through yellow and orange to a dark solid, and where it meets a water channel it throws up steam. Nearly all the light in the scene comes from the metal itself, so it tests emissive lighting, heat transfer, and phase changes together.

Glowcave is a huge cavern under a rainstorm. Water seeps through soil, gathers into waterfalls, and fills pools, while mist drifts in shafts of light from cracks in the ceiling and blue fungus glows along the walls. It is the calm scene, and it tests liquids at scale and soft lighting.

Demolition is a tall tower of brick, timber, and glass. Charges go off floor by floor, the structure breaks into hundreds of rigid bodies that tumble and shatter, dust rolls outward, and the timber catches fire. It tests rigid body creation, splitting, and free particles under load.

Firestorm is a forest on a hillside that catches fire in a crosswind. Leaves flash first, trunks burn slowly, smoke rises and lights orange from below, and the river stops the fire: flames and embers that reach the water go out. It tests fire spread, gases, and wind across a wide active area.

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
| `fe-core` | Cells, coordinates, fixed-point math, hashing, deterministic randomness |
| `fe-materials` | Material and reaction file format, compiled lookup tables |
| `fe-sim` | Cell rules, free particles, the checkerboard scheduler (the only crate with `unsafe`) |
| `fe-physics` | Bridge between pixel bodies and Rapier |
| `fe-engine` | The full tick, commands, character tools, streaming, world generation, hashing, snapshots |
| `fe-scene` | Text scene format and expectations |
| `fe-net` | Lockstep protocol, snapshots, desync detection |
| `fe-render` | CPU renderer and lighting, PNG output, overlay font |
| `fe-control` | JSON-RPC control API |
| `fe-mcp` | MCP server over the control API |
| `fe` | Command-line tool: run, test, bench, replay, diff |
| `fe-client` | Windowed client |
| `fe-server` | Headless server, soak harness, multiplayer demo recorder |

## Build order

Each phase ends with a demo that has to run before the next phase starts.

1. Phase 0, foundation. Workspace, CI with every quality gate, the scene format, and the `fe` tool skeleton. Exit: an empty scene loads, steps, and hashes in CI on all three platforms.
2. Phase 1, deterministic core. Single-threaded simulation of powder, liquid, gas, fire, temperature, and reactions; the control API and MCP server; a plain windowed viewer with no lighting. Exit: scene tests for every material, the cross-platform hash test, and an agent driving the engine end to end through MCP.
3. Phase 2, parallel and large worlds. The checkerboard scheduler, dirty rects, sleeping chunks, chunk streaming, world generation, and benchmarks. Exit: the thread-count test passes, the per-cell targets are measured and the table above is updated, and Dam break runs at target.
4. Phase 3, rendering. Texture streaming, emissive lighting, bloom, heat haze, smoke scattering, and headless video capture. Exit: Glowcave and Firestorm render to video.
5. Phase 4, bodies and particles. Free particles, pixel rigid bodies, structural support, damage and fracture, buoyancy, joints, and the cross-platform Rapier test. Exit: Demolition and Foundry run, and their hashes match on all platforms.
6. Phase 5, multiplayer. Lockstep protocol, character prediction, desync detection and repair, late join. Exit: all four multiplayer demos, with Chaos soak at zero desyncs.
7. Phase 6, scripting. Game logic and mods in WebAssembly with NaN canonicalization on so that float results stay deterministic, plus mod hashes in the join handshake. Exit: a playable prototype of one game idea, likely Siege builder because it uses every system.

## Risks

Cross-platform float determinism in Rapier is the largest risk. The Rapier docs promise it under conditions, but this project needs it to hold for 1,000,000-tick soaks with hundreds of bodies. Phase 4 tests it first. If it fails, the fallback is a fixed-point rigid body solver written for this engine, which is several weeks of work and would delay multiplayer.

The cost of simulating every player's region on every client sets the real player limit, and it is unknown until Phase 5. Eight players spread across a large world may be too much for the reference machine without the region-streaming fallback.

Desync repair is harder than resending a chunk. Rigid bodies and particles cross chunk borders, so a repair has to move them consistently too. The repair path gets its own fuzz tests.

Lockstep adds input delay to world interaction. Character prediction hides it for movement, but digging and building still lag by one round trip. Whether 150 ms feels acceptable is a playtest question, not an engineering one.

The 32-cell reach limit per tick is set by the checkerboard pattern. Anything faster has to go through the particle system, and materials that need long-range effects in one tick, such as lightning or lasers, need a separate deterministic pass that runs after the four checkerboard passes.

Rendering ended up on the CPU instead of the GPU, so headless screenshots and videos need no GPU at all. They are still for people and agents to look at; world hashes remain the only pass/fail check.

### Risks that came true

The cell pass costs about 25 ns per visited cell on one thread, against a 5 ns target. Most of the remaining cost is the neighbor lookup across chunk borders and the per-cell rule dispatch. A packed 2x2 cell layout or SIMD scanning of air runs are the next things to try.

The CPU renderer draws a 1080p frame in about 0.2 s, roughly 5 frames per second. That is fine for videos and agents but far from 60 fps at 1440p. Moving light propagation and bloom to a compute shader is the fix, and the renderer's output format already matches a texture upload.

Liquid surfaces with a partly filled top row never fully sleep, because single cells keep walking along the surface. The cost is bounded (a settled lake visits fewer than 3,000 cells per tick) but it is not zero.

Every material used to have the same heat capacity, so air cooled hot metal as fast as stone did, and air conductivity was set very low to work around it. Materials now have a heat capacity and transitions a latent heat (see the world model), so air conducts normally and holds little heat, and a pour stays liquid while it fills its molds. Cooling into the surroundings is still conduction only; there is no separate radiation or Newtonian cooling term.

Buoyancy is applied at the center of mass with no righting torque, so a floating body keeps whatever angle it lands at. Strong angular damping hides this in the demos.