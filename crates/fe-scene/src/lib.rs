//! The `.scene` text format.
//!
//! A scene is a plain text file an agent or a person can write by hand:
//!
//! ```text
//! [scene]
//! name = sand pile
//! size = 128 96
//! seed = 7
//!
//! [legend]
//! # bedrock
//! s sand
//! w wood body
//!
//! [grid 0 80]
//! ...sss.......ww.....
//! ####################
//!
//! [build]
//! fill 0 90 128 96 stone
//!
//! [events]
//! at 30: explode 64 70 8 120
//! every 10 from 0 to 200: paint 20 5 2 water
//!
//! [camera]
//! view 128 96
//! at 0: 64 48
//!
//! [expect]
//! after 300: count sand in 0 0 128 80 == 0
//! after 300: asleep
//! after 300: temp sand in 0 80 128 96 < 100
//! ```
//!
//! `drawn water in X0 Y0 X1 Y1 >= N` counts the cells drawn as a liquid in a
//! rectangle: its cells, and the air that a stream of it in the air is drawn
//! across (see `fe_sim::liquid::drawn`).
//!
//! Comments start with `//` everywhere except inside `[grid]` blocks.
//!
//! A `script = PATH` key in `[scene]` names a WebAssembly game script, relative
//! to the scene file. [`load`] reads it; scenes parsed from text get it from
//! [`attach_script`]. The [`Runner`] calls its `on_init` after setup and its
//! `on_tick` at the start of every tick.
#![forbid(unsafe_code)]

mod build;
mod parse;
mod run;

pub use build::{build, load_materials};
pub use parse::{
    Camera, CameraKey, Check, Cmp, Expect, InputSpec, LegendEntry, Op, ParamSpec, Scene,
    SceneError, ScriptBytes, Setup, Timed, When, parse,
};
pub use run::{CheckResult, Frame, Replay, RunSummary, Runner};

/// Reads and parses a scene file, and loads its script if it names one.
pub fn load(path: &std::path::Path) -> Result<Scene, SceneError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| SceneError::new(0, format!("cannot read {}: {e}", path.display())))?;
    let mut scene = parse(&text)?;
    let dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    attach_script(&mut scene, dir)?;
    Ok(scene)
}

/// Reads the script a scene names, relative to `base_dir`. Does nothing if the
/// scene has no script or it is already loaded.
pub fn attach_script(scene: &mut Scene, base_dir: &std::path::Path) -> Result<(), SceneError> {
    let Some((rel, line)) = &scene.script else { return Ok(()) };
    if scene.script_wasm.is_some() {
        return Ok(());
    }
    let path = base_dir.join(rel);
    let bytes = std::fs::read(&path).map_err(|e| {
        SceneError::new(*line, format!("cannot read script {}: {e}", path.display()))
    })?;
    scene.script_wasm = Some(ScriptBytes(bytes.into()));
    Ok(())
}

/// [`fe_script::mod_hash`] of a scene's script, or 0 if it has none.
pub fn mods_hash(scene: &Scene) -> u64 {
    scene.script_wasm.as_ref().map_or(0, |w| fe_script::mod_hash(&w.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::sync::Arc;

    const PILE: &str = "\
[scene]
name = pile
size = 128 96
seed = 3

[legend]
# bedrock
s sand
w wood body
. air

[grid 0 60]
....ssssssss....................wwwwwwww..........
....ssssssss....................wwwwwwww..........
....ssssssss....................wwwwwwww..........

[build]
fill 0 90 128 96 bedrock   // floor

[events]
at 5: paint 100 10 3 water

[expect]
after 200: count sand in 0 0 128 85 == 0
after 200: count sand in 0 0 128 96 == 24
after 200: count water in 0 0 128 96 >= 20
after 200: bodies >= 0
";

    fn mats() -> Arc<fe_materials::MaterialSet> {
        Arc::new(fe_materials::base_set())
    }

    #[test]
    fn parses_every_section() {
        let s = parse(PILE).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(s.name, "pile");
        assert_eq!(s.bounds, fe_core::IRect::new(0, 0, 128, 96));
        assert_eq!(s.legend.len(), 4);
        assert!(s.legend[&'w'].body);
        assert_eq!(s.setup.len(), 2);
        assert_eq!(s.timeline.len(), 1);
        assert_eq!(s.expects.len(), 4);
        assert_eq!(s.ticks, 200);
    }

    #[test]
    fn runs_and_checks_expectations() {
        let s = parse(PILE).unwrap_or_else(|e| panic!("{e}"));
        let mut r = Runner::new(s, mats(), 1).unwrap_or_else(|e| panic!("{e}"));
        let sum = r.run_to_end();
        assert!(sum.passed, "{:#?}", sum.checks);
        assert_eq!(sum.checks.len(), 4);
    }

    #[test]
    fn same_scene_same_hash() {
        let run = |threads| {
            let s = parse(PILE).unwrap_or_else(|e| panic!("{e}"));
            Runner::new(s, mats(), threads).map(|mut r| r.run_to_end().hash).ok()
        };
        assert_eq!(run(1), run(3));
    }

    #[test]
    fn reports_line_numbers() {
        let bad = "[scene]\nsize = 10 10\n[build]\nfill 0 0 10\n";
        assert_eq!(parse(bad).map(|_| ()).map_err(|e| e.line), Err(4));
        let unknown = "[scene]\n[legend]\nz zzz\n[grid]\nz\n";
        let s = parse(unknown).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(build(&s, &mats(), 1).map(|_| ()).map_err(|e| e.line), Err(5));
    }

    #[test]
    fn replay_round_trips_and_reproduces() {
        let s = parse(PILE).unwrap_or_else(|e| panic!("{e}"));
        let mut r = Runner::new(s, mats(), 1).unwrap_or_else(|e| panic!("{e}"));
        let mut inputs = std::collections::BTreeMap::new();
        inputs.insert(0, fe_engine::PlayerInput::default());
        r.step_with(
            &inputs,
            &[fe_engine::Command::Explode { x: 60, y: 70, radius: 8, power: 150 }],
        );
        let hash = r.run_to_end().hash;
        let replay = Replay {
            format: Replay::FORMAT,
            scene: PILE.into(),
            scene_dir: ".".into(),
            frames: r.recorded(),
            hashes: vec![],
        };
        let back = Replay::from_json(&replay.to_json()).unwrap_or_else(|e| panic!("{e}"));
        let mut r2 = back.runner(mats(), 2).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(r2.run_to_end().hash, hash);
    }

    #[test]
    fn unreachable_and_skipped_expectations_are_reported() {
        let text = "[scene]\nsize = 64 64\nticks = 10\n[events]\nat 5: run 10\n[expect]\nafter 0: bodies == 99\nafter 8: bodies == 99\nafter 50: bodies == 99\n";
        let s = parse(text).unwrap_or_else(|e| panic!("{e}"));
        let mut r = Runner::new(s, mats(), 1).unwrap_or_else(|e| panic!("{e}"));
        let sum = r.run_to_end();
        assert!(!sum.passed);
        let mut lines: Vec<usize> = sum.checks.iter().map(|c| c.line).collect();
        lines.sort_unstable();
        assert_eq!(lines, vec![7, 8, 9], "{:#?}", sum.checks);
        assert!(sum.checks.iter().all(|c| !c.passed));
        assert!(sum.checks.iter().any(|c| c.detail.contains("never reached")));
    }

    #[test]
    fn commands_applied_between_ticks_are_replayed() {
        let s = parse(PILE).unwrap_or_else(|e| panic!("{e}"));
        let mut r = Runner::new(s, mats(), 1).unwrap_or_else(|e| panic!("{e}"));
        for _ in 0..5 {
            r.step();
        }
        let sand = r.sim.material("sand").unwrap_or_default();
        r.apply_command(&fe_engine::Command::Paint {
            x: 64,
            y: 20,
            radius: 4,
            material: sand,
            replace: false,
        });
        for _ in 0..10 {
            r.step();
        }
        let hash = r.sim.hash();
        let replay = Replay {
            format: Replay::FORMAT,
            scene: PILE.into(),
            scene_dir: ".".into(),
            frames: r.recorded(),
            hashes: vec![],
        };
        let back = Replay::from_json(&replay.to_json()).unwrap_or_else(|e| panic!("{e}"));
        let mut r2 = back.runner(mats(), 1).unwrap_or_else(|e| panic!("{e}"));
        for _ in 0..15 {
            r2.step();
        }
        assert_eq!(r2.sim.hash(), hash);
    }

    #[test]
    fn last_tick_does_not_overflow() {
        let s = parse("[scene]\n[events]\nat 18446744073709551615: dig 0 0 1\n");
        assert!(s.is_ok());
    }

    fn siege(threads: usize) -> Runner {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../scenes/games/siege/siege.scene");
        let scene = load(&path).unwrap_or_else(|e| panic!("{e}"));
        Runner::new(scene, mats(), threads).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Plays the siege game: the left player builds, the right player tries to
    /// shoot during the build phase (blocked), then shells the left core.
    fn play_siege(r: &mut Runner, ticks: u64) {
        use fe_engine::{Command, PlayerInput, tools};
        let stone = r.sim.material("stone").unwrap_or_default().0;
        for t in r.sim.tick()..ticks {
            let mut inputs = std::collections::BTreeMap::new();
            let mut cmds = Vec::new();
            if t == 0 {
                cmds.push(Command::AddPlayer { id: 0, x: 130, y: 280 });
                cmds.push(Command::AddPlayer { id: 1, x: 830, y: 280 });
            }
            if t < 400 {
                let aim_x = ((t / 20) % 9) as i16 * 4 - 16;
                let paint = PlayerInput {
                    tool: tools::PAINT,
                    param: stone,
                    aim_x,
                    aim_y: -30,
                    use_tool: true,
                    ..Default::default()
                };
                inputs.insert(0, paint);
            }
            let shoot = PlayerInput {
                tool: tools::EXPLODE,
                param: 20,
                aim_x: -712 + ((t / 20) % 5) as i16 * 6,
                aim_y: -6 + ((t / 100) % 3) as i16 * 6,
                use_tool: t % 20 < 10,
                ..Default::default()
            };
            inputs.insert(1, shoot);
            r.step_with(&inputs, &cmds);
        }
    }

    #[test]
    fn siege_game_plays_by_its_rules_and_is_deterministic() {
        use fe_engine::tools;
        let mut a = siege(1);
        assert_ne!(a.mods_hash(), 0);
        let crystal = a.sim.material("crystal").unwrap_or_default();
        let core = fe_core::IRect::new(106, 264, 154, 300);
        let start = a.sim.world.count(core, crystal);
        assert!(start > 0);
        play_siege(&mut a, 600);
        assert!(a.sim.status.starts_with("BUILD"), "{}", a.sim.status);
        assert_eq!(a.sim.tool_mask(0), tools::bit(tools::PAINT));
        assert_eq!(a.sim.tool_zone(1), Some(fe_core::IRect::new(520, 40, 944, 300)));
        assert_eq!(a.sim.world.count(core, crystal), start, "explode must be off while building");
        assert!(!a.sim.status.contains("BUDGET 5000 |"), "{}", a.sim.status);

        play_siege(&mut a, 1900);
        assert!(a.sim.status.starts_with("BATTLE"), "{}", a.sim.status);
        assert_eq!(a.sim.tool_mask(1), tools::bit(tools::EXPLODE) | tools::bit(tools::CRATE));
        assert_eq!(a.sim.tool_zone(1), None);

        play_siege(&mut a, 2400);
        assert_eq!(a.sim.world.count(core, crystal), 0);
        assert!(a.sim.status.starts_with("RIGHT WINS"), "{}", a.sim.status);
        assert_eq!(a.sim.tool_mask(0), 0, "tools are off once the game is over");

        let mut b = siege(3);
        play_siege(&mut b, 2400);
        assert_eq!(a.sim.hash(), b.sim.hash(), "thread count changed the result");
        assert_eq!(a.sim.status, b.sim.status);
    }

    #[test]
    fn siege_script_continues_after_a_snapshot() {
        let mut a = siege(1);
        play_siege(&mut a, 300);
        let snap = a.sim.snapshot();
        let mut b = siege(2);
        let sim =
            fe_engine::Simulation::restore(&snap, mats(), 2).unwrap_or_else(|e| panic!("{e}"));
        b.replace_sim(sim).unwrap_or_else(|e| panic!("{e}"));
        play_siege(&mut a, 420);
        play_siege(&mut b, 420);
        assert_eq!(a.sim.hash(), b.sim.hash());
        assert_eq!(a.sim.status, b.sim.status);
    }

    #[test]
    fn scripted_scene_needs_its_script_loaded() {
        let s = parse("[scene]\nscript = nothing.wasm\n").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(s.script, Some(("nothing.wasm".into(), 2)));
        assert!(Runner::new(s.clone(), mats(), 1).is_err());
        let mut s2 = s;
        assert!(attach_script(&mut s2, std::path::Path::new(".")).is_err());
    }

    proptest! {
        #[test]
        fn parser_never_panics(text in "\\PC{0,400}") {
            let _ = parse(&text);
        }

        #[test]
        fn parser_never_panics_on_scene_like_text(lines in proptest::collection::vec(
            prop_oneof![
                Just("[scene]".to_string()), Just("[build]".to_string()), Just("[events]".to_string()),
                Just("[legend]".to_string()), Just("[grid 1 2]".to_string()), Just("[expect]".to_string()),
                Just("[camera]".to_string()), "[a-z ]{0,12}", "[0-9 =:.a-z]{0,30}",
            ], 0..30)) {
            let _ = parse(&lines.join("\n"));
        }
    }
}
