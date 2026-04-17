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
//! Comments start with `//` everywhere except inside `[grid]` blocks.
#![forbid(unsafe_code)]

mod build;
mod parse;
mod run;

pub use build::{build, load_materials};
pub use parse::{
    Camera, CameraKey, Check, Cmp, Expect, InputSpec, LegendEntry, Op, ParamSpec, Scene,
    SceneError, Setup, Timed, When, parse,
};
pub use run::{CheckResult, Frame, Replay, RunSummary, Runner};

/// Reads and parses a scene file.
pub fn load(path: &std::path::Path) -> Result<Scene, SceneError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| SceneError::new(0, format!("cannot read {}: {e}", path.display())))?;
    parse(&text)
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
            frames: r.recorded().to_vec(),
            hashes: vec![],
        };
        let back = Replay::from_json(&replay.to_json()).unwrap_or_else(|e| panic!("{e}"));
        let mut r2 = back.runner(mats(), 2).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(r2.run_to_end().hash, hash);
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
