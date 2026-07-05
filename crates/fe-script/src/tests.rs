use super::*;
use fe_core::IRect;
use std::collections::BTreeMap;
use std::sync::Arc;

fn sim() -> Simulation {
    let mats = Arc::new(fe_materials::base_set());
    let mut s = Simulation::new(mats, fe_engine::SimConfig { seed: 5, ..Default::default() });
    s.load_rect(IRect::new(0, 0, 128, 128));
    s
}

fn wasm(text: &str) -> Vec<u8> {
    wat::parse_str(text).unwrap_or_else(|e| panic!("{e}"))
}

fn load(text: &str, s: &Simulation) -> Script {
    Script::load(&wasm(text), s).unwrap_or_else(|e| panic!("{e}"))
}

/// Runs a tick the way the scene runner does: script first, then the step.
fn step(script: &mut Script, s: &mut Simulation) {
    let cmds = script.tick(s);
    s.step(&BTreeMap::new(), &cmds);
}

/// Counts ticks in a global and in memory, paints sand every tick, and shows the count.
const COUNTER: &str = r#"
(module
  (import "fe" "paint" (func $paint (param i32 i32 i32 i32)))
  (import "fe" "material_id" (func $material_id (param i32 i32) (result i32)))
  (import "fe" "set_status" (func $set_status (param i32 i32)))
  (import "fe" "set_tool_mask" (func $set_tool_mask (param i32 i32)))
  (memory (export "memory") 1)
  (global $n (export "n") (mut i32) (i32.const 0))
  (data (i32.const 0) "sand")
  (data (i32.const 16) "count ?")
  (func (export "on_init")
    (i32.store (i32.const 32) (call $material_id (i32.const 0) (i32.const 4)))
    (call $set_tool_mask (i32.const 3) (i32.const 4)))
  (func (export "on_tick")
    (global.set $n (i32.add (global.get $n) (i32.const 1)))
    (i32.store (i32.const 64) (i32.add (i32.load (i32.const 64)) (i32.const 2)))
    (i32.store8 (i32.const 22) (i32.add (i32.const 48) (i32.rem_u (global.get $n) (i32.const 10))))
    (call $set_status (i32.const 16) (i32.const 7))
    (call $paint (i32.const 40) (i32.const 20) (i32.const 2) (i32.load (i32.const 32)))))
"#;

#[test]
fn float_modules_are_rejected() {
    let s = sim();
    let text = r#"(module (func (export "on_tick") (drop (f32.add (f32.const 1) (f32.const 2)))))"#;
    let err = Script::load(&wasm(text), &s).err();
    assert_eq!(err, Some(ScriptError::Floats));
    assert!(ScriptError::Floats.to_string().contains("floating-point"));
    // A float type in a signature is rejected too.
    let sig = r#"(module (func (export "f") (param f64)))"#;
    assert_eq!(Script::load(&wasm(sig), &s).err(), Some(ScriptError::Floats));
}

#[test]
fn modules_that_hide_state_are_rejected() {
    let s = sim();
    let hidden = r#"(module (global $g (mut i32) (i32.const 0))
        (func (export "on_tick") (global.set $g (i32.const 1))))"#;
    assert!(matches!(Script::load(&wasm(hidden), &s), Err(ScriptError::Rules(_))));
    let start = r"(module (func $s) (start $s))";
    assert!(Script::load(&wasm(start), &s).is_err());
    let foreign = r#"(module (import "env" "f" (func)))"#;
    assert!(matches!(Script::load(&wasm(foreign), &s), Err(ScriptError::Rules(_))));
    assert!(matches!(Script::load(b"not wasm", &s), Err(ScriptError::Invalid(_))));
}

#[test]
fn commands_and_game_state_are_applied() {
    let mut s = sim();
    let before = s.hash();
    let mut script = load(COUNTER, &s);
    assert!(script.init(&mut s).is_empty());
    assert_eq!(s.tool_mask(3), 4);
    assert_eq!(s.tool_mask(4), fe_engine::tools::ALL);
    assert!(!s.script_state.is_empty());
    assert_ne!(s.hash(), before, "game state must enter the hash");
    for _ in 0..3 {
        step(&mut script, &mut s);
    }
    assert_eq!(s.status, "count 3");
    let sand = s.material("sand").unwrap_or_default();
    assert!(s.world.count(IRect::new(30, 10, 50, 40), sand) > 0);
    assert!(!script.faulted());
}

#[test]
fn running_out_of_fuel_is_a_deterministic_fault() {
    let text = r#"
    (module
      (import "fe" "dig" (func $dig (param i32 i32 i32)))
      (memory (export "memory") 1)
      (func (export "on_tick")
        (call $dig (i32.const 10) (i32.const 10) (i32.const 3))
        (loop $forever
          (i32.store (i32.const 0) (i32.add (i32.load (i32.const 0)) (i32.const 1)))
          (br $forever))))
    "#;
    let run = || {
        let mut s = sim();
        let mut script = load(text, &s);
        script.init(&mut s);
        let cmds = script.tick(&mut s);
        assert!(cmds.is_empty(), "commands from a faulted call are dropped");
        assert!(script.faulted());
        s.step(&BTreeMap::new(), &cmds);
        step(&mut script, &mut s);
        (s.status.clone(), s.hash(), s.script_state.clone())
    };
    let (status, hash, state) = run();
    assert!(status.contains("on_tick"), "{status}");
    assert_eq!(run(), (status, hash, state));
}

#[test]
fn state_round_trips_through_a_snapshot() {
    let mut a = sim();
    let mut sa = load(COUNTER, &a);
    for c in sa.init(&mut a) {
        a.apply(&c);
    }
    for _ in 0..5 {
        step(&mut sa, &mut a);
    }
    let bytes = a.snapshot();
    let mut b =
        Simulation::restore(&bytes, a.materials().clone(), 1).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(a.hash(), b.hash());
    // A fresh load starts over; restoring from the saved state continues.
    let mut sb = load(COUNTER, &b);
    sb.restore(&b).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(sb.save(), b.script_state);
    for _ in 0..7 {
        step(&mut sa, &mut a);
        step(&mut sb, &mut b);
        assert_eq!(a.hash(), b.hash());
    }
    assert_eq!(a.status, "count 2");
    assert_eq!(b.status, a.status);

    let mut bad = b.script_state.clone();
    bad.truncate(10);
    b.script_state = bad;
    assert!(matches!(sb.restore(&b), Err(ScriptError::State(_))));
}

#[test]
fn mod_hash_is_stable() {
    assert_eq!(mod_hash(&[]), 0);
    let w = wasm(COUNTER);
    assert_eq!(mod_hash(&w), mod_hash(&w.clone()));
    assert_ne!(mod_hash(&w), mod_hash(&w[..w.len() - 1]));
}
