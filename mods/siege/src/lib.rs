//! Siege builder, a falling-engine game mod.
//!
//! Two teams: even player ids hold the left fort, odd ids the right one.
//!
//! - BUILD (30 s): only the paint tool works, aimed inside your own half. Each
//!   team has a budget of painted cells. The script counts the cells in the
//!   team's build zone and turns the team's paint tool off once it is spent.
//! - BATTLE: the explode and crate tools work and paint is off. Each fort has
//!   a core of crystal cells. A team loses when its core count reaches zero.
//!   If both cores survive the time limit, the larger share of core left wins.
//! - OVER: every tool is off and the status line shows the winner.
//!
//! The geometry below must match `scenes/games/siege/siege.scene`.
//!
//! The module is `no_std` and uses no floats: the engine rejects modules with
//! float instructions. All state lives in atomics, which compile to plain loads
//! and stores on single-threaded wasm and need no `unsafe`.
#![no_std]

use core::sync::atomic::{AtomicI32, Ordering::Relaxed};

#[link(wasm_import_module = "fe")]
unsafe extern "C" {
    safe fn material_id(ptr: *const u8, len: i32) -> i32;
    safe fn count(x0: i32, y0: i32, x1: i32, y1: i32, mat: i32) -> i32;
    safe fn set_tool_mask(player: i32, mask: i32);
    safe fn set_tool_zone(player: i32, x0: i32, y0: i32, x1: i32, y1: i32);
    safe fn clear_tool_zone(player: i32);
    safe fn set_status(ptr: *const u8, len: i32);
    safe fn log(ptr: *const u8, len: i32);
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    core::arch::wasm32::unreachable()
}

const TICKS_PER_SECOND: i32 = 60;
const BUILD_TICKS: i32 = 30 * TICKS_PER_SECOND;
const BATTLE_TICKS: i32 = 120 * TICKS_PER_SECOND;
/// Cells each team may add during the build phase.
const BUDGET: i32 = 5000;
/// Player ids whose tools the script manages.
const MAX_PLAYERS: i32 = 16;
/// Ticks between budget recounts; the two teams are counted on different ticks.
const RECOUNT: i32 = 10;

const TOOL_PAINT: i32 = 1 << 2;
const TOOL_EXPLODE: i32 = 1 << 3;
const TOOL_CRATE: i32 = 1 << 6;

const PHASE_BUILD: i32 = 0;
const PHASE_BATTLE: i32 = 1;
const PHASE_OVER: i32 = 2;

/// A rectangle, minimum inclusive and maximum exclusive.
#[derive(Clone, Copy)]
struct Rect(i32, i32, i32, i32);

/// Where each team may paint during the build phase: its half, away from the middle.
const ZONES: [Rect; 2] = [Rect(16, 40, 440, 300), Rect(520, 40, 944, 300)];
/// Where each team's crystal core is counted.
const CORES: [Rect; 2] = [Rect(106, 264, 154, 300), Rect(806, 264, 854, 300)];

static PHASE: AtomicI32 = AtomicI32::new(PHASE_BUILD);
static TIME: AtomicI32 = AtomicI32::new(0);
static CRYSTAL: AtomicI32 = AtomicI32::new(-1);
static WINNER: AtomicI32 = AtomicI32::new(-1);
static BASE: [AtomicI32; 2] = [AtomicI32::new(0), AtomicI32::new(0)];
static SPENT: [AtomicI32; 2] = [AtomicI32::new(0), AtomicI32::new(0)];
static CORE: [AtomicI32; 2] = [AtomicI32::new(0), AtomicI32::new(0)];
static CORE_START: [AtomicI32; 2] = [AtomicI32::new(0), AtomicI32::new(0)];

fn filled(r: Rect) -> i32 {
    let area = (r.2 - r.0) * (r.3 - r.1);
    area - count(r.0, r.1, r.2, r.3, 0)
}

fn core_cells(team: usize) -> i32 {
    let r = CORES[team];
    count(r.0, r.1, r.2, r.3, CRYSTAL.load(Relaxed))
}

fn say(text: &[u8]) {
    log(text.as_ptr(), text.len() as i32);
}

/// Sets every managed player's tools for the current phase.
fn apply_tools() {
    let phase = PHASE.load(Relaxed);
    for id in 0..MAX_PLAYERS {
        let team = (id % 2) as usize;
        match phase {
            PHASE_BUILD => {
                let open = SPENT[team].load(Relaxed) < BUDGET;
                set_tool_mask(id, if open { TOOL_PAINT } else { 0 });
                let z = ZONES[team];
                set_tool_zone(id, z.0, z.1, z.2, z.3);
            }
            PHASE_BATTLE => {
                set_tool_mask(id, TOOL_EXPLODE | TOOL_CRATE);
                clear_tool_zone(id);
            }
            _ => {
                set_tool_mask(id, 0);
                clear_tool_zone(id);
            }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn on_init() {
    let name = b"crystal";
    CRYSTAL.store(material_id(name.as_ptr(), name.len() as i32), Relaxed);
    for team in 0..2 {
        BASE[team].store(filled(ZONES[team]), Relaxed);
        let c = core_cells(team);
        CORE[team].store(c, Relaxed);
        CORE_START[team].store(c, Relaxed);
    }
    apply_tools();
    show();
    say(b"siege: build phase started");
}

#[unsafe(no_mangle)]
pub extern "C" fn on_tick() {
    let t = TIME.load(Relaxed) + 1;
    TIME.store(t, Relaxed);
    match PHASE.load(Relaxed) {
        PHASE_BUILD => {
            for team in 0..2 {
                if t % RECOUNT == team as i32 * (RECOUNT / 2) {
                    let spent = filled(ZONES[team]) - BASE[team].load(Relaxed);
                    SPENT[team].store(spent.max(0), Relaxed);
                }
            }
            if t >= BUILD_TICKS {
                PHASE.store(PHASE_BATTLE, Relaxed);
                TIME.store(0, Relaxed);
                say(b"siege: battle phase started");
            }
        }
        PHASE_BATTLE => {
            for team in 0..2 {
                CORE[team].store(core_cells(team), Relaxed);
            }
            let (l, r) = (CORE[0].load(Relaxed), CORE[1].load(Relaxed));
            let winner = if l == 0 && r == 0 {
                2
            } else if l == 0 {
                1
            } else if r == 0 {
                0
            } else if t >= BATTLE_TICKS {
                // Time is up: the larger share of core left wins.
                let ls = i64::from(l) * i64::from(CORE_START[1].load(Relaxed));
                let rs = i64::from(r) * i64::from(CORE_START[0].load(Relaxed));
                if ls > rs {
                    0
                } else if rs > ls {
                    1
                } else {
                    2
                }
            } else {
                -1
            };
            if winner >= 0 {
                WINNER.store(winner, Relaxed);
                PHASE.store(PHASE_OVER, Relaxed);
                say(b"siege: game over");
            }
        }
        _ => {}
    }
    apply_tools();
    show();
}

/// A small text buffer on the stack.
struct Line {
    buf: [u8; 160],
    len: usize,
}

impl Line {
    fn text(&mut self, s: &[u8]) {
        for &b in s {
            if self.len < self.buf.len() {
                self.buf[self.len] = b;
                self.len += 1;
            }
        }
    }

    fn num(&mut self, mut v: i32) {
        if v < 0 {
            self.text(b"-");
            v = -v;
        }
        let mut digits = [0u8; 10];
        let mut n = 0;
        loop {
            digits[n] = b'0' + (v % 10) as u8;
            n += 1;
            v /= 10;
            if v == 0 {
                break;
            }
        }
        while n > 0 {
            n -= 1;
            self.text(&digits[n..=n]);
        }
    }

    fn clock(&mut self, ticks: i32) {
        let secs = (ticks.max(0) + TICKS_PER_SECOND - 1) / TICKS_PER_SECOND;
        self.num(secs / 60);
        self.text(b":");
        if secs % 60 < 10 {
            self.text(b"0");
        }
        self.num(secs % 60);
    }
}

/// Writes the status line: phase, time left, core counts, budgets, and the winner.
fn show() {
    let mut l = Line { buf: [0; 160], len: 0 };
    let phase = PHASE.load(Relaxed);
    let t = TIME.load(Relaxed);
    match phase {
        PHASE_BUILD => {
            l.text(b"BUILD ");
            l.clock(BUILD_TICKS - t);
        }
        PHASE_BATTLE => {
            l.text(b"BATTLE ");
            l.clock(BATTLE_TICKS - t);
        }
        _ => match WINNER.load(Relaxed) {
            0 => l.text(b"LEFT WINS"),
            1 => l.text(b"RIGHT WINS"),
            _ => l.text(b"DRAW"),
        },
    }
    for (team, name) in [(0usize, &b"LEFT"[..]), (1, &b"RIGHT"[..])] {
        l.text(b" | ");
        l.text(name);
        l.text(b" CORE ");
        l.num(CORE[team].load(Relaxed));
        l.text(b"/");
        l.num(CORE_START[team].load(Relaxed));
        if phase == PHASE_BUILD {
            l.text(b" BUDGET ");
            l.num((BUDGET - SPENT[team].load(Relaxed)).max(0));
        }
    }
    set_status(l.buf.as_ptr(), l.len as i32);
}
