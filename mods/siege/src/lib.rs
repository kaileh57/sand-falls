//! Siege builder, a falling-engine game mod.
//!
//! Two teams: even player ids hold the left fort, odd ids the right one.
//!
//! - BUILD (12 s): each team has a budget of cells. The team's masons, run by
//!   this script, raise the team's blueprint (towers, walls, cisterns, powder
//!   stores) course by course from the ground up and pay for it from the
//!   budget. Players may paint inside their own half from what is left. The
//!   script counts the cells in each build zone and turns a team's paint tool
//!   off, and stops its masons, once the budget is spent.
//! - BATTLE (up to 48 s): the explode and crate tools work and paint is off.
//!   Each fort has a core of crystal cells, warded for the first 20 s: the
//!   script regrows a warded core row by row. Once the wards are down, a core
//!   that loses half its crystal shatters: the script blows it apart and then
//!   blows out the base of that team's keep, and the team loses. If both cores
//!   last the time limit, the larger share of core left wins, and if the
//!   shares are equal, the larger share of fort left standing: the script
//!   counts each team's brick, stone, metal, glass and wood cells in its build
//!   zone and shows the share on the status line as WALLS.
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
    safe fn get_material(x: i32, y: i32) -> i32;
    safe fn fill(x0: i32, y0: i32, x1: i32, y1: i32, mat: i32);
    safe fn explode(x: i32, y: i32, r: i32, power: i32);
    safe fn player_count() -> i32;
    safe fn player_id(i: i32) -> i32;
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
const BUILD_TICKS: i32 = 12 * TICKS_PER_SECOND;
/// Battle time limit: the game ends at tick 3580 if no core shatters first.
const BATTLE_TICKS: i32 = 2861;
/// How long the cores are warded once the battle starts (to tick 1919).
const WARD_TICKS: i32 = 20 * TICKS_PER_SECOND;
/// Cells each team may add during the build phase, masons included.
const BUDGET: i32 = 12_500;
/// The masons lay one course of the blueprint at a time, from the bottom row
/// up to the top row, between these two ticks of the build phase.
const MASON_START: i32 = 20;
const MASON_END: i32 = 660;
const MASON_BOTTOM: i32 = 299;
const MASON_TOP: i32 = -66;
/// Player ids whose tools the script manages.
const MAX_PLAYERS: i32 = 16;
/// Ticks between budget recounts; the two teams are counted on different ticks.
const RECOUNT: i32 = 10;
/// A shattered core takes its keep with it this many ticks later.
const KEEP_DELAY: i32 = 30;

/// Materials the script uses. The first five count as fort for the tiebreak.
const MATERIALS: [&[u8]; 11] = [
    b"brick", b"stone", b"metal", b"glass", b"wood", b"tnt", b"water", b"oil", b"torch", b"crystal",
    b"bedrock",
];
const BRICK: u8 = 0;
const METAL: u8 = 2;
const GLASS: u8 = 3;
const WOOD: u8 = 4;
const TNT: u8 = 5;
const WATER: u8 = 6;
const OIL: u8 = 7;
const TORCH: u8 = 8;
const CRYSTAL: usize = 9;
const BEDROCK: usize = 10;
const FORT_MATERIALS: usize = 5;

const TOOL_PAINT: i32 = 1 << 2;
const TOOL_EXPLODE: i32 = 1 << 3;
const TOOL_CRATE: i32 = 1 << 6;

const PHASE_BUILD: i32 = 0;
const PHASE_BATTLE: i32 = 1;
const PHASE_OVER: i32 = 2;

/// A rectangle, minimum inclusive and maximum exclusive.
#[derive(Clone, Copy)]
struct Rect(i32, i32, i32, i32);

/// One piece of a blueprint: a material over a rectangle. Later pieces are
/// laid over earlier ones in the same course.
#[derive(Clone, Copy)]
struct Piece(u8, i32, i32, i32, i32);

/// The left team's blueprint: an inner bastion, a raised front wall, a front
/// tower with a powder room in its shaft and a metal cistern of water on top,
/// and a powder store on brick stilts out in front.
const LEFT_PLAN: [Piece; 17] = [
    Piece(BRICK, 158, 266, 198, 300),
    Piece(BRICK, 200, 170, 214, 300),
    Piece(BRICK, 200, 162, 203, 170),
    Piece(BRICK, 205, 162, 209, 170),
    Piece(BRICK, 211, 162, 214, 170),
    Piece(BRICK, 250, -20, 278, 300),
    Piece(GLASS, 259, 20, 269, 34),
    Piece(GLASS, 259, 80, 269, 94),
    Piece(GLASS, 259, 140, 269, 154),
    Piece(TNT, 254, 208, 274, 226),
    Piece(METAL, 214, -60, 262, -20),
    Piece(WATER, 218, -56, 258, -24),
    Piece(TORCH, 236, -66, 240, -60),
    Piece(BRICK, 298, 262, 302, 300),
    Piece(BRICK, 332, 262, 336, 300),
    Piece(BRICK, 298, 242, 336, 262),
    Piece(TNT, 306, 248, 328, 258),
];

/// The right team's blueprint: a timber palisade frame by the river, a front
/// tower with a TNT store in its foot and a cistern on top, an oil store and a
/// curtain wall in front of the core.
const RIGHT_PLAN: [Piece; 19] = [
    Piece(WOOD, 645, 258, 689, 262),
    Piece(WOOD, 645, 272, 689, 276),
    Piece(WOOD, 645, 286, 689, 290),
    Piece(WOOD, 645, 258, 649, 300),
    Piece(WOOD, 665, 258, 669, 300),
    Piece(WOOD, 685, 258, 689, 300),
    Piece(BRICK, 722, -20, 750, 300),
    Piece(GLASS, 731, 20, 741, 34),
    Piece(GLASS, 731, 80, 741, 94),
    Piece(GLASS, 731, 140, 741, 154),
    Piece(TNT, 728, 270, 744, 296),
    Piece(METAL, 736, -60, 784, -20),
    Piece(WATER, 740, -56, 780, -24),
    Piece(TORCH, 758, -66, 762, -60),
    Piece(WOOD, 754, 276, 782, 300),
    Piece(OIL, 756, 278, 780, 298),
    Piece(BRICK, 786, 170, 796, 300),
    Piece(BRICK, 786, 162, 789, 170),
    Piece(BRICK, 792, 162, 796, 170),
];

/// Where each team may paint during the build phase: its half, away from the middle.
const ZONES: [Rect; 2] = [Rect(16, 40, 440, 300), Rect(520, 40, 944, 300)];
/// Where each team's crystal core is counted.
const CORES: [Rect; 2] = [Rect(106, 264, 154, 300), Rect(806, 264, 854, 300)];
/// The crystal of each core, which a ward regrows.
const CRYSTALS: [Rect; 2] = [Rect(122, 280, 138, 296), Rect(822, 280, 838, 296)];
/// Where the script blows out each team's keep when its core shatters.
const KEEPS: [(i32, i32); 2] = [(80, 226), (876, 226)];

static PHASE: AtomicI32 = AtomicI32::new(PHASE_BUILD);
static TIME: AtomicI32 = AtomicI32::new(0);
static WINNER: AtomicI32 = AtomicI32::new(-1);
static COURSES: AtomicI32 = AtomicI32::new(0);
static MEND_ROW: AtomicI32 = AtomicI32::new(0);
static IDS: [AtomicI32; 11] = [const { AtomicI32::new(-1) }; 11];
static BASE: [AtomicI32; 2] = [AtomicI32::new(0), AtomicI32::new(0)];
static SPENT: [AtomicI32; 2] = [AtomicI32::new(0), AtomicI32::new(0)];
static CORE: [AtomicI32; 2] = [AtomicI32::new(0), AtomicI32::new(0)];
static CORE_START: [AtomicI32; 2] = [AtomicI32::new(0), AtomicI32::new(0)];
/// Battle tick at which each core shattered, or 0.
static SHATTERED: [AtomicI32; 2] = [AtomicI32::new(0), AtomicI32::new(0)];
static WALLS: [AtomicI32; 2] = [AtomicI32::new(0), AtomicI32::new(0)];
static WALLS_START: [AtomicI32; 2] = [AtomicI32::new(0), AtomicI32::new(0)];

fn id(m: usize) -> i32 {
    IDS[m].load(Relaxed)
}

fn filled(r: Rect) -> i32 {
    let area = (r.2 - r.0) * (r.3 - r.1);
    area - count(r.0, r.1, r.2, r.3, 0)
}

fn core_cells(team: usize) -> i32 {
    let r = CORES[team];
    count(r.0, r.1, r.2, r.3, id(CRYSTAL))
}

/// Fort cells standing in a team's build zone.
fn fort_cells(team: usize) -> i32 {
    let r = ZONES[team];
    (0..FORT_MATERIALS).map(|m| count(r.0, r.1, r.2, r.3, id(m))).sum()
}

/// Share of the fort left, in percent.
fn walls_pct(team: usize) -> i32 {
    let start = WALLS_START[team].load(Relaxed).max(1);
    (i64::from(WALLS[team].load(Relaxed)) * 100 / i64::from(start)) as i32
}

/// Players on a team.
fn team_size(team: usize) -> i32 {
    (0..player_count()).filter(|&i| player_id(i).rem_euclid(2) as usize == team).count() as i32
}

fn say(text: &[u8]) {
    log(text.as_ptr(), text.len() as i32);
}

/// Lays one course of each team's blueprint, if the team can still pay.
fn lay_course(y: i32) {
    for team in 0..2 {
        if SPENT[team].load(Relaxed) >= BUDGET {
            continue;
        }
        let plan: &[Piece] = if team == 0 { &LEFT_PLAN } else { &RIGHT_PLAN };
        for p in plan {
            if p.2 <= y && y < p.4 {
                fill(p.1, y, p.3, y + 1, id(usize::from(p.0)));
            }
        }
    }
}

/// Runs the masons: courses go up at a steady rate from the bottom row.
fn masons(t: i32) {
    let total = MASON_BOTTOM - MASON_TOP + 1;
    let due = ((t - MASON_START).clamp(0, MASON_END - MASON_START) * total)
        / (MASON_END - MASON_START);
    let mut done = COURSES.load(Relaxed);
    while done < due {
        lay_course(MASON_BOTTOM - done);
        done += 1;
    }
    COURSES.store(done, Relaxed);
}

/// True for the solid building materials a falling body can be made of; a
/// ward never overwrites them, so it cannot cut into a body.
fn is_masonry(m: i32) -> bool {
    m == id(TNT as usize) || m == id(BEDROCK) || (0..FORT_MATERIALS).any(|k| m == id(k))
}

/// Regrows one row of each warded core that has lost crystal, over whatever
/// loose debris, gas or water has taken its place.
fn mend() {
    let row = MEND_ROW.load(Relaxed);
    MEND_ROW.store((row + 1) % 16, Relaxed);
    for team in 0..2 {
        if core_cells(team) >= CORE_START[team].load(Relaxed) {
            continue;
        }
        let c = CRYSTALS[team];
        let y = c.1 + row % (c.3 - c.1);
        for x in c.0..c.2 {
            let m = get_material(x, y);
            if m != id(CRYSTAL) && !is_masonry(m) {
                fill(x, y, x + 1, y + 1, id(CRYSTAL));
            }
        }
    }
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
    for (slot, name) in IDS.iter().zip(MATERIALS) {
        slot.store(material_id(name.as_ptr(), name.len() as i32), Relaxed);
    }
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

/// Shatters a core that has lost half its crystal once the wards are down,
/// and blows out its keep a moment later.
fn shatter(t: i32) {
    for team in 0..2 {
        let at = SHATTERED[team].load(Relaxed);
        if at == 0 {
            let start = CORE_START[team].load(Relaxed);
            if PHASE.load(Relaxed) == PHASE_BATTLE
                && t >= WARD_TICKS
                && CORE[team].load(Relaxed) * 2 < start
            {
                SHATTERED[team].store(t, Relaxed);
                let c = CRYSTALS[team];
                explode((c.0 + c.2) / 2, (c.1 + c.3) / 2, 30, 255);
                say(b"siege: a core shattered");
            }
        } else if t == at + KEEP_DELAY {
            let (x, y) = KEEPS[team];
            explode(x, y, 40, 255);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn on_tick() {
    let t = TIME.load(Relaxed) + 1;
    TIME.store(t, Relaxed);
    match PHASE.load(Relaxed) {
        PHASE_BUILD => {
            masons(t);
            for team in 0..2 {
                if t % RECOUNT == team as i32 * (RECOUNT / 2) {
                    let spent = filled(ZONES[team]) - BASE[team].load(Relaxed);
                    SPENT[team].store(spent.max(0), Relaxed);
                }
            }
            if t >= BUILD_TICKS {
                for team in 0..2 {
                    let w = fort_cells(team);
                    WALLS[team].store(w, Relaxed);
                    WALLS_START[team].store(w, Relaxed);
                }
                PHASE.store(PHASE_BATTLE, Relaxed);
                TIME.store(0, Relaxed);
                say(b"siege: battle phase started");
            }
        }
        PHASE_BATTLE => {
            if t < WARD_TICKS {
                mend();
            } else if t == WARD_TICKS {
                say(b"siege: the wards are down");
            }
            for team in 0..2 {
                CORE[team].store(core_cells(team), Relaxed);
                if t % RECOUNT == team as i32 * (RECOUNT / 2) {
                    WALLS[team].store(fort_cells(team), Relaxed);
                }
            }
            shatter(t);
            let (l, r) = (CORE[0].load(Relaxed), CORE[1].load(Relaxed));
            let gone = |team: usize, n: i32| n == 0 || SHATTERED[team].load(Relaxed) != 0;
            let (l_gone, r_gone) = (gone(0, l), gone(1, r));
            let winner = if t < WARD_TICKS {
                -1
            } else if l_gone && r_gone {
                2
            } else if l_gone {
                1
            } else if r_gone {
                0
            } else if t >= BATTLE_TICKS {
                // Time is up: the larger share of core left wins.
                let ls = i64::from(l) * i64::from(CORE_START[1].load(Relaxed));
                let rs = i64::from(r) * i64::from(CORE_START[0].load(Relaxed));
                for team in 0..2 {
                    WALLS[team].store(fort_cells(team), Relaxed);
                }
                // Equal shares of core: the larger share of fort left wins.
                let (lw, rw) = (walls_pct(0), walls_pct(1));
                if ls > rs || (ls == rs && lw > rw) {
                    0
                } else if rs > ls || (ls == rs && rw > lw) {
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
        _ => {
            // The battle clock keeps running so a shattered keep still falls.
            shatter(t);
        }
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

/// Writes the status line: phase, time left, wards, team sizes, core counts,
/// budgets or walls, and the winner.
fn show() {
    let mut l = Line { buf: [0; 160], len: 0 };
    let phase = PHASE.load(Relaxed);
    let t = TIME.load(Relaxed);
    match phase {
        PHASE_BUILD => {
            l.text(b"BUILD ");
            l.clock(BUILD_TICKS - t);
            l.text(b" | MASONS ");
            let total = MASON_BOTTOM - MASON_TOP + 1;
            l.num(COURSES.load(Relaxed) * 100 / total);
            l.text(b"%");
        }
        PHASE_BATTLE => {
            l.text(b"BATTLE ");
            l.clock(BATTLE_TICKS - t);
            if t < WARD_TICKS {
                l.text(b" | WARDS UP ");
                l.clock(WARD_TICKS - t);
            } else {
                l.text(b" | WARDS DOWN");
            }
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
        l.text(b" ");
        l.num(team_size(team));
        l.text(b"P CORE ");
        if SHATTERED[team].load(Relaxed) != 0 {
            l.text(b"SHATTERED");
        } else {
            l.num(CORE[team].load(Relaxed));
            l.text(b"/");
            l.num(CORE_START[team].load(Relaxed));
        }
        if phase == PHASE_BUILD {
            l.text(b" BUDGET ");
            l.num((BUDGET - SPENT[team].load(Relaxed)).max(0));
        } else {
            l.text(b" WALLS ");
            l.num(walls_pct(team));
            l.text(b"%");
        }
    }
    set_status(l.buf.as_ptr(), l.len as i32);
}
