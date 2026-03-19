//! Line-based parser for `.scene` files.

use fe_core::IRect;
use std::collections::BTreeMap;

/// A parse or build error with the line it came from (1-based, 0 if none).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SceneError {
    /// Line number, or 0 when the error is not tied to a line.
    pub line: usize,
    /// What went wrong.
    pub message: String,
}

impl SceneError {
    pub(crate) fn new(line: usize, message: impl Into<String>) -> Self {
        Self { line, message: message.into() }
    }
}

impl std::fmt::Display for SceneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.line == 0 {
            write!(f, "{}", self.message)
        } else {
            write!(f, "line {}: {}", self.line, self.message)
        }
    }
}

impl std::error::Error for SceneError {}

/// Comparison used by expectations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmp {
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
}

impl Cmp {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "==" => Self::Eq,
            "!=" => Self::Ne,
            "<" => Self::Lt,
            "<=" => Self::Le,
            ">" => Self::Gt,
            ">=" => Self::Ge,
            _ => return None,
        })
    }

    /// Applies the comparison.
    pub fn test(self, a: i64, b: i64) -> bool {
        match self {
            Self::Eq => a == b,
            Self::Ne => a != b,
            Self::Lt => a < b,
            Self::Le => a <= b,
            Self::Gt => a > b,
            Self::Ge => a >= b,
        }
    }

    /// The operator as written.
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::Eq => "==",
            Self::Ne => "!=",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
        }
    }
}

/// One player-input setting from a timeline line.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct InputSpec {
    /// Player id.
    pub player: u8,
    /// -1, 0 or 1.
    pub move_x: i8,
    /// Jump held.
    pub jump: bool,
    /// Tool number.
    pub tool: u8,
    /// Tool parameter, or a material name to resolve.
    pub param: ParamSpec,
    /// Aim offset.
    pub aim: (i16, i16),
    /// Tool used this tick.
    pub use_tool: bool,
}

/// A tool parameter: a number, or a material name resolved at build time.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum ParamSpec {
    /// No parameter.
    #[default]
    None,
    /// A number.
    Num(u16),
    /// A material name.
    Material(String),
}

/// A scene operation, with materials still named.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// Fill a rectangle.
    Fill(IRect, String),
    /// Fill a disc, overwriting.
    Circle(i32, i32, i32, String),
    /// Paint a disc into air only.
    Paint(i32, i32, i32, String),
    /// Fill a rectangle, then carve it into one body.
    Body(IRect, String),
    /// Carve the static cells in a rectangle into one body.
    Carve(IRect),
    /// Fill from a noisy surface line down to `bottom`.
    Terrain {
        /// First column.
        x0: i32,
        /// Column after the last.
        x1: i32,
        /// Average surface row.
        base: i32,
        /// Largest offset of the surface from `base`.
        amp: i32,
        /// Columns between random height points.
        scale: i32,
        /// Row after the last filled row.
        bottom: i32,
        /// Fill material.
        material: String,
    },
    /// Randomly place a material into air cells.
    Scatter(IRect, String, u32),
    /// Grow a material on air cells next to solid ones: `floor`, `ceiling`, `walls` or `any`.
    Decorate(IRect, String, u32, Surface),
    /// Set temperature in a rectangle, in degrees C.
    Temp(IRect, i32),
    /// Add a character.
    Player(u8, i32, i32),
    /// Remove a character.
    RemovePlayer(u8),
    /// Pin body `n` to the world.
    Pin(u32, i32, i32, f32),
    /// Join two bodies.
    Joint(u32, u32, i32, i32, fe_engine::JointKind, f32),
    /// Explosion.
    Explode(i32, i32, u8, u8),
    /// Remove cells in a disc.
    Dig(i32, i32, i32),
    /// Change temperature in a disc by engine units.
    Heat(i32, i32, i32, i16),
    /// Push a body, in cells per second.
    Impulse(u32, f32, f32),
    /// Player input for this tick.
    Input(InputSpec),
    /// Simulate ticks during setup.
    Run(u64),
}

/// Which surfaces a decoration grows on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    /// Air with a solid cell below.
    Floor,
    /// Air with a solid cell above.
    Ceiling,
    /// Air with a solid cell to the left or right.
    Walls,
    /// Air next to any solid cell.
    Any,
}

/// When a timeline entry fires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum When {
    /// On one tick.
    At(u64),
    /// On every tick in `start..end`.
    Range(u64, u64),
    /// Every `n` ticks in `start..end`.
    Every(u64, u64, u64),
}

impl When {
    /// Returns true if the entry fires on `tick`.
    pub fn fires(&self, tick: u64) -> bool {
        match *self {
            Self::At(t) => tick == t,
            Self::Range(a, b) => tick >= a && tick < b,
            Self::Every(n, a, b) => tick >= a && tick < b && (tick - a).is_multiple_of(n.max(1)),
        }
    }

    /// Last tick the entry can fire on, plus one.
    pub fn end(&self) -> u64 {
        match *self {
            Self::At(t) => t + 1,
            Self::Range(_, b) | Self::Every(_, _, b) => b,
        }
    }
}

/// A timeline entry.
#[derive(Clone, Debug, PartialEq)]
pub struct Timed {
    /// When it fires.
    pub when: When,
    /// What it does.
    pub op: Op,
    /// Source line.
    pub line: usize,
}

/// A checked condition.
#[derive(Clone, Debug, PartialEq)]
pub enum Check {
    /// Number of cells of a material in a rectangle.
    Count(String, IRect, Cmp, i64),
    /// Whole-state hash.
    Hash(u64),
    /// Body count.
    Bodies(Cmp, i64),
    /// Particles in flight.
    Particles(Cmp, i64),
    /// Chunks with work last tick.
    Active(Cmp, i64),
    /// A character coordinate: player, `true` for x, comparison, value.
    Avatar(u8, bool, Cmp, i64),
}

/// An expectation checked after a tick.
#[derive(Clone, Debug, PartialEq)]
pub struct Expect {
    /// Checked once this many ticks have run.
    pub after: u64,
    /// The condition.
    pub check: Check,
    /// Source line.
    pub line: usize,
}

/// One legend entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegendEntry {
    /// Material name.
    pub material: String,
    /// Connected groups of this character become rigid bodies.
    pub body: bool,
    /// Starting temperature in degrees C, if set.
    pub temp: Option<i32>,
}

/// A camera keyframe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraKey {
    /// Tick.
    pub tick: u64,
    /// View center x in cells.
    pub x: f32,
    /// View center y in cells.
    pub y: f32,
    /// Zoom factor (1 shows `view` cells).
    pub zoom: f32,
}

/// Camera path for rendering.
#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    /// Width and height of the view in cells at zoom 1.
    pub view: (u32, u32),
    /// Output pixels per cell at zoom 1.
    pub scale: u32,
    /// Keyframes in tick order.
    pub keys: Vec<CameraKey>,
    /// `day`, `dusk`, `night`, or `cave`.
    pub sky: String,
    /// Ambient light in linear RGB.
    pub ambient: [f32; 3],
    /// Tone-mapping exposure.
    pub exposure: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            view: (480, 270),
            scale: 2,
            keys: Vec::new(),
            sky: "cave".into(),
            ambient: [0.05, 0.05, 0.06],
            exposure: 1.4,
        }
    }
}

/// A setup step.
#[derive(Clone, Debug, PartialEq)]
pub enum Setup {
    /// ASCII grid placed with its top-left at `(x, y)`.
    Grid {
        /// Left edge.
        x: i32,
        /// Top edge.
        y: i32,
        /// Rows of legend characters.
        rows: Vec<String>,
        /// Source line of the first row.
        line: usize,
    },
    /// Any other operation.
    Op(Op, usize),
}

/// A parsed scene.
#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    /// Name.
    pub name: String,
    /// One-line description.
    pub description: String,
    /// World bounds: chunks covering this rectangle are loaded at start.
    pub bounds: IRect,
    /// World seed.
    pub seed: u64,
    /// `base` or a path to a TOML material file, relative to the scene file.
    pub materials: String,
    /// World generator.
    pub worldgen: fe_engine::WorldGen,
    /// Streaming radius and margin, in chunks.
    pub stream: Option<(i32, i32)>,
    /// Ticks to run; 0 means up to the last timeline entry or expectation.
    pub ticks: u64,
    /// Where networked players appear, if the scene sets it. Several `spawn` lines
    /// give several spawn points; players are dealt to them in turn.
    pub spawn: Option<(i32, i32)>,
    /// Every spawn point, in file order.
    pub spawns: Vec<(i32, i32)>,
    /// Wind from -4 to 4.
    pub wind: i8,
    /// Legend characters.
    pub legend: BTreeMap<char, LegendEntry>,
    /// Setup steps in order.
    pub setup: Vec<Setup>,
    /// Timeline.
    pub timeline: Vec<Timed>,
    /// Camera.
    pub camera: Camera,
    /// Expectations.
    pub expects: Vec<Expect>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Section {
    None,
    Scene,
    Legend,
    Grid,
    Build,
    Events,
    Camera,
    Expect,
}

fn strip_comment(line: &str) -> &str {
    match line.find("//") {
        Some(i) => &line[..i],
        None => line,
    }
}

fn num<T: std::str::FromStr>(line: usize, s: &str, what: &str) -> Result<T, SceneError> {
    s.parse().map_err(|_| SceneError::new(line, format!("{what} '{s}' is not a valid number")))
}

fn hex_or_num(line: usize, s: &str) -> Result<u64, SceneError> {
    match s.strip_prefix("0x") {
        Some(h) => u64::from_str_radix(h, 16)
            .map_err(|_| SceneError::new(line, format!("'{s}' is not hex"))),
        None => num(line, s, "value"),
    }
}

fn rect(line: usize, a: &[&str]) -> Result<IRect, SceneError> {
    if a.len() < 4 {
        return Err(SceneError::new(line, "a rectangle needs x0 y0 x1 y1"));
    }
    Ok(IRect::new(
        num(line, a[0], "x0")?,
        num(line, a[1], "y0")?,
        num(line, a[2], "x1")?,
        num(line, a[3], "y1")?,
    ))
}

fn need(line: usize, a: &[&str], n: usize, usage: &str) -> Result<(), SceneError> {
    if a.len() < n { Err(SceneError::new(line, format!("expected: {usage}"))) } else { Ok(()) }
}

fn tool_number(line: usize, s: &str) -> Result<u8, SceneError> {
    use fe_engine::tools;
    Ok(match s {
        "none" => tools::NONE,
        "dig" => tools::DIG,
        "paint" => tools::PAINT,
        "explode" => tools::EXPLODE,
        "heat" => tools::HEAT,
        "freeze" => tools::FREEZE,
        "crate" => tools::CRATE,
        other => num(line, other, "tool")?,
    })
}

fn parse_input(line: usize, a: &[&str]) -> Result<InputSpec, SceneError> {
    need(
        line,
        a,
        1,
        "input PLAYER [move=-1|0|1] [jump] [tool=NAME] [param=N|MATERIAL] [aim=DX,DY] [use]",
    )?;
    let mut spec = InputSpec { player: num(line, a[0], "player")?, ..Default::default() };
    for kv in &a[1..] {
        let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
        match k {
            "move" => spec.move_x = num::<i8>(line, v, "move")?.clamp(-1, 1),
            "jump" => spec.jump = true,
            "use" => spec.use_tool = true,
            "tool" => spec.tool = tool_number(line, v)?,
            "param" => {
                spec.param = match v.parse::<u16>() {
                    Ok(n) => ParamSpec::Num(n),
                    Err(_) => ParamSpec::Material(v.to_string()),
                }
            }
            "aim" => {
                let (x, y) =
                    v.split_once(',').ok_or_else(|| SceneError::new(line, "aim must be DX,DY"))?;
                spec.aim = (num(line, x, "aim x")?, num(line, y, "aim y")?);
            }
            other => return Err(SceneError::new(line, format!("unknown input key '{other}'"))),
        }
    }
    Ok(spec)
}

/// Parses one operation line.
pub(crate) fn parse_op(line: usize, text: &str) -> Result<Op, SceneError> {
    let parts: Vec<&str> = text.split_whitespace().collect();
    let Some((&word, a)) = parts.split_first() else {
        return Err(SceneError::new(line, "empty operation"));
    };
    let op = match word {
        "fill" => {
            need(line, a, 5, "fill X0 Y0 X1 Y1 MATERIAL")?;
            Op::Fill(rect(line, a)?, a[4].to_string())
        }
        "circle" => {
            need(line, a, 4, "circle X Y R MATERIAL")?;
            Op::Circle(
                num(line, a[0], "x")?,
                num(line, a[1], "y")?,
                num(line, a[2], "r")?,
                a[3].to_string(),
            )
        }
        "paint" => {
            need(line, a, 4, "paint X Y R MATERIAL")?;
            Op::Paint(
                num(line, a[0], "x")?,
                num(line, a[1], "y")?,
                num(line, a[2], "r")?,
                a[3].to_string(),
            )
        }
        "body" => {
            need(line, a, 5, "body X0 Y0 X1 Y1 MATERIAL")?;
            Op::Body(rect(line, a)?, a[4].to_string())
        }
        "carve" => Op::Carve(rect(line, a)?),
        "terrain" => {
            need(line, a, 6, "terrain X0 X1 BASE AMP SCALE MATERIAL [BOTTOM]")?;
            Op::Terrain {
                x0: num(line, a[0], "x0")?,
                x1: num(line, a[1], "x1")?,
                base: num(line, a[2], "base")?,
                amp: num(line, a[3], "amp")?,
                scale: num::<i32>(line, a[4], "scale")?.max(1),
                material: a[5].to_string(),
                bottom: match a.get(6) {
                    Some(b) => num(line, b, "bottom")?,
                    None => i32::MAX,
                },
            }
        }
        "scatter" => {
            need(line, a, 6, "scatter X0 Y0 X1 Y1 MATERIAL PERCENT")?;
            let pct: u32 = num(line, a[5].trim_end_matches('%'), "percent")?;
            Op::Scatter(rect(line, a)?, a[4].to_string(), pct.min(100))
        }
        "decorate" => {
            need(line, a, 7, "decorate X0 Y0 X1 Y1 MATERIAL PERCENT floor|ceiling|walls|any")?;
            let pct: u32 = num(line, a[5].trim_end_matches('%'), "percent")?;
            let surface = match a[6] {
                "floor" => Surface::Floor,
                "ceiling" => Surface::Ceiling,
                "walls" => Surface::Walls,
                "any" => Surface::Any,
                other => {
                    return Err(SceneError::new(
                        line,
                        format!("surface '{other}' must be floor, ceiling, walls, or any"),
                    ));
                }
            };
            Op::Decorate(rect(line, a)?, a[4].to_string(), pct.min(100), surface)
        }
        "temp" => {
            need(line, a, 5, "temp X0 Y0 X1 Y1 CELSIUS")?;
            Op::Temp(rect(line, a)?, num(line, a[4], "celsius")?)
        }
        "player" => {
            need(line, a, 3, "player ID X Y")?;
            Op::Player(num(line, a[0], "id")?, num(line, a[1], "x")?, num(line, a[2], "y")?)
        }
        "remove-player" => {
            need(line, a, 1, "remove-player ID")?;
            Op::RemovePlayer(num(line, a[0], "id")?)
        }
        "pin" => {
            need(line, a, 3, "pin BODY X Y [BREAK_FORCE]")?;
            let brk = a.get(3).map_or(Ok(0.0), |b| num(line, b, "break force"))?;
            Op::Pin(num(line, a[0], "body")?, num(line, a[1], "x")?, num(line, a[2], "y")?, brk)
        }
        "joint" => {
            need(line, a, 5, "joint A B X Y hinge|fixed [BREAK_FORCE]")?;
            let kind = match a[4] {
                "hinge" => fe_engine::JointKind::Hinge,
                "fixed" => fe_engine::JointKind::Fixed,
                other => {
                    return Err(SceneError::new(
                        line,
                        format!("joint kind '{other}' must be hinge or fixed"),
                    ));
                }
            };
            let brk = a.get(5).map_or(Ok(0.0), |b| num(line, b, "break force"))?;
            Op::Joint(
                num(line, a[0], "a")?,
                num(line, a[1], "b")?,
                num(line, a[2], "x")?,
                num(line, a[3], "y")?,
                kind,
                brk,
            )
        }
        "explode" => {
            need(line, a, 4, "explode X Y RADIUS POWER")?;
            Op::Explode(
                num(line, a[0], "x")?,
                num(line, a[1], "y")?,
                num(line, a[2], "radius")?,
                num(line, a[3], "power")?,
            )
        }
        "dig" => {
            need(line, a, 3, "dig X Y R")?;
            Op::Dig(num(line, a[0], "x")?, num(line, a[1], "y")?, num(line, a[2], "r")?)
        }
        "heat" => {
            need(line, a, 4, "heat X Y R DELTA")?;
            Op::Heat(
                num(line, a[0], "x")?,
                num(line, a[1], "y")?,
                num(line, a[2], "r")?,
                num(line, a[3], "delta")?,
            )
        }
        "impulse" => {
            need(line, a, 3, "impulse BODY VX VY")?;
            Op::Impulse(num(line, a[0], "body")?, num(line, a[1], "vx")?, num(line, a[2], "vy")?)
        }
        "input" => Op::Input(parse_input(line, a)?),
        "run" => {
            need(line, a, 1, "run TICKS")?;
            Op::Run(num(line, a[0], "ticks")?)
        }
        other => return Err(SceneError::new(line, format!("unknown operation '{other}'"))),
    };
    Ok(op)
}

fn parse_when(line: usize, s: &str, default_end: u64) -> Result<When, SceneError> {
    let w: Vec<&str> = s.split_whitespace().collect();
    match w.as_slice() {
        ["at", t] => match t.split_once("..") {
            Some((a, b)) => Ok(When::Range(num(line, a, "start")?, num(line, b, "end")?)),
            None => Ok(When::At(num(line, t, "tick")?)),
        },
        ["every", n] => Ok(When::Every(num(line, n, "interval")?, 0, default_end)),
        ["every", n, "from", a, "to", b] => {
            Ok(When::Every(num(line, n, "interval")?, num(line, a, "start")?, num(line, b, "end")?))
        }
        _ => Err(SceneError::new(
            line,
            "timing must be 'at T', 'at A..B', 'every N', or 'every N from A to B'",
        )),
    }
}

fn parse_check(line: usize, s: &str) -> Result<Check, SceneError> {
    let w: Vec<&str> = s.split_whitespace().collect();
    let cmp = |t: &str| {
        Cmp::parse(t).ok_or_else(|| SceneError::new(line, format!("'{t}' is not a comparison")))
    };
    Ok(match w.as_slice() {
        ["count", mat, "in", x0, y0, x1, y1, op, n] => Check::Count(
            mat.to_string(),
            rect(line, &[x0, y0, x1, y1])?,
            cmp(op)?,
            num(line, n, "count")?,
        ),
        ["hash", "==", h] => Check::Hash(hex_or_num(line, h)?),
        ["bodies", op, n] => Check::Bodies(cmp(op)?, num(line, n, "count")?),
        ["particles", op, n] => Check::Particles(cmp(op)?, num(line, n, "count")?),
        ["active", op, n] => Check::Active(cmp(op)?, num(line, n, "count")?),
        ["asleep"] => Check::Active(Cmp::Eq, 0),
        ["avatar", id, axis @ ("x" | "y"), op, n] => {
            Check::Avatar(num(line, id, "player")?, *axis == "x", cmp(op)?, num(line, n, "value")?)
        }
        _ => return Err(SceneError::new(line, format!("unknown check '{s}'"))),
    })
}

/// Parses scene text. Never panics, whatever the input.
pub fn parse(text: &str) -> Result<Scene, SceneError> {
    let mut scene = Scene {
        name: String::new(),
        description: String::new(),
        bounds: IRect::new(0, 0, 256, 256),
        seed: 1,
        materials: "base".into(),
        worldgen: fe_engine::WorldGen::Empty,
        stream: None,
        ticks: 0,
        spawn: None,
        spawns: Vec::new(),
        wind: 0,
        legend: BTreeMap::new(),
        setup: Vec::new(),
        timeline: Vec::new(),
        camera: Camera::default(),
        expects: Vec::new(),
    };
    let mut section = Section::None;
    let mut grid: Option<(i32, i32, Vec<String>, usize)> = None;
    let mut pending_events: Vec<(usize, String, String)> = Vec::new();

    for (i, raw) in text.lines().enumerate() {
        let line = i + 1;
        let trimmed = raw.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            if let Some((x, y, rows, l)) = grid.take() {
                scene.setup.push(Setup::Grid { x, y, rows, line: l });
            }
            let inner: Vec<&str> = trimmed[1..trimmed.len() - 1].split_whitespace().collect();
            section = match inner.first().copied() {
                Some("scene") => Section::Scene,
                Some("legend") => Section::Legend,
                Some("grid") => {
                    let x = inner.get(1).map_or(Ok(0), |v| num(line, v, "grid x"))?;
                    let y = inner.get(2).map_or(Ok(0), |v| num(line, v, "grid y"))?;
                    grid = Some((x, y, Vec::new(), line + 1));
                    Section::Grid
                }
                Some("build") => Section::Build,
                Some("events") => Section::Events,
                Some("camera") => Section::Camera,
                Some("expect") => Section::Expect,
                _ => return Err(SceneError::new(line, format!("unknown section {trimmed}"))),
            };
            continue;
        }
        if section == Section::Grid {
            if let Some((_, _, rows, _)) = grid.as_mut()
                && !raw.trim_end().is_empty()
            {
                rows.push(raw.trim_end().to_string());
            }
            continue;
        }
        let body = strip_comment(raw).trim();
        if body.is_empty() {
            continue;
        }
        match section {
            Section::None => return Err(SceneError::new(line, "text before the first [section]")),
            Section::Scene => {
                let (k, v) = body
                    .split_once('=')
                    .ok_or_else(|| SceneError::new(line, "expected key = value"))?;
                let (k, v) = (k.trim(), v.trim());
                let w: Vec<&str> = v.split_whitespace().collect();
                match k {
                    "name" => scene.name = v.to_string(),
                    "description" => scene.description = v.to_string(),
                    "seed" => scene.seed = hex_or_num(line, v)?,
                    "materials" => scene.materials = v.to_string(),
                    "ticks" => scene.ticks = num(line, v, "ticks")?,
                    "size" => {
                        need(line, &w, 2, "size = WIDTH HEIGHT")?;
                        scene.bounds =
                            IRect::new(0, 0, num(line, w[0], "width")?, num(line, w[1], "height")?);
                    }
                    "bounds" => scene.bounds = rect(line, &w)?,
                    "wind" => scene.wind = num::<i8>(line, v, "wind")?.clamp(-4, 4),
                    "spawn" => {
                        need(line, &w, 2, "spawn = X Y")?;
                        let p = (num(line, w[0], "x")?, num(line, w[1], "y")?);
                        scene.spawn.get_or_insert(p);
                        scene.spawns.push(p);
                    }
                    "worldgen" => {
                        scene.worldgen = match w.as_slice() {
                            ["empty"] => fe_engine::WorldGen::Empty,
                            ["flat", g, b] => fe_engine::WorldGen::Flat {
                                ground: num(line, g, "ground")?,
                                bedrock: num(line, b, "bedrock")?,
                            },
                            ["caves", s] => {
                                fe_engine::WorldGen::Caves { surface: num(line, s, "surface")? }
                            }
                            _ => {
                                return Err(SceneError::new(
                                    line,
                                    "worldgen must be 'empty', 'flat GROUND BEDROCK', or 'caves SURFACE'",
                                ));
                            }
                        }
                    }
                    "stream" => {
                        need(line, &w, 2, "stream = RADIUS MARGIN")?;
                        scene.stream =
                            Some((num(line, w[0], "radius")?, num(line, w[1], "margin")?));
                    }
                    other => {
                        return Err(SceneError::new(line, format!("unknown scene key '{other}'")));
                    }
                }
            }
            Section::Legend => {
                let mut chars = body.chars();
                let ch = chars.next().ok_or_else(|| SceneError::new(line, "empty legend line"))?;
                let rest: Vec<&str> = chars.as_str().split_whitespace().collect();
                let Some((&mat, flags)) = rest.split_first() else {
                    return Err(SceneError::new(
                        line,
                        "legend lines look like: s sand [body] [temp=C]",
                    ));
                };
                let mut entry = LegendEntry { material: mat.to_string(), body: false, temp: None };
                for f in flags {
                    if *f == "body" {
                        entry.body = true;
                    } else if let Some(t) = f.strip_prefix("temp=") {
                        entry.temp = Some(num(line, t, "temp")?);
                    } else {
                        return Err(SceneError::new(line, format!("unknown legend flag '{f}'")));
                    }
                }
                scene.legend.insert(ch, entry);
            }
            Section::Build => scene.setup.push(Setup::Op(parse_op(line, body)?, line)),
            Section::Events => {
                let (when, what) = body
                    .split_once(':')
                    .ok_or_else(|| SceneError::new(line, "expected 'WHEN: ACTION'"))?;
                pending_events.push((line, when.trim().to_string(), what.trim().to_string()));
            }
            Section::Camera => {
                let w: Vec<&str> = body.split_whitespace().collect();
                match w.as_slice() {
                    ["view", vw, vh] => {
                        scene.camera.view = (num(line, vw, "width")?, num(line, vh, "height")?);
                    }
                    ["scale", s] => scene.camera.scale = num::<u32>(line, s, "scale")?.clamp(1, 16),
                    ["sky", s] => {
                        if !matches!(*s, "day" | "dusk" | "night" | "cave") {
                            return Err(SceneError::new(
                                line,
                                "sky must be day, dusk, night, or cave",
                            ));
                        }
                        scene.camera.sky = (*s).to_string();
                    }
                    ["ambient", r, g, b] => {
                        scene.camera.ambient =
                            [num(line, r, "red")?, num(line, g, "green")?, num(line, b, "blue")?];
                    }
                    ["exposure", e] => scene.camera.exposure = num(line, e, "exposure")?,
                    ["at", t, x, y, rest @ ..] => {
                        let tick = num(line, t.trim_end_matches(':'), "tick")?;
                        let zoom = rest.first().map_or(Ok(1.0), |z| {
                            num(line, z.trim_start_matches("zoom="), "zoom")
                        })?;
                        scene.camera.keys.push(CameraKey {
                            tick,
                            x: num(line, x, "x")?,
                            y: num(line, y, "y")?,
                            zoom,
                        });
                    }
                    _ => {
                        return Err(SceneError::new(
                            line,
                            "camera lines: 'view W H', 'scale N', 'sky NAME', 'ambient R G B', 'exposure E', 'at T: X Y [zoom]'",
                        ));
                    }
                }
            }
            Section::Expect => {
                let (when, what) = body
                    .split_once(':')
                    .ok_or_else(|| SceneError::new(line, "expected 'after T: CHECK'"))?;
                let t = when
                    .trim()
                    .strip_prefix("after")
                    .ok_or_else(|| SceneError::new(line, "expectations start with 'after T:'"))?;
                scene.expects.push(Expect {
                    after: num(line, t.trim(), "tick")?,
                    check: parse_check(line, what.trim())?,
                    line,
                });
            }
            Section::Grid => {}
        }
    }
    if let Some((x, y, rows, l)) = grid.take() {
        scene.setup.push(Setup::Grid { x, y, rows, line: l });
    }

    let last_expect = scene.expects.iter().map(|e| e.after).max().unwrap_or(0);
    let default_end = if scene.ticks > 0 { scene.ticks } else { last_expect.max(600) };
    for (line, when, what) in pending_events {
        let when = parse_when(line, &when, default_end)?;
        scene.timeline.push(Timed { when, op: parse_op(line, &what)?, line });
    }
    if scene.ticks == 0 {
        let last_event = scene.timeline.iter().map(|t| t.when.end()).max().unwrap_or(0);
        scene.ticks = last_expect.max(last_event).max(1);
    }
    scene.camera.keys.sort_by_key(|k| k.tick);
    Ok(scene)
}
