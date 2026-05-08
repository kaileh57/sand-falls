use crate::MaterialError;
use crate::chance::{CHANCE_ONE, parse_chance};
use crate::def::{MaterialDef, MaterialFile};
use fe_core::{MaterialId, StableHasher, celsius_to_temp};
use std::collections::BTreeMap;

/// How a material moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum State {
    /// Air. Only material 0 has this state.
    Empty,
    /// Does not move. Solids, terrain, and structures.
    Static,
    /// Falls and piles up.
    Powder,
    /// Falls and spreads sideways.
    Liquid,
    /// Rises and spreads.
    Gas,
    /// Flickers upward, burns out, and heats what it touches.
    Fire,
}

impl State {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "static" => Self::Static,
            "powder" => Self::Powder,
            "liquid" => Self::Liquid,
            "gas" => Self::Gas,
            "fire" => Self::Fire,
            _ => return None,
        })
    }

    /// Liquids, gases and fire: states other cells can sink or rise through.
    pub const fn is_fluid(self) -> bool {
        matches!(self, Self::Liquid | Self::Gas | Self::Fire)
    }

    /// Powders and liquids: states that fall.
    pub const fn falls(self) -> bool {
        matches!(self, Self::Powder | Self::Liquid)
    }

    const fn code(self) -> u32 {
        self as u32
    }
}

/// A temperature-triggered material change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transition {
    /// Threshold on the engine temperature scale.
    pub at: u8,
    /// Material to become.
    pub into: MaterialId,
    /// Latent heat in temperature steps. 0 changes at once.
    pub latent: u8,
}

/// Compiled ignition settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ignite {
    /// Ignition temperature on the engine scale.
    pub at: u8,
    /// Chance per tick out of [`CHANCE_ONE`].
    pub chance: u32,
    /// Material to become.
    pub into: MaterialId,
}

/// Compiled explosion settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Explode {
    /// Detonation temperature on the engine scale.
    pub at: u8,
    /// Blast radius in cells.
    pub radius: u8,
    /// Damage at the center.
    pub power: u8,
}

/// Which neighbors a spawn rule may fill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpawnDir {
    /// The cell above.
    Up,
    /// The cell below.
    Down,
    /// Any of the four neighbors.
    Any,
}

/// Compiled spawn rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpawnRule {
    /// Material to spawn.
    pub into: MaterialId,
    /// Chance per tick out of [`CHANCE_ONE`].
    pub chance: u32,
    /// Allowed neighbors.
    pub dir: SpawnDir,
}

/// One compiled material. Every field is an integer or a flag.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Material {
    /// Unique name.
    pub name: String,
    /// Movement state.
    pub state: State,
    /// Relative density. Air is 0, water is 100.
    pub density: i16,
    /// Display colors. Never empty.
    pub colors: Vec<[u8; 3]>,
    /// Powders: chance to slide diagonally, out of [`CHANCE_ONE`].
    pub slide: u32,
    /// Liquids and gases: sideways search distance in cells.
    pub spread: u8,
    /// Starting temperature.
    pub temp: u8,
    /// Share of a temperature difference exchanged per tick, out of 256.
    pub conductivity: u8,
    /// Chance out of 256 per tick to drift one step toward ambient.
    pub cooling: u8,
    /// Relative heat capacity per cell, at least 1.
    pub heat_capacity: u8,
    /// Hit points for static solids.
    pub hp: u8,
    /// What the material becomes at zero hit points.
    pub debris: MaterialId,
    /// Holds up connected structural cells.
    pub anchored: bool,
    /// Falls when disconnected from an anchor.
    pub structural: bool,
    /// Ignores damage.
    pub indestructible: bool,
    /// Impact strength at which a body of this material shatters. 0 never shatters.
    pub fracture: u16,
    /// Melting or boiling.
    pub melt: Option<Transition>,
    /// Freezing or condensing.
    pub freeze: Option<Transition>,
    /// Ignition.
    pub ignite: Option<Ignite>,
    /// Explosion.
    pub explode: Option<Explode>,
    /// Lifetime range in ticks.
    pub lifetime: Option<(u8, u8)>,
    /// Material after the lifetime ends.
    pub decay: MaterialId,
    /// Material sometimes left instead of `decay`, with its chance out of [`CHANCE_ONE`].
    pub residue: Option<(MaterialId, u32)>,
    /// Chance per tick out of [`CHANCE_ONE`] that the lifetime counts down.
    pub lifetime_rate: u32,
    /// Spawn rule.
    pub spawn: Option<SpawnRule>,
    /// Minimum temperature the cell holds. 0 means none.
    pub hold_temp: u8,
    /// Emitted light. All zero means none.
    pub light: [u8; 3],
    /// Light-blocking strength.
    pub opacity: u8,
    /// Drag on bodies.
    pub viscosity: u8,
    /// Whether hot cells glow.
    pub glow: bool,
}

impl Material {
    /// Returns true for cells that block movement and support characters.
    pub const fn is_solid(&self) -> bool {
        matches!(self.state, State::Static | State::Powder)
    }

    /// Returns true if the value byte holds a remaining lifetime.
    pub const fn has_lifetime(&self) -> bool {
        self.lifetime.is_some()
    }
}

/// The result of a reaction from one side's point of view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reaction {
    /// What the reacting cell becomes.
    pub into_self: MaterialId,
    /// What the neighbor becomes.
    pub into_other: MaterialId,
    /// Chance per tick out of [`CHANCE_ONE`].
    pub chance: u32,
    /// Lowest temperature the products start at, on the 8-bit scale; 0 for none.
    pub min_temp: u8,
}

/// A compiled material set: flat tables indexed by material id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaterialSet {
    materials: Vec<Material>,
    by_name: BTreeMap<String, MaterialId>,
    /// `n * n` entries; 0 means no reaction, otherwise an index + 1 into `reactions`.
    table: Vec<u32>,
    reactions: Vec<Reaction>,
    hash: u64,
}

fn parse_color(s: &str, ctx: &str) -> Result<[u8; 3], MaterialError> {
    let bad = || MaterialError::Invalid {
        context: ctx.to_string(),
        message: format!("color '{s}' is not #rrggbb"),
    };
    let hex = s.strip_prefix('#').ok_or_else(bad)?;
    if hex.len() != 6 {
        return Err(bad());
    }
    let v = u32::from_str_radix(hex, 16).map_err(|_| bad())?;
    Ok([(v >> 16) as u8, (v >> 8) as u8, v as u8])
}

fn chance(s: &str, ctx: &str) -> Result<u32, MaterialError> {
    parse_chance(s).ok_or_else(|| MaterialError::Invalid {
        context: ctx.to_string(),
        message: format!("chance '{s}' must look like \"5%\" or \"1/20\""),
    })
}

fn air_def() -> Material {
    Material {
        name: "air".into(),
        state: State::Empty,
        density: 0,
        colors: vec![[0, 0, 0]],
        slide: 0,
        spread: 0,
        temp: fe_core::AMBIENT_TEMP,
        // Air holds a fiftieth of the heat a metal cell does, so it can conduct
        // well without draining what it touches: it warms up instead, and hot
        // air rises off the heat as haze.
        conductivity: 24,
        cooling: 48,
        heat_capacity: 1,
        hp: 0,
        debris: MaterialId::AIR,
        anchored: false,
        structural: false,
        indestructible: true,
        fracture: 0,
        melt: None,
        freeze: None,
        ignite: None,
        explode: None,
        lifetime: None,
        decay: MaterialId::AIR,
        residue: None,
        lifetime_rate: CHANCE_ONE,
        spawn: None,
        hold_temp: 0,
        light: [0, 0, 0],
        opacity: 0,
        viscosity: 0,
        glow: false,
    }
}

fn boundary_def() -> Material {
    Material {
        name: "boundary".into(),
        state: State::Static,
        density: i16::MAX,
        colors: vec![[8, 8, 10]],
        conductivity: 0,
        cooling: 0,
        hp: 255,
        anchored: true,
        structural: true,
        opacity: 255,
        glow: false,
        ..air_def()
    }
}

impl MaterialSet {
    /// Parses and compiles a TOML material file.
    pub fn from_toml(text: &str) -> Result<Self, MaterialError> {
        let file: MaterialFile =
            toml::from_str(text).map_err(|e| MaterialError::Parse(e.to_string()))?;
        Self::compile(&file)
    }

    /// Compiles a parsed material file.
    pub fn compile(file: &MaterialFile) -> Result<Self, MaterialError> {
        let total = file.material.len() + 2;
        if total > MaterialId::MAX_COUNT {
            return Err(MaterialError::TooMany(total));
        }
        let mut by_name = BTreeMap::new();
        by_name.insert("air".to_string(), MaterialId::AIR);
        by_name.insert("boundary".to_string(), MaterialId::BOUNDARY);
        for (i, d) in file.material.iter().enumerate() {
            let id = MaterialId((i + 2) as u16);
            if by_name.insert(d.name.clone(), id).is_some() {
                return Err(MaterialError::Duplicate(d.name.clone()));
            }
        }
        let lookup = |name: &str, ctx: &str| {
            by_name.get(name).copied().ok_or_else(|| MaterialError::UnknownMaterial {
                name: name.to_string(),
                context: ctx.to_string(),
            })
        };

        let mut materials = vec![air_def(), boundary_def()];
        for d in &file.material {
            materials.push(compile_one(d, &lookup)?);
        }

        let n = materials.len();
        let mut table = vec![0u32; n * n];
        let mut reactions = Vec::new();
        for r in &file.reaction {
            let ctx = format!("reaction {} + {}", r.a, r.b);
            let a = lookup(&r.a, &ctx)?;
            let b = lookup(&r.b, &ctx)?;
            let ia = lookup(&r.into_a, &ctx)?;
            let ib = lookup(&r.into_b, &ctx)?;
            let c = chance(&r.chance, &ctx)?;
            let min_temp = r.temp.map_or(0, celsius_to_temp);
            reactions.push(Reaction { into_self: ia, into_other: ib, chance: c, min_temp });
            table[a.index() * n + b.index()] = reactions.len() as u32;
            if a != b {
                reactions.push(Reaction { into_self: ib, into_other: ia, chance: c, min_temp });
                table[b.index() * n + a.index()] = reactions.len() as u32;
            }
        }

        let mut set = Self { materials, by_name, table, reactions, hash: 0 };
        set.hash = set.compute_hash();
        Ok(set)
    }

    /// Number of materials, including `air` and `boundary`.
    pub fn len(&self) -> usize {
        self.materials.len()
    }

    /// Always false: a set contains at least `air` and `boundary`.
    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }

    /// Looks up a material id by name.
    pub fn id(&self, name: &str) -> Option<MaterialId> {
        self.by_name.get(name).copied()
    }

    /// Returns the material for an id. Unknown ids map to `boundary`.
    #[inline]
    pub fn get(&self, id: MaterialId) -> &Material {
        self.materials.get(id.index()).unwrap_or(&self.materials[1])
    }

    /// Iterates over all materials in id order.
    pub fn iter(&self) -> impl Iterator<Item = (MaterialId, &Material)> {
        self.materials.iter().enumerate().map(|(i, m)| (MaterialId(i as u16), m))
    }

    /// Returns the reaction for `a` touching `b`, from `a`'s side.
    #[inline]
    pub fn reaction(&self, a: MaterialId, b: MaterialId) -> Option<&Reaction> {
        let n = self.materials.len();
        if a.index() >= n || b.index() >= n {
            return None;
        }
        match self.table[a.index() * n + b.index()] {
            0 => None,
            i => self.reactions.get(i as usize - 1),
        }
    }

    /// Returns true if `a` reacts with anything.
    pub fn has_reactions(&self, a: MaterialId) -> bool {
        let n = self.materials.len();
        a.index() < n && self.table[a.index() * n..(a.index() + 1) * n].iter().any(|&v| v != 0)
    }

    /// A hash of every compiled field. Clients and servers compare it before joining.
    pub const fn hash(&self) -> u64 {
        self.hash
    }

    fn compute_hash(&self) -> u64 {
        let mut h = StableHasher::new();
        h.write_u64(self.materials.len() as u64);
        for m in &self.materials {
            h.write_bytes(m.name.as_bytes());
            h.write_u32(m.state.code());
            h.write_i32(i32::from(m.density));
            for c in &m.colors {
                h.write_u32(u32::from_le_bytes([c[0], c[1], c[2], 0]));
            }
            h.write_u32(m.slide);
            h.write_u32(u32::from_le_bytes([m.spread, m.temp, m.conductivity, m.cooling]));
            h.write_u32(u32::from_le_bytes([m.hp, m.opacity, m.viscosity, m.hold_temp]));
            h.write_u32(u32::from(m.debris.0) | u32::from(m.decay.0) << 16);
            h.write_u32(
                u32::from(m.anchored)
                    | u32::from(m.structural) << 1
                    | u32::from(m.indestructible) << 2
                    | u32::from(m.glow) << 3,
            );
            h.write_u32(u32::from(m.fracture));
            h.write_u32(u32::from(m.heat_capacity));
            for t in [m.melt, m.freeze] {
                match t {
                    Some(t) => {
                        h.write_u32(u32::from(t.at) | u32::from(t.into.0) << 8);
                        h.write_u32(u32::from(t.latent));
                    }
                    None => h.write_u32(u32::MAX),
                }
            }
            match m.ignite {
                Some(i) => {
                    h.write_u32(u32::from(i.at) | u32::from(i.into.0) << 8);
                    h.write_u32(i.chance);
                }
                None => h.write_u32(u32::MAX),
            }
            match m.explode {
                Some(e) => h.write_u32(u32::from_le_bytes([e.at, e.radius, e.power, 1])),
                None => h.write_u32(u32::MAX),
            }
            match m.lifetime {
                Some((a, b)) => h.write_u32(u32::from(a) | u32::from(b) << 8),
                None => h.write_u32(u32::MAX),
            }
            match m.residue {
                Some((into, chance)) => {
                    h.write_u32(u32::from(into.0));
                    h.write_u32(chance);
                }
                None => h.write_u32(u32::MAX),
            }
            h.write_u32(m.lifetime_rate);
            match m.spawn {
                Some(s) => {
                    h.write_u32(u32::from(s.into.0) | (s.dir as u32) << 16);
                    h.write_u32(s.chance);
                }
                None => h.write_u32(u32::MAX),
            }
            h.write_u32(u32::from_le_bytes([m.light[0], m.light[1], m.light[2], 0]));
        }
        for (i, &t) in self.table.iter().enumerate() {
            if t != 0 {
                let r = &self.reactions[t as usize - 1];
                h.write_u64(i as u64);
                h.write_u32(u32::from(r.into_self.0) | u32::from(r.into_other.0) << 16);
                h.write_u32(r.chance);
                h.write_u32(u32::from(r.min_temp));
            }
        }
        h.finish()
    }
}

fn compile_one(
    d: &MaterialDef,
    lookup: &impl Fn(&str, &str) -> Result<MaterialId, MaterialError>,
) -> Result<Material, MaterialError> {
    let ctx = format!("material '{}'", d.name);
    let state = State::parse(&d.state).ok_or_else(|| MaterialError::Invalid {
        context: ctx.clone(),
        message: format!("state '{}' must be static, powder, liquid, gas, or fire", d.state),
    })?;

    // Defaults by state: (conductivity, cooling, opacity, spread, hp, glow)
    let (cond, cool, opac, spread, hp, glow) = match state {
        State::Static => (24, 0, 255, 0, 40, true),
        State::Powder => (24, 1, 255, 0, 1, true),
        State::Liquid => (40, 1, 70, 6, 1, true),
        State::Gas => (10, 6, 24, 2, 1, false),
        State::Fire => (200, 0, 0, 1, 1, true),
        State::Empty => (6, 48, 0, 0, 0, false),
    };

    let mut colors = Vec::new();
    for c in &d.colors {
        colors.push(parse_color(c, &ctx)?);
    }
    if colors.is_empty() {
        colors.push([255, 0, 255]);
    }

    let transition = |t: &Option<crate::def::TransitionDef>| -> Result<_, MaterialError> {
        t.as_ref()
            .map(|t| {
                Ok(Transition {
                    at: celsius_to_temp(t.at),
                    into: lookup(&t.into, &ctx)?,
                    latent: (t.latent.unwrap_or(0) / 10).clamp(0, 255) as u8,
                })
            })
            .transpose()
    };

    let ignite = d
        .ignite
        .as_ref()
        .map(|i| {
            Ok::<_, MaterialError>(Ignite {
                at: celsius_to_temp(i.at),
                chance: chance(&i.chance, &ctx)?,
                into: lookup(&i.into, &ctx)?,
            })
        })
        .transpose()?;

    let spawn = d
        .spawn
        .as_ref()
        .map(|s| {
            let dir = match s.dir.as_str() {
                "up" => SpawnDir::Up,
                "down" => SpawnDir::Down,
                "any" => SpawnDir::Any,
                other => {
                    return Err(MaterialError::Invalid {
                        context: ctx.clone(),
                        message: format!("spawn dir '{other}' must be up, down, or any"),
                    });
                }
            };
            Ok(SpawnRule { into: lookup(&s.into, &ctx)?, chance: chance(&s.chance, &ctx)?, dir })
        })
        .transpose()?;

    let lifetime = match (d.lifetime, state) {
        (Some([a, b]), _) => {
            if a == 0 || b < a {
                return Err(MaterialError::Invalid {
                    context: ctx,
                    message: "lifetime must be [min, max] with 0 < min <= max".into(),
                });
            }
            Some((a, b))
        }
        (None, State::Fire) => Some((20, 40)),
        (None, _) => None,
    };

    let hold_temp = match (d.hold_temp, state) {
        (Some(c), _) => celsius_to_temp(c),
        (None, State::Fire) => celsius_to_temp(900),
        (None, _) => 0,
    };

    let light = d.light.as_deref().map(|l| parse_color(l, &ctx)).transpose()?.unwrap_or([0; 3]);
    let slide = d.slide.as_deref().map(|s| chance(s, &ctx)).transpose()?.unwrap_or(match state {
        State::Powder => CHANCE_ONE,
        _ => 0,
    });

    Ok(Material {
        name: d.name.clone(),
        state,
        density: d.density,
        colors,
        slide,
        spread: d.spread.unwrap_or(spread).min(16),
        temp: d.temp.map_or(fe_core::AMBIENT_TEMP, celsius_to_temp),
        conductivity: d.conductivity.unwrap_or(cond),
        cooling: d.cooling.unwrap_or(cool),
        heat_capacity: d
            .heat_capacity
            .unwrap_or(match state {
                State::Static => 20,
                State::Powder => 14,
                State::Liquid => 30,
                State::Gas | State::Fire | State::Empty => 1,
            })
            .max(1),
        hp: d.hp.unwrap_or(hp).max(1),
        debris: d
            .debris
            .as_deref()
            .map(|n| lookup(n, &ctx))
            .transpose()?
            .unwrap_or(MaterialId::AIR),
        anchored: d.anchored,
        structural: d.structural.unwrap_or(state == State::Static),
        indestructible: d.indestructible,
        fracture: d.fracture,
        melt: transition(&d.melt)?,
        freeze: transition(&d.freeze)?,
        ignite,
        explode: d.explode.as_ref().map(|e| Explode {
            at: celsius_to_temp(e.at),
            radius: e.radius,
            power: e.power,
        }),
        lifetime,
        decay: d.decay.as_deref().map(|n| lookup(n, &ctx)).transpose()?.unwrap_or(MaterialId::AIR),
        residue: d
            .residue
            .as_ref()
            .map(|r| Ok::<_, MaterialError>((lookup(&r.into, &ctx)?, chance(&r.chance, &ctx)?)))
            .transpose()?,
        lifetime_rate: d
            .lifetime_rate
            .as_deref()
            .map(|r| chance(r, &ctx))
            .transpose()?
            .unwrap_or(CHANCE_ONE),
        spawn,
        hold_temp,
        light,
        opacity: d.opacity.unwrap_or(opac),
        viscosity: d.viscosity.unwrap_or(match state {
            State::Liquid => 30,
            _ => 0,
        }),
        glow: d.glow.unwrap_or(glow),
    })
}
