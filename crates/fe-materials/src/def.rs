use serde::Deserialize;

/// A material file as written in TOML.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialFile {
    /// Material definitions, in id order. `air` and `boundary` are added automatically.
    #[serde(default)]
    pub material: Vec<MaterialDef>,
    /// Pairwise reactions.
    #[serde(default)]
    pub reaction: Vec<ReactionDef>,
}

/// A temperature-triggered change into another material.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransitionDef {
    /// Threshold in degrees C.
    pub at: i32,
    /// Material to become.
    pub into: String,
    /// Latent heat in degrees C of the material's own temperature: how much heat
    /// the cell takes in (melting) or gives off (freezing) at the threshold
    /// before it changes. Unset or 0 changes at once.
    #[serde(default)]
    pub latent: Option<i32>,
}

/// A material sometimes left behind when a lifetime ends.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidueDef {
    /// Material left behind.
    pub into: String,
    /// Chance, as `"10%"` or `"1/10"`, that a cell leaves it instead of its decay.
    pub chance: String,
}

/// Ignition settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IgniteDef {
    /// Ignition temperature in degrees C.
    pub at: i32,
    /// Chance per tick, as `"5%"` or `"1/20"`.
    pub chance: String,
    /// Material to become when it ignites.
    pub into: String,
}

/// Explosion settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExplodeDef {
    /// Detonation temperature in degrees C.
    pub at: i32,
    /// Blast radius in cells.
    pub radius: u8,
    /// Damage at the center.
    pub power: u8,
}

/// A rule that spawns a material into a neighboring air cell.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpawnDef {
    /// Material to spawn.
    pub into: String,
    /// Chance per tick.
    pub chance: String,
    /// `"up"`, `"down"`, or `"any"`.
    #[serde(default = "default_dir")]
    pub dir: String,
}

fn default_dir() -> String {
    "any".into()
}

/// One material as written in TOML. Temperatures are in degrees C.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialDef {
    /// Unique name.
    pub name: String,
    /// `static`, `powder`, `liquid`, `gas`, or `fire`.
    pub state: String,
    /// Relative density. Air is 0, water is 100.
    #[serde(default)]
    pub density: i16,
    /// Display colors as `#rrggbb`. A cell picks one by position.
    #[serde(default)]
    pub colors: Vec<String>,
    /// Powders: chance to slide diagonally when blocked below.
    #[serde(default)]
    pub slide: Option<String>,
    /// Liquids and gases: how many cells sideways the material looks for space.
    #[serde(default)]
    pub spread: Option<u8>,
    /// Starting temperature.
    #[serde(default)]
    pub temp: Option<i32>,
    /// Share of a temperature difference exchanged per tick, out of 255.
    #[serde(default)]
    pub conductivity: Option<u8>,
    /// Chance out of 255 per tick to drift one step toward ambient temperature.
    #[serde(default)]
    pub cooling: Option<u8>,
    /// Heat held per cell per degree, relative to other materials: when two cells
    /// exchange heat, each one's temperature changes in inverse proportion to it.
    /// Defaults: static 20, powder 14, liquid 30, gas and fire 1.
    #[serde(default)]
    pub heat_capacity: Option<u8>,
    /// Hit points for static solids.
    #[serde(default)]
    pub hp: Option<u8>,
    /// What the material becomes when its hit points reach zero.
    #[serde(default)]
    pub debris: Option<String>,
    /// Anchored solids hold up everything connected to them.
    #[serde(default)]
    pub anchored: bool,
    /// Structural solids fall when disconnected from an anchor. Defaults to true for static.
    #[serde(default)]
    pub structural: Option<bool>,
    /// Indestructible cells ignore damage.
    #[serde(default)]
    pub indestructible: bool,
    /// Impact strength at which a rigid body made of this material shatters. 0 never shatters.
    #[serde(default)]
    pub fracture: u16,
    /// Melts or boils at or above a temperature.
    #[serde(default)]
    pub melt: Option<TransitionDef>,
    /// Freezes or condenses at or below a temperature.
    #[serde(default)]
    pub freeze: Option<TransitionDef>,
    /// Catches fire.
    #[serde(default)]
    pub ignite: Option<IgniteDef>,
    /// Explodes when hot enough.
    #[serde(default)]
    pub explode: Option<ExplodeDef>,
    /// Lifetime range in ticks. Unset means the material does not expire.
    #[serde(default)]
    pub lifetime: Option<[u8; 2]>,
    /// What the material becomes when its lifetime ends.
    #[serde(default)]
    pub decay: Option<String>,
    /// A chance to leave a different material than `decay` when the lifetime
    /// ends, such as ash from a burnt plank.
    #[serde(default)]
    pub residue: Option<ResidueDef>,
    /// Chance per tick that the lifetime counts down, as `"50%"`. Defaults to
    /// 100%. Lower rates make a u8 lifetime last longer than 255 ticks.
    #[serde(default)]
    pub lifetime_rate: Option<String>,
    /// Spawns another material into neighboring air.
    #[serde(default)]
    pub spawn: Option<SpawnDef>,
    /// The cell never cools below this temperature.
    #[serde(default)]
    pub hold_temp: Option<i32>,
    /// Emitted light color as `#rrggbb`.
    #[serde(default)]
    pub light: Option<String>,
    /// Light-blocking strength, 0 to 255.
    #[serde(default)]
    pub opacity: Option<u8>,
    /// Drag on rigid bodies moving through the material, 0 to 255.
    #[serde(default)]
    pub viscosity: Option<u8>,
    /// Whether hot cells glow with their temperature.
    #[serde(default)]
    pub glow: Option<bool>,
}

/// A pairwise reaction as written in TOML.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReactionDef {
    /// First reactant.
    pub a: String,
    /// Second reactant.
    pub b: String,
    /// What `a` becomes.
    pub into_a: String,
    /// What `b` becomes.
    pub into_b: String,
    /// Chance per tick of contact.
    pub chance: String,
}
