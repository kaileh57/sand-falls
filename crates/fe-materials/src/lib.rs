//! Material and reaction definitions.
//!
//! Materials are written in TOML (see `data/base.toml`) and compiled into a
//! [`MaterialSet`]: flat arrays indexed by [`MaterialId`] with integer fields
//! only, so the simulation's inner loop never touches strings or floats.
#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic)]

mod chance;
mod def;
mod set;

pub use chance::{CHANCE_ONE, parse_chance};
pub use def::{MaterialDef, MaterialFile, ReactionDef, ResidueDef};
pub use set::{
    Explode, Ignite, Material, MaterialSet, Reaction, SpawnDir, SpawnRule, State, Transition,
};

use fe_core::MaterialId;

/// The built-in material file used when a scene does not name one.
pub const BASE_MATERIALS: &str = include_str!("../data/base.toml");

/// An error produced while loading a material file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaterialError {
    /// The TOML text could not be parsed.
    Parse(String),
    /// A material or reaction refers to a name that is not defined.
    UnknownMaterial {
        /// The name that was not found.
        name: String,
        /// Where the reference appeared.
        context: String,
    },
    /// Two materials share a name.
    Duplicate(String),
    /// A field has a value outside its allowed range.
    Invalid {
        /// Material or reaction the field belongs to.
        context: String,
        /// What is wrong.
        message: String,
    },
    /// The file defines more materials than a cell can address.
    TooMany(usize),
}

impl std::fmt::Display for MaterialError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "material file does not parse: {e}"),
            Self::UnknownMaterial { name, context } => {
                write!(f, "unknown material '{name}' referenced by {context}")
            }
            Self::Duplicate(n) => write!(f, "material '{n}' is defined twice"),
            Self::Invalid { context, message } => write!(f, "{context}: {message}"),
            Self::TooMany(n) => {
                write!(f, "{n} materials defined; the limit is {}", MaterialId::MAX_COUNT)
            }
        }
    }
}

impl std::error::Error for MaterialError {}

/// Loads and compiles the built-in material set.
///
/// # Panics
///
/// Never in a released build: the built-in file is checked by this crate's tests.
pub fn base_set() -> MaterialSet {
    match MaterialSet::from_toml(BASE_MATERIALS) {
        Ok(s) => s,
        Err(e) => panic!("built-in material file is invalid: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_set_compiles() {
        let set = base_set();
        assert!(set.len() > 30);
        assert_eq!(set.id("air"), Some(MaterialId::AIR));
        assert_eq!(set.id("boundary"), Some(MaterialId::BOUNDARY));
        let water = set.id("water").unwrap_or_default();
        assert_eq!(set.get(water).state, State::Liquid);
    }

    #[test]
    fn hash_is_stable_across_loads() {
        assert_eq!(base_set().hash(), base_set().hash());
    }

    #[test]
    fn hash_changes_when_a_material_changes() {
        let a = base_set().hash();
        let edited = BASE_MATERIALS.replacen("density = 150", "density = 151", 1);
        let b = MaterialSet::from_toml(&edited).map_or(a, |s| s.hash());
        assert_ne!(a, b);
    }

    #[test]
    fn unknown_reference_is_reported() {
        let text = r#"
            [[material]]
            name = "goo"
            state = "liquid"
            decay = "nothing"
        "#;
        assert!(matches!(MaterialSet::from_toml(text), Err(MaterialError::UnknownMaterial { .. })));
    }

    #[test]
    fn duplicate_is_reported() {
        let text = r#"
            [[material]]
            name = "goo"
            state = "liquid"
            [[material]]
            name = "goo"
            state = "liquid"
        "#;
        assert_eq!(MaterialSet::from_toml(text), Err(MaterialError::Duplicate("goo".into())));
    }
}
