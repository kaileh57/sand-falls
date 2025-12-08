use fe_core::{IRect, MaterialId};
use fe_physics::JointKind;
use serde::{Deserialize, Serialize};

/// A world edit applied at the start of a tick, before the simulation runs.
///
/// Commands come from scene timelines, the control API, tools, and the network.
/// They are applied in the order given, so the same list always has the same effect.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Command {
    /// Fills a disc with fresh cells of a material. Air cells only, unless `replace`.
    Paint {
        /// Center x.
        x: i32,
        /// Center y.
        y: i32,
        /// Radius in cells.
        radius: i32,
        /// Material id.
        material: MaterialId,
        /// Overwrite non-air cells too (never the boundary or indestructible cells).
        #[serde(default)]
        replace: bool,
    },
    /// Fills a rectangle with fresh cells of a material, overwriting what is there.
    FillRect {
        /// Area to fill.
        rect: IRect,
        /// Material id.
        material: MaterialId,
    },
    /// Removes destructible cells in a disc.
    Dig {
        /// Center x.
        x: i32,
        /// Center y.
        y: i32,
        /// Radius in cells.
        radius: i32,
    },
    /// Detonates an explosion.
    Explode {
        /// Center x.
        x: i32,
        /// Center y.
        y: i32,
        /// Radius in cells.
        radius: u8,
        /// Damage at the center.
        power: u8,
    },
    /// Adds heat (or removes it, when negative) in a disc.
    Heat {
        /// Center x.
        x: i32,
        /// Center y.
        y: i32,
        /// Radius in cells.
        radius: i32,
        /// Temperature change in engine units (10 degrees C each).
        delta: i16,
    },
    /// Cuts the static cells in a rectangle out of the world as one rigid body.
    CarveBody {
        /// Area to carve.
        rect: IRect,
    },
    /// Pushes a body, in cells per second.
    Impulse {
        /// Body id.
        body: u32,
        /// Velocity change in cells per second.
        vx: f32,
        /// Velocity change in cells per second.
        vy: f32,
    },
    /// Joins two bodies.
    Joint {
        /// First body.
        a: u32,
        /// Second body.
        b: u32,
        /// Anchor x.
        x: i32,
        /// Anchor y.
        y: i32,
        /// Hinge or fixed.
        kind: JointKind,
        /// Break force in newtons; 0 never breaks.
        break_force: f32,
    },
    /// Pins a body to the world with a hinge.
    Pin {
        /// Body id.
        body: u32,
        /// Anchor x.
        x: i32,
        /// Anchor y.
        y: i32,
        /// Break force in newtons; 0 never breaks.
        break_force: f32,
    },
    /// Adds a player character with its feet at a position.
    AddPlayer {
        /// Player id.
        id: u8,
        /// Feet x.
        x: i32,
        /// Feet y.
        y: i32,
    },
    /// Removes a player character.
    RemovePlayer {
        /// Player id.
        id: u8,
    },
}
