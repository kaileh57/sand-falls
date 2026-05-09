//! Player characters: integer-only movement against the cell grid.
//!
//! A character's whole state is a few dozen bytes, so a client can rewind and
//! replay its own character cheaply when a prediction turns out wrong.

use crate::world::World;
use fe_core::{Cell, CellPos, Fx, MaterialId, StableHasher};
use fe_materials::State;
use serde::{Deserialize, Serialize};

/// Character width in cells.
pub const AVATAR_W: i32 = 6;
/// Character height in cells.
pub const AVATAR_H: i32 = 12;

/// The world's gravity, 7/256 cells per tick squared (9.84 m/s^2).
const GRAVITY: Fx = fe_core::GRAVITY;
const MAX_FALL: Fx = Fx(8 << 16);
const WALK: Fx = Fx(1 << 16);
const ACCEL_GROUND: Fx = Fx(1 << 14);
const ACCEL_AIR: Fx = Fx(1 << 13);
/// Take-off speed: -0.85 cells per tick, a 13-cell apex and 62 ticks in the air.
const JUMP: Fx = Fx(-55_706);
const SWIM: Fx = Fx(-(3 << 15));
const STEP_UP: i32 = 3;

/// One tick of input for one player.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlayerInput {
    /// -1 left, 0 none, 1 right.
    pub move_x: i8,
    /// Jump or swim up.
    pub jump: bool,
    /// Aim offset from the character's center, in cells.
    pub aim_x: i16,
    /// Aim offset from the character's center, in cells.
    pub aim_y: i16,
    /// Selected tool, interpreted by the game layer.
    pub tool: u8,
    /// Tool parameter, such as a material id or a radius.
    pub param: u16,
    /// Use the tool this tick.
    pub use_tool: bool,
}

/// A player character.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Avatar {
    /// Player id.
    pub id: u8,
    /// Center x in cells.
    pub x: Fx,
    /// Bottom edge (feet) y in cells.
    pub y: Fx,
    /// Velocity in cells per tick.
    pub vx: Fx,
    /// Velocity in cells per tick.
    pub vy: Fx,
    /// Standing on something solid.
    pub on_ground: bool,
    /// Overlapping liquid.
    pub in_liquid: bool,
    /// Health.
    pub hp: u16,
    /// Last horizontal direction, -1 or 1.
    pub facing: i8,
}

impl Avatar {
    /// Creates a character standing with its feet at a position.
    pub fn new(id: u8, feet: CellPos) -> Self {
        Self {
            id,
            x: Fx::from_int(feet.x),
            y: Fx::from_int(feet.y),
            vx: Fx::ZERO,
            vy: Fx::ZERO,
            on_ground: false,
            in_liquid: false,
            hp: 100,
            facing: 1,
        }
    }

    /// The cell at the character's center.
    pub fn center(&self) -> CellPos {
        CellPos::new(self.x.floor(), self.y.floor() - AVATAR_H / 2)
    }

    /// The world cell an input aims at.
    pub fn aim_target(&self, input: &PlayerInput) -> CellPos {
        let c = self.center();
        CellPos::new(c.x + i32::from(input.aim_x), c.y + i32::from(input.aim_y))
    }

    /// Hash of every field.
    pub fn hash_into(&self, h: &mut StableHasher) {
        h.write_u32(u32::from(self.id));
        h.write_i32(self.x.0);
        h.write_i32(self.y.0);
        h.write_i32(self.vx.0);
        h.write_i32(self.vy.0);
        h.write_u32(u32::from(self.on_ground) | u32::from(self.in_liquid) << 1);
        h.write_u32(u32::from(self.hp));
        h.write_i32(i32::from(self.facing));
    }

    /// Advances the character by one tick. Reads the world, never writes it.
    pub fn step(&mut self, world: &World, input: &PlayerInput) {
        self.in_liquid = Self::submerged(world, self.x, self.y);
        let target = WALK * i32::from(input.move_x.signum());
        let accel = if self.on_ground { ACCEL_GROUND } else { ACCEL_AIR };
        self.vx = approach(self.vx, target, accel);
        if input.move_x != 0 {
            self.facing = input.move_x.signum();
        }
        if self.in_liquid {
            self.vy += Fx(GRAVITY.0 / 4);
            if input.jump {
                self.vy = Fx(self.vy.0.min(0)) + SWIM.mul_fx(Fx(1 << 14));
                if self.vy.0 < SWIM.0 {
                    self.vy = SWIM;
                }
            }
            // Drag: lose an eighth of the speed each tick.
            self.vx = Fx(self.vx.0 - self.vx.0 / 8);
            self.vy = Fx(self.vy.0 - self.vy.0 / 8);
        } else {
            if input.jump && self.on_ground {
                self.vy = JUMP;
            }
            self.vy = (self.vy + GRAVITY).clamp(Fx(-MAX_FALL.0), MAX_FALL);
        }
        self.move_x(world);
        self.move_y(world);
    }

    fn move_x(&mut self, world: &World) {
        let steps = (self.vx.abs().0 as u32).div_ceil(Fx::ONE.0 as u32).max(1) as i32;
        let dx = Fx(self.vx.0 / steps);
        for _ in 0..steps {
            let nx = self.x + dx;
            if !Self::blocked(world, nx, self.y) {
                self.x = nx;
                continue;
            }
            // Walk up small ledges.
            let climbed = (1..=STEP_UP).find(|&up| {
                let ny = self.y - Fx::from_int(up);
                self.on_ground && !Self::blocked(world, nx, ny)
            });
            if let Some(up) = climbed {
                self.x = nx;
                self.y -= Fx::from_int(up);
            } else {
                self.vx = Fx::ZERO;
                break;
            }
        }
    }

    fn move_y(&mut self, world: &World) {
        let steps = (self.vy.abs().0 as u32).div_ceil(Fx::ONE.0 as u32).max(1) as i32;
        let dy = Fx(self.vy.0 / steps);
        self.on_ground = false;
        for _ in 0..steps {
            let ny = self.y + dy;
            if Self::blocked(world, self.x, ny) {
                if dy.0 > 0 {
                    self.on_ground = true;
                    // Snap the feet to the top of the cell below.
                    self.y = Fx::from_int(ny.floor());
                    if Self::blocked(world, self.x, self.y) {
                        self.y -= Fx::ONE;
                    }
                }
                self.vy = Fx::ZERO;
                break;
            }
            self.y = ny;
        }
        if !self.on_ground && self.vy.0 >= 0 {
            self.on_ground = Self::blocked(world, self.x, self.y + Fx(1 << 12));
        }
    }

    /// Cells covered by the character's box with its feet at `(x, y)`.
    fn box_cells(x: Fx, y: Fx) -> (i32, i32, i32, i32) {
        let x0 = (x - Fx::from_ratio(AVATAR_W, 2)).floor();
        let y1 = (y - Fx(1)).floor();
        (x0, y1 - AVATAR_H + 1, x0 + AVATAR_W - 1, y1)
    }

    fn blocked(world: &World, x: Fx, y: Fx) -> bool {
        let (x0, y0, x1, y1) = Self::box_cells(x, y);
        for cy in y0..=y1 {
            for cx in x0..=x1 {
                let c = world.get(CellPos::new(cx, cy));
                if solid(world, c) {
                    return true;
                }
            }
        }
        false
    }

    /// True when at least a quarter of the character's box is liquid. A few
    /// drops of spray or rain in the air do not make it swim.
    fn submerged(world: &World, x: Fx, y: Fx) -> bool {
        let (x0, y0, x1, y1) = Self::box_cells(x, y);
        let mut wet = 0;
        for cy in y0..=y1 {
            for cx in x0..=x1 {
                let c = world.get(CellPos::new(cx, cy));
                if !c.is_air()
                    && !c.has(Cell::FLAG_IN_BODY)
                    && world.materials().get(c.material()).state == State::Liquid
                {
                    wet += 1;
                }
            }
        }
        wet * 4 >= (x1 - x0 + 1) * (y1 - y0 + 1)
    }
}

fn solid(world: &World, c: Cell) -> bool {
    if c.is_air() {
        return false;
    }
    if c.material() == MaterialId::BOUNDARY || c.has(Cell::FLAG_IN_BODY) {
        return true;
    }
    matches!(world.materials().get(c.material()).state, State::Static | State::Powder)
}

fn approach(v: Fx, target: Fx, step: Fx) -> Fx {
    if v.0 < target.0 { Fx((v.0 + step.0).min(target.0)) } else { Fx((v.0 - step.0).max(target.0)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fe_core::IRect;
    use std::sync::Arc;

    fn world() -> World {
        let mats = Arc::new(fe_materials::base_set());
        let stone = mats.id("stone").unwrap_or_default();
        let mut w = World::new(mats, 1, 1);
        w.ensure_rect(IRect::new(0, 0, 256, 128), Cell::AIR);
        w.fill_rect(IRect::new(0, 100, 256, 128), stone);
        w.fill_rect(IRect::new(150, 98, 256, 100), stone);
        w.fill_rect(IRect::new(200, 60, 210, 98), stone);
        w
    }

    #[test]
    fn falls_lands_walks_and_steps_up() {
        let w = world();
        let mut a = Avatar::new(0, CellPos::new(100, 50));
        let idle = PlayerInput::default();
        for _ in 0..90 {
            a.step(&w, &idle);
        }
        assert!(a.on_ground);
        assert_eq!(a.y.floor(), 100);
        let right = PlayerInput { move_x: 1, ..Default::default() };
        for _ in 0..140 {
            a.step(&w, &right);
        }
        // Climbed the 2-cell ledge at x 150 and stopped at the wall at x 200.
        assert_eq!(a.y.floor(), 98);
        assert!(a.x.floor() < 200 && a.x.floor() > 190, "x {}", a.x.floor());
    }

    #[test]
    fn jump_rises_and_lands() {
        let w = world();
        let mut a = Avatar::new(0, CellPos::new(50, 100));
        a.step(&w, &PlayerInput::default());
        assert!(a.on_ground);
        a.step(&w, &PlayerInput { jump: true, ..Default::default() });
        let mut peak = a.y.floor();
        for _ in 0..120 {
            a.step(&w, &PlayerInput::default());
            peak = peak.min(a.y.floor());
        }
        assert!(peak < 90, "peak {peak}");
        assert!(a.on_ground);
    }
}
