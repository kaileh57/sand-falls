//! Deterministic world generation from a seed, using integer value noise.

use fe_core::{CHUNK_AREA, CHUNK_SIZE, Cell, ChunkPos, MaterialId, mix64};
use fe_materials::MaterialSet;
use serde::{Deserialize, Serialize};

/// How new chunks are filled when they are first loaded.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorldGen {
    /// Every new chunk is air.
    Empty,
    /// Air above `ground`, then soil, stone, and bedrock at `bedrock` and below.
    Flat {
        /// First solid row.
        ground: i32,
        /// First bedrock row.
        bedrock: i32,
    },
    /// Caves carved from noise, with water and lava pockets, ore, and glowing fungus.
    Caves {
        /// First row of terrain. Above it is open sky.
        surface: i32,
    },
}

/// Smooth 2D value noise in `0..=65535`, from lattice values every `scale` cells.
fn value_noise(seed: u64, x: i32, y: i32, scale: i32) -> i32 {
    stretched_noise(seed, x, y, scale, scale)
}

/// Value noise with lattice values every `sx` cells across and `sy` down, for
/// layered features such as coal seams.
fn stretched_noise(seed: u64, x: i32, y: i32, sx: i32, sy: i32) -> i32 {
    let gx = x.div_euclid(sx);
    let gy = y.div_euclid(sy);
    let fx = x.rem_euclid(sx) * 256 / sx;
    let fy = y.rem_euclid(sy) * 256 / sy;
    let lattice = |ix: i32, iy: i32| -> i32 {
        let k = mix64(seed ^ (u64::from(ix as u32) << 32 | u64::from(iy as u32)));
        (k >> 48) as i32
    };
    // Smoothstep on 0..256: t * t * (3 - 2t).
    let smooth = |t: i32| t * t * (768 - 2 * t) / 65_536;
    let (sx, sy) = (smooth(fx), smooth(fy));
    let a = lattice(gx, gy);
    let b = lattice(gx + 1, gy);
    let c = lattice(gx, gy + 1);
    let d = lattice(gx + 1, gy + 1);
    let top = a + (b - a) * sx / 256;
    let bottom = c + (d - c) * sx / 256;
    top + (bottom - top) * sy / 256
}

/// Three octaves of value noise, in `0..=65535`.
fn fbm(seed: u64, x: i32, y: i32) -> i32 {
    (value_noise(seed, x, y, 96) * 4
        + value_noise(seed ^ 0xa1, x, y, 40) * 2
        + value_noise(seed ^ 0xb2, x, y, 16))
        / 7
}

struct Palette {
    stone: MaterialId,
    soil: MaterialId,
    bedrock: MaterialId,
    water: MaterialId,
    lava: MaterialId,
    coal: MaterialId,
    glowshroom: MaterialId,
    moss: MaterialId,
    crystal: MaterialId,
    sand: MaterialId,
}

impl Palette {
    fn new(m: &MaterialSet) -> Self {
        let id = |n: &str| m.id(n).unwrap_or(MaterialId::BOUNDARY);
        Self {
            stone: id("stone"),
            soil: id("soil"),
            bedrock: id("bedrock"),
            water: id("water"),
            lava: id("lava"),
            coal: id("coal"),
            glowshroom: id("glowshroom"),
            moss: id("moss"),
            crystal: id("crystal"),
            sand: id("sand"),
        }
    }
}

impl WorldGen {
    /// Generates one chunk's cells, row-major.
    pub fn generate(&self, mats: &MaterialSet, seed: u64, pos: ChunkPos) -> Vec<Cell> {
        let origin = pos.origin();
        let mut out = vec![Cell::AIR; CHUNK_AREA];
        let make = |id: MaterialId, x: i32, y: i32| {
            fe_sim::new_cell(mats, id, None, fe_core::cell_rand(seed, 0, x, y, 0x6e))
        };
        match *self {
            Self::Empty => {}
            Self::Flat { ground, bedrock } => {
                let p = Palette::new(mats);
                for ly in 0..CHUNK_SIZE {
                    for lx in 0..CHUNK_SIZE {
                        let (x, y) = (origin.x + lx, origin.y + ly);
                        let id = if y >= bedrock {
                            p.bedrock
                        } else if y >= ground + 12 {
                            p.stone
                        } else if y >= ground {
                            p.soil
                        } else {
                            continue;
                        };
                        out[(ly * CHUNK_SIZE + lx) as usize] = make(id, x, y);
                    }
                }
            }
            Self::Caves { surface } => {
                let p = Palette::new(mats);
                for ly in 0..CHUNK_SIZE {
                    for lx in 0..CHUNK_SIZE {
                        let (x, y) = (origin.x + lx, origin.y + ly);
                        if let Some((id, temp)) = cave_cell(&p, seed, surface, x, y) {
                            out[(ly * CHUNK_SIZE + lx) as usize] = fe_sim::new_cell(
                                mats,
                                id,
                                temp,
                                fe_core::cell_rand(seed, 0, x, y, 0x6e),
                            );
                        }
                    }
                }
            }
        }
        out
    }
}

/// Rock below the rolling surface line deeper than this (give or take a few
/// wavy cells) is bedrock.
const BEDROCK_DEPTH: i32 = 600;
/// Lava pools in caves deeper than this; shallower caves hold water.
const LAVA_DEPTH: i32 = 260;

/// Noise level above which rock is carved out. Lower with depth, so caves
/// grow, then higher again over the last rows above bedrock, so caves close
/// up before it and never cut into the bedrock band.
fn cave_threshold(depth: i32) -> i32 {
    let close = (depth - (BEDROCK_DEPTH - 60)).max(0) * 500;
    // Caves rarely break into the soil crust.
    let crust = (30 - depth).max(0) * 900;
    36_500 - depth.clamp(0, 400) * 6 + close + crust
}

/// A slow wobble in `-amp..=amp` cells, for wavy band boundaries.
fn wobble(seed: u64, x: i32, y: i32, amp: i32) -> i32 {
    (value_noise(seed, x, y, 40) - 32_768) * amp / 32_768
}

/// Height of the rolling surface line at column `x`.
fn hill(seed: u64, surface: i32, x: i32) -> i32 {
    surface + (value_noise(seed ^ 0x51, x, 0, 128) - 32_768) / 700
}

/// Whether noise alone carves the cell open (before small islands are removed).
fn carved(seed: u64, depth: i32, x: i32, y: i32) -> bool {
    depth > 10 && fbm(seed, x, y) > cave_threshold(depth)
}

/// A solid cell that is part of a small lump of rock floating in cave air:
/// open cave in every direction a short way off. Such lumps are carved away.
fn floating_island(seed: u64, depth: i32, x: i32, y: i32) -> bool {
    const DIRS: [(i32, i32); 8] =
        [(7, 0), (5, 5), (0, 7), (-5, 5), (-7, 0), (-5, -5), (0, -7), (5, -5)];
    DIRS.iter().all(|&(dx, dy)| carved(seed, depth + dy, x + dx, y + dy))
}

impl WorldGen {
    /// The rock band a cell of generated stone lies in, `0` near the surface
    /// to `3` deep down, with slightly wavy boundaries; `None` for worlds
    /// without depth bands. The renderer tints stone by band. Presentation
    /// only: the simulation never reads it.
    pub fn rock_band(&self, seed: u64, x: i32, y: i32) -> Option<u8> {
        let Self::Caves { surface } = *self else { return None };
        let depth = y - hill(seed, surface, x) + wobble(seed ^ 0x5a, x, y, 14);
        Some(match depth {
            ..130 => 0,
            130..280 => 1,
            280..440 => 2,
            _ => 3,
        })
    }
}

fn cave_cell(
    p: &Palette,
    seed: u64,
    surface: i32,
    x: i32,
    y: i32,
) -> Option<(MaterialId, Option<u8>)> {
    let hill = hill(seed, surface, x);
    if y < hill {
        return None;
    }
    let depth = y - hill;
    // Bedrock: a wavy band that caves never reach.
    if depth > BEDROCK_DEPTH + wobble(seed ^ 0xbe, x, y, 6) {
        return Some((p.bedrock, None));
    }
    let solid = |id| Some((id, None));
    // Caves open up with depth; the first rows are solid ground.
    let n = fbm(seed, x, y);
    let threshold = cave_threshold(depth);
    let open = depth > 10
        && (n > threshold
            // Lumps of rock too small to have stayed up are carved away too.
            || (n > threshold - 9_000 && floating_island(seed, depth, x, y)));
    let rnd = (mix64(seed ^ 0x77 ^ (u64::from(x as u32) << 32 | u64::from(y as u32))) >> 40) as i32;
    // Deep caves are hot: lava pools start as hot as they can without melting
    // the stone they lie in, and the rock around a pocket starts warm, so the
    // pockets lose heat slowly and stay molten.
    let pool = value_noise(seed ^ 0x33, x, y, 64);
    let deep = depth > LAVA_DEPTH;
    if open {
        // Liquids pool in the bottom of caves: deep caves hold lava, shallow ones water.
        if fbm(seed, x, y + 3) <= cave_threshold(depth) && pool > 52_000 {
            return Some(if deep {
                (p.lava, Some(fe_core::celsius_to_temp(1_180)))
            } else {
                (p.water, None)
            });
        }
        return None;
    }
    let warm = if deep && pool > 48_000 {
        // Up to about 550 C at the pocket's edge, cooling outward: warm, but
        // short of glowing.
        Some(fe_core::celsius_to_temp(100 + (pool - 48_000) / 9))
    } else {
        None
    };
    // Decorate cave floors and ceilings.
    let above_open = depth > 11 && carved(seed, depth - 1, x, y - 1);
    let below_open = depth > 9 && carved(seed, depth + 1, x, y + 1);
    if above_open && rnd % 9 == 0 {
        return solid(if depth > 120 { p.glowshroom } else { p.moss });
    }
    if below_open && rnd % 23 == 0 && depth > 60 {
        return solid(p.crystal);
    }
    // A thick crust of soil with a wavy underside, sand lenses near the top.
    let soil_floor = 22 + wobble(seed ^ 0x50, x, y, 6);
    if depth < 4 {
        return solid(p.soil);
    }
    if depth < 12 && value_noise(seed ^ 0x99, x, y, 24) > 50_000 {
        return solid(p.sand);
    }
    if depth < soil_floor {
        return solid(p.soil);
    }
    // Coal in long, thin seams: noise stretched sideways, gently dipping.
    if depth > soil_floor + 6
        && stretched_noise(seed ^ 0xc0, x, y + x / 7, 56, 7) > 56_000
        && value_noise(seed ^ 0xc1, x, y, 80) > 26_000
    {
        return solid(p.coal);
    }
    Some((p.stone, warm))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_in_range_and_continuous() {
        let mut prev = value_noise(1, 0, 0, 32);
        for x in 1..500 {
            let v = value_noise(1, x, 7, 32);
            assert!((0..=65_535).contains(&v));
            assert!((v - prev).abs() < 4000, "jump at {x}: {prev} -> {v}");
            prev = v;
        }
    }

    #[test]
    fn generation_is_deterministic_and_varied() {
        let mats = fe_materials::base_set();
        let g = WorldGen::Caves { surface: 0 };
        let a = g.generate(&mats, 42, ChunkPos::new(3, 4));
        let b = g.generate(&mats, 42, ChunkPos::new(3, 4));
        assert_eq!(a, b);
        let air = a.iter().filter(|c| c.is_air()).count();
        assert!(air > 100 && air < CHUNK_AREA - 100, "air {air}");
        let sky = g.generate(&mats, 42, ChunkPos::new(0, -3));
        assert!(sky.iter().all(|c| c.is_air()));
    }
}
