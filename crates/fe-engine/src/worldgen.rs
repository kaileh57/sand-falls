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
    let gx = x.div_euclid(scale);
    let gy = y.div_euclid(scale);
    let fx = x.rem_euclid(scale) * 256 / scale;
    let fy = y.rem_euclid(scale) * 256 / scale;
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
                        if let Some(id) = cave_cell(&p, seed, surface, x, y) {
                            out[(ly * CHUNK_SIZE + lx) as usize] = make(id, x, y);
                        }
                    }
                }
            }
        }
        out
    }
}

/// Noise level above which rock is carved out. Lower with depth, so caves grow.
fn cave_threshold(depth: i32) -> i32 {
    36_500 - depth.clamp(0, 400) * 6
}

fn cave_cell(p: &Palette, seed: u64, surface: i32, x: i32, y: i32) -> Option<MaterialId> {
    // Rolling surface line.
    let hill = surface + (value_noise(seed ^ 0x51, x, 0, 128) - 32_768) / 700;
    if y < hill {
        return None;
    }
    let depth = y - hill;
    let n = fbm(seed, x, y);
    // Caves open up with depth; the first rows are solid ground.
    let open = depth > 10 && n > cave_threshold(depth);
    let rnd = (mix64(seed ^ 0x77 ^ (u64::from(x as u32) << 32 | u64::from(y as u32))) >> 40) as i32;
    if open {
        // Liquids pool in the bottom of caves: deep caves hold lava, shallow ones water.
        let pool = value_noise(seed ^ 0x33, x, y, 64);
        if fbm(seed, x, y + 3) <= cave_threshold(depth) && pool > 52_000 {
            return Some(if depth > 260 { p.lava } else { p.water });
        }
        return None;
    }
    // Decorate cave floors and ceilings.
    let above_open = fbm(seed, x, y - 1) > cave_threshold(depth - 1) && depth > 11;
    let below_open = fbm(seed, x, y + 1) > cave_threshold(depth + 1) && depth > 9;
    if above_open && rnd % 9 == 0 {
        return Some(if depth > 120 { p.glowshroom } else { p.moss });
    }
    if below_open && rnd % 23 == 0 && depth > 60 {
        return Some(p.crystal);
    }
    if depth < 4 {
        return Some(p.soil);
    }
    if depth < 10 && value_noise(seed ^ 0x99, x, y, 24) > 50_000 {
        return Some(p.sand);
    }
    if value_noise(seed ^ 0xc0, x, y, 12) > 60_000 {
        return Some(p.coal);
    }
    if depth > 600 {
        return Some(p.bedrock);
    }
    Some(if depth < 14 { p.soil } else { p.stone })
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
