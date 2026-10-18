//! Liquid in the air, drawn as soft sheets and drops.
//!
//! A falling stream is simulated as whole cells spaced as far apart as they
//! move in a tick, so a still frame of the cells alone shows a dotted line,
//! and joining them with lines one cell wide drew forks like lightning. Here
//! each cell of liquid in the air is instead splatted as a soft blob drawn
//! out along its path: back to the liquid behind it when it is part of a
//! stream (see `fe_sim::liquid::air_links`), and by a short motion blur in
//! any case. The blobs add up into a coverage field, and a smooth threshold
//! on that field turns neighbouring drops of a stream into one sheet with
//! soft edges while a lone drop stays a small round drop.

use fe_core::Cell;
use fe_sim::liquid::{AirLink, air_velocity};

/// Spread of a blob across its path, in cells.
const SIGMA: f32 = 0.72;
/// Coverage below which nothing is drawn, and above which the sheet is solid.
const EDGE: (f32, f32) = (0.2, 0.62);
/// Share of the wide blur of the coverage added to it.
const FILL: f32 = 1.3;
/// Wide coverage over which the fill fades in.
const FILL_FROM: (f32, f32) = (0.1, 0.32);
/// Longest join along a stream that is drawn, in cells of travel.
const MAX_JOIN: f32 = 24.0;
/// Cells of travel over which a long join thins out.
const TAPER: f32 = 6.0;
/// Ticks of motion blur.
const BLUR_TICKS: f32 = 2.5;
/// Cells of travel over which a long join thins out behind a drop in a wide
/// falling mass, and over which the gap above a slab of it is filled.
pub(crate) const TAPER_WIDE: f32 = 18.0;
/// Longest gap above a slab of a falling mass that is filled back to the
/// liquid above it, in cells.
pub(crate) const GAP_FILL: usize = 16;

/// A soft blob of liquid added to the sheet: liquid thrown up from a pool,
/// drawn as a plume and droplets instead of a hard needle, or the pour
/// between two slabs of a falling mass.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Splash {
    /// Centre, in buffer cells.
    pub x: f32,
    pub y: f32,
    /// Peak weight of the blob.
    pub weight: f32,
    /// The cell it comes from, by index, for its look.
    pub index: usize,
}

/// Liquid in the air over a window of cells.
pub(crate) struct Sheet {
    /// Summed blob weight per cell.
    pub cov: Vec<f32>,
    /// Albedo summed by weight.
    pub albedo: Vec<[f32; 3]>,
    /// Emission summed by weight.
    pub emit: Vec<[f32; 3]>,
    /// How much of the weight is water, summed by weight.
    pub water: Vec<f32>,
    /// True if any cell is covered.
    pub any: bool,
}

/// Where a cell of liquid in the air moving at `(vx, vy)` (1/16 cells per
/// tick) was `k` cells of travel along its faster axis ago, relative to where
/// it is now: back along the arc gravity bent its velocity along, as
/// `fe_sim::liquid::drawn` traces it.
fn path_back(vx: i32, vy: i32, k: f32) -> (f32, f32) {
    let m = vx.abs().max(vy).max(1) as f32;
    (-(vx as f32) * k / m, -((2.0 * vy as f32 * k * m - 7.0 * k * k) / (2.0 * m * m)).max(0.0))
}

/// Smooth step from `e0` to `e1`.
pub(crate) fn smooth(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl Sheet {
    /// Opacity of the sheet for a coverage.
    pub fn alpha(cov: f32) -> f32 {
        smooth(EDGE.0, EDGE.1, cov)
    }

    /// Builds the sheet from the cells in the air (`in_air`, by index) and
    /// the splashes thrown up from pools (`splash`), `look` giving each one's
    /// albedo, emission and whether it is water. It covers only open cells
    /// (`open`), the cells in the air and the cells the splashes were taken
    /// from (`freed`): liquid in the air is never in front of a wall or a pool.
    #[expect(clippy::too_many_arguments, reason = "internal builder over shared buffers")]
    pub fn build(
        cells: &[Cell],
        w: usize,
        h: usize,
        in_air: impl Fn(usize) -> bool,
        freed: impl Fn(usize) -> bool,
        open: impl Fn(Cell) -> bool,
        links: &[AirLink],
        splash: &[Splash],
        look: impl Fn(usize, Cell) -> ([f32; 3], [f32; 3], bool),
    ) -> Self {
        let n = w * h;
        let mut s = Self {
            cov: vec![0.0; n],
            albedo: vec![[0.0; 3]; n],
            emit: vec![[0.0; 3]; n],
            water: vec![0.0; n],
            any: false,
        };
        let mut back = vec![(0i32, false); n];
        for l in links {
            back[l.index] = (l.back, l.stream);
        }
        let norm = SIGMA * (2.0 * std::f32::consts::PI).sqrt();
        let inv = 1.0 / (2.0 * SIGMA * SIGMA);
        for (i, &c) in cells.iter().enumerate() {
            if !in_air(i) {
                continue;
            }
            s.any = true;
            let (x0, y0) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
            let (vx, vy) = air_velocity(c);
            let m = vx.abs().max(vy).max(1) as f32;
            let speed = ((vx * vx + vy * vy) as f32).sqrt() / 16.0;
            // Cells of travel along the faster axis per cell of path.
            let stretch = speed * 16.0 / m;
            let (join, stream) = back[i];
            // Motion blur: a short streak behind a fast drop, longer along a
            // stream than for a lone drop, so spray stays drops.
            let blur = (m / 16.0 * BLUR_TICKS).min(if stream { 10.0 } else { 5.0 });
            // In a wide falling mass (liquid in the air on both sides in its
            // row) the join behind a drop thins out slowly.
            let row = |dx: i32| {
                let xx = (i % w) as i32 + dx;
                xx >= 0 && (xx as usize) < w && in_air(i - (i % w) + xx as usize)
            };
            let wide = (row(-2) || row(-1)) && (row(1) || row(2)) && (row(-1) || row(1));
            let taper_len = if wide { TAPER_WIDE } else { TAPER };
            let reach = if stream { (join as f32).min(MAX_JOIN).max(blur) } else { blur };
            let len = reach * stretch;
            let steps = (len * 2.0).ceil().max(0.0) as usize;
            let pts = steps + 1;
            // Each point's weight: a line sums to about 1 across its middle,
            // a single blob peaks at 1.
            let wt = (len / norm).max(1.0) / pts as f32;
            let (a, e, water) = look(i, c);
            for p in 0..pts {
                let k = if steps == 0 { 0.0 } else { reach * p as f32 / steps as f32 };
                // A long join thins out behind the drop, so a sparse stream
                // reads as tapering streaks, not as bars.
                let taper = if k > blur { (-(k - blur) / taper_len).exp() } else { 1.0 };
                let (dx, dy) = path_back(vx, vy, k);
                let (px, py) = (x0 + dx, y0 + dy);
                let (cx, cy) = (px.floor() as i32, py.floor() as i32);
                // The trail stops where the path leaves open air (steam and
                // smoke are open), at the ledge, wall or pool the liquid
                // came from.
                if p > 0
                    && (cx < 0 || cy < 0 || cx as usize >= w || cy as usize >= h || {
                        let j = cy as usize * w + cx as usize;
                        !open(cells[j]) && !in_air(j)
                    })
                {
                    break;
                }
                for yy in cy - 2..=cy + 2 {
                    if yy < 0 || yy as usize >= h {
                        continue;
                    }
                    let fy = yy as f32 + 0.5 - py;
                    for xx in cx - 2..=cx + 2 {
                        if xx < 0 || xx as usize >= w {
                            continue;
                        }
                        let fx = xx as f32 + 0.5 - px;
                        let d2 = fx * fx + fy * fy;
                        if d2 > 4.0 {
                            continue;
                        }
                        let v = wt * taper * (-d2 * inv).exp();
                        let j = yy as usize * w + xx as usize;
                        s.cov[j] += v;
                        let al = &mut s.albedo[j];
                        *al = [al[0] + a[0] * v, al[1] + a[1] * v, al[2] + a[2] * v];
                        let em = &mut s.emit[j];
                        *em = [em[0] + e[0] * v, em[1] + e[1] * v, em[2] + e[2] * v];
                        if water {
                            s.water[j] += v;
                        }
                    }
                }
            }
        }
        for b in splash {
            s.any = true;
            let (a, e, water) = look(b.index, cells[b.index]);
            let (cx, cy) = (b.x.floor() as i32, b.y.floor() as i32);
            for yy in cy - 2..=cy + 2 {
                for xx in cx - 2..=cx + 2 {
                    if xx < 0 || yy < 0 || xx as usize >= w || yy as usize >= h {
                        continue;
                    }
                    let (fx, fy) = (xx as f32 + 0.5 - b.x, yy as f32 + 0.5 - b.y);
                    let d2 = fx * fx + fy * fy;
                    if d2 > 4.0 {
                        continue;
                    }
                    let v = b.weight * (-d2 * inv).exp();
                    let j = yy as usize * w + xx as usize;
                    s.cov[j] += v;
                    let al = &mut s.albedo[j];
                    *al = [al[0] + a[0] * v, al[1] + a[1] * v, al[2] + a[2] * v];
                    let em = &mut s.emit[j];
                    *em = [em[0] + e[0] * v, em[1] + e[1] * v, em[2] + e[2] * v];
                    if water {
                        s.water[j] += v;
                    }
                }
            }
        }
        if s.any {
            // A thin wide share of each blob fills the spaces between the
            // drops of a curtain, which then reads as one sheet; a lone drop
            // adds too little to show.
            let pack: Vec<[f32; 3]> =
                s.cov.iter().zip(&s.water).map(|(&c, &w)| [c, w, 0.0]).collect();
            let wide = |v: &[[f32; 3]]| {
                crate::light::box_blur(&crate::light::box_blur(v, w, h, 3), w, h, 3)
            };
            let (pack, albedo, emit) = (wide(&pack), wide(&s.albedo), wide(&s.emit));
            for i in 0..n {
                if !open(cells[i]) && !in_air(i) && !freed(i) {
                    s.cov[i] = 0.0;
                    continue;
                }
                // Only where the drops are dense: a lone streak gets no halo.
                let b = pack[i][0];
                if b < 1e-4 {
                    continue;
                }
                let f = FILL * smooth(FILL_FROM.0, FILL_FROM.1, b);
                s.cov[i] += b * f;
                s.water[i] += pack[i][1] * f;
                let (a, e) = (&mut s.albedo[i], &mut s.emit[i]);
                for k in 0..3 {
                    a[k] += albedo[i][k] * f;
                    e[k] += emit[i][k] * f;
                }
            }
        }
        s
    }

    /// The sheet's opacity, albedo, emission and water share at a cell.
    pub fn at(&self, i: usize) -> Option<(f32, [f32; 3], [f32; 3], f32)> {
        let c = self.cov[i];
        if c < 1e-3 {
            return None;
        }
        let a = self.albedo[i];
        let e = self.emit[i];
        Some((
            Self::alpha(c),
            [a[0] / c, a[1] / c, a[2] / c],
            [e[0] / c, e[1] / c, e[2] / c],
            self.water[i] / c,
        ))
    }
}

/// Smooth value noise in two dimensions, 0 to 1.
pub(crate) fn noise(x: f32, y: f32, salt: i32) -> f32 {
    let cell = |gx: i32, gy: i32| {
        (fe_core::mix64((u64::from((gx ^ salt) as u32) << 32) | u64::from(gy as u32)) >> 40) as f32
            / (1u64 << 24) as f32
    };
    let (gx, gy) = (x.floor(), y.floor());
    let (tx, ty) = (x - gx, y - gy);
    let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
    let (gx, gy) = (gx as i32, gy as i32);
    let top = cell(gx, gy) + (cell(gx + 1, gy) - cell(gx, gy)) * sx;
    let bottom = cell(gx, gy + 1) + (cell(gx + 1, gy + 1) - cell(gx, gy + 1)) * sx;
    top + (bottom - top) * sy
}
