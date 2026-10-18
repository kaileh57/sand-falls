//! Light propagation, bloom, and the heat-haze field.

use fe_core::{Cell, temp_to_celsius};
use rayon::prelude::*;

/// Spreads light through the buffer: each cell takes the brighter of its own
/// light and its neighbor's light times the neighbor's transmission. Sweeping in
/// all four directions twice gives soft light that bends around corners and
/// fades into walls within a few cells.
pub(crate) fn propagate(light: &mut [[f32; 3]], trans: &[f32], w: usize, h: usize) {
    for _ in 0..2 {
        // Horizontal sweeps, one row per task.
        light.par_chunks_mut(w).zip(trans.par_chunks(w)).for_each(|(row, tr)| {
            for x in 1..w {
                let k = tr[x - 1];
                let p = row[x - 1];
                let c = &mut row[x];
                c[0] = c[0].max(p[0] * k);
                c[1] = c[1].max(p[1] * k);
                c[2] = c[2].max(p[2] * k);
            }
            for x in (0..w - 1).rev() {
                let k = tr[x + 1];
                let p = row[x + 1];
                let c = &mut row[x];
                c[0] = c[0].max(p[0] * k);
                c[1] = c[1].max(p[1] * k);
                c[2] = c[2].max(p[2] * k);
            }
        });
        // Vertical sweeps, in bands of columns so each task touches contiguous memory.
        for y in 1..h {
            let (above, below) = light.split_at_mut(y * w);
            let prev = &above[(y - 1) * w..];
            below[..w]
                .par_iter_mut()
                .zip(prev.par_iter())
                .zip(trans[(y - 1) * w..y * w].par_iter())
                .for_each(|((c, p), &k)| {
                    c[0] = c[0].max(p[0] * k);
                    c[1] = c[1].max(p[1] * k);
                    c[2] = c[2].max(p[2] * k);
                });
        }
        for y in (0..h - 1).rev() {
            let (above, below) = light.split_at_mut((y + 1) * w);
            let cur = &mut above[y * w..];
            cur.par_iter_mut()
                .zip(below[..w].par_iter())
                .zip(trans[(y + 1) * w..(y + 2) * w].par_iter())
                .for_each(|((c, p), &k)| {
                    c[0] = c[0].max(p[0] * k);
                    c[1] = c[1].max(p[1] * k);
                    c[2] = c[2].max(p[2] * k);
                });
        }
    }
}

pub(crate) fn box_blur(src: &[[f32; 3]], w: usize, h: usize, r: usize) -> Vec<[f32; 3]> {
    let mut tmp = vec![[0.0f32; 3]; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let s = &src[y * w..(y + 1) * w];
        let mut acc = [0.0f32; 3];
        let n = (2 * r + 1) as f32;
        for x in 0..=r.min(w - 1) {
            for c in 0..3 {
                acc[c] += s[x][c];
            }
        }
        for x in 0..w {
            row[x] = [acc[0] / n, acc[1] / n, acc[2] / n];
            if x + r + 1 < w {
                for c in 0..3 {
                    acc[c] += s[x + r + 1][c];
                }
            }
            if x >= r {
                for c in 0..3 {
                    acc[c] -= s[x - r][c];
                }
            }
        }
    });
    let mut out = vec![[0.0f32; 3]; w * h];
    let n = (2 * r + 1) as f32;
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r).min(h - 1);
        for (x, o) in row.iter_mut().enumerate() {
            let mut acc = [0.0f32; 3];
            for yy in y0..=y1 {
                let v = tmp[yy * w + x];
                acc[0] += v[0];
                acc[1] += v[1];
                acc[2] += v[2];
            }
            *o = [acc[0] / n, acc[1] / n, acc[2] / n];
        }
    });
    out
}

/// Blurs the light field slightly, which hides the diamond shapes that
/// max-based propagation leaves around point lights.
pub(crate) fn soften(light: &[[f32; 3]], w: usize, h: usize) -> Vec<[f32; 3]> {
    box_blur(&box_blur(light, w, h, 2), w, h, 2)
}

/// A wide blur, about six cells each way: thin gas into a cloud, and the light
/// that bounces into the rock around a lit space.
pub(crate) fn wide(src: &[[f32; 3]], w: usize, h: usize) -> Vec<[f32; 3]> {
    box_blur(&box_blur(src, w, h, 4), w, h, 4)
}

/// Adds a soft glow around bright areas.
pub(crate) fn bloom(hdr: &mut [[f32; 3]], w: usize, h: usize) {
    let bright: Vec<[f32; 3]> = hdr
        .iter()
        .map(|c| {
            let l = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
            if l > 0.9 {
                let k = (l - 0.9) / l;
                [c[0] * k, c[1] * k, c[2] * k]
            } else {
                [0.0; 3]
            }
        })
        .collect();
    let near = box_blur(&box_blur(&bright, w, h, 2), w, h, 2);
    let far = box_blur(&box_blur(&bright, w, h, 7), w, h, 7);
    hdr.par_iter_mut().zip(near.par_iter().zip(far.par_iter())).for_each(|(c, (n, f))| {
        c[0] += n[0] * 0.5 + f[0] * 0.45;
        c[1] += n[1] * 0.5 + f[1] * 0.45;
        c[2] += n[2] * 0.5 + f[2] * 0.45;
    });
}

/// How strongly each cell shimmers from hot air, 0 to 1: hot open air, fading
/// out with distance from the source of its heat (`source`, such as lava,
/// fire or molten metal) over about a dozen cells. Anything but open air is 0,
/// so the shimmer never moves an edge.
pub(crate) fn haze_field(
    cells: &[Cell],
    w: usize,
    h: usize,
    source: impl Fn(Cell) -> bool + Sync,
) -> Vec<f32> {
    let (hot, near): (Vec<[f32; 3]>, Vec<[f32; 3]>) = cells
        .par_iter()
        .map(|c| {
            let t = temp_to_celsius(c.temp);
            let v = if c.is_air() && t > 90 { ((t - 90) as f32 / 500.0).min(1.0) } else { 0.0 };
            ([v, 0.0, 0.0], [if source(*c) { 1.0 } else { 0.0 }, 0.0, 0.0])
        })
        .unzip();
    let hot = box_blur(&hot, w, h, 3);
    let near = box_blur(&box_blur(&near, w, h, 6), w, h, 6);
    hot.par_iter()
        .zip(near.par_iter())
        .zip(cells.par_iter())
        .map(|((v, n), c)| {
            if !c.is_air() {
                return 0.0;
            }
            // Within a few cells of a broad source `n` is near a half; a lone
            // hot cell far off adds little.
            let k = (n[0] * 3.0).min(1.0);
            (v[0] * k * k).min(1.0)
        })
        .collect()
}
