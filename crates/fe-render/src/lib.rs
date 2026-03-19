//! Renderer: turns a simulation into lit images.
//!
//! Rendering happens at cell resolution on the CPU, then scales up to pixels.
//! It reads the simulation and never writes to it, so it may use floats and any
//! number of threads without affecting determinism.
//!
//! The pipeline per frame:
//!
//! 1. sample cells, particles and characters into albedo, emission and opacity buffers
//! 2. propagate light from emissive and hot cells (and the sky) through open space
//! 3. compose: lit surfaces, glowing cells, light scattered by smoke and steam
//! 4. bloom bright areas, tone map, and scale up with heat haze over hot air
#![forbid(unsafe_code)]

mod font;
mod image;
mod light;

pub use image::Image;

use fe_core::{CHUNK_SIZE, Cell, CellPos, ChunkPos, MaterialId, temp_to_celsius};
use fe_engine::Simulation;
use fe_materials::{MaterialSet, State};
use rayon::prelude::*;

/// Background and sky light.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sky {
    /// Bright daylight from above.
    Day,
    /// Warm low light.
    Dusk,
    /// Dim blue light.
    Night,
    /// No sky; only emissive light.
    Cave,
}

impl Sky {
    /// Parses `day`, `dusk`, `night`, or `cave`.
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "day" => Self::Day,
            "dusk" => Self::Dusk,
            "night" => Self::Night,
            "cave" => Self::Cave,
            _ => return None,
        })
    }

    fn light(self) -> [f32; 3] {
        match self {
            Self::Day => [1.25, 1.2, 1.1],
            Self::Dusk => [1.1, 0.55, 0.3],
            Self::Night => [0.12, 0.16, 0.3],
            Self::Cave => [0.0; 3],
        }
    }

    fn background(self, t: f32) -> [f32; 3] {
        let (top, bottom) = match self {
            Self::Day => ([0.22, 0.42, 0.85], [0.62, 0.76, 0.92]),
            Self::Dusk => ([0.12, 0.08, 0.22], [0.85, 0.38, 0.18]),
            Self::Night => ([0.004, 0.006, 0.02], [0.02, 0.03, 0.07]),
            Self::Cave => ([0.008, 0.008, 0.012], [0.012, 0.01, 0.012]),
        };
        lerp3(top, bottom, t)
    }
}

/// How a frame is rendered.
#[derive(Clone, Debug)]
pub struct RenderSettings {
    /// View width in cells at zoom 1.
    pub view_w: u32,
    /// View height in cells at zoom 1.
    pub view_h: u32,
    /// Output pixels per cell at zoom 1.
    pub scale: u32,
    /// Sky.
    pub sky: Sky,
    /// Light every surface gets regardless of light sources.
    pub ambient: [f32; 3],
    /// Exposure for tone mapping.
    pub exposure: f32,
    /// Light propagation on or off.
    pub lighting: bool,
    /// Bloom on or off.
    pub bloom: bool,
    /// Heat haze on or off.
    pub haze: bool,
    /// Draw chunk borders and dirty rects.
    pub overlay: bool,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            view_w: 480,
            view_h: 270,
            scale: 2,
            sky: Sky::Cave,
            ambient: [0.05, 0.05, 0.06],
            exposure: 1.4,
            lighting: true,
            bloom: true,
            haze: true,
            overlay: false,
        }
    }
}

/// Where the camera looks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    /// Center x in cells.
    pub cx: f32,
    /// Center y in cells.
    pub cy: f32,
    /// Zoom; 2 shows half as many cells, each twice as large.
    pub zoom: f32,
}

#[derive(Clone)]
struct MatLook {
    colors: Vec<[f32; 3]>,
    light: [f32; 3],
    opacity: f32,
    state: State,
    glow: bool,
    structural: bool,
}

/// Cached per-material render data.
pub struct Renderer {
    looks: Vec<MatLook>,
}

const MARGIN: i32 = 40;

fn srgb_to_linear(c: u8) -> f32 {
    let c = f32::from(c) / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

fn linear_to_srgb(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

/// Incandescent color for a temperature in degrees C, zero below 480 C.
fn blackbody(celsius: i32) -> [f32; 3] {
    if celsius < 480 {
        return [0.0; 3];
    }
    let t = (celsius - 480) as f32;
    let i = (t / 400.0).min(3.0);
    let r = 1.0f32.min(0.25 + t / 500.0);
    let g = (t / 1100.0).powf(1.4).min(0.95);
    let b = ((t - 700.0).max(0.0) / 1600.0).min(0.85);
    [r * i, g * i, b * i]
}

fn player_color(id: u8) -> [f32; 3] {
    const C: [[f32; 3]; 8] = [
        [0.9, 0.2, 0.15],
        [0.15, 0.45, 0.95],
        [0.2, 0.8, 0.3],
        [0.95, 0.75, 0.1],
        [0.7, 0.25, 0.85],
        [0.1, 0.8, 0.8],
        [0.95, 0.45, 0.1],
        [0.85, 0.85, 0.9],
    ];
    C[id as usize % 8]
}

fn hash2(x: i32, y: i32) -> u32 {
    let h = fe_core::mix64((u64::from(x as u32) << 32) | u64::from(y as u32));
    (h >> 32) as u32
}

impl Renderer {
    /// Builds render data for a material set.
    pub fn new(mats: &MaterialSet) -> Self {
        let looks = mats
            .iter()
            .map(|(_, m)| MatLook {
                colors: m
                    .colors
                    .iter()
                    .map(|c| [srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2])])
                    .collect(),
                light: [
                    srgb_to_linear(m.light[0]) * 3.0,
                    srgb_to_linear(m.light[1]) * 3.0,
                    srgb_to_linear(m.light[2]) * 3.0,
                ],
                opacity: f32::from(m.opacity) / 255.0,
                state: m.state,
                glow: m.glow,
                structural: m.structural || m.anchored,
            })
            .collect();
        Self { looks }
    }

    fn look(&self, id: MaterialId) -> &MatLook {
        self.looks.get(id.index()).unwrap_or(&self.looks[1])
    }

    /// Renders one frame. `frame` animates haze and liquid shimmer.
    pub fn render(&self, sim: &Simulation, view: View, s: &RenderSettings, frame: u64) -> Image {
        self.render_with(sim, view, s, frame, &sim.avatars)
    }

    /// Renders one frame, drawing `avatars` instead of the simulation's own
    /// characters. Networked clients pass their predicted character here.
    pub fn render_with(
        &self,
        sim: &Simulation,
        view: View,
        s: &RenderSettings,
        frame: u64,
        avatars: &[fe_sim::Avatar],
    ) -> Image {
        let zoom = view.zoom.max(0.05);
        let vis_w = s.view_w as f32 / zoom;
        let vis_h = s.view_h as f32 / zoom;
        let x0 = (view.cx - vis_w / 2.0).floor() as i32;
        let y0 = (view.cy - vis_h / 2.0).floor() as i32;
        let cw = vis_w.ceil() as i32 + 2;
        let ch = vis_h.ceil() as i32 + 2;
        // Buffer region including the light margin.
        let bx = x0 - MARGIN;
        let by = y0 - MARGIN;
        let bw = (cw + 2 * MARGIN) as usize;
        let bh = (ch + 2 * MARGIN) as usize;

        let cells = sample_cells(sim, bx, by, bw, bh);
        let mut buf = Buffers::new(bw, bh);
        self.fill_buffers(sim, avatars, &cells, &mut buf, bx, by, frame, s);
        if s.lighting {
            light::propagate(&mut buf.light, &buf.trans, bw, bh);
            buf.light = light::soften(&buf.light, bw, bh);
            buf.shaft = light::soften(&buf.shaft, bw, bh);
        }
        // How open each cell's surroundings are: 1 deep in air, 0 deep in rock.
        let open_mask: Vec<[f32; 3]> =
            buf.trans.iter().map(|&t| if t > 0.9 { [1.0, 0.0, 0.0] } else { [0.0; 3] }).collect();
        buf.open = light::soften(&light::soften(&open_mask, bw, bh), bw, bh)
            .into_iter()
            .map(|v| v[0])
            .collect();
        let mut hdr = self.compose(&cells, &buf, bw, bh, bx, by, s);
        if s.bloom {
            light::bloom(&mut hdr, bw, bh);
        }
        let haze = if s.haze { light::haze_field(&cells, bw, bh) } else { Vec::new() };

        let out_w = s.view_w * s.scale;
        let out_h = s.view_h * s.scale;
        let mut img = Image::new(out_w, out_h);
        let px_per_cell = s.scale as f32 * zoom;
        let fx0 = view.cx - vis_w / 2.0;
        let fy0 = view.cy - vis_h / 2.0;
        let exposure = s.exposure;
        img.rgba.par_chunks_mut(out_w as usize * 4).enumerate().for_each(|(py, row)| {
            let wy = fy0 + (py as f32 + 0.5) / px_per_cell;
            let cy = (wy.floor() as i32 - by) as usize;
            for px in 0..out_w as usize {
                let wx = fx0 + (px as f32 + 0.5) / px_per_cell;
                let mut cx = wx.floor() as i32 - bx;
                if !haze.is_empty() && cy < bh && (cx as usize) < bw {
                    let hz = haze[cy * bw + cx as usize];
                    if hz > 0.02 {
                        let phase = py as f32 * 0.21 / zoom.max(0.5) + frame as f32 * 0.31;
                        cx += (phase.sin() * hz * 1.6).round() as i32;
                    }
                }
                let i = cy.min(bh - 1) * bw + (cx.max(0) as usize).min(bw - 1);
                let c = hdr[i];
                let o = px * 4;
                row[o] = linear_to_srgb(1.0 - (-c[0] * exposure).exp());
                row[o + 1] = linear_to_srgb(1.0 - (-c[1] * exposure).exp());
                row[o + 2] = linear_to_srgb(1.0 - (-c[2] * exposure).exp());
                row[o + 3] = 255;
            }
        });
        if s.overlay {
            draw_overlay(sim, &mut img, fx0, fy0, px_per_cell);
        }
        img
    }

    #[expect(clippy::too_many_arguments, reason = "internal helper over shared buffers")]
    fn fill_buffers(
        &self,
        sim: &Simulation,
        avatars: &[fe_sim::Avatar],
        cells: &[Cell],
        buf: &mut Buffers,
        bx: i32,
        by: i32,
        frame: u64,
        s: &RenderSettings,
    ) {
        let bw = buf.w;
        let bh = buf.h;
        let sky = s.sky.light();
        buf.albedo
            .par_chunks_mut(bw)
            .zip(buf.light.par_chunks_mut(bw))
            .zip(buf.trans.par_chunks_mut(bw))
            .zip(buf.emit.par_chunks_mut(bw))
            .enumerate()
            .for_each(|(y, (((alb, lig), tr), em))| {
                for x in 0..bw {
                    let c = cells[y * bw + x];
                    let look = self.look(c.material());
                    let (wx, wy) = (bx + x as i32, by + y as i32);
                    let h = hash2(wx, wy);
                    let mut a = look.colors[h as usize % look.colors.len()];
                    if look.state == State::Liquid {
                        let shimmer =
                            (((wx as f32) * 0.3 + (wy as f32) * 0.17 + frame as f32 * 0.07).sin()
                                * 0.06)
                                + 1.0;
                        a = [a[0] * shimmer, a[1] * shimmer, a[2] * shimmer];
                    }
                    let mut e = if c.is_air() { [0.0; 3] } else { look.light };
                    if look.glow && !c.is_air() {
                        let b = blackbody(temp_to_celsius(c.temp));
                        e = [e[0] + b[0], e[1] + b[1], e[2] + b[2]];
                    }
                    if look.state == State::Fire {
                        let f = 0.8 + (h % 100) as f32 / 250.0;
                        e = [
                            look.colors[h as usize % look.colors.len()][0] * 4.0 * f,
                            look.colors[h as usize % look.colors.len()][1] * 3.0 * f,
                            look.colors[h as usize % look.colors.len()][2] * 2.0 * f,
                        ];
                    }
                    alb[x] = a;
                    em[x] = e;
                    lig[x] = e;
                    // Open air carries light far; anything denser absorbs more, and solids
                    // let it reach only a few cells in, which lights their surfaces.
                    tr[x] = if c.is_air() { 0.982 } else { 0.95 - look.opacity * 0.45 };
                }
            });
        // Sky light falls down each column until it meets something opaque. Air
        // above that point shows the sky; air below it is underground.
        if s.sky != Sky::Cave {
            for x in 0..bw {
                let mut k = 1.0f32;
                let mut open = true;
                for y in 0..bh {
                    let i = y * bw + x;
                    let t = buf.trans[i];
                    // Only ground and structures end the sky; foliage, rain, mist and
                    // smoke only dim it.
                    if t < 0.7 && self.look(cells[i].material()).structural {
                        open = false;
                    }
                    buf.sky[i] = open;
                    if k < 0.02 {
                        continue;
                    }
                    let l = &mut buf.light[i];
                    *l = [l[0] + sky[0] * k, l[1] + sky[1] * k, l[2] + sky[2] * k];
                    if !open {
                        buf.shaft[i] = [sky[0] * k, sky[1] * k, sky[2] * k];
                    }
                    k *= if t > 0.9 {
                        1.0
                    } else if t > 0.7 {
                        t
                    } else {
                        t * 0.5
                    };
                }
            }
        }
        // Particles and characters draw over the cells.
        for p in sim.particles.iter() {
            let (x, y) = (p.x.floor() - bx, p.y.floor() - by);
            if x >= 0 && y >= 0 && (x as usize) < bw && (y as usize) < bh {
                let i = y as usize * bw + x as usize;
                let look = self.look(p.cell.material());
                buf.albedo[i] = look.colors[0];
                let mut e = look.light;
                if look.glow {
                    let b = blackbody(temp_to_celsius(p.cell.temp));
                    e = [e[0] + b[0], e[1] + b[1], e[2] + b[2]];
                }
                buf.emit[i] = e;
                buf.light[i] =
                    [buf.light[i][0] + e[0], buf.light[i][1] + e[1], buf.light[i][2] + e[2]];
                buf.solid_override[i] = true;
            }
        }
        for a in avatars {
            let feet = CellPos::new(a.x.floor(), a.y.floor());
            let body = player_color(a.id);
            for dy in 0..fe_sim::AVATAR_H {
                for dx in 0..fe_sim::AVATAR_W {
                    let (x, y) = (
                        feet.x - fe_sim::AVATAR_W / 2 + dx - bx,
                        feet.y - fe_sim::AVATAR_H + dy - by,
                    );
                    if x < 0 || y < 0 || x as usize >= bw || y as usize >= bh {
                        continue;
                    }
                    let head = dy < 4 && (1..5).contains(&dx);
                    let legs = dy >= 9 && (dx == 1 || dx == 4);
                    let torso = (4..9).contains(&dy);
                    let col = if head {
                        [0.85, 0.62, 0.45]
                    } else if torso {
                        body
                    } else if legs {
                        [0.12, 0.12, 0.16]
                    } else {
                        continue;
                    };
                    let i = y as usize * bw + x as usize;
                    buf.albedo[i] = col;
                    buf.emit[i] = [col[0] * 0.25, col[1] * 0.25, col[2] * 0.25];
                    buf.solid_override[i] = true;
                }
            }
        }
    }

    #[expect(clippy::too_many_arguments, reason = "internal helper over shared buffers")]
    fn compose(
        &self,
        cells: &[Cell],
        buf: &Buffers,
        bw: usize,
        bh: usize,
        bx: i32,
        by: i32,
        s: &RenderSettings,
    ) -> Vec<[f32; 3]> {
        let amb = s.ambient;
        let mut out = vec![[0.0f32; 3]; bw * bh];
        out.par_chunks_mut(bw).enumerate().for_each(|(y, row)| {
            let t = ((y as i32 + by) as f32 / 400.0).clamp(0.0, 1.0);
            let sky_bg = s.sky.background(t);
            let cave_bg = [0.004, 0.0045, 0.006];
            for x in 0..bw {
                let bg = if buf.sky[y * bw + x] { sky_bg } else { cave_bg };
                let i = y * bw + x;
                let c = cells[i];
                let l = buf.light[i];
                let a = buf.albedo[i];
                let e = buf.emit[i];
                let look = self.look(c.material());
                let lit = |k: f32| {
                    [
                        a[0] * (amb[0] + l[0] * k),
                        a[1] * (amb[1] + l[1] * k),
                        a[2] * (amb[2] + l[2] * k),
                    ]
                };
                row[x] = if buf.solid_override[i] {
                    let v = lit(1.0);
                    [v[0] + e[0], v[1] + e[1], v[2] + e[2]]
                } else if c.is_air() || (look.state == State::Static && look.opacity < 0.02) {
                    let sh = buf.shaft[i];
                    if buf.sky[i] {
                        // Open sky, plus dust lit by light passing through.
                        [
                            bg[0] + l[0] * 0.012 + sh[0] * 0.09,
                            bg[1] + l[1] * 0.012 + sh[1] * 0.09,
                            bg[2] + l[2] * 0.012 + sh[2] * 0.09,
                        ]
                    } else {
                        // Underground: a dim rock back wall that catches nearby light,
                        // plus dust lit by shafts of sky light.
                        let h = hash2(bx + x as i32, by + y as i32);
                        let n = 0.75 + (h % 64) as f32 / 160.0;
                        let wall = [0.055 * n, 0.05 * n, 0.058 * n];
                        [
                            wall[0] * (amb[0] + l[0] * 0.35) + l[0] * 0.01 + sh[0] * 0.09,
                            wall[1] * (amb[1] + l[1] * 0.35) + l[1] * 0.01 + sh[1] * 0.09,
                            wall[2] * (amb[2] + l[2] * 0.35) + l[2] * 0.01 + sh[2] * 0.09,
                        ]
                    }
                } else {
                    match look.state {
                        State::Gas => {
                            let alpha = (look.opacity * 2.2).clamp(0.04, 0.85);
                            let v = lit(1.3);
                            let back =
                                [bg[0] + l[0] * 0.035, bg[1] + l[1] * 0.035, bg[2] + l[2] * 0.035];
                            [
                                back[0] + (v[0] - back[0]) * alpha + e[0],
                                back[1] + (v[1] - back[1]) * alpha + e[1],
                                back[2] + (v[2] - back[2]) * alpha + e[2],
                            ]
                        }
                        State::Fire => [e[0] * 0.9, e[1] * 0.9, e[2] * 0.9],
                        State::Liquid => {
                            let v = lit(0.9);
                            [
                                v[0] + e[0] + l[0] * 0.04,
                                v[1] + e[1] + l[1] * 0.04,
                                v[2] + e[2] + l[2] * 0.04,
                            ]
                        }
                        _ => {
                            // Rock far from open space sinks into shadow; edges stay lit.
                            let ao = (0.12 + buf.open[i] * 2.2).min(1.0);
                            let v = lit(1.0);
                            [v[0] * ao + e[0], v[1] * ao + e[1], v[2] * ao + e[2]]
                        }
                    }
                };
            }
        });
        out
    }
}

struct Buffers {
    w: usize,
    h: usize,
    albedo: Vec<[f32; 3]>,
    emit: Vec<[f32; 3]>,
    light: Vec<[f32; 3]>,
    trans: Vec<f32>,
    solid_override: Vec<bool>,
    sky: Vec<bool>,
    /// Direct sky light below the first solid in each column: light shafts.
    shaft: Vec<[f32; 3]>,
    /// Openness of the neighborhood, for ambient occlusion.
    open: Vec<f32>,
}

impl Buffers {
    fn new(w: usize, h: usize) -> Self {
        let n = w * h;
        Self {
            w,
            h,
            albedo: vec![[0.0; 3]; n],
            emit: vec![[0.0; 3]; n],
            light: vec![[0.0; 3]; n],
            trans: vec![0.0; n],
            solid_override: vec![false; n],
            sky: vec![false; n],
            shaft: vec![[0.0; 3]; n],
            open: vec![1.0; n],
        }
    }
}

/// Reads a rectangle of cells, chunk by chunk.
fn sample_cells(sim: &Simulation, bx: i32, by: i32, bw: usize, bh: usize) -> Vec<Cell> {
    let mut out = vec![Cell::BOUNDARY; bw * bh];
    let (a, b) = fe_core::IRect::from_size(bx, by, bw as i32, bh as i32).chunk_range();
    for cy in a.y..b.y {
        for cx in a.x..b.x {
            let Some(chunk) = sim.world.grid.chunk(ChunkPos::new(cx, cy)) else { continue };
            let o = chunk.pos().origin();
            for ly in 0..CHUNK_SIZE {
                let y = o.y + ly - by;
                if y < 0 || y as usize >= bh {
                    continue;
                }
                for lx in 0..CHUNK_SIZE {
                    let x = o.x + lx - bx;
                    if x < 0 || x as usize >= bw {
                        continue;
                    }
                    out[y as usize * bw + x as usize] = chunk.get((ly * CHUNK_SIZE + lx) as usize);
                }
            }
        }
    }
    out
}

fn draw_overlay(sim: &Simulation, img: &mut Image, fx0: f32, fy0: f32, ppc: f32) {
    let to_px = |x: i32, y: i32| (((x as f32 - fx0) * ppc) as i32, ((y as f32 - fy0) * ppc) as i32);
    for c in sim.world.grid.chunks() {
        let o = c.pos().origin();
        let (x0, y0) = to_px(o.x, o.y);
        let (x1, y1) = to_px(o.x + CHUNK_SIZE, o.y + CHUNK_SIZE);
        img.rect_outline(x0, y0, x1, y1, [60, 60, 70, 255]);
        let d = c.pending();
        if !d.is_empty() {
            let (a0, b0) = to_px(o.x + i32::from(d.x0), o.y + i32::from(d.y0));
            let (a1, b1) = to_px(o.x + i32::from(d.x1) + 1, o.y + i32::from(d.y1) + 1);
            img.rect_outline(a0, b0, a1, b1, [255, 60, 200, 255]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fe_core::IRect;
    use fe_engine::{Command, SimConfig};
    use std::sync::Arc;

    #[test]
    fn renders_lava_brighter_than_stone() {
        let mats = Arc::new(fe_materials::base_set());
        let mut sim = Simulation::new(mats.clone(), SimConfig::default());
        sim.load_rect(IRect::new(0, 0, 128, 128));
        let lava = sim.material("lava").unwrap_or_default();
        let stone = sim.material("stone").unwrap_or_default();
        sim.apply(&Command::FillRect { rect: IRect::new(0, 64, 64, 128), material: lava });
        sim.apply(&Command::FillRect { rect: IRect::new(64, 64, 128, 128), material: stone });
        let r = Renderer::new(&mats);
        let s = RenderSettings { view_w: 128, view_h: 128, scale: 1, ..RenderSettings::default() };
        let img = r.render(&sim, View { cx: 64.0, cy: 64.0, zoom: 1.0 }, &s, 0);
        assert_eq!((img.width, img.height), (128, 128));
        let lum = |x: u32, y: u32| {
            let i = ((y * img.width + x) * 4) as usize;
            u32::from(img.rgba[i]) + u32::from(img.rgba[i + 1]) + u32::from(img.rgba[i + 2])
        };
        assert!(lum(30, 100) > lum(110, 120) + 100, "{} vs {}", lum(30, 100), lum(110, 120));
        // Lava lights the stone next to it more than the stone far away.
        assert!(lum(66, 100) > lum(125, 100));
        assert!(!img.encode_png().is_empty());
    }
}
