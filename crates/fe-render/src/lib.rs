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
    /// For glowing liquids that freeze (lava): the freezing point and the
    /// material's own temperature in degrees C, to dim the glow as it cools.
    cooling_glow: Option<(i32, i32)>,
    /// Water: tinted pale as it nears boiling.
    water: bool,
    /// Gas with a lifetime (smoke, dust): thins out as its lifetime runs down.
    fades: bool,
    /// Burning solids (wood, leaves, charcoal): their light flickers.
    flicker: bool,
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
    // Brightness keeps climbing up to about 2000 C, so a white-hot pour reads
    // brighter than metal near its freezing point.
    let i = if t <= 800.0 { t / 400.0 } else { (2.0 + (t - 800.0) / 900.0).min(3.5) };
    // Red first, then orange, yellow near 1500 C and white above: metal on its
    // 1300 C freezing plateau reads orange and lava cooling toward 700 C reddens.
    let r = 1.0f32.min(0.25 + t / 500.0);
    let g = (t / 1300.0).powf(1.8).min(0.95);
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
                cooling_glow: match m.freeze {
                    Some(f) if m.state == State::Liquid && m.light != [0; 3] && m.temp > f.at => {
                        Some((temp_to_celsius(f.at), temp_to_celsius(m.temp)))
                    }
                    _ => None,
                },
                water: m.name == "water",
                fades: m.state == State::Gas && m.lifetime.is_some(),
                flicker: m.state == State::Static && m.lifetime.is_some() && m.light != [0; 3],
            })
            .collect();
        Self { looks }
    }

    fn look(&self, id: MaterialId) -> &MatLook {
        self.looks.get(id.index()).unwrap_or(&self.looks[1])
    }

    /// Light a cell gives off: its material's light, dimmed toward the freezing
    /// point for cooling liquids such as lava, plus the glow of its temperature.
    fn emission(&self, c: Cell) -> [f32; 3] {
        if c.is_air() {
            return [0.0; 3];
        }
        let look = self.look(c.material());
        let celsius = temp_to_celsius(c.temp);
        let mut e = look.light;
        if let Some((freeze, hot)) = look.cooling_glow {
            let k = ((celsius - freeze) as f32 / (hot - freeze) as f32).clamp(0.15, 1.0);
            e = [e[0] * k, e[1] * k, e[2] * k];
        }
        if look.glow {
            let b = blackbody(celsius);
            e = [e[0] + b[0], e[1] + b[1], e[2] + b[2]];
        }
        e
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
            buf.bounce = light::wide(&light::wide(&buf.light, bw, bh), bw, bh);
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
            .zip(buf.gas_density.par_chunks_mut(bw))
            .zip(buf.gas_color.par_chunks_mut(bw))
            .enumerate()
            .for_each(|(y, (((((alb, lig), tr), em), gd), gc))| {
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
                        if look.water {
                            // Hot water turns pale, up to 30% at boiling.
                            let k = ((temp_to_celsius(c.temp) - 60) as f32 / 40.0).clamp(0.0, 1.0)
                                * 0.3;
                            a = lerp3(a, [0.5, 0.58, 0.66], k);
                        }
                    }
                    let mut e = self.emission(c);
                    if look.flicker {
                        // Burning surfaces flicker cell by cell, a few times a second.
                        let n = hash2(wx ^ (frame / 5) as i32, wy) % 100;
                        let f = 0.45 + n as f32 / 110.0;
                        e = [e[0] * f, e[1] * f, e[2] * f];
                    }
                    if look.state == State::Fire {
                        let f = 0.8 + (h % 100) as f32 / 250.0;
                        e = [
                            look.colors[h as usize % look.colors.len()][0] * 4.0 * f,
                            look.colors[h as usize % look.colors.len()][1] * 3.0 * f,
                            look.colors[h as usize % look.colors.len()][2] * 2.0 * f,
                        ];
                    }
                    if look.state == State::Gas {
                        // Smoke and dust thin out over the last part of their life
                        // instead of popping out of existence.
                        let o = if look.fades {
                            look.opacity * (f32::from(c.value) / 48.0).min(1.0)
                        } else {
                            look.opacity
                        };
                        gd[x] = [o, 0.0, 0.0];
                        gc[x] = [a[0] * o, a[1] * o, a[2] * o];
                    }
                    alb[x] = a;
                    em[x] = e;
                    lig[x] = e;
                    // Open air carries light far; anything denser absorbs more, and solids
                    // let it reach only a few cells in, which lights their surfaces.
                    tr[x] = if c.is_air() { 0.982 } else { 0.95 - look.opacity * 0.45 };
                }
            });
        // Gas reads as a soft volume, not as specks: its density and color are
        // blurred over neighboring cells, and the air around a puff is tinted too.
        buf.gas_density = light::wide(&light::soften(&buf.gas_density, bw, bh), bw, bh);
        buf.gas_color = light::wide(&light::soften(&buf.gas_color, bw, bh), bw, bh);
        // Sky light falls down each column until it meets something opaque. Air
        // above that point shows the sky; air below it is underground.
        // How far open air may carry the sky in under each cell's cover (see
        // `reach_sky`): further under a rigid body than under rock.
        let mut reach = vec![0u16; bw * bh];
        if s.sky != Sky::Cave {
            for x in 0..bw {
                let mut k = 1.0f32;
                let mut open = true;
                let mut under_rock = false;
                for y in 0..bh {
                    let i = y * bw + x;
                    let t = buf.trans[i];
                    // Only ground and structures end the sky; foliage, rain, mist and
                    // smoke only dim it.
                    // Cells beyond the loaded world read as the boundary; above the
                    // world that is open sky, so it never ends the sky.
                    // Air under an overhang is handed back to the sky afterwards
                    // if open air connects it to the sky (see `reach_sky`).
                    // Glass ends it too: a room behind a window is indoors.
                    let boundary = cells[i].material() == MaterialId::BOUNDARY;
                    if !boundary && self.look(cells[i].material()).structural {
                        open = false;
                        under_rock |= !cells[i].has(Cell::FLAG_IN_BODY);
                    }
                    reach[i] = if under_rock { SKY_REACH } else { SKY_REACH_BODY };
                    if boundary && open {
                        buf.sky[i] = 1.0;
                        continue;
                    }
                    buf.sky[i] = if open { 1.0 } else { 0.0 };
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
        if s.sky != Sky::Cave {
            let pass: Vec<bool> = cells
                .iter()
                .map(|&c| {
                    c.is_air()
                        || (!c.has(Cell::FLAG_IN_BODY)
                            && matches!(self.look(c.material()).state, State::Gas | State::Fire))
                })
                .collect();
            reach_sky(&mut buf.sky, &pass, &reach, bw, bh);
        }
        // Particles and characters draw over the cells.
        for p in sim.particles.iter() {
            let (x, y) = (p.x.floor() - bx, p.y.floor() - by);
            if x >= 0 && y >= 0 && (x as usize) < bw && (y as usize) < bh {
                let i = y as usize * bw + x as usize;
                let look = self.look(p.cell.material());
                buf.albedo[i] = look.colors[0];
                let e = self.emission(p.cell);
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
            let sky_light = s.sky.light();
            for x in 0..bw {
                let i = y * bw + x;
                let skyness = buf.sky[i];
                let c = cells[i];
                let l = buf.light[i];
                let a = buf.albedo[i];
                let e = buf.emit[i];
                let look = self.look(c.material());
                // A glowing cell's look is its own glow; its surface is lit only by
                // light from elsewhere, or a white-hot pour and one about to freeze
                // would both saturate to the same yellow.
                let l = [(l[0] - e[0]).max(0.0), (l[1] - e[1]).max(0.0), (l[2] - e[2]).max(0.0)];
                let lit = |k: f32| {
                    [
                        a[0] * (amb[0] + l[0] * k),
                        a[1] * (amb[1] + l[1] * k),
                        a[2] * (amb[2] + l[2] * k),
                    ]
                };
                let gas = buf.gas_density[i][0];
                let is_gas = look.state == State::Gas && !c.is_air();
                let open = c.is_air() || (look.state == State::Static && look.opacity < 0.02);
                row[x] = if buf.solid_override[i] {
                    let v = lit(1.0);
                    [v[0] + e[0], v[1] + e[1], v[2] + e[2]]
                } else if open || is_gas {
                    let sh = buf.shaft[i];
                    // Open sky, plus dust lit by light passing through.
                    let open_back = [
                        sky_bg[0] + l[0] * 0.012 + sh[0] * 0.09,
                        sky_bg[1] + l[1] * 0.012 + sh[1] * 0.09,
                        sky_bg[2] + l[2] * 0.012 + sh[2] * 0.09,
                    ];
                    let back = if skyness >= 1.0 {
                        open_back
                    } else {
                        // Underground: a dim rock back wall that catches nearby light,
                        // plus dust lit by shafts of sky light. Near an opening to
                        // the sky it fades into the sky.
                        let h = hash2(bx + x as i32, by + y as i32);
                        let n = 0.75 + (h % 64) as f32 / 160.0;
                        let wall = [0.055 * n, 0.05 * n, 0.058 * n];
                        let cave = [
                            cave_bg[0]
                                + wall[0] * (amb[0] + l[0] * 0.35)
                                + l[0] * 0.01
                                + sh[0] * 0.09,
                            cave_bg[1]
                                + wall[1] * (amb[1] + l[1] * 0.35)
                                + l[1] * 0.01
                                + sh[1] * 0.09,
                            cave_bg[2]
                                + wall[2] * (amb[2] + l[2] * 0.35)
                                + l[2] * 0.01
                                + sh[2] * 0.09,
                        ];
                        lerp3(cave, open_back, skyness)
                    };
                    if is_gas || gas > 0.02 {
                        // Smoke, steam, mist and dust: the blurred gas color, lit by
                        // the light around it, over what lies behind.
                        let alpha = (1.0 - (-gas * 3.0).exp()).min(0.88);
                        let gc = buf.gas_color[i];
                        let g =
                            if gas > 1e-4 { [gc[0] / gas, gc[1] / gas, gc[2] / gas] } else { a };
                        // Gas under the open sky also scatters light from the whole
                        // dome, so a dust cloud in the shadow of a slab is not black.
                        let dome = skyness * 0.35;
                        let v = [
                            g[0] * (amb[0] + l[0] * 1.3 + sky_light[0] * dome),
                            g[1] * (amb[1] + l[1] * 1.3 + sky_light[1] * dome),
                            g[2] * (amb[2] + l[2] * 1.3 + sky_light[2] * dome),
                        ];
                        [
                            back[0] + (v[0] - back[0]) * alpha + e[0],
                            back[1] + (v[1] - back[1]) * alpha + e[1],
                            back[2] + (v[2] - back[2]) * alpha + e[2],
                        ]
                    } else {
                        back
                    }
                } else {
                    match look.state {
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
                            // Rigid bodies are thin and never sink as far.
                            let floor = if c.has(Cell::FLAG_IN_BODY) { 0.6 } else { 0.22 };
                            let ao = (floor + buf.open[i] * 2.2).min(1.0);
                            let v = lit(1.0);
                            // Deep rock still shows relief: broad light and dark
                            // patches and strata, lit by the ambient light and by
                            // light bouncing in from the spaces around it.
                            let (wx, wy) = (bx + x as i32, by + y as i32);
                            let rl = relief(wx, wy);
                            let b = buf.bounce[i];
                            let deep = [
                                a[0] * (amb[0] * 0.35 + b[0] * 0.4) * rl,
                                a[1] * (amb[1] * 0.35 + b[1] * 0.4) * rl,
                                a[2] * (amb[2] * 0.35 + b[2] * 0.4) * rl,
                            ];
                            [
                                (v[0] * ao).max(deep[0]) + e[0],
                                (v[1] * ao).max(deep[1]) + e[1],
                                (v[2] * ao).max(deep[2]) + e[2],
                            ]
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
    /// 1 for air open to the sky, 0 underground, in between near an opening.
    sky: Vec<f32>,
    /// Light blurred far into the rock: the light that bounces off lit spaces.
    bounce: Vec<[f32; 3]>,
    /// Direct sky light below the first solid in each column: light shafts.
    shaft: Vec<[f32; 3]>,
    /// Openness of the neighborhood, for ambient occlusion.
    open: Vec<f32>,
    /// Blurred gas density (opacity, 0 to 1) in the first channel.
    gas_density: Vec<[f32; 3]>,
    /// Blurred gas color weighted by density.
    gas_color: Vec<[f32; 3]>,
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
            sky: vec![0.0; n],
            bounce: vec![[0.0; 3]; n],
            shaft: vec![[0.0; 3]; n],
            open: vec![1.0; n],
            gas_density: vec![[0.0; 3]; n],
            gas_color: vec![[0.0; 3]; n],
        }
    }
}

/// How far, in cells, air under an overhang still counts as sky.
const SKY_REACH: u16 = 64;
/// The same under a rigid body and nothing else: a leaning tower or a falling
/// slab shades the ground under it but does not make it a cave.
const SKY_REACH_BODY: u16 = 256;

/// Lets the sky reach air that the column test hides: air under an overhang,
/// a bridge or a tilted slab is sky if open air connects it to the sky within
/// `reach` cells ([`SKY_REACH`] under rock, [`SKY_REACH_BODY`] under only a
/// rigid body), fading out over the second half of that distance, so
/// only enclosed rooms and caves show the rock back wall. `pass` marks the cells
/// sky light can spread through: air, gas and fire.
fn reach_sky(sky: &mut [f32], pass: &[bool], reach: &[u16], w: usize, h: usize) {
    const FAR: u16 = u16::MAX;
    let mut d: Vec<u16> = sky.iter().map(|&k| if k >= 1.0 { 0 } else { FAR }).collect();
    let relax = |d: &mut [u16], i: usize, j: usize| {
        if pass[i] && d[j] < FAR {
            d[i] = d[i].min(d[j] + 1);
        }
    };
    for _ in 0..2 {
        for y in 0..h {
            let row = y * w;
            if y > 0 {
                for x in 0..w {
                    relax(&mut d, row + x, row + x - w);
                }
            }
            for x in 1..w {
                relax(&mut d, row + x, row + x - 1);
            }
            for x in (0..w - 1).rev() {
                relax(&mut d, row + x, row + x + 1);
            }
        }
        for y in (0..h - 1).rev() {
            let row = y * w;
            for x in 0..w {
                relax(&mut d, row + x, row + x + w);
            }
            for x in 1..w {
                relax(&mut d, row + x, row + x - 1);
            }
            for x in (0..w - 1).rev() {
                relax(&mut d, row + x, row + x + 1);
            }
        }
    }
    for ((k, &dist), &r) in sky.iter_mut().zip(&d).zip(reach) {
        if *k < 1.0 && dist < r {
            *k = ((f32::from(r) - f32::from(dist)) / f32::from(r / 2)).min(1.0);
        }
    }
}

/// Low-frequency relief for deep rock, about 0.5 to 1.3: smooth patches a
/// couple of dozen cells across, finer mottling, and faint strata.
fn relief(x: i32, y: i32) -> f32 {
    fn value_noise(x: i32, y: i32, scale: f32, salt: i32) -> f32 {
        let cell =
            |gx: i32, gy: i32| (hash2(gx.wrapping_mul(7919) ^ salt, gy) % 1000) as f32 / 1000.0;
        let (fx, fy) = (x as f32 / scale, y as f32 / scale);
        let (gx, gy) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - gx, fy - gy);
        let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
        let (gx, gy) = (gx as i32, gy as i32);
        let top = cell(gx, gy) + (cell(gx + 1, gy) - cell(gx, gy)) * sx;
        let bottom = cell(gx, gy + 1) + (cell(gx + 1, gy + 1) - cell(gx, gy + 1)) * sx;
        top + (bottom - top) * sy
    }
    let broad = value_noise(x, y, 22.0, 0x51);
    let fine = value_noise(x, y, 6.0, 0x2c3);
    let strata = ((y as f32 * 0.18 + broad * 5.0).sin() * 0.5 + 0.5) * 0.12;
    0.5 + broad * 0.55 + fine * 0.15 + strata
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

    /// Tone-mapped luminance of a cell's own glow at the default exposure.
    fn glow_lum(r: &Renderer, mats: &MaterialSet, name: &str, celsius: i32) -> f32 {
        let id = mats.id(name).unwrap_or_default();
        let e = r.emission(Cell::new(id, fe_core::celsius_to_temp(celsius), 0));
        let x = RenderSettings::default().exposure;
        let t = |v: f32| 1.0 - (-v * x).exp();
        0.2126 * t(e[0]) + 0.7152 * t(e[1]) + 0.0722 * t(e[2])
    }

    #[test]
    fn hotter_melts_glow_brighter() {
        let mats = fe_materials::base_set();
        let r = Renderer::new(&mats);
        for (name, cool, hot) in [("molten_metal", 1300, 1900), ("lava", 750, 1150)] {
            let (a, b) = (glow_lum(&r, &mats, name, cool), glow_lum(&r, &mats, name, hot));
            assert!(b > a * 1.15, "{name}: {cool} C {a}, {hot} C {b}");
        }
    }

    #[test]
    fn a_white_hot_pour_renders_whiter_than_cooling_metal() {
        let mats = Arc::new(fe_materials::base_set());
        let mut sim = Simulation::new(mats.clone(), SimConfig::default());
        sim.load_rect(IRect::new(0, 0, 192, 64));
        let id = sim.material("molten_metal").unwrap_or_default();
        for (x0, c) in [(0, 1300), (128, 1900)] {
            for y in 0..64 {
                for x in x0..x0 + 64 {
                    let cell = Cell::new(id, fe_core::celsius_to_temp(c), 0);
                    let _ = sim.world.set(CellPos::new(x, y), cell);
                }
            }
        }
        let r = Renderer::new(&mats);
        let s = RenderSettings { view_w: 192, view_h: 64, scale: 1, ..RenderSettings::default() };
        let img = r.render(&sim, View { cx: 96.0, cy: 32.0, zoom: 1.0 }, &s, 0);
        let px = |x: u32| {
            let i = ((32 * img.width + x) * 4) as usize;
            [img.rgba[i], img.rgba[i + 1], img.rgba[i + 2]]
        };
        let (cool, hot) = (px(32), px(160));
        assert!(hot[2] > cool[2] + 40, "cool {cool:?} hot {hot:?}");
        assert!(hot[1] >= cool[1], "cool {cool:?} hot {hot:?}");
    }
}
