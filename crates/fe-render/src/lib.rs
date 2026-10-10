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

/// Largest view width or height, in cells.
const MAX_VIEW: u32 = 8192;
/// Largest pixels per cell.
const MAX_SCALE: u32 = 16;
/// Most cells the renderer looks at in one frame, before the light margin.
const MAX_CELLS: u32 = 16 * 1024 * 1024;

/// The zoom actually used: at least 0.05, and high enough that the visible
/// area stays within [`MAX_CELLS`], since zooming out widens every buffer.
fn clamp_zoom(zoom: f32, view_w: u32, view_h: u32) -> f32 {
    let min_zoom = (view_w as f32 * view_h as f32 / MAX_CELLS as f32).sqrt();
    let zoom = if zoom.is_finite() { zoom } else { 1.0 };
    zoom.max(0.05).max(min_zoom)
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

impl RenderSettings {
    /// The settings with the view at least 1x1 and at most [`MAX_VIEW`] cells
    /// on a side, and the scale between 1 and [`MAX_SCALE`]. Frames are always
    /// rendered with clamped settings, so use these to size video output.
    #[must_use]
    pub fn clamped(&self) -> Self {
        Self {
            view_w: self.view_w.clamp(1, MAX_VIEW),
            view_h: self.view_h.clamp(1, MAX_VIEW),
            scale: self.scale.clamp(1, MAX_SCALE),
            ..self.clone()
        }
    }
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
    /// Stone: tinted by the world generator's depth bands.
    banded: bool,
    water: bool,
    /// Gas with a lifetime (smoke, dust): thins out as its lifetime runs down.
    fades: bool,
    /// Burning solids (wood, leaves, charcoal): their light flickers.
    flicker: bool,
    /// Longest lifetime in ticks, for materials that burn down: a flying
    /// ember's light fades with what is left of it.
    life: Option<u8>,
    /// A blast's flash: its light is spread far, like a spark's.
    flash: bool,
    /// Fuel (wood, leaves, charcoal, and the same burning): the light of
    /// nearby flames falling on it is capped, so a trunk burning at its top
    /// shows a lit tip over a dark trunk instead of glowing orange along its
    /// whole length.
    fuel: bool,
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
                banded: m.name == "stone",
                fades: m.state == State::Gas && m.lifetime.is_some(),
                flicker: m.state == State::Static && m.lifetime.is_some() && m.light != [0; 3],
                life: m.lifetime.map(|(_, max)| max),
                flash: m.name == "flash",
                fuel: m.state == State::Static
                    && (m.ignite.is_some() || (m.lifetime.is_some() && m.light != [0; 3])),
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

    /// Light a flying particle gives off: its glow by temperature, so a molten
    /// speck fades from white through orange to dull red as it cools, and for
    /// a material that burns down, such as an ember, light that fades as it
    /// burns out.
    fn particle_emission(&self, c: Cell) -> [f32; 3] {
        let e = self.emission(c);
        match self.look(c.material()).life {
            Some(max) if max > 0 => {
                let k = 0.15 + 0.85 * (f32::from(c.value) / f32::from(max)).min(1.0);
                [e[0] * k, e[1] * k, e[2] * k]
            }
            _ => e,
        }
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
        let s = &s.clamped();
        let zoom = clamp_zoom(view.zoom, s.view_w, s.view_h);
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

        // Liquid in the air is drawn with the path it swept: see
        // `fe_sim::liquid::drawn`.
        let cells = fe_sim::liquid::drawn(
            &sample_cells(sim, bx, by, bw, bh),
            bw,
            bh,
            |c| self.look(c.material()).state == State::Liquid,
            |c| {
                c.has(Cell::FLAG_IN_BODY)
                    || matches!(self.look(c.material()).state, State::Static | State::Powder)
            },
        );
        let mut buf = Buffers::new(bw, bh);
        self.fill_buffers(sim, avatars, &cells, &mut buf, bx, by, frame, s);
        if s.lighting {
            light::propagate(&mut buf.light, &buf.trans, bw, bh);
            buf.light = light::soften(&buf.light, bw, bh);
            buf.shaft = light::soften(&buf.shaft, bw, bh);
            buf.bounce = light::wide(&light::wide(&buf.light, bw, bh), bw, bh);
            self.broad_light(sim, &cells, &mut buf, bx, by);
        }
        // How open each cell's surroundings are: 1 deep in air, 0 deep in rock.
        let open_mask: Vec<[f32; 3]> =
            buf.trans.iter().map(|&t| if t > 0.9 { [1.0, 0.0, 0.0] } else { [0.0; 3] }).collect();
        buf.open = light::soften(&light::soften(&open_mask, bw, bh), bw, bh)
            .into_iter()
            .map(|v| v[0])
            .collect();
        let mut hdr = self.compose(&cells, &buf, bw, bh, bx, by, s);
        if sim.is_streamed() {
            stream_fade(sim, &mut hdr, bw, bh, bx, by, s.exposure);
        }
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

    /// Adds the light of many small, bright sources: sparks in flight and a
    /// blast's flash. The light field keeps only the brightest source at each
    /// cell, so a hundred sparks would light the ground no more than one; here
    /// their light is summed on a coarse grid and spread with a gentle falloff,
    /// so a burst of fireworks lights the roofs below it and a large charge
    /// lights the whole room for a moment.
    fn broad_light(&self, sim: &Simulation, cells: &[Cell], buf: &mut Buffers, bx: i32, by: i32) {
        const BLOCK: usize = 8;
        /// Distance in cells at which a source's light falls to half.
        const REACH: f32 = 48.0;
        const GAIN: f32 = 0.004;
        let (bw, bh) = (buf.w, buf.h);
        let (gw, gh) = (bw.div_ceil(BLOCK), bh.div_ceil(BLOCK));
        let mut src = vec![[0.0f32; 3]; gw * gh];
        let mut add = |x: i32, y: i32, e: [f32; 3]| {
            if x >= 0 && y >= 0 && (x as usize) < bw && (y as usize) < bh {
                let s = &mut src[(y as usize / BLOCK) * gw + x as usize / BLOCK];
                *s = [s[0] + e[0], s[1] + e[1], s[2] + e[2]];
            }
        };
        for p in sim.particles.iter() {
            add(p.x.floor() - bx, p.y.floor() - by, self.particle_emission(p.cell));
        }
        for (i, c) in cells.iter().enumerate() {
            if self.look(c.material()).flash {
                add((i % bw) as i32, (i / bw) as i32, buf.emit[i]);
            }
        }
        let sources: Vec<(f32, f32, [f32; 3])> = src
            .iter()
            .enumerate()
            .filter(|(_, e)| e[0] + e[1] + e[2] > 0.5)
            .map(|(i, e)| ((i % gw) as f32, (i / gw) as f32, *e))
            .collect();
        if sources.is_empty() {
            return;
        }
        let reach = REACH / BLOCK as f32;
        let mut coarse = vec![[0.0f32; 3]; gw * gh];
        coarse.par_iter_mut().enumerate().for_each(|(i, c)| {
            let (gx, gy) = ((i % gw) as f32, (i / gw) as f32);
            for &(sx, sy, e) in &sources {
                let d = ((gx - sx) * (gx - sx) + (gy - sy) * (gy - sy)).sqrt();
                let k = GAIN * reach / (reach + d);
                *c = [c[0] + e[0] * k, c[1] + e[1] * k, c[2] + e[2] * k];
            }
        });
        buf.broad.par_chunks_mut(bw).enumerate().for_each(|(y, row)| {
            // Bilinear between block centers.
            let fy = ((y as f32 + 0.5) / BLOCK as f32 - 0.5).clamp(0.0, (gh - 1) as f32);
            let (y0, ty) = (fy.floor() as usize, fy.fract());
            let y1 = (y0 + 1).min(gh - 1);
            for (x, l) in row.iter_mut().enumerate() {
                let fx = ((x as f32 + 0.5) / BLOCK as f32 - 0.5).clamp(0.0, (gw - 1) as f32);
                let (x0, tx) = (fx.floor() as usize, fx.fract());
                let x1 = (x0 + 1).min(gw - 1);
                for k in 0..3 {
                    let top = coarse[y0 * gw + x0][k] * (1.0 - tx) + coarse[y0 * gw + x1][k] * tx;
                    let bot = coarse[y1 * gw + x0][k] * (1.0 - tx) + coarse[y1 * gw + x1][k] * tx;
                    l[k] += top * (1.0 - ty) + bot * ty;
                }
            }
        });
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
                    if look.banded
                        && let Some(b) = sim.worldgen().rock_band(sim.world.seed(), wx, wy)
                    {
                        // Generated stone is tinted by depth band: warm grey
                        // near the surface, cooler and darker further down.
                        let t = ROCK_BANDS[usize::from(b).min(ROCK_BANDS.len() - 1)];
                        a = [a[0] * t[0], a[1] * t[1], a[2] * t[2]];
                    }
                    if look.state == State::Liquid {
                        let shimmer =
                            (((wx as f32) * 0.3 + (wy as f32) * 0.17 + frame as f32 * 0.07).sin()
                                * 0.06)
                                + 1.0;
                        a = [a[0] * shimmer, a[1] * shimmer, a[2] * shimmer];
                        if look.water {
                            // Hot water stays blue: near boiling it only
                            // shimmers harder and a touch lighter, cell by
                            // cell, instead of turning a flat pale mint.
                            let k = ((temp_to_celsius(c.temp) - 70) as f32 / 30.0).clamp(0.0, 1.0);
                            if k > 0.0 {
                                let n = hash2(wx ^ (frame / 3) as i32, wy) % 100;
                                let f = 1.0 + k * (n as f32 / 100.0 - 0.3) * 0.25;
                                a = [a[0] * f, a[1] * f, (a[2] * f).max(a[2])];
                            }
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
        let inside = if s.sky == Sky::Cave {
            vec![false; bw * bh]
        } else {
            body_insides(sim, |m| self.look(m).state == State::Liquid, bx, by, bw, bh)
        };
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
                    // A rigid body is an object in front of the background, not
                    // part of it: under a ship's yard or a bridge deck the sky
                    // goes on, and only the body's shadow falls there. The air a
                    // body encloses, such as a ship's hold, shows its dark inside.
                    let boundary = cells[i].material() == MaterialId::BOUNDARY;
                    let body = cells[i].has(Cell::FLAG_IN_BODY);
                    if !boundary && !body && self.look(cells[i].material()).structural {
                        open = false;
                        under_rock |= !cells[i].has(Cell::FLAG_IN_BODY);
                    }
                    reach[i] = if under_rock { SKY_REACH } else { SKY_REACH_BODY };
                    if boundary && open {
                        buf.sky[i] = 1.0;
                        continue;
                    }
                    buf.sky[i] = if open && !inside[i] { 1.0 } else { 0.0 };
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
            // A body's dark inside stays dark: the sky does not spread into it.
            let pass: Vec<bool> = cells
                .iter()
                .zip(&inside)
                .map(|(&c, &held)| {
                    !held
                        && (c.is_air()
                            || (!c.has(Cell::FLAG_IN_BODY)
                                && matches!(
                                    self.look(c.material()).state,
                                    State::Gas | State::Fire
                                )))
                })
                .collect();
            reach_sky(&mut buf.sky, &pass, &reach, bw, bh);
        }
        // Particles and characters draw over the cells. A glowing particle
        // also leaves a fading streak back along its path, so sparks read as
        // short trails of light rather than single specks.
        for p in sim.particles.iter() {
            let e = self.particle_emission(p.cell);
            let (x, y) = (p.x.floor() - bx, p.y.floor() - by);
            if x >= 0 && y >= 0 && (x as usize) < bw && (y as usize) < bh {
                let i = y as usize * bw + x as usize;
                let look = self.look(p.cell.material());
                buf.albedo[i] = look.colors[0];
                buf.emit[i] = e;
                buf.light[i] =
                    [buf.light[i][0] + e[0], buf.light[i][1] + e[1], buf.light[i][2] + e[2]];
                buf.solid_override[i] = true;
            }
            if e[0] + e[1] + e[2] < 0.05 {
                continue;
            }
            let (vx, vy) = (p.vx.0 as f32 / 65536.0, p.vy.0 as f32 / 65536.0);
            let len = ((vx * vx + vy * vy).sqrt() * 2.0).min(10.0);
            let steps = len.ceil() as i32;
            for k in 1..=steps {
                let f = k as f32 / steps as f32;
                let tx = (p.x.0 as f32 / 65536.0 - vx * 2.0 * f).floor() as i32 - bx;
                let ty = (p.y.0 as f32 / 65536.0 - vy * 2.0 * f).floor() as i32 - by;
                if tx < 0 || ty < 0 || tx as usize >= bw || ty as usize >= bh {
                    continue;
                }
                let i = ty as usize * bw + tx as usize;
                let w = 0.6 * (1.0 - f) * (1.0 - f);
                let g = &mut buf.glow[i];
                *g = [g[0] + e[0] * w, g[1] + e[1] * w, g[2] + e[2] * w];
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
                // The broad light of sparks and flashes falls on surfaces and
                // smoke; the sky behind them stays dark.
                let sky_l = l;
                let wide = buf.broad[i];
                let l = [l[0] + wide[0], l[1] + wide[1], l[2] + wide[2]];
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
                let glow = buf.glow[i];
                row[x] = if buf.solid_override[i] {
                    let v = lit(1.0);
                    [v[0] + e[0], v[1] + e[1], v[2] + e[2]]
                } else if open || is_gas {
                    let sh = buf.shaft[i];
                    // Open sky, plus dust lit by light passing through.
                    let open_back = [
                        sky_bg[0] + sky_l[0] * 0.012 + sh[0] * 0.09,
                        sky_bg[1] + sky_l[1] * 0.012 + sh[1] * 0.09,
                        sky_bg[2] + sky_l[2] * 0.012 + sh[2] * 0.09,
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
                        State::Liquid if look.water => {
                            // Water tints the light it receives: warm light from
                            // fire and smoke is taken mostly as its brightness
                            // and capped, so the top rows of a river stay blue
                            // instead of washing out to grey.
                            let lum = (l[0] + l[1] + l[2]) / 3.0;
                            let cap = sky_light[2].max(0.15) * 1.5 + 0.4;
                            let w = |k: usize| (l[k] * 0.3 + lum * 0.7).min(cap);
                            let lw = [w(0), w(1), w(2)];
                            [
                                a[0] * (amb[0] + lw[0] * 0.9) + e[0],
                                a[1] * (amb[1] + lw[1] * 0.9) + e[1],
                                a[2] * (amb[2] + lw[2] * 0.9) + e[2],
                            ]
                        }
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
                            let v = if look.fuel {
                                // Flame light on fuel is capped near the sky's
                                // own light: only the burning cells glow.
                                let cap = |k: usize| sky_light[k] + 0.12;
                                [
                                    a[0] * (amb[0] + l[0].min(cap(0))),
                                    a[1] * (amb[1] + l[1].min(cap(1))),
                                    a[2] * (amb[2] + l[2].min(cap(2))),
                                ]
                            } else {
                                lit(1.0)
                            };
                            // Deep rock still shows relief: broad light and dark
                            // patches and strata, lit by the ambient light and by
                            // light bouncing in from the spaces around it.
                            let (wx, wy) = (bx + x as i32, by + y as i32);
                            let rl = relief(wx, wy);
                            let mut b = buf.bounce[i];
                            if look.fuel {
                                b = [
                                    b[0].min(sky_light[0] + 0.12),
                                    b[1].min(sky_light[1] + 0.12),
                                    b[2].min(sky_light[2] + 0.12),
                                ];
                            }
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
                row[x] = [row[x][0] + glow[0], row[x][1] + glow[1], row[x][2] + glow[2]];
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
    /// Light added on top of everything: the streaks behind sparks.
    glow: Vec<[f32; 3]>,
    /// Light from many small bright sources, summed and spread wide; it
    /// lights surfaces and smoke but not the sky (see `broad_light`).
    broad: Vec<[f32; 3]>,
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
            glow: vec![[0.0; 3]; n],
            broad: vec![[0.0; 3]; n],
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

/// Cells of the buffer in the air a body closes in, such as a ship's sealed
/// hold, which draws as the body's dark inside rather than as sky. So does air
/// under the body under deep liquid, such as a pocket left in a sunken hull:
/// there is no sky behind it. Puddles are too thin to count, and air right
/// under liquid with no body between, such as under spray a blast throws
/// over a ship, is the sky behind the spray.
fn body_insides(
    sim: &Simulation,
    liquid: impl Fn(MaterialId) -> bool,
    bx: i32,
    by: i32,
    bw: usize,
    bh: usize,
) -> Vec<bool> {
    let mut inside = vec![false; bw * bh];
    let mut mark = |x: i32, y: i32| {
        let (x, y) = (x - bx, y - by);
        if x >= 0 && y >= 0 && (x as usize) < bw && (y as usize) < bh {
            inside[y as usize * bw + x as usize] = true;
        }
    };
    for b in sim.physics.bodies() {
        for p in &b.hollow {
            mark(p.x, p.y);
        }
        let mut cells = b.stamped.iter().map(|s| s.0);
        let Some(first) = cells.next() else { continue };
        let (mut x0, mut y0, mut x1, mut y1) = (first.x, first.y, first.x, first.y);
        for p in cells {
            (x0, y0, x1, y1) = (x0.min(p.x), y0.min(p.y), x1.max(p.x), y1.max(p.y));
        }
        // Keep to the part of the footprint in view, and start a little above
        // it so a sea over a sunken body is seen.
        let (x0, x1) = (x0.max(bx), x1.min(bx + bw as i32 - 1));
        let (y1, top) = (y1.min(by + bh as i32 - 1), y0.max(by) - DEEP_LIQUID);
        for x in x0..=x1 {
            let (mut run, mut deep, mut under) = (0, false, false);
            for y in top..=y1 {
                let c = sim.world.get(CellPos::new(x, y));
                if c.has(Cell::FLAG_IN_BODY) {
                    run = 0;
                    under |= deep;
                } else if liquid(c.material()) {
                    run += 1;
                    deep |= run >= DEEP_LIQUID;
                } else {
                    run = 0;
                    if under && c.is_air() {
                        mark(x, y);
                    }
                }
            }
        }
    }
    inside
}

/// Liquid this many cells deep hides the sky from the air under it.
const DEEP_LIQUID: i32 = 8;

/// Reads a rectangle of cells, chunk by chunk.
/// Tints of generated stone by depth band (see `WorldGen::rock_band`).
const ROCK_BANDS: [[f32; 3]; 4] =
    [[1.3, 1.06, 0.78], [1.0, 1.0, 1.0], [0.72, 0.84, 1.2], [0.72, 0.56, 0.8]];

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

/// How unloaded space looks in a streamed world, in linear display light: a
/// dim slate with faint diagonal hatching, so the edge of the loaded world
/// reads as "not generated yet" instead of as black rock or a black void.
fn unloaded_color(wx: i32, wy: i32) -> [f32; 3] {
    if (wx + wy).rem_euclid(14) < 2 { [0.020, 0.022, 0.032] } else { [0.010, 0.011, 0.016] }
}

/// Draws unloaded space in a streamed world (see [`unloaded_color`]) and fades
/// newly streamed chunks in from it over [`fe_engine::STREAM_FADE_TICKS`]
/// ticks, so terrain arriving at the edge of the world reads as streaming in.
fn stream_fade(
    sim: &Simulation,
    hdr: &mut [[f32; 3]],
    bw: usize,
    bh: usize,
    bx: i32,
    by: i32,
    exposure: f32,
) {
    let tick = sim.tick();
    let fade = fe_engine::STREAM_FADE_TICKS as f32;
    // Linear display light back to the HDR value that tone maps to it.
    let inv = |l: f32| -(1.0 - l).ln() / exposure.max(0.01);
    let (a, b) = fe_core::IRect::from_size(bx, by, bw as i32, bh as i32).chunk_range();
    for cy in a.y..b.y {
        for cx in a.x..b.x {
            let pos = ChunkPos::new(cx, cy);
            // 0 shows only unloaded space, 1 only the chunk.
            let k = if sim.world.grid.contains(pos) {
                match sim.streamed_at(pos) {
                    Some(t) => {
                        let t = (tick.saturating_sub(t) as f32 / fade).clamp(0.0, 1.0);
                        t * t * (3.0 - 2.0 * t)
                    }
                    None => continue,
                }
            } else {
                0.0
            };
            if k >= 1.0 {
                continue;
            }
            let o = pos.origin();
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
                    let u = unloaded_color(o.x + lx, o.y + ly);
                    let u = [inv(u[0]), inv(u[1]), inv(u[2])];
                    let c = &mut hdr[y as usize * bw + x as usize];
                    *c = [
                        u[0] + (c[0] - u[0]) * k,
                        u[1] + (c[1] - u[1]) * k,
                        u[2] + (c[2] - u[2]) * k,
                    ];
                }
            }
        }
    }
}

fn draw_overlay(sim: &Simulation, img: &mut Image, fx0: f32, fy0: f32, ppc: f32) {
    let to_px = |x: i32, y: i32| (((x as f32 - fx0) * ppc) as i32, ((y as f32 - fy0) * ppc) as i32);
    for c in sim.world.grid.chunks() {
        let o = c.pos().origin();
        let (x0, y0) = to_px(o.x, o.y);
        let (x1, y1) = to_px(o.x + CHUNK_SIZE, o.y + CHUNK_SIZE);
        // Chunks streamed in during the last moments are outlined in green.
        let fresh = sim.streamed_at(c.pos()).is_some();
        img.rect_outline(
            x0,
            y0,
            x1,
            y1,
            if fresh { [80, 230, 120, 255] } else { [60, 60, 70, 255] },
        );
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

    fn pixel(img: &Image, x: u32, y: u32) -> [u8; 3] {
        let i = ((y * img.width + x) * 4) as usize;
        [img.rgba[i], img.rgba[i + 1], img.rgba[i + 2]]
    }

    fn dark_world() -> (Arc<MaterialSet>, Simulation) {
        let mats = Arc::new(fe_materials::base_set());
        let mut sim = Simulation::new(mats.clone(), SimConfig::default());
        sim.load_rect(IRect::new(0, 0, 128, 128));
        (mats, sim)
    }

    fn spark(sim: &mut Simulation, name: &str, celsius: i32, x: i32, vx: i32) {
        let id = sim.material(name).unwrap_or_default();
        let c = Cell::new(id, fe_core::celsius_to_temp(celsius), 0);
        let (fx, fy) = (fe_core::Fx::from_int(x), fe_core::Fx::from_int(64));
        sim.particles.spawn(fx, fy, fe_core::Fx::from_int(vx), fe_core::Fx::ZERO, c);
    }

    #[test]
    fn a_cooling_spark_fades_from_white_gold_to_dull_red() {
        // A molten speck as it leaves the burst, as scrap after it froze, and
        // nearly cold, each alone in the dark.
        let look = |name: &str, celsius: i32| {
            let (mats, mut sim) = dark_world();
            spark(&mut sim, name, celsius, 64, 0);
            let r = Renderer::new(&mats);
            let s =
                RenderSettings { view_w: 128, view_h: 128, scale: 1, ..RenderSettings::default() };
            pixel(&r.render(&sim, View { cx: 64.0, cy: 64.0, zoom: 1.0 }, &s, 0), 64, 64)
        };
        let (hot, warm, dull) =
            (look("molten_metal", 1900), look("scrap", 1000), look("scrap", 600));
        // White-gold: bright in every channel.
        assert!(hot.iter().all(|&c| c > 200), "hot {hot:?}");
        // Orange: red stays high, blue is gone, green falls.
        assert!(warm[0] > 200 && warm[1] < hot[1] && warm[2] < 60, "warm {warm:?}");
        // Dull red: dimmer still, red well above green.
        assert!(dull[0] < warm[0] && dull[0] > dull[1] * 2, "dull {dull:?}");
    }

    #[test]
    fn a_fast_spark_leaves_a_streak_behind_it() {
        let (mats, mut sim) = dark_world();
        spark(&mut sim, "molten_metal", 1800, 64, 3);
        let r = Renderer::new(&mats);
        let s = RenderSettings {
            view_w: 128,
            view_h: 128,
            scale: 1,
            bloom: false,
            ..RenderSettings::default()
        };
        let img = r.render(&sim, View { cx: 64.0, cy: 64.0, zoom: 1.0 }, &s, 0);
        let lum = |p: [u8; 3]| u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2]);
        let (behind, ahead) = (lum(pixel(&img, 61, 64)), lum(pixel(&img, 67, 64)));
        assert!(behind > ahead + 60, "behind {behind}, ahead {ahead}");
    }

    #[test]
    fn many_sparks_light_the_ground_together() {
        let render = |sparks: bool| {
            let mats = Arc::new(fe_materials::base_set());
            let mut sim = Simulation::new(mats.clone(), SimConfig::default());
            sim.load_rect(IRect::new(0, 0, 256, 256));
            let stone = sim.material("stone").unwrap_or_default();
            sim.apply(&Command::FillRect { rect: IRect::new(0, 220, 256, 256), material: stone });
            if sparks {
                let id = sim.material("molten_metal").unwrap_or_default();
                for k in 0..120 {
                    let c = Cell::new(id, fe_core::celsius_to_temp(1800), 0);
                    let (x, y) = (98 + (k % 12) * 5, 40 + (k / 12) * 5);
                    let (fx, fy) = (fe_core::Fx::from_int(x), fe_core::Fx::from_int(y));
                    sim.particles.spawn(fx, fy, fe_core::Fx::ZERO, fe_core::Fx::ZERO, c);
                }
            }
            let r = Renderer::new(&mats);
            let s =
                RenderSettings { view_w: 256, view_h: 256, scale: 1, ..RenderSettings::default() };
            let img = r.render(&sim, View { cx: 128.0, cy: 128.0, zoom: 1.0 }, &s, 0);
            pixel(&img, 128, 221).iter().map(|&c| u32::from(c)).sum::<u32>()
        };
        let (dark, lit) = (render(false), render(true));
        // The ground is 150 cells below the sparks.
        assert!(lit > dark + 40, "ground {dark} without sparks, {lit} under them");
    }

    #[test]
    fn the_sky_goes_on_under_a_body() {
        let render = |carve: bool| {
            let mats = Arc::new(fe_materials::base_set());
            let mut sim = Simulation::new(mats.clone(), SimConfig::default());
            sim.load_rect(IRect::new(0, 0, 128, 128));
            let wood = sim.material("wood").unwrap_or_default();
            // The deck spans the view, so under terrain no open air carries
            // the sky in from its ends (see `reach_sky`).
            let deck = IRect::new(0, 40, 128, 44);
            sim.apply(&Command::FillRect { rect: deck, material: wood });
            if carve {
                sim.apply(&Command::CarveBody { rect: deck });
            }
            let r = Renderer::new(&mats);
            let s = RenderSettings {
                view_w: 128,
                view_h: 128,
                scale: 1,
                sky: Sky::Day,
                ..RenderSettings::default()
            };
            let img = r.render(&sim, View { cx: 64.0, cy: 64.0, zoom: 1.0 }, &s, 0);
            pixel(&img, 64, 70)
        };
        // Under terrain the air is a dark back wall; under a body it is sky.
        let (wall, sky) = (render(false), render(true));
        assert!(sky[2] > wall[2] + 60, "under terrain {wall:?}, under a body {sky:?}");
    }

    #[test]
    fn a_hold_open_at_one_end_draws_as_the_body_inside() {
        // A broken-off half of a hull: deck, bottom and one end, open at the
        // other end. Its hold is the body's inside, not sky.
        let mats = Arc::new(fe_materials::base_set());
        let mut sim = Simulation::new(mats.clone(), SimConfig::default());
        sim.load_rect(IRect::new(0, 0, 128, 128));
        let wood = sim.material("wood").unwrap_or_default();
        for rect in
            [IRect::new(20, 40, 108, 43), IRect::new(20, 60, 108, 63), IRect::new(20, 43, 24, 60)]
        {
            sim.apply(&Command::FillRect { rect, material: wood });
        }
        sim.apply(&Command::CarveBody { rect: IRect::new(20, 40, 108, 63) });
        let r = Renderer::new(&mats);
        let s = RenderSettings {
            view_w: 128,
            view_h: 128,
            scale: 1,
            sky: Sky::Day,
            ..RenderSettings::default()
        };
        let img = r.render(&sim, View { cx: 64.0, cy: 64.0, zoom: 1.0 }, &s, 0);
        let (hold, sky) = (pixel(&img, 64, 52), pixel(&img, 118, 52));
        assert!(sky[2] > hold[2] + 60, "hold {hold:?}, sky beside it {sky:?}");
    }

    #[test]
    fn a_hold_holed_in_its_floor_draws_as_the_body_inside() {
        // A closed hull with a hole through its floor. The air over the hole
        // has the deck over it but no floor under it; it is still the hold,
        // not a stripe of sky down the middle of the hull.
        let mats = Arc::new(fe_materials::base_set());
        let mut sim = Simulation::new(mats.clone(), SimConfig::default());
        sim.load_rect(IRect::new(0, 0, 128, 128));
        let wood = sim.material("wood").unwrap_or_default();
        sim.apply(&Command::FillRect { rect: IRect::new(20, 40, 108, 63), material: wood });
        for rect in [IRect::new(23, 43, 105, 60), IRect::new(60, 60, 68, 63)] {
            sim.apply(&Command::FillRect { rect, material: MaterialId::AIR });
        }
        sim.apply(&Command::CarveBody { rect: IRect::new(20, 40, 108, 63) });
        let r = Renderer::new(&mats);
        let s = RenderSettings {
            view_w: 128,
            view_h: 128,
            scale: 1,
            sky: Sky::Day,
            ..RenderSettings::default()
        };
        let img = r.render(&sim, View { cx: 64.0, cy: 64.0, zoom: 1.0 }, &s, 0);
        let (over_hole, hold) = (pixel(&img, 64, 50), pixel(&img, 40, 50));
        let sky = pixel(&img, 118, 50);
        assert!(sky[2] > over_hole[2] + 60, "over the hole {over_hole:?}, sky {sky:?}");
        assert!(over_hole[2] < hold[2] + 20, "over the hole {over_hole:?}, hold {hold:?}");
    }

    #[test]
    fn zero_view_and_extreme_zoom_do_not_panic_or_blow_up() {
        let mats = Arc::new(fe_materials::base_set());
        let mut sim = Simulation::new(mats.clone(), SimConfig::default());
        sim.load_rect(IRect::new(0, 0, 64, 64));
        let r = Renderer::new(&mats);
        let s = RenderSettings { view_w: 0, view_h: 0, scale: 0, ..RenderSettings::default() };
        let img = r.render(&sim, View { cx: 32.0, cy: 32.0, zoom: 1.0 }, &s, 0);
        assert_eq!((img.width, img.height), (1, 1));
        let z = clamp_zoom(0.05, 4096, 4096);
        let cells = (4096.0 / z) * (4096.0 / z);
        assert!(cells <= MAX_CELLS as f32 * 1.01, "{cells} cells at zoom {z}");
        assert!((clamp_zoom(0.3, 960, 540) - 0.3).abs() < 1e-6);
    }
}
