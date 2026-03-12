//! The windowed client.

use crate::gpu::Gpu;
use crate::{opt, threads};
use fe_core::MaterialId;
use fe_engine::{Avatar, Command, PlayerInput, Simulation, tools};
use fe_net::client::{ClientCore, ClientPhase};
use fe_net::{Peer, TICK_US, UdpTransport};
use fe_render::{RenderSettings, Renderer, Sky, View};
use fe_scene::{Runner, Scene};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const PAINTS: &[&str] = &[
    "sand",
    "water",
    "oil",
    "lava",
    "wood",
    "stone",
    "gunpowder",
    "acid",
    "ice",
    "glass",
    "brick",
    "tnt",
];
/// Screen pixels per cell at zoom 1.
const PIXELS_PER_CELL: u32 = 2;

enum Mode {
    Local(Box<Runner>),
    Net { client: Box<ClientCore>, net: UdpTransport },
}

#[derive(Default)]
struct Keys {
    left: bool,
    right: bool,
    jump: bool,
    mouse: (f64, f64),
    use_tool: bool,
}

struct App {
    mode: Mode,
    scene_path: PathBuf,
    threads: usize,
    window: Option<Arc<Window>>,
    gpu: Option<Gpu>,
    renderer: Renderer,
    settings: RenderSettings,
    start: Instant,
    next_tick: Duration,
    keys: Keys,
    tool: u8,
    paint: usize,
    palette: Vec<MaterialId>,
    zoom: f32,
    paused: bool,
    cam: (f32, f32),
    player: u8,
    frames: u64,
    fps: f32,
    last_fps: Instant,
    error: Option<String>,
}

fn load_scene(path: &str) -> Result<(String, Scene, Arc<fe_materials::MaterialSet>), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let scene = fe_scene::parse(&text).map_err(|e| format!("{path}: {e}"))?;
    let dir = Path::new(path).parent().map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    let mats =
        Arc::new(fe_scene::load_materials(&scene, &dir).map_err(|e| format!("{path}: {e}"))?);
    Ok((text, scene, mats))
}

fn local_runner(path: &str, threads: usize) -> Result<Runner, String> {
    let (_, scene, mats) = load_scene(path)?;
    let mut r = Runner::new(scene, mats, threads).map_err(|e| e.to_string())?;
    if r.sim.avatars.is_empty() {
        let b = r.scene.bounds;
        let (x, y) = r.scene.spawn.unwrap_or((i32::midpoint(b.min.x, b.max.x), b.min.y + 20));
        r.sim.apply(&Command::AddPlayer { id: 0, x, y });
    }
    Ok(r)
}

fn run(app: &mut App) -> Result<(), String> {
    let event_loop = EventLoop::new().map_err(|e| format!("no display available: {e}"))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(app).map_err(|e| e.to_string())?;
    match app.error.take() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn new_app(
    mode: Mode,
    scene_path: &str,
    threads: usize,
    scene: &Scene,
    mats: &fe_materials::MaterialSet,
) -> App {
    let palette = PAINTS.iter().filter_map(|n| mats.id(n)).collect();
    let cam = &scene.camera;
    let settings = RenderSettings {
        sky: Sky::parse(&cam.sky).unwrap_or(Sky::Cave),
        ambient: cam.ambient,
        exposure: cam.exposure,
        ..RenderSettings::default()
    };
    let b = scene.bounds;
    App {
        mode,
        scene_path: PathBuf::from(scene_path),
        threads,
        window: None,
        gpu: None,
        renderer: Renderer::new(mats),
        settings,
        start: Instant::now(),
        next_tick: Duration::ZERO,
        keys: Keys::default(),
        tool: tools::DIG,
        paint: 0,
        palette,
        zoom: 1.0,
        paused: false,
        cam: ((b.min.x + b.max.x) as f32 / 2.0, (b.min.y + b.max.y) as f32 / 2.0),
        player: 0,
        frames: 0,
        fps: 0.0,
        last_fps: Instant::now(),
        error: None,
    }
}

/// `fe-client play <scene>`.
pub(crate) fn play(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("missing scene path")?;
    let t = threads(args);
    let runner = local_runner(path, t)?;
    let scene = runner.scene.clone();
    let mats = runner.sim.materials().clone();
    let mut app = new_app(Mode::Local(Box::new(runner)), path, t, &scene, &mats);
    run(&mut app)
}

/// `fe-client connect <addr> --scene <file>`.
pub(crate) fn connect(args: &[String]) -> Result<(), String> {
    let addr = args.first().ok_or("missing host:port")?;
    let scene_path =
        opt(args, "--scene").ok_or("--scene FILE is required (it must match the server's)")?;
    let name = opt(args, "--name").unwrap_or("player");
    let t = threads(args);
    let (text, scene, mats) = load_scene(scene_path)?;
    let mut net = UdpTransport::bind("0.0.0.0:0").map_err(|e| e.to_string())?;
    let server_addr = std::net::ToSocketAddrs::to_socket_addrs(addr.as_str())
        .map_err(|e| format!("{addr}: {e}"))?
        .next()
        .ok_or_else(|| format!("{addr}: no address"))?;
    let server: Peer = net.peer_for(server_addr);
    let mut client = ClientCore::new(&text, scene.clone(), mats.clone(), server, name, t);
    client.connect(0, &mut net);
    let mut app =
        new_app(Mode::Net { client: Box::new(client), net }, scene_path, t, &scene, &mats);
    run(&mut app)
}

impl App {
    fn sim(&self) -> Option<&Simulation> {
        match &self.mode {
            Mode::Local(r) => Some(&r.sim),
            Mode::Net { client, .. } => client.sim(),
        }
    }

    fn me(&self) -> Option<Avatar> {
        match &self.mode {
            Mode::Local(r) => r.sim.avatars.iter().find(|a| a.id == self.player).copied(),
            Mode::Net { client, .. } => client.predicted().copied(),
        }
    }

    fn view(&self) -> View {
        View { cx: self.cam.0, cy: self.cam.1, zoom: self.zoom }
    }

    fn input(&self) -> PlayerInput {
        let mut i = PlayerInput {
            move_x: i8::from(self.keys.right) - i8::from(self.keys.left),
            jump: self.keys.jump,
            tool: self.tool,
            use_tool: self.keys.use_tool,
            ..Default::default()
        };
        if self.tool == tools::PAINT {
            i.param = self.palette.get(self.paint).map_or(0, |m| m.0);
        } else if self.tool == tools::DIG {
            i.param = 6;
        }
        if let (Some(me), Some(gpu)) = (self.me(), self.gpu.as_ref()) {
            let (w, h) = gpu.size();
            let ppc = PIXELS_PER_CELL as f32 * self.zoom;
            let wx = self.cam.0 + (self.keys.mouse.0 as f32 - w as f32 / 2.0) / ppc;
            let wy = self.cam.1 + (self.keys.mouse.1 as f32 - h as f32 / 2.0) / ppc;
            let c = me.center();
            i.aim_x = (wx - c.x as f32).clamp(-2000.0, 2000.0) as i16;
            i.aim_y = (wy - c.y as f32).clamp(-2000.0, 2000.0) as i16;
        }
        i
    }

    fn tick(&mut self) {
        let input = self.input();
        let now_us = self.start.elapsed().as_micros() as u64;
        match &mut self.mode {
            Mode::Local(r) => {
                if !self.paused {
                    let mut inputs = BTreeMap::new();
                    inputs.insert(self.player, input);
                    r.step_with(&inputs, &[]);
                }
            }
            Mode::Net { client, net } => {
                client.poll(now_us, net);
                if let Some(p) = client.player {
                    self.player = p;
                }
                client.set_input(input);
                client.tick(now_us, net);
            }
        }
        if let Some(me) = self.me() {
            let target = (me.x.0 as f32 / 65536.0, me.y.0 as f32 / 65536.0 - 6.0);
            self.cam.0 += (target.0 - self.cam.0) * 0.12;
            self.cam.1 += (target.1 - self.cam.1) * 0.12;
        }
    }

    fn hud(&self, img: &mut fe_render::Image) {
        let mut lines = Vec::new();
        let tool = match self.tool {
            tools::DIG => "DIG".to_string(),
            tools::PAINT => {
                let name = self.sim().and_then(|s| {
                    self.palette.get(self.paint).map(|m| s.materials().get(*m).name.clone())
                });
                format!("PAINT {}", name.unwrap_or_default())
            }
            tools::EXPLODE => "EXPLODE".into(),
            tools::HEAT => "HEAT".into(),
            tools::FREEZE => "FREEZE".into(),
            tools::CRATE => "CRATE".into(),
            _ => "NONE".into(),
        };
        if let Some(s) = self.sim() {
            let r = s.report();
            lines.push(format!(
                "TICK {}  FPS {:.0}  SIM {:.1} MS",
                s.tick(),
                self.fps,
                r.times.total as f64 / 1000.0
            ));
            lines.push(format!(
                "BODIES {}  PARTICLES {}  ACTIVE CHUNKS {}",
                r.bodies, r.particles, r.active_chunks
            ));
            lines.push(format!("TOOL {tool}"));
        }
        match &self.mode {
            Mode::Net { client, .. } => {
                let st = &client.stats;
                let secs = self.start.elapsed().as_secs_f64().max(1.0);
                lines.push(match &client.phase {
                    ClientPhase::Connecting => "CONNECTING".into(),
                    ClientPhase::Loading => "LOADING WORLD".into(),
                    ClientPhase::Playing => format!(
                        "PLAYER {}  RTT {} MS  LEAD {} TICKS",
                        self.player,
                        st.rtt_us / 1000,
                        st.lead_ticks
                    ),
                    ClientPhase::Rejected(r) => format!("REJECTED: {r}"),
                });
                lines.push(format!(
                    "UP {:.0} B/S  DOWN {:.0} B/S  REPAIRS {}",
                    st.traffic.sent as f64 / secs,
                    st.traffic.received as f64 / secs,
                    st.repairs
                ));
                if let Some(s) = client.sim() {
                    lines.push(format!("HASH {:016X}", s.hash()));
                }
            }
            Mode::Local(_) if self.paused => lines.push("PAUSED".into()),
            Mode::Local(_) => {}
        }
        img.shade(0, 0, 430, 12 + 10 * lines.len() as i32, 140);
        for (i, l) in lines.iter().enumerate() {
            img.text(6, 6 + 10 * i as i32, 1, l, [235, 235, 235, 255]);
        }
    }

    fn redraw(&mut self) {
        let Some((w, h)) = self.gpu.as_ref().map(Gpu::size) else { return };
        let mut s = self.settings.clone();
        s.scale = PIXELS_PER_CELL;
        s.view_w = (w / PIXELS_PER_CELL).max(16);
        s.view_h = (h / PIXELS_PER_CELL).max(16);
        let view = self.view();
        let mut img = match &self.mode {
            Mode::Local(r) => self.renderer.render(&r.sim, view, &s, r.sim.tick()),
            Mode::Net { client, .. } => match client.sim() {
                Some(sim) => {
                    let mut avatars = sim.avatars.clone();
                    if let Some(p) = client.predicted() {
                        for a in &mut avatars {
                            if a.id == p.id {
                                *a = *p;
                            }
                        }
                    }
                    self.renderer.render_with(sim, view, &s, sim.tick(), &avatars)
                }
                None => fe_render::Image::new(s.view_w * s.scale, s.view_h * s.scale),
            },
        };
        self.hud(&mut img);
        self.frames += 1;
        let dt = self.last_fps.elapsed().as_secs_f32();
        if dt > 0.5 {
            self.fps = self.frames as f32 / dt;
            self.frames = 0;
            self.last_fps = Instant::now();
        }
        if let Some(g) = self.gpu.as_mut() {
            g.present(&img);
        }
    }

    fn key(&mut self, code: KeyCode, down: bool, el: &ActiveEventLoop) {
        match code {
            KeyCode::KeyA | KeyCode::ArrowLeft => self.keys.left = down,
            KeyCode::KeyD | KeyCode::ArrowRight => self.keys.right = down,
            KeyCode::KeyW | KeyCode::Space | KeyCode::ArrowUp => self.keys.jump = down,
            _ if !down => {}
            KeyCode::Digit1 => self.tool = tools::DIG,
            KeyCode::Digit2 => self.tool = tools::PAINT,
            KeyCode::Digit3 => self.tool = tools::EXPLODE,
            KeyCode::Digit4 => self.tool = tools::HEAT,
            KeyCode::Digit5 => self.tool = tools::FREEZE,
            KeyCode::Digit6 => self.tool = tools::CRATE,
            KeyCode::KeyQ => {
                self.paint =
                    (self.paint + self.palette.len().max(1) - 1) % self.palette.len().max(1);
            }
            KeyCode::KeyE => self.paint = (self.paint + 1) % self.palette.len().max(1),
            KeyCode::Equal | KeyCode::NumpadAdd => self.zoom = (self.zoom * 1.25).min(8.0),
            KeyCode::Minus | KeyCode::NumpadSubtract => self.zoom = (self.zoom / 1.25).max(0.125),
            KeyCode::F1 => self.settings.overlay = !self.settings.overlay,
            KeyCode::F2 => self.settings.lighting = !self.settings.lighting,
            KeyCode::KeyP => self.paused = !self.paused,
            KeyCode::KeyR => {
                if let Mode::Local(_) = self.mode
                    && let Ok(r) =
                        local_runner(&self.scene_path.display().to_string(), self.threads)
                {
                    self.mode = Mode::Local(Box::new(r));
                }
            }
            KeyCode::F12 => {
                if let Some(sim) = self.sim() {
                    let s = RenderSettings { scale: 2, ..self.settings.clone() };
                    let img = self.renderer.render(sim, self.view(), &s, sim.tick());
                    let path = format!("fe-screenshot-{}.png", sim.tick());
                    if img.save_png(Path::new(&path)).is_ok() {
                        eprintln!("saved {path}");
                    }
                }
            }
            KeyCode::Escape => el.exit(),
            _ => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("falling-engine")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
        let window = match el.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                self.error = Some(format!("cannot open a window: {e}"));
                el.exit();
                return;
            }
        };
        match Gpu::new(window.clone()) {
            Ok(g) => self.gpu = Some(g),
            Err(e) => {
                self.error = Some(e);
                el.exit();
                return;
            }
        }
        self.window = Some(window);
        self.start = Instant::now();
        self.next_tick = Duration::ZERO;
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                if let Mode::Net { client, net } = &mut self.mode {
                    client.disconnect(net);
                }
                el.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(g) = self.gpu.as_mut() {
                    g.resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.key(code, event.state == ElementState::Pressed, el);
                }
            }
            WindowEvent::CursorMoved { position, .. } => self.keys.mouse = (position.x, position.y),
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.keys.use_tool = state == ElementState::Pressed;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let d = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y / 40.0) as f32,
                };
                let n = self.palette.len().max(1);
                if d > 0.0 {
                    self.paint = (self.paint + 1) % n;
                } else if d < 0.0 {
                    self.paint = (self.paint + n - 1) % n;
                }
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        let mut ran = 0;
        while self.start.elapsed() >= self.next_tick && ran < 4 {
            self.tick();
            self.next_tick += Duration::from_micros(TICK_US);
            ran += 1;
        }
        if ran == 4 {
            // Too slow to keep up: drop the backlog rather than spiral.
            self.next_tick = self.start.elapsed();
        }
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}
