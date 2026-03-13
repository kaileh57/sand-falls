//! Presents CPU-rendered frames: each frame is copied into the surface texture.

use std::sync::Arc;
use winit::window::Window;

pub(crate) struct Gpu {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    bgra: bool,
    staging: Vec<u8>,
}

impl Gpu {
    pub(crate) fn new(window: Arc<Window>) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let size = window.inner_size();
        let surface =
            instance.create_surface(window).map_err(|e| format!("cannot create a surface: {e}"))?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .map_err(|e| format!("no graphics adapter: {e}"))?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .map_err(|e| format!("cannot open the graphics device: {e}"))?;
        let caps = surface.get_capabilities(&adapter);
        if !caps.usages.contains(wgpu::TextureUsages::COPY_DST) {
            return Err("this surface does not accept texture copies".into());
        }
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or("the surface is not supported by the adapter")?;
        config.usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST;
        config.present_mode = wgpu::PresentMode::AutoVsync;
        let bgra = matches!(
            config.format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        if !bgra
            && !matches!(
                config.format,
                wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb
            )
        {
            return Err(format!("unsupported surface format {:?}", config.format));
        }
        surface.configure(&device, &config);
        Ok(Self { surface, device, queue, config, bgra, staging: Vec::new() })
    }

    pub(crate) fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
    }

    pub(crate) fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Copies an RGBA image (smaller or equal to the surface) to the screen, centered.
    pub(crate) fn present(&mut self, img: &fe_render::Image) {
        let (w, h) = self.size();
        self.staging.clear();
        self.staging.resize(w as usize * h as usize * 4, 0);
        let ox = (w.saturating_sub(img.width) / 2) as usize;
        let oy = (h.saturating_sub(img.height) / 2) as usize;
        for y in 0..img.height.min(h) as usize {
            let src = &img.rgba[y * img.width as usize * 4..(y + 1) * img.width as usize * 4];
            let dst_row = (y + oy) * w as usize * 4;
            for x in 0..img.width.min(w) as usize {
                let s = &src[x * 4..x * 4 + 4];
                let d = dst_row + (x + ox) * 4;
                if self.bgra {
                    self.staging[d..d + 4].copy_from_slice(&[s[2], s[1], s[0], 255]);
                } else {
                    self.staging[d..d + 4].copy_from_slice(&[s[0], s[1], s[2], 255]);
                }
            }
        }
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            _ => return,
        };
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &frame.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.staging,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit([]);
        self.queue.present(frame);
    }
}
