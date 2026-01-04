/// An RGBA8 image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixels, row-major, 4 bytes each.
    pub rgba: Vec<u8>,
}

impl Image {
    /// A black, opaque image.
    pub fn new(width: u32, height: u32) -> Self {
        let mut rgba = vec![0u8; width as usize * height as usize * 4];
        for px in rgba.chunks_exact_mut(4) {
            px[3] = 255;
        }
        Self { width, height, rgba }
    }

    /// Sets one pixel, ignoring coordinates outside the image.
    pub fn put(&mut self, x: i32, y: i32, c: [u8; 4]) {
        if x < 0 || y < 0 || x as u32 >= self.width || y as u32 >= self.height {
            return;
        }
        let i = (y as usize * self.width as usize + x as usize) * 4;
        self.rgba[i..i + 4].copy_from_slice(&c);
    }

    /// Draws a one-pixel rectangle outline.
    pub fn rect_outline(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, c: [u8; 4]) {
        for x in x0..x1 {
            self.put(x, y0, c);
            self.put(x, y1 - 1, c);
        }
        for y in y0..y1 {
            self.put(x0, y, c);
            self.put(x1 - 1, y, c);
        }
    }

    /// Encodes as PNG.
    pub fn encode_png(&self) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, self.width, self.height);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            let ok = enc.write_header().and_then(|mut w| w.write_image_data(&self.rgba));
            if ok.is_err() {
                out.clear();
            }
        }
        out
    }

    /// Writes a PNG file.
    pub fn save_png(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(path, self.encode_png())
    }
}
