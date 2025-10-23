use image::{DynamicImage, GenericImageView};
use std::path::Path;

/// BGRA image structure for GPU upload
#[derive(Debug, Clone)]
pub struct BgraImage {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub pitch: usize,
}

impl BgraImage {
    /// Load image from file and convert to BGRA format
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, image::ImageError> {
        let img = image::open(path)?;
        Self::from_dynamic_image(img)
    }

    /// Convert DynamicImage to BGRA
    pub fn from_dynamic_image(img: DynamicImage) -> Result<Self, image::ImageError> {
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();
        
        // Convert RGBA to BGRA
        let mut bgra_data = Vec::with_capacity((width * height * 4) as usize);
        for pixel in rgba.pixels() {
            bgra_data.push(pixel[2]); // B
            bgra_data.push(pixel[1]); // G
            bgra_data.push(pixel[0]); // R
            bgra_data.push(pixel[3]); // A
        }
        
        Ok(BgraImage {
            data: bgra_data,
            width,
            height,
            pitch: (width * 4) as usize,
        })
    }

    /// Create solid color BGRA image
    pub fn solid_color(width: u32, height: u32, b: u8, g: u8, r: u8, a: u8) -> Self {
        let size = (width * height * 4) as usize;
        let mut data = Vec::with_capacity(size);
        
        for _ in 0..(width * height) {
            data.push(b);
            data.push(g);
            data.push(r);
            data.push(a);
        }
        
        BgraImage {
            data,
            width,
            height,
            pitch: (width * 4) as usize,
        }
    }

    /// Get pixel at position (x, y)
    pub fn get_pixel(&self, x: u32, y: u32) -> Option<(u8, u8, u8, u8)> {
        if x >= self.width || y >= self.height {
            return None;
        }
        
        let idx = (y * self.width + x) as usize * 4;
        Some((
            self.data[idx],     // B
            self.data[idx + 1], // G
            self.data[idx + 2], // R
            self.data[idx + 3], // A
        ))
    }

    /// Set pixel at position (x, y)
    pub fn set_pixel(&mut self, x: u32, y: u32, b: u8, g: u8, r: u8, a: u8) {
        if x >= self.width || y >= self.height {
            return;
        }
        
        let idx = (y * self.width + x) as usize * 4;
        self.data[idx] = b;
        self.data[idx + 1] = g;
        self.data[idx + 2] = r;
        self.data[idx + 3] = a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solid_color() {
        let img = BgraImage::solid_color(100, 100, 255, 0, 0, 255);
        assert_eq!(img.width, 100);
        assert_eq!(img.height, 100);
        assert_eq!(img.data.len(), 100 * 100 * 4);
        
        let pixel = img.get_pixel(50, 50).unwrap();
        assert_eq!(pixel, (255, 0, 0, 255)); // Blue
    }

    #[test]
    fn test_set_get_pixel() {
        let mut img = BgraImage::solid_color(10, 10, 0, 0, 0, 255);
        img.set_pixel(5, 5, 100, 150, 200, 255);
        
        let pixel = img.get_pixel(5, 5).unwrap();
        assert_eq!(pixel, (100, 150, 200, 255));
    }
}
