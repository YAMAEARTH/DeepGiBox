//! Test BGRA text rendering with rusttype

use anyhow::Result;
use rusttype::{Font, Scale, point};
use std::fs;

fn main() -> Result<()> {
    println!("🎨 Testing BGRA Text Rendering...");
    
    // Create a 1920x1080 BGRA buffer (black/transparent background)
    let width = 1920usize;
    let height = 1080usize;
    let stride = width * 4;
    let mut buffer = vec![0u8; height * stride];
    
    // Load font
    let font_data = include_bytes!("../../../../testsupport/DejaVuSans.ttf");
    let font = Font::try_from_bytes(font_data as &[u8]).unwrap();
    
    // Draw some test text
    draw_text(&mut buffer, stride, width, height, 100, 100, "Hello, World!", 48.0, (255, 255, 255, 255));
    draw_text(&mut buffer, stride, width, height, 100, 200, "Neo 0.54 ID:1", 32.0, (0, 0, 255, 255)); // Red
    draw_text(&mut buffer, stride, width, height, 100, 300, "27/11/2020", 36.0, (255, 255, 255, 255));
    draw_text(&mut buffer, stride, width, height, 100, 400, "EGD", 48.0, (255, 255, 255, 255));
    draw_text(&mut buffer, stride, width, height, 100, 500, "Neoplastic", 40.0, (0, 0, 255, 255));
    
    // Save as raw BGRA
    fs::create_dir_all("output/test")?;
    fs::write("output/test/test_text_bgra.bin", &buffer)?;
    println!("✅ Saved to output/test/test_text_bgra.bin");
    
    // Convert to PNG for easy viewing
    let mut img = image::RgbaImage::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let idx = y * stride + x * 4;
            let b = buffer[idx + 0];
            let g = buffer[idx + 1];
            let r = buffer[idx + 2];
            let a = buffer[idx + 3];
            img.put_pixel(x as u32, y as u32, image::Rgba([r, g, b, a]));
        }
    }
    img.save("output/test/test_text.png")?;
    println!("✅ Saved PNG to output/test/test_text.png");
    
    // Statistics
    let non_zero = buffer.iter().filter(|&&b| b != 0).count();
    println!("📊 Non-zero bytes: {} / {} ({:.2}%)", 
             non_zero, buffer.len(), 100.0 * non_zero as f64 / buffer.len() as f64);
    
    Ok(())
}

fn draw_text(
    buf: &mut [u8], 
    stride: usize, 
    w: usize, 
    h: usize, 
    x: i32, 
    y: i32, 
    text: &str, 
    font_size: f32,
    color: (u8, u8, u8, u8) // BGRA
) {
    let font_data = include_bytes!("../../../../testsupport/DejaVuSans.ttf");
    let font = Font::try_from_bytes(font_data as &[u8]).unwrap();
    
    let scale = Scale::uniform(font_size);
    let v_metrics = font.v_metrics(scale);
    let offset = point(0.0, v_metrics.ascent);
    
    let glyphs: Vec<_> = font.layout(text, scale, offset).collect();
    
    for glyph in glyphs {
        if let Some(bounding_box) = glyph.pixel_bounding_box() {
            glyph.draw(|gx, gy, gv| {
                let px = x + bounding_box.min.x + gx as i32;
                let py = y + bounding_box.min.y + gy as i32;
                
                if px >= 0 && py >= 0 && (px as usize) < w && (py as usize) < h {
                    let idx = (py as usize) * stride + (px as usize) * 4;
                    if idx + 3 < buf.len() {
                        // Alpha blend with existing pixel
                        let alpha = (gv * 255.0) as u32;
                        let inv_alpha = 255 - alpha;
                        
                        buf[idx + 0] = ((color.0 as u32 * alpha + buf[idx + 0] as u32 * inv_alpha) / 255) as u8; // B
                        buf[idx + 1] = ((color.1 as u32 * alpha + buf[idx + 1] as u32 * inv_alpha) / 255) as u8; // G
                        buf[idx + 2] = ((color.2 as u32 * alpha + buf[idx + 2] as u32 * inv_alpha) / 255) as u8; // R
                        buf[idx + 3] = buf[idx + 3].max(((color.3 as u32 * alpha) / 255) as u8); // A
                    }
                }
            });
        }
    }
}
