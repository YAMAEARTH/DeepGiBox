/// Test compositor without DeckLink hardware
/// Creates fake video frames and composites with foreground.png

use anyhow::Result;
use common_io::{ColorSpace, FrameMeta, MemLoc, MemRef, PixelFormat, RawFramePacket};
use decklink_output::CompositorBuilder;
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== Compositor Test (No Hardware Required) ===\n");

    let width = 1920u32;
    let height = 1080u32;

    // Check if foreground.png exists
    if !std::path::Path::new("foreground.png").exists() {
        println!("⚠️  foreground.png not found!");
        println!("Creating test overlay image...\n");
        create_test_overlay(width, height)?;
    }

    // Create compositor with PNG
    println!("Loading compositor with foreground.png...");
    let mut compositor = CompositorBuilder::new(width, height)
        .with_png("foreground.png")
        .build()?;
    println!("✓ Compositor ready\n");

    // Create fake UYVY video frame on GPU
    println!("Creating test video frame (UYVY on GPU)...");
    let video_frame = create_fake_uyvy_frame(width, height)?;
    println!("✓ Test frame ready: {}x{} UYVY on GPU\n", width, height);

    // Warm-up run
    println!("Warming up GPU...");
    for _ in 0..10 {
        compositor.composite(&video_frame)?;
    }
    println!("✓ Warm-up complete\n");

    // Benchmark
    println!("Running benchmark (100 frames)...");
    let mut times = Vec::new();
    
    for i in 0..100 {
        let start = Instant::now();
        let _output = compositor.composite(&video_frame)?;
        let elapsed = start.elapsed();
        times.push(elapsed.as_micros() as f64 / 1000.0);
        
        if (i + 1) % 20 == 0 {
            println!("  Processed {} frames...", i + 1);
        }
    }

    // Statistics
    let total: f64 = times.iter().sum();
    let avg = total / times.len() as f64;
    let min = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    println!("\n=== Results ===");
    println!("Frames processed: {}", times.len());
    println!("Average time: {:.2} ms", avg);
    println!("Min time: {:.2} ms", min);
    println!("Max time: {:.2} ms", max);
    println!("Theoretical FPS: {:.1}", 1000.0 / avg);

    // Save output
    println!("\nDownloading composited frame from GPU...");
    let output_data = compositor.download_output()?;
    
    println!("Saving to test_output.png...");
    save_bgra_as_png(&output_data, (width, height), "test_output.png")?;
    
    println!("\n✅ Success! Check test_output.png");
    println!("   - Original: foreground.png");
    println!("   - Output: test_output.png (composited)");

    Ok(())
}

/// Create fake UYVY frame on GPU (simulates DeckLink capture)
fn create_fake_uyvy_frame(width: u32, height: u32) -> Result<RawFramePacket> {
    use std::os::raw::c_void;

    // Calculate size
    let stride = (width * 2) as usize; // UYVY = 2 bytes per pixel
    let size = stride * height as usize;

    // Allocate GPU memory
    let mut dev_ptr: *mut c_void = std::ptr::null_mut();
    unsafe {
        let ret = cudaMalloc(&mut dev_ptr, size);
        if ret != 0 {
            anyhow::bail!("cudaMalloc failed: {}", ret);
        }
    }

    // Create gradient pattern on CPU
    let mut cpu_data = vec![0u8; size];
    for y in 0..height {
        for x in 0..width {
            let idx = (y as usize * stride) + (x as usize / 2) * 4;
            
            // Create colorful gradient
            let u = ((x as f32 / width as f32) * 255.0) as u8;
            let v = ((y as f32 / height as f32) * 255.0) as u8;
            let y_val = 128u8; // Mid-gray
            
            cpu_data[idx] = u;          // U
            cpu_data[idx + 1] = y_val;  // Y0
            cpu_data[idx + 2] = v;      // V
            cpu_data[idx + 3] = y_val;  // Y1
        }
    }

    // Upload to GPU
    unsafe {
        let ret = cudaMemcpy(
            dev_ptr,
            cpu_data.as_ptr() as *const c_void,
            size,
            1, // cudaMemcpyHostToDevice
        );
        if ret != 0 {
            cudaFree(dev_ptr);
            anyhow::bail!("cudaMemcpy failed: {}", ret);
        }
    }

    Ok(RawFramePacket {
        meta: FrameMeta {
            source_id: 0,
            width,
            height,
            pixfmt: PixelFormat::YUV422_8,
            colorspace: ColorSpace::BT709,
            frame_idx: 0,
            pts_ns: 0,
            t_capture_ns: 0,
            stride_bytes: stride as u32,
        },
        data: MemRef {
            ptr: dev_ptr as *mut u8,
            len: size,
            stride,
            loc: MemLoc::Gpu { device: 0 },
        },
    })
}

/// Create test overlay PNG
fn create_test_overlay(width: u32, height: u32) -> Result<()> {
    use image::{ImageBuffer, Rgba, RgbaImage};

    let mut img: RgbaImage = ImageBuffer::new(width, height);

    // Transparent background
    for pixel in img.pixels_mut() {
        *pixel = Rgba([0, 0, 0, 0]);
    }

    // Draw red rectangle with semi-transparent fill
    let rect_x = width / 4;
    let rect_y = height / 4;
    let rect_w = width / 2;
    let rect_h = height / 2;

    for y in rect_y..(rect_y + rect_h) {
        for x in rect_x..(rect_x + rect_w) {
            // Border
            if x == rect_x || x == rect_x + rect_w - 1 || 
               y == rect_y || y == rect_y + rect_h - 1 {
                img.put_pixel(x, y, Rgba([255, 0, 0, 255])); // Red border
            } else {
                // Semi-transparent fill
                img.put_pixel(x, y, Rgba([255, 0, 0, 128])); // 50% transparent red
            }
        }
    }

    // Add text-like pattern
    let text_y = height / 2;
    for x in (width / 3)..(2 * width / 3) {
        for dy in 0..20 {
            let y = text_y + dy;
            if y < height {
                img.put_pixel(x, y, Rgba([255, 255, 0, 255])); // Yellow
            }
        }
    }

    img.save("foreground.png")?;
    println!("✓ Created test overlay: foreground.png");
    
    Ok(())
}

fn save_bgra_as_png(bgra_data: &[u8], (width, height): (u32, u32), path: &str) -> Result<()> {
    use image::{ImageBuffer, RgbaImage};

    // Convert BGRA to RGBA
    let mut rgba_data = Vec::with_capacity(bgra_data.len());
    for chunk in bgra_data.chunks_exact(4) {
        rgba_data.push(chunk[2]); // R
        rgba_data.push(chunk[1]); // G
        rgba_data.push(chunk[0]); // B
        rgba_data.push(chunk[3]); // A
    }

    let img: RgbaImage = ImageBuffer::from_raw(width, height, rgba_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;

    img.save(path)?;
    Ok(())
}

// CUDA FFI
extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut std::os::raw::c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut std::os::raw::c_void) -> i32;
    fn cudaMemcpy(
        dst: *mut std::os::raw::c_void,
        src: *const std::os::raw::c_void,
        count: usize,
        kind: i32,
    ) -> i32;
}
