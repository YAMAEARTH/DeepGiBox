// Simple Hardware Internal Keying Demo
// Uses DeckLink's hardware keyer to composite PNG overlay over SDI input
//
// Usage: internal_keying_simple foreground.png

use anyhow::{Context, Result};
use image::GenericImageView;
use std::thread;
use std::time::{Duration, Instant};

// FFI to shim.cpp
extern "C" {
    fn decklink_capture_open(device_index: i32) -> bool;
    fn decklink_capture_close();
    
    fn decklink_output_open(device_index: i32, width: i32, height: i32, fps: f64) -> bool;
    fn decklink_output_send_frame(bgra_data: *const u8, width: i32, height: i32) -> bool;
    fn decklink_output_close();
    
    // Get detected input format (for internal keying sync)
    fn decklink_get_detected_input_format(
        out_width: *mut i32, 
        out_height: *mut i32, 
        out_fps: *mut f64,
        out_mode: *mut u32
    ) -> bool;
    
    // New keyer functions
    fn decklink_keyer_enable_internal() -> bool;
    fn decklink_keyer_set_level(level: u8) -> bool;
    fn decklink_keyer_disable() -> bool;
    fn decklink_set_video_output_connection(connection: i64) -> bool;
    fn decklink_get_connection_sdi() -> i64;
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("🎬 Simple Hardware Internal Keying Demo\n");
        eprintln!("Usage: {} <foreground.png>", args[0]);
        eprintln!("\nExample:");
        eprintln!("  {} overlay.png", args[0]);
        eprintln!("\nNote: Requires DeckLink card with hardware keyer support");
        return Ok(());
    }

    let png_path = &args[1];
    
    println!("🎬 Hardware Internal Keying Demo");
    println!("=================================\n");

    // Step 1: Load PNG overlay
    println!("📸 Loading PNG overlay: {}", png_path);
    let overlay = load_png_bgra(png_path)?;
    println!("   ✓ Loaded: {}x{} pixels\n", overlay.width, overlay.height);

    let device_idx = 0;

    // Step 2: Open capture (SDI input)
    println!("🎥 Opening DeckLink capture...");
    unsafe {
        if !decklink_capture_open(device_idx) {
            anyhow::bail!("Failed to open DeckLink capture");
        }
    }
    println!("   ✓ Capture opened on device {}\n", device_idx);

    // Step 3: Wait for input signal to stabilize and detect format
    println!("⏳ Waiting for input signal and format detection...");
    
    let (width, height, fps) = {
        let mut retries = 0;
        loop {
            thread::sleep(Duration::from_millis(100));
            
            let mut w = 0i32;
            let mut h = 0i32;
            let mut f = 0.0f64;
            let mut mode = 0u32;
            
            unsafe {
                if decklink_get_detected_input_format(&mut w, &mut h, &mut f, &mut mode) {
                    println!("   ✓ Detected: {}x{}@{:.3}fps (mode=0x{:x})", w, h, f, mode);
                    break (w, h, f);
                }
            }
            
            retries += 1;
            if retries > 50 {
                eprintln!("   ⚠ Warning: Could not detect input format after 5 seconds");
                eprintln!("   Using PNG dimensions: {}x{} @ 60fps (may not match input!)", 
                    overlay.width, overlay.height);
                break (overlay.width as i32, overlay.height as i32, 60.0);
            }
        }
    };
    
    println!();

    // Step 4: Configure output connection to SDI
    println!("📡 Configuring SDI output...");
    
    // Step 5: Open output (MUST use exact detected format for internal keying!)
    println!("   Opening output: {}x{}@{:.3}fps (matching input for internal keying)", 
        width, height, fps);
    
    unsafe {
        if !decklink_output_open(device_idx, width, height, fps) {
            anyhow::bail!("Failed to open DeckLink output");
        }
    }
    println!("   ✓ Output opened\n");
    
    // Step 6: Set SDI output connection (after output is open)
    unsafe {
        let sdi_conn = decklink_get_connection_sdi();
        if !decklink_set_video_output_connection(sdi_conn) {
            eprintln!("   ⚠ Warning: Could not set SDI output connection");
        }
    }

        // Step 8: Resize overlay if needed
    let overlay_resized = if overlay.width != width as u32 || overlay.height != height as u32 {
        println!("📐 Resizing overlay: {}x{} → {}x{}", 
            overlay.width, overlay.height, width, height);
        resize_bgra(&overlay, width as u32, height as u32)?
    } else {
        overlay
    };
    println!();

    // Step 9: Send overlay frames continuously
    println!("🔑 Enabling hardware keyer...");
    unsafe {
        if !decklink_keyer_enable_internal() {
            eprintln!("   ⚠ Warning: Failed to enable internal keyer");
            eprintln!("   This device may not support hardware keying.");
            eprintln!("   Continuing anyway (overlay will show without input background)...\n");
        } else {
            println!("   ✓ Internal keyer enabled");
            
            // Set keyer level to maximum (fully visible overlay)
            if !decklink_keyer_set_level(255) {
                eprintln!("   ⚠ Warning: Could not set keyer level");
            } else {
                println!("   ✓ Keyer level: 255 (fully visible)\n");
            }
        }
    }

    // Step 8: Resize overlay if needed
    // Step 9: Send overlay frames continuously
    println!("▶️  Sending overlay frames (Ctrl+C to stop)");
    println!("─────────────────────────────────────────");
    println!("   Mode:    Hardware Internal Keying");
    println!("   Input:   {}x{}@{:.3}fps (SDI)", width, height, fps);
    println!("   Overlay: {}x{}", overlay_resized.width, overlay_resized.height);
    println!("   Output:  {}x{}@{:.3}fps (SDI composited)", width, height, fps);
    println!("─────────────────────────────────────────\n");

    let frame_interval = Duration::from_secs_f64(1.0 / fps);
    let mut frame_count = 0u64;
    let start_time = Instant::now();
    let mut last_print = Instant::now();

    loop {
        let frame_start = Instant::now();

        // Send overlay frame to DeckLink
        // Hardware keyer will composite it with SDI input automatically
        unsafe {
            if !decklink_output_send_frame(
                overlay_resized.data.as_ptr(),
                overlay_resized.width as i32,
                overlay_resized.height as i32,
            ) {
                eprintln!("⚠ Warning: Failed to send frame");
            }
        }

        frame_count += 1;

        // Print stats every second
        if last_print.elapsed() >= Duration::from_secs(1) {
            let elapsed = start_time.elapsed().as_secs_f64();
            let current_fps = frame_count as f64 / elapsed;
            
            println!("Frame #{:6} | FPS: {:.2} | Input+Overlay→SDI (Hardware Keyer)", 
                frame_count, current_fps);
            
            last_print = Instant::now();
        }

        // Sleep to match frame rate
        let elapsed = frame_start.elapsed();
        if elapsed < frame_interval {
            thread::sleep(frame_interval - elapsed);
        }
    }
}

// Cleanup on drop
struct CleanupGuard;

impl Drop for CleanupGuard {
    fn drop(&mut self) {
        unsafe {
            println!("\n🛑 Shutting down...");
            decklink_keyer_disable();
            decklink_output_close();
            decklink_capture_close();
            println!("   ✓ Cleanup complete");
        }
    }
}

// Helper: Load PNG and convert to BGRA
struct BgraImage {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

fn load_png_bgra(path: &str) -> Result<BgraImage> {
    let img = image::open(path)
        .with_context(|| format!("Failed to load PNG: {}", path))?;
    
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    
    // Convert RGBA → BGRA
    let mut bgra = Vec::with_capacity((width * height * 4) as usize);
    for pixel in rgba.pixels() {
        bgra.push(pixel[2]); // B
        bgra.push(pixel[1]); // G
        bgra.push(pixel[0]); // R
        bgra.push(pixel[3]); // A
    }
    
    Ok(BgraImage {
        width,
        height,
        data: bgra,
    })
}

fn resize_bgra(img: &BgraImage, new_width: u32, new_height: u32) -> Result<BgraImage> {
    use image::{RgbaImage, imageops};
    
    // Convert BGRA → RGBA
    let mut rgba_data = Vec::with_capacity(img.data.len());
    for chunk in img.data.chunks_exact(4) {
        rgba_data.push(chunk[2]); // R
        rgba_data.push(chunk[1]); // G
        rgba_data.push(chunk[0]); // B
        rgba_data.push(chunk[3]); // A
    }
    
    let rgba_img = RgbaImage::from_raw(img.width, img.height, rgba_data)
        .context("Failed to create RGBA image from BGRA data")?;
    
    let resized = imageops::resize(&rgba_img, new_width, new_height, imageops::FilterType::Lanczos3);
    
    // Convert back to BGRA
    let mut bgra = Vec::with_capacity((new_width * new_height * 4) as usize);
    for pixel in resized.pixels() {
        bgra.push(pixel[2]); // B
        bgra.push(pixel[1]); // G
        bgra.push(pixel[0]); // R
        bgra.push(pixel[3]); // A
    }
    
    Ok(BgraImage {
        width: new_width,
        height: new_height,
        data: bgra,
    })
}

