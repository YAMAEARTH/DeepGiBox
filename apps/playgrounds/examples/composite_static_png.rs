/// Example: Using DeckLink Output with Static PNG Overlay
/// 
/// This example shows how to composite a static PNG image over
/// captured DeckLink video frames using the alpha channel.

use anyhow::Result;
use decklink_output::CompositorBuilder;
use decklink_input::CaptureSession;
use common_io::MemLoc;
use std::time::Instant;

fn main() -> Result<()> {
    println!("DeckLink Output - Static PNG Overlay Example");
    println!("============================================\n");

    // Initialize DeckLink capture
    println!("Initializing DeckLink capture...");
    let mut capture = CaptureSession::new(0)?;
    println!("✓ DeckLink capture ready\n");

    // Create compositor with static PNG
    println!("Loading foreground.png...");
    let mut compositor = CompositorBuilder::new(1920, 1080)
        .with_png("foreground.png")
        .build()?;
    println!("✓ Compositor ready\n");

    println!("Starting compositing loop (Press Ctrl+C to exit)...\n");

    let mut frame_count = 0;
    let start_time = Instant::now();
    let mut composite_times = Vec::new();

    loop {
        // Capture video frame from DeckLink
        let video_frame = match capture.next_frame() {
            Ok(frame) => frame,
            Err(e) => {
                eprintln!("Capture error: {}", e);
                continue;
            }
        };

        // Verify frame is on GPU
        match video_frame.data.loc {
            MemLoc::Gpu { device } => {
                // Good - zero-copy from DeckLink
            }
            MemLoc::Cpu => {
                eprintln!("Warning: Frame on CPU, GPUDirect not active");
            }
        }

        // Composite PNG overlay
        let comp_start = Instant::now();
        let composited = compositor.composite(&video_frame)?;
        let comp_time = comp_start.elapsed();
        composite_times.push(comp_time.as_micros() as f64 / 1000.0);

        frame_count += 1;

        // Print stats every second
        if frame_count % 60 == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let fps = frame_count as f64 / elapsed;
            let avg_comp_time = composite_times.iter().sum::<f64>() / composite_times.len() as f64;

            println!(
                "Frame {}: {:.1} fps | Composite: {:.2}ms avg | Output: {}x{} BGRA @ GPU",
                frame_count,
                fps,
                avg_comp_time,
                compositor.dimensions().0,
                compositor.dimensions().1,
            );

            composite_times.clear();
        }

        // Optional: Download output for preview/recording
        if frame_count == 1 {
            println!("\nSaving first frame as debug_output.png...");
            let bgra_data = compositor.download_output()?;
            save_bgra_as_png(&bgra_data, compositor.dimensions(), "debug_output.png")?;
            println!("✓ Saved debug_output.png\n");
        }

        // Here you would send `composited` to:
        // - Display (via OpenGL/Vulkan)
        // - DeckLink output card
        // - Video encoder
        // - Network stream
    }
}

fn save_bgra_as_png(
    bgra_data: &[u8],
    (width, height): (u32, u32),
    path: &str,
) -> Result<()> {
    use image::{ImageBuffer, RgbaImage};

    // Convert BGRA to RGBA for image crate
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
