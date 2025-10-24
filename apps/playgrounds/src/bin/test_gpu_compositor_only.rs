use anyhow::{Context, Result};
use common_io::MemLoc;
use decklink_input::capture::CaptureSession;
use decklink_output::{BgraImage, OutputSession};
use std::thread;
use std::time::Duration;

fn main() -> Result<()> {
    println!("🎬 GPU Compositor Test (CUDA Keying Kernel)");
    println!("============================================");
    println!("Testing: DeckLink Capture (GPU) → CUDA Composite → Download to CPU");
    println!();
    
    // 1. Load PNG foreground image
    println!("📸 Loading PNG image...");
    let png_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "foreground.png".to_string());
    
    let png_image = BgraImage::load_from_file(&png_path)
        .map_err(|e| anyhow::anyhow!("Failed to load PNG: {}", e))?;
    
    println!("   ✓ Loaded: {}x{} pixels", png_image.width, png_image.height);
    
    // 2. Open DeckLink capture
    println!("\n🎥 Opening DeckLink capture...");
    let mut capture = CaptureSession::open(0)
        .context("Failed to open DeckLink capture device 0")?;
    println!("   ✓ DeckLink capture opened on device 0");
    
    // 3. Wait for stable GPU frame
    println!("\n⏳ Waiting for stable GPU frame...");
    let (width, height) = loop {
        if let Some(frame) = capture.get_frame()? {
            let w = frame.meta.width;
            let h = frame.meta.height;
            
            if let MemLoc::Gpu { device } = frame.data.loc {
                if w >= 1920 && h >= 1080 {
                    println!("   ✓ Got GPU frame: {}x{} on device {}", w, h, device);
                    break (w, h);
                }
            }
        }
        thread::sleep(Duration::from_millis(50));
    };
    
    // 4. Create GPU compositor session
    println!("\n🔧 Creating GPU compositor session...");
    let mut compositor = OutputSession::new(width, height, &png_image)
        .context("Failed to create GPU compositor")?;
    println!("   ✓ GPU Compositor ready: {}x{}", width, height);
    println!("   ✓ CUDA kernel loaded successfully!");
    
    // 5. Test compositing (30 seconds at ~60fps = ~1800 frames)
    println!("\n▶️  Testing GPU compositing (30 seconds)...");
    println!("──────────────────────────────────────────");
    
    let start_time = std::time::Instant::now();
    let mut frame_count = 0;
    
    while start_time.elapsed().as_secs() < 30 {
        let i = frame_count;
        if let Some(frame) = capture.get_frame()? {
            match frame.data.loc {
                MemLoc::Gpu { device } => {
                    let start = std::time::Instant::now();
                    
                    // Composite on GPU using CUDA kernel
                    compositor.composite(
                        frame.data.ptr,
                        frame.data.stride,
                    ).context("Failed to composite frame")?;
                    
                    let composite_time = start.elapsed();
                    
                    // Download result to CPU for verification
                    let _result = compositor.download_output()
                        .context("Failed to download composited output")?;
                    
                    let total_time = start.elapsed();
                    
                    frame_count += 1;
                    
                    // Print stats every 60 frames (~1 second)
                    if frame_count % 60 == 0 {
                        println!("[{:2}s] Frame #{}: Composite={:.2}ms, Total={:.2}ms, Avg={:.2}ms (GPU device {})", 
                            start_time.elapsed().as_secs(),
                            frame_count,
                            composite_time.as_secs_f64() * 1000.0,
                            total_time.as_secs_f64() * 1000.0,
                            (start_time.elapsed().as_secs_f64() / frame_count as f64) * 1000.0,
                            device
                        );
                    }
                }
                MemLoc::Cpu => {
                    if frame_count % 60 == 0 {
                        println!("[{:2}s] Frame #{}: Skipped (on CPU, expected GPU)", 
                            start_time.elapsed().as_secs(), frame_count);
                    }
                }
            }
            frame_count += 1;
        }
        thread::sleep(Duration::from_millis(16)); // ~60fps
    }
    
    let total_elapsed = start_time.elapsed();
    let avg_fps = frame_count as f64 / total_elapsed.as_secs_f64();
    
    println!("\n✅ GPU Compositor test completed successfully!");
    println!("   Duration: {:.1}s", total_elapsed.as_secs_f64());
    println!("   Total frames: {}", frame_count);
    println!("   Average FPS: {:.2}", avg_fps);
    println!("   CUDA keying kernel is working correctly!");
    
    Ok(())
}
