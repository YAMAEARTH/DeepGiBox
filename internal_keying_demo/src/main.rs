use anyhow::Result;
use decklink_input::{capture::CaptureSession, OutputDevice};
use decklink_output::{BgraImage, OutputSession};
use std::thread;
use std::time::Duration;

fn main() -> Result<()> {
    println!("🎬 Internal Keying Demo with SDI Output");
    println!("========================================");
    
    // 1. Load PNG foreground image
    println!("\n📸 Loading PNG image...");
    let png_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "foreground.png".to_string());
    
    let png_image = BgraImage::load_from_file(&png_path)
        .map_err(|e| anyhow::anyhow!("Failed to load PNG: {}", e))?;
    
    println!("   ✓ Loaded: {}x{} pixels", png_image.width, png_image.height);
    
    // 2. Open DeckLink capture (background)
    println!("\n🎥 Opening DeckLink capture...");
    let mut capture = CaptureSession::open(0)?;
    println!("   ✓ DeckLink capture opened on device 0");
    
    // Wait for stable frame dimensions (DeckLink may change resolution)
    println!("\n⏳ Waiting for stable frame dimensions...");
    let (width, height) = loop {
        if let Some(frame) = capture.get_frame()? {
            let w = frame.meta.width;
            let h = frame.meta.height;
            
            // Check if frame matches PNG size or is stable HD resolution
            if (w == png_image.width && h == png_image.height) || (w >= 1920 && h >= 1080) {
                println!("   ✓ Got frame: {}x{}", w, h);
                break (w, h);
            } else {
                println!("   ⏳ Frame {}x{} (waiting for {}x{})...", w, h, png_image.width, png_image.height);
            }
        }
        thread::sleep(Duration::from_millis(50));
    };
    
    // 3. Open DeckLink output (SDI)
    println!("\n📡 Opening DeckLink output...");
    let decklink_output = OutputDevice::open(0, width as i32, height as i32, 60.0)
        .map_err(|e| anyhow::anyhow!("Failed to open DeckLink output: {}", e))?;
    println!("   ✓ DeckLink output opened: {}x{}@{}fps", 
        decklink_output.width(), decklink_output.height(), decklink_output.fps());
    
    // 4. Create output session
    println!("\n🔧 Setting up output session...");
    let mut output = OutputSession::new(width, height, &png_image)?;
    println!("   ✓ Output session ready: {}x{}", width, height);
    
    // 5. Process frames and output to SDI (GPU→SDI direct!)
    println!("\n▶️  Processing frames → GPU→SDI Direct (Ctrl+C to stop)...");
    println!("   Mode: ALPHA COMPOSITE (Fast - using PNG alpha channel)");
    println!("──────────────────────────────────────────────────────────");
    
    let mut frame_count = 0;
    let start_time = std::time::Instant::now();
    
    loop {
        // Get frame from DeckLink capture
        if let Some(frame) = capture.get_frame()? {
            // Check if frame is on GPU
            match frame.data.loc {
                common_io::MemLoc::Gpu { device } => {
                    // Composite using PNG's alpha channel
                    output.composite(
                        frame.data.ptr,
                        frame.data.stride,
                    )?;
                    
                    // Send directly from GPU to SDI (zero-copy!)
                    decklink_output.send_frame_gpu(
                        output.output_gpu_ptr(),
                        output.output_pitch(),
                    ).map_err(|e| anyhow::anyhow!("Failed to send frame: {}", e))?;
                    
                    frame_count += 1;
                    
                    // Print stats every second
                    let elapsed = start_time.elapsed().as_secs_f64();
                    if frame_count % 60 == 0 {
                        let fps = frame_count as f64 / elapsed;
                        println!("Frame #{:6} | FPS: {:.2} | GPU→SDI direct (device: {})", 
                            frame_count, fps, device);
                    }
                }
                common_io::MemLoc::Cpu => {
                    eprintln!("⚠️  Warning: Frame is on CPU, not GPU (frame #{})", frame_count);
                }
            }
        }
        
        // Small delay to avoid busy loop
        thread::sleep(Duration::from_millis(1));
    }
}
