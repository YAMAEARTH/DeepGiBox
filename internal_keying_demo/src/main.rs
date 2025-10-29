use anyhow::Result;
use decklink_input::{capture::CaptureSession, OutputDevice, VideoFormat};
use decklink_output::{BgraImage, ChromaKey, OutputSession};
use std::thread;
use std::time::Duration;

/// Detect stable video format from capture
fn detect_stable_format(capture: &mut CaptureSession) -> Result<VideoFormat> {
    const STABLE_FRAMES_REQUIRED: usize = 5;
    const MAX_DETECTION_ATTEMPTS: usize = 150;
    const SKIP_INITIAL_FRAMES: usize = 10; // Skip first frames while DeckLink auto-detects
    
    let mut last_format: Option<VideoFormat> = None;
    let mut stable_count = 0;
    let mut attempts = 0;
    
    // Skip initial frames to allow DeckLink to detect proper resolution
    println!("   ⏳ Waiting for signal stabilization...");
    for _ in 0..SKIP_INITIAL_FRAMES {
        if let Some(_) = capture.get_frame()? {
            thread::sleep(Duration::from_millis(16));
        }
    }
    
    loop {
        attempts += 1;
        if attempts > MAX_DETECTION_ATTEMPTS {
            return Err(anyhow::anyhow!("Failed to detect stable video format after {} attempts", MAX_DETECTION_ATTEMPTS));
        }
        
        if let Some(frame) = capture.get_frame()? {
            let w = frame.meta.width;
            let h = frame.meta.height;
            
            // Skip invalid or SD dimensions (720x486 is initial/invalid state)
            if w == 0 || h == 0 || (w <= 720 && h <= 576) {
                thread::sleep(Duration::from_millis(16));
                continue;
            }
            
            // Estimate FPS based on resolution
            let fps = if h >= 2160 { 30.0 } else { 60.0 };
            
            let current_format = VideoFormat::detect(w, h, fps);
            
            match last_format {
                None => {
                    // First valid frame
                    println!("   📺 Detected: {}", current_format.name());
                    last_format = Some(current_format);
                    stable_count = 1;
                }
                Some(ref last) if last.width == current_format.width && last.height == current_format.height => {
                    // Same format, increment stability counter
                    stable_count += 1;
                    
                    if stable_count >= STABLE_FRAMES_REQUIRED {
                        return Ok(current_format);
                    }
                }
                Some(_) => {
                    // Format changed, reset
                    println!("   📺 Format changed: {}", current_format.name());
                    last_format = Some(current_format);
                    stable_count = 1;
                }
            }
        }
        
        thread::sleep(Duration::from_millis(16));
    }
}

fn main() -> Result<()> {
    println!("🎬 Internal Keying Demo with Auto-Format Detection");
    println!("===================================================");
    
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
    
    // 3. Detect stable video format
    println!("\n⏳ Detecting video format (waiting for stable signal)...");
    let video_format = detect_stable_format(&mut capture)?;
    println!("   ✓ Stable format detected: {}", video_format.name());
    
    // 4. Open DeckLink output (SDI) with auto-detected format
    println!("\n📡 Opening DeckLink output...");
    let decklink_output = OutputDevice::open_from_format(0, video_format)
        .map_err(|e| anyhow::anyhow!("Failed to open DeckLink output: {}", e))?;
    println!("   ✓ DeckLink output configured");
    
    // 5. Create output session
    println!("\n🔧 Setting up output session...");
    let mut output = OutputSession::new(video_format.width, video_format.height, &png_image)?;
    println!("   ✓ Output session ready: {}x{}", video_format.width, video_format.height);
    
    // 6. Configure chroma key
    let chroma_key = ChromaKey::green_screen();
    println!("\n🎨 Chroma Key: RGB({}, {}, {}) threshold={}",
        chroma_key.r, chroma_key.g, chroma_key.b, chroma_key.threshold);
    
    // 7. Process frames and output to SDI (GPU→SDI direct!)
    println!("\n▶️  Processing frames → GPU→SDI Direct (Ctrl+C to stop)...");
    println!("──────────────────────────────────────────────────────────");
    println!("   Format: {} | Resolution: {}x{} @ {:.0}fps",
        video_format.name(), video_format.width, video_format.height, video_format.fps);
    println!("──────────────────────────────────────────────────────────");
    
    let mut frame_count = 0;
    let start_time = std::time::Instant::now();
    
    loop {
        // Get frame from DeckLink capture
        if let Some(frame) = capture.get_frame()? {
            // Check if frame is on GPU
            match frame.data.loc {
                common_io::MemLoc::Gpu { device } => {
                    // Composite PNG over DeckLink on GPU
                    output.composite(
                        frame.data.ptr,
                        frame.data.stride,
                        chroma_key,
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
                        println!("Frame #{:6} | FPS: {:.2} | Format: {} | GPU device: {}", 
                            frame_count, fps, video_format.name(), device);
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
