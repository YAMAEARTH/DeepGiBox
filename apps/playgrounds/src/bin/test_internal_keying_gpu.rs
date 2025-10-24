use anyhow::{Context, Result};
use common_io::MemLoc;
use decklink_input::capture::CaptureSession;
use decklink_output::{BgraImage, OutputSession, OutputRequest};
use std::thread;
use std::time::Duration;

fn main() -> Result<()> {
    println!("🎬 Internal Keying Demo (GPU Only - Direct GPU→SDI)");
    println!("====================================================");
    println!("⚠️  Note: This demo requires GPU compositor (CUDA kernel)");
    println!("    Make sure CUDA kernel was built successfully.\n");
    
    // 1. Load PNG foreground image
    println!("📸 Loading PNG image...");
    let png_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "foreground.png".to_string());
    
    let png_image = BgraImage::load_from_file(&png_path)
        .map_err(|e| anyhow::anyhow!("Failed to load PNG: {}", e))?;
    
    println!("   ✓ Loaded: {}x{} pixels", png_image.width, png_image.height);
    
    // 2. Open DeckLink capture (background)
    println!("\n🎥 Opening DeckLink capture...");
    let mut capture = CaptureSession::open(0)
        .context("Failed to open DeckLink capture device 0")?;
    println!("   ✓ DeckLink capture opened on device 0");
    
    // 3. Wait for stable GPU frame dimensions
    println!("\n⏳ Waiting for stable GPU frame dimensions...");
    let (width, height) = loop {
        if let Some(frame) = capture.get_frame()? {
            let w = frame.meta.width;
            let h = frame.meta.height;
            
            // Check if frame is GPU and has stable HD resolution
            if let MemLoc::Gpu { device } = frame.data.loc {
                if w >= 1920 && h >= 1080 {
                    println!("   ✓ Got GPU frame: {}x{} on device {}", w, h, device);
                    break (w, h);
                } else {
                    println!("   ⏳ Frame {}x{} (waiting for HD resolution)...", w, h);
                }
            } else {
                println!("   ⚠️  Frame is on CPU, waiting for GPU frame...");
            }
        }
        thread::sleep(Duration::from_millis(50));
    };
    
    // 4. Create output session with GPU compositor
    println!("\n🔧 Setting up GPU output session with compositor...");
    let mut output = OutputSession::new(width, height, &png_image)
        .context("Failed to create output session")?;
    println!("   ✓ GPU Output session ready: {}x{}", width, height);
    println!("   ✓ CUDA Compositor initialized with PNG overlay");
    
    // 5. Initialize DeckLink output
    println!("\n📡 Initializing DeckLink output...");
    let mut decklink_out = decklink_output::from_path("configs/dev_1080p60_yuv422_fp16_trt.toml")
        .context("Failed to initialize DeckLink output from config")?;
    println!("   ✓ DeckLink output initialized");
    
    // 6. Process frames: Capture (GPU) → Composite (CUDA) → Output (GPU→SDI)
    println!("\n▶️  Processing frames (GPU→CUDA Composite→SDI) - Ctrl+C to stop");
    println!("   Pipeline: DeckLink Capture (GPU) → CUDA Composite → DeckLink Output (GPU)");
    println!("──────────────────────────────────────────────────────────");
    
    let mut frame_count = 0;
    let start_time = std::time::Instant::now();
    
    loop {
        // Get frame from DeckLink capture
        if let Some(frame) = capture.get_frame()? {
            // Verify frame is on GPU
            match frame.data.loc {
                MemLoc::Gpu { device } => {
                    // Composite PNG overlay on GPU using CUDA kernel
                    output.composite(
                        frame.data.ptr,
                        frame.data.stride,
                    ).context("Failed to composite frame on GPU")?;
                    
                    // Get composited BGRA output from GPU
                    let output_bgra_ptr = output.output_gpu_ptr();
                    let output_pitch = output.output_pitch();
                    
                    // Create frame packet for composited BGRA output
                    use common_io::{MemRef, PixelFormat, ColorSpace};
                    let composited_packet = common_io::RawFramePacket {
                        meta: common_io::FrameMeta {
                            source_id: frame.meta.source_id,
                            width,
                            height,
                            pixfmt: PixelFormat::BGRA8,
                            colorspace: ColorSpace::SRGB,
                            frame_idx: frame.meta.frame_idx,
                            pts_ns: frame.meta.pts_ns,
                            t_capture_ns: frame.meta.t_capture_ns,
                            stride_bytes: output_pitch as u32,
                        },
                        data: MemRef {
                            ptr: output_bgra_ptr as *mut u8,
                            len: output_pitch * height as usize,
                            stride: output_pitch,
                            loc: common_io::MemLoc::Gpu { device: 0 },
                        },
                    };
                    
                    // Submit composited BGRA frame to DeckLink output
                    let output_request = OutputRequest {
                        video: Some(&composited_packet),
                        overlay: None,
                    };
                    
                    decklink_out.submit(output_request)
                        .context("Failed to submit frame to DeckLink output")?;
                    
                    frame_count += 1;
                    
                    // Print stats every second
                    let elapsed = start_time.elapsed().as_secs_f64();
                    if frame_count % 60 == 0 {
                        let fps = frame_count as f64 / elapsed;
                        println!("Frame #{:6} | FPS: {:.2} | GPU device: {} | GPU→CUDA→SDI ✓", 
                            frame_count, fps, device);
                    }
                }
                MemLoc::Cpu => {
                    eprintln!("⚠️  Warning: Frame #{} is on CPU (expected GPU), skipping...", frame_count);
                    thread::sleep(Duration::from_millis(5));
                }
            }
        } else {
            // No frame available, wait a bit
            thread::sleep(Duration::from_millis(5));
        }
    }
}
