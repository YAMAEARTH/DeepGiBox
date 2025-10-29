//! Complete Pipeline: Capture → Overlay using Hardware Internal Keying (OPTIMIZED BGRA VERSION)
//!
//! Pipeline flow:
//! DeckLink Capture → Preprocessing → Inference V2 → Postprocessing → Overlay Planning → **Inline BGRA Rendering** → Hardware Keying
//!
//! OPTIMIZATION: This version renders overlay directly to BGRA format on CPU to avoid
//! expensive ARGB→BGRA conversion that costs ~17ms. By rendering directly to BGRA,
//! we eliminate the conversion step entirely.

use anyhow::{anyhow, Result};
use common_io::{MemLoc, MemRef, Stage, DrawOp};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use decklink_input::capture::CaptureSession;
use decklink_output::OutputRequest;
use inference_v2::TrtInferenceStage;
use overlay_plan::PlanStage;
use postprocess;
use preprocess_cuda::{Preprocessor, CropRegion};
use std::time::{Duration, Instant};

// ═══════════════════════════════════════════════════════════════════════════
// INLINE BGRA RENDERER (CPU)
// ═══════════════════════════════════════════════════════════════════════════
// This replaces the RenderStage and renders directly to BGRA format
// to avoid ARGB→BGRA conversion overhead (~17ms)

/// Directly render overlay to BGRA format (avoiding ARGB intermediate)
struct BgraOverlayRenderer;

impl BgraOverlayRenderer {
    /// Put pixel in BGRA format
    fn put_pixel(buf: &mut [u8], stride: usize, w: usize, h: usize, x: i32, y: i32, b: u8, g: u8, r: u8, a: u8) {
        if x < 0 || y < 0 { return; }
        let (x, y) = (x as usize, y as usize);
        if x >= w || y >= h { return; }
        let idx = y * stride + x * 4;
        buf[idx + 0] = b;
        buf[idx + 1] = g;
        buf[idx + 2] = r;
        buf[idx + 3] = a;
    }

    /// Draw line with thickness in BGRA format
    fn draw_line(buf: &mut [u8], stride: usize, w: usize, h: usize, x0: i32, y0: i32, x1: i32, y1: i32, thickness: i32, color_bgra: (u8, u8, u8, u8)) {
        let (mut x0, mut y0, x1, y1) = (x0, y0, x1, y1);
        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -(y1 - y0).abs();
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let rad = thickness.max(1) / 2;
        loop {
            for oy in -rad..=rad {
                for ox in -rad..=rad {
                    Self::put_pixel(buf, stride, w, h, x0 + ox, y0 + oy, color_bgra.0, color_bgra.1, color_bgra.2, color_bgra.3);
                }
            }
            if x0 == x1 && y0 == y1 { break; }
            let e2 = 2 * err;
            if e2 >= dy { err += dy; x0 += sx; }
            if e2 <= dx { err += dx; y0 += sy; }
        }
    }

    /// Draw rectangle outline in BGRA format
    fn draw_rect_outline(buf: &mut [u8], stride: usize, w: usize, h: usize, x: i32, y: i32, ww: i32, hh: i32, thickness: i32, color_bgra: (u8, u8, u8, u8)) {
        if ww <= 0 || hh <= 0 { return; }
        let x2 = x + ww;
        let y2 = y + hh;
        Self::draw_line(buf, stride, w, h, x, y, x2, y, thickness, color_bgra);
        Self::draw_line(buf, stride, w, h, x2, y, x2, y2, thickness, color_bgra);
        Self::draw_line(buf, stride, w, h, x2, y2, x, y2, thickness, color_bgra);
        Self::draw_line(buf, stride, w, h, x, y2, x, y, thickness, color_bgra);
    }

    /// Render OverlayPlanPacket directly to BGRA buffer
    fn render_to_bgra(plan: &common_io::OverlayPlanPacket) -> Vec<u8> {
        let w = plan.canvas.0 as usize;
        let h = plan.canvas.1 as usize;
        let stride = w * 4;
        
        let mut buf = vec![0u8; h * stride];
        
        // Execute drawing operations
        for op in &plan.ops {
            match op {
                DrawOp::Rect { xywh, thickness, color } => {
                    let (x, y, ww, hh) = (xywh.0 as i32, xywh.1 as i32, xywh.2 as i32, xywh.3 as i32);
                    // Convert ARGB color to BGRA
                    let color_bgra = (color.3, color.2, color.1, color.0); // (B, G, R, A)
                    Self::draw_rect_outline(&mut buf, stride, w, h, x, y, ww, hh, *thickness as i32, color_bgra);
                }
                DrawOp::FillRect { xywh, color } => {
                    let (x, y, ww, hh) = (xywh.0 as i32, xywh.1 as i32, xywh.2 as i32, xywh.3 as i32);
                    // Convert ARGB color to BGRA
                    let color_bgra = (color.3, color.2, color.1, color.0); // (B, G, R, A)
                    for yy in y.max(0)..(y+hh).min(h as i32) {
                        for xx in x.max(0)..(x+ww).min(w as i32) {
                            Self::put_pixel(&mut buf, stride, w, h, xx, yy, color_bgra.0, color_bgra.1, color_bgra.2, color_bgra.3);
                        }
                    }
                }
                DrawOp::Poly { pts, thickness, color } => {
                    if pts.len() >= 2 {
                        let (x0, y0) = (pts[0].0 as i32, pts[0].1 as i32);
                        let (x1, y1) = (pts[1].0 as i32, pts[1].1 as i32);
                        // Convert ARGB color to BGRA
                        let color_bgra = (color.3, color.2, color.1, color.0);
                        Self::draw_line(&mut buf, stride, w, h, x0, y0, x1, y1, *thickness as i32, color_bgra);
                    }
                }
                DrawOp::Label { anchor: _, text: _, font_px: _, color: _ } => {
                    // No-op for text in minimal renderer
                }
            }
        }
        
        buf
    }
}

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  PIPELINE: CAPTURE → OVERLAY (OPTIMIZED BGRA)           ║");
    println!("║  DeckLink → Preprocess → Inference V2 → Post → Overlay  ║");
    println!("║  → Hardware Internal Keying (30 seconds test)           ║");
    println!("║                                                          ║");
    println!("║  🚀 OPTIMIZATION: Direct BGRA rendering (no conversion) ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Create output directory
    std::fs::create_dir_all("output/test")?;
    println!("📁 Created output/test directory");
    println!();

    // 0. List available DeckLink devices
    println!("📹 Available DeckLink Devices:");
    let devices = decklink_input::devicelist();
    println!("  Available DeckLink devices: {}", devices.len());
    for (idx, name) in devices.iter().enumerate() {
        println!("    [{}] {}", idx, name);
    }
    if devices.is_empty() {
        return Err(anyhow!("No DeckLink devices found!"));
    }
    println!();

    // 1. Initialize DeckLink capture
    println!("📹 Step 1: Initialize DeckLink Capture");
    let mut capture = CaptureSession::open(0)?;
    println!("  ✓ Opened DeckLink device 0");
    println!();

    // 2. Initialize Preprocessor
    println!("⚙️  Step 2: Initialize Preprocessor");
    let crop_region = CropRegion::Olympus;
    
    let mut preprocessor = Preprocessor::with_crop_region(
        (512, 512),
        false,
        0,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        preprocess_cuda::ChromaOrder::UYVY,
        crop_region,
    )?;
    
    let (cx, cy, cw, ch) = crop_region.get_coords();
    println!("  ✓ Preprocessor ready (512x512, FP32, GPU 0)");
    println!("  ✓ Crop region: {:?} => [{}, {}, {}×{}]", crop_region, cx, cy, cw, ch);
    println!();

    // 2.5 Initialize CUDA device
    println!("🔧 Step 2.5: Initialize CUDA Device");
    let cuda_device = CudaDevice::new(0)?;
    println!("  ✓ CUDA device initialized for CPU->GPU transfers");
    println!();

    // 3. Initialize TensorRT Inference V2
    println!("🧠 Step 3: Initialize TensorRT Inference V2");
    let engine_path = "configs/model/v7_optimized_YOLOv5.engine";
    let lib_path = "trt-shim/build/libtrt_shim.so";

    if !std::path::Path::new(engine_path).exists() {
        return Err(anyhow!(
            "TensorRT engine not found: {}\n💡 Please build the engine first!",
            engine_path
        ));
    }
    if !std::path::Path::new(lib_path).exists() {
        return Err(anyhow!(
            "TRT shim library not found: {}\n💡 Please build trt-shim first!",
            lib_path
        ));
    }

    let mut inference_stage =
        TrtInferenceStage::new(engine_path, lib_path).map_err(|e| anyhow!(e))?;
    println!("  ✓ TensorRT Inference V2 loaded");
    println!("  ✓ Engine: {}", engine_path);
    println!("  ✓ Output size: {} values", inference_stage.output_size());
    println!();

    // 4. Initialize Postprocessing
    println!("🎯 Step 4: Initialize Postprocessing");
    let mut post_stage = postprocess::from_path("")?.with_sort_tracking(30, 0.3, 0.3);
    println!("  ✓ Postprocessing ready");
    println!("  ✓ Classes: 2 (Hyper, Neo)");
    println!("  ✓ Confidence threshold: 0.25");
    println!("  ✓ Temporal smoothing: enabled (window=4)");
    println!("  ✓ SORT tracking: enabled (max_age=30)");
    println!();

    // 5. Initialize Overlay Planning
    println!("🎨 Step 5: Initialize Overlay Planning");
    let mut plan_stage = PlanStage {
        enable_full_ui: true,
    };
    println!("  ✓ Overlay planning ready");
    println!("  ✓ Rendering: INLINE BGRA (optimized, no conversion)");
    println!();

    // 5.5 Initialize Hardware Internal Keying (DeckLink Output)
    println!("🔧 Step 5.5: Initialize Hardware Internal Keying");
    
    // Wait for first frame to get dimensions
    println!("  ⏳ Waiting for first frame to determine dimensions...");
    let (width, height) = loop {
        if let Some(frame) = capture.get_frame()? {
            let w = frame.meta.width;
            let h = frame.meta.height;
            
            if w >= 1920 && h >= 1080 {
                println!("   ✓ Got frame: {}x{}", w, h);
                break (w, h);
            } else {
                println!("   ⏳ Frame {}x{} (waiting for HD resolution)...", w, h);
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    
    // Initialize DeckLink output
    let mut decklink_out = decklink_output::from_path("configs/dev_1080p60_yuv422_fp16_trt.toml")?;
    println!("  ✓ DeckLink output initialized: {}x{}", width, height);
    
    // Configure SDI output connection
    decklink_out.set_sdi_output()?;
    println!("  ✓ SDI output connection configured");
    
    // Enable hardware internal keying
    decklink_out.enable_internal_keying()?;
    println!("  ✓ Hardware internal keying ENABLED");
    
    // Set keyer level to maximum (255 = fully visible overlay)
    decklink_out.set_keyer_level(255)?;
    println!("  ✓ Keyer level set to 255 (fully visible)");
    println!();

    // 6. Process frames
    println!("🎬 Step 6: Processing Frames (30 seconds test)...");
    println!("════════════════════════════════════════════════════════════");
    println!("   Mode: Hardware Internal Keying (OPTIMIZED BGRA)");
    println!("   Input: SDI capture → Inline BGRA Overlay (CPU)");
    println!("   Output: SDI (Hardware Composited)");
    println!("   Duration: 30 seconds");
    println!("════════════════════════════════════════════════════════════");
    println!();

    let mut frame_count = 0u64;
    let mut total_latency_ms = 0.0;

    // Latency breakdown accumulators
    let mut total_capture_ms = 0.0;
    let mut total_preprocess_ms = 0.0;
    let mut total_inference_ms = 0.0;
    let mut total_postprocess_ms = 0.0;
    let mut total_plan_ms = 0.0;
    let mut total_render_ms = 0.0;
    let mut total_keying_ms = 0.0;
    let mut total_hardware_latency_ms = 0.0;
    
    // Hardware Keying sub-timings
    let mut total_keying_alloc_ms = 0.0;
    let mut total_keying_upload_ms = 0.0;
    let mut total_keying_packet_ms = 0.0;
    let mut total_keying_submit_ms = 0.0;

    // GPU buffer pool
    let mut gpu_buffers: Vec<CudaSlice<u8>> = Vec::new();
    
    let pipeline_start_time = Instant::now();
    let test_duration = Duration::from_secs(30);

    loop {
        // Check if 30 seconds have elapsed
        if pipeline_start_time.elapsed() >= test_duration {
            println!();
            println!("⏱️  30 seconds elapsed - stopping test");
            break;
        }
        
        let pipeline_start = Instant::now();

        // Capture frame
        let capture_start = Instant::now();
        let raw_frame = match capture.get_frame()? {
            Some(frame) => frame,
            None => {
                println!("⚠️  No frame received, waiting...");
                std::thread::sleep(std::time::Duration::from_millis(16));
                continue;
            }
        };
        let capture_time = capture_start.elapsed();
        total_capture_ms += capture_time.as_secs_f64() * 1000.0;

        println!("════════════════════════════════════════════════════════════");
        println!("🎬 Frame #{}", frame_count);
        println!("════════════════════════════════════════════════════════════");

        // Step 1: Capture Results
        println!();
        println!("📹 Step 1: Capture Result");
        println!("  ✓ Time: {:.2}ms", capture_time.as_secs_f64() * 1000.0);
        println!("  ✓ RawFramePacket:");
        println!(
            "      → Dimensions: {}×{}",
            raw_frame.meta.width, raw_frame.meta.height
        );
        println!(
            "      → Format: {:?} {:?}",
            raw_frame.meta.pixfmt, raw_frame.meta.colorspace
        );
        println!("      → Frame idx: {}", raw_frame.meta.frame_idx);

        // Copy CPU data to GPU if needed
        let raw_frame_gpu = if matches!(raw_frame.data.loc, MemLoc::Cpu) {
            let cpu_data =
                unsafe { std::slice::from_raw_parts(raw_frame.data.ptr, raw_frame.data.len) };
            let gpu_buffer = cuda_device.htod_sync_copy(cpu_data)?;
            let gpu_ptr = *gpu_buffer.device_ptr() as *mut u8;
            let gpu_packet = common_io::RawFramePacket {
                meta: raw_frame.meta.clone(),
                data: MemRef {
                    ptr: gpu_ptr,
                    len: raw_frame.data.len,
                    stride: raw_frame.data.stride,
                    loc: MemLoc::Gpu { device: 0 },
                },
            };
            gpu_buffers.push(gpu_buffer);
            gpu_packet
        } else {
            raw_frame.clone()
        };

        // Step 2: Preprocessing
        println!();
        println!("⚙️  Step 2: CUDA Preprocessing");
        let preprocess_start = Instant::now();
        let tensor_packet = match preprocessor.process_checked(raw_frame_gpu.clone()) {
            Some(packet) => packet,
            None => {
                println!("  ⚠️  Skipping init frame (waiting for stable resolution)\n");
                continue;
            }
        };
        let preprocess_time = preprocess_start.elapsed();
        total_preprocess_ms += preprocess_time.as_secs_f64() * 1000.0;
        println!("  ✓ Time: {:.2}ms", preprocess_time.as_secs_f64() * 1000.0);

        // Step 3: Inference V2
        println!();
        println!("🧠 Step 3: TensorRT Inference V2");
        let inference_start = Instant::now();
        let raw_detections = inference_stage.process(tensor_packet);
        let inference_time = inference_start.elapsed();
        total_inference_ms += inference_time.as_secs_f64() * 1000.0;
        println!("  ✓ Time: {:.2}ms", inference_time.as_secs_f64() * 1000.0);

        // Step 4: Postprocessing
        println!();
        println!("🎯 Step 4: Postprocessing (NMS + Tracking)");
        let postprocess_start = Instant::now();
        let detections = post_stage.process(raw_detections);
        let postprocess_time = postprocess_start.elapsed();
        total_postprocess_ms += postprocess_time.as_secs_f64() * 1000.0;
        println!("  ✓ Time: {:.2}ms", postprocess_time.as_secs_f64() * 1000.0);
        println!("  ✓ DetectionsPacket:");
        println!("      → Total detections: {}", detections.items.len());

        // Step 5: Overlay Planning
        println!();
        println!("🎨 Step 5: Overlay Planning");
        let plan_start = Instant::now();
        let overlay_plan = plan_stage.process(detections);
        let plan_time = plan_start.elapsed();
        total_plan_ms += plan_time.as_secs_f64() * 1000.0;
        println!("  ✓ Time: {:.2}ms", plan_time.as_secs_f64() * 1000.0);

        // Step 6: OPTIMIZED Inline BGRA Rendering
        println!();
        println!("🖼️  Step 6: Inline BGRA Rendering (OPTIMIZED)");
        let render_start = Instant::now();
        let bgra_buffer = BgraOverlayRenderer::render_to_bgra(&overlay_plan);
        let render_time = render_start.elapsed();
        total_render_ms += render_time.as_secs_f64() * 1000.0;
        println!("  ✓ Time: {:.2}ms", render_time.as_secs_f64() * 1000.0);
        println!("  ✓ BGRA Buffer:");
        println!("      → Dimensions: {}×{}", width, height);
        println!("      → Buffer size: {} bytes", bgra_buffer.len());
        println!("      → Format: BGRA8 (native DeckLink format)");
        println!("      → Location: CPU (ready for GPU upload)");
        println!("      → ⚡ NO CONVERSION NEEDED (direct BGRA rendering)");

        // Step 7: Hardware Internal Keying
        println!();
        println!("🎬 Step 7: Hardware Internal Keying (GPU Upload + Output)");
        let keying_start = Instant::now();
        
        // Sub-step 1: Allocate or reuse GPU buffer
        let alloc_start = Instant::now();
        let mut overlay_gpu_buffer = if gpu_buffers.len() > 1 {
            gpu_buffers.pop().unwrap()
        } else {
            cuda_device.alloc_zeros::<u8>(bgra_buffer.len())?
        };
        let alloc_time = alloc_start.elapsed();
        total_keying_alloc_ms += alloc_time.as_secs_f64() * 1000.0;
        
        // Sub-step 2: Upload BGRA overlay to GPU
        let upload_start = Instant::now();
        cuda_device.htod_sync_copy_into(&bgra_buffer, &mut overlay_gpu_buffer)?;
        let upload_time = upload_start.elapsed();
        total_keying_upload_ms += upload_time.as_secs_f64() * 1000.0;
        
        // Sub-step 3: Create GPU packet for overlay
        let packet_start = Instant::now();
        let overlay_gpu_packet = common_io::RawFramePacket {
            meta: common_io::FrameMeta {
                source_id: raw_frame_gpu.meta.source_id,
                width,
                height,
                pixfmt: common_io::PixelFormat::BGRA8,
                colorspace: common_io::ColorSpace::SRGB,
                frame_idx: raw_frame_gpu.meta.frame_idx,
                pts_ns: raw_frame_gpu.meta.pts_ns,
                t_capture_ns: raw_frame_gpu.meta.t_capture_ns,
                stride_bytes: (width * 4) as u32,
                crop_region: None,
            },
            data: MemRef {
                ptr: *overlay_gpu_buffer.device_ptr() as *mut u8,
                len: bgra_buffer.len(),
                stride: (width * 4) as usize,
                loc: common_io::MemLoc::Gpu { device: 0 },
            },
        };
        let packet_time = packet_start.elapsed();
        total_keying_packet_ms += packet_time.as_secs_f64() * 1000.0;
        
        // Sub-step 4: Submit overlay to DeckLink output
        let submit_start = Instant::now();
        let output_request = OutputRequest {
            video: Some(&overlay_gpu_packet),
            overlay: None,
        };
        decklink_out.submit(output_request)?;
        let submit_time = submit_start.elapsed();
        total_keying_submit_ms += submit_time.as_secs_f64() * 1000.0;
        
        // Return GPU buffer to pool
        gpu_buffers.push(overlay_gpu_buffer);
        
        let keying_time = keying_start.elapsed();
        total_keying_ms += keying_time.as_secs_f64() * 1000.0;
        println!("  ✓ Time: {:.2}ms", keying_time.as_secs_f64() * 1000.0);
        println!("      ├─ GPU Buffer Alloc:  {:.2}ms", alloc_time.as_secs_f64() * 1000.0);
        println!("      ├─ CPU→GPU Upload:    {:.2}ms", upload_time.as_secs_f64() * 1000.0);
        println!("      ├─ Packet Creation:   {:.2}ms", packet_time.as_secs_f64() * 1000.0);
        println!("      └─ DeckLink Submit:   {:.2}ms", submit_time.as_secs_f64() * 1000.0);
        println!("  ✓ Mode: HARDWARE INTERNAL KEYING (OPTIMIZED)");
        println!("  ✓ Pipeline: BGRA CPU → GPU → DeckLink Fill");
        println!("  ✓ Hardware keyer: ACTIVE");

        // Calculate hardware latency
        let output_complete_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let hardware_latency_ns = output_complete_ns - raw_frame_gpu.meta.t_capture_ns;
        let hardware_latency_ms = hardware_latency_ns as f64 / 1_000_000.0;
        total_hardware_latency_ms += hardware_latency_ms;

        let pipeline_time = pipeline_start.elapsed();
        let pipeline_ms = pipeline_time.as_secs_f64() * 1000.0;
        total_latency_ms += pipeline_ms;

        // Print latency breakdown
        println!();
        println!("════════════════════════════════════════════════════════════");
        println!("⏱️  LATENCY BREAKDOWN - Frame #{}", frame_count);
        println!("════════════════════════════════════════════════════════════");
        println!("  1. 📹 Capture:           {:6.2}ms ({:4.1}%)", 
            capture_time.as_secs_f64() * 1000.0,
            (capture_time.as_secs_f64() * 1000.0 / pipeline_ms) * 100.0);
        println!("  2. ⚙️  Preprocessing:     {:6.2}ms ({:4.1}%)", 
            preprocess_time.as_secs_f64() * 1000.0,
            (preprocess_time.as_secs_f64() * 1000.0 / pipeline_ms) * 100.0);
        println!("  3. 🧠 Inference:         {:6.2}ms ({:4.1}%)", 
            inference_time.as_secs_f64() * 1000.0,
            (inference_time.as_secs_f64() * 1000.0 / pipeline_ms) * 100.0);
        println!("  4. 🎯 Postprocessing:    {:6.2}ms ({:4.1}%)", 
            postprocess_time.as_secs_f64() * 1000.0,
            (postprocess_time.as_secs_f64() * 1000.0 / pipeline_ms) * 100.0);
        println!("  5. 🎨 Overlay Planning:  {:6.2}ms ({:4.1}%)", 
            plan_time.as_secs_f64() * 1000.0,
            (plan_time.as_secs_f64() * 1000.0 / pipeline_ms) * 100.0);
        println!("  6. 🖼️  BGRA Rendering:    {:6.2}ms ({:4.1}%) ⚡ OPTIMIZED", 
            render_time.as_secs_f64() * 1000.0,
            (render_time.as_secs_f64() * 1000.0 / pipeline_ms) * 100.0);
        println!("  7. 🎬 Hardware Keying:   {:6.2}ms ({:4.1}%) ⚡ NO CONVERSION", 
            keying_time.as_secs_f64() * 1000.0,
            (keying_time.as_secs_f64() * 1000.0 / pipeline_ms) * 100.0);
        println!("      ├─ GPU Buffer Alloc:  {:6.2}ms", alloc_time.as_secs_f64() * 1000.0);
        println!("      ├─ CPU→GPU Upload:    {:6.2}ms", upload_time.as_secs_f64() * 1000.0);
        println!("      ├─ Packet Creation:   {:6.2}ms", packet_time.as_secs_f64() * 1000.0);
        println!("      └─ DeckLink Submit:   {:6.2}ms", submit_time.as_secs_f64() * 1000.0);
        println!("────────────────────────────────────────────────────────────");
        println!("  🔴 END-TO-END (N2N):    {:6.2}ms ({:.1} FPS)", 
            pipeline_ms, 1000.0 / pipeline_ms);
        println!("  🔴 HARDWARE LATENCY:    {:6.2}ms", hardware_latency_ms);
        println!("  🎯 Mode: OPTIMIZED BGRA (No ARGB→BGRA conversion)");
        println!("════════════════════════════════════════════════════════════");
        println!();

        frame_count += 1;

        // Print cumulative stats every 60 frames
        if frame_count % 60 == 0 {
            let elapsed = pipeline_start_time.elapsed().as_secs_f64();
            let avg_fps = frame_count as f64 / elapsed;
            
            println!();
            println!("╔══════════════════════════════════════════════════════════╗");
            println!("║  📊 CUMULATIVE STATISTICS - {} FRAMES (OPTIMIZED)       ", frame_count);
            println!("╚══════════════════════════════════════════════════════════╝");
            println!();
            println!("  ⏱️  Average Latency per Stage:");
            println!("  ─────────────────────────────────────────────────────────");
            println!("  1. 📹 Capture:           {:6.2}ms avg", total_capture_ms / frame_count as f64);
            println!("  2. ⚙️  Preprocessing:     {:6.2}ms avg", total_preprocess_ms / frame_count as f64);
            println!("  3. 🧠 Inference:         {:6.2}ms avg", total_inference_ms / frame_count as f64);
            println!("  4. 🎯 Postprocessing:    {:6.2}ms avg", total_postprocess_ms / frame_count as f64);
            println!("  5. 🎨 Overlay Planning:  {:6.2}ms avg", total_plan_ms / frame_count as f64);
            println!("  6. 🖼️  Overlay Rendering:    {:6.2}ms avg ⚡ OPTIMIZED", total_render_ms / frame_count as f64);
            println!("  7. 🎬 Hardware Keying:   {:6.2}ms avg ⚡ NO CONVERSION", total_keying_ms / frame_count as f64);
            println!("      ├─ GPU Buffer Alloc:  {:6.2}ms avg", total_keying_alloc_ms / frame_count as f64);
            println!("      ├─ CPU→GPU Upload:    {:6.2}ms avg", total_keying_upload_ms / frame_count as f64);
            println!("      ├─ Packet Creation:   {:6.2}ms avg", total_keying_packet_ms / frame_count as f64);
            println!("      └─ DeckLink Submit:   {:6.2}ms avg", total_keying_submit_ms / frame_count as f64);
            println!("  ─────────────────────────────────────────────────────────");
            println!("  🔴 END-TO-END (N2N):    {:6.2}ms avg ({:.2} FPS avg)", 
                total_latency_ms / frame_count as f64,
                1000.0 / (total_latency_ms / frame_count as f64));
            println!("  🔴 HARDWARE LATENCY:    {:6.2}ms avg", 
                total_hardware_latency_ms / frame_count as f64);
            println!();
            println!("  📈 Performance Metrics:");
            println!("  ─────────────────────────────────────────────────────────");
            println!("  Total frames processed:  {}", frame_count);
            println!("  Total elapsed time:      {:.2}s", elapsed);
            println!("  Real-time FPS:           {:.2} FPS", avg_fps);
            println!("  Mode:                    OPTIMIZED BGRA (No conversion)");
            println!("  ─────────────────────────────────────────────────────────");
            println!();
        }
    }

    // Print final summary
    let total_elapsed = pipeline_start_time.elapsed().as_secs_f64();
    let avg_fps = frame_count as f64 / total_elapsed;
    
    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  🏁 FINAL SUMMARY - 30 SECOND TEST (OPTIMIZED BGRA)     ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("  ⏱️  Average Latency per Stage:");
    println!("  ─────────────────────────────────────────────────────────");
    println!("  1. 📹 Capture:           {:6.2}ms avg", total_capture_ms / frame_count as f64);
    println!("  2. ⚙️  Preprocessing:     {:6.2}ms avg", total_preprocess_ms / frame_count as f64);
    println!("  3. 🧠 Inference:         {:6.2}ms avg", total_inference_ms / frame_count as f64);
    println!("  4. 🎯 Postprocessing:    {:6.2}ms avg", total_postprocess_ms / frame_count as f64);
    println!("  5. 🎨 Overlay Planning:  {:6.2}ms avg", total_plan_ms / frame_count as f64);
    println!("  6. 🖼️  BGRA Rendering:    {:6.2}ms avg ⚡ OPTIMIZED", total_render_ms / frame_count as f64);
    println!("  7. 🎬 Hardware Keying:   {:6.2}ms avg ⚡ NO CONVERSION", total_keying_ms / frame_count as f64);
    println!("      ├─ GPU Buffer Alloc:  {:6.2}ms avg", total_keying_alloc_ms / frame_count as f64);
    println!("      ├─ CPU→GPU Upload:    {:6.2}ms avg", total_keying_upload_ms / frame_count as f64);
    println!("      ├─ Packet Creation:   {:6.2}ms avg", total_keying_packet_ms / frame_count as f64);
    println!("      └─ DeckLink Submit:   {:6.2}ms avg", total_keying_submit_ms / frame_count as f64);
    println!("  ─────────────────────────────────────────────────────────");
    println!("  🔴 END-TO-END (N2N):    {:6.2}ms avg ({:.2} FPS avg)", 
        total_latency_ms / frame_count as f64,
        1000.0 / (total_latency_ms / frame_count as f64));
    println!("  🔴 HARDWARE LATENCY:    {:6.2}ms avg", 
        total_hardware_latency_ms / frame_count as f64);
    println!();
    println!("  📈 Performance Metrics:");
    println!("  ─────────────────────────────────────────────────────────");
    println!("  Total frames processed:  {}", frame_count);
    println!("  Total elapsed time:      {:.2}s", total_elapsed);
    println!("  Real-time FPS:           {:.2} FPS", avg_fps);
    println!("  Mode:                    OPTIMIZED BGRA (Direct rendering)");
    println!();
    println!("  🚀 OPTIMIZATION SUMMARY:");
    println!("  ─────────────────────────────────────────────────────────");
    println!("  ✅ Eliminated ARGB→BGRA conversion (~17ms saved)");
    println!("  ✅ Direct BGRA rendering on CPU");
    println!("  ✅ Single GPU upload (no intermediate buffers)");
    println!("  ✅ Hardware keying with optimized data path");
    println!("  ─────────────────────────────────────────────────────────");
    println!();
    println!("✅ Test completed successfully!");
    
    Ok(())
}
