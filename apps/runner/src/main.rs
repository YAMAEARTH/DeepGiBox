//! DeepGiBox Production Runner
//!
//! Main application runner supporting multiple pipeline modes:
//! - Hardware Internal Keying (full pipeline with DeckLink output)
//! - Inference Only (capture → inference → telemetry)
//! - Visualization Mode (save detection frames to disk)
//!
//! Configuration is loaded from TOML files in configs/ directory

use anyhow::{anyhow, Result};
use common_io::{DetectionsPacket, MemLoc, MemRef, OverlayPlanPacket, RawDetectionsPacket, Stage, TensorInputPacket, DrawOp};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use decklink_input::capture::CaptureSession;
use decklink_output::OutputRequest;
use inference_v2::TrtInferenceStage;
use overlay_plan::PlanStage;
use overlay_render;
use postprocess;
use preprocess_cuda::{ChromaOrder, CropRegion, Preprocessor};
use std::fs;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use device_query::{DeviceQuery, DeviceState, Keycode};
use std::sync::atomic::AtomicU32;

mod config_loader;
use config_loader::{PipelineConfig, PipelineMode, EndoscopeMode};

// ═══════════════════════════════════════════════════════════════════════════
// ENDOSCOPE MODE MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════

/// Get EndoscopeMode from atomic u8 value
fn mode_from_u8(value: u8) -> EndoscopeMode {
    match value {
        0 => EndoscopeMode::Fuji,
        1 => EndoscopeMode::Olympus,
        _ => EndoscopeMode::Olympus, // Default fallback
    }
}

/// Convert EndoscopeMode to atomic u8 value
fn mode_to_u8(mode: &EndoscopeMode) -> u8 {
    match mode {
        EndoscopeMode::Fuji => 0,
        EndoscopeMode::Olympus => 1,
    }
}

/// Spawn keyboard listener thread to handle endoscope mode switching
fn spawn_keyboard_listener(
    current_mode: Arc<AtomicU8>,
    running: Arc<AtomicBool>,
    confidence_threshold: Arc<AtomicU32>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let device_state = DeviceState::new();
        let mut last_keys: Vec<Keycode> = Vec::new();

        while running.load(Ordering::SeqCst) {
            let keys = device_state.get_keys();

            // Detect key press (key present now but not in last frame)
            for key in &keys {
                if !last_keys.contains(key) {
                    // New key press detected
                    match key {
                        Keycode::Key1 => {
                            current_mode.store(mode_to_u8(&EndoscopeMode::Fuji), Ordering::SeqCst);
                            println!("\n🔵 Switched to FUJI mode");
                        }
                        Keycode::Key2 => {
                            current_mode.store(mode_to_u8(&EndoscopeMode::Olympus), Ordering::SeqCst);
                            println!("\n🟢 Switched to OLYMPUS mode");
                        }
                        Keycode::Equal => {
                            // '+' key (usually Shift+= on US keyboard, but Equal works for both)
                            let current_bits = confidence_threshold.load(Ordering::SeqCst);
                            let current = f32::from_bits(current_bits);
                            let new_value = (current + 0.05).min(1.0); // Cap at 1.0
                            confidence_threshold.store(new_value.to_bits(), Ordering::SeqCst);
                            println!("\n📈 Confidence threshold: {:.2} → {:.2}", current, new_value);
                        }
                        Keycode::Minus => {
                            // '-' key
                            let current_bits = confidence_threshold.load(Ordering::SeqCst);
                            let current = f32::from_bits(current_bits);
                            let new_value = (current - 0.05).max(0.0); // Floor at 0.0
                            confidence_threshold.store(new_value.to_bits(), Ordering::SeqCst);
                            println!("\n📉 Confidence threshold: {:.2} → {:.2}", current, new_value);
                        }
                        _ => {}
                    }
                }
            }

            last_keys = keys;
            std::thread::sleep(Duration::from_millis(50)); // Poll at 20Hz
        }
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// DEBUG HELPERS
// ═══════════════════════════════════════════════════════════════════════════

fn dump_frame_detections(
    detections: &DetectionsPacket,
    frame_num: usize,
    enable: bool,
) -> Result<()> {
    if !enable {
        return Ok(());
    }

    let filename = format!("output/runner/frame_{:06}_detections.txt", frame_num);
    fs::create_dir_all("output/runner")?;
    let mut file = fs::File::create(&filename)?;

    writeln!(file, "Frame: {}", frame_num)?;
    writeln!(file, "Total detections: {}", detections.items.len())?;
    writeln!(file, "")?;

    for (i, det) in detections.items.iter().enumerate() {
        writeln!(file, "Detection #{}:", i)?;
        writeln!(
            file,
            "  BBox: ({:.1}, {:.1}, {:.1}, {:.1})",
            det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h
        )?;
        writeln!(file, "  Class ID: {}", det.class_id)?;
        writeln!(file, "  Score: {:.4}", det.score)?;
        writeln!(file, "  Track ID: {:?}", det.track_id)?;
        writeln!(file, "")?;
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// PIPELINE IMPLEMENTATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Hardware Internal Keying Pipeline
/// Full pipeline: Capture → Preprocess → Inference → Postprocess → Overlay → Hardware Keying
fn run_keying_pipeline(config: &PipelineConfig) -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  DEEPGIBOX - HARDWARE INTERNAL KEYING PIPELINE          ║");
    println!("║  Production Mode: Real-time Overlay with Hardware Key   ║");
    println!("║  Press 1=Fuji, 2=Olympus, 3=Pentax to switch modes     ║");
    println!("║  Press +/- to adjust confidence threshold (±0.05)       ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Initialize endoscope mode state
    let current_mode = Arc::new(AtomicU8::new(mode_to_u8(&config.preprocessing.initial_endoscope_mode)));
    println!("🔧 Initial endoscope mode: {}", config.preprocessing.initial_endoscope_mode.name());
    println!();

    // Initialize all stages
    println!("🚀 Initializing Pipeline Stages...");
    println!();

    // 1. DeckLink Capture
    println!("📹 [1/7] DeckLink Capture");
    let mut capture = CaptureSession::open(config.capture.device_index as i32)?;
    println!("  ✓ Device {} opened", config.capture.device_index);
    println!();

    // 2. CUDA Device
    println!("🔧 [2/7] CUDA Device");
    let cuda_device = CudaDevice::new(config.preprocessing.cuda_device)?;
    println!("  ✓ GPU {} initialized", config.preprocessing.cuda_device);
    println!();

    // 3. Preprocessor (will be updated dynamically based on current_mode)
    println!("⚙️  [3/7] Preprocessor");
    let chroma_order = match config.preprocessing.chroma_order.as_str() {
        "YUY2" => ChromaOrder::YUY2,
        _ => ChromaOrder::UYVY,
    };
    
    // Get initial crop region from current mode (uses preset with auto-scaling)
    let initial_mode = mode_from_u8(current_mode.load(Ordering::SeqCst));
    let crop_region = initial_mode.get_crop_region_preset();
    
    let mut preprocessor = Preprocessor::with_crop_region(
        (
            config.preprocessing.output_width,
            config.preprocessing.output_height,
        ),
        config.preprocessing.use_fp16,
        config.preprocessing.cuda_device as u32,
        config.preprocessing.mean,
        config.preprocessing.std,
        chroma_order,
        crop_region,
    )?;
    println!(
        "  ✓ {}x{} output, {:?} crop",
        config.preprocessing.output_width,
        config.preprocessing.output_height,
        crop_region
    );
    println!();

    // 4. TensorRT Inference
    println!("🧠 [4/7] TensorRT Inference V2");
    if !std::path::Path::new(&config.inference.engine_path).exists() {
        return Err(anyhow!("Engine not found: {}", config.inference.engine_path));
    }
    if !std::path::Path::new(&config.inference.lib_path).exists() {
        return Err(anyhow!("TRT shim not found: {}", config.inference.lib_path));
    }
    let mut inference_stage =
        TrtInferenceStage::new(&config.inference.engine_path, &config.inference.lib_path)
            .map_err(|e| anyhow!(e))?;
    println!("  ✓ Engine: {}", config.inference.engine_path);
    println!("  ✓ Output size: {}", inference_stage.output_size());
    println!();

    // 5. Postprocessing
    println!("🎯 [5/7] Postprocessing");
    let mut post_stage = if config.postprocessing.tracking.enable {
        let mut stage = postprocess::from_path("")?
            .with_sort_tracking(
                config.postprocessing.tracking.max_age,
                config.postprocessing.tracking.min_confidence,
                config.postprocessing.tracking.iou_threshold,
            )
            .with_verbose_stats(config.postprocessing.verbose_stats);
        
        // Enable EMA smoothing for smoother bounding boxes
        // alpha_position = 0.3 (smoother position changes)
        // alpha_size = 0.4 (slightly more responsive size changes)
        stage = stage.with_ema_smoothing(0.3, 0.4);
        stage
    } else {
        postprocess::from_path("")?
            .with_verbose_stats(config.postprocessing.verbose_stats)
    };
    println!("  ✓ Confidence threshold: {}", config.postprocessing.confidence_threshold);
    println!("  ✓ Tracking: {}", if config.postprocessing.tracking.enable { "enabled" } else { "disabled" });
    println!("  ✓ EMA smoothing: {} (α_pos=0.3, α_size=0.4)", if config.postprocessing.tracking.enable { "enabled" } else { "disabled" });
    println!();

    // 6. Overlay Planning & Rendering
    println!("🎨 [6/7] Overlay Planning & GPU Rendering");
    let mut plan_stage = PlanStage {
        enable_full_ui: config.overlay.enable_full_ui,
        spk: config.overlay.show_speaker,  // เปิด/ปิดไอคอนลำโพงตาม config
        display_mode: config.overlay.display_mode.name().to_string(),
        confidence_threshold: config.postprocessing.confidence_threshold,
        endoscope_mode: config.preprocessing.initial_endoscope_mode.name().to_string(),
    };
    let render_config = if config.rendering.debug_rendering {
        "gpu,device=0,debug"
    } else {
        "gpu,device=0"
    };
    let mut render_stage = overlay_render::from_path(render_config)?;
    println!("  ✓ Full UI: {}", config.overlay.enable_full_ui);
    println!("  ✓ Display mode: {}", config.overlay.display_mode.name());
    println!("  ✓ Speaker icon: {}", if config.overlay.show_speaker { "enabled" } else { "disabled" });
    println!("  ✓ GPU rendering initialized (debug: {})", config.rendering.debug_rendering);
    println!();

    // 7. Hardware Internal Keying
    println!("🔧 [7/7] Hardware Internal Keying");

    // Wait for first frame to get dimensions
    println!("  ⏳ Waiting for first frame...");
    let (width, height) = loop {
        if let Some(frame) = capture.get_frame()? {
            let w = frame.meta.width;
            let h = frame.meta.height;
            if w >= 1920 && h >= 1080 {
                println!("  ✓ Got frame: {}x{}", w, h);
                break (w, h);
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    };

    // Initialize DeckLink output
    let decklink_config = if width == 3840 && height == 2160 {
        "configs/dev_4k30_yuv422_fp16_trt.toml"
    } else {
        "configs/dev_1080p60_yuv422_fp16_trt.toml"
    };
    let mut decklink_out = decklink_output::from_path(decklink_config)?;
    println!("  ✓ Output: {}x{} ({})", width, height, decklink_config);

    // Configure SDI and keying
    decklink_out.set_sdi_output()?;
    if config.keying.enable_internal_keying {
        decklink_out.enable_internal_keying()?;
        decklink_out.set_keyer_level(config.keying.keyer_level)?;
        println!("  ✅ Hardware keying enabled (level={})", config.keying.keyer_level);
    }

    // Get frame timing
    let (frame_duration, timebase) = decklink_out
        .get_frame_timing()
        .expect("Failed to get frame timing");
    let fps_calc = timebase as f64 / frame_duration as f64;
    println!("  ✓ Frame timing: {:.2} FPS", fps_calc);
    println!();

    // Pre-roll setup
    let mut scheduled_playback_started = false;
    let preroll_count = 2;
    let mut hw_start_time: Option<u64> = None;
    let frames_ahead = 0u64;

    // Adaptive queue management - Start at optimal depth 2
    let mut max_queue_depth = 2u32;
    let mut pipeline_time_sma = 15.0;  // Expected ~15ms at queue=2
    const SMA_ALPHA: f64 = 0.1;
    let frame_period_ms = 1000.0 / fps_calc;
    
    // Debouncing counters to prevent oscillation
    let mut counter_to_increase = 0u32;
    let mut counter_to_decrease = 0u32;
    const DEBOUNCE_THRESHOLD: u32 = 10; // Must exceed threshold for 10 frames consecutively
    
    // 🎯 FRAME RATE LIMITING: Target frame time to match output FPS
    let target_frame_time = Duration::from_secs_f64(1.0 / fps_calc);
    println!("  🎯 Frame rate limiting enabled:");
    println!("      → Target FPS: {:.2}", fps_calc);
    println!("      → Target frame time: {:.2}ms", target_frame_time.as_secs_f64() * 1000.0);
    println!("      → This prevents pipeline from running faster than output");
    println!();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  PIPELINE RUNNING - Press Ctrl+C to stop               ║");
    println!("║  Press 1, 2, 3 to switch endoscope modes               ║");
    println!("║  Press +/- to adjust confidence threshold (±0.05)      ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Signal handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        println!("\n🛑 Stopping pipeline gracefully...");
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    // Shared confidence threshold (stored as f32 bits in AtomicU32)
    let confidence_threshold = Arc::new(AtomicU32::new(
        config.postprocessing.confidence_threshold.to_bits()
    ));

    // Start keyboard listener thread
    let keyboard_handle = spawn_keyboard_listener(
        current_mode.clone(),
        running.clone(),
        confidence_threshold.clone(),
    );

    // Statistics
    let mut frame_count = 0u64;
    let mut total_latency_ms = 0.0;
    let mut total_capture_ms = 0.0;
    let mut total_preprocess_ms = 0.0;
    let mut total_inference_ms = 0.0;
    let mut total_postprocess_ms = 0.0;
    let mut total_plan_ms = 0.0;
    let mut total_render_ms = 0.0;
    let mut total_keying_ms = 0.0;
    let mut total_packet_prep_ms = 0.0;
    let mut total_queue_mgmt_ms = 0.0;
    let mut total_timing_calc_ms = 0.0;

    let pipeline_start_time = Instant::now();
    let test_duration = if config.general.test_duration_seconds > 0 {
        println!("⏱️  Test duration set to {} seconds", config.general.test_duration_seconds);
        Some(Duration::from_secs(config.general.test_duration_seconds))
    } else {
        println!("⏱️  Running indefinitely (test_duration_seconds = 0)");
        None
    };

    // Main loop
    loop {
        if !running.load(Ordering::SeqCst) {
            break;
        }

        if let Some(duration) = test_duration {
            if pipeline_start_time.elapsed() >= duration {
                println!("\n⏱️  Duration reached - stopping");
                break;
            }
        }

        let pipeline_start = Instant::now();

        // 1. Capture
        let capture_start = Instant::now();
        let raw_frame = match capture.get_frame()? {
            Some(frame) => frame,
            None => {
                std::thread::sleep(Duration::from_millis(16));
                continue;
            }
        };
        let capture_complete_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let capture_latency_ns = capture_complete_ns.saturating_sub(raw_frame.meta.t_capture_ns);
        let capture_latency_ms = capture_latency_ns as f64 / 1_000_000.0;
        total_capture_ms += capture_latency_ms;

        // Frame is already on GPU via DVP capture (zero-copy DMA)
        // No need to copy CPU→GPU - use DVP frame directly
        let raw_frame_gpu = raw_frame.clone();

        // 2. Preprocessing (update crop region if mode changed)
        let current_mode_val = current_mode.load(Ordering::SeqCst);
        let active_mode = mode_from_u8(current_mode_val);
        let new_crop = active_mode.get_crop_region_preset();  // Use preset with auto-scaling
        preprocessor.update_crop_region(new_crop)?;
        
        let preprocess_start = Instant::now();
        let tensor_packet = match preprocessor.process_checked(raw_frame_gpu.clone()) {
            Some(packet) => packet,
            None => {
                eprintln!("❌ Frame {} skipped by preprocessor - size {}x{}", 
                          frame_count, 
                          raw_frame_gpu.meta.width, 
                          raw_frame_gpu.meta.height);
                continue;
            }
        };
        let preprocess_time = preprocess_start.elapsed();
        total_preprocess_ms += preprocess_time.as_secs_f64() * 1000.0;

        // 3. Inference
        let inference_start = Instant::now();
        let raw_detections = inference_stage.process(tensor_packet);
        let inference_time = inference_start.elapsed();
        total_inference_ms += inference_time.as_secs_f64() * 1000.0;

        // 4. Postprocessing
        let postprocess_start = Instant::now();
        
        // Update confidence threshold from keyboard input
        let current_threshold_bits = confidence_threshold.load(Ordering::SeqCst);
        let current_threshold = f32::from_bits(current_threshold_bits);
        post_stage.set_confidence_threshold(current_threshold);
        plan_stage.confidence_threshold = current_threshold;  // อัปเดต threshold สำหรับแสดงบน UI
        
        // Update endoscope mode for overlay positioning
        let mode = mode_from_u8(current_mode.load(Ordering::SeqCst));
        plan_stage.endoscope_mode = mode.name().to_string();  // อัปเดตตำแหน่ง overlay ตามโหมด
        
        let detections = post_stage.process(raw_detections);
        let postprocess_time = postprocess_start.elapsed();
        total_postprocess_ms += postprocess_time.as_secs_f64() * 1000.0;

        // Print detection count for first few frames
        if frame_count < 5 {
            println!("Frame {}: {} detections", frame_count, detections.items.len());
            for (i, det) in detections.items.iter().enumerate().take(3) {
                println!("  Det {}: class={} score={:.3} bbox=({:.0},{:.0},{:.0},{:.0})",
                         i, det.class_id, det.score, det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h);
            }
        }

        // Debug dumps
        if config.general.enable_debug_dumps
            && frame_count < config.general.debug_dump_frame_count
        {
            dump_frame_detections(&detections, frame_count as usize, true)?;
        }

        // 5. Overlay Planning
        let plan_start = Instant::now();
        let overlay_plan = plan_stage.process(detections);
        let plan_time = plan_start.elapsed();
        total_plan_ms += plan_time.as_secs_f64() * 1000.0;

        // 6. GPU Rendering
        let render_start = Instant::now();
        let overlay_frame = render_stage.process(overlay_plan);
        let render_time = render_start.elapsed();
        total_render_ms += render_time.as_secs_f64() * 1000.0;

        // 7. Hardware Keying (with DVP DMA)
        let keying_start = Instant::now();
        
        // 7a. Packet preparation
        let packet_prep_start = Instant::now();
        let overlay_packet = common_io::RawFramePacket {
            meta: common_io::FrameMeta {
                source_id: raw_frame_gpu.meta.source_id,
                width,
                height,
                pixfmt: common_io::PixelFormat::BGRA8,
                colorspace: common_io::ColorSpace::SRGB,
                frame_idx: raw_frame_gpu.meta.frame_idx,
                pts_ns: raw_frame_gpu.meta.pts_ns,
                t_capture_ns: raw_frame_gpu.meta.t_capture_ns,
                stride_bytes: overlay_frame.argb.stride as u32,
                crop_region: None,
            },
            data: overlay_frame.argb,
        };
        let packet_prep_time = packet_prep_start.elapsed().as_secs_f64() * 1000.0;

        // 7b. Queue management
        let queue_mgmt_start = Instant::now();
        let mut buffered_count = decklink_out.get_buffered_frame_count().unwrap_or(0);
        let mut retry_count = 0;
        while buffered_count >= max_queue_depth && retry_count < 50 {
            std::thread::yield_now();
            std::thread::sleep(Duration::from_micros(100));
            buffered_count = decklink_out.get_buffered_frame_count().unwrap_or(0);
            retry_count += 1;
        }
        let queue_mgmt_time = queue_mgmt_start.elapsed().as_secs_f64() * 1000.0;

        // 7c. Timing calculation and scheduling
        let timing_calc_start = Instant::now();
        if buffered_count < max_queue_depth {
            let output_request = OutputRequest {
                video: Some(&raw_frame_gpu),
                overlay: Some(&overlay_packet),
            };

            if !scheduled_playback_started {
                let display_time = (frame_count * frame_duration as u64) as u64;
                // Use DVP (DMA) for zero-copy GPU→DeckLink transfer
                decklink_out.schedule_frame_dvp(output_request, display_time, frame_duration as u64)?;

                if frame_count >= (preroll_count - 1) {
                    let (hw_time, _hw_timebase) = decklink_out
                        .get_hardware_time()
                        .expect("Failed to get hardware reference clock");
                    let start_time = hw_time;
                    decklink_out.start_scheduled_playback(start_time, timebase as f64)?;
                    hw_start_time = Some(hw_time);
                    scheduled_playback_started = true;
                }
            } else {
                let (hw_current_time, _hw_timebase) = decklink_out
                    .get_hardware_time()
                    .expect("Failed to get hardware reference clock");
                let display_time = hw_current_time + (frames_ahead * frame_duration as u64);
                // Use DVP (DMA) for zero-copy GPU→DeckLink transfer
                decklink_out.schedule_frame_dvp(output_request, display_time, frame_duration as u64)?;
            }
        }
        let timing_calc_time = timing_calc_start.elapsed().as_secs_f64() * 1000.0;

        let keying_time = keying_start.elapsed();
        total_keying_ms += keying_time.as_secs_f64() * 1000.0;
        total_packet_prep_ms += packet_prep_time;
        total_queue_mgmt_ms += queue_mgmt_time;
        total_timing_calc_ms += timing_calc_time;

        let pipeline_time = pipeline_start.elapsed();
        let pipeline_ms = pipeline_time.as_secs_f64() * 1000.0;
        total_latency_ms += pipeline_ms;

        // 🎯 FRAME RATE LIMITING: Sleep if pipeline finished too fast
        // This prevents pipeline from running faster than output FPS
        // which reduces queue buildup and latency
        if pipeline_time < target_frame_time {
            let sleep_duration = target_frame_time - pipeline_time;
            std::thread::sleep(sleep_duration);
            
            if frame_count % 60 == 0 {
                println!("  ⏱️  Frame rate limit: slept {:.2}ms (pipeline took {:.2}ms, target {:.2}ms)",
                    sleep_duration.as_secs_f64() * 1000.0,
                    pipeline_ms,
                    target_frame_time.as_secs_f64() * 1000.0
                );
            }
        }

        // Adaptive queue adjustment with debouncing to prevent oscillation
        // STRATEGY: Stay at queue=2 as optimal depth, only change if really necessary
        pipeline_time_sma = SMA_ALPHA * pipeline_ms + (1.0 - SMA_ALPHA) * pipeline_time_sma;
        
        let current_depth = max_queue_depth;
        let utilization = pipeline_time_sma / frame_period_ms;
        
        // Determine desired queue depth based on utilization
        // Use VERY wide thresholds to keep queue stable at depth 2
        let desired_depth = match current_depth {
            1 => {
                // Currently at 1: increase to 2 if above 80% (we almost never want to be at 1)
                if utilization >= 0.80 {
                    2
                } else {
                    1
                }
            }
            2 => {
                // Currently at 2 (OPTIMAL): only decrease if extremely fast (<60%), only increase if very slow (>100%)
                if utilization < 0.60 {
                    1  // Extremely fast - can drop to 1
                } else if utilization >= 1.00 {
                    3  // Getting slow - increase to 3
                } else {
                    2  // Stay at 2 (normal range: 60-100% = 10-16.67ms pipeline)
                }
            }
            3 => {
                // Currently at 3: decrease to 2 if below 95%, increase if above 110%
                if utilization < 0.95 {
                    2
                } else if utilization >= 1.10 {
                    4
                } else {
                    3
                }
            }
            _ => {
                // 4 or higher: decrease to 3 if below 105%
                if utilization < 1.05 {
                    3
                } else {
                    current_depth
                }
            }
        };
        
        // Apply debouncing: only change if desired_depth differs from current_depth
        // for DEBOUNCE_THRESHOLD consecutive frames
        if desired_depth > current_depth {
            // Want to increase
            counter_to_increase += 1;
            counter_to_decrease = 0;
            
            if counter_to_increase >= DEBOUNCE_THRESHOLD {
                max_queue_depth = desired_depth;
                counter_to_increase = 0;
                let utilization_pct = utilization * 100.0;
                println!("[queue] 🔧 Queue depth INCREASED: {} → {} (pipeline: {:.2}ms = {:.1}% of {:.2}ms, debounced {} frames)", 
                         current_depth, max_queue_depth, pipeline_time_sma, utilization_pct, frame_period_ms, DEBOUNCE_THRESHOLD);
            }
        } else if desired_depth < current_depth {
            // Want to decrease
            counter_to_decrease += 1;
            counter_to_increase = 0;
            
            if counter_to_decrease >= DEBOUNCE_THRESHOLD {
                max_queue_depth = desired_depth;
                counter_to_decrease = 0;
                let utilization_pct = utilization * 100.0;
                println!("[queue] 🔧 Queue depth DECREASED: {} → {} (pipeline: {:.2}ms = {:.1}% of {:.2}ms, debounced {} frames)", 
                         current_depth, max_queue_depth, pipeline_time_sma, utilization_pct, frame_period_ms, DEBOUNCE_THRESHOLD);
            }
        } else {
            // desired_depth == current_depth, reset counters
            counter_to_increase = 0;
            counter_to_decrease = 0;
        }

        frame_count += 1;

        // Print runtime statistics
        if config.general.enable_runtime_statistics && frame_count % config.general.stats_print_interval == 0 {
            let elapsed = pipeline_start_time.elapsed().as_secs_f64();
            let avg_fps = frame_count as f64 / elapsed;
            
            // Calculate average latencies
            let avg_capture = total_capture_ms / frame_count as f64;
            let avg_preprocess = total_preprocess_ms / frame_count as f64;
            let avg_inference = total_inference_ms / frame_count as f64;
            let avg_postprocess = total_postprocess_ms / frame_count as f64;
            let avg_plan = total_plan_ms / frame_count as f64;
            let avg_render = total_render_ms / frame_count as f64;
            let avg_keying = total_keying_ms / frame_count as f64;
            let avg_total = total_latency_ms / frame_count as f64;
            
            println!("╔══════════════════════════════════════════════════════════╗");
            println!("║  📊 STATISTICS - Frame {}                              ", frame_count);
            println!("╚══════════════════════════════════════════════════════════╝");
            println!("  🎯 Pipeline Performance:");
            println!("    FPS:                {:.2}", avg_fps);
            println!("    Queue:              {}/{}", buffered_count, max_queue_depth);
            println!();
            println!("  ⏱️  Average Latency Breakdown:");
            println!("    1. Capture:         {:.2}ms ({:.1}%)", avg_capture, (avg_capture / avg_total) * 100.0);
            println!("    2. Preprocessing:   {:.2}ms ({:.1}%)", avg_preprocess, (avg_preprocess / avg_total) * 100.0);
            println!("    3. Inference:       {:.2}ms ({:.1}%)", avg_inference, (avg_inference / avg_total) * 100.0);
            println!("    4. Postprocessing:  {:.2}ms ({:.1}%)", avg_postprocess, (avg_postprocess / avg_total) * 100.0);
            println!("    5. Overlay Plan:    {:.2}ms ({:.1}%)", avg_plan, (avg_plan / avg_total) * 100.0);
            println!("    6. GPU Rendering:   {:.2}ms ({:.1}%)", avg_render, (avg_render / avg_total) * 100.0);
            println!("    7. Hardware Keying: {:.2}ms ({:.1}%)", avg_keying, (avg_keying / avg_total) * 100.0);
            println!("    ─────────────────────────────────────────────────");
            println!("    TOTAL:              {:.2}ms (100%)", avg_total);
            println!();
        }
    }

    // Final summary
    let total_elapsed = pipeline_start_time.elapsed().as_secs_f64();
    let avg_fps = frame_count as f64 / total_elapsed;

    if config.general.enable_final_summary {
        println!();
        println!("╔══════════════════════════════════════════════════════════╗");
        println!("║  FINAL SUMMARY - HARDWARE KEYING PIPELINE               ║");
        println!("╚══════════════════════════════════════════════════════════╝");
        println!();
        println!("  📈 Performance:");
        println!("    Total frames:       {}", frame_count);
        println!("    Total time:         {:.2}s", total_elapsed);
        println!("    Average FPS:        {:.2}", avg_fps);
        println!();
        println!("  ⏱️  Average Latency:");
        println!(
            "    Capture:            {:.2}ms",
            total_capture_ms / frame_count as f64
        );
        println!(
            "    Preprocessing:      {:.2}ms",
            total_preprocess_ms / frame_count as f64
        );
        println!(
            "    Inference:          {:.2}ms",
            total_inference_ms / frame_count as f64
        );
        println!(
            "    Postprocessing:     {:.2}ms",
            total_postprocess_ms / frame_count as f64
        );
        println!(
            "    Overlay Planning:   {:.2}ms",
            total_plan_ms / frame_count as f64
        );
        println!(
            "    GPU Rendering:      {:.2}ms",
            total_render_ms / frame_count as f64
        );
        
        // Calculate averages for hardware keying breakdown
        let avg_keying = total_keying_ms / frame_count as f64;
        let avg_packet_prep = total_packet_prep_ms / frame_count as f64;
        let avg_queue_mgmt = total_queue_mgmt_ms / frame_count as f64;
        let avg_timing_calc = total_timing_calc_ms / frame_count as f64;
        
        // Get DVP-specific timings from last frame
        let (_packet_prep_dvp, _queue_mgmt_dvp, dma_copy, api, scheduling_dvp) = decklink_out.get_last_frame_timing();
        
        println!(
            "    Hardware Keying:    {:.2}ms",
            avg_keying
        );
        println!("      ├─ Packet prep:     {:.2}ms", avg_packet_prep);
        println!("      ├─ Queue mgmt:      {:.2}ms", avg_queue_mgmt);
        println!("      ├─ Timing calc:     {:.2}ms", avg_timing_calc);
        println!("      │   ├─ DMA transfer: {:.2}ms", dma_copy);
        println!("      │   ├─ DeckLink API: {:.2}ms", api);
        println!("      │   └─ Scheduling:   {:.2}ms", scheduling_dvp);
        println!("      └─ (Other overhead): {:.2}ms", 
            avg_keying - avg_packet_prep - avg_queue_mgmt - avg_timing_calc);
        
        println!("    ─────────────────────────────────");
        println!(
            "    Total (E2E):        {:.2}ms",
            total_latency_ms / frame_count as f64
        );
        println!();
        println!("✅ Pipeline completed successfully!");
        println!();
    }

    // Wait for keyboard listener thread to finish
    keyboard_handle.join().ok();

    Ok(())
}

/// Inference Only Pipeline
/// Minimal pipeline: Capture → Preprocess → Inference → Postprocess (no output)
fn run_inference_only_pipeline(config: &PipelineConfig) -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  DEEPGIBOX - INFERENCE ONLY PIPELINE                    ║");
    println!("║  Benchmark Mode: No output, maximum throughput          ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Initialize stages (similar to keying pipeline but without output)
    println!("🚀 Initializing Pipeline Stages...");
    println!();

    let mut capture = CaptureSession::open(config.capture.device_index as i32)?;
    println!("📹 Capture: Device {}", config.capture.device_index);

    let cuda_device = CudaDevice::new(config.preprocessing.cuda_device)?;
    println!("🔧 CUDA: GPU {}", config.preprocessing.cuda_device);

    let crop_region = match config.preprocessing.crop_region.as_str() {
        "Olympus" => CropRegion::Olympus,
        "Pentax" => CropRegion::Pentax,
        "Fuji" => CropRegion::Fuji,
        _ => CropRegion::Olympus,
    };
    let chroma_order = match config.preprocessing.chroma_order.as_str() {
        "YUY2" => ChromaOrder::YUY2,
        _ => ChromaOrder::UYVY,
    };
    let mut preprocessor = Preprocessor::with_crop_region(
        (
            config.preprocessing.output_width,
            config.preprocessing.output_height,
        ),
        config.preprocessing.use_fp16,
        config.preprocessing.cuda_device as u32,
        config.preprocessing.mean,
        config.preprocessing.std,
        chroma_order,
        crop_region,
    )?;
    println!("⚙️  Preprocessing: {}x{}", config.preprocessing.output_width, config.preprocessing.output_height);

    let mut inference_stage =
        TrtInferenceStage::new(&config.inference.engine_path, &config.inference.lib_path)
            .map_err(|e| anyhow!(e))?;
    println!("🧠 Inference: {}", config.inference.engine_path);

    let mut post_stage = if config.postprocessing.tracking.enable {
        postprocess::from_path("")?.with_sort_tracking(
            config.postprocessing.tracking.max_age,
            config.postprocessing.tracking.min_confidence,
            config.postprocessing.tracking.iou_threshold,
        )
    } else {
        postprocess::from_path("")?
    };
    println!("🎯 Postprocessing: Ready");
    println!();

    // Signal handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        println!("\n🛑 Stopping pipeline...");
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    // Shared confidence threshold (stored as f32 bits in AtomicU32)
    let confidence_threshold = Arc::new(AtomicU32::new(
        config.postprocessing.confidence_threshold.to_bits()
    ));

    let mut frame_count = 0u64;
    let mut total_latency_ms = 0.0;
    let pipeline_start_time = Instant::now();
    let test_duration = if config.general.test_duration_seconds > 0 {
        Some(Duration::from_secs(config.general.test_duration_seconds))
    } else {
        None
    };

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  PIPELINE RUNNING - Press Ctrl+C to stop               ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    loop {
        if !running.load(Ordering::SeqCst) {
            break;
        }

        if let Some(duration) = test_duration {
            if pipeline_start_time.elapsed() >= duration {
                break;
            }
        }

        let pipeline_start = Instant::now();

        // Capture
        let raw_frame = match capture.get_frame()? {
            Some(frame) => frame,
            None => {
                std::thread::sleep(Duration::from_millis(16));
                continue;
            }
        };

        // Frame is already on GPU via DVP capture (zero-copy DMA)
        let raw_frame_gpu = raw_frame.clone();

        // Preprocess
        let tensor_packet = match preprocessor.process_checked(raw_frame_gpu) {
            Some(packet) => packet,
            None => continue,
        };

        // Inference
        let raw_detections = inference_stage.process(tensor_packet);

        // Postprocess
        // Update confidence threshold from keyboard input
        let current_threshold_bits = confidence_threshold.load(Ordering::SeqCst);
        let current_threshold = f32::from_bits(current_threshold_bits);
        post_stage.set_confidence_threshold(current_threshold);
        
        let detections = post_stage.process(raw_detections);

        // Debug dumps
        if config.general.enable_debug_dumps
            && frame_count < config.general.debug_dump_frame_count
        {
            dump_frame_detections(&detections, frame_count as usize, true)?;
        }

        let pipeline_ms = pipeline_start.elapsed().as_secs_f64() * 1000.0;
        total_latency_ms += pipeline_ms;
        frame_count += 1;

        if frame_count % config.general.stats_print_interval == 0 {
            let elapsed = pipeline_start_time.elapsed().as_secs_f64();
            let avg_fps = frame_count as f64 / elapsed;
            println!(
                "📊 Frame {} | Latency: {:.2}ms | FPS: {:.2} | Detections: {}",
                frame_count,
                total_latency_ms / frame_count as f64,
                avg_fps,
                detections.items.len()
            );
        }
    }

    let total_elapsed = pipeline_start_time.elapsed().as_secs_f64();
    let avg_fps = frame_count as f64 / total_elapsed;

    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  FINAL SUMMARY - INFERENCE ONLY PIPELINE                ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("  Total frames:       {}", frame_count);
    println!("  Total time:         {:.2}s", total_elapsed);
    println!("  Average FPS:        {:.2}", avg_fps);
    println!(
        "  Average latency:    {:.2}ms",
        total_latency_ms / frame_count as f64
    );
    println!();
    println!("✅ Pipeline completed!");
    println!();

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN ENTRY POINT
// ═══════════════════════════════════════════════════════════════════════════

fn main() -> Result<()> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <config_path>", args[0]);
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} configs/runner_keying.toml", args[0]);
        eprintln!("  {} configs/runner_inference_only.toml", args[0]);
        eprintln!();
        return Err(anyhow!("Missing config path argument"));
    }

    let config_path = &args[1];

    // Load configuration
    println!("📋 Loading configuration: {}", config_path);
    let config = config_loader::load_config(config_path)?;
    println!("  ✓ Mode: {:?}", config.mode);
    println!();

    // Run appropriate pipeline
    match config.mode {
        PipelineMode::HardwareKeying => run_keying_pipeline(&config),
        PipelineMode::InferenceOnly => run_inference_only_pipeline(&config),
        PipelineMode::Visualization => {
            Err(anyhow!("Visualization mode not yet implemented"))
        }
    }
}
