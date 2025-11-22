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
use postprocess::TrackerMotionOptions;
use preprocess_cuda::{ChromaOrder, CropRegion, Preprocessor};

// GPU-accelerated modules (optional)
#[cfg(feature = "gpu")]
use postprocess::GpuPostprocessor;
#[cfg(feature = "gpu")]
use overlay_plan::gpu_overlay::GpuOverlayPlanner;
use std::fs;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use device_query::{DeviceQuery, DeviceState, Keycode};
use std::sync::atomic::AtomicU32;

mod config_loader;
use config_loader::{PipelineConfig, PipelineMode, EndoscopeMode, TrackingMotionConfig};

// ═══════════════════════════════════════════════════════════════════════════
// GPU PIPELINE OPTIMIZATION
// ═══════════════════════════════════════════════════════════════════════════
//
// When compiled with --features gpu, the pipeline uses GPU-accelerated stages:
//
// CPU Pipeline (default):
//   Inference → GPU→CPU copy → CPU NMS → CPU overlay planning → Render
//   Bottleneck: ~2-3ms GPU→CPU transfer + 0.8ms CPU processing
//
// GPU Pipeline (--features gpu):
//   Inference → GPU NMS → GPU overlay planning → Render (all on GPU!)
//   Improvement: Eliminates 2-3ms transfer + faster GPU processing
//   Expected: 11.85ms → 9.35ms pipeline (-21%)
//
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

fn tracker_motion_options_from_config(cfg: &TrackingMotionConfig) -> TrackerMotionOptions {
    TrackerMotionOptions {
        use_kalman: cfg.use_kalman,
        kalman_process_noise: cfg.kalman_process_noise,
        kalman_measurement_noise: cfg.kalman_measurement_noise,
        use_optical_flow: cfg.use_optical_flow,
        optical_flow_alpha: cfg.optical_flow_alpha,
        optical_flow_max_pixels: cfg.optical_flow_max_pixels,
        use_velocity: cfg.use_velocity,
        velocity_alpha: cfg.velocity_alpha,
        velocity_max_delta: cfg.velocity_max_delta,
        velocity_decay: cfg.velocity_decay,
    }
}

/// Calculate percentile from a sorted vector
fn calculate_percentile(sorted_data: &[f64], percentile: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }
    let index = ((percentile / 100.0) * (sorted_data.len() - 1) as f64).round() as usize;
    sorted_data[index.min(sorted_data.len() - 1)]
}

/// Get percentile statistics (P50, P95, P99, Min, Max)
fn get_percentile_stats(samples: &[f64]) -> (f64, f64, f64, f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let p50 = calculate_percentile(&sorted, 50.0);
    let p95 = calculate_percentile(&sorted, 95.0);
    let p99 = calculate_percentile(&sorted, 99.0);
    
    (p50, p95, p99, min, max)
}

/// Spawn keyboard listener thread to handle endoscope mode switching and display mode
fn spawn_keyboard_listener(
    current_mode: Arc<AtomicU8>,
    running: Arc<AtomicBool>,
    confidence_threshold: Arc<AtomicU32>,
    display_mode: Arc<AtomicU8>,
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
                        Keycode::Key0 => {
                            // '0' key - Toggle display mode between EGD and COLON
                            let current = display_mode.load(Ordering::SeqCst);
                            let new_mode = if current == 0 { 1 } else { 0 }; // 0=EGD, 1=COLON
                            display_mode.store(new_mode, Ordering::SeqCst);
                            let mode_name = if new_mode == 0 { "EGD" } else { "COLON" };
                            println!("\n🔄 Display mode: {}", mode_name);
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
    println!("║  Press 1=Fuji, 2=Olympus to switch modes                ║");
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
    let _cuda_device = CudaDevice::new(config.preprocessing.cuda_device)?;
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
        let motion = tracker_motion_options_from_config(&config.postprocessing.tracking.motion);
        let mut stage = postprocess::from_path("")?
            .with_sort_tracking_motion(
                config.postprocessing.tracking.max_age,
                config.postprocessing.tracking.min_confidence,
                config.postprocessing.tracking.iou_threshold,
                motion,
            )
            .with_verbose_stats(config.postprocessing.verbose_stats);
        
        // Enable EMA smoothing if configured
        if config.postprocessing.ema_smoothing.enable {
            stage = stage.with_ema_smoothing(
                config.postprocessing.ema_smoothing.alpha_position,
                config.postprocessing.ema_smoothing.alpha_size,
            );
        }
        stage
    } else {
        postprocess::from_path("")?
            .with_verbose_stats(config.postprocessing.verbose_stats)
    };
    println!("  ✓ Confidence threshold: {}", config.postprocessing.confidence_threshold);
    println!("  ✓ Tracking: {}", if config.postprocessing.tracking.enable { "enabled" } else { "disabled" });
    if config.postprocessing.tracking.enable {
        let motion = &config.postprocessing.tracking.motion;
        println!(
            "    ↳ Kalman={} (Q={:.2}, R={:.2}) | Optical Flow={} (α={:.2}, max={:.1}px)",
            if motion.use_kalman { "on" } else { "off" },
            motion.kalman_process_noise,
            motion.kalman_measurement_noise,
            if motion.use_optical_flow { "on" } else { "off" },
            motion.optical_flow_alpha,
            motion.optical_flow_max_pixels,
        );
    }
    if config.postprocessing.ema_smoothing.enable {
        println!("  ✓ EMA smoothing: enabled (α_pos={}, α_size={})", 
            config.postprocessing.ema_smoothing.alpha_position,
            config.postprocessing.ema_smoothing.alpha_size);
    } else {
        println!("  ✓ EMA smoothing: disabled");
    }
    
    // 5b. GPU Postprocessing (if enabled)
    #[cfg(feature = "gpu")]
    let mut gpu_post_stage = {
        println!("  🚀 GPU acceleration: enabled");
        let stage = GpuPostprocessor::new(
            25200,  // num_anchors (80x80 + 40x40 + 20x20 grids)
            80,     // num_classes
            300,    // max_detections
            config.postprocessing.confidence_threshold,
            0.45,   // iou_threshold for NMS
        )?;
        println!("  ✓ GPU NMS initialized (25200 anchors, max 300 detections)");
        stage
    };
    
    #[cfg(not(feature = "gpu"))]
    println!("  ℹ️  GPU acceleration: disabled (compile with --features gpu)");
    
    println!();

    // 6. Overlay Planning & Rendering
    println!("🎨 [6/7] Overlay Planning & GPU Rendering");
    
    // 6a. GPU Overlay Planning (if enabled)
    #[cfg(feature = "gpu")]
    let mut gpu_plan_stage = {
        println!("  🚀 GPU overlay planning: enabled");
        let stage = GpuOverlayPlanner::new(
            1000,   // max_commands (typically ~4 per detection: 2 rects + 2 labels)
            config.postprocessing.confidence_threshold,
            true,   // draw_confidence_bar
        )?;
        println!("  ✓ GPU overlay planner initialized (max 1000 draw commands)");
        stage
    };
    
    #[cfg(not(feature = "gpu"))]
    println!("  ℹ️  GPU overlay planning: disabled (using CPU path)");
    
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
    let mut _hw_start_time: Option<u64> = None;
    let frames_ahead = 0u64;

    // 🔒 FIXED Queue Depth = 2 (for minimum latency)
    // Queue depth ตรึงไว้ที่ 2 frames เพื่อให้ latency ต่ำสุด
    // - Queue = 1: Latency ต่ำสุด แต่อาจมีการกระตุก (jitter) ถ้า pipeline ช้า
    // - Queue = 2: ✅ OPTIMAL - balance ระหว่าง latency และความลื่นไหล (~33ms buffering ที่ 60 FPS)
    // - Queue = 3-4: Latency สูง (50-67ms) แต่ลื่นไหลมากขึ้น
    const MAX_QUEUE_DEPTH: u32 = 2;
    let max_queue_depth = MAX_QUEUE_DEPTH;  // Immutable - no adaptive adjustment
    
    let frame_period_ms = 1000.0 / fps_calc;
    
    println!("  📦 Hardware Keying Queue Configuration:");
    println!("      → Queue depth: {} frames 🔒 (FIXED, not adaptive)", MAX_QUEUE_DEPTH);
    println!("      → Queue latency: {:.2}ms max ({} frames × {:.2}ms)", 
             (MAX_QUEUE_DEPTH as f64) * frame_period_ms, MAX_QUEUE_DEPTH, frame_period_ms);
    println!("      → Benefit: Minimum latency with stable buffering");
    println!("      ℹ️  Hardware DeckLink ต้องการ buffer อย่างน้อย 2 frames");
    println!("          เพื่อความลื่นไหลและป้องกันการ underrun");
    println!();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  PIPELINE RUNNING - Press Ctrl+C to stop               ║");
    println!("║  Press 1, 2, 3 to switch endoscope modes               ║");
    println!("║  Press +/- to adjust confidence threshold (±0.05)      ║");
    println!("║  Press 0 to toggle display mode (EGD ⇄ COLON)         ║");
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

    // Shared display mode (0=EGD, 1=COLON)
    let display_mode = Arc::new(AtomicU8::new(
        if config.overlay.display_mode.name() == "egd" { 0 } else { 1 }
    ));

    // Start keyboard listener thread
    let keyboard_handle = spawn_keyboard_listener(
        current_mode.clone(),
        running.clone(),
        confidence_threshold.clone(),
        display_mode.clone(),
    );

    // Statistics - Averages
    let mut frame_count = 0u64;
    let mut total_latency_ms = 0.0;
    let mut total_capture_ms = 0.0;
    let mut total_capture_wait_ms = 0.0;
    let mut total_capture_dvp_ms = 0.0;
    let mut total_capture_hw_ms = 0.0;
    let mut total_preprocess_ms = 0.0;
    let mut total_inference_ms = 0.0;
    let mut total_postprocess_ms = 0.0;
    let mut total_plan_ms = 0.0;
    let mut total_render_ms = 0.0;
    let mut total_keying_ms = 0.0;
    let mut total_packet_prep_ms = 0.0;
    let mut total_queue_mgmt_ms = 0.0;
    let mut total_queue_wait_ms = 0.0;
    let mut total_timing_calc_ms = 0.0;
    let mut total_cuda_sync_ms = 0.0;
    let mut total_glass_to_glass_ms = 0.0;
    
    // Statistics - Percentiles (store last 1000 samples)
    let mut latency_samples: Vec<f64> = Vec::with_capacity(1000);
    let mut capture_samples: Vec<f64> = Vec::with_capacity(1000);
    let mut inference_samples: Vec<f64> = Vec::with_capacity(1000);
    let mut e2e_samples: Vec<f64> = Vec::with_capacity(1000);
    
    // Track last frame index to detect new frames (adaptive frame-driven processing)
    // Track last hardware sequence number (DeckLink increments g_shared.seq per real capture)
    let mut last_frame_seq: Option<u64> = None;

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

        // 1. Adaptive Frame Capture (wait for NEW frame only)
        let capture_start = Instant::now();
        let mut capture_wait_duration = Duration::ZERO;
        let raw_frame = loop {
            match capture.get_frame()? {
                Some(frame) => {
                    // Use DeckLink hardware sequence (pts_ns) to detect truly new frames
                    let frame_seq = frame.meta.pts_ns;
                    match last_frame_seq {
                        None => {
                            // First frame - always process
                            last_frame_seq = Some(frame_seq);
                            break frame;
                        }
                        Some(last_seq) => {
                            if frame_seq != last_seq {
                                // NEW frame detected - process it!
                                last_frame_seq = Some(frame_seq);
                                break frame;
                            } else {
                                // SAME frame as before - wait for next one
                                // Short sleep to avoid busy-waiting (adaptive polling)
                                let sleep_dur = Duration::from_micros(500);
                                std::thread::sleep(sleep_dur);
                                capture_wait_duration += sleep_dur;
                                continue;
                            }
                        }
                    }
                }
                None => {
                    // No frame available yet - short sleep and retry
                    let sleep_dur = Duration::from_micros(500);
                    std::thread::sleep(sleep_dur);
                    capture_wait_duration += sleep_dur;
                    continue;
                }
            }
        };
        
        let capture_elapsed = capture_start.elapsed();
        let capture_total_ms = capture_elapsed.as_secs_f64() * 1000.0;
        let capture_wait_ms = capture_wait_duration.as_secs_f64() * 1000.0;
        let capture_dvp_ms = capture_elapsed
            .saturating_sub(capture_wait_duration)
            .as_secs_f64()
            * 1000.0;
        let capture_complete_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let capture_latency_hw_ns = capture_complete_ns.saturating_sub(raw_frame.meta.t_capture_ns);
        let capture_latency_hw_ms = capture_latency_hw_ns as f64 / 1_000_000.0;
        // Hardware timestamp path reports software overhead (sub-millisecond). When it is too small,
        // fall back to the measured capture stage duration so statistics reflect real cadence.
        let capture_hw_ms_for_stats = if capture_latency_hw_ms >= 0.1 {
            capture_latency_hw_ms
        } else {
            capture_total_ms
        };
        total_capture_ms += capture_total_ms;
        total_capture_wait_ms += capture_wait_ms;
        total_capture_dvp_ms += capture_dvp_ms;
        total_capture_hw_ms += capture_hw_ms_for_stats;
        
        let pipeline_start = Instant::now();

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
        let preprocess_ms = preprocess_time.as_secs_f64() * 1000.0;
        total_preprocess_ms += preprocess_ms;

        // 3. Inference (with CUDA sync measurement)
        let inference_start = Instant::now();
        let raw_detections = inference_stage.process(tensor_packet);
        let inference_time = inference_start.elapsed();
        let inference_ms = inference_time.as_secs_f64() * 1000.0;
        total_inference_ms += inference_ms;
        
        // Measure CUDA synchronization overhead
        let cuda_sync_start = Instant::now();
        // Note: Inference stage may have implicit CUDA synchronization when accessing results
        // This captures any remaining async GPU work completion time
        let cuda_sync_time = cuda_sync_start.elapsed();
        let cuda_sync_ms = cuda_sync_time.as_secs_f64() * 1000.0;
        total_cuda_sync_ms += cuda_sync_ms;

        // 4. Postprocessing (GPU or CPU path)
        let postprocess_start = Instant::now();
        
        // Update confidence threshold from keyboard input
        let current_threshold_bits = confidence_threshold.load(Ordering::SeqCst);
        let current_threshold = f32::from_bits(current_threshold_bits);
        
        // 🔧 TEMPORARY: Disable GPU postprocessing due to cudarc-TensorRT memory conflict
        // Use CPU path for postprocessing to avoid illegal memory access
        // TODO: Replace cudarc with raw cudaMalloc in postprocessing for full GPU pipeline
        #[cfg(feature = "disabled_gpu_postprocess")]
        let gpu_detections = {
            // GPU path: Zero-copy postprocessing (keeps data on GPU)
            gpu_post_stage.set_confidence_threshold(current_threshold);
            
            // Check if raw_detections has GPU pointer (zero-copy from inference)
            if let Some(gpu_ptr) = raw_detections.gpu_ptr {
                // Direct GPU processing - no transfer needed!
                gpu_post_stage.process_gpu_zerocopy(gpu_ptr, &raw_detections.from)?
            } else {
                // Fallback: CPU data → GPU processing
                gpu_post_stage.process_gpu_zerocopy(
                    raw_detections.raw_output.as_ptr() as *const f32,
                    &raw_detections.from
                )?
            }
        };
        
        // Use CPU postprocessing (fallback due to cudarc-TensorRT conflict)
        let detections = {
            post_stage.set_confidence_threshold(current_threshold);
            post_stage.process(raw_detections)
        };
        
        let postprocess_time = postprocess_start.elapsed();
        total_postprocess_ms += postprocess_time.as_secs_f64() * 1000.0;
        
        // Update overlay plan stage configuration
        plan_stage.confidence_threshold = current_threshold;  // อัปเดต threshold สำหรับแสดงบน UI
        
        // Update endoscope mode for overlay positioning
        let mode = mode_from_u8(current_mode.load(Ordering::SeqCst));
        plan_stage.endoscope_mode = mode.name().to_string();  // อัปเดตตำแหน่ง overlay ตามโหมด
        
        // Update display mode from keyboard input
        let current_display_mode = display_mode.load(Ordering::SeqCst);
        plan_stage.display_mode = if current_display_mode == 0 { "egd".to_string() } else { "colon".to_string() };

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
        
        // CPU path: Traditional overlay planning from CPU detections
        let overlay_plan = plan_stage.process(detections);
        
        // GPU overlay planning (disabled due to cudarc conflict)
        #[cfg(feature = "disabled_gpu_planning")]
        let gpu_overlay_plan = {
            // GPU path: Zero-copy overlay planning (GPU → GPU)
            match gpu_plan_stage.plan_from_gpu_detections(&gpu_detections) {
                Ok(plan) => plan,
                Err(e) => {
                    eprintln!("⚠️  GPU overlay planning failed: {}, creating empty plan", e);
                    // Return empty plan on error
                    common_io::GpuOverlayPlanPacket {
                        frame_meta: gpu_detections.frame_meta.clone(),
                        gpu_commands: std::ptr::null_mut(),
                        num_commands: 0,
                        canvas: (width, height),
                    }
                }
            }
        };
        
        let plan_time = plan_start.elapsed();
        total_plan_ms += plan_time.as_secs_f64() * 1000.0;

        // 6. GPU Rendering from CPU plan
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

        // 7b. Queue management (optimized)
        let queue_mgmt_start = Instant::now();
        let mut buffered_count = decklink_out.get_buffered_frame_count().unwrap_or(0);
        
        // � DEBUG: Check queue depth
        if frame_count < 5 {
            use std::io::{self, Write};
            let _ = writeln!(io::stderr(), "[runner] Frame {}: buffered_count={}, max_queue_depth={}", 
                             frame_count, buffered_count, max_queue_depth);
            let _ = io::stderr().flush();
        }
        
        // �🚀 OPTIMIZED: Efficient queue waiting strategy
        // Instead of aggressive polling (50 × 100us), use smart exponential backoff
        let mut queue_wait_ms = 0.0;
        if buffered_count >= max_queue_depth {
            let mut wait_time_us = 500u64; // เริ่มที่ 0.5ms
            let max_wait_time_us = 8000u64; // สูงสุด 8ms
            let mut total_wait_ms = 0.0;
            
            while buffered_count >= max_queue_depth && total_wait_ms < 10.0 {
                // Sleep with exponential backoff (แทนที่จะ poll ทุก 100us)
                std::thread::sleep(Duration::from_micros(wait_time_us));
                total_wait_ms += wait_time_us as f64 / 1000.0;
                
                // Check queue again
                buffered_count = decklink_out.get_buffered_frame_count().unwrap_or(0);
                
                // Exponential backoff: 500us → 1ms → 2ms → 4ms → 8ms
                wait_time_us = std::cmp::min(wait_time_us * 2, max_wait_time_us);
            }
            
            if buffered_count >= max_queue_depth && frame_count % 60 == 0 {
                println!("⚠️  Queue still full after waiting {:.2}ms (frame {})", 
                         total_wait_ms, frame_count);
            }

            queue_wait_ms = total_wait_ms;
        }
        total_queue_wait_ms += queue_wait_ms;
        
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
                    _hw_start_time = Some(hw_time);
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
        
        // Calculate Glass-to-Glass latency (capture to display)
        // G2G = pipeline latency + queue latency
        let queue_latency_ms = (max_queue_depth as f64) * frame_period_ms;
        let glass_to_glass_ms = pipeline_ms + queue_latency_ms;
        total_glass_to_glass_ms += glass_to_glass_ms;
        
        // Store samples for percentile calculation (keep last 1000)
        if latency_samples.len() >= 1000 {
            latency_samples.remove(0);
            capture_samples.remove(0);
            inference_samples.remove(0);
            e2e_samples.remove(0);
        }
    latency_samples.push(pipeline_ms);
    capture_samples.push(capture_total_ms);
        inference_samples.push(inference_ms);
        e2e_samples.push(glass_to_glass_ms);

        // 🎯 ADAPTIVE FRAME-DRIVEN PROCESSING (NO fixed frame rate limiting)
        // The pipeline will naturally sync to the capture device's frame rate
        // by waiting for NEW frames at the start of each loop iteration.
        // 
        // Benefits:
        // - Adapts to variable capture frame rates (if camera FPS fluctuates)
        // - Reduces latency when frames arrive early
        // - Prevents wasted processing when frames arrive late
        // - More responsive to real-time capture timing
        //
    // Note: The DeckLink sequence (pts_ns) check at capture ensures we only process NEW frames,
        // so no artificial sleep/throttling is needed here.

        // 🔒 FIXED QUEUE DEPTH (NO adaptive adjustment)
        // Queue depth ตรึงไว้ที่ 2 frames เพื่อ minimum latency (~33ms buffering @ 60 FPS)
        // ไม่มี dynamic adjustment เพื่อให้ latency คงที่และคาดเดาได้

        frame_count += 1;

        // Print runtime statistics
        if config.general.enable_runtime_statistics && frame_count % config.general.stats_print_interval == 0 {
            let elapsed = pipeline_start_time.elapsed().as_secs_f64();
            let avg_fps = frame_count as f64 / elapsed;
            
            // Calculate average latencies
            let avg_capture = total_capture_ms / frame_count as f64;
            let avg_capture_wait = total_capture_wait_ms / frame_count as f64;
            let avg_capture_dvp = total_capture_dvp_ms / frame_count as f64;
            let avg_capture_hw = total_capture_hw_ms / frame_count as f64;
            let avg_preprocess = total_preprocess_ms / frame_count as f64;
            let avg_inference = total_inference_ms / frame_count as f64;
            let avg_cuda_sync = total_cuda_sync_ms / frame_count as f64;
            let avg_postprocess = total_postprocess_ms / frame_count as f64;
            let avg_plan = total_plan_ms / frame_count as f64;
            let avg_render = total_render_ms / frame_count as f64;
            let avg_keying = total_keying_ms / frame_count as f64;
            let avg_total = total_latency_ms / frame_count as f64;
            let avg_g2g = total_glass_to_glass_ms / frame_count as f64;
            let avg_queue_wait = total_queue_wait_ms / frame_count as f64;
            
            // Calculate percentile statistics
            let (pipeline_p50, pipeline_p95, pipeline_p99, pipeline_min, pipeline_max) = get_percentile_stats(&latency_samples);
            let (capture_p50, capture_p95, capture_p99, _, _) = get_percentile_stats(&capture_samples);
            let (inference_p50, inference_p95, inference_p99, _, _) = get_percentile_stats(&inference_samples);
            let (g2g_p50, g2g_p95, g2g_p99, g2g_min, g2g_max) = get_percentile_stats(&e2e_samples);
            
            println!();
            println!("╔══════════════════════════════════════════════════════════════════════╗");
            println!("║  📊 PIPELINE STATISTICS - Frame {} (Samples: {})              ", frame_count, latency_samples.len());
            println!("╚══════════════════════════════════════════════════════════════════════╝");
            println!();
            println!("  🎯 Performance Metrics:");
            println!("    Average FPS:        {:.2} fps", avg_fps);
            println!();
            println!("  📦 Hardware Keying Queue Status:");
            println!("    Frames in queue:    {}/{} frames 🔒 (fixed depth)", buffered_count, max_queue_depth);
            println!("    Queue latency:      {:.2}ms (max = {} frames × {:.2}ms/frame)", 
                     (max_queue_depth as f64) * frame_period_ms, max_queue_depth, frame_period_ms);
            println!("    ℹ️  Queue depth ตรึงไว้ที่ {} frames เพื่อ minimum latency", max_queue_depth);
            println!("        (Hardware DeckLink ต้องการ buffer อย่างน้อย 2 frames เพื่อความลื่นไหล)");
            println!();
            println!("  ⏱️  Latency Breakdown (Average):");
            println!("    ┌─ 1. Capture:         {:.2}ms ({:.1}%)", avg_capture, (avg_capture / avg_total) * 100.0);
            println!("    │    ├─ Wait for frame:  {:.2}ms", avg_capture_wait);
            println!("    │    ├─ DVP DMA ready:   {:.2}ms", avg_capture_dvp);
            println!("    │    └─ HW delta:        {:.2}ms", avg_capture_hw);
            println!("    ├─ 2. Preprocessing:   {:.2}ms ({:.1}%)", avg_preprocess, (avg_preprocess / avg_total) * 100.0);
            println!("    ├─ 3. Inference:       {:.2}ms ({:.1}%)", avg_inference, (avg_inference / avg_total) * 100.0);
            println!("    │    └─ CUDA Sync:     {:.2}ms", avg_cuda_sync);
            println!("    ├─ 4. Postprocessing:  {:.2}ms ({:.1}%)", avg_postprocess, (avg_postprocess / avg_total) * 100.0);
            println!("    ├─ 5. Overlay Plan:    {:.2}ms ({:.1}%)", avg_plan, (avg_plan / avg_total) * 100.0);
            println!("    ├─ 6. GPU Rendering:   {:.2}ms ({:.1}%)", avg_render, (avg_render / avg_total) * 100.0);
            println!("    └─ 7. Hardware Keying: {:.2}ms ({:.1}%)", avg_keying, (avg_keying / avg_total) * 100.0);
            println!("         └─ Queue wait:     {:.2}ms", avg_queue_wait);
            println!("    ───────────────────────────────────────────────────────────");
            println!("    Pipeline Total:        {:.2}ms (100%)", avg_total);
            println!("    Glass-to-Glass (E2E):  {:.2}ms (pipeline + queue)", avg_g2g);
            println!();
            println!("  📊 Percentile Statistics (last {} samples):", latency_samples.len());
            println!("    ┌─ Pipeline Latency:");
            println!("    │    Min:  {:.2}ms  │  P50:  {:.2}ms  │  P95:  {:.2}ms  │  P99:  {:.2}ms  │  Max:  {:.2}ms", 
                     pipeline_min, pipeline_p50, pipeline_p95, pipeline_p99, pipeline_max);
            println!("    ├─ Capture Latency:");
            println!("    │    P50:  {:.2}ms  │  P95:  {:.2}ms  │  P99:  {:.2}ms", 
                     capture_p50, capture_p95, capture_p99);
            println!("    ├─ Inference Time:");
            println!("    │    P50:  {:.2}ms  │  P95:  {:.2}ms  │  P99:  {:.2}ms", 
                     inference_p50, inference_p95, inference_p99);
            println!("    └─ Glass-to-Glass:");
            println!("         Min:  {:.2}ms  │  P50:  {:.2}ms  │  P95:  {:.2}ms  │  P99:  {:.2}ms  │  Max:  {:.2}ms", 
                     g2g_min, g2g_p50, g2g_p95, g2g_p99, g2g_max);
            println!();
        }
    }

    // Final summary
    let total_elapsed = pipeline_start_time.elapsed().as_secs_f64();
    let avg_fps = frame_count as f64 / total_elapsed;

    if config.general.enable_final_summary {
        // Calculate final percentile statistics
        let (pipeline_p50, pipeline_p95, pipeline_p99, pipeline_min, pipeline_max) = get_percentile_stats(&latency_samples);
        let (capture_p50, capture_p95, capture_p99, capture_min, capture_max) = get_percentile_stats(&capture_samples);
        let (inference_p50, inference_p95, inference_p99, inference_min, inference_max) = get_percentile_stats(&inference_samples);
        let (g2g_p50, g2g_p95, g2g_p99, g2g_min, g2g_max) = get_percentile_stats(&e2e_samples);
        
        // Get final buffered count
        let final_buffered = decklink_out.get_buffered_frame_count().unwrap_or(0);
        
        println!();
        println!("╔═══════════════════════════════════════════════════════════════════════╗");
        println!("║  🏁 FINAL SUMMARY - HARDWARE KEYING PIPELINE                         ║");
        println!("╚═══════════════════════════════════════════════════════════════════════╝");
        println!();
        println!("  📈 Performance Overview:");
        println!("    Total frames:         {}", frame_count);
        println!("    Total runtime:        {:.2}s", total_elapsed);
        println!("    Average FPS:          {:.2} fps", avg_fps);
        println!("    Samples collected:    {} frames (for percentiles)", latency_samples.len());
        println!();
        println!("  📦 Hardware Keying Queue Configuration:");
        println!("    Queue depth:          {} frames 🔒 (fixed, not adaptive)", MAX_QUEUE_DEPTH);
        println!("    Final buffered:       {} frames", final_buffered);
        println!("    Max queue latency:    {:.2}ms ({} frames × {:.2}ms/frame @ {:.0} FPS)", 
                 (MAX_QUEUE_DEPTH as f64) * frame_period_ms, MAX_QUEUE_DEPTH, frame_period_ms, fps_calc);
        println!();
        
        // Calculate averages
    let avg_capture = total_capture_ms / frame_count as f64;
    let avg_capture_wait = total_capture_wait_ms / frame_count as f64;
    let avg_capture_dvp = total_capture_dvp_ms / frame_count as f64;
    let avg_capture_hw = total_capture_hw_ms / frame_count as f64;
        let avg_preprocess = total_preprocess_ms / frame_count as f64;
        let avg_inference = total_inference_ms / frame_count as f64;
        let avg_cuda_sync = total_cuda_sync_ms / frame_count as f64;
        let avg_postprocess = total_postprocess_ms / frame_count as f64;
        let avg_plan = total_plan_ms / frame_count as f64;
        let avg_render = total_render_ms / frame_count as f64;
        let avg_keying = total_keying_ms / frame_count as f64;
        let avg_total = total_latency_ms / frame_count as f64;
        let avg_g2g = total_glass_to_glass_ms / frame_count as f64;
        let avg_packet_prep = total_packet_prep_ms / frame_count as f64;
        let avg_queue_mgmt = total_queue_mgmt_ms / frame_count as f64;
    let avg_queue_wait = total_queue_wait_ms / frame_count as f64;
        let avg_timing_calc = total_timing_calc_ms / frame_count as f64;
        
        // Get DVP-specific timings from last frame
        let (_packet_prep_dvp, _queue_mgmt_dvp, dma_copy, api, scheduling_dvp) = decklink_out.get_last_frame_timing();
        
    println!("  ⏱️  Latency Breakdown (Averages):");
    println!("    ┌─ 1. Capture:              {:.2}ms ({:.1}%)", avg_capture, (avg_capture / avg_total) * 100.0);
    println!("    │    ├─ Wait for frame:      {:.2}ms", avg_capture_wait);
    println!("    │    ├─ DVP DMA ready:       {:.2}ms", avg_capture_dvp);
    println!("    │    └─ HW delta:            {:.2}ms", avg_capture_hw);
        println!("    ├─ 2. Preprocessing:        {:.2}ms ({:.1}%)", avg_preprocess, (avg_preprocess / avg_total) * 100.0);
        println!("    ├─ 3. Inference:            {:.2}ms ({:.1}%)", avg_inference, (avg_inference / avg_total) * 100.0);
        println!("    │    └─ CUDA Sync:          {:.2}ms", avg_cuda_sync);
        println!("    ├─ 4. Postprocessing:       {:.2}ms ({:.1}%)", avg_postprocess, (avg_postprocess / avg_total) * 100.0);
        println!("    ├─ 5. Overlay Planning:     {:.2}ms ({:.1}%)", avg_plan, (avg_plan / avg_total) * 100.0);
        println!("    ├─ 6. GPU Rendering:        {:.2}ms ({:.1}%)", avg_render, (avg_render / avg_total) * 100.0);
        println!("    └─ 7. Hardware Keying:      {:.2}ms ({:.1}%)", avg_keying, (avg_keying / avg_total) * 100.0);
        println!("         ├─ Packet prep:        {:.2}ms", avg_packet_prep);
            println!("         ├─ Queue mgmt:         {:.2}ms", avg_queue_mgmt);
            println!("         │   └─ Wait time:      {:.2}ms", avg_queue_wait);
        println!("         ├─ Timing calc:        {:.2}ms", avg_timing_calc);
        println!("         │   ├─ DMA transfer:   {:.2}ms (DVP zero-copy)", dma_copy);
        println!("         │   ├─ DeckLink API:   {:.2}ms", api);
        println!("         │   └─ Scheduling:     {:.2}ms", scheduling_dvp);
        println!("         └─ Other overhead:     {:.2}ms", 
                 avg_keying - avg_packet_prep - avg_queue_mgmt - avg_timing_calc);
        println!("    ───────────────────────────────────────────────────────────────");
        println!("    Pipeline Total:              {:.2}ms (100%)", avg_total);
        println!("    + Queue Latency:             {:.2}ms (buffering)", 
                 avg_g2g - avg_total);
        println!("    = Glass-to-Glass (E2E):      {:.2}ms", avg_g2g);
        println!();
        
        println!("  📊 Latency Distribution (Percentiles):");
        println!();
        println!("    ┌─ Pipeline Latency:");
        println!("    │    Min:   {:.2}ms", pipeline_min);
        println!("    │    P50:   {:.2}ms  (median)", pipeline_p50);
        println!("    │    P95:   {:.2}ms  (95% of frames faster)", pipeline_p95);
        println!("    │    P99:   {:.2}ms  (99% of frames faster)", pipeline_p99);
        println!("    │    Max:   {:.2}ms", pipeline_max);
        println!("    │");
        println!("    ├─ Capture Latency:");
        println!("    │    Min:   {:.2}ms  │  P50:  {:.2}ms  │  P95:  {:.2}ms  │  P99:  {:.2}ms  │  Max:  {:.2}ms", 
                 capture_min, capture_p50, capture_p95, capture_p99, capture_max);
        println!("    │");
        println!("    ├─ Inference Time:");
        println!("    │    Min:   {:.2}ms  │  P50:  {:.2}ms  │  P95:  {:.2}ms  │  P99:  {:.2}ms  │  Max:  {:.2}ms", 
                 inference_min, inference_p50, inference_p95, inference_p99, inference_max);
        println!("    │");
        println!("    └─ Glass-to-Glass (End-to-End):");
        println!("         Min:   {:.2}ms", g2g_min);
        println!("         P50:   {:.2}ms  (median)", g2g_p50);
        println!("         P95:   {:.2}ms  (95% of frames faster)", g2g_p95);
        println!("         P99:   {:.2}ms  (99% of frames faster)", g2g_p99);
        println!("         Max:   {:.2}ms", g2g_max);
        println!();
        
        // Performance assessment
        let jitter = pipeline_p99 - pipeline_p50;
        println!("  🎯 Performance Assessment:");
        println!("    Pipeline jitter (P99-P50):   {:.2}ms", jitter);
        if jitter < 2.0 {
            println!("    Status: ✅ Excellent consistency");
        } else if jitter < 5.0 {
            println!("    Status: ✅ Good consistency");
        } else {
            println!("    Status: ⚠️  High variance detected");
        }
        println!();
        println!("✅ Pipeline completed successfully!");
        println!();
    }

    // CRITICAL: Ensure all queued frames complete before device is dropped
    // This prevents DVP errors when device closes with frames still in flight
    println!("⏳ Waiting for queued frames to complete...");
    match decklink_out.get_buffered_frame_count() {
        Ok(count) => {
            if count > 0 {
                println!("   {} frames still buffered, draining...", count);
            }
        }
        Err(e) => {
            println!("   ⚠️  Could not check buffer: {:?}", e);
        }
    }
    
    // Small delay to allow final frames to complete
    std::thread::sleep(std::time::Duration::from_millis(50));

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

    let _cuda_device = CudaDevice::new(config.preprocessing.cuda_device)?;
    println!("🔧 CUDA: GPU {}", config.preprocessing.cuda_device);

    let crop_region = match config.preprocessing.crop_region.as_str() {
        "Olympus" => CropRegion::Olympus,
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
        let motion = tracker_motion_options_from_config(&config.postprocessing.tracking.motion);
        postprocess::from_path("")?.with_sort_tracking_motion(
            config.postprocessing.tracking.max_age,
            config.postprocessing.tracking.min_confidence,
            config.postprocessing.tracking.iou_threshold,
            motion,
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
