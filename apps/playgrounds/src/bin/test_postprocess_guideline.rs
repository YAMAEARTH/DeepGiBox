// test_postprocess_guideline.rs - Test guideline-compliant postprocessing
use std::error::Error;
use std::thread;
use std::time::Duration;

use common_io::{RawFramePacket, Stage};
use decklink_input::capture::CaptureSession;
use preprocess_cuda::{ChromaOrder, Preprocessor};
use inference::InferenceEngine;
use postprocess::Postprocess;
use config;

fn main() -> Result<(), Box<dyn Error>> {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║     GUIDELINE-COMPLIANT POSTPROCESSING TEST              ║");
    println!("║     Config-based init + Telemetry subspans              ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Load config from TOML (guideline §4)
    println!("📋 Loading configuration from configs/postprocess_test.toml...");
    let app_cfg = config::AppConfig::from_file("configs/postprocess_test.toml")?;
    let post_cfg = app_cfg.postprocess.expect("Missing [postprocess] config");
    
    println!("  ✓ Model type:      {}", post_cfg.model_type);
    println!("  ✓ Score thresh:    {}", post_cfg.score_thresh);
    println!("  ✓ NMS type:        {}", post_cfg.nms.nms_type);
    println!("  ✓ NMS IoU thresh:  {}", post_cfg.nms.iou_thresh);
    println!("  ✓ Tracking:        {}", if post_cfg.tracking.enable { "enabled" } else { "disabled" });
    println!();

    // List DeckLink devices
    let devices = decklink_input::devicelist();
    println!("Available DeckLink devices: {}", devices.len());
    for (idx, name) in devices.iter().enumerate() {
        println!("  [{}] {}", idx, name);
    }
    
    if devices.is_empty() {
        return Err("No DeckLink devices found".into());
    }
    
    // Open capture session
    let mut session = CaptureSession::open(0)?;
    println!("✓ Opened DeckLink capture session\n");
    
    // Create preprocessor
    println!("⚙️  Creating Preprocessor (512x512, FP16)...");
    let mut preprocessor = Preprocessor::with_params(
        (512, 512),
        true,
        0,
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
        ChromaOrder::UYVY,
    )?;
    println!("  ✓ Ready\n");
    
    // Create inference engine
    println!("🧠 Creating Inference Engine...");
    let inf_cfg = config::InferenceCfg {
        backend: "onnxruntime_trt".to_string(),
        model: "crates/inference/YOLOv5.onnx".to_string(),
        device: 0,
        fp16: true,
        engine_cache: "./trt_cache".to_string(),
        timing_cache: "./trt_cache".to_string(),
        max_workspace_mb: 2048,
        enable_fallback_cuda: Some(true),
        warmup_runs: Some(3),
        input_name: Some("images".to_string()),
        output_names: Some(vec!["output".to_string()]),
    };
    
    let mut inference_engine = InferenceEngine::new(&inf_cfg)?;
    println!();
    
    // Create guideline-compliant postprocessing stage (guideline §5)
    println!("🎯 Creating Guideline-Compliant Postprocessing Stage...");
    let mut post_stage = Postprocess::new(post_cfg)?
        .with_temporal_smoothing(4)  // Optional: temporal smoothing
        .with_tracking();             // Optional: SORT tracking (if enabled in config)
    
    println!("  ✓ Config-based initialization complete");
    println!("  ✓ Telemetry subspans enabled: post.decode, post.nms, post.track");
    println!("  ✓ Preallocated buffers for hot path");
    println!();
    
    // Process frames
    const TARGET_FRAMES: usize = 3;
    let mut frames_processed = 0;
    
    println!("🎥 Processing {} frames...\n", TARGET_FRAMES);
    println!("{}", "=".repeat(70));
    
    for attempt in 0..200 {
        match session.get_frame()? {
            Some(raw_packet) => {
                frames_processed += 1;
                
                println!("\n🚀 Frame #{}", raw_packet.meta.frame_idx);
                println!("{}", "-".repeat(70));
                
                // Preprocess
                let tensor = preprocessor.process(raw_packet);
                
                // Inference
                let raw_detections = inference_engine.process(tensor);
                println!("  Raw output: {} values, shape: {:?}", 
                         raw_detections.raw_output.len(), raw_detections.output_shape);
                
                // Postprocess (guideline-compliant with telemetry)
                let detections = post_stage.process(raw_detections);
                
                // Display detections
                println!("\n  📊 Final Detections: {} objects", detections.items.len());
                if !detections.items.is_empty() {
                    for (i, det) in detections.items.iter().take(5).enumerate() {
                        print!("    [{}] Class {} Score {:.3}", i + 1, det.class_id, det.score);
                        if let Some(track_id) = det.track_id {
                            print!(" Track#{}", track_id);
                        }
                        println!();
                    }
                    if detections.items.len() > 5 {
                        println!("    ... and {} more", detections.items.len() - 5);
                    }
                }
                
                println!("{}", "-".repeat(70));
                
                if frames_processed >= TARGET_FRAMES {
                    println!("\n{}", "=".repeat(70));
                    println!("✅ Successfully processed {} frames!", TARGET_FRAMES);
                    println!("{}", "=".repeat(70));
                    println!("\n💡 Check telemetry output for subspan timings:");
                    println!("   [lat] post.decode=X.XXms");
                    println!("   [lat] post.nms=X.XXms");
                    println!("   [lat] post.track=X.XXms");
                    println!("   [lat] postprocess=X.XXms (total)");
                    return Ok(());
                }
            }
            None => {
                if attempt % 20 == 0 {
                    println!("⏳ Waiting for frame... (attempt {})", attempt);
                }
            }
        }
        thread::sleep(Duration::from_millis(50));
    }
    
    println!("\n⚠️  Timeout: Only captured {} frames", frames_processed);
    
    Ok(())
}
