// test_full_pipeline.rs - Test complete pipeline: Capture → Preprocess → Inference → Postprocess
use std::error::Error;
use std::thread;
use std::time::Duration;

use common_io::{MemLoc, RawFramePacket, Stage};
use decklink_input::capture::CaptureSession;
use preprocess_cuda::{ChromaOrder, Preprocessor};
use inference::InferenceEngine;
use postprocess::{PostStage, PostprocessConfig};
use config;
use cudarc::driver::{CudaDevice, DevicePtr};
use telemetry::{now_ns, since_ms, record_ms};

// Latency statistics tracker
struct LatencyStats {
    preprocess_times: Vec<f64>,
    inference_times: Vec<f64>,
    postprocess_times: Vec<f64>,
    e2e_times: Vec<f64>,
}

impl LatencyStats {
    fn new() -> Self {
        Self {
            preprocess_times: Vec::new(),
            inference_times: Vec::new(),
            postprocess_times: Vec::new(),
            e2e_times: Vec::new(),
        }
    }
    
    fn add_preprocess(&mut self, ms: f64) {
        self.preprocess_times.push(ms);
    }
    
    fn add_inference(&mut self, ms: f64) {
        self.inference_times.push(ms);
    }
    
    fn add_postprocess(&mut self, ms: f64) {
        self.postprocess_times.push(ms);
    }
    
    fn add_e2e(&mut self, ms: f64) {
        self.e2e_times.push(ms);
    }
    
    fn print_summary(&self) {
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║         COMPLETE PIPELINE LATENCY SUMMARY                ║");
        println!("╠══════════════════════════════════════════════════════════╣");
        
        if !self.preprocess_times.is_empty() {
            let min = self.preprocess_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = self.preprocess_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let avg = self.preprocess_times.iter().sum::<f64>() / self.preprocess_times.len() as f64;
            
            println!("║ ⚙️  Preprocessing ({} frames):", self.preprocess_times.len());
            println!("║   Min:  {:.3} ms", min);
            println!("║   Max:  {:.3} ms", max);
            println!("║   Avg:  {:.3} ms", avg);
            println!("╠══════════════════════════════════════════════════════════╣");
        }
        
        if !self.inference_times.is_empty() {
            let min = self.inference_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = self.inference_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let avg = self.inference_times.iter().sum::<f64>() / self.inference_times.len() as f64;
            
            println!("║ 🧠 Inference ({} frames):", self.inference_times.len());
            println!("║   Min:  {:.3} ms", min);
            println!("║   Max:  {:.3} ms", max);
            println!("║   Avg:  {:.3} ms", avg);
            println!("╠══════════════════════════════════════════════════════════╣");
        }
        
        if !self.postprocess_times.is_empty() {
            let min = self.postprocess_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = self.postprocess_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let avg = self.postprocess_times.iter().sum::<f64>() / self.postprocess_times.len() as f64;
            
            println!("║ 🎯 Postprocessing ({} frames):", self.postprocess_times.len());
            println!("║   Min:  {:.3} ms", min);
            println!("║   Max:  {:.3} ms", max);
            println!("║   Avg:  {:.3} ms", avg);
            println!("╠══════════════════════════════════════════════════════════╣");
        }
        
        if !self.e2e_times.is_empty() {
            let min = self.e2e_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = self.e2e_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let avg = self.e2e_times.iter().sum::<f64>() / self.e2e_times.len() as f64;
            
            println!("║ 🚀 Full Pipeline ({} frames):", self.e2e_times.len());
            println!("║   Capture → Preprocess → Inference → Postprocess");
            println!("║   Min:  {:.3} ms", min);
            println!("║   Max:  {:.3} ms", max);
            println!("║   Avg:  {:.3} ms", avg);
            
            let fps = 1000.0 / avg;
            println!("║   Throughput: {:.1} FPS", fps);
            
            // Stage contributions
            if !self.preprocess_times.is_empty() && !self.inference_times.is_empty() && !self.postprocess_times.is_empty() {
                let prep_avg = self.preprocess_times.iter().sum::<f64>() / self.preprocess_times.len() as f64;
                let inf_avg = self.inference_times.iter().sum::<f64>() / self.inference_times.len() as f64;
                let post_avg = self.postprocess_times.iter().sum::<f64>() / self.postprocess_times.len() as f64;
                
                let prep_pct = (prep_avg / avg) * 100.0;
                let inf_pct = (inf_avg / avg) * 100.0;
                let post_pct = (post_avg / avg) * 100.0;
                
                println!("║");
                println!("║ 📊 Stage Breakdown:");
                println!("║   Preprocessing:  {:.1}%", prep_pct);
                println!("║   Inference:      {:.1}%", inf_pct);
                println!("║   Postprocessing: {:.1}%", post_pct);
            }
        }
        
        println!("╚══════════════════════════════════════════════════════════╝\n");
    }
}

fn process_frame(
    mut raw_packet: RawFramePacket,
    preprocessor: &mut Preprocessor,
    inference_engine: &mut InferenceEngine,
    post_stage: &mut PostStage,
    latency_stats: &mut LatencyStats,
) -> Result<(), Box<dyn Error>> {
    let t_e2e_start = now_ns();
    
    println!("\n{}", "=".repeat(70));
    println!("🚀 Processing Frame #{}", raw_packet.meta.frame_idx);
    println!("{}", "=".repeat(70));
    
    // Stage 1: Preprocessing
    println!("\n⚙️  Stage 1: Preprocessing...");
    let t_preprocess_start = now_ns();
    let tensor = preprocessor.process(raw_packet);
    let preprocess_ms = since_ms(t_preprocess_start);
    latency_stats.add_preprocess(preprocess_ms);
    println!("  ✓ Completed in {:.3} ms", preprocess_ms);
    
    // Stage 2: Inference
    println!("\n🧠 Stage 2: Inference...");
    let t_inference_start = now_ns();
    let raw_detections = inference_engine.process(tensor);
    let inference_ms = since_ms(t_inference_start);
    latency_stats.add_inference(inference_ms);
    println!("  ✓ Completed in {:.3} ms", inference_ms);
    println!("  ✓ Raw output: {} values, shape: {:?}", 
             raw_detections.raw_output.len(), raw_detections.output_shape);
    
    // Stage 3: Postprocessing
    println!("\n🎯 Stage 3: Postprocessing...");
    let t_postprocess_start = now_ns();
    let detections = post_stage.process(raw_detections);
    let postprocess_ms = since_ms(t_postprocess_start);
    latency_stats.add_postprocess(postprocess_ms);
    
    // Display detections
    println!("\n📊 Final Detections: {} objects", detections.items.len());
    if !detections.items.is_empty() {
        println!("\n  Details:");
        for (i, det) in detections.items.iter().take(10).enumerate() {
            println!("  ┌─ Object #{}", i + 1);
            println!("  │  BBox:     ({:.1}, {:.1}, {:.1}, {:.1})", 
                     det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h);
            println!("  │  Score:    {:.3}", det.score);
            println!("  │  Class:    {}", det.class_id);
            if let Some(track_id) = det.track_id {
                println!("  │  Track ID: {}", track_id);
            }
            println!("  └─");
        }
        if detections.items.len() > 10 {
            println!("  ... and {} more", detections.items.len() - 10);
        }
    }
    
    // Calculate E2E latency
    let e2e_ms = since_ms(t_e2e_start);
    latency_stats.add_e2e(e2e_ms);
    
    println!("\n✅ Full Pipeline Summary:");
    println!("  Preprocessing:   {:.3} ms ({:.1}%)", 
             preprocess_ms, (preprocess_ms / e2e_ms) * 100.0);
    println!("  Inference:       {:.3} ms ({:.1}%)", 
             inference_ms, (inference_ms / e2e_ms) * 100.0);
    println!("  Postprocessing:  {:.3} ms ({:.1}%)", 
             postprocess_ms, (postprocess_ms / e2e_ms) * 100.0);
    println!("  ─────────────────────────────────────");
    println!("  Total E2E:       {:.3} ms", e2e_ms);
    println!("  Target FPS:      {:.1} fps", 1000.0 / e2e_ms);
    
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║          COMPLETE PIPELINE TEST - ALL STAGES            ║");
    println!("║  Capture → Preprocess → Inference → Postprocess         ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
    
    let mut latency_stats = LatencyStats::new();
    
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
    let cfg = config::InferenceCfg {
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
    
    let mut inference_engine = InferenceEngine::new(&cfg)?;
    println!();
    
    // Create postprocessing stage
    println!("🎯 Creating Postprocessing Stage...");
    let post_config = PostprocessConfig {
        num_classes: 80,
        confidence_threshold: 0.25,
        nms_threshold: 0.45,
        max_detections: 100,
        letterbox_scale: 1.0,
        letterbox_pad: (0.0, 0.0),
        original_size: (512, 512),
    };
    
    let mut post_stage = PostStage::new(post_config)
        .with_temporal_smoothing(4)
        .with_sort_tracking(5, 0.25, 0.3);
    
    println!("  ✓ Temporal smoothing enabled (window=4)");
    println!("  ✓ SORT tracking enabled");
    println!();
    
    // Process frames
    const TARGET_FRAMES: usize = 5;
    let mut frames_processed = 0;
    
    println!("🎥 Waiting for frames from DeckLink...\n");
    
    for attempt in 0..200 {
        match session.get_frame()? {
            Some(raw_packet) => {
                frames_processed += 1;
                
                process_frame(
                    raw_packet,
                    &mut preprocessor,
                    &mut inference_engine,
                    &mut post_stage,
                    &mut latency_stats,
                )?;
                
                if frames_processed >= TARGET_FRAMES {
                    println!("\n╔══════════════════════════════════════════════════════════╗");
                    println!("║    Successfully Processed {} Frames! ✓                  ║", TARGET_FRAMES);
                    println!("╚══════════════════════════════════════════════════════════╝");
                    
                    latency_stats.print_summary();
                    return Ok(());
                }
                
                println!("\n{}", "─".repeat(70));
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
    
    if frames_processed > 0 {
        latency_stats.print_summary();
    }
    
    Ok(())
}
