// test_detectionspacket.rs - Test DetectionsPacket with real inference pipeline
// Similar to test_tensorinputpacket and test_inference_pipeline
// Capture → Preprocess → Inference → Postprocess → DetectionsPacket validation

use std::error::Error;
use std::thread;
use std::time::Duration;

use common_io::Stage;
use decklink_input::capture::CaptureSession;
use preprocess_cuda::{ChromaOrder, Preprocessor};
use inference::InferenceEngine;
use postprocess::Postprocess;
use config;
use telemetry::{now_ns, since_ms};

struct LatencyStats {
    capture_times: Vec<f64>,
    preprocess_times: Vec<f64>,
    inference_times: Vec<f64>,
    postprocess_times: Vec<f64>,
    e2e_times: Vec<f64>,
    detection_counts: Vec<usize>,
}

impl LatencyStats {
    fn new() -> Self {
        Self {
            capture_times: Vec::new(),
            preprocess_times: Vec::new(),
            inference_times: Vec::new(),
            postprocess_times: Vec::new(),
            e2e_times: Vec::new(),
            detection_counts: Vec::new(),
        }
    }
    
    fn add(&mut self, capture_ms: f64, prep_ms: f64, inf_ms: f64, post_ms: f64, e2e_ms: f64, det_count: usize) {
        self.capture_times.push(capture_ms);
        self.preprocess_times.push(prep_ms);
        self.inference_times.push(inf_ms);
        self.postprocess_times.push(post_ms);
        self.e2e_times.push(e2e_ms);
        self.detection_counts.push(det_count);
    }
    
    fn calc_stats(values: &[f64]) -> (f64, f64, f64) {
        if values.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let avg = values.iter().sum::<f64>() / values.len() as f64;
        (min, max, avg)
    }
    
    fn print_summary(&self) {
        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║           DETECTIONSPACKET LATENCY ANALYSIS                      ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");
        
        let n = self.e2e_times.len();
        println!("║ Frames Processed: {}                                              ", n);
        println!("╠══════════════════════════════════════════════════════════════════╣");
        
        if !self.capture_times.is_empty() {
            let (min, max, avg) = Self::calc_stats(&self.capture_times);
            println!("║ 📷 Capture Overhead:");
            println!("║    Min: {:.3} ms  |  Max: {:.3} ms  |  Avg: {:.3} ms", min, max, avg);
        }
        
        if !self.preprocess_times.is_empty() {
            let (min, max, avg) = Self::calc_stats(&self.preprocess_times);
            println!("║ ⚙️  Preprocessing:");
            println!("║    Min: {:.3} ms  |  Max: {:.3} ms  |  Avg: {:.3} ms", min, max, avg);
        }
        
        if !self.inference_times.is_empty() {
            let (min, max, avg) = Self::calc_stats(&self.inference_times);
            println!("║ 🧠 Inference:");
            println!("║    Min: {:.3} ms  |  Max: {:.3} ms  |  Avg: {:.3} ms", min, max, avg);
        }
        
        if !self.postprocess_times.is_empty() {
            let (min, max, avg) = Self::calc_stats(&self.postprocess_times);
            println!("║ 🎯 Postprocessing:");
            println!("║    Min: {:.3} ms  |  Max: {:.3} ms  |  Avg: {:.3} ms", min, max, avg);
        }
        
        println!("╠══════════════════════════════════════════════════════════════════╣");
        
        if !self.e2e_times.is_empty() {
            let (min, max, avg) = Self::calc_stats(&self.e2e_times);
            println!("║ 🚀 End-to-End Pipeline:");
            println!("║    Min: {:.3} ms  |  Max: {:.3} ms  |  Avg: {:.3} ms", min, max, avg);
            
            let fps = 1000.0 / avg;
            println!("║    Throughput: {:.2} FPS", fps);
            
            // Stage contribution percentages
            let prep_avg = Self::calc_stats(&self.preprocess_times).2;
            let inf_avg = Self::calc_stats(&self.inference_times).2;
            let post_avg = Self::calc_stats(&self.postprocess_times).2;
            
            let total = prep_avg + inf_avg + post_avg;
            if total > 0.0 {
                println!("║");
                println!("║ 📊 Stage Contributions:");
                println!("║    Preprocessing:  {:>6.2}%  ({:.3} ms)", (prep_avg/avg)*100.0, prep_avg);
                println!("║    Inference:      {:>6.2}%  ({:.3} ms)", (inf_avg/avg)*100.0, inf_avg);
                println!("║    Postprocessing: {:>6.2}%  ({:.3} ms)", (post_avg/avg)*100.0, post_avg);
            }
        }
        
        println!("╠══════════════════════════════════════════════════════════════════╣");
        
        if !self.detection_counts.is_empty() {
            let min_det = *self.detection_counts.iter().min().unwrap();
            let max_det = *self.detection_counts.iter().max().unwrap();
            let avg_det = self.detection_counts.iter().sum::<usize>() as f64 / self.detection_counts.len() as f64;
            
            println!("║ 📦 Detections per Frame:");
            println!("║    Min: {}  |  Max: {}  |  Avg: {:.1}", min_det, max_det, avg_det);
        }
        
        println!("╚══════════════════════════════════════════════════════════════════╝\n");
    }
}

fn print_detections_packet(packet: &common_io::DetectionsPacket, frame_num: usize) {
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ 📦 DETECTIONSPACKET #{}                                          ", frame_num);
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│ Frame Metadata:");
    println!("│   Frame Index:  {}", packet.from.frame_idx);
    println!("│   Timestamp:    {} ns", packet.from.pts_ns);
    println!("│   Resolution:   {}x{}", packet.from.width, packet.from.height);
    println!("│   Format:       {:?}", packet.from.pixfmt);
    println!("│   Colorspace:   {:?}", packet.from.colorspace);
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│ Detections: {} objects", packet.items.len());
    println!("└─────────────────────────────────────────────────────────────────┘");
    
    if !packet.items.is_empty() {
        println!("\n  Detailed Detections (showing first 10):");
        println!("  ┌────┬─────────┬────────────────────────────┬───────┬──────────┐");
        println!("  │ #  │ Class   │ BBox (x, y, w, h)          │ Score │ Track ID │");
        println!("  ├────┼─────────┼────────────────────────────┼───────┼──────────┤");
        
        for (i, det) in packet.items.iter().take(10).enumerate() {
            let track_str = match det.track_id {
                Some(id) => format!("{:>8}", id),
                None => "    None".to_string(),
            };
            
            println!("  │ {:>2} │ {:>7} │ ({:>6.1}, {:>6.1}, {:>6.1}, {:>6.1}) │ {:.3} │ {} │",
                     i + 1,
                     det.class_id,
                     det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h,
                     det.score,
                     track_str);
        }
        
        println!("  └────┴─────────┴────────────────────────────┴───────┴──────────┘");
        
        if packet.items.len() > 10 {
            println!("  ... and {} more detections", packet.items.len() - 10);
        }
        
        // Class distribution
        let mut class_counts = std::collections::HashMap::new();
        for det in &packet.items {
            *class_counts.entry(det.class_id).or_insert(0) += 1;
        }
        
        if class_counts.len() > 1 {
            println!("\n  Class Distribution:");
            let mut class_vec: Vec<_> = class_counts.iter().collect();
            class_vec.sort_by_key(|&(_, count)| std::cmp::Reverse(*count));
            
            for (class_id, count) in class_vec.iter().take(5) {
                let pct = (**count as f64 / packet.items.len() as f64) * 100.0;
                println!("    Class {:>3}: {:>3} objects ({:>5.1}%)", class_id, count, pct);
            }
            
            if class_vec.len() > 5 {
                println!("    ... and {} more classes", class_vec.len() - 5);
            }
        }
        
        // Score statistics
        let scores: Vec<f32> = packet.items.iter().map(|d| d.score).collect();
        let min_score = scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let avg_score = scores.iter().sum::<f32>() / scores.len() as f32;
        
        println!("\n  Score Statistics:");
        println!("    Min: {:.3}  |  Max: {:.3}  |  Avg: {:.3}", min_score, max_score, avg_score);
        
        // Tracking statistics (if enabled)
        let tracked_count = packet.items.iter().filter(|d| d.track_id.is_some()).count();
        if tracked_count > 0 {
            println!("\n  Tracking Statistics:");
            println!("    Tracked objects: {}/{} ({:.1}%)", 
                     tracked_count, 
                     packet.items.len(),
                     (tracked_count as f64 / packet.items.len() as f64) * 100.0);
            
            let unique_tracks: std::collections::HashSet<_> = packet.items.iter()
                .filter_map(|d| d.track_id)
                .collect();
            println!("    Unique track IDs: {}", unique_tracks.len());
        }
    } else {
        println!("  (No detections in this frame)");
    }
    
    println!();
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║              DETECTIONSPACKET VALIDATION TEST                    ║");
    println!("║  Full Pipeline: Capture → Preprocess → Inference → Postprocess  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
    
    let mut latency_stats = LatencyStats::new();
    
    // List DeckLink devices
    let devices = decklink_input::devicelist();
    println!("🎥 Available DeckLink devices: {}", devices.len());
    for (idx, name) in devices.iter().enumerate() {
        println!("   [{}] {}", idx, name);
    }
    
    if devices.is_empty() {
        return Err("No DeckLink devices found".into());
    }
    println!();
    
    // Open capture session
    println!("📷 Opening DeckLink capture session...");
    let mut session = CaptureSession::open(0)?;
    println!("   ✓ Capture session ready\n");
    
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
    println!("   ✓ Preprocessor ready\n");
    
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
    
    // Create postprocessing stage
    println!("🎯 Creating Postprocessing Stage...");
    let post_cfg = config::PostprocessCfg {
        model_type: "yolo".to_string(),
        score_thresh: 0.25,
        max_dets: 300,
        nms: config::NmsCfg {
            nms_type: "classwise".to_string(),
            iou_thresh: 0.45,
            max_per_class: 100,
            max_total: 300,
        },
        tracking: config::TrackingCfg {
            enable: true,  // Enable tracking for this test
            iou_match: 0.5,
            max_age: 30,
            min_confidence: 0.25,
        },
        letterbox: config::LetterboxCfg {
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
            original_w: 512,
            original_h: 512,
        },
    };
    
    let mut post_stage = Postprocess::new(post_cfg)?
        .with_temporal_smoothing(4)
        .with_tracking();
    
    println!("   ✓ Postprocessing ready (with SORT tracking)\n");
    
    // Process frames
    const TARGET_FRAMES: usize = 10;
    let mut frames_processed = 0;
    
    println!("🎬 Starting frame capture and processing...");
    println!("{}", "=".repeat(68));
    
    for attempt in 0..200 {
        // Capture frame
        let t_capture = now_ns();
        match session.get_frame()? {
            Some(raw_packet) => {
                let capture_ms = since_ms(t_capture);
                
                frames_processed += 1;
                let frame_idx = raw_packet.meta.frame_idx;
                
                println!("\n🎬 Processing Frame #{} (attempt {})...", frame_idx, attempt);
                
                let t_e2e = now_ns();
                
                // Stage 1: Preprocessing
                let t_prep = now_ns();
                let tensor = preprocessor.process(raw_packet);
                let prep_ms = since_ms(t_prep);
                
                // Stage 2: Inference
                let t_inf = now_ns();
                let raw_detections = inference_engine.process(tensor);
                let inf_ms = since_ms(t_inf);
                
                // Stage 3: Postprocessing
                let t_post = now_ns();
                let detections_packet = post_stage.process(raw_detections);
                let post_ms = since_ms(t_post);
                
                let e2e_ms = since_ms(t_e2e);
                
                // Print DetectionsPacket details
                print_detections_packet(&detections_packet, frames_processed);
                
                // Print latency breakdown
                println!("⏱️  Latency Breakdown:");
                println!("   Capture:      {:>7.3} ms ({:>5.1}%)", capture_ms, (capture_ms/e2e_ms)*100.0);
                println!("   Preprocess:   {:>7.3} ms ({:>5.1}%)", prep_ms, (prep_ms/e2e_ms)*100.0);
                println!("   Inference:    {:>7.3} ms ({:>5.1}%)", inf_ms, (inf_ms/e2e_ms)*100.0);
                println!("   Postprocess:  {:>7.3} ms ({:>5.1}%)", post_ms, (post_ms/e2e_ms)*100.0);
                println!("   ───────────────────────────────");
                println!("   Total E2E:    {:>7.3} ms", e2e_ms);
                println!("   Throughput:   {:>7.2} fps", 1000.0 / e2e_ms);
                
                // Add to statistics
                latency_stats.add(
                    capture_ms, 
                    prep_ms, 
                    inf_ms, 
                    post_ms, 
                    e2e_ms, 
                    detections_packet.items.len()
                );
                
                println!("{}", "─".repeat(68));
                
                if frames_processed >= TARGET_FRAMES {
                    println!("\n{}", "=".repeat(68));
                    println!("✅ Completed processing {} frames!", TARGET_FRAMES);
                    println!("{}", "=".repeat(68));
                    
                    // Print summary statistics
                    latency_stats.print_summary();
                    
                    return Ok(());
                }
            }
            None => {
                if attempt % 20 == 0 && attempt > 0 {
                    println!("⏳ Waiting for frame... (attempt {})", attempt);
                }
            }
        }
        
        thread::sleep(Duration::from_millis(50));
    }
    
    println!("\n⚠️  Timeout: Only processed {} frames", frames_processed);
    
    if frames_processed > 0 {
        latency_stats.print_summary();
    }
    
    Ok(())
}
