// test_postprocess_guideline.rs - Test guideline-compliant postprocessing
use std::error::Error;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use common_io::{Stage, DetectionsPacket, RawFramePacket};
use decklink_input::capture::CaptureSession;
use preprocess_cuda::{ChromaOrder, Preprocessor};
use inference_v2::TrtInferenceStage;
use postprocess::PostStage;
use config;

// Telemetry helpers with fallback
#[cfg(feature = "telemetry")]
use telemetry::{now_ns, since_ms, record_ms};

#[cfg(not(feature = "telemetry"))]
fn now_ns() -> Instant {
    Instant::now()
}

#[cfg(not(feature = "telemetry"))]
fn since_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[cfg(not(feature = "telemetry"))]
fn record_ms(_name: &str, _start: Instant) {
    // No-op when telemetry is disabled
}

fn print_rawframepacket(packet: &RawFramePacket) {
    println!("📦 RawFramePacket Details:");
    println!("  Source ID:     {}", packet.meta.source_id);
    println!("  Dimensions:    {}x{}", packet.meta.width, packet.meta.height);
    println!("  Pixel Format:  {:?}", packet.meta.pixfmt);
    println!("  Color Space:   {:?}", packet.meta.colorspace);
    println!("  Frame Index:   {}", packet.meta.frame_idx);
    println!("  PTS:           {} ns", packet.meta.pts_ns);
    println!("  Capture Time:  {} ns", packet.meta.t_capture_ns);
    println!("  Stride:        {} bytes", packet.meta.stride_bytes);
    println!("  Data Length:   {} bytes ({:.2} MB)", 
             packet.data.len, 
             packet.data.len as f64 / (1024.0 * 1024.0));
    println!("  Memory Loc:    {:?}", packet.data.loc);
}

fn display_detections_packet(detections: &DetectionsPacket) {
    println!("\n🎯 DetectionsPacket Details:");
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║              DETECTIONS PACKET - COMPLETE INFO                ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    
    // Source metadata (preserved from original frame)
    println!("║ 📸 Source Frame Metadata (Preserved):");
    println!("║   Frame Index:    {:<42} ║", detections.from.frame_idx);
    println!("║   Source ID:      {:<42} ║", detections.from.source_id);
    println!("║   Original Size:  {}x{}{:>36} ║", 
             detections.from.width, detections.from.height, "");
    println!("║   Pixel Format:   {:?}{:>37} ║", detections.from.pixfmt, "");
    println!("║   Color Space:    {:?}{:>35} ║", detections.from.colorspace, "");
    
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ ⏱️  Timing Information:");
    println!("║   PTS:            {} ns{:>27} ║", detections.from.pts_ns, "");
    println!("║   Capture Time:   {} ns{:>27} ║", detections.from.t_capture_ns, "");
    
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ 🎯 Detection Results:");
    println!("║   Total Detections: {:<42} ║", detections.items.len());
    
    if detections.items.is_empty() {
        println!("║   Status:         ⚠️  No detections found{:>26} ║", "");
    } else {
        // Calculate statistics
        let avg_score = detections.items.iter().map(|d| d.score).sum::<f32>() / detections.items.len() as f32;
        let min_score = detections.items.iter().map(|d| d.score).fold(f32::INFINITY, |a, b| a.min(b));
        let max_score = detections.items.iter().map(|d| d.score).fold(f32::NEG_INFINITY, |a, b| a.max(b));
        
        println!("║   Score Range:    {:.3} - {:.3}{:>29} ║", min_score, max_score, "");
        println!("║   Average Score:  {:.3}{:>39} ║", avg_score, "");
        
        // Count by class
        let mut class_counts = std::collections::HashMap::new();
        for det in &detections.items {
            *class_counts.entry(det.class_id).or_insert(0) += 1;
        }
        
        println!("║   Unique Classes: {:<42} ║", class_counts.len());
        
        println!("╠═══════════════════════════════════════════════════════════════╣");
        println!("║ 📊 Detection Details (Top 10):");
        
        for (i, det) in detections.items.iter().take(10).enumerate() {
            let bbox_str = format!("[x:{:.0},y:{:.0},w:{:.0},h:{:.0}]", 
                                   det.bbox.x, det.bbox.y, 
                                   det.bbox.w, det.bbox.h);
            print!("║   [{:2}] Class {:2} Score {:.3} BBox {}", 
                   i + 1, det.class_id, det.score, bbox_str);
            
            if let Some(track_id) = det.track_id {
                print!(" Track#{}", track_id);
            }
            
            let padding = 60_usize.saturating_sub(
                format!("[{:2}] Class {:2} Score {:.3} BBox {} Track#{}", 
                        i + 1, det.class_id, det.score, bbox_str, 
                        det.track_id.unwrap_or(0)).len()
            );
            println!("{:>width$} ║", "", width = padding);
        }
        
        if detections.items.len() > 10 {
            println!("║   ... and {} more detections{:>32} ║", 
                     detections.items.len() - 10, "");
        }
    }
    
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ 📝 Postprocessing Output:");
    println!("║   Type:           Filtered & NMS detections{:>20} ║", "");
    println!("║   Format:         Class ID, Score, BBox, Track ID{:>13} ║", "");
    println!("║   Next Stage:     Overlay rendering{:>28} ║", "");
    
    println!("╚═══════════════════════════════════════════════════════════════╝");
}

// Latency statistics tracker
struct LatencyStats {
    transfer_times: Vec<f64>,
    preprocess_times: Vec<f64>,
    inference_times: Vec<f64>,
    postprocess_times: Vec<f64>,
    e2e_times: Vec<f64>,  // End-to-end (capture to postprocess done)
    gpu_direct_preprocess: Vec<f64>,
}

impl LatencyStats {
    fn new() -> Self {
        Self {
            transfer_times: Vec::new(),
            preprocess_times: Vec::new(),
            inference_times: Vec::new(),
            postprocess_times: Vec::new(),
            e2e_times: Vec::new(),
            gpu_direct_preprocess: Vec::new(),
        }
    }
    
    fn add_transfer(&mut self, ms: f64) {
        self.transfer_times.push(ms);
    }
    
    fn add_preprocess(&mut self, ms: f64, is_gpu_direct: bool) {
        self.preprocess_times.push(ms);
        if is_gpu_direct {
            self.gpu_direct_preprocess.push(ms);
        }
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
        
        // CPU→GPU Transfer
        if !self.transfer_times.is_empty() {
            let min = self.transfer_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = self.transfer_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let avg = self.transfer_times.iter().sum::<f64>() / self.transfer_times.len() as f64;
            
            println!("║ 📤 CPU→GPU Transfer ({} frames):", self.transfer_times.len());
            println!("║   Min:  {:.3} ms", min);
            println!("║   Max:  {:.3} ms", max);
            println!("║   Avg:  {:.3} ms", avg);
            println!("╠══════════════════════════════════════════════════════════╣");
        }
        
        // Preprocessing
        if !self.preprocess_times.is_empty() {
            let min = self.preprocess_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = self.preprocess_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let avg = self.preprocess_times.iter().sum::<f64>() / self.preprocess_times.len() as f64;
            
            println!("║ ⚙️  Preprocessing - All Frames ({}):", self.preprocess_times.len());
            println!("║   Min:  {:.3} ms", min);
            println!("║   Max:  {:.3} ms", max);
            println!("║   Avg:  {:.3} ms", avg);
        }
        
        // GPU Direct Preprocessing
        if !self.gpu_direct_preprocess.is_empty() {
            println!("╠══════════════════════════════════════════════════════════╣");
            let min = self.gpu_direct_preprocess.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = self.gpu_direct_preprocess.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let avg = self.gpu_direct_preprocess.iter().sum::<f64>() / self.gpu_direct_preprocess.len() as f64;
            
            println!("║ 🚀 GPU Direct Preprocessing ({} frames):", self.gpu_direct_preprocess.len());
            println!("║   Min:  {:.3} ms", min);
            println!("║   Max:  {:.3} ms", max);
            println!("║   Avg:  {:.3} ms ⚡ (Zero-copy!)", avg);
        }
        
        // Inference
        if !self.inference_times.is_empty() {
            println!("╠══════════════════════════════════════════════════════════╣");
            let min = self.inference_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = self.inference_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let avg = self.inference_times.iter().sum::<f64>() / self.inference_times.len() as f64;
            
            println!("║ 🧠 Inference ({} frames):", self.inference_times.len());
            println!("║   Min:  {:.3} ms", min);
            println!("║   Max:  {:.3} ms", max);
            println!("║   Avg:  {:.3} ms", avg);
            
            let fps = 1000.0 / avg;
            println!("║   Throughput: {:.1} FPS (inference only)", fps);
        }
        
        // Postprocessing
        if !self.postprocess_times.is_empty() {
            println!("╠══════════════════════════════════════════════════════════╣");
            let min = self.postprocess_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = self.postprocess_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let avg = self.postprocess_times.iter().sum::<f64>() / self.postprocess_times.len() as f64;
            
            println!("║ 🎯 Postprocessing ({} frames):", self.postprocess_times.len());
            println!("║   Min:  {:.3} ms", min);
            println!("║   Max:  {:.3} ms", max);
            println!("║   Avg:  {:.3} ms", avg);
        }
        
        // End-to-End
        if !self.e2e_times.is_empty() {
            println!("╠══════════════════════════════════════════════════════════╣");
            let min = self.e2e_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = self.e2e_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let avg = self.e2e_times.iter().sum::<f64>() / self.e2e_times.len() as f64;
            
            println!("║ 🎯 End-to-End Pipeline ({} frames):", self.e2e_times.len());
            println!("║   Capture → Preprocess → Inference → Postprocess");
            println!("║   Min:  {:.3} ms", min);
            println!("║   Max:  {:.3} ms", max);
            println!("║   Avg:  {:.3} ms", avg);
            
            let fps = 1000.0 / avg;
            println!("║   Target FPS: {:.1} fps", fps);
            
            // Calculate individual stage contributions
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
    inference_stage: &mut TrtInferenceStage,
    post_stage: &mut PostStage,
    latency_stats: &mut LatencyStats,
) -> Result<(), Box<dyn Error>> {
    // Start E2E timing
    let t_e2e_start = now_ns();
    
    println!("\n{}", "=".repeat(60));
    println!("🚀 Processing Frame #{}", raw_packet.meta.frame_idx);
    println!("{}", "=".repeat(60));
    
    // Print input frame info
    print_rawframepacket(&raw_packet);
    
    // Check if we need to transfer CPU -> GPU
    let is_gpu_direct = matches!(raw_packet.data.loc, common_io::MemLoc::Gpu { .. });
    let needs_gpu_transfer = !is_gpu_direct;
    
    if needs_gpu_transfer {
        println!("\n📤 Transferring frame from CPU to GPU...");
        
        let t_transfer_start = now_ns();
        
        use cudarc::driver::{CudaDevice, DevicePtr};
        let device = CudaDevice::new(0)?;
        
        // Allocate GPU buffer
        let mut gpu_buffer = device.alloc_zeros::<u8>(raw_packet.data.len)?;
        
        // Copy CPU -> GPU
        let cpu_slice = unsafe {
            std::slice::from_raw_parts(raw_packet.data.ptr, raw_packet.data.len)
        };
        device.htod_sync_copy_into(cpu_slice, &mut gpu_buffer)?;
        
        // Update packet to point to GPU memory
        let gpu_ptr = *gpu_buffer.device_ptr() as *mut u8;
        raw_packet.data.ptr = gpu_ptr;
        raw_packet.data.loc = common_io::MemLoc::Gpu { device: 0 };
        
        // Leak the buffer so it's not freed
        std::mem::forget(gpu_buffer);
        
        let transfer_ms = since_ms(t_transfer_start);
        record_ms("cpu_to_gpu_transfer", t_transfer_start);
        latency_stats.add_transfer(transfer_ms);
        println!("  ✓ Transferred {} bytes to GPU in {:.3} ms", raw_packet.data.len, transfer_ms);
    } else {
        println!("\n🚀 GPU Direct frame detected - Zero-copy processing!");
    }
    
    // Stage 1: Preprocessing
    println!("\n⚙️  Stage 1: Preprocessing...");
    let t_preprocess_start = now_ns();
    let tensor = preprocessor.process(raw_packet);
    let preprocess_ms = since_ms(t_preprocess_start);
    record_ms("preprocessing", t_preprocess_start);
    latency_stats.add_preprocess(preprocess_ms, is_gpu_direct);
    
    println!("  ✓ Preprocessing completed in {:.3} ms", preprocess_ms);
    if is_gpu_direct {
        println!("  ✓ GPU Direct latency: {:.3} ms ⚡", preprocess_ms);
    }
    
    // Stage 2: Inference
    println!("\n🧠 Stage 2: Inference...");
    let t_inference_start = now_ns();
    let raw_detections = inference_stage.process(tensor);
    let inference_ms = since_ms(t_inference_start);
    record_ms("inference", t_inference_start);
    latency_stats.add_inference(inference_ms);
    
    println!("  ✓ Inference completed in {:.3} ms", inference_ms);
    println!("  Raw output: {} values, shape: {:?}", 
             raw_detections.raw_output.len(), raw_detections.output_shape);
    
    // Stage 3: Postprocessing
    println!("\n🎯 Stage 3: Postprocessing...");
    let t_postprocess_start = now_ns();
    let detections = post_stage.process(raw_detections);
    let postprocess_ms = since_ms(t_postprocess_start);
    record_ms("postprocessing", t_postprocess_start);
    latency_stats.add_postprocess(postprocess_ms);
    
    println!("  ✓ Postprocessing completed in {:.3} ms", postprocess_ms);
    println!("  ✓ Final detections: {} objects", detections.items.len());
    
    // Calculate E2E latency
    let e2e_ms = since_ms(t_e2e_start);
    record_ms("e2e_pipeline", t_e2e_start);
    latency_stats.add_e2e(e2e_ms);
    
    println!("  ✓ E2E latency (capture→postprocess): {:.3} ms", e2e_ms);
    
    // Display detection results
    display_detections_packet(&detections);
    
    // Summary
    println!("\n✅ Pipeline Summary:");
    println!("  Stage 1 (Preprocessing):  {:.3} ms", preprocess_ms);
    println!("  Stage 2 (Inference):      {:.3} ms", inference_ms);
    println!("  Stage 3 (Postprocessing): {:.3} ms", postprocess_ms);
    println!("  Total E2E:                {:.3} ms", e2e_ms);
    println!("  Target FPS:               {:.1} fps", 1000.0 / e2e_ms);
    
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    const TARGET_FRAMES: usize = 10;
    
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║      COMPLETE POSTPROCESSING PIPELINE TEST               ║");
    println!("║    Capture → Preprocess → Inference → Postprocess        ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // 1. List DeckLink devices
    let devices = decklink_input::devicelist();
    println!("Available DeckLink devices: {}", devices.len());
    for (idx, name) in devices.iter().enumerate() {
        println!("  [{}] {}", idx, name);
    }
    
    if devices.is_empty() {
        println!("❌ No DeckLink devices found!");
        return Err("No DeckLink devices found".into());
    }
    
    // 2. Open capture session
    let mut session = CaptureSession::open(0)?;
    println!("✓ Opened DeckLink capture session on device 0\n");
    
    // 3. Initialize preprocessor (512x512 FP16, ImageNet normalization)
    println!("⚙️  Creating Preprocessor:");
    let mut preprocessor = Preprocessor::with_params(
        (512, 512),
        true,  // FP16
        0,     // GPU device 0
        [0.485, 0.456, 0.406],  // ImageNet mean
        [0.229, 0.224, 0.225],  // ImageNet std
        ChromaOrder::UYVY,      // UYVY chroma order
    )?;
    println!("  Output Size:   {}x{}", preprocessor.size.0, preprocessor.size.1);
    println!("  Data Type:     FP16");
    println!("  Device:        GPU 0");
    println!();
    
    // 4. Initialize TensorRT inference stage
    println!("🧠 Creating TensorRT Inference Stage:");
    let engine_path = "configs/model/optimized_YOLOv5.engine";
    let lib_path = "TRT_SHIM/libtrt_shim.so";
    
    if !std::path::Path::new(engine_path).exists() {
        println!("❌ TensorRT engine not found: {}", engine_path);
        return Err(format!("TensorRT engine not found: {}", engine_path).into());
    }
    
    if !std::path::Path::new(lib_path).exists() {
        println!("❌ TensorRT shim library not found: {}", lib_path);
        return Err(format!("TRT shim library not found: {}", lib_path).into());
    }
    
    let mut inference_stage = TrtInferenceStage::new(engine_path, lib_path)
        .map_err(|e| format!("Failed to create TrtInferenceStage: {}", e))?;
    
    println!("  Engine:        {}", engine_path);
    println!("  Library:       {}", lib_path);
    println!("  Output Size:   {} values", inference_stage.output_size());
    println!();
    
    // 5. Initialize postprocessing stage
    println!("🎯 Creating Postprocessing Stage:");
    use postprocess::YoloPostConfig;
    let yolo_cfg = YoloPostConfig {
        num_classes: 80,  // COCO dataset
        confidence_threshold: 0.25,
        nms_threshold: 0.45,
        max_detections: 300,
        letterbox_scale: 1.0,
        letterbox_pad: (0.0, 0.0),
        original_size: (1920, 1080),
    };
    
    let mut post_stage = PostStage::new(yolo_cfg)
        .with_temporal_smoothing(4)
        .with_sort_tracking(30, 0.3, 0.3);
    
    println!("  Confidence threshold: {}", yolo_cfg.confidence_threshold);
    println!("  NMS threshold:        {}", yolo_cfg.nms_threshold);
    println!("  Max detections:       {}", yolo_cfg.max_detections);
    println!("  Temporal smoothing:   enabled (window=4)");
    println!("  SORT tracking:        enabled (idle=30, min_conf=0.3, iou=0.3)");
    println!();
    
    // 6. Initialize latency tracker
    let mut latency_stats = LatencyStats::new();
    
    // 7. Capture and process frames
    println!("🎥 Waiting for frames from DeckLink...\n");
    let mut frames_processed = 0;
    
    for attempt in 0..200 {
        match session.get_frame()? {
            Some(raw_packet) => {
                frames_processed += 1;
                
                // Process complete pipeline
                process_frame(
                    raw_packet,
                    &mut preprocessor,
                    &mut inference_stage,
                    &mut post_stage,
                    &mut latency_stats,
                )?;
                
                if frames_processed >= TARGET_FRAMES {
                    println!("\n╔══════════════════════════════════════════════════════════╗");
                    println!("║        Successfully Processed {} Frames! ✓              ║", TARGET_FRAMES);
                    println!("╚══════════════════════════════════════════════════════════╝\n");
                    
                    // Print comprehensive summary
                    latency_stats.print_summary();
                    
                    return Ok(());
                }
                
                println!("\n{}", "─".repeat(60));
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
