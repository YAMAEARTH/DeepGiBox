// test_inference_pipeline.rs - Test complete pipeline: DeckLink → Preprocessing → Inference
use std::error::Error;
use std::thread;
use std::time::Duration;

use common_io::{MemLoc, RawFramePacket, Stage, TensorInputPacket, RawDetectionsPacket};
use decklink_input::capture::CaptureSession;
use preprocess_cuda::{ChromaOrder, Preprocessor};
use inference::InferenceEngine;
use config;
use cudarc::driver::{CudaDevice, DevicePtr, DevicePtrMut};
use telemetry::{now_ns, since_ms, record_ms};

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

fn display_tensor_info(tensor: &TensorInputPacket) {
    println!("\n🔷 TensorInputPacket Details:");
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    TENSOR INPUT PACKET                        ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    
    println!("║ 📸 Source Frame:");
    println!("║   Frame Index:    {:<42} ║", tensor.from.frame_idx);
    println!("║   Original Size:  {}x{}{:>36} ║", 
             tensor.from.width, tensor.from.height, "");
    
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ 🔢 Tensor Descriptor:");
    println!("║   Shape:          N={}, C={}, H={}, W={}{:>24} ║", 
             tensor.desc.n, tensor.desc.c, tensor.desc.h, tensor.desc.w, "");
    println!("║   Data Type:      {:?}{:>37} ║", tensor.desc.dtype, "");
    println!("║   Device:         GPU {}{:>43} ║", tensor.desc.device, "");
    
    let element_size = match tensor.desc.dtype {
        common_io::DType::Fp16 => 2,
        common_io::DType::Fp32 => 4,
    };
    let total_elements = tensor.desc.n * tensor.desc.c * tensor.desc.h * tensor.desc.w;
    let total_bytes = total_elements * element_size;
    
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ 💾 Memory:");
    println!("║   Location:       {:?}{:>29} ║", tensor.data.loc, "");
    println!("║   Total Size:     {} bytes ({:.2} MB){:>24} ║", 
             total_bytes, total_bytes as f64 / (1024.0 * 1024.0), "");
    
    println!("╚═══════════════════════════════════════════════════════════════╝");
}

fn display_raw_detections(detections: &RawDetectionsPacket) {
    println!("\n🎯 RawDetectionsPacket Details:");
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║              RAW DETECTIONS PACKET - COMPLETE INFO            ║");
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
    println!("║ 🔢 Raw Model Output:");
    println!("║   Total Values:   {}{:>43} ║", detections.raw_output.len(), "");
    
    // Display output shape
    if !detections.output_shape.is_empty() {
        let shape_str = format!("{:?}", detections.output_shape);
        println!("║   Output Shape:   {}{:>43} ║", shape_str, " ".repeat(43_usize.saturating_sub(shape_str.len())));
        
        // Parse YOLO format if applicable
        if detections.output_shape.len() == 3 {
            let (batch, anchors, features) = (
                detections.output_shape[0],
                detections.output_shape[1],
                detections.output_shape[2]
            );
            println!("║   Format:         Batch={}, Anchors={}, Features={}{} ║", 
                     batch, anchors, features, " ".repeat(43_usize.saturating_sub(format!("Batch={}, Anchors={}, Features={}", batch, anchors, features).len())));
            
            if features == 85 {
                println!("║   Detected:       YOLOv5 format (80 classes + 5 bbox){:>10} ║", "");
            } else if features == 6 {
                println!("║   Detected:       YOLOv8/v10 format (1 class + 4 bbox + 1 conf){} ║", "");
            }
        }
    }
    
    if detections.raw_output.is_empty() {
        println!("║   Status:         ⚠️  Empty output{:>33} ║", "");
    } else {
        // Calculate statistics
        let min = detections.raw_output.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = detections.raw_output.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mean = detections.raw_output.iter().sum::<f32>() / detections.raw_output.len() as f32;
        
        println!("║   Min Value:      {:.6}{:>37} ║", min, "");
        println!("║   Max Value:      {:.6}{:>37} ║", max, "");
        println!("║   Mean Value:     {:.6}{:>37} ║", mean, "");
        
        println!("╠═══════════════════════════════════════════════════════════════╣");
        println!("║ 📊 Output Preview:");
        
        // Show first 10 values
        if detections.raw_output.len() >= 10 {
            println!("║   First 10 values:");
            for i in 0..10 {
                println!("║     [{:4}] = {:12.6}{:>36} ║", i, detections.raw_output[i], "");
            }
        } else {
            println!("║   All {} values:", detections.raw_output.len());
            for (i, &val) in detections.raw_output.iter().enumerate() {
                println!("║     [{:4}] = {:12.6}{:>36} ║", i, val, "");
            }
        }
        
        // Show last 10 values if output is large
        if detections.raw_output.len() > 20 {
            println!("║   ...");
            println!("║   Last 10 values:");
            let start = detections.raw_output.len() - 10;
            for i in start..detections.raw_output.len() {
                println!("║     [{:4}] = {:12.6}{:>36} ║", i, detections.raw_output[i], "");
            }
        }
    }
    
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ 📝 Model Output Format:");
    println!("║   Type:           Raw tensor output from inference{:>13} ║", "");
    println!("║   Format:         Depends on model architecture{:>16} ║", "");
    println!("║   Next Stage:     Postprocessing (NMS, filtering){:>16} ║", "");
    println!("║");
    println!("║ 💡 Common Output Formats:");
    println!("║   • YOLO:         (1, 25200, 85) or (1, 8400, 85){:>17} ║", "");
    println!("║                   [x, y, w, h, conf, class_probs...]{:>13} ║", "");
    println!("║   • SSD:          [num_detections, boxes, scores, classes]{:>8} ║", "");
    println!("║   • RetinaNet:    Multiple scale outputs{:>26} ║", "");
    
    println!("╚═══════════════════════════════════════════════════════════════╝");
}

// Latency statistics tracker
struct LatencyStats {
    transfer_times: Vec<f64>,
    preprocess_times: Vec<f64>,
    inference_times: Vec<f64>,
    gpu_direct_preprocess: Vec<f64>,
    e2e_times: Vec<f64>,  // End-to-end (capture to inference done)
}

impl LatencyStats {
    fn new() -> Self {
        Self {
            transfer_times: Vec::new(),
            preprocess_times: Vec::new(),
            inference_times: Vec::new(),
            gpu_direct_preprocess: Vec::new(),
            e2e_times: Vec::new(),
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
        
        // End-to-End
        if !self.e2e_times.is_empty() {
            println!("╠══════════════════════════════════════════════════════════╣");
            let min = self.e2e_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = self.e2e_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let avg = self.e2e_times.iter().sum::<f64>() / self.e2e_times.len() as f64;
            
            println!("║ 🎯 End-to-End Pipeline ({} frames):", self.e2e_times.len());
            println!("║   Capture → Preprocess → Inference");
            println!("║   Min:  {:.3} ms", min);
            println!("║   Max:  {:.3} ms", max);
            println!("║   Avg:  {:.3} ms", avg);
            
            let fps = 1000.0 / avg;
            println!("║   Target FPS: {:.1} fps", fps);
            
            // Calculate individual stage contributions
            if !self.preprocess_times.is_empty() && !self.inference_times.is_empty() {
                let prep_avg = self.preprocess_times.iter().sum::<f64>() / self.preprocess_times.len() as f64;
                let inf_avg = self.inference_times.iter().sum::<f64>() / self.inference_times.len() as f64;
                let prep_pct = (prep_avg / avg) * 100.0;
                let inf_pct = (inf_avg / avg) * 100.0;
                
                println!("║");
                println!("║ 📊 Stage Breakdown:");
                println!("║   Preprocessing: {:.1}%", prep_pct);
                println!("║   Inference:     {:.1}%", inf_pct);
            }
        }
        
        println!("╚══════════════════════════════════════════════════════════╝\n");
    }
}

fn process_frame(
    mut raw_packet: RawFramePacket,
    preprocessor: &mut Preprocessor,
    inference_engine: &mut InferenceEngine,
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
    let is_gpu_direct = matches!(raw_packet.data.loc, MemLoc::Gpu { .. });
    let needs_gpu_transfer = !is_gpu_direct;
    
    if needs_gpu_transfer {
        println!("\n📤 Transferring frame from CPU to GPU...");
        
        let t_transfer_start = now_ns();
        
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
        raw_packet.data.loc = MemLoc::Gpu { device: 0 };
        
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
    
    // Display tensor info
    display_tensor_info(&tensor);
    
    // Stage 2: Inference
    println!("\n🧠 Stage 2: Inference...");
    let t_inference_start = now_ns();
    let detections = inference_engine.process(tensor);
    let inference_ms = since_ms(t_inference_start);
    record_ms("inference", t_inference_start);
    latency_stats.add_inference(inference_ms);
    
    println!("  ✓ Inference completed in {:.3} ms", inference_ms);
    
    // Calculate E2E latency
    let e2e_ms = since_ms(t_e2e_start);
    record_ms("e2e_pipeline", t_e2e_start);
    latency_stats.add_e2e(e2e_ms);
    
    println!("  ✓ E2E latency (capture→inference): {:.3} ms", e2e_ms);
    
    // Display detection results
    display_raw_detections(&detections);
    
    // Summary
    println!("\n✅ Pipeline Summary:");
    println!("  Stage 1 (Preprocessing): {:.3} ms", preprocess_ms);
    println!("  Stage 2 (Inference):     {:.3} ms", inference_ms);
    println!("  Total E2E:               {:.3} ms", e2e_ms);
    println!("  Target FPS:              {:.1} fps", 1000.0 / e2e_ms);
    
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║    Complete Inference Pipeline Test with Real DeckLink  ║");
    println!("║         Capture → Preprocessing → Inference             ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
    
    // Initialize latency tracker
    let mut latency_stats = LatencyStats::new();
    
    // List available DeckLink devices
    let devices = decklink_input::devicelist();
    println!("Available DeckLink devices: {}", devices.len());
    for (idx, name) in devices.iter().enumerate() {
        println!("  [{}] {}", idx, name);
    }
    println!();
    
    if devices.is_empty() {
        println!("❌ No DeckLink devices found!");
        println!("   Please connect a DeckLink device to test with real frames.\n");
        return Err("No DeckLink devices found".into());
    }
    
    // Open capture session
    let mut session = CaptureSession::open(0)?;
    println!("✓ Opened DeckLink capture session on device 0\n");
    
    // Create preprocessor (512x512 FP16, ImageNet normalization)
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
    
    // Create inference engine from config
    println!("🧠 Creating Inference Engine:");
    
    // Create a simple config
    use config::InferenceCfg;
    let model_path = "configs/model/YOLOv5.onnx";
    
    if !std::path::Path::new(model_path).exists() {
        println!("❌ Model file not found: {}", model_path);
        println!("   Please place a YOLO model at this location.");
        return Err("Model file not found".into());
    }
    
    let cfg = InferenceCfg {
        backend: "onnxruntime_trt".to_string(),
        model: model_path.to_string(),
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
    
    // Capture and process frames
    const TARGET_FRAMES: usize = 10;
    let mut frames_processed = 0;
    
    println!("🎥 Waiting for frames from DeckLink...\n");
    
    for attempt in 0..200 {
        match session.get_frame()? {
            Some(raw_packet) => {
                frames_processed += 1;
                
                // Process frame through complete pipeline
                process_frame(
                    raw_packet,
                    &mut preprocessor,
                    &mut inference_engine,
                    &mut latency_stats,
                )?;
                
                if frames_processed >= TARGET_FRAMES {
                    println!("\n╔══════════════════════════════════════════════════════════╗");
                    println!("║        Successfully Processed {} Frames! ✓              ║", TARGET_FRAMES);
                    println!("╚══════════════════════════════════════════════════════════╝\n");
                    
                    // Print latency summary
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
    
    if frames_processed > 0 {
        latency_stats.print_summary();
    }
    
    Ok(())
}
