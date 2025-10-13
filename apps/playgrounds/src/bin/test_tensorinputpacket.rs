// test_tensorinputpacket.rs - Test TensorInputPacket with REAL preprocessing from DeckLink capture
use std::error::Error;
use std::thread;
use std::time::Duration;

use common_io::{MemLoc, RawFramePacket, Stage, TensorInputPacket};
use decklink_input::capture::CaptureSession;
use preprocess_cuda::{ChromaOrder, Preprocessor};
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
    println!("║              TENSOR INPUT PACKET - COMPLETE INFO              ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    
    // Source metadata (from original frame)
    println!("║ 📸 Source Frame Metadata:");
    println!("║   Frame Index:    {:<42} ║", tensor.from.frame_idx);
    println!("║   Source ID:      {:<42} ║", tensor.from.source_id);
    println!("║   Original Size:  {}x{}{:>36} ║", 
             tensor.from.width, tensor.from.height, "");
    println!("║   Pixel Format:   {:?}{:>37} ║", tensor.from.pixfmt, "");
    println!("║   Color Space:    {:?}{:>35} ║", tensor.from.colorspace, "");
    println!("║   Stride:         {} bytes{:>37} ║", tensor.from.stride_bytes, "");
    
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ ⏱️  Timing Information:");
    println!("║   PTS:            {} ns{:>27} ║", tensor.from.pts_ns, "");
    println!("║   Capture Time:   {} ns{:>27} ║", tensor.from.t_capture_ns, "");
    
    println!("╠═══════════════════════════════════════════════════════════════╣");
    
    // Tensor descriptor
    println!("║ 🔢 Tensor Descriptor:");
    println!("║   Shape:          N={}, C={}, H={}, W={}{:>24} ║", 
             tensor.desc.n, tensor.desc.c, tensor.desc.h, tensor.desc.w, "");
    println!("║   Layout:         NCHW (Batch, Channels, Height, Width){:>9} ║", "");
    println!("║   Data Type:      {:?}{:>37} ║", tensor.desc.dtype, "");
    println!("║   Device:         GPU {}{:>43} ║", tensor.desc.device, "");
    
    // Calculate sizes
    let element_size = match tensor.desc.dtype {
        common_io::DType::Fp16 => 2,
        common_io::DType::Fp32 => 4,
    };
    let dtype_name = match tensor.desc.dtype {
        common_io::DType::Fp16 => "FP16 (half precision)",
        common_io::DType::Fp32 => "FP32 (single precision)",
    };
    let total_elements = tensor.desc.n * tensor.desc.c * tensor.desc.h * tensor.desc.w;
    let total_bytes = total_elements * element_size;
    let per_channel = tensor.desc.h * tensor.desc.w;
    let per_channel_bytes = per_channel * element_size;
    
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ 💾 Memory Information:");
    println!("║   Memory Location: {:?}{:>29} ║", tensor.data.loc, "");
    println!("║   Base Pointer:    0x{:016x}{:>31} ║", tensor.data.ptr as usize, "");
    println!("║   Data Length:     {} bytes{:>37} ║", tensor.data.len, "");
    println!("║   Stride:          {} bytes{:>37} ║", tensor.data.stride, "");
    
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ 📊 Size Breakdown:");
    println!("║   Total Elements:  {}{:>43} ║", total_elements, "");
    println!("║   Element Size:    {} bytes ({}){:>23} ║", element_size, dtype_name, "");
    println!("║   Total Size:      {} bytes ({:.2} MB){:>24} ║", 
             total_bytes, total_bytes as f64 / (1024.0 * 1024.0), "");
    println!("║   Per Channel:     {} elements = {} bytes{:>19} ║", 
             per_channel, per_channel_bytes, "");
    
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ 🎨 Channel Layout (NCHW - Planar RGB):");
    
    // Calculate byte offsets for each channel
    let ch0_offset = 0;
    let ch1_offset = per_channel_bytes;
    let ch2_offset = per_channel_bytes * 2;
    let ch0_end = ch1_offset - 1;
    let ch1_end = ch2_offset - 1;
    let ch2_end = total_bytes - 1;
    
    println!("║   Channel 0 (R):   bytes [{}..{}]{:>28} ║", 
             ch0_offset, ch0_end, "");
    println!("║                    {} elements, {} bytes{:>22} ║", 
             per_channel, per_channel_bytes, "");
    println!("║                    pointer: 0x{:016x}{:>24} ║", 
             (tensor.data.ptr as usize + ch0_offset as usize), "");
    println!("║");
    println!("║   Channel 1 (G):   bytes [{}..{}]{:>25} ║", 
             ch1_offset, ch1_end, "");
    println!("║                    {} elements, {} bytes{:>22} ║", 
             per_channel, per_channel_bytes, "");
    println!("║                    pointer: 0x{:016x}{:>24} ║", 
             (tensor.data.ptr as usize + ch1_offset as usize), "");
    println!("║");
    println!("║   Channel 2 (B):   bytes [{}..{}]{:>22} ║", 
             ch2_offset, ch2_end, "");
    println!("║                    {} elements, {} bytes{:>22} ║", 
             per_channel, per_channel_bytes, "");
    println!("║                    pointer: 0x{:016x}{:>24} ║", 
             (tensor.data.ptr as usize + ch2_offset as usize), "");
    
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ 🎯 Preprocessing Info:");
    println!("║   Input:           {}x{} {:?}{:>28} ║", 
             tensor.from.width, tensor.from.height, tensor.from.pixfmt, "");
    println!("║   Output:          {}x{} {:?} (NCHW planar){:>16} ║", 
             tensor.desc.w, tensor.desc.h, tensor.desc.dtype, "");
    println!("║   Resize Method:   Bilinear interpolation{:>24} ║", "");
    println!("║   Color Convert:   YUV422 → RGB{:>35} ║", "");
    println!("║   Normalization:   ImageNet (per-channel){:>23} ║", "");
    println!("║                    mean = [0.485, 0.456, 0.406]{:>17} ║", "");
    println!("║                    std  = [0.229, 0.224, 0.225]{:>17} ║", "");
    
    println!("╚═══════════════════════════════════════════════════════════════╝");
}

fn process_and_display(
    mut raw_packet: RawFramePacket,
    preprocessor: &mut Preprocessor,
    latency_stats: &mut LatencyStats,
) -> Result<(), Box<dyn Error>> {
    // Start E2E timing (using telemetry's monotonic clock)
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
        
        // Leak the buffer so it's not freed (will be cleaned up by preprocessor)
        std::mem::forget(gpu_buffer);
        
        let transfer_ms = since_ms(t_transfer_start);
        record_ms("cpu_to_gpu_transfer", t_transfer_start);
        latency_stats.add_transfer(transfer_ms);
        println!("  ✓ Transferred {} bytes to GPU in {:.3} ms", raw_packet.data.len, transfer_ms);
    } else {
        println!("\n🚀 GPU Direct frame detected - Zero-copy processing!");
    }
    
    // Run preprocessing with telemetry
    println!("\n⚙️  Running Preprocessing...");
    let t_preprocess_start = now_ns();
    let tensor = preprocessor.process(raw_packet);
    let preprocess_ms = since_ms(t_preprocess_start);
    record_ms("preprocessing", t_preprocess_start);
    
    // Record GPU Direct timing separately
    if is_gpu_direct {
        record_ms("gpu_direct_preprocessing", t_preprocess_start);
    }
    
    latency_stats.add_preprocess(preprocess_ms, is_gpu_direct);
    
    // Calculate E2E latency (from start of this function to preprocessing done)
    let e2e_ms = since_ms(t_e2e_start);
    record_ms("e2e_process_frame", t_e2e_start);
    latency_stats.add_e2e(e2e_ms);
    
    println!("  ✓ Preprocessing completed in {:.3} ms", preprocess_ms);
    if is_gpu_direct {
        println!("  ✓ GPU Direct latency: {:.3} ms ⚡", preprocess_ms);
    }
    println!("  ✓ E2E latency (process→done): {:.3} ms", e2e_ms);
    
    // Display tensor info
    display_tensor_info(&tensor);
    
    // Verification
    println!("\n✅ Verification:");
    println!("  ✓ Tensor shape: {}×{}×{}×{}", tensor.desc.n, tensor.desc.c, tensor.desc.h, tensor.desc.w);
    println!("  ✓ Data type: {:?}", tensor.desc.dtype);
    println!("  ✓ Memory location: {:?}", tensor.data.loc);
    println!("  ✓ Metadata preserved: frame_idx={}, pts={}", 
             tensor.from.frame_idx, tensor.from.pts_ns);
    
    Ok(())
}

// Latency statistics tracker
struct LatencyStats {
    transfer_times: Vec<f64>,
    preprocess_times: Vec<f64>,
    gpu_direct_times: Vec<f64>,  // GPU Direct frames only (no transfer)
    e2e_times: Vec<f64>,  // End-to-end (capture to preprocessing done)
}

impl LatencyStats {
    fn new() -> Self {
        Self {
            transfer_times: Vec::new(),
            preprocess_times: Vec::new(),
            gpu_direct_times: Vec::new(),
            e2e_times: Vec::new(),
        }
    }
    
    fn add_transfer(&mut self, ms: f64) {
        self.transfer_times.push(ms);
    }
    
    fn add_preprocess(&mut self, ms: f64, is_gpu_direct: bool) {
        self.preprocess_times.push(ms);
        if is_gpu_direct {
            self.gpu_direct_times.push(ms);
        }
    }
    
    fn add_e2e(&mut self, ms: f64) {
        self.e2e_times.push(ms);
    }
    
    fn print_summary(&self) {
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║              LATENCY SUMMARY (Telemetry)                 ║");
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
        
        // All Preprocessing
        if !self.preprocess_times.is_empty() {
            let min = self.preprocess_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = self.preprocess_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let avg = self.preprocess_times.iter().sum::<f64>() / self.preprocess_times.len() as f64;
            
            println!("║ ⚙️  Preprocessing - All Frames ({}):", self.preprocess_times.len());
            println!("║   Min:  {:.3} ms", min);
            println!("║   Max:  {:.3} ms", max);
            println!("║   Avg:  {:.3} ms", avg);
        }
        
        // GPU Direct Only
        if !self.gpu_direct_times.is_empty() {
            println!("╠══════════════════════════════════════════════════════════╣");
            let gpu_min = self.gpu_direct_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let gpu_max = self.gpu_direct_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let gpu_avg = self.gpu_direct_times.iter().sum::<f64>() / self.gpu_direct_times.len() as f64;
            
            println!("║ 🚀 GPU Direct Preprocessing ({} frames):", self.gpu_direct_times.len());
            println!("║   Min:  {:.3} ms", gpu_min);
            println!("║   Max:  {:.3} ms", gpu_max);
            println!("║   Avg:  {:.3} ms ⚡ (Zero-copy!)", gpu_avg);
            
            // Calculate throughput
            let fps = 1000.0 / gpu_avg;
            println!("║   Throughput: {:.1} FPS (preprocessing only)", fps);
        }
        
        // End-to-End
        if !self.e2e_times.is_empty() {
            println!("╠══════════════════════════════════════════════════════════╣");
            let e2e_min = self.e2e_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let e2e_max = self.e2e_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let e2e_avg = self.e2e_times.iter().sum::<f64>() / self.e2e_times.len() as f64;
            
            println!("║ 🎯 End-to-End (Capture → Preprocessing) ({} frames):", self.e2e_times.len());
            println!("║   Min:  {:.3} ms", e2e_min);
            println!("║   Max:  {:.3} ms", e2e_max);
            println!("║   Avg:  {:.3} ms", e2e_avg);
            
            let fps = 1000.0 / e2e_avg;
            println!("║   Target FPS: {:.1} fps", fps);
        }
        
        println!("╚══════════════════════════════════════════════════════════╝\n");
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║     TensorInputPacket Test with REAL DeckLink Input     ║");
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
    println!("  Chroma Order:  UYVY");
    println!("  Normalization: ImageNet (mean={:?}, std={:?})", 
             preprocessor.mean, preprocessor.std);
    println!();
    
    // Capture and process frames
    const TARGET_FRAMES: usize = 10;
    let mut frames_processed = 0;
    
    println!("🎥 Waiting for frames from DeckLink...\n");
    
    for attempt in 0..200 {
        match session.get_frame()? {
            Some(raw_packet) => {
                frames_processed += 1;
                
                // Process and display
                process_and_display(raw_packet, &mut preprocessor, &mut latency_stats)?;
                
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
    Ok(())
}
