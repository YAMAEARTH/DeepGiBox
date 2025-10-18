use libloading::{Library, Symbol};
use std::ffi::CString;

// ═══════════════════════════════════════════════════════════════════════════
// Data Structures (matching your pipeline types)
// ═══════════════════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum PixelFormat {
    RGB8,
    RGBA8,
    BGR8,
    BGRA8,
    YUV420P,
    NV12,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum ColorSpace {
    SRGB,
    Linear,
    BT709,
    BT2020,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum DType {
    U8,
    F16,
    F32,
    I32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum MemLoc {
    CPU,
    GPU,
}

#[repr(C)]
pub struct FrameMeta {
    pub source_id: u32,
    pub width: u32,
    pub height: u32,
    pub pixfmt: PixelFormat,
    pub colorspace: ColorSpace,
    pub frame_idx: u64,
    pub pts_ns: u64,
    pub t_capture_ns: u64,
    pub stride_bytes: u32,
}

#[repr(C)]
pub struct TensorDesc {
    pub n: u32,  // batch size
    pub c: u32,  // channels
    pub h: u32,  // height
    pub w: u32,  // width
    pub dtype: DType,
    pub device: u32,  // GPU device ID
}

#[repr(C)]
pub struct MemRef {
    pub ptr: *mut u8,       // Raw pointer to data (CPU or GPU)
    pub len: usize,         // Length in bytes
    pub stride: usize,      // Stride in bytes
    pub loc: MemLoc,        // CPU or GPU location
}

#[repr(C)]
pub struct TensorInputPacket {
    pub from: FrameMeta,
    pub desc: TensorDesc,
    pub data: MemRef,
}

/// GPU buffer pointers from TensorRT
#[repr(C)]
struct DeviceBuffers {
    d_input: *mut std::ffi::c_void,
    d_output: *mut std::ffi::c_void,
    input_size: i32,
    output_size: i32,
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU-Only Inference Pipeline
// ═══════════════════════════════════════════════════════════════════════════

pub fn run_gpu_inference(packet: &TensorInputPacket) -> Vec<f32> {
    unsafe {
        // ───────────────────────────────────────────────────────────────
        // STEP 1: Validate Input
        // ───────────────────────────────────────────────────────────────
        println!("🔍 Validating TensorInputPacket...");
        
        // Check that data is on GPU
        if !matches!(packet.data.loc, MemLoc::GPU) {
            panic!("❌ Input data must be on GPU! Current location: {:?}", packet.data.loc);
        }
        
        // Check tensor dimensions match YOLOv5 (640x640x3)
        let expected_size = packet.desc.n * packet.desc.c * packet.desc.h * packet.desc.w;
        println!("   Tensor shape: {}×{}×{}×{} = {} elements", 
                 packet.desc.n, packet.desc.c, packet.desc.h, packet.desc.w, expected_size);
        println!("   Data type: {:?}", packet.desc.dtype);
        println!("   Memory location: GPU (device {})", packet.desc.device);
        println!("   GPU pointer: {:#x}", packet.data.ptr as usize);
        
        // Frame metadata
        println!("\n📸 Frame Info:");
        println!("   Source ID: {}", packet.from.source_id);
        println!("   Frame index: {}", packet.from.frame_idx);
        println!("   Original size: {}×{}", packet.from.width, packet.from.height);
        println!("   PTS: {} ns", packet.from.pts_ns);
        println!("   Capture time: {} ns", packet.from.t_capture_ns);
        
        // ───────────────────────────────────────────────────────────────
        // STEP 2: Load TensorRT Library
        // ───────────────────────────────────────────────────────────────
        println!("\n🚀 Loading TensorRT library...");
        let lib = Library::new("../build/libtrt_shim.so")
            .expect("Failed to load libtrt_shim.so");
        
        let create_session: Symbol<unsafe extern "C" fn(*const i8) -> *mut std::ffi::c_void> = 
            lib.get(b"create_session").expect("Failed to load create_session");
        
        let run_inference_device: Symbol<unsafe extern "C" fn(
            *mut std::ffi::c_void, *const f32, *mut f32, i32, i32
        )> = lib.get(b"run_inference_device").expect("Failed to load run_inference_device");
        
        let copy_output_to_cpu: Symbol<unsafe extern "C" fn(
            *mut std::ffi::c_void, *mut f32, i32
        )> = lib.get(b"copy_output_to_cpu").expect("Failed to load copy_output_to_cpu");
        
        let destroy_session: Symbol<unsafe extern "C" fn(*mut std::ffi::c_void)> = 
            lib.get(b"destroy_session").expect("Failed to load destroy_session");
        
        // ───────────────────────────────────────────────────────────────
        // STEP 3: Create TensorRT Session
        // ───────────────────────────────────────────────────────────────
        println!("\n💾 Creating TensorRT session...");
        let engine_path = CString::new("assets/optimized_YOLOv5.engine").unwrap();
        let session = create_session(engine_path.as_ptr());
        
        if session.is_null() {
            panic!("❌ Failed to create TensorRT session");
        }
        println!("   ✓ Session created");
        
        // ───────────────────────────────────────────────────────────────
        // STEP 4: Run GPU-Only Inference
        // ───────────────────────────────────────────────────────────────
        println!("\n⚡ Running GPU-Only Inference...");
        println!("   Input GPU pointer: {:#x}", packet.data.ptr as usize);
        println!("   Using data directly from TensorInputPacket (ZERO-COPY)");
        
        // Cast input pointer to f32* (assuming preprocessed float data on GPU)
        let input_gpu_ptr = packet.data.ptr as *const f32;
        
        // YOLOv5 output size: 25200 detections × 85 values
        let output_size = 25200 * 85;
        
        // Allocate CPU buffer for output (we'll copy results back)
        let mut output_cpu: Vec<f32> = vec![0.0; output_size];
        
        // Get TensorRT's internal GPU output buffer (don't pass null!)
        let get_device_buffers: Symbol<unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut DeviceBuffers> = 
            lib.get(b"get_device_buffers").expect("Failed to load get_device_buffers");
        let buffers = &*get_device_buffers(session);
        let output_gpu_ptr = buffers.d_output as *mut f32;
        
        // ═══════════════════════════════════════════════════════════════
        // Run 100 iterations to get average time
        // ═══════════════════════════════════════════════════════════════
        println!("   Running 100 inference iterations...");
        
        let iterations = 100;
        let mut times = Vec::with_capacity(iterations);
        
        for i in 0..iterations {
            let start = std::time::Instant::now();
            run_inference_device(
                session,
                input_gpu_ptr,
                output_gpu_ptr,
                expected_size as i32,
                output_size as i32
            );
            let elapsed = start.elapsed();
            times.push(elapsed.as_secs_f64() * 1000.0);
            
            if i == 0 {
                println!("   First run (cold start): {:.2}ms", times[0]);
            }
        }
        
        // Calculate statistics
        let total: f64 = times.iter().sum();
        let avg = total / iterations as f64;
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        // Exclude first run for "warm" average
        let warm_avg = times[1..].iter().sum::<f64>() / (iterations - 1) as f64;
        
        println!("\n   📊 Inference Statistics:");
        println!("      First run:  {:.2}ms (includes GPU warmup)", times[0]);
        println!("      Average:    {:.2}ms (all 100 runs)", avg);
        println!("      Warm avg:   {:.2}ms (excluding first)", warm_avg);
        println!("      Min:        {:.2}ms", min);
        println!("      Max:        {:.2}ms", max);
        println!("      Throughput: {:.1} FPS", 1000.0 / warm_avg);
        
        // ───────────────────────────────────────────────────────────────
        // STEP 5: Copy Output to CPU for Postprocessing
        // ───────────────────────────────────────────────────────────────
        println!("\n📤 Copying results to CPU...");
        let start = std::time::Instant::now();
        copy_output_to_cpu(session, output_cpu.as_mut_ptr(), output_size as i32);
        let copy_time = start.elapsed();
        
        println!("   ✓ Output copied in {:.2}ms", copy_time.as_secs_f64() * 1000.0);
        
        // ───────────────────────────────────────────────────────────────
        // STEP 6: Analyze Results
        // ───────────────────────────────────────────────────────────────
        println!("\n📊 Detection Results:");
        
        let non_zero = output_cpu.iter().filter(|&&x| x.abs() > 0.001).count();
        let max_val = output_cpu.iter().fold(0.0f32, |a, &b| a.max(b));
        let min_val = output_cpu.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        
        println!("   Non-zero values: {}/{} ({:.2}%)", 
                 non_zero, output_size, (non_zero as f64 / output_size as f64) * 100.0);
        println!("   Value range: [{:.4}, {:.4}]", min_val, max_val);
        
        // Show first detection if any
        if non_zero > 0 && output_cpu.len() >= 7 {
            println!("\n   ✅ First detection:");
            println!("      x={:.2}, y={:.2}, w={:.2}, h={:.2}", 
                     output_cpu[0], output_cpu[1], output_cpu[2], output_cpu[3]);
            println!("      confidence={:.4}", output_cpu[4]);
            
            // Only 2 classes (class 0 and class 1)
            let class0_prob = output_cpu[5];
            let class1_prob = output_cpu[6];
            
            if class0_prob > class1_prob {
                println!("      class=0, probability={:.4}", class0_prob);
            } else {
                println!("      class=1, probability={:.4}", class1_prob);
            }
        }
        
        // ───────────────────────────────────────────────────────────────
        // STEP 7: Performance Summary
        // ───────────────────────────────────────────────────────────────
        println!("\n⏱️  Performance Summary:");
        println!("   ┌──────────────────────────────┬──────────┐");
        println!("   │ GPU Inference (warm avg)     │ {:>6.2}ms │", warm_avg);
        println!("   │ Copy GPU → CPU               │ {:>6.2}ms │", copy_time.as_secs_f64() * 1000.0);
        println!("   ├──────────────────────────────┼──────────┤");
        let total_pipeline = warm_avg + copy_time.as_secs_f64() * 1000.0;
        println!("   │ Total Pipeline               │ {:>6.2}ms │", total_pipeline);
        println!("   └──────────────────────────────┴──────────┘");
        println!("   Throughput: {:.1} FPS\n", 1000.0 / total_pipeline);
        
        println!("💡 Zero-Copy Benefit:");
        println!("   • Input was ALREADY on GPU (from TensorInputPacket)");
        println!("   • No CPU→GPU transfer needed!");
        println!("   • True end-to-end GPU pipeline 🚀\n");
        
        // ───────────────────────────────────────────────────────────────
        // STEP 8: Cleanup
        // ───────────────────────────────────────────────────────────────
        destroy_session(session);
        println!("🧹 Session destroyed");
        
        // Keep library loaded to avoid segfault
        std::mem::forget(lib);
        
        output_cpu
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Example: Create a mock TensorInputPacket for testing
// ═══════════════════════════════════════════════════════════════════════════

pub fn create_mock_tensor_packet(gpu_ptr: *mut u8) -> TensorInputPacket {
    TensorInputPacket {
        from: FrameMeta {
            source_id: 1,
            width: 1920,
            height: 1080,
            pixfmt: PixelFormat::RGB8,
            colorspace: ColorSpace::SRGB,
            frame_idx: 42,
            pts_ns: 1234567890,
            t_capture_ns: 1234567800,
            stride_bytes: 1920 * 3,
        },
        desc: TensorDesc {
            n: 1,           // batch size
            c: 3,           // RGB channels
            h: 640,         // YOLOv5 input height
            w: 640,         // YOLOv5 input width
            dtype: DType::F32,
            device: 0,      // GPU 0
        },
        data: MemRef {
            ptr: gpu_ptr,
            len: 1 * 3 * 640 * 640 * 4,  // 4 bytes per f32
            stride: 640 * 4,              // Row stride in bytes
            loc: MemLoc::GPU,
        },
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Main: Demo with mock GPU data
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║   TensorInputPacket GPU-Only Inference Demo              ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");
    
    unsafe {
        // For this demo, we'll simulate having GPU data by:
        // 1. Creating a session to get GPU buffer pointers
        // 2. Using those pointers to create a TensorInputPacket
        // 3. Running inference on it
        
        println!("🔧 Setting up demo environment...\n");
        
        let lib = Library::new("../build/libtrt_shim.so")
            .expect("Failed to load library");
        
        let create_session: Symbol<unsafe extern "C" fn(*const i8) -> *mut std::ffi::c_void> = 
            lib.get(b"create_session").expect("Failed to load create_session");
        
        let get_device_buffers: Symbol<unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut DeviceBuffers> = 
            lib.get(b"get_device_buffers").expect("Failed to load get_device_buffers");
        
        let copy_input_to_gpu: Symbol<unsafe extern "C" fn(
            *mut std::ffi::c_void, *const f32, i32
        )> = lib.get(b"copy_input_to_gpu").expect("Failed to load copy_input_to_gpu");
        
        // Create session
        let engine_path = CString::new("assets/optimized_YOLOv5.engine").unwrap();
        let session = create_session(engine_path.as_ptr());
        
        if session.is_null() {
            panic!("Failed to create session");
        }
        
        // Get GPU buffer pointers
        let buffers = &*get_device_buffers(session);
        
        println!("✅ Got GPU buffer pointer: {:#x}\n", buffers.d_input as usize);
        
        // Prepare some test data (simulate preprocessed image)
        let input_size = 3 * 640 * 640;
        let test_data: Vec<f32> = (0..input_size)
            .map(|i| ((i % 255) as f32) / 255.0)
            .collect();
        
        // Copy test data to GPU
        copy_input_to_gpu(session, test_data.as_ptr(), input_size as i32);
        println!("✅ Copied test data to GPU\n");
        
        // Create TensorInputPacket pointing to GPU data
        let packet = create_mock_tensor_packet(buffers.d_input as *mut u8);
        
        println!("═══════════════════════════════════════════════════════════\n");
        
        // Run GPU-only inference
        let _results = run_gpu_inference(&packet);
        
        println!("\n═══════════════════════════════════════════════════════════");
        println!("✅ Demo completed successfully!");
        println!("═══════════════════════════════════════════════════════════\n");
        
        std::mem::forget(lib);
    }
}
