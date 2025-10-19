use libloading::{Library, Symbol};
use std::ffi::CString;
use image::GenericImageView;

/// GPU buffer pointers returned from TensorRT
#[repr(C)]
struct DeviceBuffers {
    d_input: *mut std::ffi::c_void,   // GPU input buffer pointer
    d_output: *mut std::ffi::c_void,  // GPU output buffer pointer
    input_size: i32,                   // Size of input buffer in floats
    output_size: i32,                  // Size of output buffer in floats
}

/// Load image, resize to 640x640, and convert to CHW format (channels-first)
/// Returns normalized float data ready for YOLOv5
fn preprocess_image(img_path: &str) -> Vec<f32> {
    println!("📸 Loading image: {}", img_path);
    let img = image::open(img_path).expect("Failed to load image");
    
    println!("   Original size: {}x{}", img.width(), img.height());
    
    // Resize to YOLOv5 input size (640x640)
    let img = img.resize_exact(640, 640, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    
    println!("   Resized to: 640x640");
    
    // Convert HWC (Height, Width, Channels) → CHW (Channels, Height, Width)
    // YOLOv5 expects: [R_channel][G_channel][B_channel]
    let mut input_data = vec![0.0f32; 3 * 640 * 640];
    
    for y in 0..640 {
        for x in 0..640 {
            let pixel = img.get_pixel(x, y);
            
            // Normalize to [0.0, 1.0]
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            
            // CHW layout: all R values, then all G values, then all B values
            input_data[0 * 640 * 640 + y as usize * 640 + x as usize] = r;
            input_data[1 * 640 * 640 + y as usize * 640 + x as usize] = g;
            input_data[2 * 640 * 640 + y as usize * 640 + x as usize] = b;
        }
    }
    
    println!("   ✓ Preprocessed: 3×640×640 CHW format\n");
    
    input_data
}

fn main() {
    unsafe {
        // ═══════════════════════════════════════════════════════════════
        // STEP 1: Load TensorRT Shared Library
        // ═══════════════════════════════════════════════════════════════
        let lib = Library::new("../build/libtrt_shim.so")
            .expect("Failed to load library");
        
        // Load C function symbols from the shared library
        let create_session: Symbol<unsafe extern "C" fn(*const i8) -> *mut std::ffi::c_void> = 
            lib.get(b"create_session").expect("Failed to load create_session");
        
        let get_device_buffers: Symbol<unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut DeviceBuffers> = 
            lib.get(b"get_device_buffers").expect("Failed to load get_device_buffers");
        
        let run_inference_device: Symbol<unsafe extern "C" fn(
            *mut std::ffi::c_void, *const f32, *mut f32, i32, i32
        )> = lib.get(b"run_inference_device").expect("Failed to load run_inference_device");
        
        let copy_input_to_gpu: Symbol<unsafe extern "C" fn(
            *mut std::ffi::c_void, *const f32, i32
        )> = lib.get(b"copy_input_to_gpu").expect("Failed to load copy_input_to_gpu");
        
        let copy_output_to_cpu: Symbol<unsafe extern "C" fn(
            *mut std::ffi::c_void, *mut f32, i32
        )> = lib.get(b"copy_output_to_cpu").expect("Failed to load copy_output_to_cpu");
        
        let destroy_session: Symbol<unsafe extern "C" fn(*mut std::ffi::c_void)> = 
            lib.get(b"destroy_session").expect("Failed to load destroy_session");
        
        // ═══════════════════════════════════════════════════════════════
        // STEP 2: Create TensorRT Session (loads engine once)
        // ═══════════════════════════════════════════════════════════════
        println!("🚀 Creating TensorRT inference session...");
        let engine_path = CString::new("assets/optimized_YOLOv5.engine").unwrap();
        let session = create_session(engine_path.as_ptr());
        
        if session.is_null() {
            panic!("❌ Failed to create session");
        }
        println!("   ✓ Session created\n");
        
        // ═══════════════════════════════════════════════════════════════
        // STEP 3: Get GPU Buffer Pointers (pre-allocated by TensorRT)
        // ═══════════════════════════════════════════════════════════════
        let buffers_ptr = get_device_buffers(session);
        if buffers_ptr.is_null() {
            panic!("❌ Failed to get device buffers");
        }
        let buffers = &*buffers_ptr;
        
        println!("💾 GPU Buffers:");
        println!("   Input:  {:#x} ({} floats)", buffers.d_input as usize, buffers.input_size);
        println!("   Output: {:#x} ({} floats)\n", buffers.d_output as usize, buffers.output_size);
        
        // YOLOv5 tensor sizes
        let input_size = 3 * 640 * 640;   // 1.2M floats (3 channels × 640 × 640)
        let output_size = 25200 * 85;     // 2.1M floats (25200 detections × 85 values)
        
        // ═══════════════════════════════════════════════════════════════
        // STEP 4: Load and Preprocess Image
        // ═══════════════════════════════════════════════════════════════
        let input_cpu = preprocess_image("assets/sample.jpg");
        
        // ═══════════════════════════════════════════════════════════════
        // STEP 5: Zero-Copy GPU Inference Pipeline
        // ═══════════════════════════════════════════════════════════════
        println!("⚡ Zero-Copy Inference Pipeline");
        println!("   1️⃣  Copy input CPU → GPU");
        println!("   2️⃣  Run inference (GPU only)");
        println!("   3️⃣  Copy output GPU → CPU\n");
        
        // Get GPU buffer pointers
        let input_ptr = buffers.d_input as *const f32;
        let output_ptr = buffers.d_output as *mut f32;
        
        // --- 5a: Copy input to GPU ---
        let start = std::time::Instant::now();
        copy_input_to_gpu(session, input_cpu.as_ptr(), input_size as i32);
        let copy_in_time = start.elapsed();
        println!("   ✓ Input copied to GPU in {:.2}ms", copy_in_time.as_secs_f64() * 1000.0);
        
        // --- 5b: Run zero-copy inference ---
        let start = std::time::Instant::now();
        run_inference_device(session, input_ptr, output_ptr, input_size as i32, output_size as i32);
        let inference_time = start.elapsed();
        println!("   ✓ Inference completed in {:.2}ms", inference_time.as_secs_f64() * 1000.0);
        
        // --- 5c: Copy output to CPU ---
        let mut output_result: Vec<f32> = vec![0.0; output_size];
        let start = std::time::Instant::now();
        copy_output_to_cpu(session, output_result.as_mut_ptr(), output_size as i32);
        let copy_out_time = start.elapsed();
        println!("   ✓ Output copied to CPU in {:.2}ms\n", copy_out_time.as_secs_f64() * 1000.0);
        
        // ═══════════════════════════════════════════════════════════════
        // STEP 6: Analyze Results
        // ═══════════════════════════════════════════════════════════════
        println!("📊 Detection Results");
        
        // Show first 10 values
        print!("   First 10 values: ");
        for i in 0..10 {
            print!("{:.2} ", output_result[i]);
        }
        println!("...");
        
        // Count non-zero values
        let non_zero = output_result.iter().filter(|&&x| x.abs() > 0.001).count();
        println!("   Non-zero values: {}/{}", non_zero, output_size);
        
        // Find value range
        let max_val = output_result.iter().fold(0.0f32, |a, &b| a.max(b));
        let min_val = output_result.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        println!("   Value range: [{:.4}, {:.4}]", min_val, max_val);
        
        // Show first detection box if any
        if non_zero > 0 && output_result.len() >= 85 {
            println!("\n   ✅ Model produced detections!");
            println!("   First box: x={:.2}, y={:.2}, w={:.2}, h={:.2}, conf={:.4}", 
                     output_result[0], output_result[1], 
                     output_result[2], output_result[3], 
                     output_result[4]);
        }
        println!();
        
        // ═══════════════════════════════════════════════════════════════
        // STEP 7: Benchmark Zero-Copy Performance
        // ═══════════════════════════════════════════════════════════════
        println!("⏱️  Benchmarking (100 iterations)...");
        
        let warmup = 10;
        let iterations = 100;
        
        // Warmup GPU
        for _ in 0..warmup {
            run_inference_device(session, input_ptr, output_ptr, 
                               input_size as i32, output_size as i32);
        }
        
        // Measure performance
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            run_inference_device(session, input_ptr, output_ptr, 
                               input_size as i32, output_size as i32);
        }
        let total_duration = start.elapsed();
        
        let avg_time = total_duration.as_secs_f64() * 1000.0 / iterations as f64;
        let fps = 1000.0 / avg_time;
        
        println!("   Average inference: {:.2}ms", avg_time);
        println!("   Throughput: {:.1} FPS\n", fps);
        
        // ═══════════════════════════════════════════════════════════════
        // STEP 8: Performance Summary
        // ═══════════════════════════════════════════════════════════════
        println!("📈 Complete Pipeline Breakdown");
        println!("   ┌─────────────────────────────┬──────────┐");
        println!("   │ Stage                       │ Time     │");
        println!("   ├─────────────────────────────┼──────────┤");
        println!("   │ 1. Copy CPU → GPU           │ {:>6.2}ms │", copy_in_time.as_secs_f64() * 1000.0);
        println!("   │ 2. GPU Inference (avg)      │ {:>6.2}ms │", avg_time);
        println!("   │ 3. Copy GPU → CPU           │ {:>6.2}ms │", copy_out_time.as_secs_f64() * 1000.0);
        println!("   ├─────────────────────────────┼──────────┤");
        let total = copy_in_time.as_secs_f64() * 1000.0 + avg_time + copy_out_time.as_secs_f64() * 1000.0;
        println!("   │ Total Pipeline              │ {:>6.2}ms │", total);
        println!("   └─────────────────────────────┴──────────┘\n");
        
        println!("💡 Key Insights:");
        println!("   • Zero-copy inference: {:.2}ms (GPU only)", avg_time);
        println!("   • Complete pipeline: {:.2}ms (with CPU↔GPU copies)", total);
        println!("   • In a full GPU pipeline (CUDA preprocessing + postprocessing),");
        println!("     you skip steps 1 & 3 and only pay {:.2}ms per frame! 🚀\n", avg_time);
        
        // ═══════════════════════════════════════════════════════════════
        // STEP 9: Cleanup
        // ═══════════════════════════════════════════════════════════════
        destroy_session(session);
        println!("🧹 Session destroyed");
        
        // Keep library loaded to avoid segfault on exit
        std::mem::forget(lib);
    }
}
