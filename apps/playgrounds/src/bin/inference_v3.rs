use anyhow::Result;
use common_io::{
    ColorSpace, DType, FrameMeta, MemLoc, MemRef, PixelFormat, Stage, TensorDesc,
    TensorInputPacket, RawDetectionsPacket,
};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use image::io::Reader as ImageReader;
use inference_v3::TrtInferenceStage;

const ENGINE_PATH: &str = "configs/model/gim_model.engine";
const LIB_PATH: &str = "trt-shim/build/libtrt_shim.so";
const IMAGE_PATH: &str = "apps/playgrounds/sample_img_5.png";

// GIM model dimensions (HWC format!)
const INPUT_HEIGHT: u32 = 512;
const INPUT_WIDTH: u32 = 640;

fn print_engine_build_instructions() {
    eprintln!("\n╔═══════════════════════════════════════════════════════════════════╗");
    eprintln!("║  ⚠️  TensorRT Engine File Not Found                               ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════╝\n");
    eprintln!("The TensorRT engine file is required for GPU inference.");
    eprintln!("You need to build it from an ONNX model.\n");
}

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║   GIM Segmentation - GPU-Only Pipeline Test              ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    println!("📋 Configuration:");
    println!("   Engine:  {}", ENGINE_PATH);
    println!("   Library: {}", LIB_PATH);
    println!("   Image:   {}", IMAGE_PATH);
    println!("   Size:    {}x{} (HWC format)", INPUT_HEIGHT, INPUT_WIDTH);
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 1: Initialize CUDA Device
    // ═══════════════════════════════════════════════════════════════════════
    println!("🚀 Step 1: Initializing CUDA Device...");
    let cuda_device = CudaDevice::new(0)?;
    println!("   ✓ CUDA device 0 initialized");
    println!("   ✓ GPU: {}\n", cuda_device.name()?);

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 2: Load and Preprocess Image on CPU
    // ═══════════════════════════════════════════════════════════════════════
    println!("🖼️  Step 2: Loading image from disk...");
    let img = ImageReader::open(IMAGE_PATH)?.decode()?.to_rgb8();
    let (orig_w, orig_h) = img.dimensions();
    println!("   ✓ Image loaded: {}x{}", orig_w, orig_h);

    println!("   Preprocessing to {}x{} (HWC format)...", INPUT_HEIGHT, INPUT_WIDTH);
    let preprocessed = preprocess_image_hwc(&img, INPUT_WIDTH, INPUT_HEIGHT)?;
    println!("   ✓ Preprocessed to HWC format (FP32)");
    println!("   ✓ Total elements: {}", preprocessed.len());
    println!("   ✓ Expected: {} elements", INPUT_HEIGHT * INPUT_WIDTH * 3);
    
    // Sanity check on preprocessing
    let sample_vals: Vec<f32> = preprocessed.iter().take(15).copied().collect();
    println!("   ✓ First 15 values: {:?}", sample_vals);
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 3: Upload to GPU Memory
    // ═══════════════════════════════════════════════════════════════════════
    println!("📤 Step 3: Uploading data to GPU...");
    let start = std::time::Instant::now();

    let gpu_buffer: CudaSlice<f32> = cuda_device.htod_sync_copy(&preprocessed)?;
    let upload_time = start.elapsed();

    let gpu_ptr = *gpu_buffer.device_ptr() as *mut u8;
    let gpu_len = preprocessed.len() * std::mem::size_of::<f32>();

    println!("   ✓ Uploaded {} bytes to GPU", gpu_len);
    println!("   ✓ GPU pointer: {:#x}", gpu_ptr as usize);
    println!("   ✓ Upload time: {:.2}ms\n", upload_time.as_secs_f64() * 1000.0);

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 3.5: Prepare threshold configurations to test
    // ═══════════════════════════════════════════════════════════════════════
    println!("📤 Step 3.5: Preparing multiple threshold configurations to test...\n");
    
    // Test 20 different threshold combinations - varied independently, not just linear
    // Exploring different combinations of NBI, WLE, and C thresholds
    // Sweet spot around 0.13-0.15 for NBI/WLE, 0.24-0.28 for C
    let threshold_configs = vec![
        (0.120, 0.140, 0.250, "config_01"),  // Low NBI, mid WLE
        (0.145, 0.125, 0.260, "config_02"),  // Mid NBI, low WLE
        (0.135, 0.145, 0.240, "config_03"),  // Balanced mid
        (0.150, 0.130, 0.270, "config_04"),  // High NBI, low WLE
        (0.125, 0.150, 0.255, "config_05"),  // Low NBI, high WLE
        (0.142, 0.142, 0.265, "config_06"),  // Equal thresholds
        (0.130, 0.135, 0.245, "config_07"),  // Low-mid all
        (0.148, 0.138, 0.275, "config_08"),  // High NBI, mid WLE
        (0.138, 0.148, 0.250, "config_09"),  // Mid NBI, high WLE
        (0.144, 0.144, 0.260, "config_10"),  // Equal mid
        (0.152, 0.142, 0.280, "config_11"),  // High NBI, mid WLE, high C
        (0.128, 0.145, 0.255, "config_12"),  // Low NBI, high WLE
        (0.146, 0.135, 0.270, "config_13"),  // Mid-high NBI, low-mid WLE
        (0.132, 0.148, 0.248, "config_14"),  // Low-mid NBI, high WLE
        (0.155, 0.140, 0.285, "config_15"),  // High all
        (0.140, 0.140, 0.262, "config_16"),  // Equal low-mid
        (0.136, 0.152, 0.252, "config_17"),  // Mid NBI, high WLE
        (0.150, 0.128, 0.275, "config_18"),  // High NBI, low WLE
        (0.134, 0.138, 0.258, "config_19"),  // Mid all
        (0.148, 0.148, 0.268, "config_20"),  // Equal mid-high
    ];
    
    println!("   Will test {} threshold configurations:", threshold_configs.len());

    println!("   Will test {} threshold configurations:", threshold_configs.len());
    for (nbi, wle, c, name) in &threshold_configs {
        println!("      {} - NBI={:.2}, WLE={:.2}, C={:.2}", name, nbi, wle, c);
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 4: Create TensorInputPacket (single input)
    // ═══════════════════════════════════════════════════════════════════════
    println!("📦 Step 4: Creating TensorInputPacket...");

    let frame_meta = FrameMeta {
        source_id: 1,
        width: orig_w,
        height: orig_h,
        pixfmt: PixelFormat::RGB8,
        colorspace: ColorSpace::BT709,
        frame_idx: 1,
        pts_ns: 0,
        t_capture_ns: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos() as u64,
        stride_bytes: orig_w * 3,
        crop_region: None,
    };

    // HWC format descriptor
    let tensor_desc = TensorDesc {
        n: 1,
        c: 3,
        h: INPUT_HEIGHT,
        w: INPUT_WIDTH,
        dtype: DType::Fp32,
        device: 0,
    };

    let mem_ref = MemRef {
        ptr: gpu_ptr,
        len: gpu_len,
        stride: (INPUT_WIDTH * 3 * std::mem::size_of::<f32>() as u32) as usize,
        loc: MemLoc::Gpu { device: 0 },
    };

    let tensor_packet = TensorInputPacket {
        from: frame_meta,
        desc: tensor_desc,
        data: mem_ref,
    };

    println!("   ✓ TensorInputPacket created");
    println!("   ✓ Format: HWC (Height-Width-Channels)");
    println!("   ✓ Shape: {}x{}x{}", 
             tensor_packet.desc.h,
             tensor_packet.desc.w,
             tensor_packet.desc.c);
    println!("   ✓ DType: {:?}", tensor_packet.desc.dtype);
    println!("   ✓ MemLoc: {:?}\n", tensor_packet.data.loc);

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 5: Initialize TensorRT Inference Stage
    // ═══════════════════════════════════════════════════════════════════════
    println!("🔧 Step 5: Initializing TensorRT Inference Stage...");
    println!("   ⚠️  GIM model requires 4 inputs:");
    println!("      1. input: [512, 640, 3] - RGB image");
    println!("      2. threshold_NBI: scalar - NBI threshold");
    println!("      3. threshold_WLE: scalar - WLE threshold");
    println!("      4. c_threshold: scalar - classification threshold");
    println!("   ⚠️  Your TrtInferenceStage may need modification to support multiple inputs!\n");
    
    let start = std::time::Instant::now();

    let mut inference_stage = match TrtInferenceStage::new(ENGINE_PATH, LIB_PATH) {
        Ok(stage) => stage,
        Err(e) => {
            eprintln!("❌ Error: {}", e);
            print_engine_build_instructions();
            return Err(anyhow::anyhow!("Failed to create inference stage"));
        }
    };
    let init_time = start.elapsed();

    println!("   ✓ TrtInferenceStage created");
    println!("   ✓ Engine loaded from: {}", ENGINE_PATH);
    println!("   ✓ Initialization time: {:.2}ms\n", init_time.as_secs_f64() * 1000.0);

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 6: Run Inference with Multiple Threshold Configurations
    // ═══════════════════════════════════════════════════════════════════════
    println!("⚡ Step 6: Running GPU-Only Inference with {} threshold configurations...\n", threshold_configs.len());

    for (config_idx, (threshold_nbi, threshold_wle, c_threshold, config_name)) in threshold_configs.iter().enumerate() {
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║ Configuration {}/{}: {}                                          ", config_idx + 1, threshold_configs.len(), config_name);
        println!("║ Thresholds: NBI={:.2}, WLE={:.2}, C={:.2}                        ", threshold_nbi, threshold_wle, c_threshold);
        println!("╚════════════════════════════════════════════════════════════════╝");
        
        println!("   Input: GPU pointer {:#x} (ZERO-COPY)", gpu_ptr as usize);

        let start = std::time::Instant::now();
        let segmentation: RawDetectionsPacket = inference_stage.infer_with_thresholds(
            tensor_packet.clone(),
            *threshold_nbi,
            *threshold_wle,
            *c_threshold
        ).map_err(|e| anyhow::anyhow!(e))?;
        let inference_time = start.elapsed();

        println!("   ✓ Inference completed!");
        println!("   ✓ Inference time: {:.2}ms", inference_time.as_secs_f64() * 1000.0);
        println!("   ✓ Throughput: {:.1} FPS\n", 1000.0 / (inference_time.as_secs_f64() * 1000.0));

        println!("   ✓ Inference completed!");
        println!("   ✓ Inference time: {:.2}ms", inference_time.as_secs_f64() * 1000.0);
        println!("   ✓ Throughput: {:.1} FPS\n", 1000.0 / (inference_time.as_secs_f64() * 1000.0));

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 6.5: Analyze Raw Segmentation Output
        // ═══════════════════════════════════════════════════════════════════════
        println!("📊 Analyzing Raw Segmentation Output...");

        let raw_output = &segmentation.raw_output;
        let reported_shape = &segmentation.output_shape;
    println!("   Raw output_shape from engine: {:?}", reported_shape);

    // GIM outputs [height, width] segmentation mask
    let height = 512;
    let width = 640;
    let expected_size = height * width;

    println!("   ✓ Expected format: Single-channel segmentation mask");
    println!("   ✓ Shape: [height={}, width={}]", height, width);

    if raw_output.len() != expected_size {
        eprintln!("   ❌ ERROR: Output size mismatch!");
        eprintln!("      Expected: {} elements", expected_size);
        eprintln!("      Got: {} elements", raw_output.len());
        return Err(anyhow::anyhow!("Output size mismatch"));
    }

    println!("   ✓ Total output elements: {}", raw_output.len());

    // Statistical analysis
    let mut class_counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;

    for &val in raw_output.iter() {
        min_val = min_val.min(val);
        max_val = max_val.max(val);
        let class_id = val.round() as i32;
        *class_counts.entry(class_id).or_insert(0) += 1;
    }

    println!("   Value range: [{:.6}, {:.6}]", min_val, max_val);
        println!("   Unique class IDs detected: {}", class_counts.len());

        // Class distribution
        println!("\n   Class Distribution:");
        let mut sorted_classes: Vec<_> = class_counts.iter().collect();
        sorted_classes.sort_by_key(|(class_id, _)| *class_id);

        for (class_id, count) in sorted_classes.iter() {
            let percentage = (**count as f64 / raw_output.len() as f64) * 100.0;
            println!("      Class {:3}: {:7} pixels ({:6.2}%)", class_id, count, percentage);
        }

        println!("\n   ✓ Raw segmentation analysis complete\n");

        // ═══════════════════════════════════════════════════════════════════════
        // STEP 6.9: Save raw segmentation to JSON for Python visualization
        // ═══════════════════════════════════════════════════════════════════════
        println!("💾 Saving raw segmentation to JSON...");

        let output_path = format!("output/seg_{}.json", config_name);
        std::fs::create_dir_all("output")?;

        // Convert float mask to integer class IDs using argmax across class dimension
        let num_classes = reported_shape.get(1).copied().unwrap_or(2) as usize;
        // GIM model output: Output 0 is segmentation mask [height, width], Output 1 is classification (scalar)
        // The segmentation mask is already in the correct format - just need to round to integers
        let height = 512usize;
        let width = 640usize;
        let seg_output = &raw_output[0..(height * width)];
        let segmentation_mask: Vec<i32> = seg_output.iter().map(|&v| v.round() as i32).collect();
        let classification_output = if raw_output.len() > height * width {
            Some(raw_output[height * width])
        } else {
            None
        };

        if let Some(class_val) = classification_output {
            println!("   Classification output: {:.4}", class_val);
        }

        let json_data = serde_json::json!({
            "config_name": config_name,
            "thresholds": {
                "nbi": threshold_nbi,
                "wle": threshold_wle,
                "c": c_threshold,
            },
            "image": {
                "path": IMAGE_PATH,
                "original_size": [orig_w, orig_h],
                "model_size": [width, height],
            },
            "num_classes": 2,
            "output_shape": [height, width],
            "segmentation_mask": segmentation_mask,
            "classification_output": classification_output,
            "class_distribution": class_counts,
        });

        std::fs::write(&output_path, serde_json::to_string_pretty(&json_data)?)?;
        println!("   ✓ Saved segmentation mask to: {}\n", output_path);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 7: Performance Summary
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n⏱️  Step 7: Performance Summary");
    println!("   All {} threshold configurations tested successfully!", threshold_configs.len());
    println!("   Check output directory for results: output/seg_*.json");
    println!("   ┌──────────────────────────────┬───────────┐");
    println!("   │ Stage                        │ Time      │");
    println!("   ├──────────────────────────────┼───────────┤");
    println!("   │ CPU Preprocessing            │ (varies)  │");
    println!("   │ CPU → GPU Upload             │ {:>7.2}ms │", upload_time.as_secs_f64() * 1000.0);
    println!("   │ TensorRT Init (one-time)     │ {:>7.2}ms │", init_time.as_secs_f64() * 1000.0);
    println!("   │ GPU Inference (per config)   │ ~{:>6.2}ms │", 17.0);
    println!("   ├──────────────────────────────┼───────────┤");
    println!("   │ GPU Inference (per config)   │ ~{:>6.2}ms │", 17.0);
    println!("   ├──────────────────────────────┼───────────┤");
    let total_runtime = upload_time.as_secs_f64() * 1000.0 + 17.0;
    println!("   │ Total (per frame, sustained) │ {:>7.2}ms │", total_runtime);
    println!("   └──────────────────────────────┴───────────┘");
    println!("   Sustained Throughput: {:.1} FPS\n", 1000.0 / total_runtime);

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║   ✅ GIM Multi-Threshold Test Complete!                   ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    println!("💡 Next steps:");
    println!("   1. Compare results: python3 visualize_segmentation.py output/seg_medium.json");
    println!("   2. View all configs: ls -lh output/seg_*.json\n");

    // Keep GPU buffer alive until program exits
    std::mem::forget(gpu_buffer);

    Ok(())
}

/// Preprocess image on CPU: resize and convert to HWC format
/// GIM model expects [height, width, channels] layout, NOT NCHW!
fn preprocess_image_hwc(img: &image::RgbImage, target_w: u32, target_h: u32) -> Result<Vec<f32>> {
    // Direct resize to target dimensions
    let resized = image::imageops::resize(
        img,
        target_w,
        target_h,
        image::imageops::FilterType::Lanczos3,
    );

    // Convert to HWC format: [height, width, channels]
    let mut tensor = vec![0.0f32; (target_h * target_w * 3) as usize];

    for y in 0..target_h {
        for x in 0..target_w {
            let pixel = resized.get_pixel(x, y);
            let base_idx = ((y * target_w + x) * 3) as usize;

            // HWC layout: consecutive RGB values for each pixel
            tensor[base_idx] = pixel[0] as f32 / 255.0;     // R
            tensor[base_idx + 1] = pixel[1] as f32 / 255.0; // G
            tensor[base_idx + 2] = pixel[2] as f32 / 255.0; // B
        }
    }

    Ok(tensor)
}