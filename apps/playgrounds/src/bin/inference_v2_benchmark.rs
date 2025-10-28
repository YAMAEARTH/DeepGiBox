use anyhow::Result;
use common_io::{
    ColorSpace, DType, FrameMeta, MemLoc, MemRef, PixelFormat, RawDetectionsPacket, Stage,
    TensorDesc, TensorInputPacket,
};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use image::io::Reader as ImageReader;
use inference_v2::TrtInferenceStage;

const ENGINE_PATH: &str = "trt-shim/test_rust/assets/v7_optimized_YOLOv5.engine";
const LIB_PATH: &str = "trt-shim/build/libtrt_shim.so";
const IMAGE_PATH: &str = "apps/playgrounds/sample_img.jpg";
const INPUT_WIDTH: u32 = 512;
const INPUT_HEIGHT: u32 = 512;

// Number of iterations for benchmarking
const WARMUP_RUNS: usize = 5;
const BENCHMARK_RUNS: usize = 100;

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║   Inference V2 - Multi-Run Benchmark                     ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    println!("📋 Configuration:");
    println!("   Engine:  {}", ENGINE_PATH);
    println!("   Library: {}", LIB_PATH);
    println!("   Image:   {}", IMAGE_PATH);
    println!("   Size:    {}x{}", INPUT_WIDTH, INPUT_HEIGHT);
    println!("   Warmup:  {} runs", WARMUP_RUNS);
    println!("   Bench:   {} runs\n", BENCHMARK_RUNS);

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

    println!("   Preprocessing to {}x{}...", INPUT_WIDTH, INPUT_HEIGHT);
    let preprocessed = preprocess_image_cpu(&img, INPUT_WIDTH, INPUT_HEIGHT)?;
    println!("   ✓ Preprocessed to NCHW format (FP32)\n");

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 3: Upload to GPU Memory (once)
    // ═══════════════════════════════════════════════════════════════════════
    println!("📤 Step 3: Uploading data to GPU...");
    let gpu_buffer: CudaSlice<f32> = cuda_device.htod_sync_copy(&preprocessed)?;
    let gpu_ptr = *gpu_buffer.device_ptr() as *mut u8;
    println!("   ✓ Uploaded to GPU pointer: {:#x}\n", gpu_ptr as usize);

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 4: Create TensorInputPacket (reused for all runs)
    // ═══════════════════════════════════════════════════════════════════════
    println!("📦 Step 4: Creating TensorInputPacket...");
    let frame_meta = FrameMeta {
        source_id: 0,
        width: INPUT_WIDTH,
        height: INPUT_HEIGHT,
        pixfmt: PixelFormat::BGRA8,
        colorspace: ColorSpace::BT709,
        frame_idx: 1,
        pts_ns: 0,
        t_capture_ns: 0,
        stride_bytes: INPUT_WIDTH * 4,
            crop_region: None,
    };

    let tensor_desc = TensorDesc {
        n: 1,
        c: 3,
        h: INPUT_HEIGHT,
        w: INPUT_WIDTH,
        dtype: DType::Fp32,
        device: 0,
    };

    let tensor_packet = TensorInputPacket {
        from: frame_meta,
        desc: tensor_desc,
        data: MemRef {
            ptr: gpu_ptr,
            len: preprocessed.len() * std::mem::size_of::<f32>(),
            stride: INPUT_WIDTH as usize * 3 * std::mem::size_of::<f32>(),
            loc: MemLoc::Gpu { device: 0 },
        },
    };
    println!("   ✓ TensorInputPacket created\n");

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 5: Initialize TensorRT Inference Stage
    // ═══════════════════════════════════════════════════════════════════════
    println!("🔧 Step 5: Initializing TensorRT Inference Stage...");
    let start = std::time::Instant::now();
    let mut inference_stage = TrtInferenceStage::new(ENGINE_PATH, LIB_PATH)
        .map_err(|e| anyhow::anyhow!("Failed to create TensorRT inference stage: {}", e))?;
    let init_time = start.elapsed();
    println!("   ✓ TrtInferenceStage created");
    println!(
        "   ✓ Initialization time: {:.2}ms\n",
        init_time.as_secs_f64() * 1000.0
    );

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 6: Warmup Runs (exclude from benchmark)
    // ═══════════════════════════════════════════════════════════════════════
    println!("🔥 Step 6: Warmup Phase ({} runs)...", WARMUP_RUNS);
    let mut warmup_times = Vec::new();

    for i in 0..WARMUP_RUNS {
        let start = std::time::Instant::now();
        let _detections: RawDetectionsPacket = inference_stage.process(tensor_packet.clone());
        let elapsed = start.elapsed();
        warmup_times.push(elapsed.as_secs_f64() * 1000.0);

        print!(
            "   Run {}/{}: {:.2}ms",
            i + 1,
            WARMUP_RUNS,
            elapsed.as_secs_f64() * 1000.0
        );
        if i == 0 {
            print!(" ← FIRST RUN (may include cold start)");
        }
        println!();
    }

    let warmup_avg = warmup_times.iter().sum::<f64>() / warmup_times.len() as f64;
    let warmup_min = warmup_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let warmup_max = warmup_times.iter().fold(0.0f64, |a, &b| a.max(b));

    println!("   ✓ Warmup complete!");
    println!("   ✓ First run: {:.2}ms", warmup_times[0]);
    println!(
        "   ✓ Warmup avg: {:.2}ms (min: {:.2}ms, max: {:.2}ms)\n",
        warmup_avg, warmup_min, warmup_max
    );

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 7: Benchmark Runs (measure steady-state performance)
    // ═══════════════════════════════════════════════════════════════════════
    println!("⚡ Step 7: Benchmark Phase ({} runs)...", BENCHMARK_RUNS);
    let mut bench_times = Vec::new();

    let bench_start = std::time::Instant::now();
    for i in 0..BENCHMARK_RUNS {
        let start = std::time::Instant::now();
        let _detections: RawDetectionsPacket = inference_stage.process(tensor_packet.clone());
        let elapsed = start.elapsed();
        bench_times.push(elapsed.as_secs_f64() * 1000.0);

        if i < 5 || i >= BENCHMARK_RUNS - 5 {
            println!(
                "   Run {}/{}: {:.2}ms",
                i + 1,
                BENCHMARK_RUNS,
                elapsed.as_secs_f64() * 1000.0
            );
        } else if i == 5 {
            println!("   ... (runs 6-{}) ...", BENCHMARK_RUNS - 5);
        }
    }
    let bench_total = bench_start.elapsed();

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 8: Statistical Analysis
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n📊 Step 8: Statistical Analysis");

    // Sort for percentile calculations
    let mut sorted = bench_times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let bench_avg = bench_times.iter().sum::<f64>() / bench_times.len() as f64;
    let bench_min = sorted[0];
    let bench_max = sorted[sorted.len() - 1];
    let bench_p50 = sorted[sorted.len() / 2];
    let bench_p95 = sorted[(sorted.len() as f64 * 0.95) as usize];
    let bench_p99 = sorted[(sorted.len() as f64 * 0.99) as usize];

    // Calculate standard deviation
    let variance = bench_times
        .iter()
        .map(|&x| (x - bench_avg).powi(2))
        .sum::<f64>()
        / bench_times.len() as f64;
    let std_dev = variance.sqrt();

    println!("\n   🔥 COLD START ANALYSIS:");
    println!("   ┌─────────────────────────┬───────────┐");
    println!(
        "   │ First run (warmup)      │ {:>7.2}ms │",
        warmup_times[0]
    );
    println!("   │ Subsequent warmup runs  │ {:>7.2}ms │", warmup_avg);
    println!("   │ Benchmark steady-state  │ {:>7.2}ms │", bench_avg);
    println!(
        "   │ Overhead (cold start)   │ {:>7.2}ms │",
        warmup_times[0] - bench_avg
    );
    println!("   └─────────────────────────┴───────────┘");

    println!("\n   ⚡ BENCHMARK STATISTICS ({} runs):", BENCHMARK_RUNS);
    println!("   ┌─────────────────────────┬───────────┐");
    println!("   │ Average (mean)          │ {:>7.2}ms │", bench_avg);
    println!("   │ Median (p50)            │ {:>7.2}ms │", bench_p50);
    println!("   │ Minimum (best)          │ {:>7.2}ms │", bench_min);
    println!("   │ Maximum (worst)         │ {:>7.2}ms │", bench_max);
    println!("   │ 95th percentile         │ {:>7.2}ms │", bench_p95);
    println!("   │ 99th percentile         │ {:>7.2}ms │", bench_p99);
    println!("   │ Std deviation           │ {:>7.2}ms │", std_dev);
    println!("   └─────────────────────────┴───────────┘");

    println!("\n   📈 THROUGHPUT:");
    println!("   ┌─────────────────────────┬───────────┐");
    println!(
        "   │ Average FPS             │ {:>7.1}   │",
        1000.0 / bench_avg
    );
    println!(
        "   │ Peak FPS (min latency)  │ {:>7.1}   │",
        1000.0 / bench_min
    );
    println!(
        "   │ Total time ({} runs)   │ {:>7.2}ms │",
        BENCHMARK_RUNS,
        bench_total.as_secs_f64() * 1000.0
    );
    println!(
        "   │ Actual throughput       │ {:>7.1}   │",
        BENCHMARK_RUNS as f64 / bench_total.as_secs_f64()
    );
    println!("   └─────────────────────────┴───────────┘");

    println!("\n💡 VERDICT:");
    if (warmup_times[0] - bench_avg).abs() > 5.0 {
        println!(
            "   ⚠️  Cold start detected! First run was {:.2}ms slower",
            warmup_times[0] - bench_avg
        );
        println!(
            "   ✅ Steady-state performance: {:.2}ms ({:.1} FPS)",
            bench_avg,
            1000.0 / bench_avg
        );
    } else {
        println!("   ✅ No significant cold start overhead");
        println!(
            "   ✅ Consistent performance: {:.2}ms ({:.1} FPS)",
            bench_avg,
            1000.0 / bench_avg
        );
    }

    if std_dev > 1.0 {
        println!("   ⚠️  High variance detected ({:.2}ms std dev)", std_dev);
    } else {
        println!(
            "   ✅ Low variance ({:.2}ms std dev) - very stable!",
            std_dev
        );
    }

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║   ✅ Benchmark Completed Successfully!                    ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Keep GPU buffer alive
    std::mem::forget(gpu_buffer);

    Ok(())
}

/// Preprocess image on CPU: resize with letterboxing and convert to NCHW format
fn preprocess_image_cpu(img: &image::RgbImage, target_w: u32, target_h: u32) -> Result<Vec<f32>> {
    let (orig_w, orig_h) = img.dimensions();

    // Calculate letterbox scale (maintain aspect ratio)
    let scale = (target_w as f32 / orig_w as f32).min(target_h as f32 / orig_h as f32);
    let new_w = (orig_w as f32 * scale) as u32;
    let new_h = (orig_h as f32 * scale) as u32;

    // Resize image
    let resized = image::imageops::resize(img, new_w, new_h, image::imageops::FilterType::Lanczos3);

    // Create letterboxed image with gray padding
    let mut letterboxed =
        image::RgbImage::from_pixel(target_w, target_h, image::Rgb([114u8, 114u8, 114u8]));

    let pad_x = (target_w - new_w) / 2;
    let pad_y = (target_h - new_h) / 2;

    // Overlay resized image
    image::imageops::overlay(&mut letterboxed, &resized, pad_x as i64, pad_y as i64);

    // Convert to NCHW format and normalize to [0, 1]
    let mut tensor = vec![0.0f32; (target_w * target_h * 3) as usize];
    let hw = (target_w * target_h) as usize;

    for y in 0..target_h {
        for x in 0..target_w {
            let pixel = letterboxed.get_pixel(x, y);
            let idx = (y * target_w + x) as usize;

            // NCHW layout: [C, H, W]
            tensor[0 * hw + idx] = pixel[0] as f32 / 255.0; // R channel
            tensor[1 * hw + idx] = pixel[1] as f32 / 255.0; // G channel
            tensor[2 * hw + idx] = pixel[2] as f32 / 255.0; // B channel
        }
    }

    Ok(tensor)
}
