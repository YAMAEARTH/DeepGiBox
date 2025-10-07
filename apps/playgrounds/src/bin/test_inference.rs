use anyhow::Result;
use common_io::{
    ColorSpace, DType, FrameMeta, MemLoc, MemRef, PixelFormat, Stage, TensorDesc, TensorInputPacket,
};
use image::io::Reader as ImageReader;
use inference::InferenceEngine;
use std::path::Path;

const MODEL_PATH: &str = "apps/playgrounds/YOLOv5.onnx";
const IMAGE_PATH: &str = "apps/playgrounds/sample_img.jpg";
const INPUT_WIDTH: u32 = 512;
const INPUT_HEIGHT: u32 = 512;

fn main() -> Result<()> {
    println!("=== Inference Engine Test ===");
    println!("Model: {}", MODEL_PATH);
    println!("Image: {}", IMAGE_PATH);
    println!();

    // Check if model exists
    if !Path::new(MODEL_PATH).exists() {
        eprintln!("❌ Error: Model file not found at {}", MODEL_PATH);
        eprintln!();
        eprintln!("Please provide a YOLO ONNX model:");
        eprintln!("  1. Export your YOLO model to ONNX format");
        eprintln!("  2. Place it at: {}", MODEL_PATH);
        eprintln!(
            "  3. Ensure input shape is 1x3x{}x{}",
            INPUT_HEIGHT, INPUT_WIDTH
        );
        eprintln!();
        eprintln!("For testing purposes, you can:");
        eprintln!("  - Use YOLOv5: https://github.com/ultralytics/yolov5");
        eprintln!(
            "  - Export with: python export.py --weights yolov5s.pt --include onnx --imgsz 512"
        );
        anyhow::bail!("Model file not found");
    }

    // Check if image exists
    if !Path::new(IMAGE_PATH).exists() {
        anyhow::bail!("Image file not found at {}", IMAGE_PATH);
    }

    // Step 1: Load and preprocess image (mock preprocessing)
    println!("Step 1: Loading and preprocessing image...");
    let preprocessed = mock_preprocess_image(IMAGE_PATH)?;
    println!(
        "  ✓ Image loaded and preprocessed to {}x{}",
        INPUT_WIDTH, INPUT_HEIGHT
    );
    println!("  ✓ Tensor shape: 1x3x{}x{}", INPUT_HEIGHT, INPUT_WIDTH);
    println!("  ✓ Total elements: {}", preprocessed.len());
    println!();

    // Step 2: Create TensorInputPacket
    println!("Step 2: Creating TensorInputPacket...");
    let tensor_packet = create_tensor_packet(preprocessed)?;
    println!("  ✓ TensorInputPacket created");
    println!(
        "  ✓ Shape: {}x{}x{}x{}",
        tensor_packet.desc.n, tensor_packet.desc.c, tensor_packet.desc.h, tensor_packet.desc.w
    );
    println!("  ✓ DType: {:?}", tensor_packet.desc.dtype);
    println!("  ✓ Device: GPU {}", tensor_packet.desc.device);
    println!();

    // Step 3: Initialize Inference Engine
    println!("Step 3: Initializing Inference Engine...");
    let mut engine = InferenceEngine::new(MODEL_PATH)?;
    println!("  ✓ Engine initialized with TensorRT");
    println!("  ✓ FP16 optimization enabled");
    println!("  ✓ Engine cache: ./trt_cache/");
    println!();

    // Step 4: Run Inference
    println!("Step 4: Running inference...");
    let start = std::time::Instant::now();
    let output = engine.process(tensor_packet);
    let duration = start.elapsed();

    println!(
        "  ✓ Inference completed in {:.2}ms",
        duration.as_secs_f64() * 1000.0
    );
    println!("  ✓ Frame idx: {}", output.from.frame_idx);
    println!("  ✓ Output ready for postprocessing");
    println!();

    println!("=== Test Completed Successfully! ===");
    println!();
    println!("Next steps:");
    println!("  - Raw detections are ready for postprocess stage");
    println!("  - Postprocess will decode boxes, apply NMS, etc.");
    println!("  - Pipeline: Preprocess → Inference → Postprocess → Overlay");

    Ok(())
}

/// Mock preprocessing: Load image, resize, normalize to FP32 tensor
fn mock_preprocess_image(image_path: &str) -> Result<Vec<f32>> {
    let img = ImageReader::open(image_path)?.decode()?.to_rgb8();
    let (orig_w, orig_h) = img.dimensions();

    println!("  Original image: {}x{}", orig_w, orig_h);

    // Resize with letterboxing
    let scale = (INPUT_WIDTH as f32 / orig_w as f32).min(INPUT_HEIGHT as f32 / orig_h as f32);
    let new_w = (orig_w as f32 * scale) as u32;
    let new_h = (orig_h as f32 * scale) as u32;

    let resized =
        image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::Lanczos3);

    // Create letterboxed image (gray background)
    let mut letterboxed =
        image::RgbImage::from_pixel(INPUT_WIDTH, INPUT_HEIGHT, image::Rgb([114u8, 114u8, 114u8]));

    let pad_x = (INPUT_WIDTH - new_w) / 2;
    let pad_y = (INPUT_HEIGHT - new_h) / 2;

    // Overlay resized image
    image::imageops::overlay(&mut letterboxed, &resized, pad_x as i64, pad_y as i64);

    // Convert to NCHW format and normalize to [0, 1]
    let mut tensor = vec![0.0f32; (INPUT_WIDTH * INPUT_HEIGHT * 3) as usize];
    let hw = (INPUT_WIDTH * INPUT_HEIGHT) as usize;

    for y in 0..INPUT_HEIGHT {
        for x in 0..INPUT_WIDTH {
            let pixel = letterboxed.get_pixel(x, y);
            let idx = (y * INPUT_WIDTH + x) as usize;

            // NCHW: [C, H, W]
            tensor[idx] = pixel[0] as f32 / 255.0; // R channel
            tensor[hw + idx] = pixel[1] as f32 / 255.0; // G channel
            tensor[2 * hw + idx] = pixel[2] as f32 / 255.0; // B channel
        }
    }

    println!(
        "  Letterbox scale: {:.3}, pad: ({}, {})",
        scale, pad_x, pad_y
    );

    Ok(tensor)
}

/// Create TensorInputPacket from preprocessed data
fn create_tensor_packet(data: Vec<f32>) -> Result<TensorInputPacket> {
    // Create mock frame metadata
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
    };

    // Create tensor descriptor
    let tensor_desc = TensorDesc {
        n: 1,
        c: 3,
        h: INPUT_HEIGHT,
        w: INPUT_WIDTH,
        dtype: DType::Fp32,
        device: 0,
    };

    // Leak the data to get a stable pointer (in real pipeline, this is managed by CUDA)
    let data_boxed = Box::new(data);
    let ptr = Box::leak(data_boxed).as_mut_ptr() as *mut u8;
    let len = (INPUT_WIDTH * INPUT_HEIGHT * 3 * std::mem::size_of::<f32>() as u32) as usize;

    // Create memory reference
    let mem_ref = MemRef {
        ptr,
        len,
        stride: (INPUT_WIDTH * 3 * std::mem::size_of::<f32>() as u32) as usize,
        loc: MemLoc::Gpu { device: 0 }, // Mock as GPU memory
    };

    Ok(TensorInputPacket {
        from: frame_meta,
        desc: tensor_desc,
        data: mem_ref,
    })
}
