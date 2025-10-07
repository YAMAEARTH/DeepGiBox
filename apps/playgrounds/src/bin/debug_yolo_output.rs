use anyhow::Result;
use common_io::{Stage, TensorInputPacket, TensorDesc, MemRef, FrameMeta, DType, MemLoc, PixelFormat, ColorSpace};
use image::io::Reader as ImageReader;
use inference::InferenceEngine;
use std::path::Path;

const MODEL_PATH: &str = "apps/playgrounds/YOLOv5.onnx";
const IMAGE_PATH: &str = "apps/playgrounds/sample_img.jpg";
const INPUT_WIDTH: u32 = 512;
const INPUT_HEIGHT: u32 = 512;

fn main() -> Result<()> {
    println!("=== YOLO Output Debug ===");
    
    if !Path::new(MODEL_PATH).exists() {
        anyhow::bail!("Model file not found at {}", MODEL_PATH);
    }
    
    if !Path::new(IMAGE_PATH).exists() {
        anyhow::bail!("Image file not found at {}", IMAGE_PATH);
    }
    
    // Preprocess
    let preprocessed = mock_preprocess_image(IMAGE_PATH)?;
    let tensor_packet = create_tensor_packet(preprocessed)?;
    
    // Run inference
    let mut inference_engine = InferenceEngine::new(MODEL_PATH)?;
    let raw_detections = inference_engine.process(tensor_packet);
    
    let predictions = &raw_detections.raw_output;
    
    // YOLO output format: [cx, cy, w, h, objectness, class0, class1]
    // Total: 5 + 2 = 7 values per detection
    // Number of detections: 112896 / 7 = 16128 anchors
    
    let num_classes = 2;
    let stride = 5 + num_classes; // 7
    let num_detections = predictions.len() / stride;
    
    println!("Total prediction values: {}", predictions.len());
    println!("Stride (values per detection): {}", stride);
    println!("Number of detection anchors: {}", num_detections);
    println!();
    
    // Examine first 5 detections
    println!("=== First 5 Raw Detections ===");
    for i in 0..5.min(num_detections) {
        let offset = i * stride;
        let chunk = &predictions[offset..offset + stride];
        
        println!("\nDetection {}:", i + 1);
        println!("  cx={:.3}, cy={:.3}, w={:.3}, h={:.3}", 
                 chunk[0], chunk[1], chunk[2], chunk[3]);
        println!("  objectness (raw)={:.6}", chunk[4]);
        println!("  objectness (sigmoid)={:.6}", sigmoid(chunk[4]));
        
        // Find best class
        let class_scores = &chunk[5..];
        let (best_idx, best_raw) = class_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        
        println!("  best_class={}, raw_score={:.6}, sigmoid={:.6}", 
                 best_idx, best_raw, sigmoid(*best_raw));
        
        // Combined score
        let obj_sig = sigmoid(chunk[4]);
        let cls_sig = sigmoid(*best_raw);
        println!("  combined_score (obj*cls)={:.6}", obj_sig * cls_sig);
        
        // Show top 3 class scores
        let mut class_vec: Vec<(usize, f32)> = class_scores
            .iter()
            .enumerate()
            .map(|(idx, &score)| (idx, score))
            .collect();
        class_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        println!("  top 3 classes:");
        for (idx, score) in class_vec.iter().take(3) {
            println!("    class_{}: raw={:.6}, sigmoid={:.6}", idx, score, sigmoid(*score));
        }
    }
    
    println!();
    println!("=== Analysis ===");
    println!("If raw values are mostly in [0, 1], the model outputs are already activated");
    println!("If raw values span large negative/positive range, they need sigmoid activation");
    println!();
    
    // Statistics
    let mut obj_values: Vec<f32> = Vec::new();
    let mut cls_values: Vec<f32> = Vec::new();
    
    for i in 0..num_detections.min(100) {
        let offset = i * stride;
        let chunk = &predictions[offset..offset + stride];
        obj_values.push(chunk[4]);
        cls_values.extend_from_slice(&chunk[5..]);
    }
    
    obj_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cls_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    println!("Objectness stats (first 100 detections):");
    println!("  min={:.6}, max={:.6}", obj_values.first().unwrap(), obj_values.last().unwrap());
    println!("  median={:.6}", obj_values[obj_values.len()/2]);
    
    println!("\nClass score stats (first 100 detections):");
    println!("  min={:.6}, max={:.6}", cls_values.first().unwrap(), cls_values.last().unwrap());
    println!("  median={:.6}", cls_values[cls_values.len()/2]);
    
    Ok(())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn mock_preprocess_image(image_path: &str) -> Result<Vec<f32>> {
    let img = ImageReader::open(image_path)?.decode()?.to_rgb8();
    let (orig_w, orig_h) = img.dimensions();
    
    let scale = (INPUT_WIDTH as f32 / orig_w as f32).min(INPUT_HEIGHT as f32 / orig_h as f32);
    let new_w = (orig_w as f32 * scale) as u32;
    let new_h = (orig_h as f32 * scale) as u32;
    
    let resized = image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::Lanczos3);
    let mut letterboxed = image::RgbImage::from_pixel(INPUT_WIDTH, INPUT_HEIGHT, image::Rgb([114u8, 114u8, 114u8]));
    let pad_x = (INPUT_WIDTH - new_w) / 2;
    let pad_y = (INPUT_HEIGHT - new_h) / 2;
    image::imageops::overlay(&mut letterboxed, &resized, pad_x as i64, pad_y as i64);
    
    let mut tensor = vec![0.0f32; (INPUT_WIDTH * INPUT_HEIGHT * 3) as usize];
    let hw = (INPUT_WIDTH * INPUT_HEIGHT) as usize;
    
    for y in 0..INPUT_HEIGHT {
        for x in 0..INPUT_WIDTH {
            let pixel = letterboxed.get_pixel(x, y);
            let idx = (y * INPUT_WIDTH + x) as usize;
            tensor[idx] = pixel[0] as f32 / 255.0;
            tensor[hw + idx] = pixel[1] as f32 / 255.0;
            tensor[2 * hw + idx] = pixel[2] as f32 / 255.0;
        }
    }
    
    Ok(tensor)
}

fn create_tensor_packet(data: Vec<f32>) -> Result<TensorInputPacket> {
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
    
    let tensor_desc = TensorDesc {
        n: 1,
        c: 3,
        h: INPUT_HEIGHT,
        w: INPUT_WIDTH,
        dtype: DType::Fp32,
        device: 0,
    };
    
    let data_boxed = Box::new(data);
    let ptr = Box::leak(data_boxed).as_mut_ptr() as *mut u8;
    let len = (INPUT_WIDTH * INPUT_HEIGHT * 3 * std::mem::size_of::<f32>() as u32) as usize;
    
    let mem_ref = MemRef {
        ptr,
        len,
        stride: (INPUT_WIDTH * 3 * std::mem::size_of::<f32>() as u32) as usize,
        loc: MemLoc::Cpu,
    };
    
    Ok(TensorInputPacket {
        from: frame_meta,
        desc: tensor_desc,
        data: mem_ref,
    })
}
