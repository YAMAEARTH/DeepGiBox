// Exact replica of your old inference code to compare timing

use ort::{
    execution_providers::TensorRTExecutionProvider,
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::Session,
    value::Tensor,
    Result,
};
use std::path::Path;
use image::io::Reader as ImageReader;

const INPUT_WIDTH: u32 = 512;
const INPUT_HEIGHT: u32 = 512;
const INPUT_SHAPE: [usize; 4] = [1, 3, INPUT_HEIGHT as usize, INPUT_WIDTH as usize];

fn main() -> Result<()> {
    // Initialize environment (matches your old code exactly)
    let _ = ort::init()
        .with_name("tensorrt_iobinding")
        .with_execution_providers([
            TensorRTExecutionProvider::default()
                .with_device_id(0)
                .with_fp16(true)
                .with_engine_cache(true)
                .with_engine_cache_path("./trt_cache")
                .build(),
        ])
        .commit();

    let model_path = Path::new("apps/playgrounds/YOLOv5.onnx");
    
    // Create session
    let mut session = Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .commit_from_file(model_path)?;

    let cpu_allocator = Allocator::new(
        &session,
        MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::Default)?,
    )?;

    let gpu_allocator = Allocator::new(
        &session,
        MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?,
    )?;

    // Load and preprocess image
    println!("Loading image...");
    let img = ImageReader::open("apps/playgrounds/sample_img.jpg")
        .map_err(|e| ort::Error::new(format!("{}", e)))?
        .decode()
        .map_err(|e| ort::Error::new(format!("{}", e)))?
        .to_rgb8();
    let preprocessed = preprocess(&img);
    
    // Prepare input tensor (matches your old prepare_input exactly)
    println!("Preparing input tensor...");
    let mut input_tensor = Tensor::<f32>::new(&gpu_allocator, INPUT_SHAPE)?;
    
    if input_tensor.memory_info().is_cpu_accessible() {
        println!("GPU tensor is CPU-accessible - direct copy");
        let (_, tensor_data) = input_tensor.try_extract_tensor_mut::<f32>()?;
        tensor_data.copy_from_slice(&preprocessed);
    } else {
        println!("GPU tensor NOT CPU-accessible - using staging");
        let mut staging_tensor = Tensor::<f32>::new(&Allocator::default(), INPUT_SHAPE)?;
        let (_, staging_data) = staging_tensor.try_extract_tensor_mut::<f32>()?;
        staging_data.copy_from_slice(&preprocessed);
        staging_tensor.copy_into(&mut input_tensor)?;
    }
    
    // Run inference (matches your old run exactly)
    println!("\n=== Running Inference (10 iterations) ===");
    let mut timings = Vec::new();
    
    for i in 0..10 {
        let mut io_binding = session.create_binding()?;
        io_binding.bind_input("images", &input_tensor)?;
        io_binding.bind_output_to_device("output", &cpu_allocator.memory_info())?;

        let start = std::time::Instant::now();
        let _outputs = session.run_binding(&io_binding)?;
        let duration = start.elapsed();
        
        timings.push(duration.as_secs_f64() * 1000.0);
        println!("Iteration {}: {:.2}ms", i + 1, duration.as_secs_f64() * 1000.0);
    }
    
    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("\n=== Statistics ===");
    println!("Min: {:.2}ms", timings[0]);
    println!("Median: {:.2}ms", timings[5]);
    println!("Max: {:.2}ms", timings[9]);
    println!("Average: {:.2}ms", timings.iter().sum::<f64>() / timings.len() as f64);
    
    Ok(())
}

fn preprocess(img: &image::RgbImage) -> Vec<f32> {
    let (orig_w, orig_h) = img.dimensions();
    let scale = (INPUT_WIDTH as f32 / orig_w as f32).min(INPUT_HEIGHT as f32 / orig_h as f32);
    let new_w = (orig_w as f32 * scale) as u32;
    let new_h = (orig_h as f32 * scale) as u32;
    
    let resized = image::imageops::resize(img, new_w, new_h, image::imageops::FilterType::Lanczos3);
    let mut letterboxed = image::RgbImage::from_pixel(INPUT_WIDTH, INPUT_HEIGHT, image::Rgb([114, 114, 114]));
    
    let pad_x = (INPUT_WIDTH - new_w) / 2;
    let pad_y = (INPUT_HEIGHT - new_h) / 2;
    image::imageops::overlay(&mut letterboxed, &resized, pad_x as i64, pad_y as i64);
    
    let mut normalized = vec![0.0f32; (INPUT_WIDTH * INPUT_HEIGHT * 3) as usize];
    let hw = (INPUT_WIDTH * INPUT_HEIGHT) as usize;
    
    for y in 0..INPUT_HEIGHT {
        for x in 0..INPUT_WIDTH {
            let pixel = letterboxed.get_pixel(x, y);
            let idx = (y * INPUT_WIDTH + x) as usize;
            normalized[idx] = pixel[0] as f32 / 255.0;
            normalized[hw + idx] = pixel[1] as f32 / 255.0;
            normalized[2 * hw + idx] = pixel[2] as f32 / 255.0;
        }
    }
    
    normalized
}
