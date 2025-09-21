use image::{imageops::FilterType, io::Reader as ImageReader};
use ort::{
    execution_providers::TensorRTExecutionProvider,
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::Session,
    value::Tensor,
    Result,
};
use std::{path::Path, time::Instant};

const INPUT_SHAPE: [usize; 4] = [1, 3, 512, 512];
const INPUT_WIDTH: u32 = 512;
const INPUT_HEIGHT: u32 = 512;

fn load_image_as_chw(image_path: &Path) -> Result<Vec<f32>> {
    let image = ImageReader::open(image_path)
        .map_err(ort::Error::wrap)?
        .decode()
        .map_err(ort::Error::wrap)?;
    let resized = image
        .resize_exact(INPUT_WIDTH, INPUT_HEIGHT, FilterType::Triangle)
        .to_rgb8();

    let (width, height) = resized.dimensions();
    let plane_size = (width * height) as usize;
    let mut data = vec![0.0f32; 3 * plane_size];

    for y in 0..height {
        for x in 0..width {
            let pixel = resized.get_pixel(x, y);
            let idx = (y * width + x) as usize;
            data[idx] = pixel[0] as f32 / 255.0;
            data[idx + plane_size] = pixel[1] as f32 / 255.0;
            data[idx + plane_size * 2] = pixel[2] as f32 / 255.0;
        }
    }

    Ok(data)
}

fn prepare_tensors(
    input_allocator: &Allocator,
    cpu_allocator: &Allocator,
    image_path: &Path,
) -> Result<(Tensor<f32>, Tensor<f32>)> {
    let mut input_tensor = Tensor::<f32>::new(input_allocator, INPUT_SHAPE)?;
    let output_tensor = Tensor::<f32>::new(cpu_allocator, [1_usize, 16128, 7])?;

    let image_data = load_image_as_chw(image_path)?;

    if input_tensor.memory_info().is_cpu_accessible() {
        let (_, tensor_data) = input_tensor.try_extract_tensor_mut::<f32>()?;
        tensor_data.copy_from_slice(&image_data);
    } else {
        let mut staging_tensor = Tensor::<f32>::new(&Allocator::default(), INPUT_SHAPE)?;
        let (_, staging_data) = staging_tensor.try_extract_tensor_mut::<f32>()?;
        staging_data.copy_from_slice(&image_data);
        staging_tensor.copy_into(&mut input_tensor)?;
    }

    println!("Loaded input tensor from {}", image_path.display());

    Ok((input_tensor, output_tensor))
}

fn run_inference(
    session: &mut Session,
    input_tensor: &Tensor<f32>,
    output_tensor: Tensor<f32>,
) -> Result<()> {
    let mut io_binding = session.create_binding()?;
    io_binding.bind_input("images", input_tensor)?;
    io_binding.bind_output("output", output_tensor)?;

    let start = Instant::now();
    let outputs = session.run_binding(&io_binding)?;
    let duration = start.elapsed();

    println!("Inference duration: {:?}", duration);
    println!("Inference done with IOBinding!");
    println!("Output tensors count: {}", outputs.len());

    match outputs[0].try_extract_tensor::<f32>() {
        Ok((shape, data)) => {
            println!("✓ Output ready on CPU!");
            println!("Output shape: {:?}", shape);
            println!("Total elements: {}", data.len());

            println!("First 10 output values:");
            for (i, val) in data.iter().take(10).enumerate() {
                println!("  output[{i}] = {val}");
            }

            if shape.len() >= 3 {
                let batch_size = shape[0];
                let num_detections = shape[1];
                let values_per_detection = shape[2];

                println!("\nYOLO Output Analysis:");
                println!("  Batch size: {}", batch_size);
                println!("  Number of detections: {}", num_detections);
                println!("  Values per detection: {}", values_per_detection);

                println!("\nFirst 5 detections:");
                for detection_idx in 0..5.min(num_detections as usize) {
                    let start_idx = detection_idx * values_per_detection as usize;
                    let end_idx = start_idx + values_per_detection as usize;
                    let detection = &data[start_idx..end_idx];
                    println!("  Detection {}: {:?}", detection_idx, detection);
                }
            }
        }
        Err(e) => {
            println!("Error extracting tensor data: {:?}", e);
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    let _env = ort::init()
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

    let model_path = "./YOLOv5.onnx";
    let mut session = Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .commit_from_file(model_path)?;
    println!("Session ready with TensorRT!");

    let input_allocator = Allocator::new(
        &session,
        MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?,
    )?;
    let cpu_allocator = Allocator::new(
        &session,
        MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::Default)?,
    )?;

    let image_path = Path::new("/home/kot/Documents/pun/ort_binding/cat.jpg");
    let (input_tensor, output_tensor) = prepare_tensors(&input_allocator, &cpu_allocator, image_path)?;
    run_inference(&mut session, &input_tensor, output_tensor)?;

    Ok(())
}
