// infer_ort_trt_preview.rs - Preview inference engine with ORT TensorRT EP
// Following inference_guideline.md specification

use anyhow::Result;
use common_io::{
    ColorSpace, DType, FrameMeta, MemLoc, MemRef, PixelFormat, Stage, TensorDesc, TensorInputPacket,
};
use cudarc::driver::DevicePtr;

fn create_dummy_input_gpu(device: u32) -> Result<TensorInputPacket> {
    let dummy_meta = FrameMeta {
        source_id: 0,
        frame_idx: 0,
        width: 512,
        height: 512,
        pixfmt: PixelFormat::RGB8,
        colorspace: ColorSpace::BT709,
        pts_ns: 0,
        t_capture_ns: 0,
        stride_bytes: 512 * 3,
    };

    let shape = [1_usize, 3, 512, 512];
    let total_elements: usize = shape.iter().product();
    let element_size = 2; // FP16
    let total_bytes = total_elements * element_size;

    // Allocate GPU memory for dummy input
    let cuda_device = cudarc::driver::CudaDevice::new(device as usize)?;
    let dummy_buffer = cuda_device.alloc_zeros::<u8>(total_bytes)?;
    let gpu_ptr = *dummy_buffer.device_ptr() as *mut u8;

    // Leak buffer so it persists
    std::mem::forget(dummy_buffer);

    Ok(TensorInputPacket {
        from: dummy_meta,
        desc: TensorDesc {
            n: 1,
            c: 3,
            h: 512,
            w: 512,
            dtype: DType::Fp16,
            device,
        },
        data: MemRef {
            ptr: gpu_ptr,
            len: total_bytes,
            stride: 512 * 3,
            loc: MemLoc::Gpu { device },
        },
    })
}

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║     ORT TensorRT Inference Engine Preview               ║");
    println!("║     Following inference_guideline.md specification       ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let cfg_path = "configs/inference_test.toml";

    println!("📝 Loading config from: {}", cfg_path);
    let mut engine = inference::from_path(cfg_path)?;

    println!("\n✅ Inference Engine Ready!");
    println!("  Input:  {}", engine.input_name());
    println!("  Outputs: {:?}", engine.output_names());

    println!("\n🎯 Running single inference pass...");
    let tin = create_dummy_input_gpu(0)?;

    let result = telemetry::time_stage("inference_preview", &mut engine, tin);

    println!("\n📊 Results:");
    println!("  Raw output size: {} values", result.raw_output.len());
    println!("  Output shape:    {:?}", result.output_shape);

    if !result.raw_output.is_empty() {
        let min = result
            .raw_output
            .iter()
            .fold(f32::INFINITY, |a, &b| a.min(b));
        let max = result
            .raw_output
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mean = result.raw_output.iter().sum::<f32>() / result.raw_output.len() as f32;

        println!("  Min value:       {:.6}", min);
        println!("  Max value:       {:.6}", max);
        println!("  Mean value:      {:.6}", mean);
    }

    println!("\n✅ Preview complete!");

    Ok(())
}
