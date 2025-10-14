// dump_preprocessing_output.rs - Dump preprocessing output data for inspection
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::thread;
use std::time::Duration;

use common_io::{DType, MemLoc, RawFramePacket, Stage, TensorInputPacket};
use cudarc::driver::CudaDevice;
use decklink_input::capture::CaptureSession;
use preprocess_cuda::{ChromaOrder, Preprocessor};

fn dump_tensor_to_file(
    tensor: &TensorInputPacket,
    output_prefix: &str,
) -> Result<(), Box<dyn Error>> {
    println!("\n📁 Dumping tensor data to files...");

    let device = CudaDevice::new(tensor.desc.device as usize)?;
    let total_elements = (tensor.desc.n * tensor.desc.c * tensor.desc.h * tensor.desc.w) as usize;

    // Determine element size
    let element_size = match tensor.desc.dtype {
        DType::Fp16 => 2,
        DType::Fp32 => 4,
    };

    println!("  Total elements: {}", total_elements);
    println!("  Element size: {} bytes", element_size);
    println!("  Total bytes: {}", total_elements * element_size);

    // Copy GPU -> CPU
    match tensor.desc.dtype {
        DType::Fp32 => {
            println!("  Copying FP32 data from GPU to CPU...");
            let mut host_data = vec![0.0f32; total_elements];

            unsafe {
                use cudarc::driver::sys;
                let result = sys::lib().cuMemcpyDtoH_v2(
                    host_data.as_mut_ptr() as *mut std::ffi::c_void,
                    tensor.data.ptr as sys::CUdeviceptr,
                    total_elements * 4,
                );
                if result != sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("CUDA memcpy failed: {:?}", result).into());
                }
            }

            // Save as binary
            let bin_path = format!("{}_fp32.bin", output_prefix);
            let mut file = File::create(&bin_path)?;
            let bytes = unsafe {
                std::slice::from_raw_parts(host_data.as_ptr() as *const u8, total_elements * 4)
            };
            file.write_all(bytes)?;
            println!("  ✓ Saved binary: {}", bin_path);

            // Save as text (first 100 values per channel)
            let txt_path = format!("{}_sample.txt", output_prefix);
            let mut txt_file = File::create(&txt_path)?;

            writeln!(
                txt_file,
                "Tensor Shape: {}×{}×{}×{} (NCHW)",
                tensor.desc.n, tensor.desc.c, tensor.desc.h, tensor.desc.w
            )?;
            writeln!(txt_file, "Data Type: FP32")?;
            writeln!(txt_file, "Frame: #{}\n", tensor.from.frame_idx)?;

            let per_channel = (tensor.desc.h * tensor.desc.w) as usize;
            for ch in 0..3 {
                let ch_name = match ch {
                    0 => "R (Red)",
                    1 => "G (Green)",
                    2 => "B (Blue)",
                    _ => unreachable!(),
                };

                writeln!(txt_file, "=== Channel {} ({}) ===", ch, ch_name)?;

                let ch_offset = ch * per_channel;
                let ch_data = &host_data[ch_offset..ch_offset + per_channel];

                // Statistics
                let min = ch_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max = ch_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mean = ch_data.iter().sum::<f32>() / ch_data.len() as f32;

                writeln!(txt_file, "Min:  {:.6}", min)?;
                writeln!(txt_file, "Max:  {:.6}", max)?;
                writeln!(txt_file, "Mean: {:.6}", mean)?;

                // First 10x10 block
                writeln!(txt_file, "\nFirst 10×10 values:")?;
                for y in 0..10.min(tensor.desc.h as usize) {
                    for x in 0..10.min(tensor.desc.w as usize) {
                        let idx = y * tensor.desc.w as usize + x;
                        write!(txt_file, "{:8.4} ", ch_data[idx])?;
                    }
                    writeln!(txt_file)?;
                }
                writeln!(txt_file)?;
            }

            println!("  ✓ Saved text sample: {}", txt_path);
        }

        DType::Fp16 => {
            println!("  Copying FP16 data from GPU to CPU...");

            // Read as u16 (raw FP16 bits)
            let mut host_data_u16 = vec![0u16; total_elements];

            unsafe {
                use cudarc::driver::sys;
                let result = sys::lib().cuMemcpyDtoH_v2(
                    host_data_u16.as_mut_ptr() as *mut std::ffi::c_void,
                    tensor.data.ptr as sys::CUdeviceptr,
                    total_elements * 2,
                );
                if result != sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("CUDA memcpy failed: {:?}", result).into());
                }
            }

            // Save as binary
            let bin_path = format!("{}_fp16.bin", output_prefix);
            let mut file = File::create(&bin_path)?;
            let bytes = unsafe {
                std::slice::from_raw_parts(host_data_u16.as_ptr() as *const u8, total_elements * 2)
            };
            file.write_all(bytes)?;
            println!("  ✓ Saved binary: {}", bin_path);

            // Convert to FP32 for text output
            use half::f16;
            let host_data_f32: Vec<f32> = host_data_u16
                .iter()
                .map(|&bits| f16::from_bits(bits).to_f32())
                .collect();

            // Save as text (first 100 values per channel)
            let txt_path = format!("{}_sample.txt", output_prefix);
            let mut txt_file = File::create(&txt_path)?;

            writeln!(
                txt_file,
                "Tensor Shape: {}×{}×{}×{} (NCHW)",
                tensor.desc.n, tensor.desc.c, tensor.desc.h, tensor.desc.w
            )?;
            writeln!(txt_file, "Data Type: FP16")?;
            writeln!(txt_file, "Frame: #{}\n", tensor.from.frame_idx)?;

            let per_channel = (tensor.desc.h * tensor.desc.w) as usize;
            for ch in 0..3 {
                let ch_name = match ch {
                    0 => "R (Red)",
                    1 => "G (Green)",
                    2 => "B (Blue)",
                    _ => unreachable!(),
                };

                writeln!(txt_file, "=== Channel {} ({}) ===", ch, ch_name)?;

                let ch_offset = ch * per_channel;
                let ch_data = &host_data_f32[ch_offset..ch_offset + per_channel];

                // Statistics
                let min = ch_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max = ch_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mean = ch_data.iter().sum::<f32>() / ch_data.len() as f32;

                writeln!(txt_file, "Min:  {:.6}", min)?;
                writeln!(txt_file, "Max:  {:.6}", max)?;
                writeln!(txt_file, "Mean: {:.6}", mean)?;

                // First 10x10 block
                writeln!(txt_file, "\nFirst 10×10 values:")?;
                for y in 0..10.min(tensor.desc.h as usize) {
                    for x in 0..10.min(tensor.desc.w as usize) {
                        let idx = y * tensor.desc.w as usize + x;
                        write!(txt_file, "{:8.4} ", ch_data[idx])?;
                    }
                    writeln!(txt_file)?;
                }
                writeln!(txt_file)?;
            }

            println!("  ✓ Saved text sample: {}", txt_path);
        }
    }

    Ok(())
}

fn dump_raw_frame_info(raw: &RawFramePacket) {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║                RAW FRAME INFORMATION                     ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Frame Index:   {:<41} ║", raw.meta.frame_idx);
    println!("║  Source ID:     {:<41} ║", raw.meta.source_id);
    println!(
        "║  Dimensions:    {}×{}{:>37} ║",
        raw.meta.width, raw.meta.height, ""
    );
    println!("║  Pixel Format:  {:?}{:>32} ║", raw.meta.pixfmt, "");
    println!("║  Color Space:   {:?}{:>34} ║", raw.meta.colorspace, "");
    println!(
        "║  Stride:        {} bytes{:>36} ║",
        raw.meta.stride_bytes, ""
    );
    println!(
        "║  Data Length:   {} bytes ({:.2} MB){:>23} ║",
        raw.data.len,
        raw.data.len as f64 / (1024.0 * 1024.0),
        ""
    );
    println!("║  Memory Loc:    {:?}{:>30} ║", raw.data.loc, "");
    println!("║  PTS:           {} ns{:>32} ║", raw.meta.pts_ns, "");
    println!("║  Capture Time:  {} ns{:>32} ║", raw.meta.t_capture_ns, "");
    println!("╚══════════════════════════════════════════════════════════╝");
}

fn dump_tensor_info(tensor: &TensorInputPacket) {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║              TENSOR OUTPUT INFORMATION                   ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║  Shape (NCHW):  {}×{}×{}×{}{:>32} ║",
        tensor.desc.n, tensor.desc.c, tensor.desc.h, tensor.desc.w, ""
    );
    println!("║  Data Type:     {:?}{:>38} ║", tensor.desc.dtype, "");
    println!("║  Device:        GPU {}{:>44} ║", tensor.desc.device, "");

    let element_size = match tensor.desc.dtype {
        DType::Fp16 => 2,
        DType::Fp32 => 4,
    };
    let total_elements = tensor.desc.n * tensor.desc.c * tensor.desc.h * tensor.desc.w;
    let total_bytes = total_elements * element_size;

    println!("║  Total Elements: {}{:>41} ║", total_elements, "");
    println!("║  Element Size:   {} bytes{:>38} ║", element_size, "");
    println!(
        "║  Total Size:     {} bytes ({:.2} MB){:>23} ║",
        total_bytes,
        total_bytes as f64 / (1024.0 * 1024.0),
        ""
    );
    println!("║  Memory Loc:     {:?}{:>30} ║", tensor.data.loc, "");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Source Frame:   #{}{:>40} ║", tensor.from.frame_idx, "");
    println!(
        "║  Original Size:  {}×{}{:>37} ║",
        tensor.from.width, tensor.from.height, ""
    );
    println!("╚══════════════════════════════════════════════════════════╝");
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║         Preprocessing Output Dump Tool                  ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // List DeckLink devices
    let devices = decklink_input::devicelist();
    println!("📹 Available DeckLink devices: {}", devices.len());
    for (idx, name) in devices.iter().enumerate() {
        println!("  [{}] {}", idx, name);
    }
    println!();

    if devices.is_empty() {
        return Err("No DeckLink devices found".into());
    }

    // Open capture session
    println!("🔧 Opening DeckLink capture session on device 0...");
    let mut session = CaptureSession::open(0)?;
    println!("  ✓ Capture session opened");

    // Create preprocessor
    println!("\n🔧 Creating Preprocessor:");
    let mut preprocessor = Preprocessor::with_params(
        (512, 512),            // Output size
        true,                  // FP16
        0,                     // Device 0
        [0.485, 0.456, 0.406], // ImageNet mean
        [0.229, 0.224, 0.225], // ImageNet std
        ChromaOrder::UYVY,     // UYVY chroma order
    )?;
    println!(
        "  Output Size:   {}×{}",
        preprocessor.size.0, preprocessor.size.1
    );
    println!("  Data Type:     FP16");
    println!("  Device:        GPU 0");
    println!("  Chroma Order:  {:?}", preprocessor.chroma);
    println!("  Mean:          {:?}", preprocessor.mean);
    println!("  Std:           {:?}", preprocessor.std);
    println!("  ✓ Preprocessor ready");

    // Wait for a STABLE GPU frame (skip pre-detection frames)
    println!("\n🎥 Waiting for stable GPU frame from DeckLink...");
    println!("  (Skipping pre-detection CPU frames and initial GPU frames)");
    println!("  Pre-detection frames: 720×486 (should skip ~2 frames)");
    println!("  Target: 1920×1080 GPU frame");

    let mut gpu_frame_count = 0;
    const MIN_STABLE_FRAMES: usize = 3; // Skip first 2 GPU frames to avoid pre-detection

    let raw_frame = loop {
        match session.get_frame()? {
            Some(frame) => {
                // Check if it's a GPU frame
                if matches!(frame.data.loc, MemLoc::Gpu { .. }) {
                    gpu_frame_count += 1;

                    // Also verify it's the correct resolution (not pre-detection)
                    let is_stable = frame.meta.width == 1920 && frame.meta.height == 1080;

                    if is_stable && gpu_frame_count >= MIN_STABLE_FRAMES {
                        println!(
                            "  ✓ Stable GPU frame captured! ({}×{}, frame #{})",
                            frame.meta.width, frame.meta.height, gpu_frame_count
                        );
                        break frame;
                    } else if is_stable {
                        println!(
                            "  ⏭️  Skipping GPU frame #{}/{} (waiting for stable frames)...",
                            gpu_frame_count, MIN_STABLE_FRAMES
                        );
                    } else {
                        println!(
                            "  ⚠️  Unexpected GPU frame size: {}×{} (frame #{})",
                            frame.meta.width, frame.meta.height, gpu_frame_count
                        );
                    }
                } else {
                    println!(
                        "  ⏭️  Skipping CPU frame ({}×{})...",
                        frame.meta.width, frame.meta.height
                    );
                }
            }
            None => {
                thread::sleep(Duration::from_millis(50));
            }
        }
    };

    // Display raw frame info
    dump_raw_frame_info(&raw_frame);

    // Confirm GPU memory
    println!("\n✓ Frame is in GPU memory (GPU Direct enabled)");

    // Process the frame
    println!("\n⚙️  Running preprocessing...");
    let tensor = preprocessor.process(raw_frame);
    println!("  ✓ Preprocessing completed!");

    // Display tensor info
    dump_tensor_info(&tensor);

    // Dump to files
    let output_prefix = format!("preprocessing_dump_frame_{}", tensor.from.frame_idx);
    dump_tensor_to_file(&tensor, &output_prefix)?;

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║                  DUMP COMPLETE ✓                         ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Files created:                                          ║");
    println!("║    • {}_*.bin   (binary data)    ║", output_prefix);
    println!("║    • {}_sample.txt (text sample) ║", output_prefix);
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Print instructions
    println!("📖 How to use the dump files:\n");
    println!("1. Binary format:");
    println!("   - FP16: Read as uint16_t array, convert using half-precision library");
    println!("   - FP32: Read as float32 array directly");
    println!("   - Layout: NCHW (all R channel, then G, then B)");
    println!("   - Shape: 1×3×512×512 = 786,432 elements\n");

    println!("2. Text sample:");
    println!("   - Contains statistics (min/max/mean) for each channel");
    println!("   - Shows first 10×10 block of values");
    println!("   - Human-readable for quick inspection\n");

    println!("3. Python example to load binary:");
    println!("   import numpy as np");
    println!(
        "   data = np.fromfile('{}_fp16.bin', dtype=np.float16)",
        output_prefix
    );
    println!("   tensor = data.reshape(1, 3, 512, 512)  # NCHW");
    println!("   # Access channels: R=tensor[0,0], G=tensor[0,1], B=tensor[0,2]\n");

    println!("✨ Dump complete! Check the files for inspection.");

    Ok(())
}
