use decklink_rust::packets::{RawFramePacket, FrameMeta, PixelFormat, ColorSpace};
use decklink_rust::preprocessing_v2::{PreprocessingV2Stage, PreprocessingV2Config, ProcessingStageV2};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DeepGI CV-CUDA Preprocessing v.2 Test ===");
    
    // Create test configuration
    let config = PreprocessingV2Config {
        pan_x: 100,
        pan_y: 50,
        zoom: 1.2,
        target_size: (512, 512),
        use_cuda: true,
        debug: true,
        normalization: Some(([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])),
        async_processing: true,
        device_id: 0,
    };
    
    println!("Configuration:");
    println!("  Target size: {}x{}", config.target_size.0, config.target_size.1);
    println!("  Pan offset: ({}, {})", config.pan_x, config.pan_y);
    println!("  Zoom factor: {}", config.zoom);
    println!("  CUDA enabled: {}", config.use_cuda);
    println!("  Async processing: {}", config.async_processing);
    println!("  Device ID: {}", config.device_id);
    
    // Initialize preprocessing stage
    println!("\nInitializing CV-CUDA preprocessing stage...");
    let mut stage = match PreprocessingV2Stage::new(config) {
        Ok(stage) => {
            println!("✓ CV-CUDA preprocessing v.2 initialized successfully");
            stage
        },
        Err(e) => {
            println!("✗ Failed to initialize CV-CUDA: {}", e);
            println!("Falling back to CPU-only configuration...");
            let cpu_config = PreprocessingV2Config {
                use_cuda: false,
                debug: true,
                ..PreprocessingV2Config::default()
            };
            PreprocessingV2Stage::new(cpu_config)?
        }
    };
    
    // Create test frames with different resolutions
    let test_cases = vec![
        (1920, 1080, "1080p"),
        (1280, 720, "720p"),
        (3840, 2160, "4K"),
        (640, 480, "VGA"),
    ];
    
    println!("\n=== Processing Test Frames ===");
    
    for (width, height, description) in test_cases {
        println!("\nTesting {} ({}x{}):", description, width, height);
        
        // Create test frame data (BGRA format)
        let frame_size = (width * height * 4) as usize;
        let mut test_data = vec![0u8; frame_size];
        
        // Fill with gradient pattern for testing
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 4) as usize;
                if idx + 3 < test_data.len() {
                    test_data[idx] = (x % 256) as u8;     // B
                    test_data[idx + 1] = (y % 256) as u8; // G
                    test_data[idx + 2] = ((x + y) % 256) as u8; // R
                    test_data[idx + 3] = 255;             // A
                }
            }
        }
        
        let meta = FrameMeta {
            source_id: 1,
            width,
            height,
            stride: width * 4,
            pixfmt: PixelFormat::BGRA8,
            colorspace: ColorSpace::BT709,
            pts_ns: 0,
            timecode: None,
            seq_no: 1,
        };
        
        let frame = RawFramePacket::new_cpu(test_data, meta);
        
        // Process frame
        let start_time = Instant::now();
        match ProcessingStageV2::process(&mut stage, frame) {
            Ok(tensor_packet) => {
                let processing_time = start_time.elapsed();
                println!("  ✓ Processed in {:.2}ms", processing_time.as_secs_f64() * 1000.0);
                println!("  ✓ Input: {}x{} BGRA", width, height);
                println!("  ✓ Output: {}x{} {} {:?} {:?}", 
                    tensor_packet.desc.shape[3], 
                    tensor_packet.desc.shape[2],
                    match tensor_packet.desc.colors {
                        decklink_rust::packets::ColorFormat::RGB => "RGB",
                        decklink_rust::packets::ColorFormat::BGR => "BGR",
                    },
                    tensor_packet.desc.dtype,
                    tensor_packet.desc.layout
                );
                
                match &tensor_packet.mem {
                    decklink_rust::packets::TensorMem::Cuda { device_ptr } => {
                        println!("  ✓ GPU Memory: device_ptr=0x{:x}", device_ptr);
                    },
                    decklink_rust::packets::TensorMem::Cpu { data } => {
                        println!("  ✓ CPU Memory: {} bytes", data.len());
                    }
                }
                
                println!("  ✓ Frame metadata preserved: seq={}, pts={}ns", 
                    tensor_packet.meta.seq_no, tensor_packet.meta.pts_ns);
            },
            Err(e) => {
                println!("  ✗ Processing failed: {}", e);
            }
        }
    }
    
    // Performance benchmark
    println!("\n=== Performance Benchmark ===");
    let benchmark_config = PreprocessingV2Config {
        target_size: (512, 512),
        use_cuda: true,
        debug: false,
        async_processing: true,
        ..Default::default()
    };
    
    let mut benchmark_stage = match PreprocessingV2Stage::new(benchmark_config) {
        Ok(stage) => stage,
        Err(_) => {
            println!("CUDA not available for benchmark, using CPU fallback");
            PreprocessingV2Stage::new(PreprocessingV2Config {
                use_cuda: false,
                debug: false,
                ..Default::default()
            })?
        }
    };
    
    let test_frames = 100;
    let frame_data = vec![0u8; 1920 * 1080 * 4];
    let meta = FrameMeta {
        source_id: 1,
        width: 1920,
        height: 1080,
        stride: 1920 * 4,
        pixfmt: PixelFormat::BGRA8,
        colorspace: ColorSpace::BT709,
        pts_ns: 0,
        timecode: None,
        seq_no: 1,
    };
    
    println!("Processing {} frames for benchmark...", test_frames);
    let benchmark_start = Instant::now();
    let mut successful_frames = 0;
    
    for i in 0..test_frames {
        let mut frame_meta = meta.clone();
        frame_meta.seq_no = i as u64;
        frame_meta.pts_ns = (i as u64) * 16_666_666; // ~60fps
        
        let frame = RawFramePacket::new_cpu(frame_data.clone(), frame_meta);
        
        if ProcessingStageV2::process(&mut benchmark_stage, frame).is_ok() {
            successful_frames += 1;
        }
    }
    
    let total_time = benchmark_start.elapsed();
    let avg_latency = total_time.as_secs_f64() / successful_frames as f64;
    let fps = successful_frames as f64 / total_time.as_secs_f64();
    
    println!("Benchmark Results:");
    println!("  Total time: {:.2}ms", total_time.as_secs_f64() * 1000.0);
    println!("  Successful frames: {}/{}", successful_frames, test_frames);
    println!("  Average latency: {:.2}ms", avg_latency * 1000.0);
    println!("  Throughput: {:.1} FPS", fps);
    println!("  Stage name: {}", benchmark_stage.name());
    
    // Latency analysis
    if avg_latency < 0.010 {
        println!("  ✓ Excellent latency (<10ms) - suitable for real-time processing");
    } else if avg_latency < 0.033 {
        println!("  ✓ Good latency (<33ms) - suitable for 30fps processing");
    } else {
        println!("  ⚠ High latency (>33ms) - may impact real-time performance");
    }
    
    println!("\n=== CV-CUDA Preprocessing v.2 Test Complete ===");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_preprocessing_v2_basic() {
        let config = PreprocessingV2Config {
            use_cuda: false, // Use CPU for testing
            debug: true,
            ..Default::default()
        };
        
        let mut stage = PreprocessingV2Stage::new(config).unwrap();
        
        let test_data = vec![0u8; 640 * 480 * 4];
        let meta = FrameMeta {
            source_id: 1,
            width: 640,
            height: 480,
            stride: 640 * 4,
            pixfmt: PixelFormat::BGRA8,
            colorspace: ColorSpace::BT709,
            pts_ns: 0,
            timecode: None,
            seq_no: 1,
        };
        
        let frame = RawFramePacket::new_cpu(test_data, meta);
        let result = ProcessingStageV2::process(&mut stage, frame);
        
        assert!(result.is_ok());
    }
}
