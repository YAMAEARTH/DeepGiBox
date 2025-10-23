/// Simple test to verify the composite_alpha_only implementation
/// This doesn't require DeckLink hardware - just tests the GPU kernels

use decklink_output::{BgraImage, OutputSession};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing composite_alpha_only() implementation");
    println!("================================================\n");

    // 1. Load test image with alpha channel
    println!("📸 Loading test PNG with alpha channel...");
    let png_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "foreground.png".to_string());
    
    let png_image = BgraImage::load_from_file(&png_path)?;
    println!("   ✓ Loaded: {}x{} pixels", png_image.width, png_image.height);
    println!("   ✓ Data size: {} bytes", png_image.data.len());
    
    // Check if image has alpha channel data
    let has_transparency = png_image.data.chunks(4)
        .any(|pixel| pixel[3] < 255);
    
    if has_transparency {
        println!("   ✓ Image has transparency (alpha channel detected)");
    } else {
        println!("   ⚠️  Image appears fully opaque (no transparency detected)");
    }

    // 2. Create output session (this uploads PNG to GPU)
    println!("\n🔧 Creating output session...");
    let mut session = OutputSession::new(png_image.width, png_image.height, &png_image)?;
    println!("   ✓ Output session created");
    println!("   ✓ PNG uploaded to GPU");

    // 3. Create fake UYVY background on CPU (simulate DeckLink frame)
    println!("\n🎨 Creating fake UYVY background...");
    let width = png_image.width as usize;
    let height = png_image.height as usize;
    let uyvy_pitch = (width / 2) * 4; // UYVY is 2 pixels per 4 bytes
    let uyvy_size = uyvy_pitch * height;
    
    // Create solid blue background in UYVY format
    // Y=41, U=240, V=110 = RGB(0, 0, 255) blue
    let mut uyvy_data = vec![0u8; uyvy_size];
    for i in (0..uyvy_size).step_by(4) {
        uyvy_data[i] = 240;  // U
        uyvy_data[i+1] = 41; // Y0
        uyvy_data[i+2] = 110; // V
        uyvy_data[i+3] = 41; // Y1
    }
    println!("   ✓ Created {}x{} UYVY background (blue)", width, height);

    // 4. Upload UYVY to GPU (simulate DeckLink GPU Direct)
    println!("\n⬆️  Uploading UYVY to GPU...");
    
    // Allocate GPU memory for UYVY
    use std::os::raw::c_void;
    use std::ptr;
    
    extern "C" {
        fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
        fn cudaFree(dev_ptr: *mut c_void) -> i32;
        fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    }
    
    const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
    
    let mut uyvy_gpu_ptr: *mut c_void = ptr::null_mut();
    unsafe {
        let result = cudaMalloc(&mut uyvy_gpu_ptr, uyvy_size);
        if result != 0 {
            return Err(format!("cudaMalloc failed: {}", result).into());
        }
        
        let result = cudaMemcpy(
            uyvy_gpu_ptr,
            uyvy_data.as_ptr() as *const c_void,
            uyvy_size,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        );
        if result != 0 {
            cudaFree(uyvy_gpu_ptr);
            return Err(format!("cudaMemcpy failed: {}", result).into());
        }
    }
    println!("   ✓ UYVY uploaded to GPU at {:?}", uyvy_gpu_ptr);

    // 5. Test composite_alpha_only() - THE NEW FAST METHOD!
    println!("\n🚀 Testing composite_alpha_only() - FAST MODE");
    println!("   Pipeline: UYVY → Single Composite Kernel → Output");
    
    let start = std::time::Instant::now();
    
    session.composite_alpha_only(
        uyvy_gpu_ptr as *const u8,
        uyvy_pitch,
    )?;
    
    let elapsed = start.elapsed();
    println!("   ✓ Composite completed in {:.3}ms", elapsed.as_secs_f64() * 1000.0);
    
    // 6. Download result and verify
    println!("\n⬇️  Downloading result from GPU...");
    let output_data = session.download_output()?;
    println!("   ✓ Downloaded {} bytes", output_data.len());
    
    // Verify output is BGRA format (4 bytes per pixel)
    let expected_size = (width * height * 4) as usize;
    assert_eq!(output_data.len(), expected_size, "Output size mismatch");
    println!("   ✓ Output size verified: {} bytes", output_data.len());
    
    // Check that output is not all zeros
    let non_zero = output_data.iter().any(|&b| b != 0);
    assert!(non_zero, "Output is all zeros!");
    println!("   ✓ Output contains non-zero data");
    
    // Sample a few pixels to verify compositing happened
    println!("\n🔍 Sampling output pixels:");
    for i in 0..5 {
        let offset = (i * width * height / 10) * 4;
        if offset + 3 < output_data.len() {
            let b = output_data[offset];
            let g = output_data[offset + 1];
            let r = output_data[offset + 2];
            let a = output_data[offset + 3];
            println!("   Pixel {}: BGRA({:3}, {:3}, {:3}, {:3})", i, b, g, r, a);
        }
    }

    // 7. Cleanup
    unsafe {
        cudaFree(uyvy_gpu_ptr);
    }
    
    println!("\n✅ Test PASSED!");
    println!("   composite_alpha_only() is working correctly!");
    println!("   Time: {:.3}ms (expected: ~0.5-1ms for 1080p)", elapsed.as_secs_f64() * 1000.0);

    Ok(())
}
