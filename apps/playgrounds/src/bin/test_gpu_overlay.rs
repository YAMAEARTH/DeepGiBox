//! Test GPU Overlay Rendering - วาด overlay แล้ว download มาดู

use anyhow::Result;
use common_io::{DrawOp, OverlayPlanPacket, FrameMeta, PixelFormat, ColorSpace, Stage};
use std::fs;
use std::io::Write;

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  GPU OVERLAY RENDERER TEST                               ║");
    println!("║  วาด overlay บน GPU แล้ว download มาเช็คว่าวาดจริงไหม   ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    fs::create_dir_all("output/test")?;

    // Create GPU overlay renderer
    println!("🔧 Creating GPU overlay renderer...");
    let mut renderer = overlay_render::from_path("gpu,device=0")?;
    println!("  ✓ Renderer created");
    println!();

    // Create simple overlay plan with various shapes
    let width = 1920u32;
    let height = 1080u32;
    
    println!("🎨 Creating overlay plan ({}x{})...", width, height);
    
    let mut ops = Vec::new();
    
    // 1. Big red rectangle (outline) in center
    ops.push(DrawOp::Rect {
        xywh: (500.0, 300.0, 920.0, 480.0),
        thickness: 5,
        color: (255, 0, 0, 255), // Red, full opacity (R, G, B, A)
    });
    
    // 2. Green filled rectangle (semi-transparent)
    ops.push(DrawOp::FillRect {
        xywh: (600.0, 400.0, 200.0, 100.0),
        color: (0, 255, 0, 200), // Green, 200/255 opacity
    });
    
    // 3. Blue filled rectangle (opaque)
    ops.push(DrawOp::FillRect {
        xywh: (900.0, 400.0, 200.0, 100.0),
        color: (0, 0, 255, 255), // Blue, full opacity
    });
    
    // 4. Yellow lines (cross)
    ops.push(DrawOp::Poly {
        pts: vec![(100.0, 100.0), (300.0, 300.0)],
        thickness: 3,
        color: (255, 255, 0, 255), // Yellow
    });
    ops.push(DrawOp::Poly {
        pts: vec![(300.0, 100.0), (100.0, 300.0)],
        thickness: 3,
        color: (255, 255, 0, 255), // Yellow
    });
    
    // 5. White rectangle in corner
    ops.push(DrawOp::FillRect {
        xywh: (50.0, 50.0, 150.0, 80.0),
        color: (255, 255, 255, 255), // White
    });
    
    println!("  ✓ Created {} draw operations:", ops.len());
    println!("      - Red outline rectangle (center)");
    println!("      - Green filled rectangle (semi-transparent)");
    println!("      - Blue filled rectangle (opaque)");
    println!("      - Yellow cross lines");
    println!("      - White filled rectangle (corner)");
    println!();

    let overlay_plan = OverlayPlanPacket {
        from: FrameMeta {
            source_id: 0,
            width,
            height,
            pixfmt: PixelFormat::YUV422_8,
            colorspace: ColorSpace::BT709,
            frame_idx: 0,
            pts_ns: 0,
            t_capture_ns: 0,
            stride_bytes: width * 2,
            crop_region: None,
        },
        ops,
        canvas: (width, height),
    };

    // Render on GPU
    println!("🖼️  Rendering overlay on GPU...");
    let start = std::time::Instant::now();
    let overlay_frame = renderer.process(overlay_plan);
    let elapsed = start.elapsed();
    
    println!("  ✓ Rendered in {:.3}ms", elapsed.as_secs_f64() * 1000.0);
    println!("  ✓ Output:");
    println!("      → Buffer: {:p}", overlay_frame.argb.ptr);
    println!("      → Size: {} bytes", overlay_frame.argb.len);
    println!("      → Stride: {} bytes", overlay_frame.stride);
    println!("      → Location: {:?}", overlay_frame.argb.loc);
    println!();

    // Download from GPU to CPU using simple CUDA runtime API
    println!("📥 Downloading buffer from GPU...");
    
    let mut cpu_buffer = vec![0u8; overlay_frame.argb.len];
    
    // Use cudaMemcpy from CUDA runtime (simple C API)
    extern "C" {
        fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, count: usize, kind: i32) -> i32;
    }
    
    const CUDA_SUCCESS: i32 = 0;
    const cudaMemcpyDeviceToHost: i32 = 2;
    
    let result = unsafe {
        cudaMemcpy(
            cpu_buffer.as_mut_ptr() as *mut std::ffi::c_void,
            overlay_frame.argb.ptr as *const std::ffi::c_void,
            overlay_frame.argb.len,
            cudaMemcpyDeviceToHost,
        )
    };
    
    if result != CUDA_SUCCESS {
        anyhow::bail!("Failed to download from GPU: CUDA error {}", result);
    }
    
    println!("  ✓ Downloaded {} bytes", cpu_buffer.len());
    println!();

    // Analyze buffer
    println!("📊 Analyzing buffer...");
    let non_zero = cpu_buffer.iter().filter(|&&b| b != 0).count();
    let percent = (non_zero as f64 / cpu_buffer.len() as f64) * 100.0;
    
    println!("  Non-zero bytes: {} / {} ({:.2}%)", non_zero, cpu_buffer.len(), percent);
    
    // Count pixels by alpha channel (ARGB format: A, R, G, B)
    let mut transparent = 0;
    let mut semi_transparent = 0;
    let mut opaque = 0;
    
    for chunk in cpu_buffer.chunks_exact(4) {
        let alpha = chunk[0];
        match alpha {
            0 => transparent += 1,
            255 => opaque += 1,
            _ => semi_transparent += 1,
        }
    }
    
    let total_pixels = (width * height) as usize;
    println!("  Pixel alpha distribution:");
    println!("    Transparent (α=0):   {} pixels ({:.2}%)", transparent, transparent as f64 / total_pixels as f64 * 100.0);
    println!("    Semi-transparent:    {} pixels ({:.2}%)", semi_transparent, semi_transparent as f64 / total_pixels as f64 * 100.0);
    println!("    Opaque (α=255):      {} pixels ({:.2}%)", opaque, opaque as f64 / total_pixels as f64 * 100.0);
    println!();

    // Sample some pixels
    println!("  Sample pixels (ARGB format):");
    let samples = [
        (100, 100, "Top-left white rect"),
        (700, 450, "Center green rect"),
        (1000, 450, "Center blue rect"),
        (600, 400, "Red outline"),
    ];
    
    for (x, y, desc) in samples {
        let idx = (y * width + x) as usize * 4;
        if idx + 3 < cpu_buffer.len() {
            let a = cpu_buffer[idx];
            let r = cpu_buffer[idx + 1];
            let g = cpu_buffer[idx + 2];
            let b = cpu_buffer[idx + 3];
            println!("    ({:4}, {:4}): ARGB=({:3}, {:3}, {:3}, {:3}) - {}", x, y, a, r, g, b, desc);
        }
    }
    println!();

    // Save to file
    let filename = "output/test/test_gpu_overlay_argb.bin";
    fs::write(filename, &cpu_buffer)?;
    println!("💾 Saved buffer to: {}", filename);
    println!("   Use test_gpu_overlay.py to analyze further");
    println!();

    // Verdict
    if opaque > 10000 {
        println!("✅ SUCCESS: GPU overlay is rendering! ({} opaque pixels)", opaque);
    } else {
        println!("❌ FAILED: GPU overlay NOT rendering (only {} opaque pixels)", opaque);
    }

    Ok(())
}
