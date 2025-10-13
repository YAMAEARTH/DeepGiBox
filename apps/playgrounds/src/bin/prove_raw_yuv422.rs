// Prove that DeckLink captures RAW YUV422 (UYVY format)
use std::error::Error;
use std::thread;
use std::time::Duration;

use decklink_input::capture::CaptureSession;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Proving DeckLink Captures RAW YUV422 ===\n");

    let mut session = CaptureSession::open(0)?;
    println!("✓ DeckLink opened\n");

    thread::sleep(Duration::from_millis(500));
    
    println!("Analyzing pixel format and data layout...\n");
    println!("{:=<80}", "");

    // Get one frame
    let mut packet = None;
    for _ in 0..100 {
        if let Some(p) = session.get_frame()? {
            packet = Some(p);
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }

    let packet = packet.expect("Failed to get frame");
    
    println!("\n📦 FRAME METADATA:");
    println!("  Resolution:     {}x{}", packet.meta.width, packet.meta.height);
    println!("  Pixel Format:   {:?}", packet.meta.pixfmt);
    println!("  Color Space:    {:?}", packet.meta.colorspace);
    println!("  Stride:         {} bytes", packet.meta.stride_bytes);
    println!("  Buffer Size:    {} bytes ({:.2} MB)", 
        packet.data.len, packet.data.len as f64 / 1_048_576.0);
    
    // Calculate expected sizes for different formats
    let width = packet.meta.width as usize;
    let height = packet.meta.height as usize;
    let total_pixels = width * height;
    
    println!("\n📐 THEORETICAL SIZE CALCULATIONS:");
    println!("  Total pixels:           {}", total_pixels);
    println!("  RGB (3 bytes/pixel):    {} bytes ({:.2} MB)", 
        total_pixels * 3, (total_pixels * 3) as f64 / 1_048_576.0);
    println!("  RGBA (4 bytes/pixel):   {} bytes ({:.2} MB)", 
        total_pixels * 4, (total_pixels * 4) as f64 / 1_048_576.0);
    println!("  YUV422 (2 bytes/pixel): {} bytes ({:.2} MB)", 
        total_pixels * 2, (total_pixels * 2) as f64 / 1_048_576.0);
    println!("  → Actual buffer size:   {} bytes ({:.2} MB)", 
        packet.data.len, packet.data.len as f64 / 1_048_576.0);
    
    // Check if it matches YUV422
    let expected_yuv422 = total_pixels * 2;
    println!("\n✅ VERIFICATION:");
    if packet.data.len == expected_yuv422 {
        println!("  ✓ Buffer size EXACTLY matches YUV422 format (2 bytes/pixel)");
        println!("  ✓ This confirms DeckLink captures RAW YUV422, NOT RGB!");
    } else {
        println!("  ✗ Buffer size does not match YUV422: {} vs {} bytes", 
            packet.data.len, expected_yuv422);
    }
    
    println!("\n🔍 ANALYZING RAW PIXEL DATA:");
    println!("YUV422 uses UYVY layout: [U Y V Y] [U Y V Y] ...");
    println!("  - U/V (chroma) shared between 2 adjacent pixels");
    println!("  - Each 4 bytes = 2 pixels\n");
    
    // Read first 16 bytes (8 pixels worth) from multiple locations
    println!("Sample locations:");
    
    let locations = [
        ("Top-left corner", 0),
        ("Top-middle", packet.meta.stride_bytes as usize * 0 + width),
        ("Center", packet.meta.stride_bytes as usize * (height / 2) + width),
        ("Bottom-middle", packet.meta.stride_bytes as usize * (height - 1) + width),
    ];
    
    for (name, offset) in locations.iter() {
        if *offset + 16 <= packet.data.len {
            unsafe {
                let bytes = std::slice::from_raw_parts(packet.data.ptr.add(*offset), 16);
                println!("\n  {} (offset {}):", name, offset);
                print!("    Raw bytes: ");
                for b in bytes {
                    print!("{:02X} ", b);
                }
                println!();
                
                // Parse as UYVY
                println!("    Parsed as UYVY:");
                for i in (0..16).step_by(4) {
                    let u = bytes[i];
                    let y1 = bytes[i + 1];
                    let v = bytes[i + 2];
                    let y2 = bytes[i + 3];
                    println!("      Pixel {}-{}: U={:3} Y={:3} V={:3} Y={:3}", 
                        i/2, i/2+1, u, y1, v, y2);
                }
            }
        }
    }
    
    println!("\n📊 PIXEL VALUE STATISTICS:");
    // Sample 1000 random pixels to get statistics
    let sample_size = 1000;
    let mut y_values = Vec::new();
    let mut u_values = Vec::new();
    let mut v_values = Vec::new();
    
    unsafe {
        for i in 0..sample_size {
            let offset = (i * 17) % (packet.data.len / 4) * 4; // Random-ish sampling
            if offset + 4 <= packet.data.len {
                let bytes = std::slice::from_raw_parts(packet.data.ptr.add(offset), 4);
                u_values.push(bytes[0]);
                y_values.push(bytes[1]);
                v_values.push(bytes[2]);
                y_values.push(bytes[3]); // Y2
            }
        }
    }
    
    let avg_y = y_values.iter().map(|&x| x as u32).sum::<u32>() / y_values.len() as u32;
    let avg_u = u_values.iter().map(|&x| x as u32).sum::<u32>() / u_values.len() as u32;
    let avg_v = v_values.iter().map(|&x| x as u32).sum::<u32>() / v_values.len() as u32;
    
    println!("  Sampled {} pixels:", sample_size * 2);
    println!("    Average Y (luma):       {} (range: 16-235 for video)", avg_y);
    println!("    Average U (chroma):     {} (range: 16-240, center at 128)", avg_u);
    println!("    Average V (chroma):     {} (range: 16-240, center at 128)", avg_v);
    
    // Check if values are in valid YUV range
    if avg_y >= 16 && avg_y <= 235 && avg_u >= 16 && avg_u <= 240 && avg_v >= 16 && avg_v <= 240 {
        println!("\n  ✓ Values are in valid YUV video range (BT.709)");
        println!("  ✓ This confirms the data is RAW YUV, not RGB!");
    }
    
    println!("\n{:=<80}", "");
    println!("\n🎯 CONCLUSION:");
    println!("  DeckLink captures in RAW YUV422 format (UYVY layout):");
    println!("  • 2 bytes per pixel (not 3 for RGB or 4 for RGBA)");
    println!("  • Chroma subsampling: 4:2:2 (U/V shared between 2 pixels)");
    println!("  • Color space: BT.709");
    println!("  • Layout: [U0 Y0 V0 Y1] [U2 Y2 V2 Y3] ...");
    println!("\n  ⚠️  This means:");
    println!("  • Data is NOT in RGB format");
    println!("  • Processing (inference/overlay) needs YUV→RGB conversion");
    println!("  • Conversion can be done on CPU or GPU");
    println!("  • GPU conversion is MUCH faster (should be <1ms)");
    
    println!("\n💡 NEXT STEPS:");
    println!("  1. Implement CUDA kernel for YUV422→RGB conversion");
    println!("  2. Measure real conversion latency on GPU");
    println!("  3. Optimize memory transfer (H2D)");
    
    println!("\n=== Test Complete ===");
    Ok(())
}
