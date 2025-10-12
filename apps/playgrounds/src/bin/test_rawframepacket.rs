use std::error::Error;
use std::thread;
use std::time::Duration;

use decklink_input::capture::CaptureSession;
use decklink_input::{ColorSpace, MemLoc, PixelFormat, RawFramePacket};

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Test RawFramePacket from DeckLink Input ===\n");

    // List available devices first
    let devices = decklink_input::devicelist();
    println!("Available DeckLink devices: {}", devices.len());
    for (idx, name) in devices.iter().enumerate() {
        println!("  [{}] {}", idx, name);
    }
    println!();

    if devices.is_empty() {
        return Err("No DeckLink devices found".into());
    }

    // Open device 0
    let mut session = CaptureSession::open(0)?;
    println!("✓ Opened capture session on device 0\n");

    // Try to capture a few frames
    const TARGET_FRAMES: usize = 10;
    let mut frames_captured = 0;
    for attempt in 0..100 {
        match session.get_frame()? {
            Some(packet) => {
                frames_captured += 1;
                print_rawframepacket(&packet);

                if frames_captured >= TARGET_FRAMES {
                    println!("\n✓ Successfully captured {} frames!", frames_captured);
                    return Ok(());
                }

                println!("---\n");
            }
            None => {
                if attempt % 10 == 0 {
                    println!("Waiting for frame... (attempt {})", attempt);
                }
            }
        }
        thread::sleep(Duration::from_millis(50));
    }

    if frames_captured > 0 {
        println!("\n✓ Captured {} frame(s)", frames_captured);
        Ok(())
    } else {
        Err("Timed out waiting for frames".into())
    }
}

fn print_rawframepacket(packet: &RawFramePacket) {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║               RawFramePacket Details                      ║");
    println!("╠═══════════════════════════════════════════════════════════╣");

    // FrameMeta information
    println!("║ FrameMeta:");
    println!("║   source_id      : {}", packet.meta.source_id);
    println!("║   frame_idx      : {}", packet.meta.frame_idx);
    println!(
        "║   dimensions     : {}x{}",
        packet.meta.width, packet.meta.height
    );
    println!("║   pixel_format   : {:?}", packet.meta.pixfmt);
    println!("║   colorspace     : {:?}", packet.meta.colorspace);
    println!("║   stride_bytes   : {}", packet.meta.stride_bytes);
    println!("║   pts_ns         : {}", packet.meta.pts_ns);
    println!("║   t_capture_ns   : {}", packet.meta.t_capture_ns);

    // MemRef information
    println!("║");
    println!("║ MemRef (data):");
    println!("║   location       : {:?}", packet.data.loc);
    println!("║   ptr            : {:p}", packet.data.ptr);
    println!("║   len            : {} bytes", packet.data.len);
    println!("║   stride         : {} bytes", packet.data.stride);

    // Calculate expected size
    let expected_size = packet.meta.stride_bytes as usize * packet.meta.height as usize;
    let size_match = if packet.data.len == expected_size {
        "✓"
    } else {
        "✗"
    };
    println!(
        "║   expected_size  : {} bytes {}",
        expected_size, size_match
    );

    // Sample first few bytes if available
    if !packet.data.ptr.is_null() && packet.data.len > 0 {
        let sample_size = std::cmp::min(16, packet.data.len);
        let sample = unsafe { std::slice::from_raw_parts(packet.data.ptr, sample_size) };
        print!("║   first {} bytes  : ", sample_size);
        for (i, &byte) in sample.iter().enumerate() {
            if i > 0 && i % 4 == 0 {
                print!(" ");
            }
            print!("{:02X}", byte);
        }
        println!();
    }

    // Verify packet structure
    println!("║");
    println!("║ Verification:");
    verify_packet(packet);

    println!("╚═══════════════════════════════════════════════════════════╝");
}

fn verify_packet(packet: &RawFramePacket) {
    let mut all_ok = true;

    // Check pixel format
    match packet.meta.pixfmt {
        PixelFormat::YUV422_8 => {
            println!("║   [✓] PixelFormat is YUV422_8 (as expected)");
        }
        _ => {
            println!(
                "║   [✗] PixelFormat is {:?} (expected YUV422_8)",
                packet.meta.pixfmt
            );
            all_ok = false;
        }
    }

    // Check colorspace
    match packet.meta.colorspace {
        ColorSpace::BT709 => {
            println!("║   [✓] ColorSpace is BT709 (as expected)");
        }
        _ => {
            println!(
                "║   [✗] ColorSpace is {:?} (expected BT709)",
                packet.meta.colorspace
            );
            all_ok = false;
        }
    }

    // Check memory location
    match packet.data.loc {
        MemLoc::Cpu => {
            println!("║   [✓] Data is in CPU memory");
        }
        MemLoc::Gpu { device } => {
            println!("║   [✓] Data is in GPU memory (device {})", device);
        }
    }

    // Check dimensions
    if packet.meta.width > 0 && packet.meta.height > 0 {
        println!(
            "║   [✓] Dimensions are valid ({}x{})",
            packet.meta.width, packet.meta.height
        );
    } else {
        println!(
            "║   [✗] Invalid dimensions: {}x{}",
            packet.meta.width, packet.meta.height
        );
        all_ok = false;
    }

    // Check stride
    let min_stride = packet.meta.width * 2; // YUV422 is 2 bytes per pixel
    if packet.meta.stride_bytes >= min_stride {
        println!(
            "║   [✓] Stride is valid ({} >= {} min)",
            packet.meta.stride_bytes, min_stride
        );
    } else {
        println!(
            "║   [✗] Stride too small ({} < {} min)",
            packet.meta.stride_bytes, min_stride
        );
        all_ok = false;
    }

    // Check data size
    let expected = packet.meta.stride_bytes as usize * packet.meta.height as usize;
    if packet.data.len == expected {
        println!("║   [✓] Data size matches (stride × height)");
    } else {
        println!(
            "║   [✗] Data size mismatch ({} != {})",
            packet.data.len, expected
        );
        all_ok = false;
    }

    // Check pointer
    if !packet.data.ptr.is_null() {
        println!("║   [✓] Data pointer is valid");
    } else {
        println!("║   [✗] Data pointer is null");
        all_ok = false;
    }

    if all_ok {
        println!("║   [✓✓✓] All checks passed!");
    } else {
        println!("║   [!!!] Some checks failed");
    }
}
