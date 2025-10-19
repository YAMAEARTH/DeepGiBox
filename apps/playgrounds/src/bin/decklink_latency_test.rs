// DeckLink Latency Test - Measure real per-stage latency with DeckLink input
use std::error::Error;
use std::thread;
use std::time::Duration;

use decklink_input::capture::CaptureSession;
use telemetry::{now_ns, record_ms};

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== DeckLink Latency Test ===");
    println!("Testing per-stage latency measurement with real DeckLink capture\n");

    // Open DeckLink capture session
    println!("Opening DeckLink device 0...");
    let mut session = CaptureSession::open(0)?;
    println!("✓ DeckLink capture session opened successfully\n");

    // Wait for device to stabilize
    println!("Waiting for device to stabilize...");
    thread::sleep(Duration::from_millis(500));
    println!("✓ Device ready\n");

    println!("Capturing and measuring latency for 100 frames...");
    println!("Output format: [lat] stage_name=time_in_ms\n");
    println!("{:-<80}", "");

    let mut frame_count = 0;
    let mut total_capture_ms = 0.0;
    let mut total_frame_ms = 0.0;

    // Capture 100 frames and measure latency
    for attempt in 0..300 {
        // Start measuring total frame processing time
        let t_frame = now_ns();

        // Stage 1: Capture frame from DeckLink
        let t_capture = now_ns();
        let packet_option = session.get_frame()?;

        if let Some(packet) = packet_option {
            let capture_ms = (now_ns() - t_capture) as f64 / 1_000_000.0;
            record_ms("capture", t_capture);

            // Stage 2: Simulate frame validation
            let t_validate = now_ns();
            let width = packet.meta.width;
            let height = packet.meta.height;
            let stride = packet.meta.stride_bytes;
            let buffer_size = packet.data.len;
            let pixfmt = packet.meta.pixfmt;
            let colorspace = packet.meta.colorspace;
            let frame_idx = packet.meta.frame_idx;

            // Validate frame dimensions
            let is_valid = width > 0 && height > 0 && stride > 0 && buffer_size > 0;
            record_ms("validate", t_validate);

            // Stage 3: Simulate simple frame processing (read some pixels)
            let t_process = now_ns();
            if is_valid && buffer_size >= 8 {
                // Read first 8 bytes to simulate minimal processing
                let _sample_data = unsafe { std::slice::from_raw_parts(packet.data.ptr, 8) };
            }
            record_ms("process", t_process);

            // Record end-to-end frame time
            let frame_ms = (now_ns() - t_frame) as f64 / 1_000_000.0;
            record_ms("frame.e2e", t_frame);

            // Accumulate statistics
            frame_count += 1;
            total_capture_ms += capture_ms;
            total_frame_ms += frame_ms;

            // Print frame info every 10 frames
            if frame_count % 10 == 0 {
                println!(
                    "Frame #{}: {}x{} stride={} size={} bytes pixfmt={:?} colorspace={:?}",
                    frame_idx, width, height, stride, buffer_size, pixfmt, colorspace
                );
            }

            // Stop after 100 frames
            if frame_count >= 100 {
                break;
            }
        } else {
            // No frame available, wait a bit
            thread::sleep(Duration::from_micros(100));
        }

        // Safety check - don't loop forever
        if attempt >= 299 {
            println!(
                "\nWarning: Reached maximum attempts (300) with only {} frames captured",
                frame_count
            );
            break;
        }
    }

    println!("{:-<80}", "");
    println!("\n=== Statistics Summary ===");
    println!("Total frames captured: {}", frame_count);

    if frame_count > 0 {
        let avg_capture = total_capture_ms / frame_count as f64;
        let avg_frame = total_frame_ms / frame_count as f64;
        let fps = 1000.0 / avg_frame;

        println!("\nAverage Latencies:");
        println!("  Capture:       {:.3} ms", avg_capture);
        println!("  Frame (E2E):   {:.3} ms", avg_frame);
        println!("\nPerformance:");
        println!("  Average FPS:   {:.2}", fps);
        println!("  Frame budget at 60 FPS:  16.67 ms");
        println!("  Frame budget at 30 FPS:  33.33 ms");

        if avg_frame < 16.67 {
            println!("\n✓ Performance is good for 60 FPS!");
        } else if avg_frame < 33.33 {
            println!("\n⚠ Performance suitable for 30 FPS (too slow for 60 FPS)");
        } else {
            println!("\n✗ Performance is below 30 FPS");
        }
    }

    println!("\n=== Test Complete ===");
    Ok(())
}
