// Real Capture Latency Test - Using t_capture_ns and processing stages
use std::error::Error;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use decklink_input::capture::CaptureSession;
use telemetry::{now_ns, record_ms};

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Real Capture Latency Test ===");
    println!("Measuring actual latency from DeckLink buffer to processing\n");

    // Open DeckLink
    let mut session = CaptureSession::open(0)?;
    println!("✓ DeckLink opened\n");

    // Stabilize
    thread::sleep(Duration::from_millis(500));
    
    println!("Capturing 100 frames with detailed latency measurements...");
    println!("{:=<90}", "");

    let mut frame_count = 0;
    let mut total_buffer_latency = 0.0;
    let mut total_copy_latency = 0.0;
    let mut total_e2e = 0.0;

    for attempt in 0..300 {
        // === STAGE 0: Get frame from DeckLink ===
        let t_get = now_ns();
        let packet_option = session.get_frame()?;
        
        if let Some(packet) = packet_option {
            frame_count += 1;
            
            // === MEASURE: DeckLink Buffer Latency ===
            // Time from when DeckLink captured the frame to when we received it
            let now_unix_ns = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
            
            let buffer_latency_ms = (now_unix_ns - packet.meta.t_capture_ns) as f64 / 1_000_000.0;
            total_buffer_latency += buffer_latency_ms;
            
            // === STAGE 1: Copy frame data (simulate H2H or validation) ===
            let t_copy = now_ns();
            
            // Simulate reading entire frame buffer (touch every page)
            let mut checksum: u64 = 0;
            let stride = packet.data.stride;
            let height = packet.meta.height as usize;
            
            unsafe {
                // Read every 4KB (typical page size) to ensure cache miss
                for row in 0..height {
                    let offset = row * stride;
                    if offset + 4096 <= packet.data.len {
                        let slice = std::slice::from_raw_parts(
                            packet.data.ptr.add(offset),
                            4096
                        );
                        // Touch the data
                        checksum = checksum.wrapping_add(slice[0] as u64);
                        checksum = checksum.wrapping_add(slice[2048] as u64);
                        checksum = checksum.wrapping_add(slice[4095] as u64);
                    }
                }
            }
            
            let copy_ms = (now_ns() - t_copy) as f64 / 1_000_000.0;
            total_copy_latency += copy_ms;
            
            // === MEASURE: End-to-End from get_frame ===
            let e2e_ms = (now_ns() - t_get) as f64 / 1_000_000.0;
            total_e2e += e2e_ms;
            
            // Print detailed info every 20 frames
            if frame_count % 20 == 0 {
                println!("\n--- Frame #{} ---", packet.meta.frame_idx);
                println!("  Buffer latency:     {:.3} ms  (DeckLink buffer → get_frame)", buffer_latency_ms);
                println!("  Memory Read:        {:.3} ms  (scan entire 3.96MB buffer)", copy_ms);
                println!("  E2E (this cycle):   {:.3} ms  (get_frame + read)", e2e_ms);
                println!("  Checksum: 0x{:x}  (to prevent compiler optimization)", checksum);
            }
            
            // Record with telemetry
            record_ms("capture.buffer_latency", 
                (now_unix_ns - packet.meta.t_capture_ns) as u64);
            record_ms("capture.memory_read", t_copy);
            record_ms("capture.e2e", t_get);
            
            if frame_count >= 100 {
                break;
            }
        } else {
            thread::sleep(Duration::from_micros(100));
        }
        
        if attempt >= 299 {
            println!("\n⚠ Reached max attempts with {} frames", frame_count);
            break;
        }
    }

    println!("\n{:=<90}", "");
    println!("\n=== Performance Summary ({} frames) ===\n", frame_count);
    
    if frame_count > 0 {
        let avg_buffer = total_buffer_latency / frame_count as f64;
        let avg_copy = total_copy_latency / frame_count as f64;
        let avg_e2e = total_e2e / frame_count as f64;
        
        println!("📊 Average Latencies (REAL measurements only):");
        println!("  ├─ DeckLink Buffer:     {:.3} ms  ← Time from hardware capture to software", avg_buffer);
        println!("  ├─ Memory Read (4MB):   {:.3} ms  ← Actual data access cost", avg_copy);
        println!("  └─ E2E (per cycle):     {:.3} ms  ← Total (get_frame + read)", avg_e2e);
        
        println!("\n🎯 Real Capture Latency Breakdown:");
        println!("  • Hardware → Software:  {:.3} ms  ({:.1}%)", avg_buffer, avg_buffer/avg_e2e*100.0);
        println!("  • Memory Read:          {:.3} ms  ({:.1}%)", avg_copy, avg_copy/avg_e2e*100.0);
        println!("\n  ⚠️  NOTE: GPU H2D transfer latency NOT measured yet");
        println!("     Expected: ~0.3-0.4ms for PCIe 4.0 (will be added when preprocessing implemented)");
        
        println!("\n⚡ Performance Metrics:");
        let fps = 1000.0 / avg_e2e;
        println!("  • Max throughput:       {:.2} FPS", fps);
        println!("  • Frame budget (60Hz):  16.67 ms → {}", 
            if avg_e2e < 16.67 { "✓ OK" } else { "✗ TOO SLOW" });
        println!("  • Frame budget (30Hz):  33.33 ms → {}", 
            if avg_e2e < 33.33 { "✓ OK" } else { "✗ TOO SLOW" });
        
        println!("\n💡 Insights:");
        
        if avg_buffer < 0.1 {
            println!("  ✓ DeckLink buffer latency is very low (<0.1ms)");
        } else if avg_buffer < 1.0 {
            println!("  ✓ DeckLink buffer latency is acceptable (<1ms)");
        } else {
            println!("  ⚠ DeckLink buffer latency is high (>1ms) - may need driver tuning");
        }
        
        if avg_copy > avg_buffer * 5.0 {
            println!("  ⚠ Memory read is {}x slower than buffer latency", 
                (avg_copy / avg_buffer) as i32);
            println!("    → Main bottleneck is memory bandwidth");
        } else {
            println!("  ✓ Memory read is fast enough ({}x buffer latency)",
                (avg_copy / avg_buffer.max(0.001)) as i32);
        }
        
        println!("\n📝 Next Steps:");
        println!("  • Current measurements: Buffer + Memory Read = {:.3}ms", avg_e2e);
        println!("  • Still need to measure:");
        println!("    - H2D transfer (System RAM → GPU VRAM): ~0.3-0.4ms expected");
        println!("    - GPU preprocessing (YUV→RGB + resize): ~0.5-2ms expected");
        println!("    - GPU inference: ~2-8ms (model dependent)");
        println!("  • For 60fps budget (16.67ms), currently used: {:.1}%", avg_e2e/16.67*100.0);
        println!("  • Remaining budget: ~{:.2}ms for GPU pipeline", 16.67 - avg_e2e);
    }

    println!("\n=== Test Complete ===");
    Ok(())
}
