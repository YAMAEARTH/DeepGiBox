// Explore RawFramePacket and measure real capture latency
use std::error::Error;
use std::thread;
use std::time::Duration;

use decklink_input::capture::CaptureSession;
use telemetry::{now_ns, record_ms};

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== RawFramePacket Exploration & Real Latency Test ===\n");

    // Open DeckLink capture session
    println!("Opening DeckLink device 0...");
    let t_open = now_ns();
    let mut session = CaptureSession::open(0)?;
    let open_ms = (now_ns() - t_open) as f64 / 1_000_000.0;
    println!("✓ Opened in {:.3} ms\n", open_ms);

    // Wait for device to stabilize
    println!("Waiting for device to stabilize (500ms)...");
    thread::sleep(Duration::from_millis(500));
    println!("✓ Ready\n");

    println!("Capturing 50 frames and exploring RawFramePacket structure...");
    println!("{:=<100}", "");

    let mut total_get_frame_ms = 0.0;
    let mut total_memcpy_simulate_ms = 0.0;
    let mut total_validation_ms = 0.0;
    let mut frame_count = 0;
    let mut total_bytes = 0u64;

    // Capture loop
    for attempt in 0..200 {
        // Measure: get_frame() call
        let t_get = now_ns();
        let packet_option = session.get_frame()?;
        let get_ms = (now_ns() - t_get) as f64 / 1_000_000.0;
        
        if let Some(packet) = packet_option {
            frame_count += 1;
            total_get_frame_ms += get_ms;
            
            // === EXPLORE RawFramePacket ===
            let meta = &packet.meta;
            let data = &packet.data;
            
            // Measure: Frame validation
            let t_validate = now_ns();
            let is_valid = meta.width > 0 
                && meta.height > 0 
                && meta.stride_bytes > 0 
                && data.len > 0;
            let validate_ms = (now_ns() - t_validate) as f64 / 1_000_000.0;
            total_validation_ms += validate_ms;
            
            // Measure: Simulate memory access/copy (read first and last 1KB)
            let t_memcpy = now_ns();
            if is_valid && data.len >= 2048 {
                unsafe {
                    // Read first 1KB
                    let _first = std::slice::from_raw_parts(data.ptr, 1024);
                    let _sum_first: u64 = _first.iter().map(|&b| b as u64).sum();
                    
                    // Read last 1KB
                    let _last = std::slice::from_raw_parts(data.ptr.add(data.len - 1024), 1024);
                    let _sum_last: u64 = _last.iter().map(|&b| b as u64).sum();
                }
            }
            let memcpy_ms = (now_ns() - t_memcpy) as f64 / 1_000_000.0;
            total_memcpy_simulate_ms += memcpy_ms;
            
            total_bytes += data.len as u64;
            
            // Print detailed info every 10 frames
            if frame_count % 10 == 0 {
                println!("\n--- Frame #{} ---", meta.frame_idx);
                println!("  FrameMeta:");
                println!("    source_id:      {}", meta.source_id);
                println!("    width×height:   {}×{}", meta.width, meta.height);
                println!("    stride_bytes:   {}", meta.stride_bytes);
                println!("    pixfmt:         {:?}", meta.pixfmt);
                println!("    colorspace:     {:?}", meta.colorspace);
                println!("    frame_idx:      {}", meta.frame_idx);
                println!("    pts_ns:         {}", meta.pts_ns);
                println!("    t_capture_ns:   {}", meta.t_capture_ns);
                
                println!("  MemRef:");
                println!("    ptr:            {:p}", data.ptr);
                println!("    len:            {} bytes ({:.2} MB)", data.len, data.len as f64 / 1_048_576.0);
                println!("    stride:         {}", data.stride);
                println!("    loc:            {:?}", data.loc);
                
                println!("  Latencies:");
                println!("    get_frame():    {:.6} ms", get_ms);
                println!("    validation:     {:.6} ms", validate_ms);
                println!("    memcpy(2KB):    {:.6} ms", memcpy_ms);
                
                // Calculate time since capture
                let now_ns_unix = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;
                let latency_from_capture = (now_ns_unix - meta.t_capture_ns) as f64 / 1_000_000.0;
                println!("    since capture:  {:.3} ms", latency_from_capture);
            }
            
            // Record per-frame telemetry
            record_ms("frame.get", t_get);
            
            if frame_count >= 50 {
                break;
            }
        } else {
            // No frame available
            thread::sleep(Duration::from_micros(100));
        }
        
        if attempt >= 199 {
            println!("\n⚠ Warning: Reached max attempts with only {} frames", frame_count);
            break;
        }
    }

    println!("\n{:=<100}", "");
    println!("\n=== Performance Summary ===");
    
    if frame_count > 0 {
        let avg_get = total_get_frame_ms / frame_count as f64;
        let avg_validate = total_validation_ms / frame_count as f64;
        let avg_memcpy = total_memcpy_simulate_ms / frame_count as f64;
        let avg_total = avg_get + avg_validate + avg_memcpy;
        let avg_bytes = total_bytes / frame_count as u64;
        
        println!("\nFrames captured:           {}", frame_count);
        println!("Total data transferred:    {:.2} MB", total_bytes as f64 / 1_048_576.0);
        println!("Average frame size:        {:.2} MB", avg_bytes as f64 / 1_048_576.0);
        
        println!("\n--- Average Latencies ---");
        println!("get_frame():               {:.6} ms", avg_get);
        println!("validation:                {:.6} ms", avg_validate);
        println!("memcpy (2KB sample):       {:.6} ms", avg_memcpy);
        println!("total per frame:           {:.6} ms", avg_total);
        
        println!("\n--- Throughput ---");
        let fps = 1000.0 / avg_total;
        let bandwidth_mbps = (avg_bytes as f64 * fps * 8.0) / 1_000_000.0;
        
        println!("Theoretical max FPS:       {:.2}", fps);
        println!("Data bandwidth:            {:.2} Mbps", bandwidth_mbps);
        
        println!("\n--- Frame Budget Check ---");
        println!("16.67 ms (60 FPS):         {}", if avg_total < 16.67 { "✓ OK" } else { "✗ TOO SLOW" });
        println!("33.33 ms (30 FPS):         {}", if avg_total < 33.33 { "✓ OK" } else { "✗ TOO SLOW" });
        
        println!("\n--- Interpretation ---");
        if avg_get < 0.001 {
            println!("⚠ get_frame() < 0.001ms suggests it's reading pre-buffered data");
            println!("  (Real capture happens in DeckLink driver/hardware)");
        }
        if avg_memcpy > avg_get * 10.0 {
            println!("⚠ Memory access is 10x slower than get_frame()");
            println!("  (This shows the real bottleneck will be data processing)");
        }
    }

    println!("\n=== Test Complete ===");
    Ok(())
}
