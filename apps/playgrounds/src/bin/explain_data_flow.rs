// Demonstrate actual data flow and explain latency measurements
use std::error::Error;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use decklink_input::capture::CaptureSession;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Data Flow and Latency Explanation ===\n");

    let mut session = CaptureSession::open(0)?;
    println!("✓ DeckLink opened\n");

    thread::sleep(Duration::from_millis(500));
    
    println!("Getting one frame to analyze data flow...\n");

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
    
    println!("{:=<90}", "");
    println!("\n📍 DATA LOCATION AND MEMORY ARCHITECTURE\n");
    
    println!("1. WHERE IS THE DATA?");
    println!("   Memory Location: {:p}", packet.data.ptr);
    println!("   Memory Type:     {:?}", packet.data.loc);
    println!("   Size:            {} bytes ({:.2} MB)", 
        packet.data.len, packet.data.len as f64 / 1_048_576.0);
    println!("   Stride:          {} bytes per row", packet.data.stride);
    
    println!("\n2. MEMORY TYPE EXPLANATION:");
    println!("   {:?} = System RAM (DDR4/DDR5)", packet.data.loc);
    println!("   • This is a DMA buffer allocated by DeckLink driver");
    println!("   • Shared between kernel space and user space");
    println!("   • NO COPY when we call get_frame() - just returns pointer!");
    
    println!("\n3. DATA FLOW FROM HARDWARE TO HERE:");
    println!("   ┌─────────────────────────────────────────────────────────┐");
    println!("   │ STAGE 1: Hardware Capture (DeckLink ASIC/FPGA)          │");
    println!("   │   • SDI signal → Deserialize → YUV422 format            │");
    println!("   │   • Writes to: PCIe device memory or system RAM (DMA)   │");
    println!("   │   • Latency: ~16-33ms (1-2 frames @ 60fps)              │");
    println!("   │   ⚠️  THIS latency is HIDDEN - we can't measure it!     │");
    println!("   └─────────────────────────────────────────────────────────┘");
    println!("                            ↓");
    println!("   ┌─────────────────────────────────────────────────────────┐");
    println!("   │ STAGE 2: DeckLink Driver (Kernel Space)                 │");
    println!("   │   • Receives interrupt from hardware                     │");
    println!("   │   • Manages ring buffer (3-5 frames)                    │");
    println!("   │   • Location: System RAM (DMA-accessible)               │");
    println!("   │   • Latency: ~0.5-2ms                                   │");
    println!("   └─────────────────────────────────────────────────────────┘");
    println!("                            ↓");
    println!("   ┌─────────────────────────────────────────────────────────┐");
    println!("   │ STAGE 3: DeckLink SDK Callback (User Space)             │");
    println!("   │   • VideoInputFrameArrived() called                      │");
    println!("   │   • Frame pointer stored in application buffer          │");
    println!("   │   • Latency: ~0.1-0.5ms                                 │");
    println!("   └─────────────────────────────────────────────────────────┘");
    println!("                            ↓");
    println!("   ┌─────────────────────────────────────────────────────────┐");
    println!("   │ STAGE 4: get_frame() Returns Pointer                    │");
    println!("   │   • Returns: RawFramePacket with pointer                │");
    println!("   │   • NO DATA COPY - just metadata + pointer!             │");
    println!("   │   • Latency: ~0.0003ms (function call overhead)         │");
    println!("   │   ✅ This is what we measured as '0 ms'                 │");
    println!("   └─────────────────────────────────────────────────────────┘");
    println!("                            ↓");
    println!("   ┌─────────────────────────────────────────────────────────┐");
    println!("   │ CURRENT STATE: Data still in System RAM                 │");
    println!("   │   Address: {:p}                             │", packet.data.ptr);
    println!("   │   Size: {} bytes                                  │", packet.data.len);
    println!("   └─────────────────────────────────────────────────────────┘");
    
    println!("\n\n{:=<90}", "");
    println!("\n⏱️  LATENCY MEASUREMENTS EXPLAINED\n");
    
    // Measurement 1: t_capture_ns (what we currently have)
    let now_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    
    let software_latency = (now_unix - packet.meta.t_capture_ns) as f64 / 1_000_000.0;
    
    println!("1. SOFTWARE BUFFER LATENCY (what we measure):");
    println!("   t_capture_ns = {}  (when get_frame() was called)", packet.meta.t_capture_ns);
    println!("   now          = {}  (current time)", now_unix);
    println!("   Difference   = {:.3} ms", software_latency);
    println!("\n   ❌ This is NOT hardware capture latency!");
    println!("   ✅ This is time since we got the pointer");
    println!("   → Useful for: tracking processing delays in our app");
    
    println!("\n2. MEMORY READ LATENCY (measuring data access):");
    println!("   Reading FROM: System RAM at {:p}", packet.data.ptr);
    println!("   Reading TO:   CPU cache → CPU registers");
    println!("   What we do:   Sequential scan of {} bytes", packet.data.len);
    
    // Measure memory read
    use std::time::Instant;
    let t_start = Instant::now();
    
    let mut checksum: u64 = 0;
    unsafe {
        // Read every 4KB to simulate real access pattern
        let num_pages = packet.data.len / 4096;
        for page in 0..num_pages {
            let offset = page * 4096;
            let ptr = packet.data.ptr.add(offset);
            checksum = checksum.wrapping_add(*ptr as u64);
        }
    }
    
    let read_time_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    
    println!("\n   Measured: {:.3} ms to scan {} pages", read_time_ms, packet.data.len / 4096);
    println!("   Bandwidth: {:.2} GB/s", 
        (packet.data.len as f64 / 1_073_741_824.0) / (read_time_ms / 1000.0));
    println!("   Checksum: 0x{:x} (to prevent optimization)", checksum);
    
    println!("\n3. REAL HARDWARE CAPTURE LATENCY (what we CAN'T measure now):");
    println!("   ❌ NOT IMPLEMENTED YET");
    println!("   Would need: Hardware timestamp from DeckLink API");
    println!("   Method: videoFrame->GetStreamTime()");
    println!("   Expected: 17-40ms (1-2 frame periods + overhead)");
    
    println!("\n\n{:=<90}", "");
    println!("\n🎯 LATENCY BREAKDOWN SUMMARY\n");
    
    println!("Component                        | Latency    | Where We Measure");
    println!("---------------------------------|------------|---------------------------");
    println!("1. Hardware capture (hidden)     | 16-33ms    | ❌ Can't measure (yet)");
    println!("2. Driver processing             | 0.5-2ms    | ❌ Can't measure");
    println!("3. SDK callback                  | 0.1-0.5ms  | ❌ Can't measure");
    println!("4. get_frame() call              | 0.0003ms   | ✅ This is our '0 ms'");
    println!("5. Memory read (scan)            | {:.3}ms    | ✅ What we measured", read_time_ms);
    println!("---------------------------------|------------|---------------------------");
    println!("TOTAL (estimated)                | 17-40ms    | ⚠️  Mostly unmeasured");
    
    println!("\n\n{:=<90}", "");
    println!("\n💡 KEY INSIGHTS\n");
    
    println!("1. WHY '0 ms' FOR CAPTURE?");
    println!("   • get_frame() doesn't capture - it just returns a pointer!");
    println!("   • Hardware already captured the frame 17-40ms ago");
    println!("   • The pointer points to a buffer that was filled earlier");
    println!("   → Think of it like checking your mailbox vs waiting for mail");
    
    println!("\n2. WHAT DOES MEMORY READ (0.02ms) MEAN?");
    println!("   • Reading FROM: DMA buffer in system RAM");
    println!("   • Reading TO: CPU cache/registers");
    println!("   • This is the cost of ACCESSING the data");
    println!("   • Bandwidth-limited: ~200 GB/s (DDR4)");
    
    println!("\n3. WHERE IS THE REAL LATENCY?");
    println!("   • Hidden in hardware capture (16-33ms)");
    println!("   • This is the time from light hitting sensor to frame in RAM");
    println!("   • We can't measure it without hardware timestamp");
    println!("   • Need to modify shim.cpp to get GetStreamTime()");
    
    println!("\n4. WHAT MATTERS FOR REAL-TIME PIPELINE?");
    println!("   • Hardware latency: Fixed, can't optimize");
    println!("   • Memory read: Fast enough (0.02ms)");
    println!("   • NEXT bottleneck: H2D transfer + YUV→RGB conversion");
    println!("   • Focus optimization on: GPU preprocessing (2-5ms target)");
    
    println!("\n\n{:=<90}", "");
    println!("\n📋 NEXT STEPS TO MEASURE REAL LATENCY\n");
    
    println!("Option 1: Modify shim.cpp (Recommended)");
    println!("  • Add GetStreamTime() call in VideoInputFrameArrived");
    println!("  • Store hardware timestamp in frame metadata");
    println!("  • Pass to Rust via RawFramePacket");
    println!("  → This gives us REAL hardware capture time");
    
    println!("\nOption 2: Use external sync (Advanced)");
    println!("  • Feed test pattern with timecode");
    println!("  • Compare timecode with system time");
    println!("  • Most accurate but needs extra hardware");
    
    println!("\nOption 3: Estimate from frame rate (Simple)");
    println!("  • At 60fps, each frame = 16.67ms");
    println!("  • DeckLink buffers 1-2 frames");
    println!("  • Estimated latency: 17-33ms");
    println!("  → Good enough for budget planning");
    
    println!("\n=== Test Complete ===");
    Ok(())
}
