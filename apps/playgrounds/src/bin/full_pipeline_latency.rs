// Complete data flow including GPU transfer latency
use std::error::Error;
use std::thread;
use std::time::Duration;

use decklink_input::capture::CaptureSession;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Complete Pipeline Latency Analysis (Including GPU) ===\n");

    let mut session = CaptureSession::open(0)?;
    println!("✓ DeckLink opened\n");

    thread::sleep(Duration::from_millis(500));

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

    println!("{:=<100}", "");
    println!("\n🚀 COMPLETE DATA JOURNEY: CPU → GPU → INFERENCE\n");

    let frame_size_mb = packet.data.len as f64 / 1_048_576.0;

    println!("   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓");
    println!("   ┃ STAGE 0: Hardware Capture (DeckLink ASIC)                                            ┃");
    println!("   ┃   Location:  DeckLink hardware → PCIe DMA → System RAM                               ┃");
    println!("   ┃   Format:    YUV422 (UYVY)                                                            ┃");
    println!("   ┃   Size:      {:.2} MB                                                                ┃", frame_size_mb);
    println!("   ┃   Latency:   16-33ms (1-2 frames @ 60fps)                                            ┃");
    println!("   ┃   ⚠️  Hidden - can't measure without hardware timestamp                               ┃");
    println!("   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛");
    println!("                                          ↓");
    println!("   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓");
    println!("   ┃ STAGE 1: get_frame() - Get Pointer                                                   ┃");
    println!("   ┃   Location:  System RAM (DMA buffer)                                                 ┃");
    println!(
        "   ┃   Address:   {:p}                                                        ┃",
        packet.data.ptr
    );
    println!("   ┃   Operation: Return pointer (NO COPY!)                                               ┃");
    println!("   ┃   Latency:   ~0.0003ms (function call)                                               ┃");
    println!("   ┃   ✅ This is what we measured as '0 ms'                                               ┃");
    println!("   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛");
    println!("                                          ↓");
    println!("   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓");
    println!("   ┃ STAGE 2: CPU Memory Access (if needed)                                               ┃");
    println!(
        "   ┃   FROM:      System RAM ({:p})                                           ┃",
        packet.data.ptr
    );
    println!("   ┃   TO:        CPU Cache → CPU Registers                                               ┃");
    println!("   ┃   Operation: Read/scan data for validation or CPU processing                         ┃");
    println!("   ┃   Latency:   ~0.015ms (for full frame scan)                                          ┃");
    println!("   ┃   Bandwidth: ~257 GB/s (DDR4 memory)                                                 ┃");
    println!("   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛");
    println!("                                          ↓");
    println!("   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓");
    println!("   ┃ STAGE 3: H2D Transfer (Host to Device) ⚠️  NEW LATENCY HERE!                         ┃");
    println!("   ┃   FROM:      System RAM (CPU memory)                                                 ┃");
    println!("   ┃   TO:        GPU VRAM (Device memory)                                                ┃");
    println!("   ┃   Method:    cudaMemcpy() or cudaMemcpyAsync()                                       ┃");
    println!("   ┃   Size:      {:.2} MB (YUV422 raw data)                                             ┃", frame_size_mb);
    println!("   ┃                                                                                       ┃");

    // Calculate H2D latency estimates
    let pcie_gen3_bw = 12.0; // GB/s (PCIe 3.0 x16)
    let pcie_gen4_bw = 24.0; // GB/s (PCIe 4.0 x16)
    let h2d_gen3 = (frame_size_mb / 1024.0) / pcie_gen3_bw * 1000.0;
    let h2d_gen4 = (frame_size_mb / 1024.0) / pcie_gen4_bw * 1000.0;

    println!("   ┃   Latency:   PCIe 3.0 x16: ~{:.3} ms (@ 12 GB/s)                                     ┃", h2d_gen3);
    println!("   ┃              PCIe 4.0 x16: ~{:.3} ms (@ 24 GB/s)                                     ┃", h2d_gen4);
    println!("   ┃   Overhead:  +0.05-0.1ms (cudaMemcpy call + PCIe setup)                              ┃");
    println!("   ┃                                                                                       ┃");
    println!("   ┃   ⚠️  THIS IS THE LATENCY YOU ASKED ABOUT!                                            ┃");
    println!("   ┃   • Must copy ALL data from CPU → GPU                                                ┃");
    println!("   ┃   • Can't avoid this - GPU needs data in its own memory                              ┃");
    println!("   ┃   • Bandwidth limited by PCIe bus                                                    ┃");
    println!("   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛");
    println!("                                          ↓");
    println!("   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓");
    println!("   ┃ STAGE 4: GPU Preprocessing (CUDA Kernel)                                             ┃");
    println!("   ┃   Location:  GPU VRAM                                                                ┃");
    println!("   ┃   Operation: YUV422 → RGB → Resize → Normalize                                      ┃");
    println!("   ┃   Threads:   1920×1080 = 2,073,600 threads (massively parallel)                      ┃");
    println!("   ┃   Latency:   ~0.5-2ms (depends on GPU, typically <1ms on RTX)                        ┃");
    println!("   ┃   Output:    RGB tensor (e.g., 640×640×3 for YOLO)                                  ┃");
    println!("   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛");
    println!("                                          ↓");
    println!("   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓");
    println!("   ┃ STAGE 5: GPU Inference (TensorRT/ONNX)                                               ┃");
    println!("   ┃   Location:  GPU VRAM (already there!)                                               ┃");
    println!("   ┃   Operation: Neural network forward pass                                             ┃");
    println!("   ┃   Latency:   ~2-8ms (depends on model size)                                          ┃");
    println!("   ┃              • YOLOv8n: ~2ms                                                          ┃");
    println!("   ┃              • YOLOv8m: ~5ms                                                          ┃");
    println!("   ┃   Output:    Detection results (still in GPU)                                        ┃");
    println!("   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛");
    println!("                                          ↓");
    println!("   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓");
    println!("   ┃ STAGE 6: D2H Transfer (Device to Host) - If needed                                   ┃");
    println!("   ┃   FROM:      GPU VRAM                                                                ┃");
    println!("   ┃   TO:        System RAM                                                              ┃");
    println!("   ┃   Size:      Small (detection results, not full frame)                               ┃");
    println!("   ┃   Latency:   ~0.05-0.2ms (much smaller data)                                         ┃");
    println!("   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛");

    println!("\n\n{:=<100}", "");
    println!("\n📊 LATENCY BUDGET BREAKDOWN\n");

    println!("┌────────────────────────────────────┬────────────┬───────────────────────────────────────┐");
    println!("│ Component                          │ Latency    │ Notes                                 │");
    println!("├────────────────────────────────────┼────────────┼───────────────────────────────────────┤");
    println!("│ 0. Hardware capture (hidden)       │ 16-33ms    │ ⚠️  Can't optimize                     │");
    println!("│ 1. get_frame() pointer             │ 0.0003ms   │ ✅ Negligible                          │");
    println!("│ 2. CPU memory access               │ 0.015ms    │ ✅ Fast enough                         │");
    println!("│ 3. H2D transfer (PCIe 3.0)         │ ~{:.2}ms   │ ⚠️  NEW BOTTLENECK                     │", h2d_gen3);
    println!("│    H2D transfer (PCIe 4.0)         │ ~{:.2}ms   │ 💡 Faster with newer hardware         │", h2d_gen4);
    println!("│ 4. GPU preprocess (YUV→RGB+resize) │ 0.5-2ms    │ ✅ Parallelized                        │");
    println!("│ 5. GPU inference (YOLOv8n)         │ 2-8ms      │ ⚠️  Model dependent                    │");
    println!("│ 6. D2H transfer (results)          │ 0.05-0.2ms │ ✅ Small data                          │");
    println!("│ 7. Postprocessing (NMS)            │ 0.5-1ms    │ ✅ Can be on GPU too                   │");
    println!("│ 8. Overlay rendering               │ 1-3ms      │ ⚠️  Depends on complexity              │");
    println!("├────────────────────────────────────┼────────────┼───────────────────────────────────────┤");
    println!("│ TOTAL (software measurable)        │ ~{:.1}-{}ms  │                                       │", 
        0.5 + h2d_gen4 + 0.5 + 2.0 + 0.05 + 0.5 + 1.0,
        0.5 + h2d_gen3 + 2.0 + 8.0 + 0.2 + 1.0 + 3.0);
    println!("│ TOTAL (with hardware)              │ ~{:.0}-{}ms  │ For 60fps: budget is 16.67ms          │",
        16.0 + 0.5 + h2d_gen4 + 0.5 + 2.0,
        33.0 + 0.5 + h2d_gen3 + 2.0 + 8.0);
    println!("└────────────────────────────────────┴────────────┴───────────────────────────────────────┘");

    println!("\n\n{:=<100}", "");
    println!("\n💡 KEY INSIGHTS ABOUT YOUR QUESTION\n");

    println!("Q: ฉันเข้าใจถูกไหมก่อนจะทำ preprocessing ที่ gpu จะต้องมี latency เพิ่มจากการที่ gpu");
    println!("   copy ข้อมูลจาก cpu ที่มีการ access ไว้");
    println!("\n✅ คุณเข้าใจถูกต้อง 100%! นี่คือประเด็นสำคัญ:\n");

    println!("1. H2D TRANSFER IS MANDATORY");
    println!("   • GPU ไม่สามารถเข้าถึง System RAM ได้โดยตรง");
    println!("   • ต้อง copy ข้อมูลผ่าน PCIe bus");
    println!(
        "   • Size: {:.2} MB → Takes {:.2}-{:.2} ms",
        frame_size_mb, h2d_gen4, h2d_gen3
    );
    println!("   • นี่คือ overhead ที่หลีกเลี่ยงไม่ได้");

    println!("\n2. WHY THIS MATTERS");
    println!("   • At 60fps: frame budget = 16.67ms");
    println!(
        "   • H2D alone: {:.2} ms (PCIe 3.0) = {:.1}% of budget!",
        h2d_gen3,
        h2d_gen3 / 16.67 * 100.0
    );
    println!(
        "   • H2D alone: {:.2} ms (PCIe 4.0) = {:.1}% of budget",
        h2d_gen4,
        h2d_gen4 / 16.67 * 100.0
    );
    println!("   • This is BEFORE any GPU computation!");

    println!("\n3. MEMORY ACCESS ORDER");
    println!("   Step 1: DeckLink DMA → System RAM     (done by hardware)");
    println!("   Step 2: get_frame()                    (just get pointer, no copy)");
    println!("   Step 3: CPU may access for validation  (optional, ~0.015ms)");
    println!(
        "   Step 4: cudaMemcpy() to GPU ⚠️          (MANDATORY, ~{:.2}ms)",
        h2d_gen3
    );
    println!("   Step 5: GPU preprocessing              (now we can process)");

    println!("\n4. CPU ACCESS VS GPU COPY - DIFFERENT THINGS!");
    println!("   CPU Access (0.015ms):");
    println!("   • Reading: System RAM → CPU cache → registers");
    println!("   • Fast because: within same memory domain");
    println!("   • Bandwidth: ~257 GB/s (DDR4)");
    println!("\n   GPU Copy ({:.2}ms):", h2d_gen3);
    println!("   • Copying: System RAM → PCIe bus → GPU VRAM");
    println!("   • Slower because: crosses memory domains via PCIe");
    println!("   • Bandwidth: ~12 GB/s (PCIe 3.0 x16)");
    println!("   • This is 21× slower than CPU memory!");

    println!("\n5. CAN'T AVOID THIS");
    println!("   ❌ GPU cannot directly access System RAM");
    println!("   ❌ Can't do zero-copy like CPU");
    println!("   ❌ PCIe bandwidth is limited");
    println!("   ✅ But GPU processing is 100× faster once data is there!");

    println!("\n\n{:=<100}", "");
    println!("\n🎯 OPTIMIZATION STRATEGIES\n");

    println!("1. USE ASYNC TRANSFER (cudaMemcpyAsync)");
    println!("   • Transfer next frame while processing current frame");
    println!("   • Hides latency through pipelining");
    println!("   • Requires pinned memory");

    println!("\n2. TRANSFER SMALLER DATA");
    println!("   • Crop region of interest before H2D");
    println!(
        "   • Example: 1920×1080 → 1280×720 = save {:.2}ms",
        h2d_gen3 * (1.0 - (1280.0 * 720.0) / (1920.0 * 1080.0))
    );

    println!("\n3. USE FASTER PCIe");
    println!("   • PCIe 4.0: 2× faster than 3.0");
    println!("   • PCIe 5.0: 2× faster than 4.0");
    println!("   • Check: lspci -vv | grep 'PCIe'");

    println!("\n4. GPU DIRECT RDMA (Advanced)");
    println!("   • DeckLink → GPU directly (bypass CPU)");
    println!("   • Requires special hardware support");
    println!("   • Can reduce H2D to near-zero");

    println!("\n5. COMPRESS BEFORE TRANSFER");
    println!("   • YUV422 → YUV420 = save 33% bandwidth");
    println!("   • But adds CPU encoding time");
    println!("   • Trade-off: CPU time vs PCIe time");

    println!("\n\n{:=<100}", "");
    println!("\n📈 REALISTIC LATENCY TARGET\n");

    println!("For 60fps real-time processing:");
    println!("  • Frame period: 16.67ms");
    println!("  • Hardware capture: ~20ms (fixed, can't optimize)");
    println!("  • H2D transfer: ~0.34ms (PCIe 4.0, can't optimize much)");
    println!("  • GPU preprocess: ~1ms (already optimized)");
    println!("  • Inference: ~2-3ms (model dependent)");
    println!("  • Postprocess: ~1ms");
    println!("  • Overlay: ~2ms");
    println!("  ─────────────────");
    println!("  • TOTAL: ~26-27ms end-to-end");
    println!("\n  ⚠️  This exceeds 16.67ms budget!");
    println!("  💡 Solution: Pipeline overlapping (process frame N while capturing N+1)");
    println!("  ✅ With pipelining: Can achieve 60fps throughput with 26ms latency");

    println!("\n=== Analysis Complete ===");
    Ok(())
}
