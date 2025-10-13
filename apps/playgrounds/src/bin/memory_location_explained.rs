// Explain exactly where data is located at each stage
use std::error::Error;
use std::thread;
use std::time::Duration;

use decklink_input::capture::CaptureSession;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Memory Location Deep Dive ===\n");

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
    println!("\n🗺️  PHYSICAL MEMORY MAP\n");
    
    println!("┌─────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ HARDWARE COMPONENTS IN YOUR SYSTEM                                                      │");
    println!("└─────────────────────────────────────────────────────────────────────────────────────────┘");
    
    println!("\n1️⃣  DeckLink Card (PCIe Device)");
    println!("   ┌────────────────────────────────────────┐");
    println!("   │  • SDI Input Port                      │");
    println!("   │  • ASIC/FPGA (capture logic)           │");
    println!("   │  • Small onboard buffer (~8-16 MB)     │  ← Hardware buffer (temporary)");
    println!("   │  • PCIe interface                      │");
    println!("   └────────────────────────────────────────┘");
    println!("              │");
    println!("              │ PCIe Bus (DMA transfer)");
    println!("              ↓");
    
    println!("\n2️⃣  System RAM (Computer's Main Memory)");
    println!("   ┌────────────────────────────────────────┐");
    println!("   │  DDR4/DDR5 Memory Modules              │");
    println!("   │  Size: 16GB / 32GB / 64GB / etc.       │");
    println!("   │  Speed: 3200MHz / 4800MHz / etc.       │");
    println!("   │                                        │");
    println!("   │  DMA Buffer Area:                      │  ← **THIS IS WHERE DATA IS!**");
    println!("   │  Address: {:p}            │", packet.data.ptr);
    println!("   │  Size: {:.2} MB (one frame)           │", packet.data.len as f64 / 1_048_576.0);
    println!("   └────────────────────────────────────────┘");
    println!("              │");
    println!("              │ PCIe Bus (H2D transfer)");
    println!("              ↓");
    
    println!("\n3️⃣  GPU Card (NVIDIA RTX/etc.)");
    println!("   ┌────────────────────────────────────────┐");
    println!("   │  GPU Chip (CUDA cores, Tensor cores)   │");
    println!("   │  VRAM: 8GB / 12GB / 24GB / etc.        │");
    println!("   │  Speed: GDDR6/GDDR6X (very fast)       │");
    println!("   │                                        │");
    println!("   │  (Data not here yet!)                  │  ← Will be copied here later
    println!("   └────────────────────────────────────────┘");
    
    println!("\n\n{:=<100}", "");
    println!("\n📍 DATA JOURNEY IN DETAIL\n");
    
    println!("╔═══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║ STEP 1: Hardware Capture                                                             ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════╝");
    println!("  SDI Signal");
    println!("     ↓");
    println!("  [DeckLink ASIC] ← Deserializes SDI to YUV422 pixels");
    println!("     ↓");
    println!("  [DeckLink onboard buffer] ← Tiny hardware buffer (8-16 MB)");
    println!("     │                         • Only holds 2-4 frames");
    println!("     │                         • Very fast, on-card memory");
    println!("     │                         • ⚠️  Data doesn't stay here long!");
    println!("     │");
    println!("     │ DMA Transfer (Direct Memory Access)");
    println!("     │ • Hardware automatically copies");
    println!("     │ • No CPU involvement");
    println!("     │ • Speed: ~3-4 GB/s (PCIe 3.0 x4)");
    println!("     ↓");
    println!("  [System RAM] ← **DATA ARRIVES HERE**");
    println!("     • Address: {:p}", packet.data.ptr);
    println!("     • This is DDR4/DDR5 on motherboard");
    println!("     • Shared between kernel and user space");
    println!("     • Size: {:.2} MB per frame", packet.data.len as f64 / 1_048_576.0);
    
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║ STEP 2: get_frame() Returns Pointer                                                  ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════╝");
    println!("  [System RAM] ← Data is STILL here");
    println!("     • No copy happened!");
    println!("     • get_frame() just returns a pointer");
    println!("     • RawFramePacket.data.ptr = {:p}", packet.data.ptr);
    println!("     • This pointer points to System RAM");
    
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║ STEP 3: CPU Access (if needed)                                                       ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════╝");
    println!("  [System RAM] ← Still here");
    println!("     ↓ Read (via CPU cache)");
    println!("  [CPU Registers]");
    println!("     • CPU can read directly from System RAM");
    println!("     • Very fast: ~257 GB/s bandwidth");
    println!("     • No copy needed");
    
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║ STEP 4: H2D Transfer (THIS IS YOUR QUESTION!)                                        ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════╝");
    println!("  [System RAM] ← **THIS IS THE SOURCE**");
    println!("     • Address: {:p}", packet.data.ptr);
    println!("     • Location: Computer's DDR4/DDR5");
    println!("     • NOT on DeckLink card anymore!");
    println!("     ↓");
    println!("     │ cudaMemcpy(dst_gpu, src_cpu, size, cudaMemcpyHostToDevice)");
    println!("     │ • src_cpu = {:p} (System RAM)", packet.data.ptr);
    println!("     │ • dst_gpu = 0x... (GPU VRAM address)");
    println!("     │ • Copies via PCIe bus");
    println!("     │ • Speed: ~12 GB/s (PCIe 3.0 x16) or ~24 GB/s (PCIe 4.0 x16)");
    println!("     ↓");
    println!("  [GPU VRAM] ← **DATA COPIED HERE**");
    println!("     • Now GPU can process it");
    println!("     • Original in System RAM still exists");
    
    println!("\n\n{:=<100}", "");
    println!("\n❓ ANSWERING YOUR QUESTION\n");
    
    println!("Q: System RAM หมายถึงใน DeckLink หรือ Computer ที่จะ copy ไป VRAM?");
    println!("\n✅ ANSWER: System RAM = Computer's RAM (ไม่ใช่ DeckLink!)\n");
    
    println!("DETAILED EXPLANATION:");
    println!("\n1. DeckLink Card มี memory เล็กๆ ของตัวเอง");
    println!("   • ขนาด: ~8-16 MB (เก็บได้แค่ 2-4 frames)");
    println!("   • ใช้เป็น buffer ชั่วคราว");
    println!("   • ⚠️  Data ไม่อยู่ที่นี่นาน! จะถูก DMA ส่งไปทันที");
    
    println!("\n2. System RAM = Computer's Main Memory");
    println!("   • นี่คือ RAM ที่ติดบน motherboard");
    println!("   • DDR4/DDR5 modules (16GB, 32GB, etc.)");
    println!("   • ✅ DATA IS HERE after DMA transfer");
    println!("   • Address space: 0x00007... (64-bit virtual address)");
    println!("   • Shared between all applications and kernel");
    
    println!("\n3. When we call cudaMemcpy():");
    println!("   • SOURCE: System RAM (Computer's DDR4/DDR5)");
    println!("   •         Address: {:p}", packet.data.ptr);
    println!("   • DESTINATION: GPU VRAM (on GPU card)");
    println!("   • PATH: System RAM → PCIe Bus → GPU VRAM");
    println!("   • This is the '0.3-0.4ms' H2D latency you asked about");
    
    println!("\n4. Why can't GPU use data in System RAM directly?");
    println!("   • GPU and System RAM are in different address spaces");
    println!("   • GPU can only process data in its own VRAM");
    println!("   • PCIe bus is too slow for real-time GPU access");
    println!("   • Must copy first: System RAM → VRAM");
    
    println!("\n\n{:=<100}", "");
    println!("\n🔍 MEMORY HIERARCHY SUMMARY\n");
    
    println!("┌────────────────────┬──────────────┬────────────┬─────────────────────────────┐");
    println!("│ Memory Type        │ Location     │ Size       │ Access Speed                │");
    println!("├────────────────────┼──────────────┼────────────┼─────────────────────────────┤");
    println!("│ DeckLink buffer    │ DeckLink card│ 8-16 MB    │ N/A (internal, auto DMA)    │");
    println!("│ System RAM         │ Motherboard  │ 16-64 GB   │ ~257 GB/s (CPU access)      │");
    println!("│                    │              │            │ ~12-24 GB/s (PCIe to GPU)   │");
    println!("│ GPU VRAM           │ GPU card     │ 8-24 GB    │ ~900 GB/s (GPU access)      │");
    println!("│ CPU Cache          │ CPU chip     │ 8-64 MB    │ ~1 TB/s                     │");
    println!("│ CPU Registers      │ CPU chip     │ <1 KB      │ Instant (1 cycle)           │");
    println!("└────────────────────┴──────────────┴────────────┴─────────────────────────────┘");
    
    println!("\n🎯 CURRENT STATE:");
    println!("   • Data location: System RAM (computer's DDR4/DDR5)");
    println!("   • Pointer: {:p}", packet.data.ptr);
    println!("   • Next step: Copy to GPU VRAM via cudaMemcpy()");
    println!("   • H2D latency: ~0.3-0.4 ms (must happen!)");
    
    println!("\n=== Explanation Complete ===");
    Ok(())
}