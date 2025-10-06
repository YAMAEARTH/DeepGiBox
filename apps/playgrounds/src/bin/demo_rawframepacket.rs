use decklink_input::{ColorSpace, MemLoc, MemRef, PixelFormat, RawFramePacket, FrameMeta};

fn main() {
    println!("=== Demo: Creating and Displaying RawFramePacket ===\n");
    
    // Create a mock RawFramePacket (simulating what DeckLink would produce)
    let mock_packet = create_mock_rawframepacket();
    
    println!("Created a mock RawFramePacket simulating DeckLink output:\n");
    print_packet_details(&mock_packet);
    
    println!("\n=== Verification ===");
    verify_packet_structure(&mock_packet);
    
    println!("\n✓ RawFramePacket structure is ready for the pipeline!");
    println!("  Next stage: PreprocessCUDA will receive this packet and:");
    println!("  1. Read meta.pixfmt (YUV422_8) to select YUV→RGB kernel");
    println!("  2. Use meta.colorspace (BT709) for correct color conversion");
    println!("  3. Access data from data.ptr with data.stride and data.len");
    println!("  4. Pass meta (FrameMeta) to next stages in the pipeline");
}

fn create_mock_rawframepacket() -> RawFramePacket {
    // Simulate 1920x1080 YUV422 frame
    let width = 1920u32;
    let height = 1080u32;
    let stride_bytes = width * 2; // YUV422 = 2 bytes per pixel
    let total_size = stride_bytes as usize * height as usize;
    
    // Allocate mock buffer (in real scenario, this comes from DeckLink)
    let mock_buffer: Vec<u8> = vec![0x80; total_size]; // Fill with mid-gray YUV values
    let buffer_ptr = mock_buffer.as_ptr() as *mut u8;
    
    // Don't drop the buffer (in real scenario, DeckLink manages this)
    std::mem::forget(mock_buffer);
    
    let meta = FrameMeta {
        source_id: 0,
        width,
        height,
        pixfmt: PixelFormat::YUV422_8,
        colorspace: ColorSpace::BT709,
        frame_idx: 42,
        pts_ns: 1234567890123456789,
        t_capture_ns: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64,
        stride_bytes,
    };
    
    let data = MemRef {
        ptr: buffer_ptr,
        len: total_size,
        stride: stride_bytes as usize,
        loc: MemLoc::Cpu,
    };
    
    RawFramePacket { meta, data }
}

fn print_packet_details(packet: &RawFramePacket) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    RawFramePacket                             ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║");
    println!("║ 📦 FrameMeta:");
    println!("║    ├─ source_id      : {}", packet.meta.source_id);
    println!("║    ├─ frame_idx      : {}", packet.meta.frame_idx);
    println!("║    ├─ dimensions     : {}x{} pixels", packet.meta.width, packet.meta.height);
    println!("║    ├─ pixel_format   : {:?}", packet.meta.pixfmt);
    println!("║    ├─ colorspace     : {:?}", packet.meta.colorspace);
    println!("║    ├─ stride_bytes   : {} bytes/row", packet.meta.stride_bytes);
    println!("║    ├─ pts_ns         : {}", packet.meta.pts_ns);
    println!("║    └─ t_capture_ns   : {}", packet.meta.t_capture_ns);
    println!("║");
    println!("║ 💾 MemRef (data):");
    println!("║    ├─ location       : {:?}", packet.data.loc);
    println!("║    ├─ ptr            : {:p}", packet.data.ptr);
    println!("║    ├─ len            : {} bytes ({:.2} MB)", 
             packet.data.len, 
             packet.data.len as f64 / 1_048_576.0);
    println!("║    └─ stride         : {} bytes/row", packet.data.stride);
    println!("║");
    
    // Calculate some statistics
    let bytes_per_pixel = 2; // YUV422
    let pixels = packet.meta.width as usize * packet.meta.height as usize;
    let expected_min = pixels * bytes_per_pixel;
    
    println!("║ 📊 Statistics:");
    println!("║    ├─ total pixels   : {}", pixels);
    println!("║    ├─ bytes/pixel    : {}", bytes_per_pixel);
    println!("║    ├─ expected min   : {} bytes", expected_min);
    println!("║    └─ actual size    : {} bytes", packet.data.len);
    
    println!("║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}

fn verify_packet_structure(packet: &RawFramePacket) {
    println!("\n┌─────────────────────────────────────────────────────────┐");
    println!("│ Checking packet structure according to INSTRUCTION.md  │");
    println!("└─────────────────────────────────────────────────────────┘");
    
    let mut checks_passed = 0;
    let mut checks_total = 0;
    
    // Check 1: PixelFormat
    checks_total += 1;
    if matches!(packet.meta.pixfmt, PixelFormat::YUV422_8) {
        println!("✓ [1] PixelFormat is YUV422_8 (8-bit YUV UYVY format)");
        checks_passed += 1;
    } else {
        println!("✗ [1] PixelFormat is {:?}, expected YUV422_8", packet.meta.pixfmt);
    }
    
    // Check 2: ColorSpace
    checks_total += 1;
    if matches!(packet.meta.colorspace, ColorSpace::BT709) {
        println!("✓ [2] ColorSpace is BT709 (HD standard)");
        checks_passed += 1;
    } else {
        println!("✗ [2] ColorSpace is {:?}, expected BT709", packet.meta.colorspace);
    }
    
    // Check 3: Dimensions
    checks_total += 1;
    if packet.meta.width > 0 && packet.meta.height > 0 {
        println!("✓ [3] Dimensions are valid: {}x{}", packet.meta.width, packet.meta.height);
        checks_passed += 1;
    } else {
        println!("✗ [3] Invalid dimensions: {}x{}", packet.meta.width, packet.meta.height);
    }
    
    // Check 4: Stride
    checks_total += 1;
    let min_stride = packet.meta.width * 2; // YUV422 = 2 bytes/pixel
    if packet.meta.stride_bytes >= min_stride {
        println!("✓ [4] Stride is valid: {} bytes (>= {} minimum)", 
                 packet.meta.stride_bytes, min_stride);
        checks_passed += 1;
    } else {
        println!("✗ [4] Stride too small: {} < {}", packet.meta.stride_bytes, min_stride);
    }
    
    // Check 5: Data size
    checks_total += 1;
    let expected_size = packet.meta.stride_bytes as usize * packet.meta.height as usize;
    if packet.data.len == expected_size {
        println!("✓ [5] Data size matches: {} bytes (stride × height)", packet.data.len);
        checks_passed += 1;
    } else {
        println!("✗ [5] Data size mismatch: {} != {}", packet.data.len, expected_size);
    }
    
    // Check 6: Memory location
    checks_total += 1;
    if matches!(packet.data.loc, MemLoc::Cpu) {
        println!("✓ [6] Data is in CPU memory (as captured from DeckLink)");
        checks_passed += 1;
    } else {
        println!("✗ [6] Data location is {:?}, expected CPU", packet.data.loc);
    }
    
    // Check 7: Pointer validity
    checks_total += 1;
    if !packet.data.ptr.is_null() {
        println!("✓ [7] Data pointer is valid (not null)");
        checks_passed += 1;
    } else {
        println!("✗ [7] Data pointer is null");
    }
    
    // Check 8: Frame metadata
    checks_total += 1;
    if packet.meta.pts_ns > 0 && packet.meta.t_capture_ns > 0 {
        println!("✓ [8] Timestamps are populated (pts_ns and t_capture_ns)");
        checks_passed += 1;
    } else {
        println!("✗ [8] Missing timestamps");
    }
    
    println!("\n┌─────────────────────────────────────────────────────────┐");
    if checks_passed == checks_total {
        println!("│ ✓✓✓ All {} checks PASSED! Packet is pipeline-ready!  │", checks_total);
    } else {
        println!("│ {}/{} checks passed                                     │", 
                 checks_passed, checks_total);
    }
    println!("└─────────────────────────────────────────────────────────┘");
}
