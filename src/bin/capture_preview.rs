// Simple live preview window for DeckLink capture using the existing C++ shim
// This binary polls the shim for the latest BGRA frame and displays it via minifb

use std::mem::MaybeUninit;
use std::time::{Duration, Instant};

use minifb::{Key, Window, WindowOptions};

#[repr(C)]
struct CaptureFrame {
    data: *const u8,
    width: i32,
    height: i32,
    row_bytes: i32,
    seq: u64,
}

extern "C" {
    fn decklink_capture_open(device_index: i32) -> bool;
    fn decklink_capture_get_frame(out: *mut CaptureFrame) -> bool;
    fn decklink_capture_close();
}

fn bgra_to_argb_u32(src: &[u8], width: usize, height: usize, row_bytes: usize, dst: &mut [u32]) {
    // Convert BGRA bytes (little-endian) into u32 ARGB (0xFFRRGGBB)
    for y in 0..height {
        let src_row = &src[y * row_bytes..][..width * 4];
        let dst_row = &mut dst[y * width..][..width];
        let mut sx = 0;
        for dx in 0..width {
            let b = src_row[sx] as u32;
            let g = src_row[sx + 1] as u32;
            let r = src_row[sx + 2] as u32;
            // src_row[sx + 3] is alpha, ignore and set to 0xFF
            dst_row[dx] = (0xFF << 24) | (r << 16) | (g << 8) | b;
            sx += 4;
        }
    }
}

fn main() {
    // Ensure the library target (and its C++ shim linking) is brought into this binary
    let _ = decklink_rust::decklink_devicelist_count();
    // Choose first device (index 0) for now
    let device_index = 0i32;

    unsafe {
        if !decklink_capture_open(device_index) {
            eprintln!("Failed to open DeckLink capture on device {}", device_index);
            return;
        }
    }

    let mut last_seq: u64 = 0;
    // Create a window immediately so the user sees something even before frames arrive
    let mut width: usize = 640;
    let mut height: usize = 360;
    let mut buffer: Vec<u32> = vec![0xFF202020; width * height];
    let mut window: Window = Window::new(
        &format!("DeckLink Preview - {}x{} (waiting for signal)", width, height),
        width,
        height,
        WindowOptions::default(),
    )
    .expect("Failed to create window");
    // Uncap update rate for lowest latency preview
    window.limit_update_rate(None);

    let mut last_fps_time = Instant::now();
    let mut frames = 0u32;

    'outer: loop {
        let mut cf = MaybeUninit::<CaptureFrame>::uninit();
        let ok = unsafe { decklink_capture_get_frame(cf.as_mut_ptr()) };
        if ok {
            let cf = unsafe { cf.assume_init() };
            if cf.seq != 0 && cf.seq != last_seq && !cf.data.is_null() && cf.width > 0 && cf.height > 0 {
                last_seq = cf.seq;
                let w = cf.width as usize;
                let h = cf.height as usize;
                let rb = cf.row_bytes as usize;

                // Resize window and buffer if needed (recreate window; minifb 0.24 has no set_size)
                if width != w || height != h {
                    width = w;
                    height = h;
                    buffer = vec![0u32; width * height];
                    window = Window::new(
                        &format!("DeckLink Preview - {}x{}", width, height),
                        width,
                        height,
                        WindowOptions::default(),
                    )
                    .expect("Failed to recreate window");
                    window.limit_update_rate(Some(Duration::from_millis(16)));
                }

                let size = (rb) * height;
                // SAFETY: we copy only, cf.data must remain valid during this call
                let src = unsafe { std::slice::from_raw_parts(cf.data, size) };
                bgra_to_argb_u32(src, width, height, rb, &mut buffer);
                frames += 1;
            }
        }

        // Show frame (submit even if same buffer)
        if !window.is_open() || window.is_key_down(Key::Escape) {
            break 'outer;
        }
        if let Err(e) = window.update_with_buffer(&buffer, width, height) {
            eprintln!("Window update error: {}", e);
            break 'outer;
        }

        // FPS logging
        if last_fps_time.elapsed() >= Duration::from_secs(1) {
            println!("fps: {}", frames);
            frames = 0;
            last_fps_time = Instant::now();
        }
    }

    unsafe {
        decklink_capture_close();
    }
}
