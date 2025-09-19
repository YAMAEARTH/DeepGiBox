//! โหมดพรีวิวสองแบบ: บน macOS จะใช้ DeckLink Cocoa Screen Preview (zero-copy)
//! ส่วน Linux และแพลตฟอร์มอื่น fallback เป็นการดึงเฟรมจาก shim แล้ววาดผ่าน minifb

#[cfg(target_os = "macos")]
mod platform {
    use std::ffi::c_void;
    use std::time::{Duration, Instant};

    use winit::event::{ElementState, Event, WindowEvent};
    use winit::event_loop::{ControlFlow, EventLoop};
    use winit::keyboard::{Key, NamedKey};
    use winit::raw_window_handle::{AppKitWindowHandle, HasWindowHandle, RawWindowHandle};
    use winit::window::WindowBuilder;

    extern "C" {
        fn decklink_capture_open(device_index: i32) -> bool;
        fn decklink_capture_close();
        fn decklink_preview_attach_nsview(nsview: *mut c_void) -> bool;
        fn decklink_preview_detach();
        fn decklink_preview_seq() -> u64;
    }

    pub fn run() {
        // ดึงไลบรารีหลักขึ้นมาเพื่อให้ลิงก์กับ shim
        let _ = decklink_rust::decklink_devicelist_count();

        let event_loop = EventLoop::new().expect("event loop");
        let window = WindowBuilder::new()
            .with_title("DeckLink Native Preview (Cocoa)")
            .with_inner_size(winit::dpi::LogicalSize::new(960.0, 540.0))
            .build(&event_loop)
            .expect("create window");

        let ns_view_ptr: *mut c_void = {
            let handle = window.window_handle().expect("window handle").as_raw();
            match handle {
                RawWindowHandle::AppKit(AppKitWindowHandle { ns_view, .. }) => {
                    ns_view.cast::<c_void>().as_ptr()
                }
                _ => {
                    eprintln!("This preview is supported on macOS (AppKit) only");
                    return;
                }
            }
        };

        unsafe {
            if !decklink_preview_attach_nsview(ns_view_ptr) {
                eprintln!("Failed to attach NSView for DeckLink screen preview");
                return;
            }
            if !decklink_capture_open(0) {
                eprintln!("Failed to open DeckLink capture");
                decklink_preview_detach();
                return;
            }
        }

        let mut last_seq: u64 = unsafe { decklink_preview_seq() };
        let mut frames: u32 = 0;
        let mut last_fps = Instant::now();

        event_loop
            .run(|event, elwt| match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => {
                        unsafe {
                            decklink_preview_detach();
                            decklink_capture_close();
                        }
                        elwt.exit();
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.state == ElementState::Pressed {
                            if let Key::Named(NamedKey::Escape) = event.logical_key {
                                unsafe {
                                    decklink_preview_detach();
                                    decklink_capture_close();
                                }
                                elwt.exit();
                            }
                        }
                    }
                    _ => {}
                },
                Event::AboutToWait => {
                    let seq = unsafe { decklink_preview_seq() };
                    if seq > last_seq {
                        frames = frames.saturating_add((seq - last_seq) as u32);
                        last_seq = seq;
                    }
                    if last_fps.elapsed() >= Duration::from_secs(1) {
                        println!("fps: {}", frames);
                        frames = 0;
                        last_fps = Instant::now();
                    }
                    elwt.set_control_flow(ControlFlow::WaitUntil(
                        Instant::now() + Duration::from_millis(16),
                    ));
                }
                _ => {}
            })
            .expect("run");
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
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

    fn bgra_to_argb_u32(
        src: &[u8],
        width: usize,
        height: usize,
        row_bytes: usize,
        dst: &mut [u32],
    ) {
        for y in 0..height {
            let src_row = &src[y * row_bytes..][..width * 4];
            let dst_row = &mut dst[y * width..][..width];
            let mut sx = 0;
            for dx in 0..width {
                let b = src_row[sx] as u32;
                let g = src_row[sx + 1] as u32;
                let r = src_row[sx + 2] as u32;
                dst_row[dx] = (0xFF << 24) | (r << 16) | (g << 8) | b;
                sx += 4;
            }
        }
    }

    pub fn run() {
        let _ = decklink_rust::decklink_devicelist_count();
        let device_index = 0i32;

        unsafe {
            if !decklink_capture_open(device_index) {
                eprintln!("Failed to open DeckLink capture on device {}", device_index);
                return;
            }
        }

        let mut last_seq: u64 = 0;
        let mut width: usize = 640;
        let mut height: usize = 360;
        let mut buffer: Vec<u32> = vec![0xFF202020; width * height];
        let mut window = Window::new(
            &format!(
                "DeckLink Preview - {}x{} (waiting for signal)",
                width, height
            ),
            width,
            height,
            WindowOptions::default(),
        )
        .expect("Failed to create window");
        window.limit_update_rate(None);

        let mut last_fps_time = Instant::now();
        let mut frames = 0u32;

        'outer: loop {
            let mut cf = MaybeUninit::<CaptureFrame>::uninit();
            let ok = unsafe { decklink_capture_get_frame(cf.as_mut_ptr()) };
            if ok {
                let cf = unsafe { cf.assume_init() };
                if cf.seq != 0
                    && cf.seq != last_seq
                    && !cf.data.is_null()
                    && cf.width > 0
                    && cf.height > 0
                {
                    last_seq = cf.seq;
                    let w = cf.width as usize;
                    let h = cf.height as usize;
                    let rb = cf.row_bytes as usize;

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

                    let size = rb * height;
                    let src = unsafe { std::slice::from_raw_parts(cf.data, size) };
                    bgra_to_argb_u32(src, width, height, rb, &mut buffer);
                    frames += 1;
                }
            }

            if !window.is_open() || window.is_key_down(Key::Escape) {
                break 'outer;
            }
            if let Err(e) = window.update_with_buffer(&buffer, width, height) {
                eprintln!("Window update error: {}", e);
                break 'outer;
            }

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
}

fn main() {
    platform::run();
}
