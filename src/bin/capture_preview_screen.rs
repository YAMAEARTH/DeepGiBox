// Zero-copy/zero-convert live preview by using DeckLink's Cocoa Screen Preview
// Renders directly into the NSView of a winit window via the DeckLink API helper.
// macOS only.

use std::ffi::c_void;
use std::time::{Duration, Instant};

use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::WindowBuilder;

// Bring in raw window handle to extract the NSView pointer (AppKit)
use winit::raw_window_handle::{AppKitWindowHandle, HasWindowHandle, RawWindowHandle};

extern "C" {
    fn decklink_capture_open(device_index: i32) -> bool;
    fn decklink_capture_close();
    fn decklink_preview_attach_nsview(nsview: *mut c_void) -> bool;
    fn decklink_preview_detach();
    fn decklink_preview_seq() -> u64;
}

fn main() {
    // Ensure crate lib gets linked (brings in C++ shim static lib)
    let _ = decklink_rust::decklink_devicelist_count();

    // Create window (AppKit/NSView under the hood)
    let event_loop = EventLoop::new().expect("event loop");
    let window = WindowBuilder::new()
        .with_title("DeckLink Native Preview (Cocoa)")
        .with_inner_size(winit::dpi::LogicalSize::new(960.0, 540.0))
        .build(&event_loop)
        .expect("create window");

    // Extract NSView pointer
    let ns_view_ptr: *mut c_void = {
        let handle = window
            .window_handle()
            .expect("window handle")
            .as_raw();
        match handle {
            RawWindowHandle::AppKit(AppKitWindowHandle { ns_view, .. }) => ns_view.cast::<c_void>().as_ptr(),
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
        // Choose first device for now
        if !decklink_capture_open(0) {
            eprintln!("Failed to open DeckLink capture");
            decklink_preview_detach();
            return;
        }
    }

    // FPS counter based on sequence increments reported by the shim
    let mut last_seq: u64 = unsafe { decklink_preview_seq() };
    let mut frames: u32 = 0;
    let mut last_fps = Instant::now();

    // Run event loop; DeckLink renders directly into the NSView
    event_loop
        .run(|event, elwt| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    // Cleanup and exit
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
                // Poll sequence and compute FPS once per second
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
                // Tick roughly at 60Hz to keep printing timely
                elwt.set_control_flow(ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(16)));
            }
            _ => {}
        })
        .expect("run");
}
