use std::ffi::{c_void, CString};
use std::time::{Duration, Instant};

use anyhow::Result;
use decklink_rust::capture::CaptureSession;
use decklink_rust::preview_gl::PreviewGl;
use glow::HasContext;
use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextAttributesBuilder, GlProfile};
use glutin::display::GetGlDisplay;
use glutin::prelude::*;
use glutin::surface::{SurfaceAttributesBuilder, SwapInterval, WindowSurface};
use glutin_winit::{DisplayBuilder, GlWindow};
use raw_window_handle::HasRawWindowHandle;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::Window;

struct GlState {
    context: glutin::context::PossiblyCurrentContext,
    surface: glutin::surface::Surface<WindowSurface>,
    window: Window,
    glow: glow::Context,
}

impl GlState {
    fn new(event_loop: &EventLoop<()>) -> Result<Self> {
        let window_builder = winit::window::WindowBuilder::new()
            .with_title("DeckLink OpenGL Preview")
            .with_inner_size(LogicalSize::new(1280.0, 720.0));

        let template = ConfigTemplateBuilder::new()
            .with_alpha_size(8)
            .with_depth_size(24)
            .with_stencil_size(8)
            .with_transparency(false);

        let display_builder = DisplayBuilder::new().with_window_builder(Some(window_builder));
        let (window_opt, gl_config) = display_builder
            .build(event_loop, template, |configs| {
                configs
                    .reduce(|accum, config| {
                        if config.num_samples() > accum.num_samples() {
                            config
                        } else {
                            accum
                        }
                    })
                    .unwrap()
            })
            .expect("failed to initialize OpenGL window");

        let window = window_opt.unwrap();
        let raw_window_handle = window.raw_window_handle();
        let context_attributes = ContextAttributesBuilder::new()
            .with_profile(GlProfile::Compatibility)
            .build(Some(raw_window_handle));
        let fallback_attributes = ContextAttributesBuilder::new().build(Some(raw_window_handle));

        let not_current = unsafe {
            gl_config
                .display()
                .create_context(&gl_config, &context_attributes)
                .or_else(|_| {
                    gl_config
                        .display()
                        .create_context(&gl_config, &fallback_attributes)
                })?
        };

        let attrs =
            window.build_surface_attributes(SurfaceAttributesBuilder::<WindowSurface>::new());
        let surface = unsafe {
            gl_config
                .display()
                .create_window_surface(&gl_config, &attrs)?
        };
        let context = not_current.make_current(&surface)?;
        surface
            .set_swap_interval(&context, SwapInterval::DontWait)
            .ok();

        let glow = unsafe {
            glow::Context::from_loader_function(|name| {
                let c_name = CString::new(name).expect("invalid symbol name");
                gl_config
                    .display()
                    .get_proc_address(c_name.as_c_str())
                    .cast::<c_void>()
            })
        };

        window.resize_surface(&surface, &context);

        Ok(Self {
            context,
            surface,
            window,
            glow,
        })
    }

    fn resize(&self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.window.resize_surface(&self.surface, &self.context);
        unsafe {
            self.glow.viewport(0, 0, width as i32, height as i32);
        }
    }

    fn swap_buffers(&self) {
        let _ = self.surface.swap_buffers(&self.context);
    }
}

fn main() -> Result<()> {
    let _ = decklink_rust::decklink_devicelist_count();

    let event_loop = EventLoop::new()?;
    let gl_state = GlState::new(&event_loop)?;
    let initial_size = gl_state.window.inner_size();
    gl_state.resize(initial_size.width, initial_size.height);

    let mut preview = PreviewGl::create()?;
    preview.initialize_gl()?;
    let _capture = CaptureSession::open(0)?;
    preview.enable()?;

    let mut last_seq = 0u64;
    let mut frames = 0u32;
    let mut last_fps_instant = Instant::now();

    event_loop
        .run(move |event, elwt| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    preview.disable();
                    elwt.exit();
                }
                WindowEvent::Resized(size) => {
                    gl_state.resize(size.width, size.height);
                }
                WindowEvent::RedrawRequested => {
                    let rendered = preview.render();
                    if rendered {
                        gl_state.swap_buffers();
                    }

                    let seq = preview.seq();
                    if seq != last_seq {
                        frames = frames.saturating_add(seq.saturating_sub(last_seq) as u32);
                        last_seq = seq;
                    }
                    if last_fps_instant.elapsed() >= Duration::from_secs(1) {
                        let latency_ns = preview.last_latency_ns();
                        let latency_ms = latency_ns as f64 / 1_000_000.0;
                        println!("fps: {frames}, latency: {latency_ms:.2} ms");
                        frames = 0;
                        last_fps_instant = Instant::now();
                    }
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    if event.state == ElementState::Pressed {
                        if let Key::Named(NamedKey::Escape) = event.logical_key {
                            preview.disable();
                            elwt.exit();
                        }
                    }
                }
                _ => {}
            },
            Event::AboutToWait => {
                gl_state.window.request_redraw();
            }
            Event::LoopExiting => {
                preview.disable();
            }
            _ => {}
        })
        .map_err(|e| anyhow::anyhow!(e))?;

    Ok(())
}
