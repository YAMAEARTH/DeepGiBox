use std::ffi::{c_void, CString};
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
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

use decklink_rust::{PipelineBuilder, CaptureConfig, PreviewConfig, FrameInfoStage, ColorSpace};

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
    println!("Starting DeepGI Pipeline with separated capture and preview...");
    
    // List available devices
    let devices = decklink_rust::devicelist();
    if devices.is_empty() {
        bail!("No DeckLink devices found");
    }
    
    println!("Available DeckLink devices:");
    for (i, name) in devices.iter().enumerate() {
        println!("  {}: {}", i, name);
    }

    let event_loop = EventLoop::new()?;
    let gl_state = GlState::new(&event_loop)?;
    let initial_size = gl_state.window.inner_size();
    gl_state.resize(initial_size.width, initial_size.height);

    // Create pipeline configuration
    let capture_config = CaptureConfig {
        device_index: 0,
        source_id: 1,
        expected_colorspace: ColorSpace::BT709,
    };

    let preview_config = PreviewConfig {
        enable_stats: true,
        stats_interval: Duration::from_secs(1),
    };

    // Build pipeline with processing stage
    let mut pipeline = PipelineBuilder::new()
        .with_capture_config(capture_config)
        .with_preview_config(preview_config)
        .with_processing_stage(Box::new(FrameInfoStage::new()))
        .build();

    // Initialize pipeline (must be done in OpenGL context)
    if let Err(e) = pipeline.initialize() {
        bail!("Failed to initialize pipeline: {}", e);
    }

    // Start the pipeline
    if let Err(e) = pipeline.start() {
        bail!("Failed to start pipeline: {}", e);
    }

    println!("Pipeline started successfully!");

    let mut last_stats_time = Instant::now();

    event_loop
        .run(move |event, elwt| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    println!("Shutting down pipeline...");
                    let _ = pipeline.stop();
                    elwt.exit();
                }
                WindowEvent::Resized(size) => {
                    gl_state.resize(size.width, size.height);
                }
                WindowEvent::RedrawRequested => {
                    let rendered = pipeline.render();
                    if rendered {
                        gl_state.swap_buffers();
                    }

                    // Print stats periodically
                    if last_stats_time.elapsed() >= Duration::from_secs(1) {
                        let pipeline_stats = pipeline.get_performance_stats();
                        let preview_stats = pipeline.get_preview_stats();
                        
                        println!(
                            "Pipeline - Processed: {}, Dropped: {}, Preview FPS: {:.1}, Latency: {:.2}ms",
                            pipeline_stats.frames_processed,
                            pipeline_stats.frames_dropped,
                            preview_stats.fps,
                            preview_stats.latency_ms
                        );
                        
                        last_stats_time = Instant::now();
                    }
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    if event.state == ElementState::Pressed {
                        if let Key::Named(NamedKey::Escape) = event.logical_key {
                            println!("Shutting down pipeline...");
                            let _ = pipeline.stop();
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
                let _ = pipeline.stop();
            },
            _ => {}
        })
        .map_err(|e| anyhow::anyhow!(e))?;

    Ok(())
}
