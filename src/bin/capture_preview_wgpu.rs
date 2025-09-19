// Ultra low-latency preview using winit + wgpu (Metal backend on macOS)
// Uploads BGRA frames from the C++ shim into a GPU texture and renders a full-screen quad.

use std::mem::MaybeUninit;
use std::time::{Duration, Instant};

use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::WindowBuilder;

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

fn main() {
    // Ensure crate lib gets linked (brings in C++ shim static lib)
    let _ = decklink_rust::decklink_devicelist_count();
    // Initialize capture
    let device_index = 0i32;
    unsafe {
        if !decklink_capture_open(device_index) {
            eprintln!("Failed to open DeckLink capture on device {}", device_index);
            return;
        }
    }

    // Create window
    let event_loop = EventLoop::new().expect("event loop");
    let window = WindowBuilder::new()
        .with_title("DeckLink Preview (wgpu)")
        .with_inner_size(winit::dpi::LogicalSize::new(960.0, 540.0))
        .build(&event_loop)
        .expect("create window");

    // Init wgpu (Metal on macOS)
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..Default::default()
    });
    let surface = unsafe { instance.create_surface(&window) }.expect("create surface");
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .expect("request adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .expect("request device");

    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| {
            matches!(
                f,
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
            )
        })
        .unwrap_or(caps.formats[0]);
    let mut size = window.inner_size();
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: if caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
            wgpu::PresentMode::Immediate
        } else {
            wgpu::PresentMode::Fifo
        },
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 1,
    };
    surface.configure(&device, &config);

    // Simple textured quad pipeline
    let shader_src = r#"
        struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
        @vertex fn vs_main(@builtin(vertex_index) vi : u32) -> VsOut {
            var p = array<vec2<f32>, 3>(
                vec2<f32>(-1.0, -3.0),
                vec2<f32>( 3.0,  1.0),
                vec2<f32>(-1.0,  1.0)
            );
            var uv = array<vec2<f32>, 3>(
                vec2<f32>(0.0, 2.0),
                vec2<f32>(2.0, 0.0),
                vec2<f32>(0.0, 0.0)
            );
            var o: VsOut;
            o.pos = vec4<f32>(p[vi], 0.0, 1.0);
            o.uv = uv[vi];
            return o;
        }
        @group(0) @binding(0) var samp: sampler;
        @group(0) @binding(1) var tex: texture_2d<f32>;
        @fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
            return textureSample(tex, samp, in.uv);
        }
    "#;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shader"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("linear-sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    // Frame texture (BGRA8) — created on demand when first frame arrives or size changes
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("pipe"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    // State for frame upload
    let mut video_tex: Option<wgpu::Texture> = None;
    let mut video_view: Option<wgpu::TextureView> = None;
    let mut bind_group: Option<wgpu::BindGroup> = None;
    let mut v_width: u32 = 0;
    let mut v_height: u32 = 0;
    let mut last_seq: u64 = 0;

    let bgl_sampler = &sampler;

    // FPS counter
    let mut last_fps = Instant::now();
    let mut frames = 0u32;

    event_loop
        .run(|event, elwt| {
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.state == ElementState::Pressed {
                            if let Key::Named(NamedKey::Escape) = event.logical_key {
                                elwt.exit();
                            }
                        }
                    }
                    WindowEvent::Resized(new_size) => {
                        size = new_size;
                        if size.width > 0 && size.height > 0 {
                            config.width = size.width;
                            config.height = size.height;
                            surface.configure(&device, &config);
                        }
                    }
                    _ => {}
                },
                Event::AboutToWait => {
                    // Poll capture frame
                    let mut cf = MaybeUninit::<CaptureFrame>::uninit();
                    let got = unsafe { decklink_capture_get_frame(cf.as_mut_ptr()) };
                    if got {
                        let cf = unsafe { cf.assume_init() };
                        if cf.seq != 0
                            && cf.seq != last_seq
                            && !cf.data.is_null()
                            && cf.width > 0
                            && cf.height > 0
                        {
                            last_seq = cf.seq;
                            let w = cf.width as u32;
                            let h = cf.height as u32;
                            let rb = cf.row_bytes as usize;
                            // Create video texture on demand (BGRA8)
                            if v_width != w || v_height != h || video_tex.is_none() {
                                v_width = w;
                                v_height = h;
                                let tex = device.create_texture(&wgpu::TextureDescriptor {
                                    label: Some("video"),
                                    size: wgpu::Extent3d {
                                        width: w,
                                        height: h,
                                        depth_or_array_layers: 1,
                                    },
                                    mip_level_count: 1,
                                    sample_count: 1,
                                    dimension: wgpu::TextureDimension::D2,
                                    format: wgpu::TextureFormat::Bgra8Unorm,
                                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                                        | wgpu::TextureUsages::COPY_DST,
                                    view_formats: &[],
                                });
                                let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some("bg"),
                                    layout: &bind_group_layout,
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: wgpu::BindingResource::Sampler(bgl_sampler),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: wgpu::BindingResource::TextureView(&view),
                                        },
                                    ],
                                });
                                video_view = Some(view);
                                video_tex = Some(tex);
                                bind_group = Some(bg);
                            }
                            // Upload with 256-byte row alignment padding if needed
                            if let Some(tex) = &video_tex {
                                let bytes_per_row = rb;
                                let padded_bpr = ((bytes_per_row + 255) / 256) * 256;
                                let size_bytes = padded_bpr * (h as usize);
                                let mut staging = vec![0u8; size_bytes];
                                // SAFETY: read from capture buffer for this call only
                                let src = unsafe {
                                    std::slice::from_raw_parts(cf.data, (rb) * (h as usize))
                                };
                                for row in 0..(h as usize) {
                                    let s = &src[row * rb..(row + 1) * rb];
                                    let d = &mut staging[row * padded_bpr..row * padded_bpr + rb];
                                    d.copy_from_slice(s);
                                }
                                queue.write_texture(
                                    wgpu::ImageCopyTexture {
                                        texture: tex,
                                        mip_level: 0,
                                        origin: wgpu::Origin3d::ZERO,
                                        aspect: wgpu::TextureAspect::All,
                                    },
                                    &staging,
                                    wgpu::ImageDataLayout {
                                        offset: 0,
                                        bytes_per_row: Some(padded_bpr as u32),
                                        rows_per_image: Some(h),
                                    },
                                    wgpu::Extent3d {
                                        width: w,
                                        height: h,
                                        depth_or_array_layers: 1,
                                    },
                                );
                            }
                        }
                    }

                    // Render
                    let frame = match surface.get_current_texture() {
                        Ok(f) => f,
                        Err(_) => {
                            surface.configure(&device, &config);
                            return;
                        }
                    };
                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("enc"),
                        });

                    {
                        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("rpass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                        r: 0.125,
                                        g: 0.125,
                                        b: 0.125,
                                        a: 1.0,
                                    }),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            occlusion_query_set: None,
                            timestamp_writes: None,
                        });
                        rpass.set_pipeline(&pipeline);
                        if let Some(bg) = &bind_group {
                            rpass.set_bind_group(0, bg, &[]);
                        }
                        rpass.draw(0..3, 0..1);
                    }

                    queue.submit(Some(encoder.finish()));
                    frame.present();

                    frames += 1;
                    if last_fps.elapsed() >= Duration::from_secs(1) {
                        println!("fps: {}", frames);
                        frames = 0;
                        last_fps = Instant::now();
                    }
                }
                _ => {}
            }
        })
        .expect("run");
}
