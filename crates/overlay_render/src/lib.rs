use anyhow::Result;
use common_io::{MemLoc, MemRef, OverlayFramePacket, OverlayPlanPacket, Stage, DrawOp};
use rusttype::{Font, Scale, point};
use std::cmp::{max, min};
use std::os::raw::{c_int, c_uint, c_void};
use std::ptr;
use std::slice;

// ==================== CUDA FFI ====================
#[allow(dead_code)]
extern "C" {
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn cudaMemcpy2DAsync(
        dst: *mut c_void,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> c_int;
    fn cudaFreeHost(ptr: *mut c_void) -> c_int;
    fn cudaStreamCreate(stream: *mut *mut c_void) -> c_int;
    fn cudaStreamDestroy(stream: *mut c_void) -> c_int;
    fn cudaStreamSynchronize(stream: *mut c_void) -> c_int;
    fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: c_uint) -> c_int;
    fn cudaEventDestroy(event: *mut c_void) -> c_int;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> c_int;
    fn cudaEventSynchronize(event: *mut c_void) -> c_int;
    fn cudaStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: c_uint) -> c_int;
    
    // DVP-compatible allocation (from shim.cpp)
    fn decklink_allocate_dvp_compatible_buffer(size: usize) -> *mut c_void;
    fn decklink_free_dvp_compatible_buffer(ptr: *mut c_void);
    
    // CUDA kernels จาก overlay_render.cu
    fn launch_clear_buffer(
        gpu_buf: *mut u8,
        stride: c_int,
        width: c_int,
        height: c_int,
        stream: *mut c_void,
    );
    
    fn launch_draw_line(
        gpu_buf: *mut u8,
        stride: c_int,
        width: c_int,
        height: c_int,
        x0: c_int,
        y0: c_int,
        x1: c_int,
        y1: c_int,
        thickness: c_int,
        a: u8,
        r: u8,
        g: u8,
        b: u8,
        stream: *mut c_void,
    );
    
    fn launch_draw_rect(
        gpu_buf: *mut u8,
        stride: c_int,
        width: c_int,
        height: c_int,
        x: c_int,
        y: c_int,
        w: c_int,
        h: c_int,
        thickness: c_int,
        a: u8,
        r: u8,
        g: u8,
        b: u8,
        stream: *mut c_void,
    );
    
    fn launch_execute_commands(
        gpu_buf: *mut u8,
        stride: c_int,
        width: c_int,
        height: c_int,
        gpu_commands: *const c_void,
        num_commands: c_int,
        stream: *mut c_void,
    );
    
    fn launch_fill_rect(
        gpu_buf: *mut u8,
        stride: c_int,
        width: c_int,
        height: c_int,
        x: c_int,
        y: c_int,
        w: c_int,
        h: c_int,
        a: u8,
        r: u8,
        g: u8,
        b: u8,
        stream: *mut c_void,
    );
}

const CUDA_SUCCESS: c_int = 0;
const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
const CUDA_EVENT_DISABLE_TIMING: c_uint = 0x02;

struct PinnedHostBuffer {
    ptr: *mut u8,
    len: usize,
}

impl PinnedHostBuffer {
    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cudaFreeHost(self.ptr as *mut c_void);
            }
            self.ptr = ptr::null_mut();
            self.len = 0;
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Region {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl Region {
    fn union_with(&mut self, other: &Region) {
        let min_x = min(self.x, other.x);
        let min_y = min(self.y, other.y);
        let max_x = max(self.x + self.width, other.x + other.width);
        let max_y = max(self.y + self.height, other.y + other.height);
        self.x = min_x;
        self.y = min_y;
        self.width = max_x - min_x;
        self.height = max_y - min_y;
    }
}

struct LabelCommand {
    anchor: (i32, i32),
    text: String,
    font_px: f32,
    color: (u8, u8, u8, u8),
    bounds: Region,
}

fn measure_label_bounds(
    font: &Font<'static>,
    text: &str,
    font_px: f32,
    anchor: (i32, i32),
    canvas_w: u32,
    canvas_h: u32,
) -> Option<Region> {
    if text.is_empty() || font_px <= 0.0 {
        return None;
    }

    let scale = Scale::uniform(font_px);
    let v_metrics = font.v_metrics(scale);
    let offset = point(0.0, v_metrics.ascent);

    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;

    for glyph in font.layout(text, scale, offset) {
        if let Some(bb) = glyph.pixel_bounding_box() {
            let gx_min = anchor.0 + bb.min.x;
            let gy_min = anchor.1 + bb.min.y;
            let gx_max = anchor.0 + bb.max.x;
            let gy_max = anchor.1 + bb.max.y;

            min_x = min(min_x, gx_min);
            min_y = min(min_y, gy_min);
            max_x = max(max_x, gx_max);
            max_y = max(max_y, gy_max);
        }
    }

    if min_x == i32::MAX || min_y == i32::MAX {
        return None;
    }

    let clamp_min_x = max(min_x, 0);
    let clamp_min_y = max(min_y, 0);
    let clamp_max_x = min(max_x, canvas_w as i32);
    let clamp_max_y = min(max_y, canvas_h as i32);

    if clamp_min_x >= clamp_max_x || clamp_min_y >= clamp_max_y {
        return None;
    }

    Some(Region {
        x: clamp_min_x as u32,
        y: clamp_min_y as u32,
        width: (clamp_max_x - clamp_min_x) as u32,
        height: (clamp_max_y - clamp_min_y) as u32,
    })
}

fn union_region(regions: &[Region]) -> Option<Region> {
    let mut iter = regions.iter();
    let first = iter.next()?.clone();
    let mut acc = first;
    for region in iter {
        acc.union_with(region);
    }
    Some(acc)
}

// ==================== GPU Render Stage ====================
pub struct RenderStage {
    gpu_buf: Option<*mut u8>,
    cpu_buf: Option<PinnedHostBuffer>,  // CPU buffer สำหรับ text rendering (pinned)
    stream: *mut c_void,
    text_stream: *mut c_void,
    render_done_event: *mut c_void,
    text_done_event: *mut c_void,
    width: u32,
    height: u32,
    stride: usize,
    device_id: u32,
    debug_mode: bool,
    font: Font<'static>,  // Font สำหรับ text rendering
}

// Helper function for text rendering on CPU buffer
fn draw_text_on_buffer(
    buf: &mut [u8],
    stride: usize,
    w: u32,
    h: u32,
    x: i32,
    y: i32,
    text: &str,
    font_size: f32,
    color: (u8, u8, u8, u8), // R, G, B, A
    font: &Font,
) {
    let scale = Scale::uniform(font_size);
    let v_metrics = font.v_metrics(scale);
    let offset = point(0.0, v_metrics.ascent);
    
    let glyphs: Vec<_> = font.layout(text, scale, offset).collect();
    
    for glyph in glyphs {
        if let Some(bounding_box) = glyph.pixel_bounding_box() {
            glyph.draw(|gx, gy, gv| {
                let px = x + bounding_box.min.x + gx as i32;
                let py = y + bounding_box.min.y + gy as i32;
                
                if px >= 0 && py >= 0 && (px as u32) < w && (py as u32) < h {
                    let idx = (py as usize) * stride + (px as usize) * 4;
                    if idx + 3 < buf.len() {
                        // Alpha blend with existing pixel
                        // Buffer format is BGRA, color input is RGBA
                        let alpha = (gv * color.3 as f32) as u32;
                        let inv_alpha = 255 - alpha;
                        
                        buf[idx + 0] = ((color.2 as u32 * alpha + buf[idx + 0] as u32 * inv_alpha) / 255) as u8; // B
                        buf[idx + 1] = ((color.1 as u32 * alpha + buf[idx + 1] as u32 * inv_alpha) / 255) as u8; // G
                        buf[idx + 2] = ((color.0 as u32 * alpha + buf[idx + 2] as u32 * inv_alpha) / 255) as u8; // R
                        buf[idx + 3] = buf[idx + 3].max((alpha) as u8); // A
                    }
                }
            });
        }
    }
}

impl RenderStage {
    fn ensure_buffer(&mut self, width: u32, height: u32) -> Result<()> {
        // ถ้ามี buffer แล้วและขนาดตรง ไม่ต้องทำอะไร
        if self.gpu_buf.is_some() && self.width == width && self.height == height {
            return Ok(());
        }
        
        // ถ้ามี buffer เก่า free ก่อน
        if let Some(ptr) = self.gpu_buf {
            unsafe { decklink_free_dvp_compatible_buffer(ptr as *mut c_void); }
        }
        
        // Allocate DVP-compatible GPU buffer ใหม่
        let stride = (width * 4) as usize;
        let size = stride * height as usize;
        
        let dev_ptr = unsafe { decklink_allocate_dvp_compatible_buffer(size) };
        if dev_ptr.is_null() {
            anyhow::bail!("Failed to allocate DVP-compatible GPU buffer for overlay rendering");
        }
        
        eprintln!("[overlay_render] ✅ Allocated DVP-compatible buffer: {}x{} ({} bytes)", 
                  width, height, size);
        
        self.gpu_buf = Some(dev_ptr as *mut u8);
        self.width = width;
        self.height = height;
        self.stride = stride;
        
        Ok(())
    }

    fn ensure_cpu_buffer(&mut self, size: usize) -> Result<&mut PinnedHostBuffer> {
        if size == 0 {
            anyhow::bail!("Pinned buffer size must be > 0");
        }

        let needs_alloc = match &self.cpu_buf {
            Some(buf) => buf.len != size,
            None => true,
        };

        if needs_alloc {
            self.cpu_buf.take();

            let mut host_ptr: *mut c_void = ptr::null_mut();
            let result = unsafe { cudaHostAlloc(&mut host_ptr, size, 0) };
            if result != CUDA_SUCCESS || host_ptr.is_null() {
                anyhow::bail!("Failed to allocate pinned host buffer for overlay text (code {})", result);
            }

            self.cpu_buf = Some(PinnedHostBuffer {
                ptr: host_ptr as *mut u8,
                len: size,
            });
        }

        Ok(self.cpu_buf.as_mut().unwrap())
    }
}

pub fn from_path(cfg: &str) -> Result<RenderStage> {
    // Parse device ID from config (default: 0)
    let device_id = cfg
        .split(',')
        .find(|s| s.starts_with("device="))
        .and_then(|s| s.trim_start_matches("device=").parse::<u32>().ok())
        .unwrap_or(0);
    
    // Parse debug mode from config (default: false)
    let debug_mode = cfg
        .split(',')
        .any(|s| s.trim() == "debug" || s.trim() == "debug=true");
    
    // Load font (do before heavy CUDA init to avoid cleanup noise)
    let font_data = include_bytes!("../../../testsupport/DejaVuSans.ttf");
    let font = Font::try_from_bytes(font_data as &[u8])
        .ok_or_else(|| anyhow::anyhow!("Failed to load DejaVuSans.ttf font"))?;

    // Create CUDA streams
    let mut stream: *mut c_void = ptr::null_mut();
    let mut text_stream: *mut c_void = ptr::null_mut();
    let mut render_done_event: *mut c_void = ptr::null_mut();
    let mut text_done_event: *mut c_void = ptr::null_mut();

    let result = unsafe { cudaStreamCreate(&mut stream) };
    if result != CUDA_SUCCESS {
        anyhow::bail!("Failed to create CUDA stream for overlay rendering");
    }

    let text_result = unsafe { cudaStreamCreate(&mut text_stream) };
    if text_result != CUDA_SUCCESS {
        unsafe { cudaStreamDestroy(stream); }
        anyhow::bail!("Failed to create CUDA text stream for overlay rendering");
    }

    let render_event_result = unsafe { cudaEventCreateWithFlags(&mut render_done_event, CUDA_EVENT_DISABLE_TIMING) };
    if render_event_result != CUDA_SUCCESS {
        unsafe {
            cudaStreamDestroy(text_stream);
            cudaStreamDestroy(stream);
        }
        anyhow::bail!("Failed to create CUDA event for render_done");
    }

    let text_event_result = unsafe { cudaEventCreateWithFlags(&mut text_done_event, CUDA_EVENT_DISABLE_TIMING) };
    if text_event_result != CUDA_SUCCESS {
        unsafe {
            cudaEventDestroy(render_done_event);
            cudaStreamDestroy(text_stream);
            cudaStreamDestroy(stream);
        }
        anyhow::bail!("Failed to create CUDA event for text_done");
    }
    
    Ok(RenderStage {
        gpu_buf: None,
        cpu_buf: None,
        stream,
        text_stream,
        render_done_event,
        text_done_event,
        width: 0,
        height: 0,
        stride: 0,
        device_id,
        debug_mode,
        font,
    })
}

impl Stage<OverlayPlanPacket, OverlayFramePacket> for RenderStage {
    fn process(&mut self, input: OverlayPlanPacket) -> OverlayFramePacket {
        let w = input.canvas.0;
        let h = input.canvas.1;
        
        // Ensure GPU buffer is allocated
        if let Err(e) = self.ensure_buffer(w, h) {
            eprintln!("GPU overlay render error: {}", e);
            // Fallback: return empty overlay
            return OverlayFramePacket {
                from: input.from,
                argb: MemRef {
                    ptr: ptr::null_mut(),
                    len: 0,
                    stride: 0,
                    loc: MemLoc::Gpu { device: self.device_id },
                },
                stride: 0,
            };
        }
        
        let gpu_ptr = self.gpu_buf.unwrap();
        
        // Clear buffer (transparent)
        unsafe {
            launch_clear_buffer(
                gpu_ptr,
                self.stride as c_int,
                w as c_int,
                h as c_int,
                self.stream,
            );
        }
        
        // Execute drawing operations on GPU
        for (i, op) in input.ops.iter().enumerate() {
            match op {
                DrawOp::Rect { xywh, thickness, color } => {
                    if self.debug_mode {
                        eprintln!("[DEBUG] Op #{}: Rect at ({},{}) size {}x{} thickness={} color=RGBA({},{},{},{})",
                                  i, xywh.0, xywh.1, xywh.2, xywh.3, thickness, color.0, color.1, color.2, color.3);
                    }
                    unsafe {
                        launch_draw_rect(
                            gpu_ptr,
                            self.stride as c_int,
                            w as c_int,
                            h as c_int,
                            xywh.0 as c_int,
                            xywh.1 as c_int,
                            xywh.2 as c_int,
                            xywh.3 as c_int,
                            *thickness as c_int,
                            color.3, color.0, color.1, color.2, // A, R, G, B (color is R,G,B,A)
                            self.stream,
                        );
                    }
                }
                DrawOp::FillRect { xywh, color } => {
                    if self.debug_mode {
                        eprintln!("[DEBUG] Op #{}: FillRect at ({},{}) size {}x{} color=RGBA({},{},{},{})",
                                  i, xywh.0, xywh.1, xywh.2, xywh.3, color.0, color.1, color.2, color.3);
                    }
                    unsafe {
                        launch_fill_rect(
                            gpu_ptr,
                            self.stride as c_int,
                            w as c_int,
                            h as c_int,
                            xywh.0 as c_int,
                            xywh.1 as c_int,
                            xywh.2 as c_int,
                            xywh.3 as c_int,
                            color.3, color.0, color.1, color.2, // A, R, G, B (color is R,G,B,A)
                            self.stream,
                        );
                    }
                }
                DrawOp::Poly { pts, thickness, color } => {
                    if pts.len() >= 2 {
                        if self.debug_mode {
                            eprintln!("[DEBUG] Op #{}: Line from ({},{}) to ({},{}) thickness={} color=RGBA({},{},{},{})",
                                      i, pts[0].0, pts[0].1, pts[1].0, pts[1].1, thickness, color.0, color.1, color.2, color.3);
                        }
                        unsafe {
                            launch_draw_line(
                                gpu_ptr,
                                self.stride as c_int,
                                w as c_int,
                                h as c_int,
                                pts[0].0 as c_int,
                                pts[0].1 as c_int,
                                pts[1].0 as c_int,
                                pts[1].1 as c_int,
                                *thickness as c_int,
                                color.3, color.0, color.1, color.2, // A, R, G, B (color is R,G,B,A)
                                self.stream,
                            );
                        }
                    }
                }
                DrawOp::Label { anchor, text, font_px, color } => {
                    if self.debug_mode {
                        eprintln!("[DEBUG] Op #{}: Label at ({},{}) text='{}' size={} color=RGBA({},{},{},{})",
                                  i, anchor.0, anchor.1, text, font_px, color.0, color.1, color.2, color.3);
                    }
                    // Text rendering ทำบน CPU แล้ว composite กับ GPU
                    // (จะทำข้างล่างหลังจาก process ทุก op)
                }
            }
        }
        
        // Check for kernel errors
        extern "C" {
            fn cudaPeekAtLastError() -> c_int;
        }
        let last_err = unsafe { cudaPeekAtLastError() };
        if last_err != CUDA_SUCCESS {
            eprintln!("[ERROR] CUDA kernel error detected: {}", last_err);
        }
        
        // === TEXT RENDERING (CPU-based partial copy) ===
        let mut label_cmds: Vec<LabelCommand> = Vec::new();
        for op in input.ops.iter() {
            if let DrawOp::Label { anchor, text, font_px, color } = op {
                let anchor_i = (anchor.0 as i32, anchor.1 as i32);
                let font_px_f32 = *font_px as f32;
                let bounds = measure_label_bounds(&self.font, text, font_px_f32, anchor_i, w, h);
                if let Some(bounds) = bounds {
                    label_cmds.push(LabelCommand {
                        anchor: anchor_i,
                        text: text.clone(),
                        font_px: font_px_f32,
                        color: *color,
                        bounds,
                    });
                } else if self.debug_mode {
                    eprintln!("[DEBUG] Skipping label '{}' (outside canvas or empty)", text);
                }
            }
        }

        let mut text_region: Option<Region> = None;
        if !label_cmds.is_empty() {
            let label_regions: Vec<Region> = label_cmds.iter().map(|cmd| cmd.bounds).collect();
            text_region = union_region(&label_regions);
        }

        let mut text_updated = false;
        if let Some(region) = &text_region {
            if region.width > 0 && region.height > 0 {
                if self.debug_mode {
                    eprintln!(
                        "[DEBUG] Rendering {} labels in region x:{} y:{} w:{} h:{}",
                        label_cmds.len(),
                        region.x,
                        region.y,
                        region.width,
                        region.height
                    );
                }

                let row_bytes = region.width as usize * 4;
                let buf_size = row_bytes * region.height as usize;
                if buf_size > 0 {
                    if let Err(err) = self.ensure_cpu_buffer(buf_size) {
                        eprintln!("[ERROR] {}", err);
                    } else {
                        let gpu_offset = region.y as usize * self.stride + region.x as usize * 4;
                        let gpu_region_ptr = unsafe { gpu_ptr.add(gpu_offset) } as *const c_void;
                        let host_ptr = self.cpu_buf.as_ref().unwrap().ptr as *mut c_void;

                        // Ensure draw kernels complete before copy
                        let record_event = unsafe { cudaEventRecord(self.render_done_event, self.stream) };
                        if record_event != CUDA_SUCCESS {
                            eprintln!("[ERROR] cudaEventRecord (render_done) failed: {}", record_event);
                        } else {
                            let wait_result = unsafe { cudaStreamWaitEvent(self.text_stream, self.render_done_event, 0) };
                            if wait_result != CUDA_SUCCESS {
                                eprintln!("[ERROR] cudaStreamWaitEvent failed: {}", wait_result);
                            } else {
                                let copy_result = unsafe {
                                    cudaMemcpy2DAsync(
                                        host_ptr,
                                        row_bytes,
                                        gpu_region_ptr,
                                        self.stride,
                                        row_bytes,
                                        region.height as usize,
                                        CUDA_MEMCPY_DEVICE_TO_HOST,
                                        self.text_stream,
                                    )
                                };
                                if copy_result != CUDA_SUCCESS {
                                    eprintln!("[ERROR] cudaMemcpy2DAsync D2H failed: {}", copy_result);
                                } else {
                                    let sync_after_copy = unsafe { cudaStreamSynchronize(self.text_stream) };
                                    if sync_after_copy != CUDA_SUCCESS {
                                        eprintln!("[ERROR] cudaStreamSynchronize text_stream failed: {}", sync_after_copy);
                                    } else {
                                        let stride = row_bytes;
                                        let font_ptr: *const Font<'static> = &self.font;
                                        {
                                            let cpu_slice = self.cpu_buf.as_mut().unwrap().as_mut_slice();
                                            for cmd in &label_cmds {
                                                let local_x = cmd.anchor.0 - region.x as i32;
                                                let local_y = cmd.anchor.1 - region.y as i32;
                                                unsafe {
                                                    draw_text_on_buffer(
                                                        cpu_slice,
                                                        stride,
                                                        region.width,
                                                        region.height,
                                                        local_x,
                                                        local_y,
                                                        &cmd.text,
                                                        cmd.font_px,
                                                        cmd.color,
                                                        &*font_ptr,
                                                    );
                                                }
                                            }
                                        }

                                        let cpu_src = self.cpu_buf.as_ref().unwrap().ptr as *const c_void;
                                        let copy_back = unsafe {
                                            cudaMemcpy2DAsync(
                                                gpu_ptr.add(gpu_offset) as *mut c_void,
                                                self.stride,
                                                cpu_src,
                                                row_bytes,
                                                row_bytes,
                                                region.height as usize,
                                                CUDA_MEMCPY_HOST_TO_DEVICE,
                                                self.text_stream,
                                            )
                                        };
                                        if copy_back != CUDA_SUCCESS {
                                            eprintln!("[ERROR] cudaMemcpy2DAsync H2D failed: {}", copy_back);
                                        } else {
                                            let record_text_done = unsafe { cudaEventRecord(self.text_done_event, self.text_stream) };
                                            if record_text_done != CUDA_SUCCESS {
                                                eprintln!("[ERROR] cudaEventRecord (text_done) failed: {}", record_text_done);
                                            } else {
                                                text_updated = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if text_updated {
            let wait_main = unsafe { cudaStreamWaitEvent(self.stream, self.text_done_event, 0) };
            if wait_main != CUDA_SUCCESS {
                eprintln!("[ERROR] cudaStreamWaitEvent (text_done) failed: {}", wait_main);
            }
        }
        
        // Synchronize stream เพื่อให้แน่ใจว่า rendering เสร็จ
        let sync_result = unsafe { cudaStreamSynchronize(self.stream) };
        if sync_result != CUDA_SUCCESS {
            eprintln!("[ERROR] cudaStreamSynchronize failed: {}", sync_result);
        } else if self.debug_mode {
            eprintln!("[DEBUG] Stream synchronized successfully");
        }
        
        // Return GPU buffer in BGRA format (ไม่มี CPU copy!)
        OverlayFramePacket {
            from: input.from,
            argb: MemRef {  // Note: field name is 'argb' but actual format is BGRA
                ptr: gpu_ptr,
                len: self.stride * h as usize,
                stride: self.stride,
                loc: MemLoc::Gpu { device: self.device_id },
            },
            stride: self.stride,
        }
    }
}

// GPU-specific methods (outside trait impl)
#[cfg(feature = "gpu")]
impl RenderStage {
    /// Process GPU overlay plan (zero-copy from GPU overlay planning)
    /// Takes GpuOverlayPlanPacket directly without CPU conversion
    /// This is the full zero-copy path: GPU detections → GPU commands → GPU rendering
    pub fn process_gpu(&mut self, input: common_io::GpuOverlayPlanPacket) -> OverlayFramePacket {
        let w = input.canvas.0;
        let h = input.canvas.1;
        
        // Ensure GPU buffer is allocated
        if let Err(e) = self.ensure_buffer(w, h) {
            eprintln!("GPU overlay render error: {}", e);
            return OverlayFramePacket {
                from: input.frame_meta,
                argb: MemRef {
                    ptr: ptr::null_mut(),
                    len: 0,
                    stride: 0,
                    loc: MemLoc::Gpu { device: self.device_id },
                },
                stride: 0,
            };
        }
        
        let gpu_ptr = self.gpu_buf.unwrap();
        
        // Clear buffer (transparent)
        unsafe {
            launch_clear_buffer(
                gpu_ptr,
                self.stride as c_int,
                w as c_int,
                h as c_int,
                self.stream,
            );
        }
        
        // Execute GPU commands directly (FULL ZERO-COPY!)
        // Commands are already on GPU - no CPU transfers except 2 × 4-byte counts
        // GPU kernel separates commands by type and executes in parallel
        
        if self.debug_mode {
            eprintln!("[DEBUG] process_gpu: num_commands={}, gpu_commands={:?}", 
                     input.num_commands, input.gpu_commands);
        }
        
        if input.num_commands > 0 && !input.gpu_commands.is_null() {
            if self.debug_mode {
                eprintln!("[DEBUG] Executing {} GPU commands (full GPU path)", input.num_commands);
            }
            
            // Execute commands on GPU (zero-copy!)
            // Only transfers: 2 × 4 bytes for command counts (rect_count, fill_count)
            unsafe {
                launch_execute_commands(
                    gpu_ptr,
                    self.stride as c_int,
                    w as c_int,
                    h as c_int,
                    input.gpu_commands as *const c_void,
                    input.num_commands as c_int,
                    self.stream,
                );
            }
            
            // Note: Labels (cmd_type == 2) are skipped in GPU path
            // Text rendering requires CPU rasterization
        }        // Synchronize
        let sync_result = unsafe {
            cudaStreamSynchronize(self.stream)
        };
        if sync_result != CUDA_SUCCESS {
            eprintln!("[ERROR] cudaStreamSynchronize failed: {}", sync_result);
        }
        
        // Return GPU buffer (zero-copy!)
        OverlayFramePacket {
            from: input.frame_meta,
            argb: MemRef {
                ptr: gpu_ptr,
                len: self.stride * h as usize,
                stride: self.stride,
                loc: MemLoc::Gpu { device: self.device_id },
            },
            stride: self.stride,
        }
    }
}

impl Drop for RenderStage {
    fn drop(&mut self) {
        if let Some(ptr) = self.gpu_buf {
            unsafe {
                decklink_free_dvp_compatible_buffer(ptr as *mut c_void);
            }
        }
        if !self.render_done_event.is_null() {
            unsafe {
                cudaEventDestroy(self.render_done_event);
            }
        }
        if !self.text_done_event.is_null() {
            unsafe {
                cudaEventDestroy(self.text_done_event);
            }
        }
        if !self.text_stream.is_null() {
            unsafe {
                cudaStreamDestroy(self.text_stream);
            }
        }
        if !self.stream.is_null() {
            unsafe {
                cudaStreamDestroy(self.stream);
            }
        }
    }
}

unsafe impl Send for RenderStage {}
unsafe impl Sync for RenderStage {}
