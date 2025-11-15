use anyhow::Result;
use common_io::{MemLoc, MemRef, OverlayFramePacket, OverlayPlanPacket, Stage, DrawOp};
use rusttype::{Font, Scale, point};
use std::os::raw::{c_int, c_void};
use std::ptr;

// ==================== CUDA FFI ====================
extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> c_int;
    fn cudaFree(dev_ptr: *mut c_void) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
    fn cudaStreamCreate(stream: *mut *mut c_void) -> c_int;
    fn cudaStreamDestroy(stream: *mut c_void) -> c_int;
    fn cudaStreamSynchronize(stream: *mut c_void) -> c_int;
    
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

// ==================== GPU Render Stage ====================
pub struct RenderStage {
    gpu_buf: Option<*mut u8>,
    cpu_buf: Option<Vec<u8>>,  // CPU buffer สำหรับ text rendering
    stream: *mut c_void,
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
            unsafe { cudaFree(ptr as *mut c_void); }
        }
        
        // Allocate GPU buffer ใหม่
        let stride = (width * 4) as usize;
        let size = stride * height as usize;
        let mut dev_ptr: *mut c_void = ptr::null_mut();
        
        let result = unsafe { cudaMalloc(&mut dev_ptr, size) };
        if result != CUDA_SUCCESS || dev_ptr.is_null() {
            anyhow::bail!("Failed to allocate GPU buffer for overlay rendering");
        }
        
        self.gpu_buf = Some(dev_ptr as *mut u8);
        self.width = width;
        self.height = height;
        self.stride = stride;
        
        Ok(())
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
    
    // Create CUDA stream
    let mut stream: *mut c_void = ptr::null_mut();
    let result = unsafe { cudaStreamCreate(&mut stream) };
    if result != CUDA_SUCCESS {
        anyhow::bail!("Failed to create CUDA stream for overlay rendering");
    }
    
    // Load font
    let font_data = include_bytes!("../../../testsupport/DejaVuSans.ttf");
    let font = Font::try_from_bytes(font_data as &[u8])
        .ok_or_else(|| anyhow::anyhow!("Failed to load DejaVuSans.ttf font"))?;
    
    Ok(RenderStage {
        gpu_buf: None,
        cpu_buf: None,
        stream,
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
        
        // Synchronize stream เพื่อให้แน่ใจว่า rendering เสร็จ
        let sync_result = unsafe {
            cudaStreamSynchronize(self.stream)
        };
        if sync_result != CUDA_SUCCESS {
            eprintln!("[ERROR] cudaStreamSynchronize failed: {}", sync_result);
        } else if self.debug_mode {
            eprintln!("[DEBUG] Stream synchronized successfully");
        }
        
        // Check for kernel errors
        extern "C" {
            fn cudaGetLastError() -> c_int;
            fn cudaPeekAtLastError() -> c_int;
        }
        let last_err = unsafe { cudaPeekAtLastError() };
        if last_err != CUDA_SUCCESS {
            eprintln!("[ERROR] CUDA kernel error detected: {}", last_err);
        }
        
        // === TEXT RENDERING (CPU-based) ===
        // Collect all Label operations
        let label_ops: Vec<_> = input.ops.iter()
            .filter_map(|op| match op {
                DrawOp::Label { anchor, text, font_px, color } => Some((*anchor, text.clone(), *font_px, *color)),
                _ => None,
            })
            .collect();
        
        if !label_ops.is_empty() {
            if self.debug_mode {
                eprintln!("[DEBUG] Rendering {} text labels on CPU", label_ops.len());
            }
            
            // Ensure CPU buffer
            let buf_size = self.stride * h as usize;
            if self.cpu_buf.is_none() || self.cpu_buf.as_ref().unwrap().len() != buf_size {
                self.cpu_buf = Some(vec![0u8; buf_size]);
            }
            
            // Copy GPU buffer to CPU
            let cpu_buf = self.cpu_buf.as_mut().unwrap();
            let copy_result = unsafe {
                cudaMemcpy(
                    cpu_buf.as_mut_ptr() as *mut c_void,
                    gpu_ptr as *const c_void,
                    buf_size,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                )
            };
            if copy_result != CUDA_SUCCESS {
                eprintln!("[ERROR] cudaMemcpy D2H failed: {}", copy_result);
            } else {
                // Draw text on CPU buffer (need to split borrow)
                let stride = self.stride;
                let font = &self.font;
                
                for (anchor, text, font_px, color) in label_ops {
                    draw_text_on_buffer(
                        cpu_buf, 
                        stride, 
                        w, 
                        h, 
                        anchor.0 as i32, 
                        anchor.1 as i32, 
                        &text, 
                        font_px as f32, 
                        color,
                        font,
                    );
                }
                
                // Copy back to GPU
                let copy_back = unsafe {
                    cudaMemcpy(
                        gpu_ptr as *mut c_void,
                        cpu_buf.as_ptr() as *const c_void,
                        buf_size,
                        CUDA_MEMCPY_HOST_TO_DEVICE,
                    )
                };
                if copy_back != CUDA_SUCCESS {
                    eprintln!("[ERROR] cudaMemcpy H2D failed: {}", copy_back);
                } else if self.debug_mode {
                    eprintln!("[DEBUG] Text rendering completed, copied back to GPU");
                }
            }
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

impl Drop for RenderStage {
    fn drop(&mut self) {
        if let Some(ptr) = self.gpu_buf {
            unsafe {
                cudaFree(ptr as *mut c_void);
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
