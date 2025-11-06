use anyhow::Result;
use common_io::{MemLoc, MemRef, OverlayFramePacket, OverlayPlanPacket, Stage, DrawOp};
use std::os::raw::{c_int, c_void};
use std::ptr;

// ==================== CUDA FFI ====================
extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> c_int;
    fn cudaFree(dev_ptr: *mut c_void) -> c_int;
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

// ==================== GPU Render Stage ====================
pub struct RenderStage {
    gpu_buf: Option<*mut u8>,
    stream: *mut c_void,
    width: u32,
    height: u32,
    stride: usize,
    device_id: u32,
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
    
    // Create CUDA stream
    let mut stream: *mut c_void = ptr::null_mut();
    let result = unsafe { cudaStreamCreate(&mut stream) };
    if result != CUDA_SUCCESS {
        anyhow::bail!("Failed to create CUDA stream for overlay rendering");
    }
    
    Ok(RenderStage {
        gpu_buf: None,
        stream,
        width: 0,
        height: 0,
        stride: 0,
        device_id,
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
                    eprintln!("[DEBUG] Op #{}: Rect at ({},{}) size {}x{} thickness={} color=RGBA({},{},{},{})",
                              i, xywh.0, xywh.1, xywh.2, xywh.3, thickness, color.0, color.1, color.2, color.3);
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
                    eprintln!("[DEBUG] Op #{}: FillRect at ({},{}) size {}x{} color=RGBA({},{},{},{})",
                              i, xywh.0, xywh.1, xywh.2, xywh.3, color.0, color.1, color.2, color.3);
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
                        eprintln!("[DEBUG] Op #{}: Line from ({},{}) to ({},{}) thickness={} color=RGBA({},{},{},{})",
                                  i, pts[0].0, pts[0].1, pts[1].0, pts[1].1, thickness, color.0, color.1, color.2, color.3);
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
                DrawOp::Label { .. } => {
                    // TODO: GPU text rendering (ใช้ texture atlas หรือ SDF fonts)
                    // ตอนนี้ skip ไปก่อน
                }
            }
        }
        
        // Synchronize stream เพื่อให้แน่ใจว่า rendering เสร็จ
        let sync_result = unsafe {
            cudaStreamSynchronize(self.stream)
        };
        if sync_result != CUDA_SUCCESS {
            eprintln!("[ERROR] cudaStreamSynchronize failed: {}", sync_result);
        } else {
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
        
        // Return GPU buffer (ไม่มี CPU copy!)
        OverlayFramePacket {
            from: input.from,
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
