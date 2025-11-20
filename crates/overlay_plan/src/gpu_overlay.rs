// GPU Overlay Planning Module
// Generates draw commands directly on GPU from detection results

use anyhow::Result;
use common_io::FrameMeta;

// Helper macro for CUDA error checking (must be defined before use)
macro_rules! cuda_call {
    ($expr:expr) => {{
        let err = $expr;
        if err != 0 {
            Err(anyhow::anyhow!("CUDA error: {}", err))
        } else {
            Ok(())
        }
    }};
}

// CUDA types
#[repr(C)]
#[allow(non_camel_case_types)]
enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
}

// External CUDA runtime functions
extern "C" {
    fn cudaMalloc(devPtr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaFree(devPtr: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> i32;
    fn cudaMemcpyAsync(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: cudaMemcpyKind,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    fn cudaMemsetAsync(
        devPtr: *mut std::ffi::c_void,
        value: i32,
        count: usize,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    fn cudaStreamCreate(stream: *mut *mut std::ffi::c_void) -> i32;
    fn cudaStreamDestroy(stream: *mut std::ffi::c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut std::ffi::c_void) -> i32;
}

// GPU draw command structures (match CUDA side)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuDrawRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub thickness: u8,
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuDrawLabel {
    pub x: f32,
    pub y: f32,
    pub font_size: u16,
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
    pub text: [u8; 64],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union GpuDrawData {
    pub rect: GpuDrawRect,
    pub label: GpuDrawLabel,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuDrawCommand {
    pub cmd_type: i32,  // 0=RECT, 1=FILL_RECT, 2=LABEL
    pub data: GpuDrawData,
}

// Manual Debug impl (union can't derive Debug)
impl std::fmt::Debug for GpuDrawCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDrawCommand")
            .field("cmd_type", &self.cmd_type)
            .finish()
    }
}

// External CUDA kernel functions
extern "C" {
    fn gpu_unpack_detections(
        d_detections: *const std::ffi::c_void,
        d_boxes: *mut f32,
        d_scores: *mut f32,
        d_classes: *mut i32,
        num_detections: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;  // cudaError_t
    
    fn gpu_generate_overlay(
        d_boxes: *const f32,
        d_scores: *const f32,
        d_classes: *const i32,
        num_detections: i32,
        d_commands: *mut GpuDrawCommand,
        d_command_count: *mut i32,
        img_width: i32,
        img_height: i32,
        conf_threshold: f32,
        max_commands: i32,
        draw_confidence_bar: bool,
        stream: *mut std::ffi::c_void,
    ) -> i32;  // cudaError_t
}

/// GPU-accelerated overlay planning
/// Generates draw commands directly on GPU
pub struct GpuOverlayPlanner {
    // GPU buffers
    d_commands: *mut GpuDrawCommand,
    d_command_count: *mut i32,
    
    // Host staging
    h_commands: Vec<GpuDrawCommand>,
    
    // Configuration
    max_commands: i32,
    conf_threshold: f32,
    draw_confidence_bar: bool,
    
    // CUDA stream
    cuda_stream: *mut std::ffi::c_void,
}

impl GpuOverlayPlanner {
    pub fn new(max_commands: usize, conf_threshold: f32, draw_confidence_bar: bool) -> Result<Self> {
        unsafe {
            // Allocate GPU buffers
            let mut d_commands: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_command_count: *mut std::ffi::c_void = std::ptr::null_mut();
            
            cuda_call!(cudaMalloc(
                &mut d_commands,
                max_commands * std::mem::size_of::<GpuDrawCommand>(),
            ))?;
            
            cuda_call!(cudaMalloc(
                &mut d_command_count,
                std::mem::size_of::<i32>(),
            ))?;
            
            // Create CUDA stream
            let mut stream: *mut std::ffi::c_void = std::ptr::null_mut();
            cuda_call!(cudaStreamCreate(&mut stream))?;
            
            // Allocate host staging buffer
            let h_commands = vec![
                GpuDrawCommand {
                    cmd_type: 0,
                    data: GpuDrawData {
                        rect: GpuDrawRect {
                            x: 0.0, y: 0.0, w: 0.0, h: 0.0,
                            thickness: 0, r: 0, g: 0, b: 0, a: 0,
                        }
                    }
                };
                max_commands
            ];
            
            Ok(Self {
                d_commands: d_commands as *mut GpuDrawCommand,
                d_command_count: d_command_count as *mut i32,
                h_commands,
                max_commands: max_commands as i32,
                conf_threshold,
                draw_confidence_bar,
                cuda_stream: stream,
            })
        }
    }
    
    /// Generate overlay commands from DetectionsPacket (high-level API)
    /// Automatically handles CPU→GPU transfer
    pub fn plan_from_detections(
        &mut self,
        detections: &common_io::DetectionsPacket,
    ) -> Result<GpuOverlayPlan> {
        // Prepare detection data
        let (boxes, scores, classes) = prepare_detections_for_gpu(detections);
        let num_dets = detections.items.len();
        
        if num_dets == 0 {
            // No detections - return empty plan
            return Ok(GpuOverlayPlan {
                commands: Vec::new(),
                frame_meta: detections.from.clone(),
            });
        }
        
        unsafe {
            // Allocate temporary GPU buffers for input data
            let mut d_boxes_temp: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_scores_temp: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_classes_temp: *mut std::ffi::c_void = std::ptr::null_mut();
            
            cuda_call!(cudaMalloc(&mut d_boxes_temp, boxes.len() * std::mem::size_of::<f32>()))?;
            cuda_call!(cudaMalloc(&mut d_scores_temp, scores.len() * std::mem::size_of::<f32>()))?;
            cuda_call!(cudaMalloc(&mut d_classes_temp, classes.len() * std::mem::size_of::<i32>()))?;
            
            // Copy data to GPU
            cuda_call!(cudaMemcpyAsync(
                d_boxes_temp,
                boxes.as_ptr() as *const std::ffi::c_void,
                boxes.len() * std::mem::size_of::<f32>(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.cuda_stream,
            ))?;
            
            cuda_call!(cudaMemcpyAsync(
                d_scores_temp,
                scores.as_ptr() as *const std::ffi::c_void,
                scores.len() * std::mem::size_of::<f32>(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.cuda_stream,
            ))?;
            
            cuda_call!(cudaMemcpyAsync(
                d_classes_temp,
                classes.as_ptr() as *const std::ffi::c_void,
                classes.len() * std::mem::size_of::<i32>(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.cuda_stream,
            ))?;
            
            // Call kernel
            let result = self.plan_gpu(
                d_boxes_temp as *const f32,
                d_scores_temp as *const f32,
                d_classes_temp as *const i32,
                num_dets,
                &detections.from,
            );
            
            // Cleanup temporary buffers
            let _ = cuda_call!(cudaFree(d_boxes_temp));
            let _ = cuda_call!(cudaFree(d_scores_temp));
            let _ = cuda_call!(cudaFree(d_classes_temp));
            
            result
        }
    }
    
    /// Generate overlay commands on GPU (low-level API)
    /// Takes GPU pointers from postprocessing stage
    pub fn plan_gpu(
        &mut self,
        d_boxes: *const f32,      // GPU pointer from postprocessing
        d_scores: *const f32,     // GPU pointer
        d_classes: *const i32,    // GPU pointer
        num_detections: usize,
        frame_meta: &FrameMeta,
    ) -> Result<GpuOverlayPlan> {
        unsafe {
            // Call CUDA kernel to generate commands
            let err = gpu_generate_overlay(
                d_boxes,
                d_scores,
                d_classes,
                num_detections as i32,
                self.d_commands,
                self.d_command_count,
                frame_meta.width as i32,
                frame_meta.height as i32,
                self.conf_threshold,
                self.max_commands,
                self.draw_confidence_bar,
                self.cuda_stream,
            );
            
            if err != 0 {
                return Err(anyhow::anyhow!("CUDA kernel failed with error code {}", err));
            }
            
            // Synchronize to ensure completion
            cuda_call!(cudaStreamSynchronize(self.cuda_stream))?;
            
            // Copy command count back to host
            let mut h_count: i32 = 0;
            cuda_call!(cudaMemcpy(
                &mut h_count as *mut i32 as *mut std::ffi::c_void,
                self.d_command_count as *const std::ffi::c_void,
                std::mem::size_of::<i32>(),
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            ))?;
            
            // Copy commands back to host
            if h_count > 0 {
                let count = h_count.min(self.max_commands) as usize;
                cuda_call!(cudaMemcpy(
                    self.h_commands.as_mut_ptr() as *mut std::ffi::c_void,
                    self.d_commands as *const std::ffi::c_void,
                    count * std::mem::size_of::<GpuDrawCommand>(),
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                ))?;
            }
            
            Ok(GpuOverlayPlan {
                commands: self.h_commands[..h_count as usize].to_vec(),
                frame_meta: frame_meta.clone(),
            })
        }
    }
    
    pub fn set_confidence_threshold(&mut self, threshold: f32) {
        self.conf_threshold = threshold;
    }
    
    pub fn set_draw_confidence_bar(&mut self, enabled: bool) {
        self.draw_confidence_bar = enabled;
    }
    
    /// Generate overlay commands from GPU detections (zero-copy)
    /// Takes GpuDetectionsPacket directly, no CPU transfers
    /// Returns GpuOverlayPlanPacket with commands on GPU
    pub fn plan_from_gpu_detections(
        &mut self,
        gpu_detections: &common_io::GpuDetectionsPacket,
    ) -> Result<common_io::GpuOverlayPlanPacket> {
        if gpu_detections.num_detections == 0 {
            return Ok(common_io::GpuOverlayPlanPacket {
                frame_meta: gpu_detections.frame_meta.clone(),
                gpu_commands: std::ptr::null_mut(),
                num_commands: 0,
                canvas: (gpu_detections.frame_meta.width, gpu_detections.frame_meta.height),
            });
        }
        
        unsafe {
            // Allocate temporary GPU arrays for unpacked data
            let num_dets = gpu_detections.num_detections;
            let mut d_boxes_temp: *mut f32 = std::ptr::null_mut();
            let mut d_scores_temp: *mut f32 = std::ptr::null_mut();
            let mut d_classes_temp: *mut i32 = std::ptr::null_mut();
            
            cuda_call!(cudaMalloc(
                &mut d_boxes_temp as *mut *mut f32 as *mut *mut std::ffi::c_void,
                num_dets * 4 * std::mem::size_of::<f32>()
            ))?;
            cuda_call!(cudaMalloc(
                &mut d_scores_temp as *mut *mut f32 as *mut *mut std::ffi::c_void,
                num_dets * std::mem::size_of::<f32>()
            ))?;
            cuda_call!(cudaMalloc(
                &mut d_classes_temp as *mut *mut i32 as *mut *mut std::ffi::c_void,
                num_dets * std::mem::size_of::<i32>()
            ))?;
            
            // Unpack Detection structs on GPU (zero-copy!)
            let unpack_err = gpu_unpack_detections(
                gpu_detections.gpu_detections as *const std::ffi::c_void,
                d_boxes_temp,
                d_scores_temp,
                d_classes_temp,
                num_dets as i32,
                self.cuda_stream
            );
            
            if unpack_err != 0 {
                // Clean up on error
                let _ = cudaFree(d_boxes_temp as *mut std::ffi::c_void);
                let _ = cudaFree(d_scores_temp as *mut std::ffi::c_void);
                let _ = cudaFree(d_classes_temp as *mut std::ffi::c_void);
                return Err(anyhow::anyhow!("GPU unpack failed: {}", unpack_err));
            }
            
            // Generate overlay commands on GPU (zero-copy!)
            let err = gpu_generate_overlay(
                d_boxes_temp,
                d_scores_temp,
                d_classes_temp,
                num_dets as i32,
                self.d_commands,
                self.d_command_count,
                gpu_detections.frame_meta.width as i32,
                gpu_detections.frame_meta.height as i32,
                self.conf_threshold,
                self.max_commands,
                self.draw_confidence_bar,
                self.cuda_stream,
            );
            
            if err != 0 {
                // Cleanup
                let _ = cuda_call!(cudaFree(d_boxes_temp as *mut std::ffi::c_void));
                let _ = cuda_call!(cudaFree(d_scores_temp as *mut std::ffi::c_void));
                let _ = cuda_call!(cudaFree(d_classes_temp as *mut std::ffi::c_void));
                return Err(anyhow::anyhow!("CUDA kernel failed with error code {}", err));
            }
            
            // Get command count (minimal transfer)
            let mut h_count: i32 = 0;
            cuda_call!(cudaMemcpy(
                &mut h_count as *mut i32 as *mut std::ffi::c_void,
                self.d_command_count as *const std::ffi::c_void,
                std::mem::size_of::<i32>(),
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            ))?;
            
            // Cleanup temporary buffers
            let _ = cuda_call!(cudaFree(d_boxes_temp as *mut std::ffi::c_void));
            let _ = cuda_call!(cudaFree(d_scores_temp as *mut std::ffi::c_void));
            let _ = cuda_call!(cudaFree(d_classes_temp as *mut std::ffi::c_void));
            
            // Return GPU command buffer (zero-copy!)
            Ok(common_io::GpuOverlayPlanPacket {
                frame_meta: gpu_detections.frame_meta.clone(),
                gpu_commands: self.d_commands as *mut u8,
                num_commands: h_count.min(self.max_commands) as usize,
                canvas: (gpu_detections.frame_meta.width, gpu_detections.frame_meta.height),
            })
        }
    }
}

impl Drop for GpuOverlayPlanner {
    fn drop(&mut self) {
        unsafe {
            if !self.d_commands.is_null() {
                let _ = cuda_call!(cudaFree(self.d_commands as *mut std::ffi::c_void));
            }
            if !self.d_command_count.is_null() {
                let _ = cuda_call!(cudaFree(self.d_command_count as *mut std::ffi::c_void));
            }
            if !self.cuda_stream.is_null() {
                let _ = cuda_call!(cudaStreamDestroy(self.cuda_stream));
            }
        }
    }
}

// GPU overlay plan result
#[derive(Clone, Debug)]
pub struct GpuOverlayPlan {
    pub commands: Vec<GpuDrawCommand>,
    pub frame_meta: FrameMeta,
}

// Helper to convert GpuDrawCommand to something usable by renderer
impl GpuDrawCommand {
    pub fn command_type(&self) -> DrawCommandType {
        match self.cmd_type {
            0 => DrawCommandType::Rect,
            1 => DrawCommandType::FillRect,
            2 => DrawCommandType::Label,
            _ => DrawCommandType::Rect,
        }
    }
    
    pub fn get_rect(&self) -> Option<DrawRect> {
        if self.cmd_type == 0 || self.cmd_type == 1 {
            unsafe {
                Some(DrawRect {
                    x: self.data.rect.x,
                    y: self.data.rect.y,
                    w: self.data.rect.w,
                    h: self.data.rect.h,
                    thickness: self.data.rect.thickness,
                    r: self.data.rect.r,
                    g: self.data.rect.g,
                    b: self.data.rect.b,
                    a: self.data.rect.a,
                })
            }
        } else {
            None
        }
    }
    
    pub fn get_label(&self) -> Option<DrawLabel> {
        if self.cmd_type == 2 {
            unsafe {
                // Convert [u8; 64] to String
                let null_pos = self.data.label.text.iter().position(|&c| c == 0).unwrap_or(64);
                let text = String::from_utf8_lossy(&self.data.label.text[..null_pos]).to_string();
                
                Some(DrawLabel {
                    x: self.data.label.x,
                    y: self.data.label.y,
                    font_size: self.data.label.font_size,
                    r: self.data.label.r,
                    g: self.data.label.g,
                    b: self.data.label.b,
                    a: self.data.label.a,
                    text,
                })
            }
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum DrawCommandType {
    Rect,
    FillRect,
    Label,
}

#[derive(Clone, Debug)]
pub struct DrawRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub thickness: u8,
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

#[derive(Clone, Debug)]
pub struct DrawLabel {
    pub x: f32,
    pub y: f32,
    pub font_size: u16,
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
    pub text: String,
}

// Make thread-safe (required for Send trait)
unsafe impl Send for GpuOverlayPlanner {}
unsafe impl Sync for GpuOverlayPlanner {}

// ============================================================
// CONVERSION LAYER: GPU → CPU format
// ============================================================

/// Helper to prepare detection data for GPU processing
/// Converts DetectionsPacket to GPU-compatible arrays
pub fn prepare_detections_for_gpu(
    detections: &common_io::DetectionsPacket,
) -> (Vec<f32>, Vec<f32>, Vec<i32>) {
    let num_dets = detections.items.len();
    
    // Allocate arrays
    let mut boxes = Vec::with_capacity(num_dets * 4);
    let mut scores = Vec::with_capacity(num_dets);
    let mut classes = Vec::with_capacity(num_dets);
    
    // Pack detection data into flat arrays
    for det in &detections.items {
        boxes.push(det.bbox.x);
        boxes.push(det.bbox.y);
        boxes.push(det.bbox.w);
        boxes.push(det.bbox.h);
        scores.push(det.score);
        classes.push(det.class_id as i32);
    }
    
    (boxes, scores, classes)
}

/// Convert GpuOverlayPlan to OverlayPlanPacket for CPU render stage
/// This allows GPU overlay planning to work with existing render pipeline
pub fn convert_gpu_to_overlay_plan(
    gpu_plan: GpuOverlayPlan,
    canvas: (u32, u32),
) -> Result<common_io::OverlayPlanPacket> {
    use common_io::DrawOp;
    
    let mut ops = Vec::with_capacity(gpu_plan.commands.len());
    
    for cmd in &gpu_plan.commands {
        match cmd.command_type() {
            DrawCommandType::Rect => {
                if let Some(rect) = cmd.get_rect() {
                    ops.push(DrawOp::Rect {
                        xywh: (rect.x, rect.y, rect.w, rect.h),
                        thickness: rect.thickness,
                        color: (rect.r, rect.g, rect.b, rect.a),
                    });
                }
            }
            DrawCommandType::FillRect => {
                if let Some(rect) = cmd.get_rect() {
                    ops.push(DrawOp::FillRect {
                        xywh: (rect.x, rect.y, rect.w, rect.h),
                        color: (rect.r, rect.g, rect.b, rect.a),
                    });
                }
            }
            DrawCommandType::Label => {
                if let Some(label) = cmd.get_label() {
                    ops.push(DrawOp::Label {
                        anchor: (label.x, label.y),
                        text: label.text,
                        font_px: label.font_size,
                        color: (label.r, label.g, label.b, label.a),
                    });
                }
            }
        }
    }
    
    Ok(common_io::OverlayPlanPacket {
        from: gpu_plan.frame_meta,
        ops,
        canvas,
    })
}
