// GPU postprocessing module
// Wraps CUDA NMS kernel to eliminate CPU bottleneck

use anyhow::Result;
use common_io::{BBox, Detection, DetectionsPacket, FrameMeta, GpuDetectionsPacket};
use std::ptr;

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

// CUDA types (minimal definitions)
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
    fn cudaStreamCreate(stream: *mut *mut std::ffi::c_void) -> i32;
    fn cudaStreamDestroy(stream: *mut std::ffi::c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut std::ffi::c_void) -> i32;
}

// External CUDA functions (compiled from nms_gpu.cu)
extern "C" {
    fn gpu_yolo_parse(
        d_raw_output: *const f32,
        d_boxes: *mut f32,
        d_scores: *mut f32,
        d_classes: *mut i32,
        d_count: *mut i32,
        num_anchors: i32,
        num_classes: i32,
        conf_threshold: f32,
        max_detections: i32,
        img_width: i32,
        img_height: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;  // cudaError_t
    
    fn gpu_pack_detections(
        d_boxes: *const f32,
        d_scores: *const f32,
        d_classes: *const i32,
        d_detections: *mut std::ffi::c_void,
        num_detections: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;  // cudaError_t
}

/// GPU-accelerated YOLO postprocessing
/// Keeps inference output on GPU and processes there
pub struct GpuPostprocessor {
    // GPU buffers (persistent across frames)
    d_boxes: *mut f32,      // [MAX_DETS, 4]
    d_scores: *mut f32,     // [MAX_DETS]
    d_classes: *mut i32,    // [MAX_DETS]
    d_count: *mut i32,      // [1]
    
    // Host staging buffer (for final results)
    h_boxes: Vec<f32>,
    h_scores: Vec<f32>,
    h_classes: Vec<i32>,
    
    // Last detection count (cached to avoid extra GPU reads)
    last_count: usize,
    
    // Configuration
    num_anchors: i32,       // 25200 for YOLOv5
    num_classes: i32,       // 80
    max_detections: i32,    // 300
    conf_threshold: f32,
    nms_threshold: f32,
    
    cuda_stream: *mut std::ffi::c_void,
}

impl GpuPostprocessor {
    pub fn new(
        num_anchors: usize,
        num_classes: usize,
        max_detections: usize,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Self> {
        unsafe {
            // Allocate GPU buffers
            let mut d_boxes: *mut f32 = ptr::null_mut();
            let mut d_scores: *mut f32 = ptr::null_mut();
            let mut d_classes: *mut i32 = ptr::null_mut();
            let mut d_count: *mut i32 = ptr::null_mut();
            
            cuda_call!(cudaMalloc(
                &mut d_boxes as *mut *mut f32 as *mut *mut std::ffi::c_void,
                max_detections * 4 * std::mem::size_of::<f32>()
            ))?;
            
            cuda_call!(cudaMalloc(
                &mut d_scores as *mut *mut f32 as *mut *mut std::ffi::c_void,
                max_detections * std::mem::size_of::<f32>()
            ))?;
            
            cuda_call!(cudaMalloc(
                &mut d_classes as *mut *mut i32 as *mut *mut std::ffi::c_void,
                max_detections * std::mem::size_of::<i32>()
            ))?;
            
            cuda_call!(cudaMalloc(
                &mut d_count as *mut *mut i32 as *mut *mut std::ffi::c_void,
                std::mem::size_of::<i32>()
            ))?;
            
            // Create CUDA stream for async operations
            let mut stream: *mut std::ffi::c_void = ptr::null_mut();
            cuda_call!(cudaStreamCreate(&mut stream))?;
            
            // Allocate host buffers
            let h_boxes = vec![0.0f32; max_detections * 4];
            let h_scores = vec![0.0f32; max_detections];
            let h_classes = vec![0i32; max_detections];
            
            Ok(Self {
                d_boxes,
                d_scores,
                d_classes,
                d_count,
                h_boxes,
                h_scores,
                h_classes,
                last_count: 0,
                num_anchors: num_anchors as i32,
                num_classes: num_classes as i32,
                max_detections: max_detections as i32,
                conf_threshold,
                nms_threshold,
                cuda_stream: stream,
            })
        }
    }
    
    /// Process YOLO output on GPU
    /// Input: GPU pointer to raw output [1, 25200, 85]
    /// Output: DetectionsPacket with filtered detections
    pub fn process_gpu(
        &mut self,
        d_raw_output: *const f32,
        meta: &FrameMeta,
    ) -> Result<DetectionsPacket> {
        unsafe {
            // Call CUDA kernel to parse and filter detections
            let err = gpu_yolo_parse(
                d_raw_output,
                self.d_boxes,
                self.d_scores,
                self.d_classes,
                self.d_count,
                self.num_anchors,
                self.num_classes,
                self.conf_threshold,
                self.max_detections,
                meta.width as i32,
                meta.height as i32,
                self.cuda_stream,
            );
            
            if err != 0 {
                return Err(anyhow::anyhow!("CUDA kernel failed with error code {}", err));
            }
            
            // Get detection count
            let mut count: i32 = 0;
            cuda_call!(cudaMemcpyAsync(
                &mut count as *mut i32 as *mut std::ffi::c_void,
                self.d_count as *const std::ffi::c_void,
                std::mem::size_of::<i32>(),
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                self.cuda_stream
            ))?;
            
            // Wait for GPU to finish
            cuda_call!(cudaStreamSynchronize(self.cuda_stream))?;
            
            let count = count.min(self.max_detections) as usize;
            self.last_count = count;  // Save for get_gpu_buffers()
            
            if count == 0 {
                return Ok(DetectionsPacket {
                    from: meta.clone(),
                    items: Vec::new(),
                });
            }
            
            // Copy results to host
            cuda_call!(cudaMemcpy(
                self.h_boxes.as_mut_ptr() as *mut std::ffi::c_void,
                self.d_boxes as *const std::ffi::c_void,
                count * 4 * std::mem::size_of::<f32>(),
                cudaMemcpyKind::cudaMemcpyDeviceToHost
            ))?;
            
            cuda_call!(cudaMemcpy(
                self.h_scores.as_mut_ptr() as *mut std::ffi::c_void,
                self.d_scores as *const std::ffi::c_void,
                count * std::mem::size_of::<f32>(),
                cudaMemcpyKind::cudaMemcpyDeviceToHost
            ))?;
            
            cuda_call!(cudaMemcpy(
                self.h_classes.as_mut_ptr() as *mut std::ffi::c_void,
                self.d_classes as *const std::ffi::c_void,
                count * std::mem::size_of::<i32>(),
                cudaMemcpyKind::cudaMemcpyDeviceToHost
            ))?;
            
            // Convert to DetectionsPacket
            let mut items = Vec::with_capacity(count);
            for i in 0..count {
                let x = self.h_boxes[i * 4 + 0];
                let y = self.h_boxes[i * 4 + 1];
                let w = self.h_boxes[i * 4 + 2];
                let h = self.h_boxes[i * 4 + 3];
                let score = self.h_scores[i];
                let class_id = self.h_classes[i];
                
                items.push(Detection {
                    bbox: BBox { x, y, w, h },
                    score,
                    class_id,
                    track_id: None,
                });
            }
            
            Ok(DetectionsPacket {
                from: meta.clone(),
                items,
            })
        }
    }
    
    pub fn set_confidence_threshold(&mut self, threshold: f32) {
        self.conf_threshold = threshold;
    }
    
    /// Get GPU pointers for zero-copy pipeline (for GPU overlay planning)
    /// Returns (d_boxes, d_scores, d_classes, detection_count)
    /// NOTE: These pointers are only valid until next process_gpu() call
    /// Must call process_gpu() before this to populate buffers
    pub fn get_gpu_buffers(&self) -> (*const f32, *const f32, *const i32, usize) {
        (self.d_boxes, self.d_scores, self.d_classes, self.last_count)
    }
    
    /// Process YOLO output on GPU with zero-copy output (full GPU pipeline)
    /// This method does NOT copy results to CPU, keeping data on GPU for downstream stages.
    /// 
    /// Input: GPU pointer to raw YOLO output [1, 25200, 85]
    /// Output: GpuDetectionsPacket with GPU-resident detection data
    /// 
    /// Use this for full zero-copy GPU pipeline (no CPU transfers).
    /// For debugging or CPU-based overlay, use process_gpu() instead.
    pub fn process_gpu_zerocopy(
        &mut self,
        d_raw_output: *const f32,
        meta: &FrameMeta,
    ) -> Result<GpuDetectionsPacket> {
        unsafe {
            // Call CUDA kernel to parse and filter detections
            let err = gpu_yolo_parse(
                d_raw_output,
                self.d_boxes,
                self.d_scores,
                self.d_classes,
                self.d_count,
                self.num_anchors,
                self.num_classes,
                self.conf_threshold,
                self.max_detections,
                meta.width as i32,
                meta.height as i32,
                self.cuda_stream,
            );
            
            if err != 0 {
                return Err(anyhow::anyhow!("CUDA kernel failed with error code {}", err));
            }
            
            // Get detection count (minimal CPU transfer - just 4 bytes)
            let mut count: i32 = 0;
            cuda_call!(cudaMemcpyAsync(
                &mut count as *mut i32 as *mut std::ffi::c_void,
                self.d_count as *const std::ffi::c_void,
                std::mem::size_of::<i32>(),
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                self.cuda_stream
            ))?;
            
            // Wait for GPU to finish
            cuda_call!(cudaStreamSynchronize(self.cuda_stream))?;
            
            let count = count.min(self.max_detections) as usize;
            self.last_count = count;
            
            // Pack separate arrays into unified Detection format on GPU (zero-copy!)
            let detection_size = std::mem::size_of::<Detection>();
            let mut d_detections: *mut Detection = ptr::null_mut();
            
            if count > 0 {
                cuda_call!(cudaMalloc(
                    &mut d_detections as *mut *mut Detection as *mut *mut std::ffi::c_void,
                    count * detection_size
                ))?;
                
                // Launch GPU kernel to pack detections (zero-copy!)
                let pack_err = gpu_pack_detections(
                    self.d_boxes,
                    self.d_scores,
                    self.d_classes,
                    d_detections as *mut std::ffi::c_void,
                    count as i32,
                    self.cuda_stream
                );
                
                if pack_err != 0 {
                    let _ = cudaFree(d_detections as *mut std::ffi::c_void);
                    return Err(anyhow::anyhow!("GPU pack kernel failed: {}", pack_err));
                }
            }
            
            Ok(GpuDetectionsPacket {
                frame_meta: meta.clone(),
                gpu_detections: d_detections,
                num_detections: count,
                capacity: count,
            })
        }
    }
}

impl Drop for GpuPostprocessor {
    fn drop(&mut self) {
        unsafe {
            if !self.d_boxes.is_null() {
                let _ = cuda_call!(cudaFree(self.d_boxes as *mut std::ffi::c_void));
            }
            if !self.d_scores.is_null() {
                let _ = cuda_call!(cudaFree(self.d_scores as *mut std::ffi::c_void));
            }
            if !self.d_classes.is_null() {
                let _ = cuda_call!(cudaFree(self.d_classes as *mut std::ffi::c_void));
            }
            if !self.d_count.is_null() {
                let _ = cuda_call!(cudaFree(self.d_count as *mut std::ffi::c_void));
            }
            if !self.cuda_stream.is_null() {
                let _ = cuda_call!(cudaStreamDestroy(self.cuda_stream));
            }
        }
    }
}
