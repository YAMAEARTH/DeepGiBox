# Packet Usage Guide - RawFramePacket & TensorInputPacket

## Overview

คู่มือนี้อธิบายการใช้งาน `RawFramePacket` และ `TensorInputPacket` ในระบบ DeepGIBox Pipeline สำหรับการพัฒนาต่อยอดในอนาคต เช่น การต่อ end-to-end pipeline, การวัด latency, debugging, และ optimization

---

## Table of Contents

1. [Packet Structures](#packet-structures)
2. [Getting Started](#getting-started)
3. [Common Use Cases](#common-use-cases)
4. [End-to-End Pipeline Integration](#end-to-end-pipeline-integration)
5. [Latency Measurement](#latency-measurement)
6. [Performance Optimization](#performance-optimization)
7. [Debugging & Validation](#debugging--validation)
8. [Best Practices](#best-practices)

---

## Packet Structures

### RawFramePacket

```rust
pub struct RawFramePacket {
    pub meta: FrameMeta,
    pub data: MemRef,
}

pub struct FrameMeta {
    pub source_id: u32,          // DeckLink device ID
    pub width: u32,              // Frame width in pixels
    pub height: u32,             // Frame height in pixels
    pub pixfmt: PixelFormat,     // YUV422_8, NV12, BGRA8
    pub colorspace: ColorSpace,  // BT709, BT601, BT2020
    pub frame_idx: u64,          // Sequential frame number
    pub pts_ns: u64,             // Presentation timestamp (ns)
    pub t_capture_ns: u64,       // Capture timestamp (ns)
    pub stride_bytes: u32,       // Bytes per row
}

pub struct MemRef {
    pub ptr: *mut u8,            // Pointer to data
    pub len: usize,              // Total data size
    pub stride: usize,           // Stride per row
    pub loc: MemLoc,             // Cpu or Gpu { device }
}
```

**Key Points:**
- Raw video frame from DeckLink capture
- Supports GPU memory (GPUDirect) for zero-copy
- Contains original frame metadata for traceability
- YUV422_8 format (UYVY chroma order) for 1080p60/4K30

### TensorInputPacket

```rust
pub struct TensorInputPacket {
    pub from: FrameMeta,         // Original frame metadata
    pub desc: TensorDesc,        // Tensor descriptor
    pub data: MemRef,            // GPU memory reference
}

pub struct TensorDesc {
    pub n: u32,                  // Batch size (always 1)
    pub c: u32,                  // Channels (always 3 for RGB)
    pub h: u32,                  // Height in pixels
    pub w: u32,                  // Width in pixels
    pub dtype: DType,            // Fp16 or Fp32
    pub device: u32,             // GPU device ID
}
```

**Key Points:**
- Preprocessed tensor ready for inference
- NCHW layout (Batch, Channels, Height, Width)
- FP16 or FP32 data type
- Preserves original frame metadata for traceability

---

## Getting Started

### 1. Capture RawFramePacket from DeckLink

```rust
use decklink_input::capture::CaptureSession;
use common_io::RawFramePacket;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open DeckLink device 0
    let mut session = CaptureSession::open(0)?;
    
    // Get a frame
    loop {
        if let Some(packet) = session.get_frame()? {
            // Process packet
            println!("Captured frame #{}", packet.meta.frame_idx);
            println!("  Size: {}x{}", packet.meta.width, packet.meta.height);
            println!("  Memory: {:?}", packet.data.loc);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    
    Ok(())
}
```

### 2. Preprocess RawFramePacket → TensorInputPacket

```rust
use preprocess_cuda::{ChromaOrder, Preprocessor};
use common_io::Stage;

fn preprocess_example(raw_packet: RawFramePacket) -> Result<TensorInputPacket, Box<dyn std::error::Error>> {
    // Create preprocessor (512x512, FP16, ImageNet normalization)
    let mut preprocessor = Preprocessor::with_params(
        (512, 512),              // Output size
        true,                    // Use FP16
        0,                       // GPU device 0
        [0.485, 0.456, 0.406],  // ImageNet mean
        [0.229, 0.224, 0.225],  // ImageNet std
        ChromaOrder::UYVY,       // Chroma order
    )?;
    
    // Process frame (handles CPU→GPU transfer if needed)
    let tensor = preprocessor.process(raw_packet);
    
    Ok(tensor)
}
```

### 3. Load Preprocessor from Config

```rust
// From TOML config file
let preprocessor = preprocess_cuda::from_path(
    "configs/dev_1080p60_yuv422_fp16_trt.toml"
)?;

let tensor = preprocessor.process(raw_packet);
```

---

## Common Use Cases

### Use Case 1: Basic Frame Inspection

```rust
fn inspect_frame(packet: &RawFramePacket) {
    println!("=== Frame Information ===");
    println!("Frame #{}: {}x{}", 
             packet.meta.frame_idx, 
             packet.meta.width, 
             packet.meta.height);
    println!("Format: {:?}, Color: {:?}", 
             packet.meta.pixfmt, 
             packet.meta.colorspace);
    println!("Memory: {:?}, Size: {} bytes", 
             packet.data.loc, 
             packet.data.len);
    println!("PTS: {} ns, Captured: {} ns", 
             packet.meta.pts_ns, 
             packet.meta.t_capture_ns);
}
```

### Use Case 2: Handle CPU/GPU Memory

```rust
use common_io::MemLoc;
use cudarc::driver::{CudaDevice, DevicePtr, DevicePtrMut};

fn ensure_gpu_memory(mut packet: RawFramePacket) -> Result<RawFramePacket, Box<dyn std::error::Error>> {
    // Check if already on GPU
    if matches!(packet.data.loc, MemLoc::Gpu { .. }) {
        return Ok(packet);
    }
    
    // Transfer CPU → GPU
    let device = CudaDevice::new(0)?;
    let mut gpu_buffer = device.alloc_zeros::<u8>(packet.data.len)?;
    
    let cpu_slice = unsafe {
        std::slice::from_raw_parts(packet.data.ptr, packet.data.len)
    };
    device.htod_sync_copy_into(cpu_slice, &mut gpu_buffer)?;
    
    // Update packet
    packet.data.ptr = *gpu_buffer.device_ptr() as *mut u8;
    packet.data.loc = MemLoc::Gpu { device: 0 };
    std::mem::forget(gpu_buffer);
    
    Ok(packet)
}
```

### Use Case 3: Tensor to Host Memory

```rust
use cudarc::driver::CudaDevice;

fn tensor_to_host(tensor: &TensorInputPacket) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Only works for FP32 tensors
    assert!(matches!(tensor.desc.dtype, common_io::DType::Fp32));
    
    let device = CudaDevice::new(tensor.desc.device)?;
    let total_elements = (tensor.desc.n * tensor.desc.c * tensor.desc.h * tensor.desc.w) as usize;
    
    // Allocate host buffer
    let mut host_data = vec![0.0f32; total_elements];
    
    // Copy GPU → CPU
    let gpu_slice = unsafe {
        std::slice::from_raw_parts(tensor.data.ptr as *const f32, total_elements)
    };
    device.dtoh_sync_copy_into(gpu_slice, &mut host_data)?;
    
    Ok(host_data)
}
```

---

## End-to-End Pipeline Integration

### Complete Pipeline Example

```rust
use decklink_input::capture::CaptureSession;
use preprocess_cuda::Preprocessor;
use inference::TrtInference;
use postprocess::Postprocessor;
use common_io::Stage;

struct Pipeline {
    capture: CaptureSession,
    preprocess: Preprocessor,
    inference: TrtInference,
    postprocess: Postprocessor,
}

impl Pipeline {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            capture: CaptureSession::open(0)?,
            preprocess: preprocess_cuda::from_path("configs/dev_1080p60_yuv422_fp16_trt.toml")?,
            inference: inference::from_path("configs/dev_1080p60_yuv422_fp16_trt.toml")?,
            postprocess: postprocess::from_path("configs/dev_1080p60_yuv422_fp16_trt.toml")?,
        })
    }
    
    pub fn process_frame(&mut self) -> Result<Option<DetectionsPacket>, Box<dyn std::error::Error>> {
        // Stage 1: Capture
        let Some(raw_frame) = self.capture.get_frame()? else {
            return Ok(None);
        };
        
        // Stage 2: Preprocess
        let tensor = self.preprocess.process(raw_frame);
        
        // Stage 3: Inference
        let raw_detections = self.inference.process(tensor);
        
        // Stage 4: Postprocess
        let detections = self.postprocess.process(raw_detections);
        
        Ok(Some(detections))
    }
    
    pub fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        loop {
            if let Some(detections) = self.process_frame()? {
                println!("Frame {}: {} detections", 
                         detections.from.frame_idx,
                         detections.detections.len());
            }
        }
    }
}
```

### Pipeline with Error Handling

```rust
impl Pipeline {
    pub fn process_with_retry(&mut self, max_retries: usize) -> Result<DetectionsPacket, Box<dyn std::error::Error>> {
        for attempt in 0..max_retries {
            match self.process_frame() {
                Ok(Some(result)) => return Ok(result),
                Ok(None) => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    continue;
                }
                Err(e) if attempt < max_retries - 1 => {
                    eprintln!("Attempt {} failed: {}, retrying...", attempt + 1, e);
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        Err("Max retries exceeded".into())
    }
}
```

---

## Latency Measurement

### Timestamp-Based Latency Tracking

```rust
use std::time::{SystemTime, UNIX_EPOCH};

struct LatencyTracker {
    t_capture_ns: u64,
    t_preprocess_start_ns: u64,
    t_preprocess_end_ns: u64,
    t_inference_start_ns: u64,
    t_inference_end_ns: u64,
    t_postprocess_start_ns: u64,
    t_postprocess_end_ns: u64,
}

impl LatencyTracker {
    fn new(raw_frame: &RawFramePacket) -> Self {
        Self {
            t_capture_ns: raw_frame.meta.t_capture_ns,
            t_preprocess_start_ns: 0,
            t_preprocess_end_ns: 0,
            t_inference_start_ns: 0,
            t_inference_end_ns: 0,
            t_postprocess_start_ns: 0,
            t_postprocess_end_ns: 0,
        }
    }
    
    fn now_ns() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
    
    fn mark_preprocess_start(&mut self) {
        self.t_preprocess_start_ns = Self::now_ns();
    }
    
    fn mark_preprocess_end(&mut self) {
        self.t_preprocess_end_ns = Self::now_ns();
    }
    
    fn mark_inference_start(&mut self) {
        self.t_inference_start_ns = Self::now_ns();
    }
    
    fn mark_inference_end(&mut self) {
        self.t_inference_end_ns = Self::now_ns();
    }
    
    fn mark_postprocess_start(&mut self) {
        self.t_postprocess_start_ns = Self::now_ns();
    }
    
    fn mark_postprocess_end(&mut self) {
        self.t_postprocess_end_ns = Self::now_ns();
    }
    
    fn report(&self) {
        let capture_to_preprocess = (self.t_preprocess_start_ns - self.t_capture_ns) as f64 / 1_000_000.0;
        let preprocess_time = (self.t_preprocess_end_ns - self.t_preprocess_start_ns) as f64 / 1_000_000.0;
        let inference_time = (self.t_inference_end_ns - self.t_inference_start_ns) as f64 / 1_000_000.0;
        let postprocess_time = (self.t_postprocess_end_ns - self.t_postprocess_start_ns) as f64 / 1_000_000.0;
        let total = (self.t_postprocess_end_ns - self.t_capture_ns) as f64 / 1_000_000.0;
        
        println!("=== Latency Report ===");
        println!("  Capture → Preprocess: {:.3} ms", capture_to_preprocess);
        println!("  Preprocess:           {:.3} ms", preprocess_time);
        println!("  Inference:            {:.3} ms", inference_time);
        println!("  Postprocess:          {:.3} ms", postprocess_time);
        println!("  Total (E2E):          {:.3} ms", total);
        println!("  Target FPS:           {:.1} fps", 1000.0 / total);
    }
}
```

### Usage Example

```rust
fn process_with_latency_tracking(
    mut pipeline: Pipeline,
    raw_frame: RawFramePacket,
) -> Result<DetectionsPacket, Box<dyn std::error::Error>> {
    let mut tracker = LatencyTracker::new(&raw_frame);
    
    // Preprocess
    tracker.mark_preprocess_start();
    let tensor = pipeline.preprocess.process(raw_frame);
    tracker.mark_preprocess_end();
    
    // Inference
    tracker.mark_inference_start();
    let raw_detections = pipeline.inference.process(tensor);
    tracker.mark_inference_end();
    
    // Postprocess
    tracker.mark_postprocess_start();
    let detections = pipeline.postprocess.process(raw_detections);
    tracker.mark_postprocess_end();
    
    // Report latency
    tracker.report();
    
    Ok(detections)
}
```

### CUDA Event-Based GPU Timing

```rust
use cudarc::driver::{CudaDevice, CudaStream};

struct GpuTimer {
    device: CudaDevice,
    start_event: cudarc::driver::sys::CUevent,
    end_event: cudarc::driver::sys::CUevent,
}

impl GpuTimer {
    fn new(device_id: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let device = CudaDevice::new(device_id)?;
        
        // Create CUDA events
        let mut start_event = std::ptr::null_mut();
        let mut end_event = std::ptr::null_mut();
        
        unsafe {
            cudarc::driver::sys::cuEventCreate(&mut start_event, 0);
            cudarc::driver::sys::cuEventCreate(&mut end_event, 0);
        }
        
        Ok(Self {
            device,
            start_event,
            end_event,
        })
    }
    
    fn start(&self, stream: &CudaStream) {
        unsafe {
            cudarc::driver::sys::cuEventRecord(self.start_event, stream.stream);
        }
    }
    
    fn stop(&self, stream: &CudaStream) {
        unsafe {
            cudarc::driver::sys::cuEventRecord(self.end_event, stream.stream);
        }
    }
    
    fn elapsed_ms(&self) -> f32 {
        unsafe {
            cudarc::driver::sys::cuEventSynchronize(self.end_event);
            let mut elapsed_ms = 0.0f32;
            cudarc::driver::sys::cuEventElapsedTime(&mut elapsed_ms, self.start_event, self.end_event);
            elapsed_ms
        }
    }
}

// Usage
fn measure_preprocessing_gpu_time(preprocessor: &mut Preprocessor, raw_frame: RawFramePacket) -> f32 {
    let timer = GpuTimer::new(0).unwrap();
    
    // Start timing
    timer.start(&preprocessor.cuda_stream);
    
    // Process
    let _tensor = preprocessor.process(raw_frame);
    
    // Stop timing
    timer.stop(&preprocessor.cuda_stream);
    
    timer.elapsed_ms()
}
```

---

## Performance Optimization

### 1. Batch Processing (Future)

```rust
// For future batch inference support
struct BatchProcessor {
    batch_size: usize,
    buffer: Vec<RawFramePacket>,
}

impl BatchProcessor {
    fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            buffer: Vec::with_capacity(batch_size),
        }
    }
    
    fn add_frame(&mut self, frame: RawFramePacket) -> Option<Vec<RawFramePacket>> {
        self.buffer.push(frame);
        
        if self.buffer.len() >= self.batch_size {
            Some(std::mem::take(&mut self.buffer))
        } else {
            None
        }
    }
}
```

### 2. Memory Pool for Tensors

```rust
use std::collections::VecDeque;

struct TensorPool {
    pool: VecDeque<MemRef>,
    max_size: usize,
}

impl TensorPool {
    fn new(max_size: usize) -> Self {
        Self {
            pool: VecDeque::with_capacity(max_size),
            max_size,
        }
    }
    
    fn get(&mut self, size: usize) -> Option<MemRef> {
        self.pool.iter()
            .position(|mem| mem.len >= size)
            .and_then(|idx| self.pool.remove(idx))
    }
    
    fn return_buffer(&mut self, buffer: MemRef) {
        if self.pool.len() < self.max_size {
            self.pool.push_back(buffer);
        }
    }
}
```

### 3. Pipeline Parallelism

```rust
use std::sync::mpsc;
use std::thread;

struct ParallelPipeline {
    capture_tx: mpsc::Sender<RawFramePacket>,
    result_rx: mpsc::Receiver<DetectionsPacket>,
}

impl ParallelPipeline {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let (capture_tx, capture_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        
        // Preprocessing thread
        let (preprocess_tx, preprocess_rx) = mpsc::channel();
        thread::spawn(move || {
            let mut preprocessor = Preprocessor::with_params(...).unwrap();
            for raw_frame in capture_rx {
                let tensor = preprocessor.process(raw_frame);
                preprocess_tx.send(tensor).unwrap();
            }
        });
        
        // Inference thread
        let (inference_tx, inference_rx) = mpsc::channel();
        thread::spawn(move || {
            let mut inference = TrtInference::from_path(...).unwrap();
            for tensor in preprocess_rx {
                let raw_detections = inference.process(tensor);
                inference_tx.send(raw_detections).unwrap();
            }
        });
        
        // Postprocessing thread
        thread::spawn(move || {
            let mut postprocessor = Postprocessor::from_path(...).unwrap();
            for raw_detections in inference_rx {
                let detections = postprocessor.process(raw_detections);
                result_tx.send(detections).unwrap();
            }
        });
        
        Ok(Self { capture_tx, result_rx })
    }
}
```

---

## Debugging & Validation

### 1. Save Raw Frame to Disk

```rust
use std::fs::File;
use std::io::Write;

fn save_raw_frame(packet: &RawFramePacket, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let data = if matches!(packet.data.loc, MemLoc::Gpu { .. }) {
        // Copy GPU → CPU
        let device = CudaDevice::new(0)?;
        let mut host_data = vec![0u8; packet.data.len];
        let gpu_slice = unsafe {
            std::slice::from_raw_parts(packet.data.ptr, packet.data.len)
        };
        device.dtoh_sync_copy_into(gpu_slice, &mut host_data)?;
        host_data
    } else {
        // Already on CPU
        unsafe {
            std::slice::from_raw_parts(packet.data.ptr, packet.data.len).to_vec()
        }
    };
    
    let mut file = File::create(filename)?;
    file.write_all(&data)?;
    
    println!("Saved {} bytes to {}", data.len(), filename);
    Ok(())
}
```

### 2. Visualize Tensor Values

```rust
fn print_tensor_statistics(tensor: &TensorInputPacket) -> Result<(), Box<dyn std::error::Error>> {
    // Only for FP32
    if !matches!(tensor.desc.dtype, common_io::DType::Fp32) {
        return Err("Only FP32 tensors supported".into());
    }
    
    let host_data = tensor_to_host(tensor)?;
    
    let min = host_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = host_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mean = host_data.iter().sum::<f32>() / host_data.len() as f32;
    
    println!("=== Tensor Statistics ===");
    println!("  Shape: {}×{}×{}×{}", tensor.desc.n, tensor.desc.c, tensor.desc.h, tensor.desc.w);
    println!("  Min:   {:.6}", min);
    println!("  Max:   {:.6}", max);
    println!("  Mean:  {:.6}", mean);
    
    Ok(())
}
```

### 3. Frame Comparison

```rust
fn compare_frames(frame1: &RawFramePacket, frame2: &RawFramePacket) -> bool {
    if frame1.meta.width != frame2.meta.width || 
       frame1.meta.height != frame2.meta.height ||
       frame1.data.len != frame2.data.len {
        return false;
    }
    
    // Compare memory (simplified)
    unsafe {
        let slice1 = std::slice::from_raw_parts(frame1.data.ptr, frame1.data.len);
        let slice2 = std::slice::from_raw_parts(frame2.data.ptr, frame2.data.len);
        slice1 == slice2
    }
}
```

---

## Best Practices

### 1. Memory Management

✅ **DO:**
- Use GPUDirect memory when available (check `MemLoc::Gpu`)
- Reuse allocated GPU buffers with buffer pools
- Let preprocessing module handle CPU→GPU transfers automatically

❌ **DON'T:**
- Manually free GPU memory managed by packets (handled internally)
- Assume all frames are GPU memory (first frames may be CPU)
- Copy GPU→CPU unless necessary for debugging

### 2. Error Handling

✅ **DO:**
```rust
// Check frame availability
if let Some(frame) = session.get_frame()? {
    // Process frame
} else {
    // No frame available, continue polling
}

// Handle format changes
if frame.meta.width != expected_width {
    eprintln!("Format changed: {}x{}", frame.meta.width, frame.meta.height);
}
```

❌ **DON'T:**
```rust
// Panic on missing frame
let frame = session.get_frame()?.unwrap();  // May panic!

// Ignore format changes
preprocessor.process(frame);  // May fail if format changed
```

### 3. Metadata Preservation

✅ **DO:**
- Always propagate `FrameMeta` through pipeline stages
- Use `frame_idx` for frame tracking and debugging
- Use `t_capture_ns` for latency measurement

```rust
// Good: Metadata flows through pipeline
RawFramePacket { meta, data } 
  → TensorInputPacket { from: meta, desc, data }
  → RawDetectionsPacket { from: meta, raw_output }
  → DetectionsPacket { from: meta, detections }
```

### 4. Configuration Management

✅ **DO:**
```rust
// Load from config file
let preprocessor = preprocess_cuda::from_path("configs/production.toml")?;
let inference = inference::from_path("configs/production.toml")?;
```

❌ **DON'T:**
```rust
// Hardcode parameters
let preprocessor = Preprocessor::with_params(
    (512, 512), true, 0, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], ChromaOrder::UYVY
)?;  // Hard to maintain!
```

### 5. Performance Monitoring

✅ **DO:**
- Measure latency for each stage
- Log frame drops and format changes
- Monitor GPU memory usage
- Track FPS over time

```rust
struct PerformanceMonitor {
    frame_count: u64,
    start_time: Instant,
    last_report: Instant,
}

impl PerformanceMonitor {
    fn update(&mut self) {
        self.frame_count += 1;
        
        if self.last_report.elapsed() >= Duration::from_secs(1) {
            let elapsed = self.start_time.elapsed().as_secs_f64();
            let fps = self.frame_count as f64 / elapsed;
            println!("FPS: {:.2}", fps);
            self.last_report = Instant::now();
        }
    }
}
```

---

## Testing & Validation

### Test Programs

1. **`test_rawframepacket`**: Validate DeckLink capture
   ```bash
   cargo run -p playgrounds --bin test_rawframepacket
   ```

2. **`test_tensorinputpacket`**: Validate preprocessing
   ```bash
   cargo run -p playgrounds --bin test_tensorinputpacket
   ```

### Integration Tests

```rust
#[test]
fn test_end_to_end_pipeline() {
    let mut pipeline = Pipeline::new().unwrap();
    
    // Process 10 frames
    for i in 0..10 {
        let detections = pipeline.process_with_retry(5).unwrap();
        assert_eq!(detections.from.frame_idx, i);
        assert!(detections.detections.len() > 0);
    }
}
```

---

## Future Enhancements

### Planned Features

1. **Multi-stream Support**: Process multiple DeckLink devices simultaneously
2. **Batch Inference**: Process multiple frames in one inference call
3. **Dynamic Resolution**: Support runtime resolution changes
4. **Hardware Encoding**: Add H.264/HEVC encoding for output
5. **Network Streaming**: RTSP/WebRTC output support

### Extension Points

```rust
// Custom stage implementation
pub trait CustomStage {
    type Input;
    type Output;
    
    fn process(&mut self, input: Self::Input) -> Self::Output;
}

// Example: Custom preprocessing
impl CustomStage for MyPreprocessor {
    type Input = RawFramePacket;
    type Output = TensorInputPacket;
    
    fn process(&mut self, input: Self::Input) -> Self::Output {
        // Custom logic here
    }
}
```

---

## Troubleshooting

### Common Issues

**Issue**: Frame is CPU memory instead of GPU
```rust
// Solution: Check DeckLink GPUDirect setup
if matches!(frame.data.loc, MemLoc::Cpu) {
    eprintln!("Warning: Frame on CPU, GPUDirect may not be enabled");
    // Automatic transfer will happen in preprocessing
}
```

**Issue**: Preprocessing latency too high
```rust
// Solution: Ensure GPU memory is used
assert!(matches!(frame.data.loc, MemLoc::Gpu { .. }));
// GPU preprocessing: ~0.14 ms
// CPU→GPU + preprocessing: ~1.5 ms
```

**Issue**: Format change during capture
```rust
// Solution: Handle format changes gracefully
if frame.meta.width != last_width {
    eprintln!("Format changed: {}x{} → {}x{}", 
              last_width, last_height, 
              frame.meta.width, frame.meta.height);
    // Recreate preprocessor if needed
}
```

---

## References

- **INSTRUCTION.md**: Pipeline architecture and data flow
- **preprocessing_guideline.md**: Preprocessing specifications
- **CHANGES_DECKLINK_INPUT.md**: DeckLink capture changes
- **PREPROCESSING_STATUS.md**: Preprocessing implementation status
- **TEST_RAWFRAMEPACKET_RESULTS.md**: RawFramePacket test results
- **TEST_TENSORINPUTPACKET_RESULTS.md**: TensorInputPacket test results

---

## Contact & Support

สำหรับคำถามหรือปัญหา:
1. ตรวจสอบ documentation ใน project root
2. รัน test programs เพื่อ validate setup
3. ตรวจสอบ GPU memory และ CUDA version
4. ดู logs ใน `/tmp` สำหรับ debugging

---

**Last Updated**: October 13, 2025  
**Version**: 1.0  
**Status**: Production Ready ✅
