Pipeline: Capture → Preprocess → Inference → Post → Plan → GPU BGRA → Internal Keying (Async)

Overview
- Source: SDI via DeckLink capture
- Processing: CUDA preprocessing → TensorRT inference (GPU) → CPU postprocess + tracking → overlay plan
- Rendering: GPU-only BGRA overlay (CUDA kernels)
- Output: DeckLink scheduled playback with hardware internal keying (uses BGRA key over live fill)
- Timing: Hardware reference clock, pre-roll, callback-driven scheduling, adaptive queue

This document explains each stage used by apps/playgrounds/src/bin/pipeline_capture_to_output_v5_keying.rs, including the libraries/shims involved and where memory is allocated (CPU vs GPU) at every step.

Pipeline Steps (Technical)
1) SDI Capture (DeckLink)
- Code: apps/playgrounds/src/bin/pipeline_capture_to_output_v5_keying.rs:270 (CaptureSession::open), 468 (capture.get_frame())
- Crate: crates/decklink_input (Rust)
- Shim: shim/shim.cpp (C++ DeckLink SDK bridge)
- Pixel/Color: UYVY YUV422 8-bit, BT.709
- Data type: common_io::RawFramePacket with common_io::MemRef
- Memory location:
  - Prefer GPU: The shim attempts GPU-backed capture and reports MemLoc::Gpu if available; otherwise MemLoc::Cpu
  - File: crates/decklink_input/src/capture.rs: returns RawFramePacket with MemLoc::{Gpu|Cpu}
- Timestamps:
  - The shim surfaces a hardware capture timestamp (ns). Capture code computes true capture latency from hardware vs system clocks (see crates/decklink_input/src/capture.rs and pipeline log prints)
- Notes:
  - The shim integrates NVIDIA DVP (dvpapi_cuda.h) for zero-copy DMA paths when available; see shim/shim.cpp (DVPContext, DvpAllocator)

2) CPU→GPU staging (if needed)
- Code: apps/playgrounds/src/bin/pipeline_capture_to_output_v5_keying.rs:500–534
- Library: cudarc::driver (Rust CUDA wrapper)
- Operation:
  - If capture arrives in CPU memory, the code allocates a GPU buffer and performs htod_sync_copy
  - Keeps an owning Vec<CudaSlice<u8>> as a small (triple) buffer pool so CUdevice memory remains alive for downstream stages
- Result: RawFramePacket in GPU memory (UYVY), MemLoc::Gpu

3) CUDA Preprocessing (UYVY → NCHW float, crop/resize/normalize)
- Code: apps/playgrounds/src/bin/pipeline_capture_to_output_v5_keying.rs:536–560
- Crate: crates/preprocess_cuda (Rust + CUDA kernel)
- Kernels: crates/preprocess_cuda/preprocess.cu compiled to PTX via crates/preprocess_cuda/build.rs
- Library: cudarc (loads PTX, launches kernels, manages CUDA stream)
- Input: RawFramePacket (UYVY, GPU)
- Output: common_io::TensorInputPacket { desc: N=1,C=3,H=512,W=512, dtype=Fp32|Fp16, device=0; data: MemLoc::Gpu }
- Details:
  - Crop region (Olympus by default) is computed per input resolution (1080p/4K) then fused resize/normalize
  - Pixel formats supported: BGRA8, NV12, YUV422_8 (uses chroma order UYVY or YUY2)
  - Buffering: Preprocessor caches/owns output CUdeviceptr in an internal buffer pool keyed by (W,H,FP16) to avoid re-allocations
  - Files: crates/preprocess_cuda/src/lib.rs (ChromaOrder, CropRegion, buffer_pool, launch_kernel)

4) TensorRT Inference (GPU-only I/O)
- Code: apps/playgrounds/src/bin/pipeline_capture_to_output_v5_keying.rs:562–569
- Crate: crates/inference_v2 (Rust)
- Shim: trt-shim (C++/CUDA TensorRT bridge), built as libtrt_shim.so
- API used (via libloading): create_session, get_device_buffers, run_inference_device, copy_output_to_cpu
- Flow:
  - Zero-copy input: pass GPU pointer from TensorInputPacket directly to TensorRT (run_inference_device)
  - Output buffer: TensorRT-managed GPU buffer queried once (get_device_buffers)
  - Copy result to CPU: copy_output_to_cpu (for CPU-side postprocessing)
- Output: common_io::RawDetectionsPacket { raw_output: Vec<f32> (CPU), output_shape }
- Files: crates/inference_v2/src/lib.rs, trt-shim/src/trt_shim.cpp

5) Postprocessing (CPU: decode + NMS + optional SORT)
- Code: apps/playgrounds/src/bin/pipeline_capture_to_output_v5_keying.rs:573–594
- Crate: crates/postprocess (Rust)
- Input: RawDetectionsPacket (CPU)
- Work:
  - YOLO decode, NMS, temporal smoothing, optional SORT tracking
  - Coordinate transform accounts for crop + stretch-resize used by CUDA preprocessor
- Output: common_io::DetectionsPacket { items: Vec<Detection>, from: FrameMeta }
- Files: crates/postprocess/src/lib.rs (PostStage)

6) Overlay Planning (CPU → draw ops)
- Code: apps/playgrounds/src/bin/pipeline_capture_to_output_v5_keying.rs:596–611
- Crate: crates/overlay_plan (Rust)
- Input: DetectionsPacket
- Output: common_io::OverlayPlanPacket { ops: Vec<DrawOp>, canvas: (w,h) }
- Notes: Produces UI primitives (rects, polys, labels) scaled for 1080p/4K
- File: crates/overlay_plan/src/lib.rs

7) GPU Overlay Rendering (BGRA, GPU-only)
- Code: apps/playgrounds/src/bin/pipeline_capture_to_output_v5_keying.rs:613–638
- Crate: crates/overlay_render (Rust + CUDA)
- Kernels: crates/overlay_render/overlay_render.cu (clear, line, rect, fill_rect → BGRA)
- Library/FFI: Direct calls to cudaMalloc/cudaFree/cudaStream* and kernel launch stubs
- Input: OverlayPlanPacket (CPU ops)
- Output: common_io::OverlayFramePacket { argb: MemRef { ptr: GPU BGRA, loc: MemLoc::Gpu }, stride }
- Memory:
  - Allocates and reuses a persistent GPU BGRA framebuffer sized to output resolution (via cudaMalloc)
  - No CPU copy back; output remains on GPU
- Files: crates/overlay_render/src/lib.rs, crates/overlay_render/overlay_render.cu

8) Hardware Internal Keying + Scheduled Playback (DeckLink Output)
- Code: apps/playgrounds/src/bin/pipeline_capture_to_output_v5_keying.rs:340–447 (setup), 640–760 (schedule path)
- Crate: crates/decklink_output (Rust)
- Shim: shim/shim.cpp (C++ DeckLink SDK bridge)
- Setup sequence (real-time scheduled playback best practice):
  - Get frame timing: decklink_output_get_frame_timing → (frameDuration, timeScale)
  - Set SDI output connection (set_sdi_output)
  - Enable internal keyer (decklink_keyer_enable_internal), set keyer level=255 (fully visible)
  - Pre-roll 2 frames: ScheduleVideoFrame for frames 0..N-1
  - Start scheduled playback at hardware reference time (GetHardwareReferenceClock)
- Per-frame scheduling (async, callback-driven):
  - Build overlay RawFramePacket in BGRA8 (key) from the GPU overlay buffer
  - OutputRequest { video: Some(fill UYVY), overlay: Some(key BGRA) }
  - schedule_frame(...):
    - Preferred: DVP DMA path (decklink_output_schedule_frame_gpu_dvp)
    - Fallback: cudaMemcpy2D GPU→CPU into DeckLink video frame (decklink_output_schedule_frame_gpu)
  - Queue depth and backpressure: GetBufferedVideoFrameCount guides when to yield; adaptive queue adjusts depth between 2–5
- Fill vs Key:
  - Internal keying uses the device’s internal keyer to alpha blend the BGRA key over the live fill signal (the video program path on DeckLink). No GPU compositing on our side.
- Files: crates/decklink_output/src/device.rs, shim/shim.cpp (decklink_output_open, decklink_output_schedule_frame_gpu[_dvp], decklink_keyer_*).

Memory Flow Summary
- Capture
  - Preferred: MemLoc::Gpu (DVP-enabled capture path); Otherwise MemLoc::Cpu
  - If CPU, we allocate GPU memory (cudarc::driver::CudaDevice.htod_sync_copy) and copy UYVY once per frame
- Preprocessing (GPU)
  - Output tensor allocated on GPU (buffer_pool caches/leaks CUdeviceptr for reuse)
- Inference (GPU → CPU result copy)
  - Input GPU pointer passed directly to TensorRT (run_inference_device)
  - Output lives in TensorRT GPU buffer; then copied to CPU Vec<f32> for postprocess
- Postprocess + Plan (CPU)
  - No GPU memory
- Overlay Rendering (GPU)
  - cudaMalloc persistent BGRA framebuffer; kernels draw directly; stays on GPU
- Output Scheduling (GPU → DeckLink)
  - Preferred: DVP DMA to DeckLink frame (zero-copy from GPU perspective)
  - Fallback: cudaMemcpy2D GPU→CPU into DeckLink-allocated frame buffer, then ScheduleVideoFrame

Libraries and Shims Used
- DeckLink capture/output: shim/shim.cpp (C++), exposes C ABI for Rust
  - DeckLink SDK v14.4 APIs (EnableVideoOutput, ScheduleVideoFrame, GetHardwareReferenceClock, GetBufferedVideoFrameCount)
  - Internal keying control via IDeckLinkKeyer (Enable(false) for internal, SetLevel)
  - DVP integration for zero-copy DMA paths and custom IDeckLinkMemoryAllocator
- CUDA in Rust: cudarc::driver (device, stream, alloc, kernel launch)
- CUDA kernels: preprocess.cu (PTX) and overlay_render.cu (BGRA drawing)
- TensorRT: trt-shim (C++/CUDA) wrapped by crates/inference_v2 (Rust)
  - libtrt_shim.so exports create_session, run_inference_device, get_device_buffers, copy_output_to_cpu, destroy_session

Scheduling, Timing, and Latency
- Hardware reference clock
  - StartScheduledPlayback at hardware time; per-frame display_time computed as now + frames_ahead*frameDuration
- Pre-roll
  - 2–3 frames scheduled before starting playback to lock timing
- Backpressure
  - Poll GetBufferedVideoFrameCount; yield when queue full; no sleeps in steady state
- Adaptive queue
  - Smooth moving average (SMA) of pipeline time adjusts target queue depth (2–5 frames) to balance latency vs smoothness
- Latency accounting (per frame and cumulative)
  - Capture latency: hardware-measured using device timestamp
  - Stage timings for preprocess, inference, postprocess, plan, render, keying packet/schedule calls, and end-to-end

Answering “are we writing shim or what it use?”
- We use existing shims in this repo:
  - DeckLink shim: shim/shim.cpp (C++), compiled and loaded by decklink_input/decklink_output crates via FFI
  - TensorRT shim: trt-shim/src/trt_shim.cpp (C++/CUDA), built into libtrt_shim.so and dlopen’d by crates/inference_v2
  - CUDA rendering/preprocess: implemented as CUDA C kernels (overlay_render.cu, preprocess.cu) invoked from Rust
- In this pipeline we do not author a new shim; we consume these shims/libraries.

Key File References
- Pipeline driver: apps/playgrounds/src/bin/pipeline_capture_to_output_v5_keying.rs:220, 252, 291, 300, 310, 316, 347, 355, 359, 408, 539, 562, 573, 596, 613, 640, 700, 811
- Capture crate: crates/decklink_input/src/capture.rs: types, MemLoc, hardware timestamping, GPU/CPU selection
- Output crate: crates/decklink_output/src/device.rs: schedule_frame, keyer control, frame timing, hardware clock, queue depth
- Overlay render: crates/overlay_render/src/lib.rs, crates/overlay_render/overlay_render.cu
- Preprocess: crates/preprocess_cuda/src/lib.rs, crates/preprocess_cuda/build.rs
- Inference: crates/inference_v2/src/lib.rs, trt-shim/src/trt_shim.cpp
- Overlay plan: crates/overlay_plan/src/lib.rs
- DeckLink shim internals: shim/shim.cpp (DVP allocator, scheduled callback, keyer)

Operational Notes
- Engine/library paths must exist:
  - TensorRT engine: configs/model/v7_optimized_YOLOv5.engine
  - TRT shim library: trt-shim/build/libtrt_shim.so
- Keying mode relies on device internal keyer: you send only the BGRA key frame; the device’s fill is the live SDI program path
- For true zero-copy GPU→DeckLink, ensure DVP is available and use schedule_frame_dvp() path
- All GPU buffers stay device-resident through preprocess, inference, and render; the only device↔host copies are:
  - Capture CPU→GPU (only if capture delivered CPU memory)
  - Inference GPU→CPU for model output (for CPU postprocess)
  - Output scheduling GPU→CPU (if DVP not active), otherwise DVP DMA

End
