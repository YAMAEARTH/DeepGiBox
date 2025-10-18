# DeepGIBox Project Structure

## 📁 Project Overview

```
DeepGiBox/
├── 📂 apps/                          # Application binaries
│   ├── 📂 playgrounds/               # Testing and demonstration programs
│   │   ├── 📂 src/bin/              # Binary programs (18 programs)
│   │   │   ├── capture_preview.rs             # Preview capture
│   │   │   ├── capture_yuv_dump.rs            # Dump YUV data
│   │   │   ├── decklink_latency_test.rs       # Latency measurement v1
│   │   │   ├── decklink_latency_test2.rs      # Latency measurement v2
│   │   │   ├── demo_rawframepacket.rs         # RawFramePacket demo
│   │   │   ├── explain_data_flow.rs           # Data flow explanation
│   │   │   ├── explore_rawframepacket.rs      # Explore packet structure
│   │   │   ├── full_pipeline_latency.rs       # Complete pipeline latency analysis
│   │   │   ├── inference_preview.rs           # Inference preview
│   │   │   ├── latency_test.rs                # Basic latency test
│   │   │   ├── list_decklink_devices.rs       # List DeckLink devices
│   │   │   ├── memory_location_explained.rs   # Memory architecture explanation
│   │   │   ├── overlay_preview.rs             # Overlay rendering preview
│   │   │   ├── preprocessing_1080p60_test.rs  # Preprocessing test
│   │   │   ├── prove_raw_yuv422.rs            # Prove YUV422 format
│   │   │   ├── real_capture_latency.rs        # Real capture latency measurement
│   │   │   ├── telemetry_demo.rs              # Telemetry demonstration
│   │   │   └── test_rawframepacket.rs         # Test RawFramePacket
│   │   └── Cargo.toml
│   └── 📂 runner/                    # Main application runner
│       ├── src/main.rs
│       └── Cargo.toml
│
├── 📂 crates/                        # Library crates (pipeline stages)
│   ├── 📦 common_io/                # Common I/O types and traits
│   │   ├── src/lib.rs              # PixelFormat, ColorSpace, MemLoc, MemRef,
│   │   │                           # FrameMeta, RawFramePacket, TensorInputPacket,
│   │   │                           # DetectionsPacket, OverlayPlanPacket, Stage trait
│   │   └── Cargo.toml
│   │
│   ├── 📦 config/                   # Configuration management
│   │   ├── src/lib.rs
│   │   └── Cargo.toml
│   │
│   ├── 📦 decklink_input/           # DeckLink video capture
│   │   ├── src/
│   │   │   ├── lib.rs              # Public API, re-exports
│   │   │   └── capture.rs          # CaptureSession, FFI to C++ shim
│   │   ├── build.rs                # Build script (compiles shim.cpp)
│   │   └── Cargo.toml
│   │
│   ├── 📦 decklink_output/          # DeckLink video output
│   │   ├── src/lib.rs
│   │   └── Cargo.toml
│   │
│   ├── 📦 preprocess_cuda/          # GPU preprocessing (YUV→RGB, resize, normalize)
│   │   ├── src/lib.rs
│   │   └── Cargo.toml
│   │
│   ├── 📦 inference/                # TensorRT/ONNX inference
│   │   ├── src/lib.rs
│   │   └── Cargo.toml
│   │
│   ├── 📦 postprocess/              # Post-processing (NMS, tracking)
│   │   ├── src/lib.rs
│   │   └── Cargo.toml
│   │
│   ├── 📦 overlay_plan/             # Overlay planning
│   │   ├── src/lib.rs
│   │   └── Cargo.toml
│   │
│   ├── 📦 overlay_render/           # Overlay rendering
│   │   ├── src/lib.rs
│   │   └── Cargo.toml
│   │
│   └── 📦 telemetry/                # Telemetry and latency measurement
│       ├── src/
│       │   ├── lib.rs              # Main API (record_ms, time_stage)
│       │   ├── time.rs             # Time utilities (now_ns, since_ms)
│       │   ├── log.rs              # Human-readable log backend
│       │   └── json.rs             # JSON log backend
│       ├── Cargo.toml              # Features: human-log, json, stats, cuda-events
│       ├── README.md               # Comprehensive documentation
│       └── QUICKREF.md             # Quick reference
│
├── 📂 include/                      # DeckLink SDK C++ headers
│   ├── DeckLinkAPI*.h              # DeckLink API headers (multiple versions)
│   ├── DeckLinkAPIDispatch.cpp     # DeckLink dispatch implementation
│   └── LinuxCOM.h                  # Linux COM interface
│
├── 📂 shim/                         # C++ to Rust FFI bridge
│   └── shim.cpp                    # C++ wrapper for DeckLink SDK
│
├── 📂 configs/                      # Configuration files
│   ├── dev_1080p60_yuv422_fp16_trt.toml  # 1080p60 config
│   └── dev_4k30_yuv422_fp16_trt.toml     # 4K30 config
│
├── 📂 testsupport/                  # Test support utilities
│   ├── src/lib.rs
│   └── Cargo.toml
│
├── 📄 Cargo.toml                    # Workspace root
├── 📄 Cargo.lock                    # Dependency lock file
│
└── 📚 Documentation/
    ├── README.md                            # Project README
    ├── INSTRUCTION.md                       # Pipeline specification
    ├── CHANGES_DECKLINK_INPUT.md            # DeckLink implementation changes
    ├── DATA_FLOW_AND_LATENCY_EXPLAINED.md   # Data flow explanation
    ├── GPUDIRECT_RDMA_GUIDE.md              # GPUDirect RDMA guide
    ├── HOW_TO_MEASURE_REAL_CAPTURE_LATENCY.md  # Latency measurement guide
    ├── TELEMETRY_IMPLEMENTATION.md          # Telemetry implementation details
    ├── telemetry_guideline.md               # Telemetry guidelines
    └── TEST_RAWFRAMEPACKET_RESULTS.md       # Test results
```

## 🔄 Pipeline Flow

```
┌────────────────────┐
│  DeckLink Input    │  crates/decklink_input
│  (Video Capture)   │  → RawFramePacket (YUV422, System RAM)
└──────────┬─────────┘
           ↓
┌────────────────────┐
│  Preprocess CUDA   │  crates/preprocess_cuda
│  (GPU Pipeline)    │  → TensorInputPacket (RGB, GPU VRAM)
└──────────┬─────────┘  • H2D transfer (System RAM → GPU VRAM)
           ↓            • YUV422 → RGB conversion
┌────────────────────┐  • Resize to model input size
│  Inference         │  crates/inference
│  (TensorRT/ONNX)   │  → DetectionsPacket (bounding boxes)
└──────────┬─────────┘
           ↓
┌────────────────────┐
│  Postprocess       │  crates/postprocess
│  (NMS, Tracking)   │  → DetectionsPacket (filtered, tracked)
└──────────┬─────────┘
           ↓
┌────────────────────┐
│  Overlay Plan      │  crates/overlay_plan
│  (Generate Ops)    │  → OverlayPlanPacket (draw operations)
└──────────┬─────────┘
           ↓
┌────────────────────┐
│  Overlay Render    │  crates/overlay_render
│  (Draw Overlays)   │  → OverlayFramePacket (ARGB frame)
└──────────┬─────────┘
           ↓
┌────────────────────┐
│  DeckLink Output   │  crates/decklink_output
│  (Video Output)    │  → Display on monitor
└────────────────────┘

     ⏱️  Telemetry measures latency at each stage
```

## 📊 Data Structures (common_io)

### Core Types

```rust
// Memory location
enum MemLoc {
    Cpu,                    // System RAM
    Gpu { device: u32 }     // GPU VRAM
}

// Memory reference
struct MemRef {
    ptr: *mut u8,          // Pointer to data
    len: usize,            // Total size in bytes
    stride: usize,         // Stride per row
    loc: MemLoc,           // Where the data is
}

// Frame metadata
struct FrameMeta {
    source_id: u32,        // Input source ID
    width: u32,            // Frame width
    height: u32,           // Frame height
    pixfmt: PixelFormat,   // YUV422_8, BGRA8, etc.
    colorspace: ColorSpace,// BT601, BT709, BT2020
    frame_idx: u64,        // Frame sequence number
    pts_ns: u64,           // Presentation timestamp
    t_capture_ns: u64,     // Capture timestamp
    stride_bytes: u32,     // Bytes per row
}
```

### Packet Types

```rust
// Raw frame from capture
struct RawFramePacket {
    meta: FrameMeta,       // Metadata
    data: MemRef,          // Raw pixel data (YUV422)
}

// Preprocessed tensor input
struct TensorInputPacket {
    from: FrameMeta,       // Original frame metadata
    desc: TensorDesc,      // Tensor dimensions (NCHW)
    data: MemRef,          // Tensor data (RGB, normalized)
}

// Detection results
struct DetectionsPacket {
    from: FrameMeta,       // Source frame metadata
    items: Vec<Detection>, // Detected objects
}

// Overlay operations
struct OverlayPlanPacket {
    from: FrameMeta,       // Source frame metadata
    ops: Vec<DrawOp>,      // Draw operations (rect, label, poly)
    canvas: (u32, u32),    // Canvas dimensions
}
```

## 🛠️ Key Components

### DeckLink Input (`crates/decklink_input`)

**Files:**
- `src/lib.rs`: Public API
- `src/capture.rs`: CaptureSession implementation
- `build.rs`: Compiles C++ shim
- `../shim/shim.cpp`: C++ FFI bridge to DeckLink SDK

**Functionality:**
- Opens DeckLink device
- Captures video frames (1080p60, 4K30)
- Returns `RawFramePacket` with YUV422 data in System RAM
- Metadata includes: width, height, pixel format (YUV422_8), colorspace (BT709)

### Telemetry (`crates/telemetry`)

**Features:**
- `human-log` (default): Human-readable output to stderr
- `json`: JSON structured output
- `stats`: Rolling statistics (planned)
- `cuda-events`: GPU timing (planned)

**API:**
```rust
// Record measurement
record_ms(name: &str, start_ns: u64);

// Time a stage
time_stage<I, O, S: Stage<I, O>>(
    name: &str, 
    stage: &mut S, 
    input: I
) -> O;

// Get current time
now_ns() -> u64;

// Convert to milliseconds
since_ms(start_ns: u64) -> f64;
```

### Preprocessing CUDA (`crates/preprocess_cuda`)

**Planned functionality:**
- H2D transfer: System RAM → GPU VRAM (~0.3-0.4ms)
- YUV422 → RGB conversion (CUDA kernel)
- Resize to model input size (e.g., 640×640)
- Normalize pixel values (0-255 → 0.0-1.0)
- Output: `TensorInputPacket` in GPU VRAM

## 🧪 Testing Programs

### Capture & Validation
- `capture_preview.rs` - Quick capture preview
- `capture_yuv_dump.rs` - Dump raw YUV data to file
- `test_rawframepacket.rs` - Test RawFramePacket structure
- `prove_raw_yuv422.rs` - Prove DeckLink captures YUV422 format

### Latency Measurement
- `latency_test.rs` - Basic latency test
- `decklink_latency_test.rs` - DeckLink-specific latency
- `real_capture_latency.rs` - Real capture latency measurement
- `full_pipeline_latency.rs` - Complete pipeline analysis

### Educational
- `explain_data_flow.rs` - Explain data flow from hardware to app
- `memory_location_explained.rs` - Memory architecture explanation
- `telemetry_demo.rs` - Telemetry usage demonstration

## 📈 Performance Targets (60fps = 16.67ms budget)

| Stage | Target Latency | Status |
|-------|----------------|--------|
| Hardware capture | 16-33ms (fixed) | ⚠️ Unmeasurable currently |
| get_frame() | <0.001ms | ✅ Measured: 0.0003ms |
| Memory read | <0.1ms | ✅ Measured: 0.02ms |
| H2D transfer | 0.3-0.4ms | ⏳ Not implemented |
| GPU preprocess | 0.5-2ms | ⏳ Not implemented |
| Inference | 2-8ms | ⏳ Not implemented |
| Postprocess | 0.5-1ms | ⏳ Not implemented |
| Overlay | 1-3ms | ⏳ Not implemented |
| **Total (measurable)** | **~5-15ms** | **Target** |

## 🔧 Build & Run

```bash
# Build entire workspace
cargo build --release

# Run specific binary
cargo run -p playgrounds --bin real_capture_latency

# Build with specific features
cargo build -p telemetry --features json

# Run runner application
cargo run -p runner --release
```

## 📝 Documentation Files

### Implementation Guides
- **INSTRUCTION.md** - Original pipeline specification
- **CHANGES_DECKLINK_INPUT.md** - How DeckLink was adapted
- **TELEMETRY_IMPLEMENTATION.md** - Telemetry implementation details

### Technical Explanations
- **DATA_FLOW_AND_LATENCY_EXPLAINED.md** - Data flow from hardware to app
- **HOW_TO_MEASURE_REAL_CAPTURE_LATENCY.md** - Latency measurement methods
- **GPUDIRECT_RDMA_GUIDE.md** - GPUDirect RDMA setup and usage

### Test Results
- **TEST_RAWFRAMEPACKET_RESULTS.md** - RawFramePacket test results

### Guidelines
- **telemetry_guideline.md** - Telemetry usage guidelines
- **README.md** - Project overview

## 🎯 Current Status

### ✅ Completed
- Common I/O types and traits
- DeckLink input capture (YUV422, BT709)
- Telemetry framework (per-stage timing)
- Test programs and benchmarks
- Documentation

### ⏳ In Progress
- GPU preprocessing (YUV→RGB conversion)
- H2D transfer optimization

### 📋 Planned
- Inference engine integration
- Postprocessing (NMS, tracking)
- Overlay rendering
- DeckLink output
- GPUDirect RDMA support
- CUDA events for GPU timing
- Rolling statistics

## 🔗 Dependencies

- **Rust**: Edition 2021
- **CUDA**: For GPU preprocessing
- **DeckLink SDK**: For video capture/output
- **TensorRT/ONNX**: For inference (planned)

## 📊 Memory Architecture

```
┌─────────────────────────────────────────────────────────┐
│ System RAM (DDR4/DDR5)                                  │
│                                                          │
│  ┌────────────────────────────────────────────┐        │
│  │ DeckLink DMA Buffer                         │        │
│  │ • 4,147,200 bytes (1920×1080×2)            │        │
│  │ • YUV422 (UYVY) format                     │        │
│  │ • Address: e.g., 0x70bc46607010            │        │
│  │ • MemLoc::Cpu                               │        │
│  └────────────────────────────────────────────┘        │
│                                                          │
└─────────────────────────────────────────────────────────┘
                      ↓ H2D Transfer (~0.3-0.4ms)
┌─────────────────────────────────────────────────────────┐
│ GPU VRAM (GDDR6/GDDR6X)                                 │
│                                                          │
│  ┌────────────────────────────────────────────┐        │
│  │ Preprocessed Tensor                         │        │
│  │ • RGB format (3 channels)                   │        │
│  │ • Resized (e.g., 640×640)                  │        │
│  │ • Normalized (0.0-1.0)                      │        │
│  │ • MemLoc::Gpu { device: 0 }                │        │
│  └────────────────────────────────────────────┘        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

**Project**: DeepGIBox - Real-time Video Processing Pipeline  
**Status**: Active Development  
**License**: (TBD)
