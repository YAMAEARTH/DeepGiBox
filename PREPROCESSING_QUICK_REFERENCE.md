# Preprocessing Quick Reference

## การใช้งานพื้นฐาน

### 1. สร้าง Preprocessor จาก Config

```rust
use preprocess_cuda::Preprocessor;

// แบบที่ 1: อ่านจาก TOML config
let mut preprocessor = preprocess_cuda::from_path(
    "configs/dev_1080p60_yuv422_fp16_trt.toml"
)?;

// แบบที่ 2: สร้างโดยตรง
let mut preprocessor = Preprocessor::new(
    (512, 512),  // output size
    true,        // fp16
    0            // device
)?;

// แบบที่ 3: สร้างแบบ full parameters
use preprocess_cuda::ChromaOrder;
let mut preprocessor = Preprocessor::with_params(
    (512, 512),                   // output size
    true,                         // fp16
    0,                            // device
    [0.485, 0.456, 0.406],       // mean (ImageNet)
    [0.229, 0.224, 0.225],       // std (ImageNet)
    ChromaOrder::UYVY             // chroma order
)?;
```

### 2. Process Frame

```rust
use common_io::Stage;

let raw_frame: RawFramePacket = ...; // จาก DeckLink capture (GPU)
let tensor_input = preprocessor.process(raw_frame);

// Output:
// - tensor_input.desc: { n:1, c:3, h:512, w:512, dtype:Fp16 }
// - tensor_input.data: NCHW layout on GPU
```

## Config Format (TOML)

```toml
[preprocess]
size = [512, 512]         # Output size (W, H)
fp16 = true               # Use FP16 (true) or FP32 (false)
device = 0                # CUDA device ID
chroma = "UYVY"           # "UYVY" or "YUY2"
mean = [0.0, 0.0, 0.0]   # RGB mean
std = [1.0, 1.0, 1.0]    # RGB std
```

### Normalization Presets

**No normalization (0-1 only):**
```toml
mean = [0.0, 0.0, 0.0]
std = [1.0, 1.0, 1.0]
```

**ImageNet:**
```toml
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

## Pixel Format Support

| Format | Description | Stride Formula |
|--------|-------------|----------------|
| YUV422_8 | 4:2:2 interleaved (UYVY/YUY2) | `width * 2` |
| NV12 | 4:2:0 planar Y + UV | `width` |
| BGRA8 | 8-bit interleaved | `width * 4` |

## Chroma Orders (YUV422_8 only)

**UYVY (default):**
```
Bytes: U0 Y0 V0 Y1 | U2 Y2 V2 Y3 | ...
```

**YUY2:**
```
Bytes: Y0 U0 Y1 V0 | Y2 U2 Y3 V2 | ...
```

## Common Issues

### 1. Wrong chroma order → Incorrect colors
```rust
// แก้: เปลี่ยน chroma order ใน config
chroma = "YUY2"  // แทน "UYVY"
```

### 2. Panic: "Expected GPU input"
```rust
// Input frame ต้องอยู่บน GPU (MemLoc::Gpu)
// ตรวจสอบว่า decklink_input ส่ง GPU pointer มา
```

### 3. Panic: "Invalid stride"
```rust
// stride_bytes ต้อง >= expected minimum:
// YUV422_8: width * 2
// NV12: width
// BGRA8: width * 4
```

### 4. Build error: "nvcc not found"
```bash
# ติดตั้ง CUDA toolkit หรือ set path
export CUDA_PATH=/usr/local/cuda
export PATH=$PATH:/usr/local/cuda/bin
```

## Testing

```bash
# Check compilation (skip if no CUDA)
cargo check -p preprocess_cuda

# Run basic tests
cargo test -p preprocess_cuda

# Run GPU tests (requires CUDA)
cargo test -p preprocess_cuda -- --ignored

# Preview/demo
cargo run -p playgrounds --bin preprocess_gpu_preview
```

## Performance Tips

1. **Reuse preprocessor**: Create once, process many frames
2. **Match input format**: YUV422_8 is fastest for DeckLink
3. **Use FP16**: ~2x faster than FP32 for inference
4. **Keep data on GPU**: Avoid D2H/H2D copies
5. **Buffer pooling**: Already handled automatically

## API Reference

```rust
// Preprocessor struct
pub struct Preprocessor {
    pub size: (u32, u32),    // Output (W, H)
    pub fp16: bool,          // Data type
    pub device: u32,         // GPU ID
    pub mean: [f32; 3],      // RGB mean
    pub std: [f32; 3],       // RGB std
    pub chroma: ChromaOrder, // UYVY or YUY2
}

// Chroma order enum
pub enum ChromaOrder {
    UYVY = 0,
    YUY2 = 1,
}

// Construction
impl Preprocessor {
    pub fn new(size: (u32, u32), fp16: bool, device: u32) -> Result<Self>;
    pub fn with_params(...) -> Result<Self>;
}

// Usage (implements Stage trait)
impl Stage<RawFramePacket, TensorInputPacket> for Preprocessor {
    fn process(&mut self, input: RawFramePacket) -> TensorInputPacket;
}

// Config loading
pub fn from_path(cfg_path: &str) -> Result<Preprocessor>;
```

## ตัวอย่างการใช้ใน Pipeline

```rust
// runner/src/main.rs
use preprocess_cuda::Preprocessor;
use common_io::Stage;

fn main() -> anyhow::Result<()> {
    // 1. Create stages
    let mut capture = decklink_input::from_path("configs/dev.toml")?;
    let mut preprocess = preprocess_cuda::from_path("configs/dev.toml")?;
    let mut inference = inference::from_path("configs/dev.toml")?;
    
    // 2. Pipeline loop
    loop {
        // Capture (output on GPU)
        let raw_frame = capture.capture_frame()?;
        
        // Preprocess (input/output on GPU)
        let tensor = preprocess.process(raw_frame);
        
        // Inference (input on GPU)
        let detections = inference.process(tensor);
        
        // ... continue pipeline
    }
}
```

## Telemetry Integration

```rust
use telemetry::time_stage;

// Automatic timing
let tensor = time_stage("preprocess", &mut preprocessor, raw_frame);

// Log format: [lat] preprocess=2.1ms
```

## ข้อมูลเพิ่มเติม

- Full spec: [preprocessing_guideline.md](preprocessing_guideline.md)
- Implementation: [PREPROCESSING_IMPLEMENTATION_SUMMARY.md](PREPROCESSING_IMPLEMENTATION_SUMMARY.md)
- Module README: [crates/preprocess_cuda/README.md](crates/preprocess_cuda/README.md)
