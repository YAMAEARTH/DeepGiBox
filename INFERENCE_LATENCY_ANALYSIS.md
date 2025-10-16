# 🔍 Inference Latency Analysis

## 📊 Current Performance Breakdown

จากผลการทดสอบ 10 frames:

```
╔═══════════════════════════════════════════════════════════╗
║ 🧠 Inference Stage Breakdown (Average):
║   Total Inference Time: 123.0 ms (100%)
║   
║   1. Tensor Staging (GPU→CPU Copy): 34.5 ms (28%)  ⚠️ BOTTLENECK
║   2. IO Binding Setup:              0.02 ms (0.02%)
║   3. TensorRT Execution:            87.0 ms (71%)
║   4. Output Extraction:             0.05 ms (0.04%)
╚═══════════════════════════════════════════════════════════╝
```

## 🔴 Major Bottlenecks

### 1. **Tensor Staging (34.5ms - 28% of total time)**
**ปัญหา**: GPU FP16 → CPU FP32 conversion
- Copy 786,432 FP16 values (1.5 MB) from GPU to CPU
- Convert u16 → half::f16 → f32 (element-by-element)
- Synchronous blocking operation

**สาเหตุ**:
```rust
// Current implementation (SLOW):
let mut cpu_fp16_raw = vec![0u16; total_elements];
cudarc::driver::result::memcpy_dtoh_sync(&mut cpu_fp16_raw, gpu_ptr)?;
for (i, &raw_val) in cpu_fp16_raw.iter().enumerate() {
    staging_data[i] = half::f16::from_bits(raw_val).to_f32();
}
```

### 2. **TensorRT Execution (87ms - 71% of total time)**
**ปัญหา**: Model complexity
- Input: 1×3×512×512 (FP16) = 786,432 elements
- Output: 1×16,128×7 = 112,896 elements
- 16,128 anchor boxes (high resolution detection grid)

---

## 🏗️ Inference Engine Architecture

### **Data Flow Pipeline**:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INPUT: TensorInputPacket (FP16 on GPU)                  │
│    - Shape: [1, 3, 512, 512]                               │
│    - Type: FP16                                            │
│    - Location: GPU Memory (Zero-copy from preprocessing)   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. TENSOR STAGING (34.5ms) ⚠️                              │
│    Why needed? ORT + TensorRT expects CPU-accessible input │
│                                                             │
│    Steps:                                                   │
│    a) GPU FP16 → CPU u16 buffer (CUDA memcpy)             │
│    b) Convert u16 → half::f16 → f32 (CPU loop)            │
│    c) Create ORT tensor from FP32 CPU buffer               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. IO BINDING SETUP (0.02ms) ✅                            │
│    - Bind input tensor "images"                            │
│    - Bind output to CPU memory                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. TENSORRT EXECUTION (87ms)                               │
│    What happens inside TensorRT:                           │
│    a) Copy input from CPU → GPU (internal)                 │
│    b) Run optimized CUDA kernels                           │
│    c) Copy output from GPU → CPU (internal)                │
│                                                             │
│    Model Architecture (detected):                          │
│    - Input: [1, 3, 512, 512] FP16                         │
│    - Backbone: Feature extraction                          │
│    - Neck: Feature pyramid / PAN                           │
│    - Head: Detection head                                  │
│    - Output: [1, 16128, 7] FP32                           │
│      ├─ 16,128 anchor boxes                               │
│      └─ 7 features: [x, y, w, h, conf, cls_id, ?]        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. OUTPUT EXTRACTION (0.05ms) ✅                           │
│    - Extract tensor from ORT session                       │
│    - Convert to Vec<f32>                                   │
│    - Extract output shape                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. OUTPUT: RawDetectionsPacket                             │
│    - raw_output: Vec<f32> (112,896 values)                │
│    - output_shape: [1, 16128, 7]                          │
│    - from: Original frame metadata                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤔 Why This Design?

### **Problem: ORT + TensorRT Limitation**
Current ONNX Runtime with TensorRT EP has limitations:
1. **Cannot directly consume GPU tensors** from external CUDA code
2. **Requires CPU-accessible input** even though TensorRT runs on GPU
3. **Internal GPU transfers** happen anyway (CPU→GPU→CPU)

### **Current Flow (Inefficient)**:
```
Preprocessing (GPU) → GPU FP16
                         ↓ (34ms - WASTE!)
                      CPU FP32
                         ↓ (internal - hidden)
                      GPU FP32 (TensorRT)
                         ↓ (87ms - actual inference)
                      GPU Output
                         ↓ (internal - hidden)
                      CPU Output
```

### **Ideal Flow (Not currently possible with ORT)**:
```
Preprocessing (GPU) → GPU FP16
                         ↓ (0ms - zero copy)
                      GPU FP32 (TensorRT) 
                         ↓ (87ms)
                      GPU Output
                         ↓ (only if needed)
                      CPU Output
```

---

## 💡 Why Latency is High

### **1. Redundant GPU↔CPU Transfers**
- **First transfer**: GPU FP16 → CPU FP32 (34ms) - *our code*
- **Second transfer**: CPU FP32 → GPU FP32 (hidden) - *TensorRT internal*
- **Third transfer**: GPU output → CPU output (hidden) - *TensorRT internal*

### **2. FP16→FP32 Conversion Overhead**
```rust
// Current: Element-by-element conversion (CPU)
for (i, &raw_val) in cpu_fp16_raw.iter().enumerate() {
    staging_data[i] = half::f16::from_bits(raw_val).to_f32();
}
```
- 786,432 iterations
- No vectorization (SIMD)
- No GPU acceleration

### **3. Model Complexity**
- **16,128 anchors** = high-resolution detection grid
- YOLOv5-640 typically has ~25,200 anchors
- This model has fewer anchors but custom architecture
- 512×512 input still requires significant computation

### **4. TensorRT Engine Not Fully Optimized**
- Using FP16 precision (good!)
- But model might not be fully optimized for target GPU
- First run builds TensorRT engine (cached for later)

---

## 🚀 Optimization Opportunities

### **Priority 1: Eliminate GPU→CPU Transfer (Target: -30ms)**

#### Option A: Use TensorRT C++ API directly
```cpp
// Native TensorRT approach (C++)
cudaStream_t stream;
cudaStreamCreate(&stream);

// Direct GPU input binding
trtContext->setTensorAddress("images", gpu_fp16_ptr);
trtContext->enqueueV3(stream);

// Output stays on GPU for postprocessing
```

#### Option B: Custom ORT Execution Provider
Create custom EP that accepts GPU tensors directly

#### Option C: Async GPU→CPU transfer (easiest)
```rust
// Use async CUDA streams
let stream = device.fork_default_stream()?;
device.htod_copy_into_async(cpu_data, &mut gpu_buffer, &stream)?;
// Continue with other work...
stream.synchronize()?;
```

### **Priority 2: Vectorized FP16→FP32 Conversion (Target: -5ms)**

Use SIMD or GPU kernel:
```rust
// Option 1: SIMD (x86_64)
use std::simd::{f16x8, f32x8};
// Batch convert 8 values at once

// Option 2: CUDA kernel
__global__ void fp16_to_fp32(half* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = __half2float(input[idx]);
}
```

### **Priority 3: Model Optimization (Target: -20ms)**

1. **Reduce input resolution**: 512×512 → 416×416 or 384×384
2. **Use INT8 quantization** (if acceptable accuracy loss)
3. **Optimize anchor grid**: Reduce from 16,128 to ~8,400
4. **Layer fusion**: Ensure TensorRT fully fuses layers

### **Priority 4: Async Pipeline (Target: Hide latency)**

```rust
// Pipelined execution
Frame N:   [Capture] [Preprocess] [Inference] [Postprocess]
Frame N+1:          [Capture] [Preprocess] [Inference] [Postprocess]
Frame N+2:                   [Capture] [Preprocess] [Inference]

// Throughput: ~8 FPS → ~24 FPS (3x improvement)
```

---

## 📈 Estimated Performance After Optimization

### **Current: 133ms total (7.5 FPS)**
```
Preprocessing:  0.14ms (GPU Direct)
Inference:    123.00ms
  ├─ Staging:   34.50ms ⚠️
  ├─ Binding:    0.02ms
  ├─ TensorRT:  87.00ms
  └─ Extract:    0.05ms
```

### **After GPU-Direct Inference: 98ms (10.2 FPS)**
```
Preprocessing:  0.14ms (GPU Direct)
Inference:     88.00ms (-35ms!)
  ├─ Staging:    0.00ms ✅ ELIMINATED
  ├─ Binding:    0.50ms (GPU binding overhead)
  ├─ TensorRT:  87.00ms
  └─ Extract:    0.50ms (GPU→CPU only if needed)
```

### **After Model Optimization: 68ms (14.7 FPS)**
```
Preprocessing:  0.14ms (GPU Direct)
Inference:     58.00ms (-30ms from TensorRT!)
  ├─ Binding:    0.50ms
  ├─ TensorRT:  57.00ms (smaller model/quantization)
  └─ Extract:    0.50ms
```

### **After Async Pipeline: 15ms effective (66 FPS)**
```
Parallel processing of 3 frames simultaneously
Effective latency per frame: 68ms / 3 = ~15ms
```

---

## 🎯 Recommended Next Steps

### **Immediate (Easy wins)**:
1. ✅ Add output shape to RawDetectionsPacket (DONE!)
2. 🔨 Profile actual GPU transfer time separately
3. 🔨 Try async CUDA streams for GPU→CPU copy

### **Short-term (Moderate effort)**:
4. 🔨 Investigate TensorRT C++ API integration
5. 🔨 Test with smaller input resolution (384×384)
6. 🔨 Benchmark INT8 quantization

### **Long-term (High impact)**:
7. 🔨 Implement fully GPU-resident pipeline
8. 🔨 Custom postprocessing on GPU (NMS, bbox decode)
9. 🔨 Multi-stream async processing

---

## 📚 Technical Details

### **Memory Layout**:
```
Input Tensor (FP16):
  Shape:  [1, 3, 512, 512]
  Size:   1 × 3 × 512 × 512 × 2 bytes = 1,572,864 bytes (1.5 MB)
  Layout: NCHW (batch, channels, height, width)
  
Output Tensor (FP32):
  Shape:  [1, 16128, 7]
  Size:   1 × 16128 × 7 × 4 bytes = 451,584 bytes (441 KB)
  Format: [x_center?, y_center?, width?, height?, confidence, class_id, ?]
```

### **CUDA Memory Copy Performance**:
```
PCIe 3.0 x16: ~12 GB/s theoretical
Actual: ~8-10 GB/s practical
1.5 MB transfer: ~0.15-0.19 ms (theoretical minimum)
Observed: 34ms (180x slower!) ⚠️

Why so slow?
- Synchronous blocking call
- CPU-side conversion overhead
- Small transfers (not saturating bandwidth)
- Kernel launch overhead
```

---

## 🔬 Debug Commands

```bash
# Profile CUDA operations
nsys profile --trace=cuda,nvtx cargo run --release --bin test_inference_pipeline

# Check TensorRT engine optimization
trtexec --onnx=YOLOv5.onnx --saveEngine=optimized.trt --fp16 --verbose

# Monitor GPU utilization
nvidia-smi dmon -s u

# Measure pure TensorRT performance
trtexec --loadEngine=optimized.trt --warmUp=100 --iterations=100
```

---

## ✅ Conclusion

**Current bottleneck**: 34.5ms (28%) spent on unnecessary GPU→CPU transfer and FP16→FP32 conversion

**Root cause**: ONNX Runtime + TensorRT limitation requiring CPU-accessible input despite GPU execution

**Best solution**: Direct TensorRT C++ integration for fully GPU-resident inference

**Quick win**: Async CUDA streams + vectorized conversion could save ~10-15ms

**Target achievable**: 60-80ms inference (12-16 FPS) with moderate optimization
