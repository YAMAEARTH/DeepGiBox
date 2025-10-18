# GPUDirect RDMA Integration Guide for DeckLink

## Overview

GPUDirect RDMA allows DeckLink card to write captured video frames directly to GPU VRAM, bypassing System RAM and CPU overhead.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        NORMAL PATH (Current)                              │
└──────────────────────────────────────────────────────────────────────────┘

SDI Input → DeckLink → System RAM → CPU copy → GPU VRAM → Processing
            (PCIe)      (4.1 MB)     (0.3ms)    (VRAM)

Latency: ~0.3-0.4ms
CPU Usage: High (manages copy)
Bandwidth: Uses System RAM bandwidth


┌──────────────────────────────────────────────────────────────────────────┐
│                    GPUDirect RDMA PATH (Target)                           │
└──────────────────────────────────────────────────────────────────────────┘

SDI Input → DeckLink ═══════════════════════════▶ GPU VRAM → Processing
            (PCIe)      Direct DMA transfer       (VRAM)

Latency: ~0.1-0.2ms (2-3x faster!)
CPU Usage: Minimal (DMA engine handles everything)
Bandwidth: Saves System RAM bandwidth
```

## Hardware Requirements

### ✅ Required Components

1. **NVIDIA GPU with GPUDirect RDMA Support**
   - Consumer: RTX 3060+, RTX 4060+
   - Professional: Quadro P2000+, RTX A4000+, A6000
   - Data Center: Tesla T4, A100, H100
   
   Check support:
   ```bash
   nvidia-smi -q | grep -i "GPUDirect RDMA"
   ```

2. **DeckLink Card with RDMA Support**
   - ✅ DeckLink 4K Extreme 12G
   - ✅ DeckLink 8K Pro
   - ✅ DeckLink Quad HDMI Recorder
   - ⚠️  DeckLink Mini models may not support RDMA
   
   Check card:
   ```bash
   lspci | grep -i blackmagic
   ```

3. **PCIe Topology**
   - Both GPU and DeckLink must be on same PCIe root complex
   - Or connected via PCIe switch
   
   Verify topology:
   ```bash
   lspci -tv
   # Look for GPU and DeckLink under same PCIe bridge
   ```

4. **Motherboard Requirements**
   - IOMMU support (VT-d for Intel, AMD-Vi for AMD)
   - Sufficient PCIe lanes (8x for DeckLink, 16x for GPU minimum)
   - Above 4G Decoding support
   - Resizable BAR support (recommended)

## Software Requirements

### Operating System

**Linux (Recommended)**
- Ubuntu 20.04 LTS or later
- RHEL/CentOS 7.0 or later
- Kernel 4.15+ (5.x recommended)

**Windows (Limited Support)**
- Windows 10/11 Pro/Enterprise
- Special configuration required
- Performance may be lower

### NVIDIA Components

1. **NVIDIA Driver**
   - Version 470.x or later
   - Data Center Driver (recommended for production)
   
   ```bash
   nvidia-smi
   # Should show driver version 470+
   ```

2. **CUDA Toolkit**
   - CUDA 11.0 or later (12.x recommended)
   
   ```bash
   nvcc --version
   ```

3. **NVIDIA Peer Memory Driver** (nv_peer_mem)
   ```bash
   # Install from source
   git clone https://github.com/Mellanox/nv_peer_memory.git
   cd nv_peer_memory
   make
   sudo make install
   
   # Load module
   sudo modprobe nv_peer_mem
   
   # Verify
   lsmod | grep nv_peer_mem
   ```

### Blackmagic Components

1. **DeckLink SDK**
   - Version 12.0 or later
   - Download from Blackmagic website

2. **Desktop Video Software**
   - Latest version
   - Includes drivers for DeckLink cards

## System Configuration

### 1. BIOS Settings

Access BIOS/UEFI and enable:

```
✅ VT-d (Intel) or AMD-Vi (AMD)
✅ IOMMU
✅ Above 4G Decoding
✅ Resizable BAR (if available)
✅ PCIe ASPM: Disabled (for lowest latency)
```

### 2. Linux Kernel Configuration

Edit `/etc/default/grub`:

```bash
# For Intel CPU
GRUB_CMDLINE_LINUX="intel_iommu=on iommu=pt"

# For AMD CPU
GRUB_CMDLINE_LINUX="amd_iommu=on iommu=pt"

# Update grub
sudo update-grub
sudo reboot
```

### 3. Verify IOMMU

```bash
# Check if IOMMU is enabled
dmesg | grep -i iommu

# Should see output like:
# DMAR: IOMMU enabled
# or
# AMD-Vi: AMD IOMMUv2 loaded and initialized
```

### 4. Check GPU Peer-to-Peer Support

```bash
# CUDA sample
cd /usr/local/cuda/samples/1_Utilities/p2pBandwidthLatencyTest
make
./p2pBandwidthLatencyTest

# Should show peer access available
```

## Implementation Steps

### Step 1: Allocate GPU Memory for Capture

Instead of using System RAM DMA buffer, allocate VRAM:

```cpp
// In shim.cpp or capture initialization

#include <cuda_runtime.h>

// Allocate GPU memory for frames
void* d_frameBuffer;
size_t frameSize = width * height * 2; // YUV422 = 2 bytes per pixel

cudaError_t err = cudaMalloc(&d_frameBuffer, frameSize);
if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate GPU memory: %s\n", 
            cudaGetErrorString(err));
    return;
}

// Get device pointer as physical address for DMA
CUdeviceptr devicePtr = (CUdeviceptr)d_frameBuffer;
```

### Step 2: Configure DeckLink for GPUDirect

Modify DeckLink capture to use GPU memory:

```cpp
// In VideoInputFrameArrived callback

// Option 1: Use IDeckLinkVideoFrame with GPU memory
HRESULT result = videoFrame->GetBytes(&buffer);

// Instead, configure DeckLink to write to GPU memory directly
// This requires DeckLink SDK 12.0+ with GPUDirect support

IDeckLinkVideoFrameGPUMemory* gpuFrame = nullptr;
result = videoFrame->QueryInterface(
    IID_IDeckLinkVideoFrameGPUMemory, 
    (void**)&gpuFrame
);

if (result == S_OK) {
    // Frame is in GPU memory
    CUdeviceptr gpuPtr;
    gpuFrame->GetGPUMemoryAddress(&gpuPtr);
    
    // Pass GPU pointer to Rust (no copy needed!)
    // ...
}
```

### Step 3: Modify Rust Code

Update `RawFramePacket` to handle GPU pointers:

```rust
// In crates/common_io/src/lib.rs

#[derive(Debug, Clone, Copy)]
pub enum MemoryLocation {
    Cpu,    // System RAM
    Gpu,    // GPU VRAM (GPUDirect)
}

pub struct RawFrameData {
    pub ptr: *mut u8,
    pub len: usize,
    pub stride: usize,
    pub loc: MemoryLocation,  // Already exists!
}

// In crates/decklink_input/src/capture.rs

// When GPUDirect is active, set:
let data = RawFrameData {
    ptr: gpu_ptr as *mut u8,
    len: frame_size,
    stride: stride,
    loc: MemoryLocation::Gpu,  // ✅ Important!
};
```

### Step 4: Skip H2D Transfer in Preprocessing

```rust
// In preprocess_cuda crate

pub fn preprocess(input: &RawFramePacket) -> Result<Tensor> {
    match input.data.loc {
        MemoryLocation::Gpu => {
            // Already in GPU! Just process directly
            yuv422_to_rgb_kernel(input.data.ptr, ...);
            // No cudaMemcpy needed! ✅
        }
        MemoryLocation::Cpu => {
            // Need H2D transfer
            let mut d_input = cuda_malloc(input.data.len)?;
            cuda_memcpy_h2d(d_input, input.data.ptr, input.data.len)?;
            yuv422_to_rgb_kernel(d_input, ...);
        }
    }
}
```

## Performance Comparison

### Latency Budget (1080p60 YUV422)

| Stage | Normal Path | GPUDirect RDMA | Savings |
|-------|-------------|----------------|---------|
| DeckLink capture | 16-33ms | 16-33ms | 0ms |
| DMA to RAM | ~0.0003ms | - | - |
| CPU access | ~0.02ms | - | 0.02ms |
| H2D transfer | ~0.3-0.4ms | ~0.1-0.2ms | **0.2ms** |
| GPU preprocess | ~1ms | ~1ms | 0ms |
| GPU inference | ~3ms | ~3ms | 0ms |
| **TOTAL** | **~20-37ms** | **~20-37ms** | **0.2ms** |

### Real Impact

- **Latency Reduction**: 0.2ms (not huge, but consistent)
- **CPU Usage Reduction**: ~5-10% (significant!)
- **System RAM Bandwidth**: Saves 4.1 MB × 60 fps = ~240 MB/s
- **Jitter Reduction**: More consistent frame times (important for real-time)

## Verification and Testing

### 1. Check if GPUDirect is Active

```bash
# Monitor GPU memory usage
nvidia-smi dmon -s u

# Should see memory usage increase when capturing

# Check peer memory access
cat /sys/class/infiniband/mlx5_0/device/gpu_direct_rdma/supported
# Should return 1 if supported
```

### 2. Benchmark Latency

Create test program:

```rust
// apps/playgrounds/src/bin/test_gpudirect.rs

fn main() {
    let mut session = CaptureSession::open(0)?;
    
    for _ in 0..100 {
        let packet = session.get_frame()?.unwrap();
        
        match packet.data.loc {
            MemoryLocation::Gpu => {
                println!("✅ Frame in GPU memory");
                // Measure preprocessing time (should be faster)
            }
            MemoryLocation::Cpu => {
                println!("⚠️  Frame in CPU memory (fallback)");
            }
        }
    }
}
```

### 3. Validate Data Integrity

```rust
// Ensure data is correctly captured to GPU

// Read back from GPU
let mut host_buffer = vec![0u8; packet.data.len];
unsafe {
    cuda_memcpy_d2h(
        host_buffer.as_mut_ptr(),
        packet.data.ptr,
        packet.data.len
    );
}

// Verify YUV422 format
assert_eq!(host_buffer.len(), 1920 * 1080 * 2);
```

## Troubleshooting

### Problem: "GPUDirect RDMA not available"

**Solution:**
1. Check IOMMU is enabled: `dmesg | grep -i iommu`
2. Verify nv_peer_mem loaded: `lsmod | grep nv_peer`
3. Check PCIe topology: `lspci -tv`
4. Ensure GPU driver supports RDMA: `nvidia-smi -q | grep RDMA`

### Problem: "DeckLink cannot write to GPU memory"

**Solution:**
1. Update DeckLink SDK to 12.0+
2. Check DeckLink firmware: `BlackmagicFirmwareUpdater`
3. Verify card supports RDMA (4K Extreme, 8K Pro)
4. Check Blackmagic forum for specific card support

### Problem: "Performance no better than normal path"

**Solution:**
1. Verify data is actually in GPU: `nvidia-smi dmon`
2. Check PCIe bandwidth: `nvidia-smi topo -m`
3. Ensure no unnecessary D2H transfers
4. Profile with `nvprof` or Nsight Systems

### Problem: "System crashes or freezes"

**Solution:**
1. Check IOMMU groups: `find /sys/kernel/iommu_groups/ -type l`
2. Verify devices in same IOMMU group
3. Try `iommu=pt` instead of `iommu=on`
4. Check kernel logs: `dmesg -T | tail -100`

## Production Checklist

Before deploying GPUDirect RDMA in production:

- [ ] Verified hardware compatibility
- [ ] IOMMU enabled and tested
- [ ] nv_peer_mem loaded automatically on boot
- [ ] Tested capture→process pipeline
- [ ] Validated data integrity
- [ ] Benchmarked latency improvement
- [ ] Tested error recovery (cable disconnect, etc.)
- [ ] Documented fallback to normal path if RDMA fails
- [ ] Load tested at full frame rate (60fps)
- [ ] Monitored system stability over 24+ hours

## Alternative: NVIDIA GPUDirect Storage

If your goal is maximum throughput (not minimum latency), consider **GPUDirect Storage** (GDS):

```
DeckLink → NVMe SSD → GPU VRAM
           (via DMA)
```

Benefits:
- Record directly to NVMe at full bandwidth
- Replay from NVMe to GPU without CPU
- Good for video editing workflows

## Summary

**Is GPUDirect RDMA Worth It?**

✅ **YES if:**
- You need absolute minimum latency (<20ms total)
- You have high-end hardware (Quadro/RTX professional)
- CPU is a bottleneck in your system
- You're building a production system

❌ **NO if:**
- You have consumer hardware (limited RDMA support)
- 0.2ms latency saving doesn't matter for your use case
- Complexity and debugging time not worth savings
- You don't have Linux expertise for setup

**Recommendation for DeepGIBox:**
1. **Start without GPUDirect** - get pipeline working first
2. **Benchmark current performance** - measure actual bottlenecks
3. **If latency <25ms is required** - invest time in GPUDirect
4. **If CPU usage >80%** - GPUDirect will help significantly

The 0.2ms savings is nice, but the **real value is consistent latency and lower CPU usage**.

## References

- [NVIDIA GPUDirect RDMA Documentation](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [Blackmagic DeckLink SDK Manual](https://www.blackmagicdesign.com/support/)
- [Linux IOMMU Configuration](https://www.kernel.org/doc/Documentation/IOMMU.txt)
- [CUDA Peer-to-Peer Access](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#peer-to-peer-memory-access)
