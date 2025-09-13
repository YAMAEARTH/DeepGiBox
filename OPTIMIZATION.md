# DeckLink Rust Optimization for Ubuntu 24.04

## Overview
This document tracks the optimization of the DeckLink Rust project for Ubuntu 24.04 LTS. The project currently focuses on macOS but requires comprehensive changes to work efficiently on Linux.

## Current Status
- **Platform**: Ubuntu 24.04.3 LTS (noble)
- **DeckLink Libraries**: Installed at `/usr/lib/libDeckLinkAPI.so` and `/usr/lib/libDeckLinkPreviewAPI.so`
- **DeckLink SDK**: Found at `/opt/Blackmagic DeckLink SDK 15.0/Linux/include/`
- **Build Status**: ✅ Successfully building with Linux support
- **Architecture**: Rust FFI → C++ shim → DeckLink SDK

## Optimization Task Schedule

### Phase 1: Critical Fixes
- [x] **Task 1**: Analyze Project Architecture ✅
- [x] **Task 2**: Fix Linux Build Configuration ✅
- [x] **Task 3**: Update C++ Shim for Linux Compatibility ✅
- [x] **Task 4**: Install Development Dependencies ✅

### Phase 2: Core Optimizations
- [ ] **Task 5**: Optimize Memory Management
- [ ] **Task 6**: Remove macOS-specific Features
- [ ] **Task 7**: Optimize GPU Preview Path
- [ ] **Task 8**: Add Linux-specific GPU Integration

### Phase 3: System Integration
- [ ] **Task 9**: Implement System Optimization
- [ ] **Task 10**: Create Performance Benchmarks
- [ ] **Task 11**: Add Error Handling and Logging
- [ ] **Task 12**: Optimize Build System
- [ ] **Task 13**: Add Package Management

### Phase 4: Production Ready
- [ ] **Task 14**: Implement Threading Optimizations
- [ ] **Task 15**: Add Configuration Management
- [ ] **Task 16**: Validate and Test
- [ ] **Task 17**: Documentation and Deployment

## Key Issues Identified

### 1. Build Configuration Issues ✅ RESOLVED
```bash
# Previous error:
shim/shim.cpp:17:10: fatal error: DeckLinkAPI.h: No such file or directory
```

**Root Cause**: The build.rs file didn't properly configure include paths for Ubuntu 24.04 system installation.
**Solution**: Updated build.rs to search for DeckLink SDK in `/opt/Blackmagic DeckLink SDK 15.0/Linux/include/` and added proper library linking paths.

### 2. macOS-specific Dependencies ✅ RESOLVED
- CoreFoundation/CoreVideo frameworks
- NSView screen preview functionality
- macOS-specific DeckLinkAPIDispatch.cpp

**Solution**: Added conditional compilation with `#if defined(__APPLE__)` blocks and Linux equivalents.

### 3. Missing Linux Integration
- No X11/Wayland screen preview
- No Linux-specific GPU optimizations
- Missing system-level performance tuning

## Performance Targets

### Latency Goals
- **Frame Capture**: < 16ms (sub-frame latency)
- **Processing Pipeline**: < 8ms CPU processing
- **GPU Preview**: < 4ms render latency
- **End-to-end**: < 33ms total (2-frame latency max)

### Throughput Goals
- **4K60**: Full bandwidth capture and preview
- **Multiple Streams**: Support 2+ concurrent captures
- **CPU Usage**: < 25% on modern 8-core systems
- **Memory**: < 1GB for 4K60 pipeline

## System Requirements

### Ubuntu 24.04 Dependencies
```bash
# Build tools
sudo apt install build-essential pkg-config libclang-dev

# DeckLink drivers (from Blackmagic)
# - Desktop Video 12.x or later
# - DeckLink SDK headers

# GPU acceleration
sudo apt install mesa-dev vulkan-dev

# Real-time support (optional)
sudo apt install linux-lowlatency
```

### Hardware Requirements
- **CPU**: 8+ cores, 3.0GHz+ (Intel/AMD)
- **Memory**: 16GB+ DDR4
- **GPU**: Vulkan 1.2+ compatible
- **Storage**: NVMe SSD for high-bandwidth recording

## Optimization Strategies

### 1. Memory Optimization
- SIMD-accelerated format conversions
- Zero-copy GPU transfers via DMA-BUF
- Lock-free circular buffers for frame data
- NUMA-aware memory allocation

### 2. Threading Optimization
- Dedicated capture thread with RT priority
- GPU submission on separate thread
- Lock-free producer-consumer queues
- CPU affinity for video processing

### 3. GPU Optimization
- Vulkan backend for lowest latency
- Direct texture uploads from capture buffers
- Multi-threaded command submission
- Optimal present modes and swap chains

### 4. System Integration
- RT kernel scheduling policies
- Large page memory allocation
- IOMMU bypass for DMA performance
- Power management tuning

## Completed Changes

### Task 2: Linux Build Configuration ✅
**Files Modified**: `build.rs`

**Changes Made**:
- Added automatic detection of DeckLink SDK at `/opt/Blackmagic DeckLink SDK 15.0/`
- Enhanced include path candidates for Linux:
  - `/opt/Blackmagic DeckLink SDK 15.0/Linux/include/`
  - `/usr/include/decklink/` (system installation)
  - Project's local `include/` directory
- Added Linux library search paths:
  - `/usr/lib/` and `/usr/lib/x86_64-linux-gnu/`
  - SDK-specific paths under `Linux/Libraries/`
- Added Linux DeckLinkAPIDispatch.cpp compilation

### Task 4: Development Dependencies ✅
**Packages Verified/Installed**:

**Already Available**:
- `build-essential` 12.10ubuntu1 ✅
- `pkg-config` 1.8.1-2build1 ✅  
- `libclang-common-18-dev` 1:18.1.3-1ubuntu1 ✅
- `rustc` 1.89.0 (latest stable) ✅
- `cargo` 1.89.0 ✅

**Newly Installed**:
- `vulkan-tools` 1.3.275.0+dfsg1-1 ✅
- `vulkan-utility-libraries-dev` 1.3.275.0-1 ✅
- `libvulkan-dev` 1.3.275.0-1build1 ✅
- `mesa-vulkan-drivers` 25.0.7-0ubuntu0.24.04.2 ✅

**GPU Support Verified**:
- NVIDIA GeForce RTX 3070 (discrete GPU, Vulkan 1.4.303) ✅
- Mesa LLVMpipe (CPU fallback, Vulkan 1.4.305) ✅
- DeckLink device detected: "DeckLink SDI 4K" ✅

### Task 3: C++ Shim Linux Compatibility ✅
**Files Modified**: `shim/shim.cpp`

**Changes Made**:
- Fixed `IID_IUnknown` address-of-rvalue error with static const declaration
- Added conditional compilation for macOS-specific code:
  - CoreVideo/CoreFoundation APIs wrapped in `#if defined(__APPLE__)`
  - IDeckLinkMacVideoBuffer usage only on macOS
  - CVPixelBuffer handling only on macOS
- Added Linux placeholder for screen preview functionality
- Removed direct references to `g_screenPreview` on non-macOS platforms
- Enhanced error handling for platform-specific video buffer access

**Build Status**: ✅ Successfully compiles on Ubuntu 24.04 with only minor warnings

---

## Progress Log

### September 13, 2025
- ✅ Completed project architecture analysis
- ✅ Identified DeckLink libraries at `/usr/lib/libDeckLinkAPI.so`
- ✅ Found DeckLink SDK at `/opt/Blackmagic DeckLink SDK 15.0/`
- ✅ Fixed build.rs configuration for Linux header inclusion
- ✅ Updated C++ shim for Linux compatibility
- ✅ Removed macOS-specific CoreVideo/CoreFoundation dependencies
- ✅ Added conditional compilation for platform-specific code
- ✅ Successfully building on Ubuntu 24.04
- ✅ Installed Vulkan development stack (1.3.275.0)
- ✅ Verified GPU support: NVIDIA RTX 3070 + Mesa fallback
- ✅ Device detection working: "DeckLink SDI 4K" found
- ✅ **Phase 1 Complete**: All critical fixes implemented
- 🔄 Starting Phase 2: Core Optimizations

---

## Next Steps
1. Fix build.rs to find DeckLink headers on Ubuntu
2. Update C++ shim for Linux compatibility
3. Install complete development environment
4. Begin core optimizations

*Last updated: September 13, 2025*
