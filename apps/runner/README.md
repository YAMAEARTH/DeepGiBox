# DeepGiBox Runner

Production application runner for DeepGiBox real-time detection and overlay system.

## Overview

The runner provides a flexible, configuration-based interface to run different pipeline modes:

- **Hardware Keying** - Full production pipeline with real-time overlay output via DeckLink hardware keying
- **Inference Only** - Benchmark mode for maximum throughput testing (no output overhead)
- **Visualization** - Offline analysis mode that saves detection frames to disk (not yet implemented)

## Building

```bash
# Build in release mode (required for real-time performance)
cargo build --release --bin runner

# Or run directly
cargo run --release --bin runner -- <config_path>
```

## Usage

```bash
# Run with a specific configuration file
./target/release/runner configs/runner_keying.toml

# Or use cargo run
cargo run --release --bin runner -- configs/runner_keying.toml
```

### Available Configurations

1. **`configs/runner_keying.toml`** - Hardware Internal Keying Pipeline
   - Full production pipeline
   - Real-time overlay with DeckLink hardware keying
   - Runs indefinitely until Ctrl+C
   - Use for: Production deployment, live events

2. **`configs/runner_inference_only.toml`** - Inference Only Pipeline
   - Minimal overhead for maximum throughput
   - No overlay rendering or output
   - 60-second test duration
   - Use for: Performance benchmarking, throughput testing

3. **`configs/runner_visualization.toml`** - Visualization Pipeline (placeholder)
   - Saves detection frames to disk
   - Not yet implemented
   - Use for: Offline analysis, quality review

## Configuration File Structure

All configuration files follow this structure:

```toml
# Pipeline mode: "hardware_keying", "inference_only", or "visualization"
mode = "hardware_keying"

[general]
test_duration_seconds = 0  # 0 = unlimited
enable_debug_dumps = false
stats_print_interval = 60

[capture]
device_index = 0
expected_resolution = "1080p60"

[preprocessing]
output_width = 512
output_height = 512
crop_region = "Olympus"
# ... more preprocessing settings

[inference]
engine_path = "configs/model/v7_optimized_YOLOv5.engine"
lib_path = "trt-shim/build/libtrt_shim.so"

[postprocessing]
confidence_threshold = 0.25
# ... tracking, smoothing settings

[overlay]
enable_full_ui = true
# ... bbox, label settings

[keying]  # Hardware keying mode only
enable_internal_keying = true
keyer_level = 255

[performance]
print_per_frame_latency = true
# ... buffer settings
```

## Pipeline Modes

### 1. Hardware Keying Mode (`hardware_keying`)

**Full Pipeline:**
```
DeckLink Capture → CUDA Preprocessing → TensorRT Inference → 
Postprocessing (NMS + Tracking) → Overlay Planning → 
GPU Rendering → Hardware Internal Keying → SDI Output
```

**Features:**
- Real-time overlay with DeckLink hardware alpha blending
- Adaptive queue management (2-5 frames based on performance)
- Scheduled frame completion callbacks (async, non-blocking)
- Zero-copy GPU pipeline (all buffers stay on GPU)
- Hardware-controlled timing for minimal jitter

**Output:**
- Fill signal: Original UYVY video
- Key signal: BGRA overlay with alpha channel
- Hardware keyer blends them in real-time

**Use Case:** Production deployment in live medical procedures

### 2. Inference Only Mode (`inference_only`)

**Minimal Pipeline:**
```
DeckLink Capture → CUDA Preprocessing → TensorRT Inference → 
Postprocessing (NMS + Tracking)
```

**Features:**
- No overlay rendering overhead
- No output hardware overhead
- Maximum throughput for benchmarking
- Optional debug dumps for first N frames

**Output:** Console statistics only (FPS, latency, detection counts)

**Use Case:** Performance testing, algorithm validation

### 3. Visualization Mode (`visualization`)

**Pipeline (planned):**
```
DeckLink Capture → CUDA Preprocessing → TensorRT Inference → 
Postprocessing (NMS + Tracking) → Overlay Planning → 
CPU Rendering → Save to Disk (PNG/JPEG/MP4)
```

**Status:** Not yet implemented

**Use Case:** Offline quality review, creating demo videos

## Performance Monitoring

The runner provides detailed performance statistics:

```
📊 Frame 180 | Latency: 34.52ms | FPS: 29.87 | Queue: 2/3
```

Press Ctrl+C to stop and see final summary:

```
╔══════════════════════════════════════════════════════════╗
║  FINAL SUMMARY - HARDWARE KEYING PIPELINE               ║
╚══════════════════════════════════════════════════════════╝

  📈 Performance:
    Total frames:       1800
    Total time:         60.24s
    Average FPS:        29.88

  ⏱️  Average Latency:
    Capture:            12.34ms
    Preprocessing:       4.21ms
    Inference:          14.67ms
    Postprocessing:      1.85ms
    Overlay Planning:    0.73ms
    GPU Rendering:       2.15ms
    Hardware Keying:     0.58ms
    ─────────────────────────────────
    Total (E2E):        36.53ms
```

## Customization

### Creating a Custom Configuration

1. Copy an existing config file:
   ```bash
   cp configs/runner_keying.toml configs/my_config.toml
   ```

2. Edit the settings:
   - Adjust `crop_region` for different endoscope types
   - Change `confidence_threshold` for detection sensitivity
   - Enable/disable tracking or temporal smoothing
   - Adjust overlay appearance (colors, sizes, etc.)

3. Run with your config:
   ```bash
   cargo run --release --bin runner -- configs/my_config.toml
   ```

### Common Adjustments

**For 4K input:**
```toml
[capture]
expected_resolution = "4k30"
```

**For different crop regions:**
```toml
[preprocessing]
crop_region = "Pentax"  # or "Fuji", "Olympus", "None"
```

**For higher confidence threshold:**
```toml
[postprocessing]
confidence_threshold = 0.50  # Only show high-confidence detections
```

**For minimal UI (bboxes only):**
```toml
[overlay]
enable_full_ui = false
```

## Troubleshooting

### "No DeckLink devices found"
- Check DeckLink card is installed and recognized by OS
- Run `apps/playgrounds/bin/pipeline_capture_to_output_v5_keying` to test

### "TensorRT engine not found"
- Build the TensorRT engine first:
  ```bash
  python rebuild_engine_640.py
  ```

### "TRT shim library not found"
- Build the TRT shim:
  ```bash
  cd trt-shim
  mkdir build && cd build
  cmake .. && make
  ```

### Low FPS or high latency
- Check GPU load: `nvidia-smi`
- Reduce queue depth in adaptive mode
- Disable temporal smoothing or tracking if not needed
- Use FP16 mode for faster preprocessing (if model supports it)

### Overlay not visible
- Check keyer level: `keyer_level = 255` (fully visible)
- Verify `enable_internal_keying = true`
- Check SDI output connection on DeckLink card

## Architecture

```
apps/runner/
├── Cargo.toml           # Dependencies
├── README.md           # This file
└── src/
    ├── main.rs         # Main entry point, pipeline implementations
    └── config_loader.rs # Configuration loading and validation

configs/
├── runner_keying.toml              # Hardware keying config
├── runner_inference_only.toml      # Benchmark config
└── runner_visualization.toml       # Visualization config (placeholder)
```

## Dependencies

### Internal Crates
- `common_io` - Shared data structures (FramePacket, DetectionsPacket, etc.)
- `decklink_input` - DeckLink capture interface
- `preprocess_cuda` - CUDA preprocessing (YUV→RGB, crop, resize, normalize)
- `inference_v2` - TensorRT inference wrapper
- `postprocess` - NMS, tracking (SORT), temporal smoothing
- `overlay_plan` - Overlay planning (detection → draw operations)
- `overlay_render` - GPU overlay rendering (CUDA kernels → BGRA)
- `decklink_output` - DeckLink output with hardware keying support

### External Crates
- `anyhow` - Error handling
- `serde` + `toml` - Configuration parsing
- `ctrlc` - Graceful shutdown on Ctrl+C
- `cudarc` - CUDA memory management

## License

Copyright © 2024 YAMAEARTH. All rights reserved.
