# DeepGiBox Configuration Files

## 📁 Config Files Overview

### 🚀 Main Runner Configs (Production)

#### `runner.toml` ⭐ **RECOMMENDED**
**Universal config for all endoscope types with runtime mode switching**

- **Pipeline**: Hardware Internal Keying (full production)
- **Endoscope Modes**: Switch on-the-fly with keyboard
  - Press `1` → Fuji mode
  - Press `2` → Olympus mode
- **Features**:
  - Single config file for all endoscopes
  - No restart needed to change modes
  - Runtime crop region switching
  - Model: v7_optimized_YOLOv5 (512x512)

**Usage**:
```bash
cargo run --release -p runner -- configs/runner.toml
# Press 1 or 2 during runtime to switch endoscope modes
```

#### `runner_keying.toml`
**Hardware keying pipeline with fixed Olympus crop**

- **Pipeline**: Hardware Internal Keying
- **Endoscope**: Fixed Olympus crop region
- **Features**: Same as runner.toml but without runtime switching
- **Model**: v7_optimized_YOLOv5 (512x512)

**Usage**:
```bash
cargo run --release -p runner -- configs/runner_keying.toml
```

**Note**: Consider using `runner.toml` instead for more flexibility

---

### 🔬 Testing & Development Configs

#### `runner_inference_only.toml`
**Inference-only pipeline (no output, for benchmarking)**

- **Pipeline**: Inference Only
- **Purpose**: Performance testing, latency measurement
- **Features**:
  - No DeckLink output
  - No overlay rendering
  - Fastest pipeline for pure inference testing
  - Capture → Preprocess → Inference → Postprocess

**Usage**:
```bash
cargo run --release -p runner -- configs/runner_inference_only.toml
```

#### `runner_visualization.toml`
**Visualization mode (saves detection frames to disk)**

- **Pipeline**: Visualization
- **Purpose**: Debug, visual inspection of detections
- **Features**:
  - Saves annotated frames to disk
  - No DeckLink output
  - Useful for offline analysis

**Usage**:
```bash
cargo run --release -p runner -- configs/runner_visualization.toml
```

---

### 🧪 Component Testing Configs

#### `pipeline_config.toml`
**Legacy pipeline configuration**

- **Purpose**: Old pipeline configuration format
- **Status**: Deprecated (use runner configs instead)

#### `postprocess_test.toml`
**Postprocessing component test configuration**

- **Purpose**: Test postprocessing stage in isolation
- **Features**: NMS, tracking, temporal smoothing tests

---

## 📊 Config Comparison

| Config | Pipeline Mode | Endoscope | Runtime Switch | Output | Use Case |
|--------|--------------|-----------|----------------|--------|----------|
| **runner.toml** ⭐ | Hardware Keying | All (1,2 keys) | ✅ Yes | DeckLink | **Production** |
| runner_keying.toml | Hardware Keying | Olympus | ❌ No | DeckLink | Production (fixed) |
| runner_inference_only.toml | Inference Only | Olympus | ❌ No | None | Performance test |
| runner_visualization.toml | Visualization | Olympus | ❌ No | Disk | Debug/Analysis |

---

## 🗑️ Removed Files (Deprecated)

The following files were removed as they are replaced by `runner.toml`:

- ~~`runner_fuji.toml`~~ → Use `runner.toml` and press `1`
- ~~`runner_olympus.toml`~~ → Use `runner.toml` and press `2`
- ~~`runner_pentax.toml`~~ → **Removed (no longer supported)**

**Migration**: Simply use `runner.toml` and switch modes with keyboard shortcuts during runtime.

---

## ⚙️ Common Configuration Sections

All runner configs share these sections:

### `[general]`
- Test duration, debug dumps, stats interval

### `[capture]`
- DeckLink device index, input resolution

### `[preprocessing]`
- Model input size, FP16, CUDA device, normalization
- Crop region (or initial_endoscope_mode for runner.toml)

### `[inference]`
- TensorRT engine path, TRT shim library

### `[postprocessing]`
- Confidence/NMS thresholds, tracking, temporal smoothing

### `[overlay]`
- Bounding box style, label appearance, full UI mode

### `[rendering]`
- Text antialiasing, debug rendering

### `[keying]` (Hardware Keying mode only)
- Internal keying, keyer level

### `[performance]`
- GPU buffer pool, timing prints

### `[classes]`
- Class labels (e.g., ["Hyper", "Neo"])

---

## 🚀 Quick Start

### For Production (Recommended)
```bash
# Use unified config with runtime mode switching
cargo run --release -p runner -- configs/runner.toml

# During runtime:
# - Press 1 for Fuji
# - Press 2 for Olympus
# - Press Ctrl+C to stop
```

### For Performance Testing
```bash
cargo run --release -p runner -- configs/runner_inference_only.toml
```

### For Debug/Visualization
```bash
cargo run --release -p runner -- configs/runner_visualization.toml
```

---

## 📝 Notes

- **Model Location**: `configs/model/v7_optimized_YOLOv5.engine`
- **TRT Shim**: `trt-shim/build/libtrt_shim.so`
- **Default Resolution**: 1080p60 (auto-detects 4K)
- **Default CUDA Device**: GPU 0

---

**Last Updated**: November 7, 2025  
**DeepGiBox Version**: Production
