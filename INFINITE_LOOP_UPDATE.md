# Pipeline Infinite Loop Update

## 🔄 Changes Made

### Previous Behavior:
- Pipeline processed **10 frames** then stopped
- Showed summary statistics at the end
- Required manual restart for continuous operation

### New Behavior:
- Pipeline runs **continuously (infinite loop)**
- Automatically displays statistics every **60 frames**
- Must press **Ctrl+C** to stop
- Real-time performance monitoring

---

## 📊 What Changed

### 1. Loop Type
```rust
// BEFORE:
let frames_to_process = 10;
while frame_count < frames_to_process { ... }

// AFTER:
loop {  // Infinite loop
    ...
}
```

### 2. Frame Counter
```rust
// BEFORE:
let mut frame_count = 0;  // u32, limited to frames_to_process

// AFTER:
let mut frame_count = 0u64;  // u64, can count indefinitely
```

### 3. Statistics Display
```rust
// BEFORE:
// Summary at end after 10 frames

// AFTER:
// Real-time stats every 60 frames
if frame_count > 0 && frame_count % 60 == 0 {
    println!("📊 Stats after {} frames", frame_count);
    println!("  Average FPS: {:.2}", avg_fps);
    println!("  Average E2E Latency: {:.2}ms", avg_latency);
    // ... breakdown
}
```

### 4. Delays
```rust
// BEFORE:
std::thread::sleep(Duration::from_millis(100));  // 100ms delay per frame

// AFTER:
// No delay - maximum throughput!
// (commented out for max speed)
```

---

## 🚀 How to Use

### Start Pipeline (Infinite Loop):
```bash
cargo run --release --bin pipeline_capture_to_output_v1
```

### Stop Pipeline:
Press **Ctrl+C** to interrupt

---

## 📈 Real-Time Statistics

The pipeline now shows statistics every 60 frames:

```
════════════════════════════════════════════════════════════
📊 Stats after 60 frames (10.5s elapsed):
  Average FPS: 5.71
  Average E2E Latency: 175.23ms
  Breakdown:
    Capture:      0.00ms
    Preprocess:   5.15ms
    Inference:    16.78ms
    Postprocess:  0.31ms
    Overlay Plan: 0.02ms
    Overlay Render:2.65ms
    Internal Key: 22.40ms
════════════════════════════════════════════════════════════
```

### Stats Include:
- ✅ **Total frames processed**
- ✅ **Elapsed time**
- ✅ **Average FPS** (frames per second)
- ✅ **Average E2E latency** (end-to-end)
- ✅ **Latency breakdown** for each pipeline stage:
  - Capture
  - Preprocessing
  - Inference
  - Postprocessing
  - Overlay Planning
  - Overlay Rendering
  - Internal Keying

---

## 🎯 Expected Performance

### Target Performance (Real-time):
- **60 FPS** (16.67ms per frame)
- Suitable for live SDI output

### Current Performance (with file saves):
- **~6-10 FPS** (due to PNG file saving)
- To achieve 60 FPS:
  - Disable PNG file saves (comment out Step 9)
  - Keep only internal keying output

### Optimization Tips:
```rust
// For maximum FPS, comment out file saving:
// Step 9: Save Images
/*
if frame_count % 60 == 0 {  // Save only every 60th frame
    save_original_frame(&raw_frame_cpu, &original_path)?;
    save_overlay_as_image(&overlay_frame, &overlay_path)?;
    save_comparison_image(&original_path, &overlay_path, &comparison_path)?;
}
*/
```

---

## 🔍 Monitoring

### Watch Frame Processing:
```bash
cargo run --release --bin pipeline_capture_to_output_v1 2>&1 | grep "Frame #"
```

### Watch FPS Stats Only:
```bash
cargo run --release --bin pipeline_capture_to_output_v1 2>&1 | grep "Average FPS"
```

### Full Output:
```bash
cargo run --release --bin pipeline_capture_to_output_v1
```

---

## ✅ Benefits

1. **Continuous Operation** - No manual restarts needed
2. **Real-time Monitoring** - See stats every 60 frames
3. **Live Output** - Continuous DeckLink SDI output
4. **Long-term Testing** - Can run for hours/days
5. **Performance Profiling** - Track FPS over time

---

## 📝 Notes

- Pipeline outputs **live video** to DeckLink SDI continuously
- PNG files are still saved for every frame (can be disabled for better FPS)
- Press **Ctrl+C** to gracefully stop the pipeline
- Frame counter uses `u64` to support billions of frames

---

## 🎊 Status

**✅ Pipeline now runs continuously!**

```
🎬 Step 6: Processing Frames (Infinite Loop - Press Ctrl+C to stop)...
════════════════════════════════════════════════════════════

🎬 Frame #0
🎬 Frame #1
🎬 Frame #2
...
[continues indefinitely]
```

**Press Ctrl+C when you want to stop! 🛑**
