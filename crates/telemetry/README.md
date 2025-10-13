# Telemetry Crate

Lightweight observability for DeepGIBox real-time video pipeline.

## Features

- ✅ Measure per-stage latency (capture, preprocess, inference, etc.)
- ✅ Measure end-to-end frame time
- ✅ Minimal overhead (<1-2%)
- ✅ Feature-gated backends (human-log, json)
- ⏳ Optional fine-grained spans (H2D, kernels, D2D) - not yet implemented
- ⏳ Rolling statistics - not yet implemented
- ⏳ CUDA events - not yet implemented

## Structure

```
crates/telemetry/
├── Cargo.toml
└── src/
    ├── lib.rs      # Main API: record_ms(), time_stage()
    ├── time.rs     # Core timing: now_ns(), since_ms()
    ├── log.rs      # Human-readable output backend
    └── json.rs     # JSON structured output backend
```

## API

### Core Functions

```rust
pub fn now_ns() -> u64
```
Returns nanoseconds elapsed since first call (monotonic clock).

```rust
pub fn since_ms(start_ns: u64) -> f64
```
Returns milliseconds elapsed since `start_ns`.

```rust
pub fn record_ms(name: &str, start_ns: u64)
```
Records a measurement and emits it to the configured backend.

```rust
pub fn time_stage<I, O, S: common_io::Stage<I, O>>(
    name: &'static str,
    stage: &mut S,
    input: I,
) -> O
```
Wraps a `Stage::process()` call with automatic timing measurement.

## Feature Flags

- `human-log` (default): Human-readable output format
- `json`: Machine-parseable JSON output format
- `stats`: Rolling statistics (not yet implemented)
- `cuda-events`: CUDA event timing (not yet implemented)

### Using Different Backends

**Default (human-readable):**
```toml
[dependencies]
telemetry = { path = "../../crates/telemetry" }
```

**JSON output:**
```toml
[dependencies]
telemetry = { path = "../../crates/telemetry", default-features = false, features = ["json"] }
```

## Usage Examples

### Basic Per-Stage Timing

```rust
use common_io::Stage;
use telemetry;

fn main() -> anyhow::Result<()> {
    let mut capture = /* ... */;
    let mut preprocess = /* ... */;
    let mut inference = /* ... */;
    
    // Time each stage individually
    let raw = telemetry::time_stage("capture", &mut capture, ());
    let tensor = telemetry::time_stage("preprocess", &mut preprocess, raw);
    let detections = telemetry::time_stage("inference", &mut inference, tensor);
    
    Ok(())
}
```

Output (human-log):
```
[lat] capture=1.23ms
[lat] preprocess=2.45ms
[lat] inference=15.67ms
```

Output (json):
```json
{"ts":1234567890,"name":"capture","ms":1.230}
{"ts":1234568901,"name":"preprocess","ms":2.450}
{"ts":1234584568,"name":"inference","ms":15.670}
```

### End-to-End Frame Timing

```rust
fn main() -> anyhow::Result<()> {
    let mut capture = /* ... */;
    let mut preprocess = /* ... */;
    let mut inference = /* ... */;
    let mut postprocess = /* ... */;
    let mut overlay = /* ... */;
    let mut output = /* ... */;
    
    // Start measuring total frame time
    let t_frame = telemetry::now_ns();
    
    // Process pipeline stages
    let raw = telemetry::time_stage("capture", &mut capture, ());
    let tensor = telemetry::time_stage("preprocess", &mut preprocess, raw);
    let detections = telemetry::time_stage("inference", &mut inference, tensor);
    let processed = telemetry::time_stage("postprocess", &mut postprocess, detections);
    let rendered = telemetry::time_stage("overlay", &mut overlay, processed);
    output.submit(rendered)?;
    
    // Record total frame time
    telemetry::record_ms("frame.e2e", t_frame);
    
    Ok(())
}
```

Output:
```
[lat] capture=1.23ms
[lat] preprocess=2.45ms
[lat] inference=15.67ms
[lat] postprocess=0.89ms
[lat] overlay=3.21ms
[lat] frame.e2e=23.45ms
```

### Manual Timing (without Stage trait)

```rust
fn do_work() {
    let t0 = telemetry::now_ns();
    
    // ... expensive operation ...
    
    telemetry::record_ms("my_operation", t0);
}
```

### Fine-Grained Sub-Spans (Future)

This feature is planned but not yet implemented. The intended usage:

```rust
fn process(&mut self, input: RawFramePacket) -> TensorInputPacket {
    let t_all = telemetry::now_ns();
    
    let t0 = telemetry::now_ns();
    self.upload_h2d(&input)?;
    telemetry::record_ms("preprocess.h2d", t0);
    
    let t1 = telemetry::now_ns();
    self.yuv422_to_rgb_bt709()?;
    self.resize_and_normalize()?;
    telemetry::record_ms("preprocess.kernels", t1);
    
    let t2 = telemetry::now_ns();
    self.pack_nchw_fp16()?;
    telemetry::record_ms("preprocess.d2d", t2);
    
    let out = self.make_output(input)?;
    telemetry::record_ms("preprocess.total", t_all);
    out
}
```

## Naming Convention

### Stage Names
- `capture` - Frame capture from input device
- `preprocess` - Video preprocessing (color conversion, resize, normalize)
- `inference` - Model inference
- `postprocess` - Detection postprocessing
- `overlay_plan` - Overlay planning
- `overlay_render` - Overlay rendering
- `frame.e2e` - End-to-end frame processing time

### Sub-Span Names (Future)
Use dot notation for hierarchical spans:
- `preprocess.h2d` - Host to device transfer
- `preprocess.kernels` - GPU kernel execution
- `preprocess.d2d` - Device to device transfer
- `preprocess.total` - Total preprocess time

## Performance

The telemetry overhead is designed to be minimal:

- `now_ns()`: ~10-20ns (single `Instant::elapsed()` call)
- `record_ms()`: ~1-2μs with `human-log`, slightly more with `json`
- `time_stage()`: ~10-20ns overhead (just wraps the Stage call)

For a typical 60 FPS pipeline (~16.7ms per frame), telemetry adds <0.1% overhead.

## Implementation Details

### Time Measurement

Uses `std::time::Instant` for monotonic clock measurements. The first call to `now_ns()` initializes a static reference point (`T0`), and all subsequent calls return nanoseconds elapsed since that point.

### Thread Safety

⚠️ **Current implementation is NOT thread-safe** - the static `T0` uses `unsafe` with mutable static. For single-threaded pipelines this is fine. For multi-threaded scenarios, consider:
- Using per-thread initialization
- Synchronization primitives
- Or accepting that all threads share the same T0

### Output

All telemetry output goes to `stderr` via `eprintln!()` to keep `stdout` clean for application data.

## Testing

```bash
# Build with default features (human-log)
cargo build -p telemetry

# Build with JSON output
cargo build -p telemetry --no-default-features --features json

# Run tests
cargo test -p telemetry
```

## Future Enhancements

1. **Rolling Statistics** (`stats` feature)
   - Track p50, p95, p99 latencies over a window
   - Automatic periodic summaries

2. **CUDA Events** (`cuda-events` feature)
   - GPU-accurate timing using CUDA events
   - Measure actual GPU execution time

3. **Fine-Grained Spans**
   - Add H2D, D2D, kernel-level measurements
   - Per-module sub-span tracking

4. **Thread Safety**
   - Make time measurement thread-safe
   - Per-thread or synchronized T0

5. **Sampling/Rate Limiting**
   - Sample every Nth frame to reduce overhead
   - Rate-limited logging for high FPS scenarios

## License

Same as parent project.
