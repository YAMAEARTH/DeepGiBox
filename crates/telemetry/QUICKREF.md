# Telemetry Quick Reference

## Import
```rust
use telemetry::{now_ns, record_ms, time_stage};
```

## Measure End-to-End Frame Time
```rust
let t_frame = now_ns();

// ... process pipeline ...

record_ms("frame.e2e", t_frame);
```

## Measure Per-Stage Latency

### Using time_stage() (Recommended)
```rust
let output = time_stage("stage_name", &mut stage, input);
```

### Manual Timing
```rust
let t0 = now_ns();
// ... do work ...
record_ms("stage_name", t0);
```

## Complete Pipeline Example
```rust
fn main() -> Result<()> {
    let mut cap = capture::CaptureSession::open(0)?;
    let mut pre = preprocess::Preprocessor::new()?;
    let mut infer = inference::InferenceEngine::new()?;
    
    loop {
        // Start frame timer
        let t_frame = now_ns();
        
        // Process pipeline with per-stage timing
        let raw = time_stage("capture", &mut cap, ());
        let tensor = time_stage("preprocess", &mut pre, raw);
        let detections = time_stage("inference", &mut infer, tensor);
        
        // End-to-end timing
        record_ms("frame.e2e", t_frame);
    }
}
```

## Output Formats

### Human-Readable (default)
```
[lat] capture=1.23ms
[lat] preprocess=2.45ms
[lat] inference=15.67ms
[lat] frame.e2e=19.35ms
```

### JSON
```json
{"ts":1234567890,"name":"capture","ms":1.230}
{"ts":1234569120,"name":"preprocess","ms":2.450}
{"ts":1234584687,"name":"inference","ms":15.670}
```

## Switching Output Format

### Build with JSON
```bash
cargo build -p myapp --no-default-features --features json
```

### Or in Cargo.toml
```toml
[dependencies]
telemetry = { path = "../../crates/telemetry", default-features = false, features = ["json"] }
```

## Standard Stage Names
- `capture` - Frame capture
- `preprocess` - Video preprocessing
- `inference` - Model inference
- `postprocess` - Detection postprocessing
- `overlay_plan` - Overlay planning
- `overlay_render` - Overlay rendering
- `frame.e2e` - End-to-end time

## Demo
```bash
cargo run -p playgrounds --bin telemetry_demo
```
