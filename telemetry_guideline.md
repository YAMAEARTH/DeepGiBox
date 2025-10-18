
# Telemetry Coding Guideline

This document defines how to implement and use the `telemetry` crate for DeepGIBox.

## Purpose
Provide lightweight observability for real‑time video pipelines:
- Measure per‑stage latency (capture, preprocess, inference, etc.)
- Measure end‑to‑end frame time
- Optional fine‑grained spans (H2D, kernels, D2D)
- Minimal overhead (<1–2%)

## Design Principles
1. Use `std::time::Instant` (monotonic).
2. Low overhead — avoid allocations on hot paths.
3. Simple API: `now_ns()`, `record_ms()`, `time_stage()`.
4. Consistent metric names: `snake_case`, use `.` for subspans.
5. Backend‑agnostic — can emit human log or JSON.
6. Small and feature‑gated (e.g., rolling stats, CUDA events).

## Crate Structure
```
crates/telemetry/
├─ Cargo.toml
└─ src/
   ├─ lib.rs
   ├─ time.rs
   ├─ log.rs
   ├─ json.rs
   ├─ stats.rs
   └─ gpu.rs
```

### Cargo.toml Example
```toml
[package]
name = "telemetry"
version = "0.1.0"
edition = "2021"

[features]
default = ["log"]
json = []
stats = []
cuda-events = []

[dependencies]
common_io = { path = "../common_io" }
```

## Core API
```rust
// time.rs
use std::time::Instant;
static mut T0: Option<Instant> = None;

#[inline] pub fn now_ns() -> u64 {
    unsafe {
        if T0.is_none() { T0 = Some(Instant::now()); }
        T0.unwrap().elapsed().as_nanos() as u64
    }
}
#[inline] pub fn since_ms(start_ns: u64) -> f64 {
    (now_ns() - start_ns) as f64 / 1_000_000.0
}
```

```rust
// lib.rs
mod time;
#[cfg(feature="stats")] mod stats;
#[cfg(feature="cuda-events")] mod gpu;
#[cfg(feature="json")] mod json;
#[cfg(feature="log")] mod log;

pub use time::{now_ns, since_ms};

pub fn record_ms(name: &str, start_ns: u64) {
    let ms = since_ms(start_ns);
    #[cfg(feature="json")] json::emit(name, ms);
    #[cfg(all(not(feature="json"), feature="log"))] log::emit(name, ms);
    #[cfg(feature="stats")] stats::update(name, ms);
}

pub fn time_stage<I, O, S: common_io::Stage<I, O>>(name: &'static str, s: &mut S, x: I) -> O {
    let t0 = now_ns();
    let y = s.process(x);
    record_ms(name, t0);
    y
}
```

## Backends
### Human‑Readable Log
```rust
#[inline] pub fn emit(name: &str, ms: f64) {
    eprintln!("[lat] {name}={ms:.2}ms");
}
```

### JSON Log
```rust
#[inline] pub fn emit(name: &str, ms: f64) {
    eprintln!("{{\"ts\":{},\"name\":\"{}\",\"ms\":{:.3}}}", super::now_ns(), name, ms);
}
```

### Rolling Stats (feature = "stats")
```rust
use std::collections::VecDeque;
use std::sync::Mutex;
use once_cell::sync::Lazy;

static STORE: Lazy<Mutex<std::collections::HashMap<&'static str, WindowStats>>> =
    Lazy::new(|| Mutex::new(Default::default()));

pub struct WindowStats { buf: VecDeque<f64>, cap: usize }
impl Default for WindowStats { fn default() -> Self { Self{ buf: VecDeque::with_capacity(128), cap: 128 } } }

impl WindowStats {
    pub fn push(&mut self, v: f64) {
        if self.buf.len() == self.cap { self.buf.pop_front(); }
        self.buf.push_back(v);
    }
    pub fn p50(&self) -> f64 { self.percentile(0.50) }
    pub fn p95(&self) -> f64 { self.percentile(0.95) }
    fn percentile(&self, q: f64) -> f64 {
        if self.buf.is_empty() { return 0.0; }
        let mut v: Vec<_> = self.buf.iter().copied().collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let i = ((v.len() as f64) * q).ceil() as usize - 1;
        v[i.min(v.len()-1)]
    }
}
pub fn update(name: &'static str, ms: f64) {
    let mut m = STORE.lock().unwrap();
    m.entry(name).or_default().push(ms);
}
```

## Usage Examples

### Runner
```rust
let t_frame = telemetry::now_ns();

let raw  = telemetry::time_stage("capture",    &mut cap,   ());
let tin  = telemetry::time_stage("preprocess", &mut pre,   raw);
let rdet = telemetry::time_stage("inference",  &mut infer, tin);
let dets = telemetry::time_stage("postprocess",&mut post,  rdet);
let ops  = telemetry::time_stage("overlay_plan",&mut plan, dets);
let ofrm = telemetry::time_stage("overlay_render",&mut rend, ops);
out.submit(ofrm)?;

telemetry::record_ms("frame.e2e", t_frame);
```

### Sub‑Spans Inside Module
```rust
use telemetry::{now_ns, record_ms};

fn process(&mut self, input: RawFramePacket) -> TensorInputPacket {
    let t_all = now_ns();
    let t0 = now_ns();
    self.upload_h2d(&input)?;
    record_ms("preprocess.h2d", t0);

    let t1 = now_ns();
    self.yuv422_to_rgb_bt709()?;
    self.resize_and_normalize()?;
    record_ms("preprocess.kernels", t1);

    let t2 = now_ns();
    self.pack_nchw_fp16()?;
    record_ms("preprocess.d2d", t2);

    let out = self.make_output(input)?;
    record_ms("preprocess.total", t_all);
    out
}
```

### Unit Test
```rust
#[test]
fn preprocess_under_budget() {
    let raw = testsupport::make_dummy_frame(1920,1080);
    let mut pre = preprocess_cuda::Preprocessor::new((512,512), true, 0).unwrap();
    let t0 = telemetry::now_ns();
    let _ = pre.process(raw);
    let ms = telemetry::since_ms(t0);
    assert!(ms <= 5.0, "preprocess too slow: {ms:.2}ms");
}
```

## Feature Flags
- `log`: human‑readable (default)
- `json`: structured log for machine parsing
- `stats`: enable rolling p50/p95 collection
- `cuda-events`: measure GPU time via CUDA events

## Naming Convention
- Stages: `capture`, `preprocess`, `inference`, `postprocess`, `overlay_plan`, `overlay_render`, `frame.e2e`
- Subspans: `preprocess.h2d`, `preprocess.kernels`, `preprocess.d2d`

## Best Practices
- Use `Instant` (monotonic)
- Apply `time_stage()` on every major stage
- Add subspans in heavy modules only
- Limit window size if using rolling stats
- Avoid frequent logs in production (sample or rate limit)

## Optional Summaries
```rust
#[cfg(feature="stats")]
pub fn summarize_every(n: u64) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let c = COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
    if c % n == 0 {
        if let Some((p50, p95)) = stats::peek("inference") {
            #[cfg(feature="json")] eprintln!("{{\"name\":\"inference.roll\",\"p50\":{:.3},\"p95\":{:.3}}}", p50, p95);
            #[cfg(all(not(feature="json"), feature="log"))] eprintln!("[lat/roll] inference p50={:.2}ms p95={:.2}ms", p50, p95);
        }
    }
}
```

## Checklist
- [ ] Wrap each stage with `time_stage()`.
- [ ] Add subspans in preprocess/inference.
- [ ] Correctly set stride_bytes and GPU copy regions.
- [ ] Use one telemetry output mode in production.
- [ ] Sample or rate‑limit logs if high frame rate.

---
