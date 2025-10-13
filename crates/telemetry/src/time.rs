// time.rs - Core time measurement utilities
use std::time::Instant;

static mut T0: Option<Instant> = None;

/// Returns nanoseconds since first call (monotonic)
#[inline]
pub fn now_ns() -> u64 {
    unsafe {
        if T0.is_none() {
            T0 = Some(Instant::now());
        }
        T0.unwrap().elapsed().as_nanos() as u64
    }
}

/// Returns milliseconds elapsed since start_ns
#[inline]
pub fn since_ms(start_ns: u64) -> f64 {
    (now_ns() - start_ns) as f64 / 1_000_000.0
}
