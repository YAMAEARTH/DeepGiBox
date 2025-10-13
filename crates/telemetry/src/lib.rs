// lib.rs - Main telemetry API
mod time;

#[cfg(feature = "json")]
mod json;
#[cfg(feature = "human-log")]
mod log;

pub use time::{now_ns, since_ms};

/// Record a measurement in milliseconds
/// 
/// Emits the measurement to the configured backend (log or json)
pub fn record_ms(name: &str, start_ns: u64) {
    let ms = since_ms(start_ns);
    
    #[cfg(feature = "json")]
    json::emit(name, ms);
    
    #[cfg(all(not(feature = "json"), feature = "human-log"))]
    log::emit(name, ms);
}

/// Time a stage by wrapping a Stage::process call
/// 
/// This is the main helper for measuring per-stage latency
pub fn time_stage<I, O, S: common_io::Stage<I, O>>(
    name: &'static str,
    stage: &mut S,
    input: I,
) -> O {
    let t0 = now_ns();
    let output = stage.process(input);
    record_ms(name, t0);
    output
}
