// log.rs - Human-readable log backend
/// Emit telemetry data in human-readable format to stderr
#[inline]
pub fn emit(name: &str, ms: f64) {
    eprintln!("[lat] {name}={ms:.2}ms");
}
