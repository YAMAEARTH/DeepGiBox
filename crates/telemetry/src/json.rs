// json.rs - JSON structured log backend
/// Emit telemetry data in JSON format to stderr for machine parsing
#[inline]
pub fn emit(name: &str, ms: f64) {
    eprintln!(
        "{{\"ts\":{},\"name\":\"{}\",\"ms\":{:.3}}}",
        super::now_ns(),
        name,
        ms
    );
}
