use std::time::Instant;
static mut T0: Option<Instant> = None;
pub fn now_ns() -> u64 {
    unsafe {
        if T0.is_none() {
            T0 = Some(Instant::now())
        }
        T0.unwrap().elapsed().as_nanos() as u64
    }
}
pub fn record_ms(name: &str, start_ns: u64) {
    let dt_ms = (now_ns() - start_ns) as f64 / 1_000_000.0;
    eprintln!("[lat] {}={:.2}ms", name, dt_ms);
}
pub fn time_stage<I, O, S: common_io::Stage<I, O>>(name: &'static str, s: &mut S, x: I) -> O {
    let t0 = now_ns();
    let y = s.process(x);
    record_ms(name, t0);
    y
}
