// Benchmark telemetry overhead
// Measures the actual cost of timing measurements

use std::time::Instant;

fn main() {
    println!("=== Telemetry Overhead Benchmark ===\n");
    
    const ITERATIONS: usize = 1_000_000;
    
    // ===================================================================
    // Test 1: Baseline - std::time::Instant overhead
    // ===================================================================
    println!("Test 1: std::time::Instant overhead");
    println!("────────────────────────────────────");
    
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _t = Instant::now();
    }
    let baseline_total = start.elapsed();
    let baseline_per_call = baseline_total.as_nanos() as f64 / ITERATIONS as f64;
    
    println!("Total time:      {:.3} ms", baseline_total.as_secs_f64() * 1000.0);
    println!("Per call:        {:.3} ns", baseline_per_call);
    println!("Frequency:       {:.2} MHz (calls per second)", 1000.0 / baseline_per_call);
    println!();
    
    // ===================================================================
    // Test 2: Instant::now() + elapsed() overhead
    // ===================================================================
    println!("Test 2: Instant::now() + elapsed() overhead");
    println!("────────────────────────────────────────────");
    
    let t0 = Instant::now();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _elapsed = t0.elapsed();
    }
    let elapsed_total = start.elapsed();
    let elapsed_per_call = elapsed_total.as_nanos() as f64 / ITERATIONS as f64;
    
    println!("Total time:      {:.3} ms", elapsed_total.as_secs_f64() * 1000.0);
    println!("Per call:        {:.3} ns", elapsed_per_call);
    println!("vs Baseline:     {:.2}x", elapsed_per_call / baseline_per_call);
    println!();
    
    // ===================================================================
    // Test 3: telemetry::now_ns() overhead
    // ===================================================================
    println!("Test 3: telemetry::now_ns() overhead");
    println!("─────────────────────────────────────");
    
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _t = telemetry::now_ns();
    }
    let now_ns_total = start.elapsed();
    let now_ns_per_call = now_ns_total.as_nanos() as f64 / ITERATIONS as f64;
    
    println!("Total time:      {:.3} ms", now_ns_total.as_secs_f64() * 1000.0);
    println!("Per call:        {:.3} ns", now_ns_per_call);
    println!("vs Baseline:     {:.2}x", now_ns_per_call / baseline_per_call);
    println!();
    
    // ===================================================================
    // Test 4: telemetry::since_ms() overhead
    // ===================================================================
    println!("Test 4: telemetry::since_ms() overhead");
    println!("───────────────────────────────────────");
    
    let t_start = telemetry::now_ns();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ms = telemetry::since_ms(t_start);
    }
    let since_ms_total = start.elapsed();
    let since_ms_per_call = since_ms_total.as_nanos() as f64 / ITERATIONS as f64;
    
    println!("Total time:      {:.3} ms", since_ms_total.as_secs_f64() * 1000.0);
    println!("Per call:        {:.3} ns", since_ms_per_call);
    println!("vs Baseline:     {:.2}x", since_ms_per_call / baseline_per_call);
    println!();
    
    // ===================================================================
    // Test 5: Full telemetry::record_ms() overhead (with logging)
    // ===================================================================
    println!("Test 5: telemetry::record_ms() overhead (with stderr output)");
    println!("─────────────────────────────────────────────────────────────");
    println!("Running {} iterations (this will print to stderr)...", ITERATIONS / 1000);
    
    let start = Instant::now();
    for i in 0..(ITERATIONS / 1000) {  // Reduced iterations to avoid spamming
        let t = telemetry::now_ns();
        telemetry::record_ms("test", t);
        if i == 0 {
            // Capture first output time
        }
    }
    let record_ms_total = start.elapsed();
    let record_ms_per_call = record_ms_total.as_nanos() as f64 / (ITERATIONS / 1000) as f64;
    
    println!("\nTotal time:      {:.3} ms", record_ms_total.as_secs_f64() * 1000.0);
    println!("Per call:        {:.3} ns", record_ms_per_call);
    println!("vs Baseline:     {:.2}x", record_ms_per_call / baseline_per_call);
    println!("⚠️  Note: Includes stderr I/O overhead (~1000-5000ns per call)");
    println!();
    
    // ===================================================================
    // Test 6: Measure actual frame processing overhead
    // ===================================================================
    println!("Test 6: Simulated frame processing with telemetry");
    println!("──────────────────────────────────────────────────");
    
    // Simulate processing without telemetry
    let start = Instant::now();
    for _ in 0..1000 {
        // Simulate work: 1ms processing
        std::thread::sleep(std::time::Duration::from_micros(1000));
    }
    let without_telemetry = start.elapsed();
    
    // Simulate processing with telemetry (silent - no output)
    let start = Instant::now();
    for _ in 0..1000 {
        let t0 = telemetry::now_ns();
        // Simulate work: 1ms processing
        std::thread::sleep(std::time::Duration::from_micros(1000));
        let _ms = telemetry::since_ms(t0);  // Calculate but don't output
    }
    let with_telemetry = start.elapsed();
    
    let overhead = (with_telemetry - without_telemetry).as_nanos() as f64 / 1000.0;
    let percentage = (overhead / (without_telemetry.as_nanos() as f64 / 1000.0)) * 100.0;
    
    println!("Without telemetry: {:.3} ms", without_telemetry.as_secs_f64() * 1000.0);
    println!("With telemetry:    {:.3} ms", with_telemetry.as_secs_f64() * 1000.0);
    println!("Overhead per call: {:.3} ns", overhead);
    println!("Percentage:        {:.4}%", percentage);
    println!();
    
    // ===================================================================
    // Summary Table
    // ===================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("┌────────────────────────────────┬──────────────┬───────────────┐");
    println!("│ Operation                      │ Time (ns)    │ vs Baseline   │");
    println!("├────────────────────────────────┼──────────────┼───────────────┤");
    println!("│ std::time::Instant::now()      │ {:>12.2} │      1.00x    │", baseline_per_call);
    println!("│ Instant::elapsed()             │ {:>12.2} │ {:>10.2}x    │", elapsed_per_call, elapsed_per_call / baseline_per_call);
    println!("│ telemetry::now_ns()            │ {:>12.2} │ {:>10.2}x    │", now_ns_per_call, now_ns_per_call / baseline_per_call);
    println!("│ telemetry::since_ms()          │ {:>12.2} │ {:>10.2}x    │", since_ms_per_call, since_ms_per_call / baseline_per_call);
    println!("│ telemetry::record_ms() (I/O)   │ {:>12.2} │ {:>10.2}x    │", record_ms_per_call, record_ms_per_call / baseline_per_call);
    println!("└────────────────────────────────┴──────────────┴───────────────┘");
    println!();
    
    // ===================================================================
    // Interpretation
    // ===================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("INTERPRETATION");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("✅ Pure timing overhead (now_ns + since_ms):");
    println!("   ~{:.2} ns per measurement", now_ns_per_call + since_ms_per_call);
    println!("   = {:.6} ms per measurement", (now_ns_per_call + since_ms_per_call) / 1_000_000.0);
    println!();
    println!("⚠️  With stderr output (record_ms):");
    println!("   ~{:.2} ns per measurement", record_ms_per_call);
    println!("   = {:.6} ms per measurement", record_ms_per_call / 1_000_000.0);
    println!();
    println!("📊 For real-time video processing (60 fps = 16.67ms budget):");
    println!("   • Pure timing: {:.4}% of frame budget", 
        ((now_ns_per_call + since_ms_per_call) / 1_000_000.0) / 16.67 * 100.0);
    println!("   • With logging to stderr: {:.4}% of frame budget",
        (record_ms_per_call / 1_000_000.0) / 16.67 * 100.0);
    println!();
    println!("💡 Recommendation:");
    println!("   • Use pure timing (now_ns/since_ms) in hot path: NEGLIGIBLE overhead");
    println!("   • Disable logging in production or log to buffer: ~{:.2} ns overhead", now_ns_per_call + since_ms_per_call);
    println!("   • Keep stderr logging for development: ~{:.2} ns overhead", record_ms_per_call);
    println!();
    
    // ===================================================================
    // References
    // ===================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("REFERENCES");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("1. std::time::Instant documentation:");
    println!("   https://doc.rust-lang.org/std/time/struct.Instant.html");
    println!("   • Monotonic clock (no backwards time jumps)");
    println!("   • Uses CLOCK_MONOTONIC on Linux");
    println!("   • Overhead: ~10-30ns per call (system dependent)");
    println!();
    println!("2. Linux CLOCK_MONOTONIC:");
    println!("   https://man7.org/linux/man-pages/man2/clock_gettime.2.html");
    println!("   • Resolution: nanoseconds");
    println!("   • Latency: 20-50ns (modern CPUs with vDSO)");
    println!();
    println!("3. Benchmarking methodology:");
    println!("   • Iteration count: {}", ITERATIONS);
    println!("   • CPU: {}", std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "Unknown".to_string()));
    println!("   • OS: {}", std::env::consts::OS);
    println!();
    
    println!("=== Benchmark Complete ===");
}
