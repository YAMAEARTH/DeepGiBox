//! Unit tests for inference engine

use super::*;
use config::InferenceCfg;

#[test]
fn test_inference_config_default() {
    let cfg = InferenceCfg::default();
    assert_eq!(cfg.backend, "onnxruntime_trt");
    assert_eq!(cfg.device, 0);
    assert_eq!(cfg.fp16, true);
    assert_eq!(cfg.warmup_runs, 5);
    assert_eq!(cfg.max_workspace_mb, 2048);
}

#[test]
fn test_engine_creation_requires_model() {
    // This test would require an actual model file
    // Skipping for now as it's an integration test
    // Real tests should be in a separate integration test suite
}

// Note: Full integration tests with actual models should be
// in tests/ directory or as part of playground binaries
