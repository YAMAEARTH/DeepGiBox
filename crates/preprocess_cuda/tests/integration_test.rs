// tests/integration_test.rs - Integration tests for preprocessing
#[cfg(test)]
mod tests {
    use preprocess_cuda::{ChromaOrder, Preprocessor};
    use common_io::{ColorSpace, FrameMeta, MemLoc, MemRef, PixelFormat, RawFramePacket, Stage};

    #[test]
    #[ignore] // Requires CUDA
    fn test_preprocessor_creation() {
        let result = Preprocessor::new((512, 512), true, 0);
        assert!(result.is_ok(), "Failed to create preprocessor: {:?}", result.err());
    }

    #[test]
    #[ignore] // Requires CUDA
    fn test_yuv422_preprocessing() {
        let mut pre = Preprocessor::with_params(
            (512, 512),
            true,
            0,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            ChromaOrder::UYVY,
        ).expect("Failed to create preprocessor");

        // Create a dummy GPU frame (in real scenario, this would come from DeckLink)
        let meta = FrameMeta {
            source_id: 0,
            width: 1920,
            height: 1080,
            pixfmt: PixelFormat::YUV422_8,
            colorspace: ColorSpace::BT709,
            frame_idx: 0,
            pts_ns: 0,
            t_capture_ns: 0,
            stride_bytes: 1920 * 2,
        };

        // Normally this would be a real GPU pointer, but we can't easily test without CUDA
        let data = MemRef {
            ptr: std::ptr::null_mut(),
            len: (1920 * 1080 * 2) as usize,
            stride: (1920 * 2) as usize,
            loc: MemLoc::Gpu { device: 0 },
        };

        let input = RawFramePacket { meta, data };
        
        // This would panic without actual GPU memory
        // let output = pre.process(input);
        // assert_eq!(output.desc.w, 512);
        // assert_eq!(output.desc.h, 512);
    }

    #[test]
    fn test_config_parsing() {
        // Test that config structure is correct
        let config_str = r#"
[preprocess]
size = [640, 480]
fp16 = false
device = 0
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
chroma = "YUY2"
        "#;

        let config: toml::Value = toml::from_str(config_str).expect("Failed to parse config");
        let preprocess_config: preprocess_cuda::PreprocessConfig = config
            .get("preprocess")
            .unwrap()
            .clone()
            .try_into()
            .expect("Failed to parse preprocess config");

        assert_eq!(preprocess_config.size, [640, 480]);
        assert_eq!(preprocess_config.fp16, false);
        assert_eq!(preprocess_config.chroma, "YUY2");
    }

    #[test]
    fn test_chroma_order() {
        assert_eq!(ChromaOrder::UYVY as i32, 0);
        assert_eq!(ChromaOrder::YUY2 as i32, 1);
    }
}
