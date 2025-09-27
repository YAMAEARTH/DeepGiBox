use anyhow::Result;
use common_io::Stage;
fn main()->Result<()>{
    let mut cap   = decklink_input::from_path("configs/dev_1080p60_yuv422_fp16_trt.toml")?;
    let mut pre   = preprocess_cuda::from_path("configs/dev_1080p60_yuv422_fp16_trt.toml")?;
    let mut infer = inference::from_path("configs/dev_1080p60_yuv422_fp16_trt.toml")?;
    let mut post  = postprocess::from_path("configs/dev_1080p60_yuv422_fp16_trt.toml")?;
    let raw  = cap.process(());
    let tin  = pre.process(raw);
    let rdet = infer.process(tin);
    let dets = post.process(rdet);
    println!("detections: {}", dets.items.len());
    Ok(())
}
