use anyhow::Result;
use common_io::Stage;
fn main()->Result<()>{
    let mut cap = decklink_input::from_path("configs/dev_1080p60_yuv422_fp16_trt.toml")?;
    let mut pre = preprocess_cuda::from_path("configs/dev_1080p60_yuv422_fp16_trt.toml")?;
    let raw = cap.process(());
    let t = pre.process(raw);
    println!("tensor desc: 1x{}x{} dtype=Fp16 dev={}", t.desc.h, t.desc.w, t.desc.device);
    Ok(())
}
