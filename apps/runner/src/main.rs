use anyhow::Result;
use common_io::Stage;
fn main()->Result<()>{
    let cfg_path = std::env::args().skip_while(|a| a != "--config").nth(1)
        .unwrap_or_else(|| "configs/dev_1080p60_yuv422_fp16_trt.toml".into());
    let mut cap   = decklink_input::from_path(&cfg_path)?;
    let mut pre   = preprocess_cuda::from_path(&cfg_path)?;
    let mut infer = inference::from_path(&cfg_path)?;
    let mut post  = postprocess::from_path(&cfg_path)?;
    let mut plan  = overlay_plan::from_path(&cfg_path)?;
    let mut rend  = overlay_render::from_path(&cfg_path)?;
    let mut out   = decklink_output::from_path(&cfg_path)?;
    let raw  = telemetry::time_stage("capture",    &mut cap,   ());
    let tin  = telemetry::time_stage("preprocess", &mut pre,   raw);
    let rdet = telemetry::time_stage("inference",  &mut infer, tin);
    let dets = telemetry::time_stage("postprocess",&mut post,  rdet);
    let ops  = telemetry::time_stage("overlay_plan",&mut plan, dets);
    let ofrm = telemetry::time_stage("overlay_render",&mut rend, ops);
    out.submit(ofrm)?;
    Ok(())
}
