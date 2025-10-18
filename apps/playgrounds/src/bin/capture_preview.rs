use anyhow::Result;
use common_io::{ColorSpace, PixelFormat, RawFramePacket};
fn main() -> Result<()> {
    let mut cap = decklink_input::DeckLinkSource::new(
        0,
        "1080p59.94",
        PixelFormat::YUV422_8,
        ColorSpace::BT709,
    )?;
    for i in 0..5 {
        let f: RawFramePacket = cap.next_frame()?;
        println!(
            "#{i} {}x{} fmt={:?} stride={} pts={}",
            f.meta.width, f.meta.height, f.meta.pixfmt, f.meta.stride_bytes, f.meta.pts_ns
        );
    }
    Ok(())
}
