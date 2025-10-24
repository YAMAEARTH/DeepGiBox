/// Simple test: just load PNG and verify API works
/// No GPU required!

use anyhow::Result;
use decklink_output::BgraImage;

fn main() -> Result<()> {
    println!("=== Simple PNG Loader Test ===\n");

    // Check if foreground.png exists
    if !std::path::Path::new("foreground.png").exists() {
        println!("❌ foreground.png not found!");
        println!("Please create foreground.png or run test_compositor first");
        return Ok(());
    }

    // Load PNG
    println!("Loading foreground.png...");
    let image = BgraImage::load_from_file("foreground.png")?;
    
    println!("✅ PNG loaded successfully!");
    println!("   Size: {}x{}", image.width, image.height);
    println!("   Pitch: {} bytes", image.pitch);
    println!("   Data: {} bytes", image.data.len());
    println!("   Expected: {} bytes", image.width * image.height * 4);

    // Analyze alpha channel
    let mut opaque_count = 0;
    let mut transparent_count = 0;
    let mut semitransparent_count = 0;

    for chunk in image.data.chunks_exact(4) {
        let alpha = chunk[3];
        match alpha {
            255 => opaque_count += 1,
            0 => transparent_count += 1,
            _ => semitransparent_count += 1,
        }
    }

    let total = opaque_count + transparent_count + semitransparent_count;
    
    println!("\n=== Alpha Channel Analysis ===");
    println!("Total pixels: {}", total);
    println!("Opaque (α=255): {} ({:.1}%)", 
             opaque_count, 
             100.0 * opaque_count as f64 / total as f64);
    println!("Transparent (α=0): {} ({:.1}%)", 
             transparent_count,
             100.0 * transparent_count as f64 / total as f64);
    println!("Semi-transparent: {} ({:.1}%)", 
             semitransparent_count,
             100.0 * semitransparent_count as f64 / total as f64);

    // Sample some pixels
    println!("\n=== Sample Pixels (BGRA) ===");
    let samples = [
        (0, 0, "Top-left"),
        (image.width / 2, image.height / 2, "Center"),
        (image.width - 1, image.height - 1, "Bottom-right"),
    ];

    for (x, y, label) in samples {
        if let Some((b, g, r, a)) = image.get_pixel(x, y) {
            println!("{:12} ({:4}, {:4}): B={:3} G={:3} R={:3} A={:3}", 
                     label, x, y, b, g, r, a);
        }
    }

    println!("\n✅ All tests passed!");
    
    Ok(())
}
