#!/usr/bin/env python3
"""
Visualization tool for preprocessing output dump
Converts binary tensor data to images for inspection
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def denormalize_imagenet(tensor):
    """
    Denormalize ImageNet-normalized tensor back to [0, 1] range
    
    Args:
        tensor: numpy array with ImageNet normalization applied
                (pixel - mean) / std where mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    
    Returns:
        Denormalized tensor in [0, 1] range
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    # Reverse: pixel = (normalized * std) + mean
    denorm = (tensor * std) + mean
    
    # Clip to [0, 1] for valid RGB
    denorm = np.clip(denorm, 0, 1)
    
    return denorm

def load_tensor(bin_path, dtype=np.float16):
    """Load binary tensor file"""
    data = np.fromfile(bin_path, dtype=dtype)
    
    # Reshape to NCHW (1, 3, 512, 512)
    tensor = data.reshape(1, 3, 512, 512)
    
    # Convert to float32 for better compatibility with matplotlib
    tensor = tensor.astype(np.float32)
    
    return tensor

def visualize_channels(tensor, title="Preprocessing Output", save_path=None):
    """Visualize RGB channels separately and combined"""
    
    # Extract single image (remove batch dimension)
    img_nchw = tensor[0]  # (3, 512, 512)
    
    # Denormalize from ImageNet normalization
    img_denorm = denormalize_imagenet(img_nchw)
    
    # Convert NCHW -> HWC for display
    img_hwc = np.transpose(img_denorm, (1, 2, 0))  # (512, 512, 3)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Show individual channels
    channel_names = ['Red Channel', 'Green Channel', 'Blue Channel']
    for i, (ax, name) in enumerate(zip(axes[0], channel_names)):
        ax.imshow(img_denorm[i], cmap='gray', vmin=0, vmax=1)
        ax.set_title(name)
        ax.axis('off')
        
        # Add statistics
        stats_text = f"Min: {img_denorm[i].min():.3f}\nMax: {img_denorm[i].max():.3f}\nMean: {img_denorm[i].mean():.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Show combined RGB image
    axes[1, 0].imshow(img_hwc)
    axes[1, 0].set_title('Combined RGB Image')
    axes[1, 0].axis('off')
    
    # Show normalized data (original format for inference)
    axes[1, 1].imshow(np.transpose(img_nchw, (1, 2, 0)))
    axes[1, 1].set_title('Normalized (as fed to model)')
    axes[1, 1].axis('off')
    
    # Show histogram
    ax_hist = axes[1, 2]
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        ax_hist.hist(img_denorm[i].flatten(), bins=50, alpha=0.5, 
                     color=color, label=channel_names[i])
    ax_hist.set_xlabel('Pixel Value')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title('Pixel Value Distribution')
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization: {save_path}")
    
    plt.show()

def analyze_tensor(tensor):
    """Print detailed analysis of tensor"""
    img = tensor[0]  # Remove batch dimension
    
    print("\n" + "="*60)
    print("TENSOR ANALYSIS")
    print("="*60)
    
    print(f"\nShape: {tensor.shape} (NCHW format)")
    print(f"Data type: {tensor.dtype}")
    print(f"Total elements: {tensor.size:,}")
    print(f"Memory size: {tensor.nbytes / (1024*1024):.2f} MB")
    
    print("\n" + "-"*60)
    print("NORMALIZED DATA (as fed to model)")
    print("-"*60)
    channel_names = ['Red', 'Green', 'Blue']
    for i, name in enumerate(channel_names):
        ch = img[i]
        print(f"\n{name} Channel:")
        print(f"  Min:  {ch.min():.6f}")
        print(f"  Max:  {ch.max():.6f}")
        print(f"  Mean: {ch.mean():.6f}")
        print(f"  Std:  {ch.std():.6f}")
    
    # Denormalize for human interpretation
    img_denorm = denormalize_imagenet(img)
    
    print("\n" + "-"*60)
    print("DENORMALIZED DATA (human-readable RGB)")
    print("-"*60)
    print("(After reversing ImageNet normalization)")
    for i, name in enumerate(channel_names):
        ch = img_denorm[i]
        print(f"\n{name} Channel:")
        print(f"  Min:  {ch.min():.6f}")
        print(f"  Max:  {ch.max():.6f}")
        print(f"  Mean: {ch.mean():.6f}")
        print(f"  Std:  {ch.std():.6f}")
    
    # Check for common issues
    print("\n" + "-"*60)
    print("QUALITY CHECKS")
    print("-"*60)
    
    # Check for constant regions
    for i, name in enumerate(channel_names):
        ch = img_denorm[i]
        unique_vals = len(np.unique(ch))
        if unique_vals < 100:
            print(f"⚠️  {name}: Only {unique_vals} unique values (might be too uniform)")
        else:
            print(f"✓ {name}: {unique_vals} unique values")
    
    # Check brightness
    brightness = img_denorm.mean()
    if brightness < 0.2:
        print(f"⚠️  Image is very dark (brightness: {brightness:.3f})")
    elif brightness > 0.8:
        print(f"⚠️  Image is very bright (brightness: {brightness:.3f})")
    else:
        print(f"✓ Brightness looks normal ({brightness:.3f})")
    
    # Check contrast
    contrast = img_denorm.std()
    if contrast < 0.1:
        print(f"⚠️  Low contrast (std: {contrast:.3f})")
    else:
        print(f"✓ Contrast looks good (std: {contrast:.3f})")
    
    print("\n" + "="*60)

def save_as_image(tensor, output_path):
    """Save tensor as PNG image"""
    img = tensor[0]  # Remove batch dimension
    img_denorm = denormalize_imagenet(img)
    img_hwc = np.transpose(img_denorm, (1, 2, 0))
    
    # Convert to uint8 for saving
    img_uint8 = (img_hwc * 255).astype(np.uint8)
    
    plt.imsave(output_path, img_uint8)
    print(f"✓ Saved image: {output_path}")

def main():
    """Main visualization routine"""
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║     Preprocessing Output Visualization Tool             ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    
    # Find the most recent dump file
    bin_files = list(Path('.').glob('preprocessing_dump_frame_*_fp16.bin'))
    
    if not bin_files:
        print("❌ No dump files found!")
        print("   Run: cargo run --release -p playgrounds --bin dump_preprocessing_output")
        return
    
    # Use the most recent file
    bin_path = sorted(bin_files)[-1]
    frame_num = bin_path.stem.split('_')[3]  # Extract frame number
    
    print(f"📁 Loading: {bin_path.name}")
    print(f"   Frame: #{frame_num}")
    
    # Load tensor
    tensor = load_tensor(bin_path, dtype=np.float16)
    print(f"   ✓ Loaded tensor: {tensor.shape}")
    
    # Analyze
    analyze_tensor(tensor)
    
    # Save denormalized image
    img_path = f"preprocessing_frame_{frame_num}_denormalized.png"
    save_as_image(tensor, img_path)
    
    # Visualize
    viz_path = f"preprocessing_frame_{frame_num}_analysis.png"
    print(f"\n📊 Creating visualization...")
    visualize_channels(tensor, 
                      title=f"Preprocessing Output - Frame #{frame_num}",
                      save_path=viz_path)
    
    print("\n" + "="*60)
    print("📖 INTERPRETATION GUIDE")
    print("="*60)
    print("""
1. NORMALIZED DATA (for model):
   - Values typically in range [-3, 3] after ImageNet normalization
   - Mean close to 0, Std close to 1 per channel
   - This is what the inference model sees

2. DENORMALIZED DATA (for humans):
   - Values in range [0, 1] (or [0, 255] for uint8)
   - This is the actual RGB image content
   - Use this to verify preprocessing quality

3. WHAT TO LOOK FOR:
   - ✓ Clear image details (not blurry)
   - ✓ Correct colors (not tinted green/magenta)
   - ✓ Good contrast (not washed out)
   - ✓ Proper brightness
   - ⚠️  Dark corners = vignetting (might be lens)
   - ⚠️  Color cast = wrong color space (BT.709 vs BT.601)
   - ⚠️  Blocky = chroma subsampling artifacts

4. FILES GENERATED:
   - {img_path} - Clean RGB image
   - {viz_path} - Full analysis with channels
    """)
    
    print("✨ Visualization complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
