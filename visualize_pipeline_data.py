#!/usr/bin/env python3
"""
Visualize Pipeline Data: Preprocessing Tensor & Overlay Render
แปลง preprocessing tensor และ BGRA buffer เป็นรูปภาพ PNG
"""

import numpy as np
import cv2
import os
import sys
from pathlib import Path
import struct

def visualize_preprocessing_tensor(frame_num=0):
    """
    แปลง preprocessing tensor (FP16) เป็นภาพ
    Tensor format: [N=1, C=3, H=640, W=640] FP16
    """
    print(f"\n📊 Visualizing Preprocessing Tensor (Frame {frame_num})...")
    
    # Read tensor info from text file
    txt_file = f"output/test/debug_frame_{frame_num:04d}_preprocessing.txt"
    if not os.path.exists(txt_file):
        print(f"  ❌ File not found: {txt_file}")
        return None
    
    # Parse tensor shape from text file
    with open(txt_file, 'r') as f:
        content = f.read()
        if 'Shape:' in content:
            # Extract N, C, H, W
            for line in content.split('\n'):
                if 'Shape:' in line:
                    print(f"  📝 {line.strip()}")
                    # Parse: Shape: [N=1, C=3, H=640, W=640]
                    import re
                    match = re.search(r'N=(\d+).*C=(\d+).*H=(\d+).*W=(\d+)', line)
                    if match:
                        n, c, h, w = map(int, match.groups())
                        print(f"  ✓ Parsed shape: N={n}, C={c}, H={h}, W={w}")
    
    # Note: เราไม่มี raw tensor data เพราะมันอยู่บน GPU
    # แต่เราสามารถแสดงข้อมูลจาก input frame แทน
    print(f"  ℹ️  Tensor data is on GPU memory (not dumped)")
    print(f"  💡 Suggestion: Use raw YUV422 frame to visualize input")
    
    return None

def visualize_bgra_buffer(frame_num=0):
    """
    แปลง BGRA buffer เป็นภาพ PNG
    Format: BGRA8 (4 bytes per pixel)
    """
    print(f"\n🎨 Visualizing BGRA Overlay Buffer (Frame {frame_num})...")
    
    bin_file = f"output/test/debug_frame_{frame_num:04d}_overlay_bgra.bin"
    if not os.path.exists(bin_file):
        print(f"  ❌ File not found: {bin_file}")
        return None
    
    # Get file size
    file_size = os.path.getsize(bin_file)
    print(f"  📁 File size: {file_size:,} bytes ({file_size/(1024*1024):.1f} MB)")
    
    # Calculate dimensions
    # Common resolutions: 1920x1080, 3840x2160
    possible_sizes = [
        (3840, 2160, "4K"),
        (1920, 1080, "Full HD"),
        (1280, 720, "HD"),
    ]
    
    width, height = None, None
    for w, h, name in possible_sizes:
        expected_size = w * h * 4  # BGRA = 4 bytes per pixel
        if file_size == expected_size:
            width, height = w, h
            print(f"  ✓ Detected resolution: {w}x{h} ({name})")
            break
    
    if width is None:
        print(f"  ⚠️  Unknown resolution, trying to guess...")
        # Try to calculate from file size
        pixels = file_size // 4
        # Assume 16:9 aspect ratio
        width = int(np.sqrt(pixels * 16 / 9))
        height = int(width * 9 / 16)
        print(f"  ⚠️  Guessed: {width}x{height}")
    
    # Read BGRA data
    print(f"  📖 Reading BGRA data...")
    with open(bin_file, 'rb') as f:
        bgra_data = np.frombuffer(f.read(), dtype=np.uint8)
    
    # Reshape to image
    try:
        bgra_image = bgra_data.reshape((height, width, 4))
        print(f"  ✓ Reshaped to: {bgra_image.shape}")
    except ValueError as e:
        print(f"  ❌ Reshape failed: {e}")
        return None
    
    # Statistics
    non_zero = np.count_nonzero(bgra_data)
    total = bgra_data.size
    print(f"  📊 Non-zero bytes: {non_zero:,} / {total:,} ({100*non_zero/total:.2f}%)")
    
    # Convert BGRA to RGBA for saving
    # OpenCV uses BGR, PIL uses RGB
    b, g, r, a = cv2.split(bgra_image)
    rgba_image = cv2.merge([r, g, b, a])
    
    # Save as PNG
    output_file = f"output/test/frame_{frame_num:04d}_overlay_bgra.png"
    cv2.imwrite(output_file, cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA))
    print(f"  ✅ Saved to: {output_file}")
    
    # Also create version with white background (for easier viewing)
    # Create white background
    white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Alpha blend
    alpha = a.astype(float) / 255.0
    alpha = np.stack([alpha, alpha, alpha], axis=2)
    
    rgb_image = cv2.merge([r, g, b])
    blended = (rgb_image * alpha + white_bg * (1 - alpha)).astype(np.uint8)
    
    output_file_bg = f"output/test/frame_{frame_num:04d}_overlay_on_white.png"
    cv2.imwrite(output_file_bg, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    print(f"  ✅ Saved (with white background): {output_file_bg}")
    
    # Create version with black background
    black_bg = np.zeros((height, width, 3), dtype=np.uint8)
    blended_black = (rgb_image * alpha + black_bg * (1 - alpha)).astype(np.uint8)
    
    output_file_black = f"output/test/frame_{frame_num:04d}_overlay_on_black.png"
    cv2.imwrite(output_file_black, cv2.cvtColor(blended_black, cv2.COLOR_RGB2BGR))
    print(f"  ✅ Saved (with black background): {output_file_black}")
    
    return rgba_image

def visualize_raw_yuv422(frame_num=0):
    """
    แปลง raw YUV422 frame เป็นภาพ PNG
    """
    print(f"\n📹 Visualizing Raw YUV422 Frame (Frame {frame_num})...")
    
    bin_file = f"output/test/debug_frame_{frame_num:04d}_raw_yuv422.bin"
    if not os.path.exists(bin_file):
        print(f"  ❌ File not found: {bin_file}")
        return None
    
    # Get file size
    file_size = os.path.getsize(bin_file)
    print(f"  📁 File size: {file_size:,} bytes ({file_size/(1024*1024):.1f} MB)")
    
    # Calculate dimensions for YUV422
    # YUV422 = 2 bytes per pixel (Y=1byte, U+V=1byte for 2 pixels)
    possible_sizes = [
        (3840, 2160, "4K"),
        (1920, 1080, "Full HD"),
        (1280, 720, "HD"),
    ]
    
    width, height = None, None
    for w, h, name in possible_sizes:
        expected_size = w * h * 2  # YUV422 = 2 bytes per pixel
        if file_size == expected_size:
            width, height = w, h
            print(f"  ✓ Detected resolution: {w}x{h} ({name})")
            break
    
    if width is None:
        print(f"  ⚠️  Unknown resolution")
        return None
    
    # Read YUV422 data
    print(f"  📖 Reading YUV422 data...")
    with open(bin_file, 'rb') as f:
        yuv_data = np.frombuffer(f.read(), dtype=np.uint8)
    
    print(f"  🔄 Converting YUV422 to RGB...")
    # YUV422 format: YUYV (Y0 U0 Y1 V0)
    # Need to convert to YUV444 first, then to RGB
    
    try:
        # Reshape to YUYV pairs
        yuyv = yuv_data.reshape((height, width * 2))
        
        # Extract Y, U, V
        y = yuyv[:, 0::2].reshape(height, width)
        u = yuyv[:, 1::4].repeat(2, axis=1).reshape(height, width)
        v = yuyv[:, 3::4].repeat(2, axis=1).reshape(height, width)
        
        # Convert YUV to RGB
        yuv_image = cv2.merge([y, u, v])
        rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)
        
        # Save as PNG
        output_file = f"output/test/frame_{frame_num:04d}_input_raw.png"
        cv2.imwrite(output_file, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        print(f"  ✅ Saved to: {output_file}")
        
        # Also save a thumbnail
        thumb = cv2.resize(rgb_image, (960, 540))
        output_thumb = f"output/test/frame_{frame_num:04d}_input_raw_thumb.png"
        cv2.imwrite(output_thumb, cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
        print(f"  ✅ Saved thumbnail: {output_thumb}")
        
        return rgb_image
        
    except Exception as e:
        print(f"  ❌ Conversion failed: {e}")
        return None

def create_comparison_image(frame_num=0):
    """
    สร้างภาพเปรียบเทียบ input vs overlay
    """
    print(f"\n🖼️  Creating Comparison Image (Frame {frame_num})...")
    
    # Load images if they exist
    input_file = f"output/test/frame_{frame_num:04d}_input_raw.png"
    overlay_file = f"output/test/frame_{frame_num:04d}_overlay_on_black.png"
    
    if not os.path.exists(input_file):
        print(f"  ⚠️  Input image not found: {input_file}")
        return
    
    if not os.path.exists(overlay_file):
        print(f"  ⚠️  Overlay image not found: {overlay_file}")
        return
    
    input_img = cv2.imread(input_file)
    overlay_img = cv2.imread(overlay_file)
    
    # Resize overlay to match input if needed
    if input_img.shape != overlay_img.shape:
        overlay_img = cv2.resize(overlay_img, (input_img.shape[1], input_img.shape[0]))
    
    # Create side-by-side comparison
    comparison = np.hstack([input_img, overlay_img])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Input (Raw YUV422)", (50, 100), font, 3, (255, 255, 255), 4)
    cv2.putText(comparison, "Overlay (BGRA)", (input_img.shape[1] + 50, 100), font, 3, (255, 255, 255), 4)
    
    output_file = f"output/test/frame_{frame_num:04d}_comparison.png"
    cv2.imwrite(output_file, comparison)
    print(f"  ✅ Saved comparison: {output_file}")
    
    # Create thumbnail
    thumb = cv2.resize(comparison, (comparison.shape[1]//2, comparison.shape[0]//2))
    output_thumb = f"output/test/frame_{frame_num:04d}_comparison_thumb.png"
    cv2.imwrite(output_thumb, thumb)
    print(f"  ✅ Saved thumbnail: {output_thumb}")

def main():
    print("=" * 70)
    print("🎨 Pipeline Data Visualization")
    print("=" * 70)
    
    # Create output directory
    os.makedirs("output/test", exist_ok=True)
    
    # Get list of frames to process
    frames_to_process = []
    for i in range(5):
        bgra_file = f"output/test/debug_frame_{i:04d}_overlay_bgra.bin"
        if os.path.exists(bgra_file):
            frames_to_process.append(i)
    
    if not frames_to_process:
        print("\n❌ No debug files found!")
        print("   Please run the pipeline first to generate debug files.")
        print("   Expected files: output/test/debug_frame_0000_overlay_bgra.bin")
        return
    
    print(f"\n✓ Found {len(frames_to_process)} frame(s) to process: {frames_to_process}")
    
    # Process each frame
    for frame_num in frames_to_process:
        print("\n" + "=" * 70)
        print(f"Processing Frame {frame_num}")
        print("=" * 70)
        
        # 1. Visualize preprocessing tensor (info only)
        visualize_preprocessing_tensor(frame_num)
        
        # 2. Visualize raw input frame (if available)
        visualize_raw_yuv422(frame_num)
        
        # 3. Visualize BGRA overlay
        visualize_bgra_buffer(frame_num)
        
        # 4. Create comparison image
        create_comparison_image(frame_num)
    
    print("\n" + "=" * 70)
    print("✅ Visualization Complete!")
    print("=" * 70)
    print(f"\n📂 Output files in: output/test/")
    print("\nGenerated files per frame:")
    print("  • frame_XXXX_input_raw.png          - Input frame (YUV422 → RGB)")
    print("  • frame_XXXX_input_raw_thumb.png    - Input thumbnail (960x540)")
    print("  • frame_XXXX_overlay_bgra.png       - Overlay with transparency")
    print("  • frame_XXXX_overlay_on_white.png   - Overlay on white background")
    print("  • frame_XXXX_overlay_on_black.png   - Overlay on black background")
    print("  • frame_XXXX_comparison.png         - Side-by-side comparison")
    print("  • frame_XXXX_comparison_thumb.png   - Comparison thumbnail")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
