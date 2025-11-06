#!/usr/bin/env python3
"""
Test GPU Overlay Rendering - Download buffer and check if overlay is drawn
"""
import numpy as np
import struct

def analyze_argb_buffer(filename):
    """Analyze ARGB buffer from GPU overlay rendering"""
    with open(filename, 'rb') as f:
        data = f.read()
    
    # 3840x2160 ARGB = 33177600 bytes
    expected_size = 3840 * 2160 * 4
    if len(data) != expected_size:
        print(f"❌ Wrong size: got {len(data)}, expected {expected_size}")
        return
    
    # Convert to numpy array (ARGB format)
    buf = np.frombuffer(data, dtype=np.uint8).reshape((2160, 3840, 4))
    
    # Extract channels
    alpha = buf[:, :, 0]
    red = buf[:, :, 1]
    green = buf[:, :, 2]
    blue = buf[:, :, 3]
    
    # Statistics
    print(f"\n📊 ARGB Buffer Statistics:")
    print(f"  Alpha channel:")
    print(f"    Non-zero pixels: {np.count_nonzero(alpha)} / {alpha.size} ({np.count_nonzero(alpha)/alpha.size*100:.2f}%)")
    print(f"    Min: {alpha.min()}, Max: {alpha.max()}, Mean: {alpha.mean():.2f}")
    
    print(f"  Red channel:")
    print(f"    Non-zero pixels: {np.count_nonzero(red)} / {red.size} ({np.count_nonzero(red)/red.size*100:.2f}%)")
    print(f"    Min: {red.min()}, Max: {red.max()}, Mean: {red.mean():.2f}")
    
    print(f"  Green channel:")
    print(f"    Non-zero pixels: {np.count_nonzero(green)} / {green.size} ({np.count_nonzero(green)/green.size*100:.2f}%)")
    print(f"    Min: {green.min()}, Max: {green.max()}, Mean: {green.mean():.2f}")
    
    print(f"  Blue channel:")
    print(f"    Non-zero pixels: {np.count_nonzero(blue)} / {blue.size} ({np.count_nonzero(blue)/blue.size*100:.2f}%)")
    print(f"    Min: {blue.min()}, Max: {blue.max()}, Mean: {blue.mean():.2f}")
    
    # Check for actual drawing
    opaque_pixels = np.sum(alpha == 255)
    semi_transparent = np.sum((alpha > 0) & (alpha < 255))
    transparent = np.sum(alpha == 0)
    
    print(f"\n🎨 Overlay Rendering Check:")
    print(f"  Fully opaque pixels (alpha=255): {opaque_pixels} ({opaque_pixels/alpha.size*100:.3f}%)")
    print(f"  Semi-transparent pixels (0 < alpha < 255): {semi_transparent} ({semi_transparent/alpha.size*100:.3f}%)")
    print(f"  Fully transparent pixels (alpha=0): {transparent} ({transparent/alpha.size*100:.3f}%)")
    
    if opaque_pixels > 10000:  # มากกว่า 10k pixels
        print(f"  ✅ Overlay appears to be drawn! ({opaque_pixels} opaque pixels)")
    else:
        print(f"  ❌ Overlay NOT drawn (only {opaque_pixels} opaque pixels)")
    
    # Find regions with high alpha
    if opaque_pixels > 0:
        print(f"\n📍 Opaque regions (sample):")
        opaque_mask = (alpha == 255)
        y_coords, x_coords = np.where(opaque_mask)
        if len(y_coords) > 0:
            # Sample first 10 opaque pixels
            for i in range(min(10, len(y_coords))):
                y, x = y_coords[i], x_coords[i]
                a, r, g, b = buf[y, x]
                print(f"    Pixel ({x}, {y}): ARGB=({a}, {r}, {g}, {b})")

if __name__ == '__main__':
    import sys
    import glob
    
    # Find GPU overlay dumps
    files = glob.glob('output/test/*overlay*gpu*.bin')
    if not files:
        print("❌ No GPU overlay dump files found")
        print("   Run pipeline and wait for dumps to be created")
        sys.exit(1)
    
    # Analyze latest file
    latest = sorted(files)[-1]
    print(f"📂 Analyzing: {latest}\n")
    analyze_argb_buffer(latest)
