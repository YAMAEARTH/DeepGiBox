#!/usr/bin/env python3
"""
Visualize what the model actually sees after preprocessing
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image that was used
img_path = "apps/playgrounds/sample_img_4.png"
img = Image.open(img_path).convert('RGB')
print(f"Original image size: {img.size}")
print(f"Original image mode: {Image.open(img_path).mode}")

# Resize like the model does
img_resized = img.resize((640, 512))
img_array = np.array(img_resized, dtype=np.float32) / 255.0

print(f"\nPreprocessed image stats:")
print(f"  Shape: {img_array.shape}")
print(f"  Min: {img_array.min():.4f}")
print(f"  Max: {img_array.max():.4f}")
print(f"  Mean: {img_array.mean():.4f}")
print(f"  Std: {img_array.std():.4f}")

# Check for unusual patterns
print(f"\nChannel statistics:")
for i, channel in enumerate(['R', 'G', 'B']):
    ch_data = img_array[:, :, i]
    print(f"  {channel}: min={ch_data.min():.4f}, max={ch_data.max():.4f}, "
          f"mean={ch_data.mean():.4f}, std={ch_data.std():.4f}")

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original
axes[0, 0].imshow(img_resized)
axes[0, 0].set_title('Original Image (640x512)')
axes[0, 0].axis('off')

# Each channel
for i, (ax, channel, color) in enumerate(zip(axes[0, 1:], ['R', 'G', 'B'], ['Reds', 'Greens', 'Blues'])):
    ax.imshow(img_array[:, :, i], cmap=color, vmin=0, vmax=1)
    ax.set_title(f'{channel} Channel')
    ax.axis('off')

# Grayscale
gray = img_array.mean(axis=2)
axes[1, 0].imshow(gray, cmap='gray', vmin=0, vmax=1)
axes[1, 0].set_title('Grayscale (mean)')
axes[1, 0].axis('off')

# Histogram
axes[1, 1].hist(img_array[:, :, 0].flatten(), bins=50, alpha=0.5, label='R', color='red')
axes[1, 1].hist(img_array[:, :, 1].flatten(), bins=50, alpha=0.5, label='G', color='green')
axes[1, 1].hist(img_array[:, :, 2].flatten(), bins=50, alpha=0.5, label='B', color='blue')
axes[1, 1].set_title('Pixel Value Distribution')
axes[1, 1].set_xlabel('Pixel Value')
axes[1, 1].set_ylabel('Count')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Horizontal profile (average each row)
row_profile = img_array.mean(axis=(1, 2))
axes[1, 2].plot(row_profile)
axes[1, 2].set_title('Horizontal Profile\n(avg brightness per row)')
axes[1, 2].set_xlabel('Row (y)')
axes[1, 2].set_ylabel('Mean pixel value')
axes[1, 2].grid(alpha=0.3)
axes[1, 2].axhline(y=row_profile.mean(), color='r', linestyle='--', label=f'Mean: {row_profile.mean():.3f}')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('output/input_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Saved input analysis to: output/input_analysis.png")

# Check if there are horizontal bands in the input
print(f"\n🔍 Checking for horizontal banding in input...")
# Look at variance across rows
row_means = img_array.mean(axis=(1, 2))
row_variance = np.var(row_means)
print(f"  Row mean variance: {row_variance:.6f}")
if row_variance > 0.01:
    print(f"  ⚠️  High variance detected - image has horizontal banding in brightness!")
else:
    print(f"  ✅ Low variance - image is relatively uniform")
