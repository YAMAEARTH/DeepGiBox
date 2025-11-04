#!/usr/bin/env python3
"""
Visualize BiSeNet segmentation output with CamVid color scheme
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# CamVid dataset 12-class color palette (RGB format)
CAMVID_COLORS = [
    (128, 128, 128),  # 0:  Sky
    (128, 0, 0),      # 1:  Building
    (192, 192, 128),  # 2:  Pole
    (128, 64, 128),   # 3:  Road
    (0, 0, 192),      # 4:  Pavement
    (128, 128, 0),    # 5:  Tree
    (192, 128, 128),  # 6:  SignSymbol
    (64, 64, 128),    # 7:  Fence
    (64, 0, 128),     # 8:  Car
    (64, 64, 0),      # 9:  Pedestrian
    (0, 128, 192),    # 10: Bicyclist
    (0, 0, 0),        # 11: Void/Unlabeled
]

CAMVID_LABELS = [
    "Sky",
    "Building", 
    "Pole",
    "Road",
    "Pavement",
    "Tree",
    "SignSymbol",
    "Fence",
    "Car",
    "Pedestrian",
    "Bicyclist",
    "Void",
]

def apply_color_map(segmentation_mask, height, width, num_classes=12):
    """Convert class indices to RGB color map"""
    color_map = np.zeros((height, width, 3), dtype=np.uint8)
    
    if num_classes == 2:
        # GIM model: class 0 = background (black), class 255 = foreground (red)
        background_mask = (segmentation_mask == 0)
        foreground_mask = (segmentation_mask == 255)
        color_map[background_mask] = [0, 0, 0]  # Black for background
        color_map[foreground_mask] = [255, 0, 0]  # Red for foreground
    else:
        # BiSeNet/CamVid: use predefined colors for 12 classes
        for class_id in range(12):
            mask = (segmentation_mask == class_id)
            color_map[mask] = CAMVID_COLORS[class_id]
    
    return color_map

def create_legend_image():
    """Create a legend showing all classes and their colors"""
    fig, ax = plt.subplots(figsize=(3, 6))
    ax.axis('off')
    
    # Create color patches
    legend_elements = []
    for i, (color, label) in enumerate(zip(CAMVID_COLORS, CAMVID_LABELS)):
        color_norm = tuple(c / 255.0 for c in color)
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color_norm, label=label))
    
    ax.legend(handles=legend_elements, loc='center', fontsize=10, frameon=True)
    
    return fig

def visualize_segmentation(json_path, output_dir="output"):
    """Main visualization function"""
    print("🎨 BiSeNet Segmentation Visualization")
    print("=" * 60)
    
    # Load JSON data
    print(f"\n📂 Loading: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    image_path = data['image']['path']
    orig_w, orig_h = data['image']['original_size']
    model_w, model_h = data['image']['model_size']
    num_classes = data['num_classes']
    output_shape = data['output_shape']
    segmentation_mask = np.array(data['segmentation_mask'], dtype=np.int32)
    
    print(f"   ✓ Image: {image_path}")
    print(f"   ✓ Original size: {orig_w}x{orig_h}")
    print(f"   ✓ Model size: {model_w}x{model_h}")
    print(f"   ✓ Classes: {num_classes}")
    print(f"   ✓ Output shape: {output_shape}")
    
    # Reshape segmentation mask based on output shape
    if len(output_shape) == 4:
        # BiSeNet format: [batch, classes, height, width]
        batch, classes, height, width = output_shape
        segmentation_mask = segmentation_mask.reshape(height, width)
    elif len(output_shape) == 2:
        # GIM format: [height, width] - already in correct shape
        height, width = output_shape
        segmentation_mask = np.array(segmentation_mask).reshape(height, width)
    else:
        raise ValueError(f"Unexpected output shape: {output_shape}")
    
    print(f"\n📊 Segmentation Statistics:")
    print(f"   Mask shape: {segmentation_mask.shape}")
    print(f"   Value range: [{segmentation_mask.min()}, {segmentation_mask.max()}]")
    
    # Class distribution
    print(f"\n📈 Class Distribution:")
    unique, counts = np.unique(segmentation_mask, return_counts=True)
    total_pixels = height * width
    
    for class_id, count in zip(unique, counts):
        percentage = (count / total_pixels) * 100
        # For GIM model, use simple foreground/background labels
        if num_classes == 2:
            label = "Background" if class_id == 0 else "Foreground"
        else:
            label = CAMVID_LABELS[class_id] if class_id < len(CAMVID_LABELS) else f"Class {class_id}"
        print(f"   {label:12s} (Class {class_id:2d}): {count:6d} pixels ({percentage:5.2f}%)")
    
    # Load original image
    print(f"\n🖼️  Loading original image...")
    if not os.path.exists(image_path):
        print(f"   ⚠️  Warning: Original image not found at {image_path}")
        print(f"   Creating dummy image...")
        original_img = np.ones((orig_h, orig_w, 3), dtype=np.uint8) * 128
    else:
        original_img = np.array(Image.open(image_path).convert('RGB'))
        print(f"   ✓ Loaded: {original_img.shape}")
    
    # Apply color map
    print(f"\n🎨 Applying color map...")
    colored_segmentation = apply_color_map(segmentation_mask, height, width, num_classes)
    
    # Resize segmentation to match original image size for overlay
    colored_segmentation_resized = np.array(
        Image.fromarray(colored_segmentation).resize((orig_w, orig_h), Image.NEAREST)
    )
    
    # Create overlay (50% original + 50% segmentation)
    overlay = (0.5 * original_img + 0.5 * colored_segmentation_resized).astype(np.uint8)
    
    # Create visualization
    print(f"\n📊 Creating visualization...")
    fig = plt.figure(figsize=(20, 12))
    
    # Original image
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(original_img)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Segmentation mask (colored)
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(colored_segmentation)
    ax2.set_title('Segmentation Output', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Overlay
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(overlay)
    ax3.set_title('Overlay (50/50)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Class distribution histogram
    ax4 = plt.subplot(2, 3, 4)
    
    if num_classes == 2:
        # GIM model: only class 0 and 255
        actual_classes = unique  # Use the actual classes found in the mask
        class_counts = [np.sum(segmentation_mask == cls) for cls in actual_classes]
        colors_norm = [[0, 0, 0], [1, 0, 0]]  # Black for 0, Red for 255
        labels = ['Background\n(Class 0)', 'Foreground\n(Class 255)']
        
        bars = ax4.bar(range(len(actual_classes)), class_counts, color=colors_norm, 
                      edgecolor='black', linewidth=1.5)
        ax4.set_xlabel('Class', fontsize=12)
        ax4.set_ylabel('Pixel Count', fontsize=12)
        ax4.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(len(actual_classes)))
        ax4.set_xticklabels(labels)
        ax4.grid(axis='y', alpha=0.3)
        
        # Add percentage labels on bars
        for i, (bar, count) in enumerate(zip(bars, class_counts)):
            percentage = (count / total_pixels) * 100
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{percentage:.1f}%\n({count:,} px)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        # BiSeNet/CamVid: 12 classes
        class_ids = list(range(num_classes))
        class_counts = [np.sum(segmentation_mask == i) for i in class_ids]
        colors_norm = [tuple(c/255.0 for c in color) for color in CAMVID_COLORS]
        
        bars = ax4.bar(class_ids, class_counts, color=colors_norm, edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('Class ID', fontsize=12)
        ax4.set_ylabel('Pixel Count', fontsize=12)
        ax4.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax4.set_xticks(class_ids)
        ax4.grid(axis='y', alpha=0.3)
        
        # Add percentage labels on bars
        for i, (bar, count) in enumerate(zip(bars, class_counts)):
            if count > 0:
                percentage = (count / total_pixels) * 100
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{percentage:.1f}%',
                        ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Legend
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    if num_classes == 2:
        # GIM model: simple 2-class legend
        legend_elements = []
        for class_id in unique:
            if class_id == 0:
                color_norm = (0, 0, 0)
                label = "Background"
            else:  # class 255
                color_norm = (1, 0, 0)
                label = "Foreground"
            count = np.sum(segmentation_mask == class_id)
            percentage = (count / total_pixels) * 100
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, fc=color_norm, 
                             label=f"Class {class_id}: {label} ({percentage:.1f}%)")
            )
        ax5.legend(handles=legend_elements, loc='center', fontsize=14, frameon=True)
        ax5.set_title('GIM Classes', fontsize=14, fontweight='bold')
    else:
        # BiSeNet/CamVid: 12-class legend
        legend_elements = []
        for i, (color, label) in enumerate(zip(CAMVID_COLORS, CAMVID_LABELS)):
            color_norm = tuple(c / 255.0 for c in color)
            count = np.sum(segmentation_mask == i)
            percentage = (count / total_pixels) * 100
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, fc=color_norm, 
                             label=f"{i}: {label} ({percentage:.1f}%)")
            )
        ax5.legend(handles=legend_elements, loc='center', fontsize=11, frameon=True)
        ax5.set_title('CamVid Classes', fontsize=14, fontweight='bold')
    
    # Segmentation mask (class IDs as heatmap)
    ax6 = plt.subplot(2, 3, 6)
    if num_classes == 2:
        # For binary segmentation, use a simple colormap
        im = ax6.imshow(segmentation_mask, cmap='gray', vmin=0, vmax=255)
        ax6.set_title('Class ID Map (0=Black, 255=White)', fontsize=14, fontweight='bold')
    else:
        im = ax6.imshow(segmentation_mask, cmap='tab20', vmin=0, vmax=11)
        ax6.set_title('Class ID Map', fontsize=14, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    
    # Update main title based on model type
    if num_classes == 2:
        title = 'GIM Medical Image Segmentation (Binary Classification)'
    else:
        title = 'BiSeNet Semantic Segmentation (CamVid Dataset)'
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'segmentation_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {output_path}")
    
    # Save only the overlay image
    overlay_path = os.path.join(output_dir, 'overlay.png')
    Image.fromarray(overlay).save(overlay_path)
    print(f"✅ Saved overlay image to: {overlay_path}")

if __name__ == "__main__":
    import sys
    
    json_path = "output/raw_segmentation.json"
    
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    
    if not os.path.exists(json_path):
        print(f"❌ Error: File not found: {json_path}")
        print(f"\n💡 Usage: python3 visualize_segmentation.py [json_path]")
        print(f"   Default: {json_path}")
        sys.exit(1)
    
    visualize_segmentation(json_path)