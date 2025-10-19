#!/usr/bin/env python3
"""
Visualize YOLOv5 detections from inference_v2 output
Draws bounding boxes on the sample image
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import sys

# Configuration
IMAGE_PATH = "apps/playgrounds/sample_img.jpg"
INPUT_SIZE = 512  # Model input size

# Class names (customize for your model)
CLASS_NAMES = {
    0: "hyperplasia",
    1: "neoplasia"
}

def load_image(image_path):
    """Load image with PIL"""
    img = Image.open(image_path)
    return np.array(img)

def parse_detections_from_output():
    """
    Parse detections from the terminal output of inference_v2
    Returns list of (x, y, w, h, confidence, class_id)
    """
    # These are the actual detections from your last run
    # Format: (x, y, w, h, confidence, class_id, objectness, class_score)
    detections = [
        (147.5, 292.2, 104.7, 162.0, 0.5665, 1, 0.7206, 0.7861),
        (149.4, 292.4, 111.5, 163.8, 0.5601, 1, 0.6971, 0.8034),
        (149.9, 293.7, 107.3, 121.2, 0.5435, 1, 0.6825, 0.7964),
        (150.0, 292.7, 112.7, 153.8, 0.5415, 1, 0.6711, 0.8069),
        (150.4, 292.9, 106.6, 148.8, 0.5400, 1, 0.6105, 0.8845),
    ]
    return detections

def scale_bbox_to_original(bbox, input_size, orig_size):
    """
    Scale bounding box from model input size to original image size
    Accounts for letterboxing (maintains aspect ratio with gray padding)
    """
    x, y, w, h = bbox
    orig_w, orig_h = orig_size
    
    # Calculate scale factor (same as preprocessing)
    scale = min(input_size / orig_w, input_size / orig_h)
    
    # Calculate padding
    new_w = orig_w * scale
    new_h = orig_h * scale
    pad_x = (input_size - new_w) / 2
    pad_y = (input_size - new_h) / 2
    
    # Remove padding and scale back to original
    x_orig = (x - pad_x) / scale
    y_orig = (y - pad_y) / scale
    w_orig = w / scale
    h_orig = h / scale
    
    return (x_orig, y_orig, w_orig, h_orig)

def draw_detections(image, detections, input_size=512, top_k=5):
    """
    Draw bounding boxes on image
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    orig_h, orig_w = image.shape[:2]
    
    # Color map for different classes
    colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
    
    print(f"\n📦 Drawing top {min(top_k, len(detections))} detections:")
    print(f"Original image size: {orig_w}x{orig_h}")
    print(f"Model input size: {input_size}x{input_size}\n")
    
    for idx, detection in enumerate(detections[:top_k]):
        x, y, w, h, conf, class_id, obj, cls_score = detection
        
        # Scale bbox to original image size
        x_orig, y_orig, w_orig, h_orig = scale_bbox_to_original(
            (x, y, w, h), input_size, (orig_w, orig_h)
        )
        
        # Convert center coords to top-left corner (matplotlib uses top-left)
        x1 = x_orig - w_orig / 2
        y1 = y_orig - h_orig / 2
        
        # Get class name
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        color = colors[class_id % len(colors)]
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), w_orig, h_orig,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Draw label
        label = f"{class_name}: {conf:.2f}"
        ax.text(
            x1, y1 - 5,
            label,
            color='white',
            fontsize=10,
            bbox=dict(facecolor=color, alpha=0.7, boxstyle='round,pad=0.3')
        )
        
        print(f"Detection #{idx + 1}:")
        print(f"  Class: {class_name} (id={class_id})")
        print(f"  Confidence: {conf:.4f} (obj={obj:.4f}, cls={cls_score:.4f})")
        print(f"  Model coords: ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})")
        print(f"  Image coords: ({x1:.1f}, {y1:.1f}, {w_orig:.1f}, {h_orig:.1f})")
        print()
    
    ax.axis('off')
    plt.title(f"YOLOv5 Detections (Top {min(top_k, len(detections))})", fontsize=14, pad=10)
    plt.tight_layout()
    
    # Save output
    output_path = "detection_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_path}")
    
    plt.show()

def main():
    """Main visualization routine"""
    print("╔═══════════════════════════════════════════════════════╗")
    print("║  YOLOv5 Detection Visualization                      ║")
    print("╚═══════════════════════════════════════════════════════╝\n")
    
    # Load image
    print(f"📸 Loading image: {IMAGE_PATH}")
    try:
        image = load_image(IMAGE_PATH)
        print(f"   ✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels\n")
    except FileNotFoundError:
        print(f"❌ Error: Image not found at {IMAGE_PATH}")
        print("   Make sure you're running from the project root directory")
        sys.exit(1)
    
    # Parse detections (from inference_v2 output)
    print("🔍 Parsing detections from inference_v2 output...")
    detections = parse_detections_from_output()
    print(f"   ✓ Found {len(detections)} high-confidence detections\n")
    
    # Draw and save
    draw_detections(image, detections, INPUT_SIZE, top_k=5)
    
    print("\n╔═══════════════════════════════════════════════════════╗")
    print("║  ✅ Visualization Complete!                           ║")
    print("╚═══════════════════════════════════════════════════════╝")

if __name__ == "__main__":
    main()
