#!/usr/bin/env python3
"""
Visualize raw YOLOv5 detections directly from inference output
NO postprocessing - just raw model predictions with coordinate transformation
NOTE: Model output already has sigmoid applied (values in [0,1] range)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
import sys

# Configuration
JSON_PATH = "output/raw_detections.json"

# Class names (customize for your model)
CLASS_NAMES = {
    0: "hyperplasia",
    1: "neoplasia"
}

def load_detections(json_path):
    """Load raw detections from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def transform_bbox_to_image(cx, cy, w, h, letterbox, orig_size, model_size):
    """
    Transform bounding box from model space to original image space
    
    Args:
        cx, cy, w, h: Center coordinates and size in model space (512x512)
        letterbox: Dict with 'scale', 'pad_x', 'pad_y'
        orig_size: Original image size [width, height]
        model_size: Model input size [width, height]
    
    Returns:
        (x1, y1, x2, y2) in original image coordinates
    """
    scale = letterbox['scale']
    pad_x = letterbox['pad_x']
    pad_y = letterbox['pad_y']
    
    inv_scale = 1.0 / scale
    
    # Convert from center format to corner format in model space
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    
    # Remove padding
    x1 -= pad_x
    y1 -= pad_y
    x2 -= pad_x
    y2 -= pad_y
    
    # Scale back to original image size
    x1 *= inv_scale
    y1 *= inv_scale
    x2 *= inv_scale
    y2 *= inv_scale
    
    # Clamp to image boundaries
    orig_w, orig_h = orig_size
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)
    
    return (x1, y1, x2, y2)

def process_and_visualize(data, top_k=20, min_objectness=0.1):
    """
    Process raw detections and visualize top-K
    
    Args:
        data: JSON data with detections
        top_k: Number of top detections to show
        min_objectness: Minimum objectness threshold (already a probability, NOT a logit)
    """
    print("╔═══════════════════════════════════════════════════════╗")
    print("║  Raw YOLOv5 Detection Visualization                  ║")
    print("╚═══════════════════════════════════════════════════════╝\n")
    
    # Load image
    image_path = data['image']['path']
    orig_size = data['image']['original_size']
    model_size = data['image']['model_size']
    letterbox = data['letterbox']
    
    print(f"📸 Image: {image_path}")
    print(f"   Original size: {orig_size[0]}x{orig_size[1]}")
    print(f"   Model size: {model_size[0]}x{model_size[1]}")
    print(f"   Letterbox: scale={letterbox['scale']:.4f}, pad=({letterbox['pad_x']:.1f}, {letterbox['pad_y']:.1f})\n")
    
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
    except FileNotFoundError:
        print(f"❌ Error: Image not found at {image_path}")
        sys.exit(1)
    
    # Process detections
    detections = data['detections']
    print(f"📦 Total raw detections: {len(detections)}\n")
    
    # Calculate final confidence for each detection
    # NOTE: Values are already probabilities (sigmoid applied in model/engine)
    processed = []
    for det in detections:
        cx = det['cx']
        cy = det['cy']
        w = det['w']
        h = det['h']
        obj = det['objectness_raw']  # Already a probability!
        class_scores_raw = det['class_scores_raw']  # Already probabilities!
        
        # Find best class (no sigmoid needed)
        best_class = np.argmax(class_scores_raw)
        best_class_score = class_scores_raw[best_class]
        
        # Final confidence = objectness * class_score
        conf = obj * best_class_score
        
        # Skip very low objectness detections
        if obj < min_objectness:
            continue
        
        # Transform to image coordinates
        x1, y1, x2, y2 = transform_bbox_to_image(
            cx, cy, w, h, letterbox, orig_size, model_size
        )
        
        # Skip boxes with zero or negative area
        if x2 <= x1 or y2 <= y1:
            continue
        
        processed.append({
            'det_idx': det['det_idx'],
            'bbox': (x1, y1, x2, y2),
            'bbox_model': (cx, cy, w, h),
            'objectness': obj,
            'class_id': best_class,
            'class_score': best_class_score,
            'confidence': conf,
        })
    
    # Sort by objectness (descending)
    processed.sort(key=lambda x: x['objectness'], reverse=True)
    
    print(f"🎯 Filtered detections (obj >= {min_objectness}): {len(processed)}\n")
    
    if len(processed) == 0:
        print("❌ No detections passed the filter!")
        print(f"   Try lowering min_objectness (currently {min_objectness})")
        return
    
    # Display top K
    display_count = min(top_k, len(processed))
    print(f"📊 Top {display_count} detections (by objectness):\n")
    
    for i, det in enumerate(processed[:display_count]):
        class_name = CLASS_NAMES.get(det['class_id'], f"class_{det['class_id']}")
        x1, y1, x2, y2 = det['bbox']
        cx, cy, w, h = det['bbox_model']
        
        print(f"   #{i+1}: det_idx={det['det_idx']}")
        print(f"      Objectness: {det['objectness']:.4f} (already a probability)")
        print(f"      Class: {class_name} (score={det['class_score']:.4f})")
        print(f"      Confidence: {det['confidence']:.4f}")
        print(f"      Model coords: ({cx:.1f}, {cy:.1f}, {w:.1f}, {h:.1f})")
        print(f"      Image coords: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f})")
        print(f"      Size: {x2-x1:.1f} x {y2-y1:.1f}")
        print()
    
    # Visualize
    fig, ax = plt.subplots(1, figsize=(14, 10))
    ax.imshow(img_array)
    
    colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
    
    for i, det in enumerate(processed[:display_count]):
        x1, y1, x2, y2 = det['bbox']
        class_name = CLASS_NAMES.get(det['class_id'], f"class_{det['class_id']}")
        color = colors[det['class_id'] % len(colors)]
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Draw label with rank
        label = f"#{i+1} {class_name}\nobj:{det['objectness']:.3f}\nconf:{det['confidence']:.3f}"
        ax.text(
            x1, y1 - 5,
            label,
            color='white',
            fontsize=9,
            bbox=dict(facecolor=color, alpha=0.7, boxstyle='round,pad=0.5')
        )
    
    ax.axis('off')
    plt.title(f"Raw YOLOv5 Detections (Top {display_count}, min_obj={min_objectness})", 
              fontsize=14, pad=10)
    plt.tight_layout()
    
    # Save output
    output_path = "output/raw_detection_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_path}\n")
    
    plt.show()

def main():
    """Main visualization routine"""
    # Load data
    print(f"📂 Loading detections from: {JSON_PATH}\n")
    try:
        data = load_detections(JSON_PATH)
    except FileNotFoundError:
        print(f"❌ Error: JSON file not found at {JSON_PATH}")
        print("   Run inference_v2 first to generate the file:")
        print("   cargo run --release --bin inference_v2 --features full")
        sys.exit(1)
    
    # Process and visualize
    # Adjust min_objectness based on your model:
    # - 0.5: Very strict (only high confidence)
    # - 0.3: Moderate
    # - 0.1: Permissive (includes lower confidence)
    process_and_visualize(data, top_k=20, min_objectness=0.3)
    
    print("╔═══════════════════════════════════════════════════════╗")
    print("║  ✅ Visualization Complete!                           ║")
    print("╚═══════════════════════════════════════════════════════╝")

if __name__ == "__main__":
    main()
