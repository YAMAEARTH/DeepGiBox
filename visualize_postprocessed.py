#!/usr/bin/env python3
"""
Visualize postprocessed detections (after NMS and tracking)
Shows the final detection results that would be used in production
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
import sys

# Configuration
JSON_PATH = "output/postprocessed_detections.json"

# Class names (customize for your model)
CLASS_NAMES = {
    0: "hyperplasia",
    1: "neoplasia"
}

def load_detections(json_path):
    """Load postprocessed detections from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def visualize(data):
    """
    Visualize postprocessed detections
    
    Args:
        data: JSON data with postprocessed detections
    """
    print("╔═══════════════════════════════════════════════════════╗")
    print("║  Postprocessed Detection Visualization               ║")
    print("║  (After NMS + Tracking)                               ║")
    print("╚═══════════════════════════════════════════════════════╝\n")
    
    # Load image
    image_path = data['image']['path']
    orig_size = data['image']['size']
    
    print(f"📸 Image: {image_path}")
    print(f"   Size: {orig_size[0]}x{orig_size[1]}\n")
    
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
    except FileNotFoundError:
        print(f"❌ Error: Image not found at {image_path}")
        sys.exit(1)
    
    # Get detections
    detections = data['detections']
    print(f"📦 Total postprocessed detections: {len(detections)}\n")
    
    if len(detections) == 0:
        print("❌ No detections found!")
        return
    
    # Display detection info
    print(f"🎯 Detections:\n")
    
    for i, det in enumerate(detections):
        bbox = det['bbox']
        class_name = CLASS_NAMES.get(det['class_id'], f"class_{det['class_id']}")
        
        print(f"   #{i+1}:")
        print(f"      Class: {class_name}")
        print(f"      Confidence: {det['score']:.4f}")
        print(f"      BBox: ({bbox['x']:.1f}, {bbox['y']:.1f}, {bbox['w']:.1f}, {bbox['h']:.1f})")
        if det['track_id'] is not None:
            print(f"      Track ID: {det['track_id']}")
        print()
    
    # Visualize
    fig, ax = plt.subplots(1, figsize=(14, 10))
    ax.imshow(img_array)
    
    colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
    
    for i, det in enumerate(detections):
        bbox = det['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        class_name = CLASS_NAMES.get(det['class_id'], f"class_{det['class_id']}")
        color = colors[det['class_id'] % len(colors)]
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=3,
            edgecolor=color,
            facecolor='none',
            alpha=0.9
        )
        ax.add_patch(rect)
        
        # Draw label
        label = f"#{i+1} {class_name}\nconf:{det['score']:.3f}"
        if det['track_id'] is not None:
            label += f"\nID:{det['track_id']}"
        
        ax.text(
            x, y - 10,
            label,
            color='white',
            fontsize=11,
            weight='bold',
            bbox=dict(facecolor=color, alpha=0.8, boxstyle='round,pad=0.5')
        )
    
    ax.axis('off')
    plt.title(f"Postprocessed Detections (After NMS) - {len(detections)} objects", 
              fontsize=14, pad=10, weight='bold')
    plt.tight_layout()
    
    # Save output
    output_path = "output/postprocessed_visualization.png"
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
    
    # Visualize
    visualize(data)
    
    print("╔═══════════════════════════════════════════════════════╗")
    print("║  ✅ Visualization Complete!                           ║")
    print("╚═══════════════════════════════════════════════════════╝")

if __name__ == "__main__":
    main()
