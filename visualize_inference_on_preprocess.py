#!/usr/bin/env python3
"""
Visualize inference results on preprocessed image
Draw bounding boxes from inference output on the 512x512 preprocessed frame
"""

import cv2
import numpy as np
from pathlib import Path

# Paths
PREPROCESS_IMG = "output/test/preprocess_frame_0000.png"
INFERENCE_TXT = "output/test/inference_frame_0000.txt"
OUTPUT_IMG = "output/test/inference_visualization.png"

# Colors for visualization
COLOR_BBOX = (0, 255, 0)      # Green for bounding box
COLOR_CENTER = (0, 0, 255)    # Red for center point
COLOR_TEXT_BG = (0, 0, 0)     # Black background for text
COLOR_TEXT = (255, 255, 255)  # White text
THICKNESS = 2

def parse_inference_output(txt_path):
    """Parse inference output text file and extract detection coordinates"""
    detections = []
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    # Find the start of detection data
    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Rank #"):
            start_idx = i
            break
    
    if start_idx is None:
        print("No detections found in file")
        return detections
    
    # Parse each detection line
    for line in lines[start_idx:]:
        if not line.startswith("Rank #"):
            continue
            
        # Parse line like:
        # Rank #1 (orig_idx=15598): x=446.951, y=470.093, w=68.048, h=119.207, obj=0.007, cls0=0.305, cls1=0.767
        parts = line.split(":")
        if len(parts) < 2:
            continue
        
        # Extract rank number
        rank = int(parts[0].split("#")[1].split("(")[0].strip())
        
        # Extract coordinates and confidences
        coords_part = parts[1].strip()
        values = {}
        for item in coords_part.split(","):
            item = item.strip()
            if "=" in item:
                key, val = item.split("=")
                values[key.strip()] = float(val.strip())
        
        detection = {
            'rank': rank,
            'x': values.get('x', 0),
            'y': values.get('y', 0),
            'w': values.get('w', 0),
            'h': values.get('h', 0),
            'obj': values.get('obj', 0),
            'cls0': values.get('cls0', 0),
            'cls1': values.get('cls1', 0),
        }
        detections.append(detection)
    
    return detections

def draw_detections(img, detections, top_n=10):
    """Draw bounding boxes on image"""
    img_vis = img.copy()
    h, w = img_vis.shape[:2]
    
    print(f"\nDrawing top {top_n} detections on {w}x{h} image:")
    print(f"{'Rank':<6} {'Center':<20} {'Size':<20} {'Obj':<8} {'Cls0':<8} {'Cls1':<8}")
    print("-" * 80)
    
    for det in detections[:top_n]:
        cx, cy = det['x'], det['y']
        box_w, box_h = det['w'], det['h']
        obj_conf = det['obj']
        cls0_conf = det['cls0']
        cls1_conf = det['cls1']
        
        # Calculate corner coordinates
        x1 = int(cx - box_w / 2)
        y1 = int(cy - box_h / 2)
        x2 = int(cx + box_w / 2)
        y2 = int(cy + box_h / 2)
        
        # Clamp to image bounds
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        # Determine class (higher confidence wins)
        if cls0_conf > cls1_conf:
            class_name = "Hyper"
            color = (255, 0, 0)  # Blue
        else:
            class_name = "Neo"
            color = (0, 255, 255)  # Yellow
        
        # Print detection info
        print(f"#{det['rank']:<5} ({cx:.1f}, {cy:.1f})    {box_w:.1f}×{box_h:.1f}    "
              f"{obj_conf:.4f}  {cls0_conf:.3f}   {cls1_conf:.3f}")
        
        # Draw bounding box
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, THICKNESS)
        
        # Draw center point
        cv2.circle(img_vis, (int(cx), int(cy)), 3, COLOR_CENTER, -1)
        
        # Draw label with background
        label = f"#{det['rank']} {class_name} {obj_conf:.3f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Label position (above box if possible)
        label_x = x1
        label_y = y1 - 5
        if label_y < label_h + 5:
            label_y = y2 + label_h + 5
        
        # Draw text background
        cv2.rectangle(img_vis, 
                     (label_x, label_y - label_h - baseline),
                     (label_x + label_w, label_y + baseline),
                     COLOR_TEXT_BG, -1)
        
        # Draw text
        cv2.putText(img_vis, label, (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    return img_vis

def main():
    print("=" * 80)
    print("Inference Result Visualization on Preprocessed Image")
    print("=" * 80)
    
    # Check if files exist
    if not Path(PREPROCESS_IMG).exists():
        print(f"❌ Preprocessed image not found: {PREPROCESS_IMG}")
        return
    
    if not Path(INFERENCE_TXT).exists():
        print(f"❌ Inference output not found: {INFERENCE_TXT}")
        return
    
    # Load preprocessed image
    print(f"\n📸 Loading preprocessed image: {PREPROCESS_IMG}")
    img = cv2.imread(PREPROCESS_IMG)
    if img is None:
        print(f"❌ Failed to load image: {PREPROCESS_IMG}")
        return
    
    h, w = img.shape[:2]
    print(f"   Image size: {w}×{h}")
    
    # Parse inference output
    print(f"\n📄 Parsing inference output: {INFERENCE_TXT}")
    detections = parse_inference_output(INFERENCE_TXT)
    print(f"   Found {len(detections)} detections")
    
    if len(detections) == 0:
        print("❌ No detections to visualize")
        return
    
    # Draw detections
    print(f"\n🎨 Drawing bounding boxes...")
    img_vis = draw_detections(img, detections, top_n=10)
    
    # Save result
    cv2.imwrite(OUTPUT_IMG, img_vis)
    print(f"\n✅ Saved visualization: {OUTPUT_IMG}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  • Input image: {w}×{h} (preprocessed tensor)")
    print(f"  • Total detections: {len(detections)}")
    print(f"  • Visualized: top 10")
    print(f"  • Output: {OUTPUT_IMG}")
    print("=" * 80)
    
    # Show coordinate space info
    print("\n📊 Coordinate Space Analysis:")
    print(f"  All coordinates are in MODEL SPACE (512×512)")
    print(f"  • x, y: center point in range [0, 512]")
    print(f"  • w, h: box size in pixels")
    print(f"  • To convert to original image, need to:")
    print(f"    1. Scale from 512×512 → crop size (1372×1080)")
    print(f"    2. Add crop offset (548, 0) for Olympus")
    print(f"    3. Result in original space (1920×1080)")
    print()

if __name__ == "__main__":
    main()
