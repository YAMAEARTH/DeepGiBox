#!/usr/bin/env python3
"""
Visualize and compare all 20 GIM segmentation configurations.
Shows original image with overlay for each threshold configuration.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

def load_segmentation(json_path):
    """Load segmentation data from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    seg_mask = np.array(data['segmentation_mask'], dtype=np.uint8)
    height, width = data['output_shape']
    seg_mask = seg_mask.reshape(height, width)
    
    return {
        'mask': seg_mask,
        'config_name': data['config_name'],
        'thresholds': data['thresholds'],
        'class_distribution': data['class_distribution'],
        'image_path': data['image']['path']
    }

def main():
    # Load all 20 configurations
    configs = [f'config_{i:02d}' for i in range(1, 21)]
    
    results = []
    for config_name in configs:
        json_path = f'output/seg_{config_name}.json'
        if Path(json_path).exists():
            results.append(load_segmentation(json_path))
            print(f"✓ Loaded {config_name}")
        else:
            print(f"✗ Missing {json_path}")
    
    if not results:
        print("No segmentation results found!")
        return
    
    # Load original image
    img_path = results[0]['image_path']
    original_img = Image.open(img_path).convert('RGB')
    # Resize to model size
    original_img = original_img.resize((640, 512), Image.Resampling.LANCZOS)
    original_img = np.array(original_img)
    
    # Create figure with 5x4 grid (20 subplots)
    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
    fig.suptitle('GIM Segmentation: 20 Threshold Configurations with Overlay', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    axes = axes.flatten()
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        mask = result['mask']
        thresholds = result['thresholds']
        config_name = result['config_name']
        class_dist = result['class_distribution']
        
        # Create overlay: original image with red mask on foreground
        overlay = original_img.copy().astype(np.float32)
        
        # Red overlay where mask is 255 (foreground)
        red_mask = (mask == 255)
        overlay[red_mask, 0] = overlay[red_mask, 0] * 0.4 + 255 * 0.6  # Red channel
        overlay[red_mask, 1] = overlay[red_mask, 1] * 0.4              # Green channel  
        overlay[red_mask, 2] = overlay[red_mask, 2] * 0.4              # Blue channel
        
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        # Display overlay
        ax.imshow(overlay)
        
        # Calculate percentages
        total_pixels = mask.shape[0] * mask.shape[1]
        class_0_count = class_dist.get('0', 0)
        class_255_count = class_dist.get('255', 0)
        class_0_pct = (class_0_count / total_pixels) * 100
        class_255_pct = (class_255_count / total_pixels) * 100
        
        # Title with config name and thresholds
        title = f"{config_name.upper()}\n"
        title += f"NBI={thresholds['nbi']:.3f}, WLE={thresholds['wle']:.3f}, C={thresholds['c']:.3f}\n"
        title += f"BG: {class_0_pct:.1f}%  FG: {class_255_pct:.1f}%"
        
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.993])
    
    # Save figure
    output_path = 'output/all_configs_overlay.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ All configs visualization saved to: {output_path}")
    
    # Don't show - just save
    plt.close()

if __name__ == '__main__':
    main()
