#!/usr/bin/env python3
"""
Visualize and compare GIM segmentation results across multiple threshold configurations.
Creates a grid layout showing all 6 threshold configurations side-by-side.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
        'class_distribution': data['class_distribution']
    }

def main():
    # Load all 6 configurations (narrow range around 0.15)
    configs = [
        'config_1',  # 0.12
        'config_2',  # 0.13
        'config_3',  # 0.14
        'config_4',  # 0.15
        'config_5',  # 0.16
        'config_6',  # 0.17
    ]
    
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
    
    # Create figure with 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('GIM Segmentation: Threshold Configuration Comparison', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    axes = axes.flatten()
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        mask = result['mask']
        thresholds = result['thresholds']
        config_name = result['config_name']
        class_dist = result['class_distribution']
        
        # Display segmentation mask
        im = ax.imshow(mask, cmap='gray', vmin=0, vmax=255)
        
        # Calculate percentages
        total_pixels = mask.shape[0] * mask.shape[1]
        class_0_count = class_dist.get('0', 0)
        class_255_count = class_dist.get('255', 0)
        class_0_pct = (class_0_count / total_pixels) * 100
        class_255_pct = (class_255_count / total_pixels) * 100
        
        # Title with config name and thresholds
        title = f"{config_name.upper().replace('_', ' ')}\n"
        title += f"NBI={thresholds['nbi']:.2f}, WLE={thresholds['wle']:.2f}, C={thresholds['c']:.2f}\n"
        title += f"BG: {class_0_pct:.1f}%  FG: {class_255_pct:.1f}%"
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar to the right of each subplot
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Class', rotation=270, labelpad=15)
        cbar.set_ticks([0, 255])
        cbar.set_ticklabels(['BG (0)', 'FG (255)'])
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    output_path = 'output/threshold_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison visualization saved to: {output_path}")
    
    plt.show()

if __name__ == '__main__':
    main()
