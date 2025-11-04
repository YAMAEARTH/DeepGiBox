#!/usr/bin/env python3
"""Show summary of all 20 threshold configurations"""
import json
from pathlib import Path

configs = []
for i in range(1, 21):
    json_path = f'output/seg_config_{i:02d}.json'
    if Path(json_path).exists():
        with open(json_path) as f:
            data = json.load(f)
            thresholds = data['thresholds']
            class_dist = data['class_distribution']
            total = sum(class_dist.values())
            fg_pct = (class_dist.get('255', 0) / total) * 100
            configs.append({
                'name': data['config_name'],
                'nbi': thresholds['nbi'],
                'wle': thresholds['wle'],
                'c': thresholds['c'],
                'fg_pct': fg_pct
            })

print("\n" + "="*80)
print("GIM Segmentation - Threshold Configuration Summary")
print("="*80)
print(f"{'Config':<12} {'NBI':<8} {'WLE':<8} {'C':<8} {'Foreground %':<15} {'Balance':<15}")
print("-"*80)

for cfg in configs:
    balance = ""
    if cfg['fg_pct'] > 80:
        balance = "❌ Too much FG"
    elif cfg['fg_pct'] < 10:
        balance = "❌ Too much BG"
    elif 40 <= cfg['fg_pct'] <= 60:
        balance = "✅ BALANCED"
    elif 25 <= cfg['fg_pct'] <= 75:
        balance = "✓ Good"
    else:
        balance = "⚠ OK"
    
    print(f"{cfg['name']:<12} {cfg['nbi']:<8.3f} {cfg['wle']:<8.3f} {cfg['c']:<8.3f} {cfg['fg_pct']:>6.1f}%          {balance:<15}")

print("="*80)
print("\n✅ Best balanced configs are between 40-60% foreground")
print("📁 Check output/all_configs_overlay.png for visual comparison\n")
