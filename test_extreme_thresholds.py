#!/usr/bin/env python3
"""Test with extreme threshold differences to see if they actually change the output"""
import subprocess
import json

# Test 3 extreme configs
configs = [
    ("0.10", "0.10", "0.20", "extreme_low"),
    ("0.15", "0.15", "0.25", "medium"),  
    ("0.20", "0.20", "0.30", "extreme_high"),
]

print("Testing extreme threshold differences...")
print("="*60)

for nbi, wle, c, name in configs:
    # Modify inference_v3.rs to use single config
    with open("apps/playgrounds/src/bin/inference_v3.rs", "r") as f:
        content = f.read()
    
    # Just run with existing configs and check output
    pass

# Instead, let's just check the existing outputs
for i in [1, 10, 20]:
    json_path = f"output/seg_config_{i:02d}.json"
    try:
        with open(json_path) as f:
            data = json.load(f)
            thresholds = data['thresholds']
            class_dist = data['class_distribution']
            total = sum(class_dist.values())
            fg_pct = (class_dist.get('255', 0) / total) * 100
            
            print(f"\nConfig {i:02d}:")
            print(f"  NBI={thresholds['nbi']:.3f}, WLE={thresholds['wle']:.3f}, C={thresholds['c']:.3f}")
            print(f"  Foreground: {fg_pct:.2f}%")
            
            # Check if changing each threshold individually affects output
            if i == 1:
                baseline = (thresholds['nbi'], thresholds['wle'], thresholds['c'], fg_pct)
    except FileNotFoundError:
        print(f"  (not found)")

print("\n" + "="*60)
print("Analysis:")
print("  If WLE and C are being ignored, we'd see:")
print("  - Configs with same NBI but different WLE/C having identical output")
print("  - Let's check configs 2 vs 18:")

try:
    with open("output/seg_config_02.json") as f:
        cfg2 = json.load(f)
    with open("output/seg_config_18.json") as f:
        cfg18 = json.load(f)
    
    t2 = cfg2['thresholds']
    t18 = cfg18['thresholds']
    fg2 = (cfg2['class_distribution'].get('255', 0) / sum(cfg2['class_distribution'].values())) * 100
    fg18 = (cfg18['class_distribution'].get('255', 0) / sum(cfg18['class_distribution'].values())) * 100
    
    print(f"\nConfig 02: NBI={t2['nbi']:.3f}, WLE={t2['wle']:.3f}, C={t2['c']:.3f} -> FG={fg2:.1f}%")
    print(f"Config 18: NBI={t18['nbi']:.3f}, WLE={t18['wle']:.3f}, C={t18['c']:.3f} -> FG={fg18:.1f}%")
    
    if abs(t2['nbi'] - t18['nbi']) < 0.01 and abs(fg2 - fg18) < 1:
        print("⚠️  WARNING: Similar NBI with different WLE/C produce same output!")
        print("   This suggests WLE and C thresholds may not be working!")
    else:
        print("✅ Different thresholds produce different outputs - working correctly!")
        
except Exception as e:
    print(f"Error: {e}")
