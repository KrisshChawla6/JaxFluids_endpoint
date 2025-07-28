#!/usr/bin/env python3
"""
Fix JAX-Fluids configuration for single-device testing
"""

import json
from pathlib import Path

def fix_config_for_single_device():
    """Fix the domain decomposition to work with single device"""
    
    # Find the latest config
    config_dirs = [d for d in Path("production_test_config").iterdir() 
                   if d.is_dir() and d.name.startswith("jaxfluids_external_flow")]
    
    if not config_dirs:
        print("❌ No config directories found")
        return False
    
    latest_config = max(config_dirs, key=lambda x: x.name)
    case_file = latest_config / f"{latest_config.name}.json"
    
    print(f"🔧 Fixing config: {case_file}")
    
    # Load and modify the config
    with open(case_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Fix decomposition for single device
    if 'domain' in config and 'decomposition' in config['domain']:
        print("📊 Original decomposition:", config['domain']['decomposition'])
        config['domain']['decomposition'] = {
            "split_x": 1,
            "split_y": 1, 
            "split_z": 1
        }
        print("✅ Fixed decomposition:", config['domain']['decomposition'])
    
    # Also reduce problem size for faster testing
    if 'domain' in config:
        for dim in ['x', 'y', 'z']:
            if dim in config['domain'] and 'cells' in config['domain'][dim]:
                original_cells = config['domain'][dim]['cells']
                new_cells = 32  # Much smaller for testing
                config['domain'][dim]['cells'] = new_cells
                print(f"📐 Reduced {dim} cells: {original_cells} → {new_cells}")
    
    # Save the fixed config
    with open(case_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration fixed for single-device testing")
    return True

if __name__ == "__main__":
    fix_config_for_single_device() 