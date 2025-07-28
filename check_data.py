#!/usr/bin/env python3
import h5py
import numpy as np
from pathlib import Path

def examine_jax_fluids_data():
    """Examine what's actually in the JAX-Fluids output files"""
    print("🔍 EXAMINING JAX-FLUIDS DATA STRUCTURE")
    print("=" * 50)
    
    data_dir = Path("output/rocket_nozzle_internal_supersonic_production/domain")
    
    # Get the latest file
    h5_files = list(data_dir.glob("*.h5"))
    if not h5_files:
        print("❌ No HDF5 files found!")
        return
    
    latest_file = sorted(h5_files)[-1]
    print(f"📁 Examining: {latest_file}")
    
    with h5py.File(latest_file, 'r') as f:
        print("\n📊 HDF5 Structure:")
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"📁 Group: {name}")
            else:
                print(f"📄 Dataset: {name}")
                print(f"   Shape: {obj.shape}")
                print(f"   Type: {obj.dtype}")
                if obj.size < 1000:  # Small datasets
                    print(f"   Min/Max: {obj[...].min():.6f} / {obj[...].max():.6f}")
                    print(f"   Unique values: {len(np.unique(obj[...]))}")
                else:
                    print(f"   Min/Max: {obj[...].min():.6f} / {obj[...].max():.6f}")
                print()
        
        f.visititems(print_structure)
        
        # Check specifically for common fields
        common_fields = ['density', 'velocity', 'pressure', 'temperature', 'mach_number', 'levelset']
        print("🔍 Looking for common fields:")
        for field in common_fields:
            if field in f:
                data = f[field][...]
                print(f"✅ {field}: shape={data.shape}, range=[{data.min():.6f}, {data.max():.6f}]")
                if data.max() == data.min():
                    print(f"   ⚠️  CONSTANT DATA - all values are {data.min():.6f}")
            else:
                print(f"❌ {field}: not found")
    
    # Also check XDMF structure
    xdmf_file = latest_file.with_suffix('.xdmf')
    if xdmf_file.exists():
        print(f"\n📄 XDMF Content: {xdmf_file}")
        with open(xdmf_file, 'r') as f:
            content = f.read()
            print(content[:1000] + "..." if len(content) > 1000 else content)

if __name__ == "__main__":
    examine_jax_fluids_data() 