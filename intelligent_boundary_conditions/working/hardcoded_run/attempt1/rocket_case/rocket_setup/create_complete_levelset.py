#!/usr/bin/env python3
"""
Create Complete Levelset for JAX-Fluids Rocket Nozzle Simulation

This script generates a comprehensive levelset that includes:
1. Walls (solid boundaries) - levelset > 0
2. Inlet region - levelset < 0 (special tag)  
3. Outlet region - levelset < 0 (special tag)
4. Fluid region - levelset < 0

JAX-Fluids uses levelset values and additional tags to determine boundary conditions.
"""

import numpy as np
import sys
from pathlib import Path

def load_existing_sdf():
    """Load the existing SDF for nozzle walls"""
    sdf_path = Path("../20250728_003202/Rocket Engine_sdf_matrix.npy")
    
    if not sdf_path.exists():
        print(f"❌ SDF file not found at {sdf_path}")
        return None, None
        
    print(f"📂 Loading existing SDF from {sdf_path}")
    wall_sdf = np.load(sdf_path)
    
    # Also load grid info if available
    try:
        grid_info = np.load(sdf_path.parent / "grid_info.npy", allow_pickle=True).item()
        print(f"📐 Grid info: {grid_info}")
    except:
        # Create default grid info based on setup.json domain
        grid_info = {
            'x_range': [-200.0, 1800.0],
            'y_range': [-800.0, 800.0], 
            'z_range': [-800.0, 800.0],
            'shape': wall_sdf.shape
        }
        print(f"📐 Using default grid info: {grid_info}")
    
    return wall_sdf, grid_info

def create_virtual_face_regions(grid_info):
    """Create levelset regions for virtual inlet and outlet faces"""
    
    x_range = grid_info['x_range']
    y_range = grid_info['y_range']
    z_range = grid_info['z_range']
    shape = grid_info['shape']
    
    # Create coordinate grids
    nx, ny, nz = shape
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    z = np.linspace(z_range[0], z_range[1], nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Define inlet region (smaller opening at X ≈ 0)
    inlet_center_x = 0.0
    inlet_center_y = 0.0  
    inlet_center_z = 0.0
    inlet_radius = 313.6  # From our virtual face detection
    
    # Distance from inlet center
    inlet_dist = np.sqrt((Y - inlet_center_y)**2 + (Z - inlet_center_z)**2)
    
    # Inlet region: circular opening at X ≈ 0
    inlet_region = (np.abs(X - inlet_center_x) < 50.0) & (inlet_dist <= inlet_radius)
    
    # Define outlet region (larger opening at X ≈ 1717)
    outlet_center_x = 1717.2
    outlet_center_y = 0.0
    outlet_center_z = 0.0  
    outlet_radius = 602.7  # From our virtual face detection
    
    # Distance from outlet center
    outlet_dist = np.sqrt((Y - outlet_center_y)**2 + (Z - outlet_center_z)**2)
    
    # Outlet region: circular opening at X ≈ 1717
    outlet_region = (np.abs(X - outlet_center_x) < 50.0) & (outlet_dist <= outlet_radius)
    
    print(f"✅ Created virtual regions:")
    print(f"   🔵 Inlet: {np.sum(inlet_region):,} grid points at X≈{inlet_center_x}, R={inlet_radius}")
    print(f"   🔴 Outlet: {np.sum(outlet_region):,} grid points at X≈{outlet_center_x}, R={outlet_radius}")
    
    return inlet_region, outlet_region

def create_complete_levelset():
    """Create the complete levelset with proper boundary region tags"""
    
    # Load existing wall SDF
    wall_sdf, grid_info = load_existing_sdf()
    if wall_sdf is None:
        return False
    
    print(f"📊 Wall SDF shape: {wall_sdf.shape}")
    print(f"📊 Wall SDF range: [{np.min(wall_sdf):.3f}, {np.max(wall_sdf):.3f}]")
    
    # Create virtual face regions
    inlet_region, outlet_region = create_virtual_face_regions(grid_info)
    
    # Create complete levelset
    # Start with wall SDF (positive inside solid, negative in fluid)
    complete_levelset = wall_sdf.copy()
    
    # Create boundary condition tags
    # 0 = fluid, 1 = wall, 2 = inlet, 3 = outlet
    bc_tags = np.zeros_like(wall_sdf, dtype=np.int32)
    
    # Tag walls (where levelset > 0)
    bc_tags[wall_sdf > 0] = 1
    
    # Tag inlet region (override to be fluid with inlet BC)
    bc_tags[inlet_region] = 2
    complete_levelset[inlet_region] = -1.0  # Ensure fluid region
    
    # Tag outlet region (override to be fluid with outlet BC)  
    bc_tags[outlet_region] = 3
    complete_levelset[outlet_region] = -1.0  # Ensure fluid region
    
    print(f"📊 Complete levelset range: [{np.min(complete_levelset):.3f}, {np.max(complete_levelset):.3f}]")
    print(f"📊 BC tags distribution:")
    print(f"   Fluid (0): {np.sum(bc_tags == 0):,} points")
    print(f"   Wall (1): {np.sum(bc_tags == 1):,} points")  
    print(f"   Inlet (2): {np.sum(bc_tags == 2):,} points")
    print(f"   Outlet (3): {np.sum(bc_tags == 3):,} points")
    
    # Save complete levelset and boundary condition tags
    output_dir = Path(".")
    
    levelset_file = output_dir / "complete_rocket_levelset.npy"
    np.save(levelset_file, complete_levelset)
    print(f"💾 Saved complete levelset to {levelset_file}")
    
    bc_tags_file = output_dir / "boundary_condition_tags.npy"
    np.save(bc_tags_file, bc_tags)
    print(f"💾 Saved BC tags to {bc_tags_file}")
    
    # Save grid info for reference
    grid_file = output_dir / "grid_info.npy"
    np.save(grid_file, grid_info)
    print(f"💾 Saved grid info to {grid_file}")
    
    return True

if __name__ == "__main__":
    print("🚀 Creating Complete Levelset for JAX-Fluids Rocket Nozzle")
    print("=" * 60)
    
    success = create_complete_levelset()
    
    if success:
        print("\n✅ Complete levelset created successfully!")
        print("\nNext steps:")
        print("1. Update setup.json to use 'complete_rocket_levelset.npy'")
        print("2. Configure boundary conditions for different levelset regions")
        print("3. Run JAX-Fluids simulation")
    else:
        print("\n❌ Failed to create complete levelset")
        sys.exit(1) 