#!/usr/bin/env python3
"""
Generate JAX-Fluids Compatible Masks
From Circular Virtual Faces for Rocket Nozzle
"""

import numpy as np
import pyvista as pv
from pathlib import Path
from circular_face_creator import find_circular_boundary_edges, fit_circle_and_create_face

def create_jax_masks():
    """Create JAX-Fluids compatible inlet/outlet masks"""
    
    print("🎯 GENERATING JAX-FLUIDS MASKS")
    print("=" * 50)
    
    # Load the rocket mesh
    mesh_file = Path("../../mesh/Rocket Engine.msh")
    if not mesh_file.exists():
        # Try alternative paths
        mesh_file = Path("../../../mesh/Rocket Engine.msh") 
        if not mesh_file.exists():
            raise FileNotFoundError(f"❌ Mesh file not found: {mesh_file}")
    
    print(f"📁 Using mesh: {mesh_file}")
    
    # Find circular boundary edges
    print("🔍 Finding circular boundary edges...")
    inlet_points, outlet_points = find_circular_boundary_edges(str(mesh_file))
    
    if inlet_points is None or outlet_points is None:
        raise RuntimeError("❌ Failed to find inlet/outlet points")
    
    print(f"   ✅ Inlet: {len(inlet_points)} points")
    print(f"   ✅ Outlet: {len(outlet_points)} points")
    
    # Create virtual faces
    print("🔧 Creating virtual faces...")
    inlet_face = fit_circle_and_create_face(inlet_points)
    outlet_face = fit_circle_and_create_face(outlet_points)
    
    # JAX-Fluids domain setup (matching our setup.json)
    domain_bounds = [-200.0, -800.0, -800.0, 1800.0, 800.0, 800.0]  # x_min, y_min, z_min, x_max, y_max, z_max
    grid_shape = (128, 64, 64)  # nx, ny, nz
    
    print(f"🔧 Creating masks for JAX-Fluids grid...")
    print(f"   Domain: X=[{domain_bounds[0]}, {domain_bounds[3]}]")
    print(f"   Domain: Y=[{domain_bounds[1]}, {domain_bounds[4]}]") 
    print(f"   Domain: Z=[{domain_bounds[2]}, {domain_bounds[5]}]")
    print(f"   Grid: {grid_shape}")
    
    # Create structured grid
    nx, ny, nz = grid_shape
    x = np.linspace(domain_bounds[0], domain_bounds[3], nx)
    y = np.linspace(domain_bounds[1], domain_bounds[4], ny)
    z = np.linspace(domain_bounds[2], domain_bounds[5], nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Initialize masks
    inlet_mask = np.zeros(grid_shape, dtype=bool)
    outlet_mask = np.zeros(grid_shape, dtype=bool)
    
    # Get inlet face parameters
    inlet_center = inlet_face['center']
    inlet_radius = inlet_face['radius']
    inlet_x = inlet_center[0]
    
    # Get outlet face parameters  
    outlet_center = outlet_face['center']
    outlet_radius = outlet_face['radius']
    outlet_x = outlet_center[0]
    
    print(f"   🔵 Inlet: center=({inlet_center[0]:.1f}, {inlet_center[1]:.1f}, {inlet_center[2]:.1f}), radius={inlet_radius:.1f}")
    print(f"   🔴 Outlet: center=({outlet_center[0]:.1f}, {outlet_center[1]:.1f}, {outlet_center[2]:.1f}), radius={outlet_radius:.1f}")
    
    # Create inlet mask
    x_tolerance = (domain_bounds[3] - domain_bounds[0]) / nx * 2  # 2 grid spacings
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Check inlet region
                if abs(X[i,j,k] - inlet_x) < x_tolerance:
                    distance_from_center = np.sqrt((Y[i,j,k] - inlet_center[1])**2 + (Z[i,j,k] - inlet_center[2])**2)
                    if distance_from_center <= inlet_radius:
                        inlet_mask[i,j,k] = True
                
                # Check outlet region
                if abs(X[i,j,k] - outlet_x) < x_tolerance:
                    distance_from_center = np.sqrt((Y[i,j,k] - outlet_center[1])**2 + (Z[i,j,k] - outlet_center[2])**2)
                    if distance_from_center <= outlet_radius:
                        outlet_mask[i,j,k] = True
    
    # Create output directory
    output_dir = Path("hardcoded_run/attempt1/rocket_nozzle_jaxfluids_simulation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save masks
    inlet_file = output_dir / "inlet_boundary_mask.npy"
    outlet_file = output_dir / "outlet_boundary_mask.npy"
    
    np.save(inlet_file, inlet_mask)
    np.save(outlet_file, outlet_mask)
    
    print(f"💾 Saved masks:")
    print(f"   ✅ Inlet mask: {inlet_file} ({np.sum(inlet_mask):,} active points)")
    print(f"   ✅ Outlet mask: {outlet_file} ({np.sum(outlet_mask):,} active points)")
    
    return inlet_face, outlet_face, inlet_mask, outlet_mask

if __name__ == "__main__":
    try:
        inlet_face, outlet_face, inlet_mask, outlet_mask = create_jax_masks()
        print("🎉 JAX-Fluids masks generated successfully!")
        
    except Exception as e:
        print(f"❌ Failed to generate masks: {e}")
        import traceback
        traceback.print_exc() 