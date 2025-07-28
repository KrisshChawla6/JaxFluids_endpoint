#!/usr/bin/env python3
"""
Visualize Complete Levelset for JAX-Fluids Rocket Nozzle

This script loads and visualizes the complete levelset with:
- Walls (red)
- Inlet region (blue) 
- Outlet region (green)
- Fluid region (transparent)
"""

import numpy as np
import pyvista as pv
from pathlib import Path

def load_complete_levelset():
    """Load the complete levelset and boundary condition tags"""
    
    levelset_file = Path("complete_rocket_levelset.npy")
    bc_tags_file = Path("boundary_condition_tags.npy")
    grid_info_file = Path("grid_info.npy")
    
    if not all([f.exists() for f in [levelset_file, bc_tags_file, grid_info_file]]):
        print("❌ Missing levelset files. Run create_complete_levelset.py first")
        return None, None, None
    
    levelset = np.load(levelset_file)
    bc_tags = np.load(bc_tags_file)
    grid_info = np.load(grid_info_file, allow_pickle=True).item()
    
    print(f"📂 Loaded complete levelset:")
    print(f"   Shape: {levelset.shape}")
    print(f"   Levelset range: [{np.min(levelset):.3f}, {np.max(levelset):.3f}]")
    print(f"   BC tags: {np.unique(bc_tags)}")
    
    return levelset, bc_tags, grid_info

def create_visualization():
    """Create PyVista visualization of the complete levelset"""
    
    levelset, bc_tags, grid_info = load_complete_levelset()
    if levelset is None:
        return
    
    # Create PyVista structured grid
    x_range = grid_info['x_range']
    y_range = grid_info['y_range'] 
    z_range = grid_info['z_range']
    nx, ny, nz = levelset.shape
    
    # Create coordinate arrays
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    z = np.linspace(z_range[0], z_range[1], nz)
    
    # Create structured grid
    grid = pv.StructuredGrid()
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid.points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    grid.dimensions = (nx, ny, nz)
    
    # Add data to grid
    grid["levelset"] = levelset.ravel()
    grid["bc_tags"] = bc_tags.ravel()
    
    print(f"✅ Created PyVista grid with {grid.n_points:,} points")
    
    # Create plotter
    plotter = pv.Plotter(window_size=(1200, 800))
    plotter.add_text("JAX-Fluids Rocket Nozzle Complete Levelset", position='upper_left', font_size=14)
    
    # 1. Visualize walls (bc_tags == 1) - Red
    wall_mask = bc_tags == 1
    if np.any(wall_mask):
        wall_grid = grid.threshold(value=[0.5, 1.5], scalars="bc_tags")
        if wall_grid.n_points > 0:
            plotter.add_mesh(wall_grid, color='red', opacity=0.7, label='Walls')
            print(f"🔴 Walls: {np.sum(wall_mask):,} points")
    
    # 2. Visualize inlet (bc_tags == 2) - Blue
    inlet_mask = bc_tags == 2
    if np.any(inlet_mask):
        inlet_grid = grid.threshold(value=[1.5, 2.5], scalars="bc_tags")
        if inlet_grid.n_points > 0:
            plotter.add_mesh(inlet_grid, color='blue', opacity=0.9, label='Inlet')
            print(f"🔵 Inlet: {np.sum(inlet_mask):,} points")
    
    # 3. Visualize outlet (bc_tags == 3) - Green  
    outlet_mask = bc_tags == 3
    if np.any(outlet_mask):
        outlet_grid = grid.threshold(value=[2.5, 3.5], scalars="bc_tags")
        if outlet_grid.n_points > 0:
            plotter.add_mesh(outlet_grid, color='green', opacity=0.9, label='Outlet')
            print(f"🟢 Outlet: {np.sum(outlet_mask):,} points")
    
    # 4. Show levelset zero contour (wall surface)
    contour = grid.contour(isosurfaces=[0.0], scalars="levelset")
    if contour.n_points > 0:
        plotter.add_mesh(contour, color='orange', opacity=0.5, line_width=2, label='Zero Levelset')
        print(f"🟠 Zero levelset contour: {contour.n_points:,} points")
    
    # Add coordinate axes and legend
    plotter.add_axes()
    plotter.add_legend()
    
    # Set camera for good view
    plotter.camera_position = 'xz'
    plotter.show_bounds(grid='front', location='outer')
    
    # Add info text
    info_text = f"""
Complete Levelset Information:
• Grid: {nx} × {ny} × {nz} = {nx*ny*nz:,} points
• X: [{x_range[0]:.0f}, {x_range[1]:.0f}]
• Y: [{y_range[0]:.0f}, {y_range[1]:.0f}]  
• Z: [{z_range[0]:.0f}, {z_range[1]:.0f}]
• Walls: {np.sum(wall_mask):,} points
• Inlet: {np.sum(inlet_mask):,} points
• Outlet: {np.sum(outlet_mask):,} points
• Fluid: {np.sum(bc_tags == 0):,} points
"""
    plotter.add_text(info_text, position='upper_right', font_size=10)
    
    print("\n🚀 Launching visualization...")
    print("Legend:")
    print("🔴 Red = Walls (solid boundaries)")
    print("🔵 Blue = Inlet region") 
    print("🟢 Green = Outlet region")
    print("🟠 Orange = Zero levelset contour (wall surface)")
    
    plotter.show()

def create_cross_section_views():
    """Create cross-section views of the levelset"""
    
    levelset, bc_tags, grid_info = load_complete_levelset()
    if levelset is None:
        return
    
    # Create figure with multiple subplots
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('JAX-Fluids Rocket Nozzle Complete Levelset Cross-Sections', fontsize=16)
    
    nx, ny, nz = levelset.shape
    
    # XY cross-section (middle Z)
    mid_z = nz // 2
    xy_levelset = levelset[:, :, mid_z]
    xy_tags = bc_tags[:, :, mid_z]
    
    im1 = axes[0,0].imshow(xy_levelset.T, origin='lower', cmap='RdBu', aspect='auto')
    axes[0,0].set_title(f'XY Cross-section (Z={mid_z})')
    axes[0,0].set_xlabel('X')
    axes[0,0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0,0], label='Levelset')
    
    # XY boundary tags
    im2 = axes[0,1].imshow(xy_tags.T, origin='lower', cmap='viridis', aspect='auto')
    axes[0,1].set_title(f'XY Boundary Tags (Z={mid_z})')
    axes[0,1].set_xlabel('X')
    axes[0,1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[0,1], label='BC Tags')
    
    # XZ cross-section (middle Y)
    mid_y = ny // 2  
    xz_levelset = levelset[:, mid_y, :]
    xz_tags = bc_tags[:, mid_y, :]
    
    im3 = axes[1,0].imshow(xz_levelset.T, origin='lower', cmap='RdBu', aspect='auto')
    axes[1,0].set_title(f'XZ Cross-section (Y={mid_y})')
    axes[1,0].set_xlabel('X')
    axes[1,0].set_ylabel('Z')
    plt.colorbar(im3, ax=axes[1,0], label='Levelset')
    
    # XZ boundary tags
    im4 = axes[1,1].imshow(xz_tags.T, origin='lower', cmap='viridis', aspect='auto')
    axes[1,1].set_title(f'XZ Boundary Tags (Y={mid_y})')
    axes[1,1].set_xlabel('X')
    axes[1,1].set_ylabel('Z')
    plt.colorbar(im4, ax=axes[1,1], label='BC Tags')
    
    plt.tight_layout()
    plt.show()
    
    print("📊 Cross-section views displayed")

if __name__ == "__main__":
    print("🎨 Visualizing Complete JAX-Fluids Levelset")
    print("=" * 50)
    
    # Create 3D visualization
    create_visualization()
    
    # Create 2D cross-section views
    create_cross_section_views()
    
    print("\n✅ Visualization complete!") 