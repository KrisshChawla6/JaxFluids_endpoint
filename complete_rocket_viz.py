#!/usr/bin/env python3
"""
Complete JAX-Fluids Rocket Visualization
Shows SDF geometry + inlet/outlet markers + flow field
"""

import pyvista as pv
import numpy as np
import h5py
from pathlib import Path

def load_jaxfluids_data_proper(h5_file):
    """Load JAX-Fluids data from proper HDF5 structure"""
    data = {}
    
    with h5py.File(h5_file, 'r') as f:
        print(f"Loading data from: {h5_file}")
        
        # Load primitives (density, pressure, velocity, temperature)
        if 'primitives' in f:
            for var in f['primitives'].keys():
                data[var] = f['primitives'][var][...]
                print(f"  {var}: shape={data[var].shape}, range=[{data[var].min():.6f}, {data[var].max():.6f}]")
        
        # Load miscellaneous (mach number, etc.)
        if 'miscellaneous' in f:
            for var in f['miscellaneous'].keys():
                data[var] = f['miscellaneous'][var][...]
                print(f"  {var}: shape={data[var].shape}, range=[{data[var].min():.6f}, {data[var].max():.6f}]")
    
    return data

def load_rocket_geometry_and_masks():
    """Load rocket SDF and inlet/outlet masks"""
    geometry_data = {}
    
    # Load rocket SDF
    sdf_paths = [
        "intelligent_boundary_conditions/working/hardcoded_run/attempt1/rocket_case/20250728_003202/Rocket Engine_sdf_matrix.npy",
        "intelligent_boundary_conditions/working/rocket_simulation_final/masks/Rocket Engine_sdf_matrix.npy",
        "intelligent_boundary_conditions/working/rocket_simulation_final/../hardcoded_run/attempt1/rocket_case/20250728_003202/Rocket Engine_sdf_matrix.npy"
    ]
    
    for sdf_path in sdf_paths:
        if Path(sdf_path).exists():
            print(f"Loading rocket SDF from: {sdf_path}")
            geometry_data['sdf'] = np.load(sdf_path)
            print(f"  SDF shape: {geometry_data['sdf'].shape}, range=[{geometry_data['sdf'].min():.3f}, {geometry_data['sdf'].max():.3f}]")
            break
    
    # Load inlet/outlet masks
    mask_paths = [
        ("intelligent_boundary_conditions/working/rocket_simulation_final/masks/inlet_boundary_mask.npy", "inlet"),
        ("intelligent_boundary_conditions/working/rocket_simulation_final/masks/outlet_boundary_mask.npy", "outlet"),
        ("intelligent_boundary_conditions/working/hardcoded_run/attempt1/rocket_nozzle_jaxfluids_simulation/inlet_boundary_mask.npy", "inlet"),
        ("intelligent_boundary_conditions/working/hardcoded_run/attempt1/rocket_nozzle_jaxfluids_simulation/outlet_boundary_mask.npy", "outlet")
    ]
    
    for mask_path, mask_type in mask_paths:
        if Path(mask_path).exists():
            print(f"Loading {mask_type} mask from: {mask_path}")
            mask_data = np.load(mask_path)
            geometry_data[f'{mask_type}_mask'] = mask_data
            print(f"  {mask_type} mask: shape={mask_data.shape}, active points={np.sum(mask_data):,}")
    
    return geometry_data

def create_complete_rocket_visualization():
    """Create complete visualization with SDF + masks + flow field"""
    print("🚀 COMPLETE ROCKET VISUALIZATION")
    print("=" * 60)
    
    # Find latest data file
    data_dir = Path("intelligent_boundary_conditions/working/rocket_simulation_final/output/rocket_nozzle_internal_supersonic_production/domain")
    if not data_dir.exists():
        print("❌ No output directory found!")
        return
    
    h5_files = list(data_dir.glob("*.h5"))
    if not h5_files:
        print("❌ No HDF5 files found!")
        return
    
    latest_file = sorted(h5_files)[-1]
    
    # Load simulation data
    flow_data = load_jaxfluids_data_proper(latest_file)
    
    # Load geometry and masks
    geometry_data = load_rocket_geometry_and_masks()
    
    print(f"\n🎨 Creating complete visualization...")
    
    # Domain bounds and grid from our configuration
    domain_bounds = [-200.0, -800.0, -800.0, 1800.0, 800.0, 800.0]
    
    # Get grid shape from flow data
    if 'density' in flow_data:
        grid_shape = flow_data['density'].shape  # (64, 64, 128)
        print(f"Grid shape: {grid_shape}")
    else:
        print("❌ No density data found!")
        return
    
    # Create coordinate arrays
    nz, ny, nx = grid_shape  # JAX-Fluids uses (z, y, x) ordering
    x = np.linspace(domain_bounds[0], domain_bounds[3], nx)
    y = np.linspace(domain_bounds[1], domain_bounds[4], ny)
    z = np.linspace(domain_bounds[2], domain_bounds[5], nz)
    
    # Create PyVista structured grid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    grid = pv.StructuredGrid()
    points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    grid.points = points
    grid.dimensions = [nx, ny, nz]
    
    # Add flow data to grid
    for var_name, var_data in flow_data.items():
        if isinstance(var_data, np.ndarray) and var_data.shape == grid_shape:
            grid[var_name] = var_data.transpose(2, 1, 0).ravel()  # (z,y,x) -> (x,y,z)
    
    # Add geometry data to grid
    if 'sdf' in geometry_data:
        sdf = geometry_data['sdf']
        # Handle different SDF orientations
        if sdf.shape == (128, 64, 64):  # (x, y, z)
            sdf_reoriented = sdf.transpose(2, 1, 0)  # -> (z, y, x)
        elif sdf.shape == grid_shape:  # (z, y, x)
            sdf_reoriented = sdf
        else:
            print(f"Warning: SDF shape {sdf.shape} doesn't match grid {grid_shape}")
            sdf_reoriented = None
        
        if sdf_reoriented is not None:
            grid['rocket_sdf'] = sdf_reoriented.ravel()
    
    # Add inlet/outlet masks
    for mask_type in ['inlet', 'outlet']:
        mask_key = f'{mask_type}_mask'
        if mask_key in geometry_data:
            mask = geometry_data[mask_key].astype(float)
            # Handle different mask orientations
            if mask.shape == grid_shape:
                grid[mask_key] = mask.transpose(2, 1, 0).ravel()
            else:
                print(f"Warning: {mask_type} mask shape {mask.shape} doesn't match grid")
    
    # Create plotter
    plotter = pv.Plotter(window_size=[1800, 1200])
    
    print("🔧 Adding rocket geometry...")
    
    # 1. Add rocket nozzle surface (SDF = 0)
    if 'rocket_sdf' in grid.array_names:
        rocket_surface = grid.contour([0.0], scalars='rocket_sdf')
        if rocket_surface.n_points > 0:
            plotter.add_mesh(rocket_surface, 
                           color='lightgray', 
                           opacity=0.6,
                           label='Rocket Nozzle')
            print(f"   ✅ Rocket surface: {rocket_surface.n_points:,} points")
    
    print("🔴 Adding inlet marker...")
    
    # 2. Add inlet region (red)
    if 'inlet_mask' in grid.array_names:
        inlet_threshold = grid.threshold(0.5, scalars='inlet_mask')
        if inlet_threshold.n_points > 0:
            plotter.add_mesh(inlet_threshold,
                           color='red',
                           opacity=0.8,
                           label='Inlet (Virtual Face)')
            print(f"   ✅ Inlet region: {inlet_threshold.n_points:,} points")
    
    print("🟢 Adding outlet marker...")
    
    # 3. Add outlet region (green)
    if 'outlet_mask' in grid.array_names:
        outlet_threshold = grid.threshold(0.5, scalars='outlet_mask')
        if outlet_threshold.n_points > 0:
            plotter.add_mesh(outlet_threshold,
                           color='green',
                           opacity=0.8,
                           label='Outlet (Virtual Face)')
            print(f"   ✅ Outlet region: {outlet_threshold.n_points:,} points")
    
    print("🌊 Adding flow field slices...")
    
    # 4. Add flow field visualization
    flow_vars = ['pressure', 'mach_number', 'temperature', 'density']
    selected_var = None
    
    for var in flow_vars:
        if var in grid.array_names:
            selected_var = var
            break
    
    if selected_var:
        print(f"   📊 Visualizing: {selected_var}")
        
        # Longitudinal slice through nozzle center
        center_x = (domain_bounds[0] + domain_bounds[3]) / 2
        center_y = (domain_bounds[1] + domain_bounds[4]) / 2
        center_z = (domain_bounds[2] + domain_bounds[5]) / 2
        
        # X-slice (shows flow through nozzle)
        slice_x = grid.slice(normal='x', origin=[center_x, center_y, center_z])
        if slice_x.n_points > 0:
            plotter.add_mesh(slice_x,
                           scalars=selected_var,
                           cmap='viridis',
                           opacity=0.7,
                           show_scalar_bar=True,
                           scalar_bar_args={'title': f'{selected_var.title()}'})
        
        # Y-slice (side view)
        slice_y = grid.slice(normal='y', origin=[center_x, center_y, center_z])
        if slice_y.n_points > 0:
            plotter.add_mesh(slice_y,
                           scalars=selected_var,
                           cmap='plasma',
                           opacity=0.5)
    
    # 5. Add domain outline
    plotter.add_mesh(grid.outline(), color='black', line_width=2)
    
    # Setup view and labels
    plotter.add_axes(line_width=4)
    plotter.add_text("Complete JAX-Fluids Rocket Nozzle Visualization\n"
                    "Gray: Rocket Walls | Red: Inlet | Green: Outlet | Color: Flow Field", 
                    font_size=12, position='upper_left')
    
    # Add legend
    plotter.add_text("🔴 Inlet (Virtual Face)\n🟢 Outlet (Virtual Face)\n⚫ Rocket Nozzle Walls", 
                    font_size=10, position='lower_left')
    
    # Set camera for optimal rocket nozzle view
    plotter.camera_position = [
        (2500, 1200, 1000),  # camera position
        (400, 0, -200),      # focal point (center of nozzle)
        (0, 0, 1)           # view up
    ]
    
    print("\n🎮 Controls:")
    print("   Mouse: Rotate/Pan/Zoom")
    print("   'q': Quit")
    print("   'r': Reset view")
    print("   'w': Wireframe mode")
    print("   's': Surface mode")
    
    # Show the complete visualization
    plotter.show()

if __name__ == "__main__":
    try:
        create_complete_rocket_visualization()
        print("✅ Complete visualization finished!")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc() 