#!/usr/bin/env python3
"""
Proper JAX-Fluids Visualization for Rocket Nozzle Internal Flow
Loads data from correct HDF5 structure and includes geometry
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
        
        # Load domain information
        if 'domain' in f:
            domain_info = {}
            for key in ['gridX', 'gridY', 'gridZ']:
                if key in f['domain']:
                    domain_info[key] = f['domain'][key][...]
            data['domain_info'] = domain_info
            
        # Check for levelset in quantities
        if 'quantities' in f and 'levelset' in f['quantities']:
            data['levelset'] = f['quantities']['levelset'][...]
            print(f"  levelset: shape={data['levelset'].shape}, range=[{data['levelset'].min():.6f}, {data['levelset'].max():.6f}]")
    
    return data

def load_rocket_levelset():
    """Load the rocket levelset/SDF for geometry visualization"""
    # Try to find the SDF file from our configuration
    sdf_file = Path("../../hardcoded_run/attempt1/rocket_case/20250728_003202/Rocket Engine_sdf_matrix.npy")
    
    if not sdf_file.exists():
        # Try alternative paths
        alternative_paths = [
            "masks/Rocket Engine_sdf_matrix.npy",
            "../hardcoded_run/attempt1/rocket_case/20250728_003202/Rocket Engine_sdf_matrix.npy"
        ]
        
        for path in alternative_paths:
            if Path(path).exists():
                sdf_file = Path(path)
                break
    
    if sdf_file.exists():
        print(f"Loading rocket SDF from: {sdf_file}")
        sdf = np.load(sdf_file)
        print(f"  SDF shape: {sdf.shape}, range=[{sdf.min():.3f}, {sdf.max():.3f}]")
        return sdf
    else:
        print("WARNING: Could not find rocket SDF file")
        return None

def create_proper_jaxfluids_visualization():
    """Create proper visualization of JAX-Fluids rocket simulation"""
    print("🚀 PROPER JAX-FLUIDS ROCKET VISUALIZATION")
    print("=" * 60)
    
    # Find latest data file
    data_dir = Path("output/rocket_nozzle_internal_supersonic_production/domain")
    if not data_dir.exists():
        print("❌ No output directory found!")
        return
    
    h5_files = list(data_dir.glob("*.h5"))
    if not h5_files:
        print("❌ No HDF5 files found!")
        return
    
    latest_file = sorted(h5_files)[-1]
    
    # Load simulation data
    data = load_jaxfluids_data_proper(latest_file)
    
    # Load rocket geometry
    rocket_sdf = load_rocket_levelset()
    
    # Create visualization
    print("\n🎨 Creating visualization...")
    
    # Domain bounds and grid from our configuration
    domain_bounds = [-200.0, -800.0, -800.0, 1800.0, 800.0, 800.0]  # x_min, y_min, z_min, x_max, y_max, z_max
    
    # Get actual grid shape from data
    if 'density' in data:
        grid_shape = data['density'].shape  # Should be (64, 64, 128) as we saw
        print(f"Grid shape: {grid_shape}")
    else:
        print("❌ No density data found!")
        return
    
    # Create coordinate arrays
    nz, ny, nx = grid_shape  # Note: JAX-Fluids uses (z, y, x) ordering
    x = np.linspace(domain_bounds[0], domain_bounds[3], nx)
    y = np.linspace(domain_bounds[1], domain_bounds[4], ny)
    z = np.linspace(domain_bounds[2], domain_bounds[5], nz)
    
    # Create PyVista structured grid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create structured grid
    grid = pv.StructuredGrid()
    points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    grid.points = points
    grid.dimensions = [nx, ny, nz]
    
    # Add flow data to grid
    for var_name, var_data in data.items():
        if isinstance(var_data, np.ndarray) and var_data.shape == grid_shape:
            # Transpose to match PyVista ordering
            grid[var_name] = var_data.transpose(2, 1, 0).ravel()  # (z,y,x) -> (x,y,z)
    
    # Add rocket geometry if available
    if rocket_sdf is not None and rocket_sdf.shape == grid_shape:
        grid['rocket_levelset'] = rocket_sdf.transpose(2, 1, 0).ravel()
    
    # Create plotter
    plotter = pv.Plotter(window_size=[1600, 1000])
    
    # Visualize the rocket geometry first
    if 'rocket_levelset' in grid.array_names:
        print("🔧 Adding rocket geometry...")
        
        # Extract rocket surface (levelset = 0)
        rocket_surface = grid.contour([0.0], scalars='rocket_levelset')
        if rocket_surface.n_points > 0:
            plotter.add_mesh(rocket_surface, 
                           color='gray', 
                           opacity=0.7,
                           label='Rocket Nozzle')
    
    # Visualize flow field
    print("🌊 Adding flow field...")
    
    # Choose which variable to visualize
    flow_vars = ['density', 'pressure', 'mach_number', 'temperature']
    selected_var = None
    
    for var in flow_vars:
        if var in grid.array_names:
            selected_var = var
            break
    
    if selected_var:
        print(f"📊 Visualizing: {selected_var}")
        
        # Create slices through the domain
        center_x = (domain_bounds[0] + domain_bounds[3]) / 2
        center_y = (domain_bounds[1] + domain_bounds[4]) / 2
        center_z = (domain_bounds[2] + domain_bounds[5]) / 2
        
        # X-slice (longitudinal cut through nozzle)
        slice_x = grid.slice(normal='x', origin=[center_x, center_y, center_z])
        if slice_x.n_points > 0:
            plotter.add_mesh(slice_x,
                           scalars=selected_var,
                           cmap='viridis',
                           opacity=0.9,
                           show_scalar_bar=True,
                           scalar_bar_args={'title': selected_var.title()})
        
        # Y-slice (side view)
        slice_y = grid.slice(normal='y', origin=[center_x, center_y, center_z])
        if slice_y.n_points > 0:
            plotter.add_mesh(slice_y,
                           scalars=selected_var,
                           cmap='plasma',
                           opacity=0.7)
    
    # Add domain outline
    plotter.add_mesh(grid.outline(), color='black', line_width=2)
    
    # Setup view
    plotter.add_axes(line_width=3)
    plotter.add_text("JAX-Fluids Rocket Nozzle Internal Supersonic Flow\n(Proper Data Loading)", 
                    font_size=14, position='upper_left')
    
    # Set camera for good view of rocket nozzle
    plotter.camera_position = [
        (2000, 1000, 1000),  # camera position
        (400, 0, 0),         # focal point (roughly center of nozzle)
        (0, 0, 1)           # view up
    ]
    
    print("🎮 Controls:")
    print("   Mouse: Rotate/Pan/Zoom")
    print("   'q': Quit")
    print("   'r': Reset view")
    print("   'w': Wireframe mode")
    print("   's': Surface mode")
    
    # Show the visualization
    plotter.show()

if __name__ == "__main__":
    try:
        create_proper_jaxfluids_visualization()
        print("✅ Visualization complete!")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc() 