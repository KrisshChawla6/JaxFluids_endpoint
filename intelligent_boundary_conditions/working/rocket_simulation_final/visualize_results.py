#!/usr/bin/env python3
"""
Proper JAX-Fluids Visualization for Rocket Nozzle Internal Flow
Loads data from correct HDF5 structure and includes geometry
"""

import pyvista as pv
import numpy as np
from pathlib import Path
import h5py

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

def create_structured_grid(data, domain_bounds, grid_shape):
    """Create PyVista structured grid from JAX-Fluids data"""
    nx, ny, nz = grid_shape
    x_min, y_min, z_min, x_max, y_max, z_max = domain_bounds
    
    # Create coordinate arrays
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny) 
    z = np.linspace(z_min, z_max, nz)
    
    # Create structured grid
    grid = pv.StructuredGrid()
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Set points (need to reshape for PyVista)
    points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    grid.points = points
    grid.dimensions = [nx, ny, nz]
    
    return grid

def visualize_simulation_results():
    """Main visualization function"""
    print("🎯 VISUALIZING JAX-FLUIDS ROCKET SIMULATION RESULTS")
    print("=" * 60)
    
    # Find the latest data file
    data_dir = Path("output/rocket_nozzle_internal_supersonic_production/domain")
    
    if not data_dir.exists():
        print("❌ No output data directory found!")
        return
    
    # Get all HDF5 files and use the latest one
    h5_files = list(data_dir.glob("*.h5"))
    if not h5_files:
        print("❌ No HDF5 data files found!")
        return
    
    # Use the last (latest time) data file
    latest_file = sorted(h5_files)[-1]
    print(f"📁 Loading data from: {latest_file}")
    
    # Try to load using PyVista's XDMF reader first
    try:
        xdmf_file = latest_file.with_suffix('.xdmf')
        if xdmf_file.exists():
            print(f"📊 Loading XDMF file: {xdmf_file}")
            mesh = pv.read(str(xdmf_file))
            
            print(f"✅ Loaded mesh with {mesh.n_points} points and {mesh.n_cells} cells")
            print(f"Available arrays: {mesh.array_names}")
            
            # Create visualization
            plotter = pv.Plotter(window_size=[1200, 800])
            
            # Try to find density field
            density_names = ['density', 'rho', 'Density', 'DENSITY']
            density_array = None
            
            for name in density_names:
                if name in mesh.array_names:
                    density_array = name
                    break
            
            if density_array:
                print(f"📈 Visualizing density field: {density_array}")
                
                # Add mesh with density coloring
                plotter.add_mesh(mesh, 
                               scalars=density_array,
                               cmap='viridis',
                               opacity=0.8,
                               show_scalar_bar=True,
                               scalar_bar_args={'title': 'Density'})
                
                # Add some slices for internal view
                slices = mesh.slice_orthogonal(x=500.0, y=0.0, z=0.0)
                plotter.add_mesh(slices, 
                               scalars=density_array, 
                               cmap='plasma',
                               opacity=0.9)
                
            else:
                # Just show the mesh structure
                print("📊 No density field found, showing mesh structure")
                plotter.add_mesh(mesh, opacity=0.5, color='lightblue')
            
            # Set up the view
            plotter.add_axes()
            plotter.camera_position = 'iso'
            plotter.add_text("JAX-Fluids Rocket Nozzle Simulation", font_size=14)
            
            print("🎮 Opening interactive viewer...")
            print("   - Use mouse to rotate/zoom")
            print("   - Press 'q' to quit")
            
            plotter.show()
            
        else:
            print("❌ No XDMF file found, trying direct HDF5 loading...")
            raise FileNotFoundError("XDMF not found")
            
    except Exception as e:
        print(f"⚠️  XDMF loading failed: {e}")
        print("🔄 Trying direct HDF5 approach...")
        
        # Fallback: Direct HDF5 loading
        data = load_jax_fluids_data(latest_file)
        
        if data:
            print("📊 Creating visualization from raw data...")
            
            # Create a simple visualization of the domain
            domain_bounds = [-200.0, -800.0, -800.0, 1800.0, 800.0, 800.0]
            grid_shape = (128, 64, 64)
            
            grid = create_structured_grid(data, domain_bounds, grid_shape)
            
            # Try to add any available scalar data
            for key, array in data.items():
                if len(array.shape) == 3 and array.shape == grid_shape:
                    grid[key] = array.ravel()
                    print(f"Added {key} to visualization")
            
            # Simple visualization
            plotter = pv.Plotter()
            plotter.add_mesh(grid.outline(), color='black', line_width=2)
            
            # Show some slices through the domain
            if grid.array_names:
                slice_x = grid.slice(normal='x', origin=[500, 0, 0])
                plotter.add_mesh(slice_x, opacity=0.8, cmap='viridis')
            
            plotter.add_axes()
            plotter.camera_position = 'iso'
            plotter.show()
        
        else:
            print("❌ Could not load any data for visualization")

def quick_data_summary():
    """Quick summary of available data"""
    print("\n📋 DATA SUMMARY")
    print("=" * 40)
    
    data_dir = Path("output/rocket_nozzle_internal_supersonic_production/domain")
    
    if data_dir.exists():
        h5_files = list(data_dir.glob("*.h5"))
        xdmf_files = list(data_dir.glob("*.xdmf"))
        
        print(f"📁 Found {len(h5_files)} HDF5 files")
        print(f"📁 Found {len(xdmf_files)} XDMF files")
        
        if h5_files:
            print(f"🕐 Time range: {h5_files[0].stem.split('_')[-1]} → {h5_files[-1].stem.split('_')[-1]}")
            
            # Check file sizes
            total_size = sum(f.stat().st_size for f in h5_files)
            print(f"💾 Total data size: {total_size/1e6:.1f} MB")

if __name__ == "__main__":
    try:
        quick_data_summary()
        visualize_simulation_results()
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n✅ Visualization complete!") 