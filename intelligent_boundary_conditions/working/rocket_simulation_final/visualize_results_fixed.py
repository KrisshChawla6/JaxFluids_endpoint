#!/usr/bin/env python3
"""
Fixed 3D Visualization of JAX-Fluids Rocket Simulation Results
Ensures visible data with proper scaling and visualization
"""

import pyvista as pv
import numpy as np
from pathlib import Path
import h5py

def visualize_results_fixed():
    """Fixed visualization that will definitely show something"""
    print("🎯 FIXED JAX-FLUIDS VISUALIZATION")
    print("=" * 50)
    
    # Find the latest data file
    data_dir = Path("output/rocket_nozzle_internal_supersonic_production/domain")
    
    if not data_dir.exists():
        print("❌ No output data directory found!")
        return
    
    # Get all XDMF files
    xdmf_files = list(data_dir.glob("*.xdmf"))
    if not xdmf_files:
        print("❌ No XDMF files found!")
        return
    
    # Use the time series file if it exists, otherwise use latest
    time_series_file = data_dir / "data_time_series.xdmf"
    if time_series_file.exists():
        print(f"📊 Loading time series: {time_series_file}")
        mesh = pv.read(str(time_series_file))
    else:
        # Use the latest single file
        latest_file = sorted([f for f in xdmf_files if "time_series" not in f.name])[-1]
        print(f"📊 Loading latest file: {latest_file}")
        mesh = pv.read(str(latest_file))
    
    print(f"✅ Loaded mesh:")
    print(f"   Points: {mesh.n_points:,}")
    print(f"   Cells: {mesh.n_cells:,}")
    print(f"   Arrays: {mesh.array_names}")
    print(f"   Bounds: {mesh.bounds}")
    
    # Create plotter
    plotter = pv.Plotter(window_size=[1400, 900])
    
    # Try different visualization approaches
    if mesh.array_names:
        # Use the first available array
        array_name = mesh.array_names[0]
        print(f"📈 Visualizing: {array_name}")
        
        # Get data range
        data_range = mesh.get_array(array_name).min(), mesh.get_array(array_name).max()
        print(f"   Data range: {data_range[0]:.6f} to {data_range[1]:.6f}")
        
        # Approach 1: Volume rendering with clipping
        if mesh.n_cells > 100000:  # For large datasets, use clipping
            print("🔄 Using clipped visualization for large dataset...")
            
            # Create a box to clip the data
            bounds = mesh.bounds
            clip_box = pv.Box(bounds=[
                bounds[0] + 0.1 * (bounds[1] - bounds[0]),  # x_min + 10%
                bounds[1] - 0.1 * (bounds[1] - bounds[0]),  # x_max - 10%
                bounds[2], bounds[3],  # y full
                bounds[4], bounds[5]   # z full
            ])
            
            clipped = mesh.clip_box(clip_box)
            
            # Add the clipped mesh
            plotter.add_mesh(clipped, 
                           scalars=array_name,
                           cmap='viridis',
                           opacity=0.8,
                           show_scalar_bar=True)
            
            # Add outline
            plotter.add_mesh(mesh.outline(), color='red', line_width=3)
            
        else:
            # For smaller datasets, show directly
            print("🔄 Direct visualization...")
            plotter.add_mesh(mesh, 
                           scalars=array_name,
                           cmap='viridis',
                           opacity=0.7,
                           show_scalar_bar=True)
        
        # Add slices for internal view
        print("🔄 Adding cross-sectional slices...")
        bounds = mesh.bounds
        center_x = (bounds[0] + bounds[1]) / 2
        center_y = (bounds[2] + bounds[3]) / 2
        center_z = (bounds[4] + bounds[5]) / 2
        
        # X slice (through center)
        slice_x = mesh.slice(normal='x', origin=[center_x, center_y, center_z])
        if slice_x.n_points > 0:
            plotter.add_mesh(slice_x, 
                           scalars=array_name, 
                           cmap='plasma',
                           opacity=1.0,
                           name='x_slice')
        
        # Y slice
        slice_y = mesh.slice(normal='y', origin=[center_x, center_y, center_z])
        if slice_y.n_points > 0:
            plotter.add_mesh(slice_y, 
                           scalars=array_name, 
                           cmap='coolwarm',
                           opacity=0.8,
                           name='y_slice')
    
    else:
        print("📊 No data arrays found, showing mesh structure...")
        plotter.add_mesh(mesh, color='lightblue', opacity=0.6)
        plotter.add_mesh(mesh.outline(), color='red', line_width=3)
    
    # Set up the view
    plotter.add_axes(line_width=5, labels_off=False)
    plotter.add_text("JAX-Fluids Rocket Nozzle - Internal Supersonic Flow", 
                    font_size=16, position='upper_left')
    
    # Set camera position
    plotter.camera_position = 'iso'
    plotter.reset_camera()
    
    # Add some lighting
    plotter.add_light(pv.Light(position=(2, 2, 2), light_type='scene light'))
    
    print("🎮 Controls:")
    print("   - Mouse: Rotate/Zoom")
    print("   - 'q': Quit")
    print("   - 'r': Reset view")
    print("   - 's': Surface mode")
    print("   - 'w': Wireframe mode")
    
    # Show the plot
    plotter.show(window_size=[1400, 900])

def quick_mesh_info():
    """Quick info about the mesh data"""
    print("📋 MESH INFO")
    print("=" * 30)
    
    data_dir = Path("output/rocket_nozzle_internal_supersonic_production/domain")
    
    if not data_dir.exists():
        print("❌ No data directory!")
        return
    
    # Try to load one file to get info
    xdmf_files = list(data_dir.glob("data_*.xdmf"))
    if xdmf_files:
        try:
            mesh = pv.read(str(xdmf_files[0]))
            print(f"📊 Sample file: {xdmf_files[0].name}")
            print(f"   Grid type: {type(mesh).__name__}")
            print(f"   Points: {mesh.n_points:,}")
            print(f"   Cells: {mesh.n_cells:,}")
            print(f"   Cell types: {set(mesh.celltypes)}")
            print(f"   Bounds: {[f'{b:.1f}' for b in mesh.bounds]}")
            print(f"   Arrays: {mesh.array_names}")
            
            # Show data ranges
            for name in mesh.array_names:
                data = mesh.get_array(name)
                print(f"   {name}: [{data.min():.6f}, {data.max():.6f}]")
                
        except Exception as e:
            print(f"❌ Error loading mesh: {e}")

def create_simple_test_viz():
    """Create a simple test visualization to make sure PyVista works"""
    print("🧪 CREATING TEST VISUALIZATION")
    print("=" * 40)
    
    # Create a simple test mesh
    sphere = pv.Sphere(radius=1.0, center=(0, 0, 0))
    
    # Add some test data
    sphere['test_data'] = np.sin(sphere.points[:, 0]) * np.cos(sphere.points[:, 1])
    
    # Create plotter
    plotter = pv.Plotter(window_size=[800, 600])
    plotter.add_mesh(sphere, scalars='test_data', cmap='viridis', show_scalar_bar=True)
    plotter.add_text("PyVista Test - If you see this, PyVista works!", font_size=14)
    plotter.add_axes()
    
    print("🎮 Showing test visualization...")
    plotter.show()

if __name__ == "__main__":
    try:
        # First check mesh info
        quick_mesh_info()
        
        print("\n" + "="*60)
        
        # Try the fixed visualization
        visualize_results_fixed()
        
    except Exception as e:
        print(f"❌ Main visualization failed: {e}")
        print("🧪 Trying simple test...")
        try:
            create_simple_test_viz()
        except Exception as e2:
            print(f"❌ Even test failed: {e2}")
            import traceback
            traceback.print_exc()
        
    print("\n✅ Visualization attempt complete!") 