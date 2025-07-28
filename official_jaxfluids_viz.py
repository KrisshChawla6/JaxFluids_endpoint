#!/usr/bin/env python3
"""
Official JAX-Fluids Rocket Visualization
Uses the official jaxfluids_postprocess functions for proper visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Use the official JAX-Fluids postprocessing
from jaxfluids_postprocess import load_data, create_2D_animation, create_2D_figure

def create_official_jaxfluids_visualization():
    """Create visualization using official JAX-Fluids functions"""
    print("🚀 OFFICIAL JAX-FLUIDS ROCKET VISUALIZATION")
    print("=" * 60)
    
    # Path to our simulation data
    data_path = Path("intelligent_boundary_conditions/working/rocket_simulation_final/output/rocket_nozzle_internal_supersonic_production/domain")
    
    if not data_path.exists():
        print("❌ Simulation data not found!")
        return
    
    print(f"📊 Loading data from: {data_path}")
    
    # Load data using official JAX-Fluids function
    quantities = ['density', 'velocity', 'pressure', 'temperature', 'mach_number']
    
    try:
        jxf_data = load_data(str(data_path), quantities)
        
        cell_centers = jxf_data.cell_centers
        data = jxf_data.data
        times = jxf_data.times
        
        print(f"✅ Loaded {len(quantities)} quantities over {len(times)} time steps")
        print(f"Domain bounds: X=[{cell_centers[0].min():.1f}, {cell_centers[0].max():.1f}]")
        print(f"Domain bounds: Y=[{cell_centers[1].min():.1f}, {cell_centers[1].max():.1f}]") 
        print(f"Domain bounds: Z=[{cell_centers[2].min():.1f}, {cell_centers[2].max():.1f}]")
        
        # Display data ranges
        for qty, qty_data in data.items():
            if hasattr(qty_data, 'shape'):
                if len(qty_data.shape) > 3:  # Vector quantity
                    print(f"{qty}: shape={qty_data.shape}, range=[{qty_data.min():.6f}, {qty_data.max():.6f}]")
                else:  # Scalar quantity
                    print(f"{qty}: shape={qty_data.shape}, range=[{qty_data.min():.6f}, {qty_data.max():.6f}]")
        
        # Create output directory
        os.makedirs("official_viz", exist_ok=True)
        
        print("\n📊 Creating 2D figure slices...")
        
        # Create 2D slices through the rocket nozzle
        plot_dict = {
            "density": data["density"],
            "pressure": data["pressure"], 
            "mach_number": data["mach_number"],
            "temperature": data["temperature"]
        }
        
        # X-Y plane slice (looking down the nozzle axis)
        create_2D_figure(
            plot_dict,
            nrows_ncols=(2, 2),
            cell_centers=cell_centers,
            plane="xy", 
            plane_value=0.0,  # Center slice
            time_step=-1,     # Latest time step
            dpi=300,
            save_fig="official_viz/rocket_nozzle_xy_slice.png",
            fig_args={"figsize": (12, 10)}
        )
        print("✅ X-Y plane slice saved")
        
        # X-Z plane slice (side view of nozzle)
        create_2D_figure(
            plot_dict,
            nrows_ncols=(2, 2),
            cell_centers=cell_centers,
            plane="xz",
            plane_value=0.0,  # Center slice
            time_step=-1,
            dpi=300,
            save_fig="official_viz/rocket_nozzle_xz_slice.png",
            fig_args={"figsize": (12, 10)}
        )
        print("✅ X-Z plane slice saved")
        
        # Y-Z plane slice (cross-section view)
        create_2D_figure(
            plot_dict,
            nrows_ncols=(2, 2), 
            cell_centers=cell_centers,
            plane="yz",
            plane_value=400.0,  # Through nozzle center
            time_step=-1,
            dpi=300,
            save_fig="official_viz/rocket_nozzle_yz_slice.png",
            fig_args={"figsize": (12, 10)}
        )
        print("✅ Y-Z plane slice saved")
        
        print("\n🎬 Creating animation...")
        
        # Create animation of pressure evolution
        animation_plot_dict = {
            "pressure": data["pressure"],
            "mach_number": data["mach_number"]
        }
        
        create_2D_animation(
            animation_plot_dict,
            cell_centers,
            times,
            nrows_ncols=(1, 2),
            plane="xz",
            plane_value=0.0,
            interval=200,  # 200ms between frames
            save_png="official_viz/frames",
            save_mp4="official_viz/rocket_nozzle_evolution.mp4",
            fig_args={"figsize": (16, 6)},
            dpi=200
        )
        print("✅ Animation saved")
        
        # Create velocity vector plot
        print("\n🌊 Creating velocity field visualization...")
        
        if "velocity" in data:
            velocity_plot_dict = {
                "u": data["velocity"][:,:,:,:,0],  # X-velocity
                "v": data["velocity"][:,:,:,:,1],  # Y-velocity  
                "w": data["velocity"][:,:,:,:,2],  # Z-velocity
                "magnitude": np.sqrt(
                    data["velocity"][:,:,:,:,0]**2 + 
                    data["velocity"][:,:,:,:,1]**2 + 
                    data["velocity"][:,:,:,:,2]**2
                )
            }
            
                         create_2D_figure(
                 velocity_plot_dict,
                 nrows_ncols=(2, 2),
                 cell_centers=cell_centers,
                 plane="xz",
                 plane_value=0.0,
                 time_step=-1,
                 dpi=300,
                 save_fig="official_viz/rocket_nozzle_velocity.png",
                 fig_args={"figsize": (12, 10)}
             )
            print("✅ Velocity field saved")
        
        print(f"\n🎉 Official JAX-Fluids visualization complete!")
        print(f"📁 Results saved in: ./official_viz/")
        print(f"   - rocket_nozzle_xy_slice.png (top view)")
        print(f"   - rocket_nozzle_xz_slice.png (side view)")  
        print(f"   - rocket_nozzle_yz_slice.png (cross-section)")
        print(f"   - rocket_nozzle_velocity.png (velocity field)")
        print(f"   - rocket_nozzle_evolution.mp4 (animation)")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()

def create_inlet_outlet_analysis():
    """Analyze the inlet/outlet regions using our masks"""
    print("\n🔍 INLET/OUTLET ANALYSIS")
    print("=" * 40)
    
    # Load our inlet/outlet masks
    mask_paths = [
        ("intelligent_boundary_conditions/working/rocket_simulation_final/masks/inlet_boundary_mask.npy", "inlet"),
        ("intelligent_boundary_conditions/working/rocket_simulation_final/masks/outlet_boundary_mask.npy", "outlet")
    ]
    
    for mask_path, mask_type in mask_paths:
        if Path(mask_path).exists():
            mask = np.load(mask_path)
            active_points = np.sum(mask)
            print(f"✅ {mask_type.title()}: {active_points:,} active grid points")
            
            # Find centroid
            indices = np.where(mask)
            if len(indices[0]) > 0:
                # Convert indices to physical coordinates
                # Domain: [-200, -800, -800] to [1800, 800, 800]
                # Grid: (128, 64, 64)
                x_coords = -200 + (indices[0] / 127) * 2000  # X range
                y_coords = -800 + (indices[1] / 63) * 1600   # Y range  
                z_coords = -800 + (indices[2] / 63) * 1600   # Z range
                
                centroid_x = np.mean(x_coords)
                centroid_y = np.mean(y_coords)
                centroid_z = np.mean(z_coords)
                
                print(f"   Centroid: ({centroid_x:.1f}, {centroid_y:.1f}, {centroid_z:.1f})")
        else:
            print(f"❌ {mask_type.title()} mask not found")

if __name__ == "__main__":
    try:
        create_official_jaxfluids_visualization()
        create_inlet_outlet_analysis()
        
    except Exception as e:
        print(f"❌ Script failed: {e}")
        import traceback
        traceback.print_exc() 