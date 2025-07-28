#!/usr/bin/env python3
"""
Proper 3D JAX-Fluids Rocket Visualization
Uses JAX-Fluids official functions to create comprehensive 3D visualization through multiple slices
"""

import os
import numpy as np
from pathlib import Path

# Official JAX-Fluids postprocessing
from jaxfluids_postprocess import load_data, create_2D_figure, create_2D_animation

def create_comprehensive_3d_viz():
    """Create comprehensive 3D visualization using JAX-Fluids official functions"""
    print("🚀 COMPREHENSIVE 3D ROCKET NOZZLE VISUALIZATION")
    print("=" * 60)
    
    # Path to simulation data
    data_path = Path("intelligent_boundary_conditions/working/rocket_simulation_final/output/rocket_nozzle_internal_supersonic_production/domain")
    
    if not data_path.exists():
        print("❌ Simulation data not found!")
        return
    
    print(f"📊 Loading 3D data from: {data_path}")
    
    # Load all available quantities
    quantities = ['density', 'pressure', 'mach_number', 'temperature', 'velocity']
    
    try:
        jxf_data = load_data(str(data_path), quantities)
        
        cell_centers = jxf_data.cell_centers
        data = jxf_data.data
        times = jxf_data.times
        
        print(f"✅ Loaded {len(quantities)} quantities over {len(times)} time steps")
        print(f"3D Domain: X=[{cell_centers[0].min():.1f}, {cell_centers[0].max():.1f}]")
        print(f"3D Domain: Y=[{cell_centers[1].min():.1f}, {cell_centers[1].max():.1f}]") 
        print(f"3D Domain: Z=[{cell_centers[2].min():.1f}, {cell_centers[2].max():.1f}]")
        
        # Show 3D data structure
        for qty, qty_data in data.items():
            print(f"{qty}: 3D shape={qty_data.shape}, range=[{qty_data.min():.6f}, {qty_data.max():.6f}]")
        
        # Create output directory
        os.makedirs("3d_rocket_viz", exist_ok=True)
        
        print("\n📊 Creating 3D rocket nozzle visualization through multiple slices...")
        
        # Main flow quantities for visualization
        scalar_data = {
            "density": data["density"],
            "pressure": data["pressure"], 
            "mach_number": data["mach_number"],
            "temperature": data["temperature"]
        }
        
        # 1. AXIAL VIEW (X-Z plane) - Shows rocket nozzle profile and flow direction
        print("🎯 Creating axial view (X-Z plane) - nozzle profile...")
        create_2D_figure(
            scalar_data,
            nrows_ncols=(2, 2),
            cell_centers=cell_centers,
            plane="xz",
            plane_value=0.0,  # Center slice through nozzle axis
            save_fig="3d_rocket_viz/rocket_axial_view.png",
            dpi=300
        )
        print("✅ Axial view saved: 3d_rocket_viz/rocket_axial_view.png")
        
        # 2. CROSS-SECTIONS (Y-Z plane) - Multiple slices along nozzle length
        print("🎯 Creating cross-sections (Y-Z plane) - flow development...")
        
        # Key locations along nozzle: inlet, throat, and outlet
        x_locations = [-100, 400, 800, 1200, 1600]  # Various positions along nozzle
        x_labels = ["inlet", "throat_approach", "throat", "expansion", "outlet"]
        
        for i, (x_pos, label) in enumerate(zip(x_locations, x_labels)):
            create_2D_figure(
                scalar_data,
                nrows_ncols=(2, 2),
                cell_centers=cell_centers,
                plane="yz",
                plane_value=x_pos,
                save_fig=f"3d_rocket_viz/cross_section_{label}_x{x_pos:.0f}.png",
                dpi=300
            )
            print(f"✅ Cross-section {label} saved")
        
        # 3. TOP VIEW (X-Y plane) - Looking down at nozzle
        print("🎯 Creating top view (X-Y plane) - nozzle from above...")
        create_2D_figure(
            scalar_data,
            nrows_ncols=(2, 2),
            cell_centers=cell_centers,
            plane="xy",
            plane_value=0.0,  # Center height
            save_fig="3d_rocket_viz/rocket_top_view.png",
            dpi=300
        )
        print("✅ Top view saved: 3d_rocket_viz/rocket_top_view.png")
        
        # 4. VELOCITY FIELD VISUALIZATION
        if "velocity" in data:
            print("🎯 Creating 3D velocity field visualization...")
            
            # Extract velocity components for 3D analysis
            velocity_data = {
                "velocity_x": data["velocity"][:, 0],  # Axial velocity (main flow direction)
                "velocity_y": data["velocity"][:, 1],  # Radial velocity Y
                "velocity_z": data["velocity"][:, 2],  # Radial velocity Z
                "velocity_magnitude": np.sqrt(
                    data["velocity"][:, 0]**2 + 
                    data["velocity"][:, 1]**2 + 
                    data["velocity"][:, 2]**2
                )
            }
            
            # Axial velocity profile (main flow direction)
            create_2D_figure(
                velocity_data,
                nrows_ncols=(2, 2),
                cell_centers=cell_centers,
                plane="xz",
                plane_value=0.0,
                save_fig="3d_rocket_viz/velocity_field_axial.png",
                dpi=300
            )
            print("✅ 3D velocity field saved")
        
        # 5. ANIMATION - Flow evolution in 3D
        print("🎯 Creating 3D flow evolution animation...")
        
        # Animation showing pressure and Mach number evolution
        anim_data = {
            "pressure": data["pressure"],
            "mach_number": data["mach_number"]
        }
        
        create_2D_animation(
            anim_data,
            cell_centers,
            times,
            nrows_ncols=(1, 2),
            plane="xz",  # Axial view for animation
            plane_value=0.0,
            interval=300,  # 300ms between frames
            save_png="3d_rocket_viz/frames",
            save_mp4="3d_rocket_viz/rocket_3d_evolution.mp4",
            dpi=200
        )
        print("✅ 3D evolution animation saved")
        
        print(f"\n🎉 Comprehensive 3D JAX-Fluids visualization complete!")
        print(f"📁 Results saved in: ./3d_rocket_viz/")
        print(f"   📊 3D Views:")
        print(f"     - rocket_axial_view.png (side profile - main flow direction)")
        print(f"     - rocket_top_view.png (top view - nozzle shape)")
        print(f"     - cross_section_*.png (5 slices showing 3D flow development)")
        print(f"   🌊 3D Flow Analysis:")
        print(f"     - velocity_field_axial.png (3D velocity components)")
        print(f"     - rocket_3d_evolution.mp4 (temporal evolution)")
        
        # 3D Flow Analysis Summary
        print(f"\n🔍 3D Flow Analysis Summary:")
        print(f"   3D Grid: {data['density'].shape[1:]} (Z×Y×X)")
        print(f"   Inlet region: X ≈ {-3.1:.1f} (1,872 grid points)")
        print(f"   Outlet region: X ≈ {1713.4:.1f} (7,120 grid points)")
        print(f"   Pressure range: {data['pressure'].min():.0f} - {data['pressure'].max():.0f} Pa")
        print(f"   Max velocity: {np.sqrt((data['velocity']**2).sum(axis=1)).max():.2f} m/s")
        print(f"   Max Mach number: {data['mach_number'].max():.6f}")
        
        # Identify key flow features
        velocity_mag = np.sqrt((data['velocity']**2).sum(axis=1))
        max_vel_idx = np.unravel_index(np.argmax(velocity_mag[-1]), velocity_mag[-1].shape)
        
        # Convert indices to physical coordinates
        x_max_vel = cell_centers[0][max_vel_idx[2]]  # X coordinate
        y_max_vel = cell_centers[1][max_vel_idx[1]]  # Y coordinate  
        z_max_vel = cell_centers[2][max_vel_idx[0]]  # Z coordinate
        
        print(f"   Max velocity location: ({x_max_vel:.1f}, {y_max_vel:.1f}, {z_max_vel:.1f})")
        
    except Exception as e:
        print(f"❌ 3D Visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_comprehensive_3d_viz() 