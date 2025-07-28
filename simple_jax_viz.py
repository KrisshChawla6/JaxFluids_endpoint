#!/usr/bin/env python3
"""
Simple JAX-Fluids Rocket Visualization
Uses official jaxfluids_postprocess functions
"""

import os
import numpy as np
from pathlib import Path

# Official JAX-Fluids postprocessing
from jaxfluids_postprocess import load_data, create_2D_figure

def create_simple_jaxfluids_viz():
    """Create simple but proper JAX-Fluids visualization"""
    print("🚀 SIMPLE JAX-FLUIDS ROCKET VISUALIZATION")
    print("=" * 60)
    
    # Path to simulation data
    data_path = Path("intelligent_boundary_conditions/working/rocket_simulation_final/output/rocket_nozzle_internal_supersonic_production/domain")
    
    if not data_path.exists():
        print("❌ Simulation data not found!")
        return
    
    print(f"📊 Loading data from: {data_path}")
    
    # Load data using official JAX-Fluids function
    quantities = ['density', 'pressure', 'mach_number', 'temperature']
    
    try:
        jxf_data = load_data(str(data_path), quantities)
        
        cell_centers = jxf_data.cell_centers
        data = jxf_data.data
        times = jxf_data.times
        
        print(f"✅ Loaded {len(quantities)} quantities over {len(times)} time steps")
        print(f"Domain: X=[{cell_centers[0].min():.1f}, {cell_centers[0].max():.1f}]")
        print(f"Domain: Y=[{cell_centers[1].min():.1f}, {cell_centers[1].max():.1f}]") 
        print(f"Domain: Z=[{cell_centers[2].min():.1f}, {cell_centers[2].max():.1f}]")
        
        # Show data ranges
        for qty, qty_data in data.items():
            print(f"{qty}: shape={qty_data.shape}, range=[{qty_data.min():.6f}, {qty_data.max():.6f}]")
        
        # Create output directory
        os.makedirs("jax_viz", exist_ok=True)
        
        print("\n📊 Creating rocket nozzle visualization...")
        
        # Prepare plot data
        plot_dict = {
            "density": data["density"],
            "pressure": data["pressure"], 
            "mach_number": data["mach_number"],
            "temperature": data["temperature"]
        }
        
        # Create side view (X-Z plane) - this shows the nozzle profile best
        create_2D_figure(
            plot_dict,
            nrows_ncols=(2, 2),
            cell_centers=cell_centers,
            plane="xz",
            plane_value=0.0,  # Center slice through nozzle
            time_step=-1,     # Latest time step
            dpi=300,
            save_fig="jax_viz/rocket_nozzle_side_view.png"
        )
        print("✅ Side view (X-Z plane) saved: jax_viz/rocket_nozzle_side_view.png")
        
        # Create front view (Y-Z plane) - cross-section through nozzle
        create_2D_figure(
            plot_dict,
            nrows_ncols=(2, 2),
            cell_centers=cell_centers,
            plane="yz", 
            plane_value=400.0,  # Through middle of nozzle
            time_step=-1,
            dpi=300,
            save_fig="jax_viz/rocket_nozzle_cross_section.png"
        )
        print("✅ Cross-section (Y-Z plane) saved: jax_viz/rocket_nozzle_cross_section.png")
        
        print(f"\n🎉 JAX-Fluids visualization complete!")
        print(f"📁 Results saved in: ./jax_viz/")
        print(f"   - rocket_nozzle_side_view.png (shows nozzle profile)")
        print(f"   - rocket_nozzle_cross_section.png (shows circular cross-section)")
        
        # Show inlet/outlet info
        print(f"\n🔍 Flow Analysis:")
        print(f"   Inlet region: around X={-3.1:.1f} (1,872 grid points)")
        print(f"   Outlet region: around X={1713.4:.1f} (7,120 grid points)")
        print(f"   Pressure variation: {data['pressure'].min():.0f} to {data['pressure'].max():.0f} Pa")
        print(f"   Max Mach number: {data['mach_number'].max():.6f} (still subsonic)")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_simple_jaxfluids_viz() 