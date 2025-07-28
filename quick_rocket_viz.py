#!/usr/bin/env python3
"""
Quick JAX-Fluids visualization showing rocket geometry + flow
"""

import os
import numpy as np
from pathlib import Path
from jaxfluids_postprocess import load_data, create_2D_figure

def quick_viz_with_geometry():
    print("🚀 QUICK ROCKET VISUALIZATION - GEOMETRY + FLOW")
    
    # Load simulation data
    data_path = Path("intelligent_boundary_conditions/working/rocket_simulation_final/output/rocket_nozzle_internal_supersonic_production/domain")
    
    # Load flow data AND levelset
    quantities = ['density', 'pressure', 'mach_number', 'levelset']
    jxf_data = load_data(str(data_path), quantities)
    
    cell_centers = jxf_data.cell_centers
    data = jxf_data.data
    
    print(f"Loaded: {list(data.keys())}")
    
    os.makedirs("quick_viz", exist_ok=True)
    
    # Plot with levelset to show geometry
    plot_data = {
        "pressure": data["pressure"],
        "mach_number": data["mach_number"],
        "levelset": data["levelset"],  # This shows the rocket geometry!
        "density": data["density"]
    }
    
    # Side view showing rocket nozzle shape
    create_2D_figure(
        plot_data,
        nrows_ncols=(2, 2),
        cell_centers=cell_centers,
        plane="xz",
        plane_value=0.0,
        save_fig="quick_viz/rocket_with_geometry.png",
        dpi=300
    )
    
    print("✅ Rocket with geometry saved: quick_viz/rocket_with_geometry.png")
    
    # Cross-section at nozzle throat
    create_2D_figure(
        plot_data,
        nrows_ncols=(2, 2),
        cell_centers=cell_centers,
        plane="yz",
        plane_value=800.0,  # Throat location
        save_fig="quick_viz/throat_cross_section.png",
        dpi=300
    )
    
    print("✅ Throat cross-section saved: quick_viz/throat_cross_section.png")

if __name__ == "__main__":
    quick_viz_with_geometry() 