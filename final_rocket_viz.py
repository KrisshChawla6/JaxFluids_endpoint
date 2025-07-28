#!/usr/bin/env python3
"""
Final Rocket Visualization - JAX-Fluids flow + SDF geometry
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from jaxfluids_postprocess import load_data

def final_rocket_viz():
    print("🚀 FINAL ROCKET VISUALIZATION")
    
    # 1. Load JAX-Fluids simulation data
    data_path = Path("intelligent_boundary_conditions/working/rocket_simulation_final/output/rocket_nozzle_internal_supersonic_production/domain")
    jxf_data = load_data(str(data_path), ['pressure', 'mach_number', 'density'])
    
    cell_centers = jxf_data.cell_centers
    data = jxf_data.data
    
    # 2. Load rocket SDF geometry
    sdf_path = Path("intelligent_boundary_conditions/working/hardcoded_run/attempt1/rocket_case/20250728_003202/Rocket Engine_sdf_matrix.npy")
    if sdf_path.exists():
        sdf = np.load(sdf_path)
        print(f"✅ Loaded SDF: {sdf.shape}")
    else:
        print("❌ SDF not found")
        sdf = None
    
    # 3. Create combined visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Get latest time step data
    pressure = data['pressure'][-1]  # Shape: (128, 64, 64)
    mach = data['mach_number'][-1]
    density = data['density'][-1]
    
    # Center slices
    x_center = pressure.shape[0] // 2  # X slice
    y_center = pressure.shape[1] // 2  # Y slice
    z_center = pressure.shape[2] // 2  # Z slice
    
    # Plot 1: Pressure (X-Z plane, Y=center)
    ax = axes[0,0]
    im1 = ax.imshow(pressure[:, y_center, :].T, origin='lower', cmap='viridis', aspect='auto')
    if sdf is not None and sdf.shape == pressure.shape:
        contour = ax.contour(sdf[:, y_center, :].T, levels=[0], colors='white', linewidths=2)
    ax.set_title('Pressure + Rocket Geometry (Side View)')
    ax.set_xlabel('X (flow direction)')
    ax.set_ylabel('Z')
    plt.colorbar(im1, ax=ax)
    
    # Plot 2: Mach number (X-Z plane, Y=center)  
    ax = axes[0,1]
    im2 = ax.imshow(mach[:, y_center, :].T, origin='lower', cmap='plasma', aspect='auto')
    if sdf is not None and sdf.shape == mach.shape:
        ax.contour(sdf[:, y_center, :].T, levels=[0], colors='white', linewidths=2)
    ax.set_title('Mach Number + Rocket Geometry')
    ax.set_xlabel('X (flow direction)')
    ax.set_ylabel('Z')
    plt.colorbar(im2, ax=ax)
    
    # Plot 3: Cross-section at throat (Y-Z plane)
    ax = axes[1,0]
    throat_x = int(0.7 * pressure.shape[0])  # Approx throat location
    im3 = ax.imshow(pressure[throat_x, :, :], origin='lower', cmap='viridis')
    if sdf is not None and sdf.shape == pressure.shape:
        ax.contour(sdf[throat_x, :, :], levels=[0], colors='white', linewidths=2)
    ax.set_title(f'Throat Cross-section (X={throat_x})')
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    plt.colorbar(im3, ax=ax)
    
    # Plot 4: Density
    ax = axes[1,1]
    im4 = ax.imshow(density[:, y_center, :].T, origin='lower', cmap='coolwarm', aspect='auto')
    if sdf is not None and sdf.shape == density.shape:
        ax.contour(sdf[:, y_center, :].T, levels=[0], colors='black', linewidths=2)
    ax.set_title('Density + Rocket Geometry')
    ax.set_xlabel('X (flow direction)')
    ax.set_ylabel('Z')
    plt.colorbar(im4, ax=ax)
    
    plt.tight_layout()
    plt.savefig('final_rocket_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Final visualization saved: final_rocket_visualization.png")
    
    # Summary
    print(f"\n📊 Flow Summary:")
    print(f"   Pressure range: {data['pressure'].min():.0f} - {data['pressure'].max():.0f} Pa")
    print(f"   Max Mach: {data['mach_number'].max():.6f}")
    print(f"   Density range: {data['density'].min():.3f} - {data['density'].max():.3f} kg/m³")

if __name__ == "__main__":
    final_rocket_viz() 