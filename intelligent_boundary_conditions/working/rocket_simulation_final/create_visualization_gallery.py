#!/usr/bin/env python3
"""
Create Visualization Gallery from JAX-Fluids Results
Generates multiple views and saves as images for documentation
"""

import pyvista as pv
import numpy as np
from pathlib import Path

def create_visualization_gallery():
    """Create a gallery of visualizations"""
    print("📸 CREATING VISUALIZATION GALLERY")
    print("=" * 50)
    
    # Set up PyVista for off-screen rendering
    pv.set_plot_theme("document")
    pv.OFF_SCREEN = True
    
    # Load the latest data
    data_dir = Path("output/rocket_nozzle_internal_supersonic_production/domain")
    h5_files = sorted(list(data_dir.glob("*.h5")))
    
    if not h5_files:
        print("❌ No data files found!")
        return
    
    # Create output directory for images
    gallery_dir = Path("visualization_gallery")
    gallery_dir.mkdir(exist_ok=True)
    
    # Load the final timestep data
    latest_file = h5_files[-1]
    xdmf_file = latest_file.with_suffix('.xdmf')
    
    print(f"📁 Loading: {xdmf_file}")
    mesh = pv.read(str(xdmf_file))
    
    print(f"✅ Loaded: {mesh.n_points} points, {mesh.n_cells} cells")
    print(f"📊 Fields: {mesh.array_names}")
    
    # Define visualization configurations
    views = [
        {
            "name": "density_iso",
            "title": "Density Field - Isometric View",
            "field": "density",
            "cmap": "viridis",
            "camera": "iso"
        },
        {
            "name": "pressure_iso", 
            "title": "Pressure Field - Isometric View",
            "field": "pressure",
            "cmap": "plasma",
            "camera": "iso"
        },
        {
            "name": "mach_number_iso",
            "title": "Mach Number - Isometric View", 
            "field": "mach_number",
            "cmap": "jet",
            "camera": "iso"
        },
        {
            "name": "temperature_iso",
            "title": "Temperature Field - Isometric View",
            "field": "temperature", 
            "cmap": "hot",
            "camera": "iso"
        },
        {
            "name": "velocity_magnitude_iso",
            "title": "Velocity Magnitude - Isometric View",
            "field": "velocity",
            "cmap": "coolwarm",
            "camera": "iso",
            "compute_magnitude": True
        }
    ]
    
    # Generate visualizations
    for view_config in views:
        try:
            print(f"🎨 Creating {view_config['name']}...")
            
            plotter = pv.Plotter(off_screen=True, window_size=[1200, 800])
            
            field = view_config["field"]
            
            # Handle velocity magnitude calculation
            if view_config.get("compute_magnitude", False) and field == "velocity":
                velocity = mesh[field].reshape(-1, 3)
                magnitude = np.linalg.norm(velocity, axis=1)
                mesh["velocity_magnitude"] = magnitude
                field = "velocity_magnitude"
            
            # Check if field exists
            if field not in mesh.array_names:
                print(f"⚠️  Field '{field}' not found, skipping...")
                continue
            
            # Add main mesh with field coloring
            plotter.add_mesh(mesh,
                           scalars=field,
                           cmap=view_config["cmap"],
                           opacity=0.9,
                           show_scalar_bar=True,
                           scalar_bar_args={
                               'title': view_config["title"],
                               'title_font_size': 14,
                               'label_font_size': 12
                           })
            
            # Add cross-sectional slices for internal view
            slice_x = mesh.slice(normal='x', origin=[800, 0, 0])
            plotter.add_mesh(slice_x,
                           scalars=field,
                           cmap=view_config["cmap"],
                           opacity=0.8)
            
            # Add wireframe outline
            plotter.add_mesh(mesh.outline(), color='black', line_width=2)
            
            # Set camera position
            if view_config["camera"] == "iso":
                plotter.camera_position = [(2000, 1500, 1000), (800, 0, 0), (0, 0, 1)]
            
            # Add title and axes
            plotter.add_text(view_config["title"], font_size=16, color='black')
            plotter.add_axes(color='black')
            
            # Save image
            image_file = gallery_dir / f"{view_config['name']}.png"
            plotter.screenshot(str(image_file), transparent_background=False)
            plotter.close()
            
            print(f"   ✅ Saved: {image_file}")
            
        except Exception as e:
            print(f"   ❌ Failed to create {view_config['name']}: {e}")
    
    # Create a summary image with multiple views
    print("🖼️  Creating summary composite...")
    try:
        plotter = pv.Plotter(shape=(2, 2), off_screen=True, window_size=[1600, 1200])
        
        # Density
        plotter.subplot(0, 0)
        plotter.add_mesh(mesh, scalars="density", cmap="viridis", opacity=0.8)
        plotter.add_text("Density", font_size=14)
        plotter.camera_position = "iso"
        
        # Pressure  
        plotter.subplot(0, 1)
        plotter.add_mesh(mesh, scalars="pressure", cmap="plasma", opacity=0.8)
        plotter.add_text("Pressure", font_size=14)
        plotter.camera_position = "iso"
        
        # Mach number
        plotter.subplot(1, 0)
        plotter.add_mesh(mesh, scalars="mach_number", cmap="jet", opacity=0.8)
        plotter.add_text("Mach Number", font_size=14)
        plotter.camera_position = "iso"
        
        # Temperature
        plotter.subplot(1, 1)
        plotter.add_mesh(mesh, scalars="temperature", cmap="hot", opacity=0.8)
        plotter.add_text("Temperature", font_size=14) 
        plotter.camera_position = "iso"
        
        summary_file = gallery_dir / "summary_all_fields.png"
        plotter.screenshot(str(summary_file))
        plotter.close()
        
        print(f"   ✅ Summary saved: {summary_file}")
        
    except Exception as e:
        print(f"   ❌ Failed to create summary: {e}")
    
    print("\n📂 GALLERY CREATED!")
    print(f"   Location: {gallery_dir.absolute()}")
    print(f"   Files: {len(list(gallery_dir.glob('*.png')))} images")
    
    return gallery_dir

if __name__ == "__main__":
    try:
        gallery_dir = create_visualization_gallery()
        
        # List created files
        print("\n📋 CREATED FILES:")
        for img in sorted(gallery_dir.glob("*.png")):
            size_mb = img.stat().st_size / 1e6
            print(f"   📸 {img.name} ({size_mb:.1f} MB)")
            
    except Exception as e:
        print(f"❌ Gallery creation failed: {e}")
        import traceback
        traceback.print_exc() 