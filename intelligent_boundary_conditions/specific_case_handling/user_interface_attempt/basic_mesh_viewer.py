#!/usr/bin/env python3
"""
Basic PyVista Mesh Viewer
Very simple test to load and display the rocket engine mesh
"""

import os
import sys

try:
    import pyvista as pv
    print(f"✅ PyVista {pv.__version__} available")
except ImportError:
    print("❌ PyVista not available")
    exit()

try:
    import meshio
    print(f"✅ Meshio available")
except ImportError:
    print("❌ Meshio not available")
    exit()

def main():
    """Load and display the mesh"""
    
    # Mesh file path
    mesh_file = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\mesh\Rocket Engine.msh"
    
    print(f"🔄 Loading: {mesh_file}")
    
    # Load with meshio first
    try:
        mesh = meshio.read(mesh_file)
        print(f"📊 Meshio loaded:")
        print(f"   Points: {len(mesh.points)}")
        print(f"   Cell types: {[block.type for block in mesh.cells]}")
        
        # Convert to PyVista using from_meshio
        pv_mesh = pv.from_meshio(mesh)
        print(f"✅ PyVista mesh: {pv_mesh.n_cells} cells, {pv_mesh.n_points} points")
        
        # Extract surface if needed
        if pv_mesh.n_faces == 0:
            print("🔄 Extracting surface...")
            surface = pv_mesh.extract_surface()
            print(f"📊 Surface mesh: {surface.n_faces} faces")
            pv_mesh = surface
        
        # Simple display
        print("🎮 Displaying mesh...")
        
        plotter = pv.Plotter()
        plotter.add_mesh(pv_mesh, show_edges=True, color='lightblue')
        plotter.add_axes()
        plotter.camera_position = 'iso'
        
        # Add simple text
        plotter.add_text("Rocket Engine Mesh", font_size=12)
        
        print("✅ Click to interact, press 'q' to quit")
        plotter.show()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 