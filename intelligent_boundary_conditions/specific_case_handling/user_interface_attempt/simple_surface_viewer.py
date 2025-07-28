#!/usr/bin/env python3
"""
Simple Surface Viewer
Uses existing .msh file, extracts surface, and color-codes based on Gmsh analysis
"""

import os
import sys
import numpy as np

try:
    import gmsh
    print(f"✅ Gmsh {gmsh.__version__} available")
except ImportError:
    print("❌ Gmsh not available")
    exit()

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

def get_face_classification_from_gmsh(mesh_file):
    """Quick Gmsh analysis to get inlet/outlet face positions"""
    
    print("🔍 Quick Gmsh analysis for face classification...")
    
    gmsh.initialize()
    try:
        gmsh.open(mesh_file)
        
        # Get boundary face positions
        entities = gmsh.model.getEntities()
        volume_entities = [e for e in entities if e[0] == 3]
        
        all_boundary_faces = []
        for vol_entity in volume_entities:
            boundary = gmsh.model.getBoundary([vol_entity], oriented=False, recursive=False)
            boundary_faces = [e for e in boundary if e[0] == 2]
            all_boundary_faces.extend(boundary_faces)
        
        unique_faces = list(set(all_boundary_faces))
        
        # Get face centroids
        face_positions = {}
        for face_dim, face_tag in unique_faces:
            face_nodes = gmsh.model.mesh.getNodes(face_dim, face_tag)
            if len(face_nodes[1]) > 0:
                coords = face_nodes[1].reshape(-1, 3)
                centroid = coords.mean(axis=0)
                face_positions[face_tag] = centroid
        
        # Find inlet/outlet by X-position
        x_coords = [pos[0] for pos in face_positions.values()]
        min_x, max_x = min(x_coords), max(x_coords)
        
        print(f"   X range: {min_x:.1f} to {max_x:.1f}")
        print(f"   Inlet at X≈{min_x:.1f}, Outlet at X≈{max_x:.1f}")
        
        return min_x, max_x
        
    finally:
        gmsh.finalize()

def create_surface_mesh_with_colors(mesh_file, inlet_x, outlet_x):
    """Load mesh, extract surface, and add color coding"""
    
    print("🔄 Loading mesh and extracting surface...")
    
    # Load with meshio and convert to PyVista
    mesh = meshio.read(mesh_file)
    pv_mesh = pv.from_meshio(mesh)
    
    print(f"   Volume mesh: {pv_mesh.n_cells:,} cells")
    
    # Extract surface
    surface = pv_mesh.extract_surface()
    print(f"   Surface mesh: {surface.n_cells:,} faces")
    
    # Calculate face centroids
    face_centers = surface.cell_centers()
    x_positions = face_centers.points[:, 0]  # X coordinates of face centers
    
    print(f"   Surface X range: {x_positions.min():.1f} to {x_positions.max():.1f}")
    
    # Classify faces based on X position
    tolerance = (outlet_x - inlet_x) * 0.05  # 5% tolerance
    
    # Create color array (0=inlet, 1=wall, 2=outlet)
    colors = np.ones(len(x_positions))  # Default to wall (1)
    
    # Mark inlet faces (near min X)
    inlet_mask = np.abs(x_positions - inlet_x) < tolerance
    colors[inlet_mask] = 0  # Inlet
    
    # Mark outlet faces (near max X)  
    outlet_mask = np.abs(x_positions - outlet_x) < tolerance
    colors[outlet_mask] = 2  # Outlet
    
    # Count faces
    n_inlet = np.sum(inlet_mask)
    n_outlet = np.sum(outlet_mask)
    n_wall = len(colors) - n_inlet - n_outlet
    
    print(f"   Classified: {n_inlet} inlet, {n_outlet} outlet, {n_wall} wall faces")
    
    # Add color data to mesh
    surface['face_type'] = colors
    
    return surface

def visualize_surface(surface):
    """Create PyVista visualization with color-coded faces"""
    
    print("🎮 Creating visualization...")
    
    # Create plotter
    plotter = pv.Plotter(window_size=[1400, 900])
    
    # Add surface with color mapping
    plotter.add_mesh(
        surface,
        scalars='face_type',
        cmap=['red', 'lightgray', 'green'],  # 0=red, 1=gray, 2=green
        show_edges=True,
        edge_color='white',
        line_width=0.1,
        opacity=0.9
    )
    
    # Add color bar
    plotter.add_scalar_bar(
        title="Face Type",
        n_labels=3,
        color='white',
        label_font_size=12
    )
    
    # Add instructions
    instructions = [
        "🚀 ROCKET ENGINE NOZZLE",
        "",
        "COLOR CODING:",
        "🔴 RED = Inlet (combustion chamber)",
        "🟢 GREEN = Outlet (nozzle exit)",
        "⚪ GRAY = Wall surfaces",
        "",
        "CONTROLS:",
        "• Drag to rotate view",
        "• Scroll to zoom",
        "• Right-click drag to pan",
        "• 'r' = Reset camera",
        "• 'q' = Quit viewer",
        "",
        f"Total faces: {surface.n_cells:,}"
    ]
    
    plotter.add_text(
        "\n".join(instructions),
        position='upper_left',
        font_size=11,
        color='white'
    )
    
    # Setup camera and lighting
    plotter.camera_position = 'iso'
    plotter.add_axes()
    plotter.set_background('navy')
    
    print("✅ Visualization ready!")
    print("   🔴 Red = Inlet faces")
    print("   🟢 Green = Outlet faces")
    print("   ⚪ Gray = Wall faces")
    
    # Show the visualization
    plotter.show()

def main():
    """Main function"""
    
    print("🚀 SIMPLE ROCKET NOZZLE SURFACE VIEWER")
    print("=" * 60)
    
    # Mesh file
    mesh_file = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\mesh\Rocket Engine.msh"
    
    if not os.path.exists(mesh_file):
        print(f"❌ Mesh file not found: {mesh_file}")
        return
    
    try:
        # Step 1: Quick Gmsh analysis to get inlet/outlet positions
        inlet_x, outlet_x = get_face_classification_from_gmsh(mesh_file)
        
        # Step 2: Load mesh, extract surface, add colors
        surface = create_surface_mesh_with_colors(mesh_file, inlet_x, outlet_x)
        
        # Step 3: Visualize with PyVista
        visualize_surface(surface)
        
        print("✅ Visualization complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 