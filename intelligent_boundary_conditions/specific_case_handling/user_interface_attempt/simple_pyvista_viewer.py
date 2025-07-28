#!/usr/bin/env python3
"""
Simple PyVista Mesh Viewer
Basic viewer to properly display the mesh and test face highlighting
"""

import os
import sys
import numpy as np

# Add the parent directories to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("❌ PyVista not available. Install with: pip install pyvista")
    exit()

try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False

def load_mesh_with_meshio(file_path):
    """Load mesh using meshio and convert to PyVista"""
    
    print(f"🔄 Loading mesh with meshio: {file_path}")
    
    try:
        mesh = meshio.read(file_path)
        print(f"📊 Mesh info:")
        print(f"   Points: {len(mesh.points)}")
        print(f"   Cell blocks: {len(mesh.cells)}")
        
        for i, cell_block in enumerate(mesh.cells):
            print(f"   Block {i}: {cell_block.type} ({len(cell_block.data)} elements)")
        
        # Try to convert to PyVista directly
        try:
            pv_mesh = pv.from_meshio(mesh)
            print(f"✅ Direct conversion successful: {pv_mesh.n_cells} cells, {pv_mesh.n_points} points")
            return pv_mesh
        except Exception as e:
            print(f"⚠️  Direct conversion failed: {e}")
            return extract_surface_from_volume(mesh)
            
    except Exception as e:
        print(f"❌ Meshio loading failed: {e}")
        return None

def extract_surface_from_volume(mesh):
    """Extract surface from volume mesh"""
    
    print("🔄 Extracting surface from volume mesh...")
    
    vertices = mesh.points
    surface_faces = []
    
    for cell_block in mesh.cells:
        if cell_block.type == 'tetra':
            print(f"   Processing {len(cell_block.data)} tetrahedra...")
            
            # Define tetrahedron faces
            tetra_faces = [
                [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]
            ]
            
            face_count = {}
            for tetra in cell_block.data:
                for face_indices in tetra_faces:
                    face = tuple(sorted([tetra[face_indices[0]], tetra[face_indices[1]], tetra[face_indices[2]]]))
                    face_count[face] = face_count.get(face, 0) + 1
            
            # Boundary faces appear only once
            boundary_faces = [list(face) for face, count in face_count.items() if count == 1]
            surface_faces.extend(boundary_faces)
            print(f"   Found {len(boundary_faces)} boundary faces")
            break
        elif cell_block.type == 'triangle':
            # Direct surface mesh
            surface_faces = cell_block.data.tolist()
            print(f"   Found {len(surface_faces)} surface triangles")
            break
    
    if not surface_faces:
        print("❌ No surface faces found")
        return None
    
    # Create PyVista mesh
    faces_array = []
    for face in surface_faces:
        faces_array.extend([3] + face)  # PyVista format: [n_points, p1, p2, p3]
    
    pv_mesh = pv.PolyData(vertices, faces_array)
    print(f"✅ Created surface mesh: {pv_mesh.n_faces} faces, {pv_mesh.n_points} points")
    
    return pv_mesh

def load_mesh_directly(file_path):
    """Try to load mesh directly with PyVista"""
    
    print(f"🔄 Trying direct PyVista load: {file_path}")
    
    try:
        pv_mesh = pv.read(file_path)
        print(f"✅ Direct PyVista load successful: {pv_mesh.n_cells} cells, {pv_mesh.n_points} points")
        return pv_mesh
    except Exception as e:
        print(f"❌ Direct PyVista load failed (this is expected for .msh files): {str(e)[:100]}...")
        return None

def create_simple_viewer(mesh_file):
    """Create a simple PyVista viewer for the mesh"""
    
    print("🚀 SIMPLE PYVISTA MESH VIEWER")
    print("=" * 50)
    
    if not PYVISTA_AVAILABLE:
        print("❌ PyVista is required for this viewer.")
        return
    
    # Try different loading methods
    pv_mesh = None
    
    # Method 1: Via meshio (preferred for .msh files)
    if MESHIO_AVAILABLE:
        pv_mesh = load_mesh_with_meshio(mesh_file)
    
    # Method 2: Direct PyVista (fallback)
    if pv_mesh is None:
        pv_mesh = load_mesh_directly(mesh_file)
    
    if pv_mesh is None:
        print("❌ Failed to load mesh with any method")
        return
    
    # Clean and prepare mesh
    print(f"🔧 Preparing mesh for visualization...")
    
    # Extract surface if it's a volume mesh
    if pv_mesh.n_faces == 0 and pv_mesh.n_cells > 0:
        print("   Extracting surface from volume mesh...")
        pv_mesh = pv_mesh.extract_surface()
    
    print(f"📊 Final mesh: {pv_mesh.n_faces} faces, {pv_mesh.n_points} points")
    
    # Create plotter
    plotter = pv.Plotter()
    
    # Add mesh with basic styling
    plotter.add_mesh(
        pv_mesh,
        color='lightblue',
        show_edges=True,
        edge_color='white',
        line_width=0.5,
        opacity=0.8,
        pickable=True,
        name='main_mesh'
    )
    
    # Enable face picking to test highlighting
    def pick_callback(mesh, picked_face_id):
        if picked_face_id is not None:
            print(f"👆 Picked face ID: {picked_face_id}")
            
            # Try to highlight the picked face
            try:
                # Extract the picked face
                picked_face = mesh.extract_cells(picked_face_id)
                
                # Add it as a separate mesh with different color
                plotter.add_mesh(
                    picked_face,
                    color='red',
                    opacity=1.0,
                    name=f'picked_face_{picked_face_id}',
                    reset_camera=False
                )
                
                print(f"✅ Highlighted face {picked_face_id}")
                
            except Exception as e:
                print(f"❌ Failed to highlight face: {e}")
    
    # Enable face picking
    plotter.enable_cell_picking(
        callback=pick_callback,
        show_message=True,
        style='wireframe',
        line_width=3,
        color='yellow'
    )
    
    # Add instructions
    instructions = [
        "🎯 SIMPLE MESH VIEWER",
        "",
        "MOUSE:",
        "• Click on faces to highlight them",
        "• Drag to rotate view",
        "• Scroll to zoom",
        "",
        "KEYBOARD:",
        "• 'r' - Reset view",
        "• 'q' - Quit",
        "",
        "Face picking is enabled!"
    ]
    
    plotter.add_text(
        "\n".join(instructions),
        position='upper_left',
        font_size=10,
        color='white'
    )
    
    # Set up camera for good view
    plotter.camera_position = 'iso'
    plotter.add_axes()
    
    print("🎮 Starting interactive viewer...")
    print("   Click on faces to test highlighting!")
    print("   Press 'q' to quit")
    
    # Show the viewer
    plotter.show()
    
    print("✅ Viewer closed")

def main():
    """Main function"""
    
    # Default mesh file
    mesh_file = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\mesh\Rocket Engine.msh"
    
    if not os.path.exists(mesh_file):
        print(f"❌ Mesh file not found: {mesh_file}")
        return
    
    create_simple_viewer(mesh_file)

if __name__ == "__main__":
    main() 