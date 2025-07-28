#!/usr/bin/env python3
"""
Gmsh + PyVista Hybrid Viewer
Uses Gmsh to identify inlet/outlet faces, then visualizes in PyVista
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

def extract_faces_with_gmsh(mesh_file):
    """Use Gmsh to extract and classify faces"""
    
    print("🔍 Using Gmsh to analyze face classification...")
    
    # Initialize Gmsh
    gmsh.initialize()
    
    try:
        # Load mesh
        gmsh.open(mesh_file)
        
        # Get boundary faces
        entities = gmsh.model.getEntities()
        volume_entities = [e for e in entities if e[0] == 3]
        
        # Get all boundary faces
        all_boundary_faces = []
        for vol_entity in volume_entities:
            boundary = gmsh.model.getBoundary([vol_entity], oriented=False, recursive=False)
            boundary_faces = [e for e in boundary if e[0] == 2]
            all_boundary_faces.extend(boundary_faces)
        
        unique_faces = list(set(all_boundary_faces))
        print(f"   Found {len(unique_faces)} boundary faces")
        
        # Analyze face positions
        face_data = {}
        face_positions = {}
        
        for face_dim, face_tag in unique_faces:
            # Get face mesh data
            face_nodes = gmsh.model.mesh.getNodes(face_dim, face_tag)
            face_elements = gmsh.model.mesh.getElements(face_dim, face_tag)
            
            if len(face_nodes[1]) > 0:
                # Get coordinates
                coords = face_nodes[1].reshape(-1, 3)
                centroid = coords.mean(axis=0)
                face_positions[face_tag] = centroid
                
                # Store face data for PyVista
                face_data[face_tag] = {
                    'nodes': face_nodes,
                    'elements': face_elements,
                    'coordinates': coords,
                    'centroid': centroid
                }
        
        # Classify faces as inlet/outlet/wall
        x_coords = [pos[0] for pos in face_positions.values()]
        min_x, max_x = min(x_coords), max(x_coords)
        tolerance = (max_x - min_x) * 0.01
        
        inlet_faces = [tag for tag, pos in face_positions.items() 
                      if abs(pos[0] - min_x) < tolerance]
        outlet_faces = [tag for tag, pos in face_positions.items() 
                       if abs(pos[0] - max_x) < tolerance]
        wall_faces = [tag for tag in face_positions.keys() 
                     if tag not in inlet_faces and tag not in outlet_faces]
        
        print(f"   Classified: {len(inlet_faces)} inlet, {len(outlet_faces)} outlet, {len(wall_faces)} wall faces")
        
        # Return classification and face data
        classification = {
            'inlet': inlet_faces,
            'outlet': outlet_faces,
            'wall': wall_faces
        }
        
        return face_data, classification
        
    finally:
        gmsh.finalize()

def create_pyvista_mesh_from_faces(face_data, classification):
    """Create PyVista meshes from Gmsh face data"""
    
    print("🔄 Converting to PyVista meshes...")
    
    meshes = {}
    
    for face_type, face_tags in classification.items():
        if not face_tags:
            continue
            
        print(f"   Processing {len(face_tags)} {face_type} faces...")
        
        # Collect all points and faces for this type
        all_points = []
        all_faces = []
        point_offset = 0
        
        for face_tag in face_tags:
            if face_tag not in face_data:
                continue
                
            face_info = face_data[face_tag]
            coords = face_info['coordinates']
            elements = face_info['elements']
            
            # Add points
            all_points.append(coords)
            
            # Process elements (triangles)
            for elem_type, elem_tags, elem_nodes in zip(elements[0], elements[1], elements[2]):
                if elem_type == 2:  # Triangle
                    # Convert to 0-indexed
                    triangles = elem_nodes.reshape(-1, 3) - 1
                    # Adjust for point offset
                    triangles += point_offset
                    
                    # Add to faces (PyVista format: [3, p1, p2, p3])
                    for triangle in triangles:
                        all_faces.extend([3] + triangle.tolist())
            
            point_offset += len(coords)
        
        if all_points and all_faces:
            # Combine all points
            combined_points = np.vstack(all_points)
            
            # Create PyVista mesh
            mesh = pv.PolyData(combined_points, all_faces)
            meshes[face_type] = mesh
            
            print(f"   ✅ {face_type}: {mesh.n_points} points, {mesh.n_faces} faces")
    
    return meshes

def visualize_with_pyvista(meshes):
    """Create interactive PyVista visualization"""
    
    print("🎮 Creating PyVista visualization...")
    
    # Create plotter
    plotter = pv.Plotter(window_size=[1400, 900])
    
    # Colors for different face types
    colors = {
        'inlet': 'red',
        'outlet': 'green',
        'wall': 'lightgray'
    }
    
    # Add each mesh type
    for face_type, mesh in meshes.items():
        color = colors.get(face_type, 'blue')
        
        # Different styling for different types
        if face_type == 'wall':
            plotter.add_mesh(
                mesh,
                color=color,
                opacity=0.3,
                show_edges=True,
                edge_color='white',
                line_width=0.1,
                name=f'mesh_{face_type}'
            )
        else:  # inlet or outlet
            plotter.add_mesh(
                mesh,
                color=color,
                opacity=0.9,
                show_edges=True,
                edge_color='black',
                line_width=1.0,
                name=f'mesh_{face_type}'
            )
    
    # Add instructions
    instructions = [
        "🚀 ROCKET ENGINE NOZZLE",
        "",
        "FACE CLASSIFICATION:",
        "🔴 RED = Inlet (combustion chamber)",
        "🟢 GREEN = Outlet (nozzle exit)",
        "⚪ GRAY = Wall surfaces",
        "",
        "CONTROLS:",
        "• Drag to rotate",
        "• Scroll to zoom",
        "• Right-click drag to pan",
        "• 'r' = Reset view",
        "• 'q' = Quit",
        "",
        f"Faces: {sum(m.n_faces for m in meshes.values()):,} total"
    ]
    
    plotter.add_text(
        "\n".join(instructions),
        position='upper_left',
        font_size=12,
        color='white'
    )
    
    # Setup camera
    plotter.camera_position = 'iso'
    plotter.add_axes()
    
    # Add a nice background
    plotter.set_background('navy')
    
    print("✅ PyVista viewer ready!")
    print("   🔴 Red faces = Inlet")
    print("   🟢 Green faces = Outlet") 
    print("   ⚪ Gray faces = Walls")
    
    # Show the visualization
    plotter.show()

def main():
    """Main function"""
    
    print("🚀 GMSH + PYVISTA HYBRID VIEWER")
    print("=" * 60)
    
    # Mesh file
    mesh_file = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\mesh\Rocket Engine.msh"
    
    if not os.path.exists(mesh_file):
        print(f"❌ Mesh file not found: {mesh_file}")
        return
    
    try:
        # Step 1: Use Gmsh to extract and classify faces
        face_data, classification = extract_faces_with_gmsh(mesh_file)
        
        # Step 2: Convert to PyVista meshes
        meshes = create_pyvista_mesh_from_faces(face_data, classification)
        
        if not meshes:
            print("❌ No meshes created")
            return
        
        # Step 3: Visualize with PyVista
        visualize_with_pyvista(meshes)
        
        print("✅ Visualization complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 