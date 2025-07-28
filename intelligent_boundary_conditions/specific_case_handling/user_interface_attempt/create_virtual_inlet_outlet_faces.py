#!/usr/bin/env python3
"""
Create Virtual Inlet and Outlet Faces
Creates virtual face surfaces across hollow openings for CFD boundary conditions
Based on established CFD practices for boundary face creation
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

try:
    from scipy.spatial import ConvexHull
    print(f"✅ SciPy available")
except ImportError:
    print("❌ SciPy not available")
    exit()

def find_boundary_loops_with_gmsh(mesh_file):
    """Use Gmsh to find boundary loops at inlet and outlet"""
    
    print("🔍 Finding boundary loops with Gmsh...")
    
    gmsh.initialize()
    try:
        gmsh.open(mesh_file)
        
        # Get all surface entities
        entities = gmsh.model.getEntities(dim=2)  # 2D surfaces
        
        inlet_edges = []
        outlet_edges = []
        
        for surf_dim, surf_tag in entities:
            # Get the boundary edges of each surface
            boundary = gmsh.model.getBoundary([(surf_dim, surf_tag)], oriented=False)
            edge_nodes = []
            
            for edge_dim, edge_tag in boundary:
                if edge_dim == 1:  # 1D edges
                    nodes = gmsh.model.mesh.getNodes(edge_dim, edge_tag)
                    if len(nodes[1]) > 0:
                        coords = nodes[1].reshape(-1, 3)
                        edge_nodes.extend(coords)
            
            if len(edge_nodes) > 0:
                edge_points = np.array(edge_nodes)
                
                # Classify by X position (flow direction)
                x_center = edge_points[:, 0].mean()
                
                if x_center < 500:  # Inlet region
                    inlet_edges.extend(edge_points)
                elif x_center > 1200:  # Outlet region
                    outlet_edges.extend(edge_points)
        
        return np.array(inlet_edges), np.array(outlet_edges)
        
    finally:
        gmsh.finalize()

def create_virtual_face_from_boundary_points(boundary_points, face_type="inlet"):
    """Create a virtual face surface from boundary points"""
    
    print(f"🔧 Creating virtual {face_type} face...")
    
    if len(boundary_points) < 3:
        print(f"❌ Not enough boundary points for {face_type}")
        return None
    
    # Get the main plane (assume flow is along X-axis)
    x_pos = boundary_points[:, 0].mean()
    
    # Project points to the cross-sectional plane
    # For inlet/outlet, we want the Y-Z plane at a specific X
    yz_points = boundary_points[:, 1:]  # Y,Z coordinates
    
    # Create a convex hull in the Y-Z plane
    try:
        hull = ConvexHull(yz_points)
        hull_points = yz_points[hull.vertices]
        
        # Create triangulated face
        triangles = []
        center_yz = hull_points.mean(axis=0)
        
        # Create triangles from center to each edge
        for i in range(len(hull_points)):
            j = (i + 1) % len(hull_points)
            triangle = [
                [x_pos, center_yz[0], center_yz[1]],  # Center point
                [x_pos, hull_points[i][0], hull_points[i][1]],  # Hull point i
                [x_pos, hull_points[j][0], hull_points[j][1]]   # Hull point j
            ]
            triangles.append(triangle)
        
        return np.array(triangles), hull_points, x_pos
        
    except Exception as e:
        print(f"❌ Could not create convex hull for {face_type}: {e}")
        return None

def create_pyvista_mesh_with_virtual_faces(original_mesh_file, inlet_triangles, outlet_triangles):
    """Create PyVista mesh including original surface + virtual faces"""
    
    print("🔄 Creating combined mesh with virtual faces...")
    
    # Load original surface mesh
    mesh = meshio.read(original_mesh_file)
    pv_mesh = pv.from_meshio(mesh)
    surface = pv_mesh.extract_surface()
    
    # Create virtual inlet face
    if inlet_triangles is not None:
        inlet_points = inlet_triangles.reshape(-1, 3)
        inlet_faces = []
        for i in range(0, len(inlet_points), 3):
            inlet_faces.append([3, i, i+1, i+2])  # Triangle with 3 vertices
        
        inlet_mesh = pv.PolyData(inlet_points, np.array(inlet_faces))
    else:
        inlet_mesh = None
    
    # Create virtual outlet face
    if outlet_triangles is not None:
        outlet_points = outlet_triangles.reshape(-1, 3)
        outlet_faces = []
        for i in range(0, len(outlet_points), 3):
            outlet_faces.append([3, i, i+1, i+2])  # Triangle with 3 vertices
        
        outlet_mesh = pv.PolyData(outlet_points, np.array(outlet_faces))
    else:
        outlet_mesh = None
    
    # Combine all meshes
    combined_meshes = [surface]
    if inlet_mesh is not None:
        combined_meshes.append(inlet_mesh)
    if outlet_mesh is not None:
        combined_meshes.append(outlet_mesh)
    
    # Create tags for each region
    n_wall_faces = surface.n_cells
    n_inlet_faces = inlet_mesh.n_cells if inlet_mesh else 0
    n_outlet_faces = outlet_mesh.n_cells if outlet_mesh else 0
    
    # Create face type array
    face_types = np.ones(n_wall_faces + n_inlet_faces + n_outlet_faces)
    face_types[:n_wall_faces] = 1  # Wall = 1
    face_types[n_wall_faces:n_wall_faces + n_inlet_faces] = 0  # Inlet = 0
    face_types[n_wall_faces + n_inlet_faces:] = 2  # Outlet = 2
    
    # Combine all meshes
    if len(combined_meshes) > 1:
        combined_mesh = combined_meshes[0]
        for mesh_part in combined_meshes[1:]:
            combined_mesh = combined_mesh.merge(mesh_part)
    else:
        combined_mesh = combined_meshes[0]
    
    # Add face type data
    combined_mesh['face_type'] = face_types
    
    return combined_mesh, n_wall_faces, n_inlet_faces, n_outlet_faces

def visualize_with_virtual_faces(combined_mesh, inlet_points=None, outlet_points=None):
    """Visualize the mesh with virtual inlet and outlet faces"""
    
    print("🎮 Creating visualization with virtual faces...")
    
    # Create plotter
    plotter = pv.Plotter(window_size=[1600, 1000])
    
    # Add main mesh with face type coloring
    plotter.add_mesh(
        combined_mesh,
        scalars='face_type',
        cmap=['red', 'lightgray', 'green'],  # 0=red (inlet), 1=gray (wall), 2=green (outlet)
        show_edges=True,
        edge_color='white',
        line_width=0.2,
        opacity=0.9
    )
    
    # Add boundary points if available
    if inlet_points is not None:
        plotter.add_points(
            inlet_points,
            color='darkred',
            point_size=8,
            render_points_as_spheres=True
        )
    
    if outlet_points is not None:
        plotter.add_points(
            outlet_points,
            color='darkgreen',
            point_size=8,
            render_points_as_spheres=True
        )
    
    # Add color bar
    plotter.add_scalar_bar(
        title="Face Type",
        n_labels=3,
        color='white',
        label_font_size=14
    )
    
    # Add instructions
    instructions = [
        "🚀 ROCKET NOZZLE WITH VIRTUAL FACES",
        "",
        "FACE TYPES:",
        "🔴 RED = Virtual Inlet Face",
        "🟢 GREEN = Virtual Outlet Face", 
        "⚪ GRAY = Wall Surfaces",
        "",
        "VIRTUAL FACES:",
        "• Created across hollow openings",
        "• Ready for CFD boundary conditions",
        "• Properly sealed flow domain",
        "",
        "CONTROLS:",
        "• Drag to rotate • Scroll to zoom",
        "• Right-click drag to pan",
        "• 'r' = Reset • 'q' = Quit",
        "",
        f"Total faces: {combined_mesh.n_cells:,}"
    ]
    
    plotter.add_text(
        "\n".join(instructions),
        position='upper_left',
        font_size=12,
        color='white'
    )
    
    # Setup camera and lighting
    plotter.camera_position = 'iso'
    plotter.add_axes()
    plotter.set_background('navy')
    
    print("✅ Visualization ready!")
    print("   🔴 Red faces = Virtual inlet")
    print("   🟢 Green faces = Virtual outlet")
    print("   ⚪ Gray faces = Wall surfaces")
    
    # Show the visualization
    plotter.show()

def main():
    """Main function"""
    
    print("🚀 VIRTUAL INLET/OUTLET FACE CREATOR")
    print("=" * 60)
    
    # Mesh file
    mesh_file = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\mesh\Rocket Engine.msh"
    
    if not os.path.exists(mesh_file):
        print(f"❌ Mesh file not found: {mesh_file}")
        return
    
    try:
        # Step 1: Find boundary loops
        inlet_boundary, outlet_boundary = find_boundary_loops_with_gmsh(mesh_file)
        
        if len(inlet_boundary) == 0 or len(outlet_boundary) == 0:
            print("❌ Could not find inlet or outlet boundary points")
            return
        
        print(f"   Found {len(inlet_boundary)} inlet boundary points")
        print(f"   Found {len(outlet_boundary)} outlet boundary points")
        
        # Step 2: Create virtual faces
        inlet_result = create_virtual_face_from_boundary_points(inlet_boundary, "inlet")
        outlet_result = create_virtual_face_from_boundary_points(outlet_boundary, "outlet")
        
        if inlet_result is None or outlet_result is None:
            print("❌ Could not create virtual faces")
            return
        
        inlet_triangles, inlet_hull, inlet_x = inlet_result
        outlet_triangles, outlet_hull, outlet_x = outlet_result
        
        print(f"   Created inlet face at X={inlet_x:.1f} with {len(inlet_triangles)} triangles")
        print(f"   Created outlet face at X={outlet_x:.1f} with {len(outlet_triangles)} triangles")
        
        # Step 3: Create combined mesh
        combined_mesh, n_wall, n_inlet, n_outlet = create_pyvista_mesh_with_virtual_faces(
            mesh_file, inlet_triangles, outlet_triangles
        )
        
        print(f"   Combined mesh: {n_wall} wall + {n_inlet} inlet + {n_outlet} outlet faces")
        
        # Step 4: Visualize
        visualize_with_virtual_faces(combined_mesh, inlet_boundary, outlet_boundary)
        
        print("✅ Virtual faces created successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 