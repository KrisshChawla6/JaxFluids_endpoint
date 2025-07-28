#!/usr/bin/env python3
"""
Proper boundary detection using Open3D and trimesh
Uses established libraries to detect actual hollow faces
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add the parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False

def convert_msh_to_ply():
    """Convert MSH file to PLY format for Open3D"""
    
    if not MESHIO_AVAILABLE:
        print("❌ meshio not available. Install with: pip install meshio")
        return None
    
    print("🔄 Converting MSH to PLY format...")
    
    msh_file = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\mesh\Rocket Engine.msh"
    ply_file = "rocket_engine_temp.ply"
    
    # Load MSH file
    mesh = meshio.read(msh_file)
    
    # Extract surface from volume elements (if needed)
    vertices = mesh.points
    surface_faces = []
    
    for cell_block in mesh.cells:
        if cell_block.type == 'tetra':
            # Extract boundary faces from tetrahedra
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
            break
    
    if surface_faces:
        # Create new mesh with only surface faces
        surface_mesh = meshio.Mesh(
            points=vertices,
            cells=[meshio.CellBlock("triangle", np.array(surface_faces))]
        )
        
        # Save as PLY
        meshio.write(ply_file, surface_mesh)
        print(f"✅ Converted to {ply_file} with {len(surface_faces)} faces")
        return ply_file
    
    return None

def detect_boundaries_with_open3d():
    """Use Open3D to detect boundary loops (hollow faces)"""
    
    if not OPEN3D_AVAILABLE:
        print("❌ Open3D not available. Install with: pip install open3d")
        return None
    
    print("\n🔍 DETECTING BOUNDARIES WITH OPEN3D")
    print("=" * 50)
    
    # Convert MSH to PLY first
    ply_file = convert_msh_to_ply()
    if not ply_file:
        print("❌ Failed to convert mesh")
        return None
    
    try:
        # Load mesh with Open3D
        print(f"📂 Loading {ply_file} with Open3D...")
        mesh = o3d.io.read_triangle_mesh(ply_file)
        
        print(f"✅ Loaded mesh:")
        print(f"   Vertices: {len(mesh.vertices)}")
        print(f"   Triangles: {len(mesh.triangles)}")
        
        # Check if mesh is watertight
        print(f"   Is watertight: {mesh.is_watertight()}")
        print(f"   Is orientable: {mesh.is_orientable()}")
        print(f"   Has vertex normals: {mesh.has_vertex_normals()}")
        
        # Get boundary loops using Open3D
        print("\n🎯 Detecting boundary loops...")
        
        # Method 1: Get boundary loops directly
        try:
            # This function finds boundary loops in the mesh
            boundary_loops = mesh.get_boundary_loops()
            print(f"✅ Found {len(boundary_loops)} boundary loops!")
            
            for i, loop in enumerate(boundary_loops):
                loop_vertices = np.asarray(mesh.vertices)[loop]
                loop_length = len(loop)
                
                # Calculate loop perimeter and area
                perimeter = 0
                for j in range(loop_length):
                    v1 = loop_vertices[j]
                    v2 = loop_vertices[(j + 1) % loop_length]
                    perimeter += np.linalg.norm(v2 - v1)
                
                # Estimate area (approximate as circle)
                radius = perimeter / (2 * np.pi)
                estimated_area = np.pi * radius ** 2
                
                # Get center
                center = np.mean(loop_vertices, axis=0)
                
                print(f"\n🔸 Boundary Loop {i}:")
                print(f"   Vertices: {loop_length}")
                print(f"   Perimeter: {perimeter:.2f}")
                print(f"   Estimated area: {estimated_area:.2f}")
                print(f"   Center: {center}")
                
                # Determine if this is inlet or outlet based on size and position
                if estimated_area < 500000:  # Smaller opening
                    print(f"   → INLET (smaller opening)")
                else:
                    print(f"   → OUTLET (larger opening)")
            
            # Visualize the results
            visualize_boundary_loops(mesh, boundary_loops)
            
            return boundary_loops
            
        except Exception as e:
            print(f"⚠️  Direct boundary loop detection failed: {e}")
            
            # Method 2: Alternative approach using edge detection
            print("\n🔄 Trying alternative edge-based approach...")
            return detect_boundaries_alternative(mesh)
    
    finally:
        # Clean up temporary file
        if os.path.exists(ply_file):
            os.remove(ply_file)

def detect_boundaries_alternative(mesh):
    """Alternative boundary detection using edge analysis"""
    
    print("🔍 Using edge-based boundary detection...")
    
    try:
        # Get vertices and triangles
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        print(f"   Analyzing {len(triangles)} triangles...")
        
        # Build edge count dictionary
        edge_count = {}
        triangle_edges = {}
        
        for tri_idx, tri in enumerate(triangles):
            edges = [
                tuple(sorted([tri[0], tri[1]])),
                tuple(sorted([tri[1], tri[2]])),
                tuple(sorted([tri[2], tri[0]]))
            ]
            triangle_edges[tri_idx] = edges
            
            for edge in edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # Find boundary edges (appear only once)
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        
        print(f"✅ Found {len(boundary_edges)} boundary edges")
        
        if not boundary_edges:
            print("❌ No boundary edges found - mesh might be watertight")
            return []
        
        # Group boundary edges into loops
        boundary_loops = find_edge_loops(boundary_edges, vertices)
        
        print(f"✅ Organized into {len(boundary_loops)} boundary loops")
        
        # Analyze each loop
        for i, loop in enumerate(boundary_loops):
            loop_vertices = vertices[loop]
            loop_length = len(loop)
            
            # Calculate perimeter
            perimeter = 0
            for j in range(loop_length):
                v1 = loop_vertices[j]
                v2 = loop_vertices[(j + 1) % loop_length]
                perimeter += np.linalg.norm(v2 - v1)
            
            # Estimate area
            radius = perimeter / (2 * np.pi)
            estimated_area = np.pi * radius ** 2
            center = np.mean(loop_vertices, axis=0)
            
            print(f"\n🔸 Boundary Loop {i}:")
            print(f"   Vertices: {loop_length}")
            print(f"   Perimeter: {perimeter:.2f}")
            print(f"   Estimated area: {estimated_area:.2f}")
            print(f"   Center: {center}")
            
            # Tag as inlet/outlet
            if estimated_area < 500000:
                print(f"   → INLET (smaller opening)")
            else:
                print(f"   → OUTLET (larger opening)")
        
        # Visualize
        visualize_boundary_loops_alternative(vertices, boundary_loops)
        
        return boundary_loops
        
    except Exception as e:
        print(f"❌ Alternative detection failed: {e}")
        return []

def find_edge_loops(boundary_edges, vertices):
    """Find closed loops from boundary edges"""
    
    # Build adjacency graph
    graph = {}
    for edge in boundary_edges:
        v1, v2 = edge
        if v1 not in graph:
            graph[v1] = []
        if v2 not in graph:
            graph[v2] = []
        graph[v1].append(v2)
        graph[v2].append(v1)
    
    # Find loops
    loops = []
    visited = set()
    
    for start_vertex in graph:
        if start_vertex in visited:
            continue
        
        # Trace the loop
        loop = []
        current = start_vertex
        
        while current not in visited:
            visited.add(current)
            loop.append(current)
            
            # Find next vertex
            neighbors = [v for v in graph[current] if v not in visited]
            if not neighbors:
                # Try to close the loop
                if len(loop) > 2 and start_vertex in graph[current]:
                    break
                else:
                    # Dead end
                    break
            
            current = neighbors[0]
        
        if len(loop) > 2:  # Valid loop
            loops.append(loop)
    
    return loops

def visualize_boundary_loops(mesh, boundary_loops):
    """Visualize boundary loops using matplotlib"""
    
    print("\n🎨 Creating visualization...")
    
    vertices = np.asarray(mesh.vertices)
    
    fig = plt.figure(figsize=(15, 10))
    
    # 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Plot each boundary loop with different colors
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    
    for i, loop in enumerate(boundary_loops):
        loop_vertices = vertices[loop]
        color = colors[i % len(colors)]
        
        # Close the loop for plotting
        closed_loop = np.vstack([loop_vertices, loop_vertices[0]])
        
        ax1.plot(closed_loop[:, 0], closed_loop[:, 1], closed_loop[:, 2], 
                color=color, linewidth=3, label=f'Loop {i}')
        ax1.scatter(loop_vertices[:, 0], loop_vertices[:, 1], loop_vertices[:, 2], 
                   color=color, s=50, alpha=0.7)
    
    ax1.set_xlabel('X (Flow Direction)')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Boundary Loops')
    ax1.legend()
    
    # X-Y projection
    ax2 = fig.add_subplot(222)
    for i, loop in enumerate(boundary_loops):
        loop_vertices = vertices[loop]
        color = colors[i % len(colors)]
        closed_loop = np.vstack([loop_vertices, loop_vertices[0]])
        ax2.plot(closed_loop[:, 0], closed_loop[:, 1], color=color, linewidth=2, label=f'Loop {i}')
    ax2.set_xlabel('X (Flow Direction)')
    ax2.set_ylabel('Y')
    ax2.set_title('X-Y Projection')
    ax2.legend()
    ax2.grid(True)
    
    # X-Z projection  
    ax3 = fig.add_subplot(223)
    for i, loop in enumerate(boundary_loops):
        loop_vertices = vertices[loop]
        color = colors[i % len(colors)]
        closed_loop = np.vstack([loop_vertices, loop_vertices[0]])
        ax3.plot(closed_loop[:, 0], closed_loop[:, 2], color=color, linewidth=2, label=f'Loop {i}')
    ax3.set_xlabel('X (Flow Direction)')
    ax3.set_ylabel('Z')
    ax3.set_title('X-Z Projection')
    ax3.legend()
    ax3.grid(True)
    
    # Size comparison
    ax4 = fig.add_subplot(224)
    areas = []
    labels = []
    for i, loop in enumerate(boundary_loops):
        loop_vertices = vertices[loop]
        perimeter = 0
        for j in range(len(loop_vertices)):
            v1 = loop_vertices[j]
            v2 = loop_vertices[(j + 1) % len(loop_vertices)]
            perimeter += np.linalg.norm(v2 - v1)
        area = np.pi * (perimeter / (2 * np.pi)) ** 2
        areas.append(area)
        labels.append(f'Loop {i}')
    
    ax4.bar(labels, areas, color=colors[:len(areas)])
    ax4.set_ylabel('Estimated Area')
    ax4.set_title('Boundary Loop Sizes')
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'boundary_loops_detection.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Saved visualization to: {output_file}")
    
    try:
        plt.show()
        print("📊 Visualization displayed!")
    except:
        print("📊 Interactive display not available, but image saved!")

def visualize_boundary_loops_alternative(vertices, boundary_loops):
    """Alternative visualization for edge-based detection"""
    visualize_boundary_loops_simple(vertices, boundary_loops)

def visualize_boundary_loops_simple(vertices, boundary_loops):
    """Simple visualization function"""
    
    print("\n🎨 Creating boundary visualization...")
    
    fig = plt.figure(figsize=(12, 8))
    
    # 3D plot
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    
    for i, loop in enumerate(boundary_loops):
        loop_vertices = vertices[loop]
        color = colors[i % len(colors)]
        
        # Close the loop
        closed_loop = np.vstack([loop_vertices, loop_vertices[0]])
        
        ax.plot(closed_loop[:, 0], closed_loop[:, 1], closed_loop[:, 2], 
               color=color, linewidth=4, label=f'Loop {i}')
        ax.scatter(loop_vertices[:, 0], loop_vertices[:, 1], loop_vertices[:, 2], 
                  color=color, s=100, alpha=0.8)
    
    ax.set_xlabel('X (Flow Direction)')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Detected Boundary Loops (Hollow Openings)')
    ax.legend()
    
    plt.tight_layout()
    
    output_file = 'boundary_detection_simple.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Saved visualization to: {output_file}")
    
    try:
        plt.show()
    except:
        print("📊 Interactive display not available, but image saved!")

if __name__ == "__main__":
    print("🚀 PROPER BOUNDARY DETECTION FOR ROCKET NOZZLE")
    print("=" * 60)
    
    boundary_loops = detect_boundaries_with_open3d()
    
    if boundary_loops:
        print(f"\n🎉 SUCCESS! Found {len(boundary_loops)} boundary loops")
        print("These represent the actual hollow faces (inlet/outlet openings)")
    else:
        print("\n❌ No boundary loops detected") 