#!/usr/bin/env python3
"""
Extract hollow faces directly from mesh topology
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
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False

def analyze_mesh_topology():
    """Analyze the mesh topology to find hollow faces"""
    
    if not MESHIO_AVAILABLE:
        print("❌ meshio not available. Install with: pip install meshio")
        return
    
    print("🔍 ANALYZING MESH TOPOLOGY FOR HOLLOW FACES")
    print("=" * 50)
    
    # Load the mesh file
    mesh_file = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\mesh\Rocket Engine.msh"
    print(f"📂 Loading: {mesh_file}")
    
    mesh = meshio.read(mesh_file)
    
    print(f"✅ Loaded mesh with {len(mesh.points)} points")
    print(f"📊 Cell blocks: {len(mesh.cells)}")
    
    # Analyze each cell block
    for i, cell_block in enumerate(mesh.cells):
        print(f"\n📋 Cell Block {i}:")
        print(f"   Type: {cell_block.type}")
        print(f"   Count: {len(cell_block.data)}")
        
        if hasattr(cell_block, 'tags') and cell_block.tags:
            print(f"   Tags: {cell_block.tags}")
    
    # Check if there are physical groups or tags that might indicate inlet/outlet
    if hasattr(mesh, 'field_data') and mesh.field_data:
        print(f"\n🏷️  Field Data (Physical Groups):")
        for name, data in mesh.field_data.items():
            print(f"   {name}: {data}")
    
    if hasattr(mesh, 'cell_data') and mesh.cell_data:
        print(f"\n📋 Cell Data:")
        for key, data in mesh.cell_data.items():
            print(f"   {key}: {len(data)} entries")
    
    if hasattr(mesh, 'point_data') and mesh.point_data:
        print(f"\n📍 Point Data:")
        for key, data in mesh.point_data.items():
            print(f"   {key}: shape {data.shape}")
    
    # Look for boundary faces specifically
    print(f"\n🔍 ANALYZING BOUNDARY STRUCTURE:")
    
    surface_cells = []
    volume_cells = []
    
    for cell_block in mesh.cells:
        if cell_block.type in ['triangle', 'quad']:
            surface_cells.append(cell_block)
            print(f"   Found surface elements: {cell_block.type} ({len(cell_block.data)})")
        elif cell_block.type in ['tetra', 'hexahedron', 'pyramid', 'wedge']:
            volume_cells.append(cell_block)
            print(f"   Found volume elements: {cell_block.type} ({len(cell_block.data)})")
    
    # If we have surface cells, analyze them for potential inlet/outlet
    if surface_cells:
        print(f"\n🎯 ANALYZING SURFACE ELEMENTS FOR OPENINGS:")
        analyze_surface_elements(mesh, surface_cells)
    else:
        print(f"\n⚠️  No direct surface elements found. Need to extract from volume.")
        extract_boundary_from_volume(mesh, volume_cells)

def analyze_surface_elements(mesh, surface_cells):
    """Analyze surface elements to find inlet/outlet openings"""
    
    vertices = mesh.points
    
    for i, surface_block in enumerate(surface_cells):
        print(f"\n🔍 Surface Block {i} ({surface_block.type}):")
        
        faces = surface_block.data
        face_centroids = []
        face_normals = []
        face_areas = []
        
        for face_indices in faces:
            face_vertices = vertices[face_indices]
            
            # Calculate centroid
            centroid = face_vertices.mean(axis=0)
            face_centroids.append(centroid)
            
            # Calculate normal and area
            if len(face_indices) == 3:  # Triangle
                v1 = face_vertices[1] - face_vertices[0]
                v2 = face_vertices[2] - face_vertices[0]
                normal = np.cross(v1, v2)
                area = 0.5 * np.linalg.norm(normal)
                normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else normal
            else:  # Quad
                v1 = face_vertices[1] - face_vertices[0]
                v2 = face_vertices[2] - face_vertices[0]
                normal = np.cross(v1, v2)
                area = np.linalg.norm(normal) * 0.5
                v3 = face_vertices[3] - face_vertices[0]
                area += np.linalg.norm(np.cross(v2, v3)) * 0.5
                normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else normal
            
            face_normals.append(normal)
            face_areas.append(area)
        
        face_centroids = np.array(face_centroids)
        face_normals = np.array(face_normals)
        face_areas = np.array(face_areas)
        
        # Analyze spatial distribution
        print(f"   📍 Centroids range:")
        print(f"      X: {face_centroids[:, 0].min():.2f} to {face_centroids[:, 0].max():.2f}")
        print(f"      Y: {face_centroids[:, 1].min():.2f} to {face_centroids[:, 1].max():.2f}")
        print(f"      Z: {face_centroids[:, 2].min():.2f} to {face_centroids[:, 2].max():.2f}")
        
        # Find potential end faces by clustering normals
        find_end_faces_by_normals(face_centroids, face_normals, face_areas, i)

def find_end_faces_by_normals(centroids, normals, areas, block_id):
    """Find end faces by analyzing normal directions"""
    
    print(f"\n🧭 NORMAL ANALYSIS for Block {block_id}:")
    
    # Find dominant normal directions
    primary_axes = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    axis_names = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
    
    for axis_idx, axis_dir in enumerate(primary_axes):
        axis_dir = np.array(axis_dir)
        
        # Find faces with normals aligned to this direction
        dots = np.abs(np.dot(normals, axis_dir))
        aligned_faces = dots > 0.8  # Faces with normals within ~37 degrees
        
        if np.sum(aligned_faces) > 0:
            aligned_centroids = centroids[aligned_faces]
            aligned_areas = areas[aligned_faces]
            
            print(f"   {axis_names[axis_idx]} aligned: {np.sum(aligned_faces)} faces, total area: {np.sum(aligned_areas):.2f}")
            
            if np.sum(aligned_faces) > 10:  # Significant number of faces
                # Check if they're clustered at geometry extremes
                coord_idx = axis_idx // 2  # 0 for X, 1 for Y, 2 for Z
                coords = aligned_centroids[:, coord_idx]
                
                if axis_idx % 2 == 0:  # Positive direction
                    extreme_value = coords.max()
                    extreme_faces = coords > (extreme_value - 50)  # Within 50 units
                else:  # Negative direction
                    extreme_value = coords.min()
                    extreme_faces = coords < (extreme_value + 50)  # Within 50 units
                
                if np.sum(extreme_faces) > 5:
                    extreme_area = np.sum(aligned_areas[extreme_faces])
                    print(f"      -> Potential opening: {np.sum(extreme_faces)} faces, area: {extreme_area:.2f}")
                    
                    # Estimate opening size
                    extreme_centroids = aligned_centroids[extreme_faces]
                    other_axes = [i for i in range(3) if i != coord_idx]
                    
                    ranges = []
                    for other_axis in other_axes:
                        axis_range = extreme_centroids[:, other_axis].max() - extreme_centroids[:, other_axis].min()
                        ranges.append(axis_range)
                    
                    diameter = np.mean(ranges)
                    estimated_area = np.pi * (diameter / 2) ** 2
                    
                    print(f"      -> Estimated opening diameter: {diameter:.2f}")
                    print(f"      -> Estimated circular area: {estimated_area:.2f}")

def extract_boundary_from_volume(mesh, volume_cells):
    """Extract boundary faces from volume elements"""
    
    print(f"\n🔧 EXTRACTING BOUNDARY FROM VOLUME ELEMENTS:")
    
    if not volume_cells:
        print("   ❌ No volume elements found")
        return
    
    # Use the first volume element block
    volume_block = volume_cells[0]
    print(f"   📊 Analyzing {volume_block.type} elements: {len(volume_block.data)}")
    
    vertices = mesh.points
    
    if volume_block.type == 'tetra':
        # For tetrahedra, extract boundary faces
        face_count = {}
        tetra_faces = [
            [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]  # 4 faces per tetrahedron
        ]
        
        for tetra in volume_block.data:
            for face_indices in tetra_faces:
                # Create face tuple (sorted for consistent identification)
                face = tuple(sorted([tetra[face_indices[0]], tetra[face_indices[1]], tetra[face_indices[2]]]))
                face_count[face] = face_count.get(face, 0) + 1
        
        # Boundary faces appear only once
        boundary_faces = [list(face) for face, count in face_count.items() if count == 1]
        
        print(f"   ✅ Extracted {len(boundary_faces)} boundary faces")
        
        # Now analyze these boundary faces for openings
        if boundary_faces:
            analyze_extracted_boundary(vertices, boundary_faces)

def analyze_extracted_boundary(vertices, boundary_faces):
    """Analyze extracted boundary faces to find openings"""
    
    print(f"\n🎯 ANALYZING EXTRACTED BOUNDARY FACES:")
    
    face_centroids = []
    face_normals = []
    face_areas = []
    
    for face_indices in boundary_faces:
        face_vertices = vertices[face_indices]
        
        # Calculate properties
        centroid = face_vertices.mean(axis=0)
        v1 = face_vertices[1] - face_vertices[0]
        v2 = face_vertices[2] - face_vertices[0]
        normal = np.cross(v1, v2)
        area = 0.5 * np.linalg.norm(normal)
        normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else normal
        
        face_centroids.append(centroid)
        face_normals.append(normal)
        face_areas.append(area)
    
    face_centroids = np.array(face_centroids)
    face_normals = np.array(face_normals)
    face_areas = np.array(face_areas)
    
    # Find openings by normal analysis
    find_end_faces_by_normals(face_centroids, face_normals, face_areas, "extracted")
    
    # Also create a simple visualization
    create_boundary_visualization(face_centroids, face_normals, face_areas)

def create_boundary_visualization(centroids, normals, areas):
    """Create a visualization of the boundary faces"""
    
    print(f"\n🎨 Creating boundary visualization...")
    
    fig = plt.figure(figsize=(15, 10))
    
    # 3D scatter plot colored by normal direction
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Color by X-normal component (flow direction)
    x_normals = normals[:, 0]
    scatter = ax1.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
                         c=x_normals, s=areas/areas.max()*50, cmap='RdYlBu', alpha=0.7)
    
    ax1.set_xlabel('X (Flow Direction)')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Boundary Faces (colored by X-normal)')
    plt.colorbar(scatter, ax=ax1, label='X-normal component')
    
    # X-Y projection
    ax2 = fig.add_subplot(222)
    scatter2 = ax2.scatter(centroids[:, 0], centroids[:, 1], c=x_normals, s=20, cmap='RdYlBu', alpha=0.7)
    ax2.set_xlabel('X (Flow Direction)')
    ax2.set_ylabel('Y')
    ax2.set_title('X-Y Projection')
    ax2.grid(True)
    plt.colorbar(scatter2, ax=ax2, label='X-normal')
    
    # X-Z projection
    ax3 = fig.add_subplot(223)
    scatter3 = ax3.scatter(centroids[:, 0], centroids[:, 2], c=x_normals, s=20, cmap='RdYlBu', alpha=0.7)
    ax3.set_xlabel('X (Flow Direction)')
    ax3.set_ylabel('Z')
    ax3.set_title('X-Z Projection')
    ax3.grid(True)
    plt.colorbar(scatter3, ax=ax3, label='X-normal')
    
    # Histogram of X coordinates
    ax4 = fig.add_subplot(224)
    ax4.hist(centroids[:, 0], bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    ax4.set_xlabel('X Coordinate')
    ax4.set_ylabel('Number of Boundary Faces')
    ax4.set_title('Boundary Face Distribution Along Flow Axis')
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'boundary_face_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Saved boundary analysis to: {output_file}")
    
    # Try to show
    try:
        plt.show()
        print("📊 Visualization displayed!")
    except:
        print("📊 Interactive display not available, but image saved!")

if __name__ == "__main__":
    analyze_mesh_topology() 