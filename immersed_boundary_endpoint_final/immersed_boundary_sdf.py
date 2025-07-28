#!/usr/bin/env python3
"""
Final Immersed Boundary SDF Endpoint

Professional signed distance function computation for immersed boundary methods.
Uses production-quality pysdf library for NASA-grade SDF computation.

Usage:
    python immersed_boundary_sdf.py mesh.msh --domain "(-100,-150,-150,150,150,150)" --resolution "(100,100,100)" --plot
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import time
import json
from pathlib import Path
from collections import Counter
from datetime import datetime

# Import the production-quality pysdf library
from pysdf import SDF

try:
    import mcubes
    MCUBES_AVAILABLE = True
except ImportError:
    MCUBES_AVAILABLE = False
    try:
        from skimage import measure
        SKIMAGE_AVAILABLE = True
    except ImportError:
        SKIMAGE_AVAILABLE = False


def parse_gmsh_mesh(mesh_file):
    """Parse Gmsh mesh file and extract boundary triangles"""
    print(f"📁 Reading mesh: {mesh_file}")
    
    vertices = []
    tetrahedra = []
    
    with open(mesh_file, 'r') as f:
        lines = f.readlines()
    
    # Parse nodes
    node_start = None
    for i, line in enumerate(lines):
        if line.strip() == '$Nodes':
            node_start = i + 1
            break
    
    if node_start is None:
        raise ValueError("No $Nodes section found")
    
    # Parse nodes (Gmsh 4.1 format)
    node_info = lines[node_start].strip().split()
    num_entity_blocks = int(node_info[0])
    total_nodes = int(node_info[1])
    print(f"📊 Found {total_nodes} nodes in {num_entity_blocks} entity blocks")
    
    line_idx = node_start + 1
    for block in range(num_entity_blocks):
        entity_header = lines[line_idx].strip().split()
        num_nodes_in_block = int(entity_header[3])
        line_idx += 1
        
        # Skip node tags
        for i in range(num_nodes_in_block):
            line_idx += 1
        
        # Read coordinates
        for i in range(num_nodes_in_block):
            parts = lines[line_idx].strip().split()
            if len(parts) >= 3:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                vertices.append([x, y, z])
            line_idx += 1
    
    vertices = np.array(vertices, dtype=np.float32)
    
    # Parse elements
    elem_start = None
    for i, line in enumerate(lines):
        if line.strip() == '$Elements':
            elem_start = i + 1
            break
    
    if elem_start is None:
        raise ValueError("No $Elements section found")
    
    elem_info = lines[elem_start].strip().split()
    num_entity_blocks = int(elem_info[0])
    total_elements = int(elem_info[1])
    print(f"🔺 Found {total_elements} elements in {num_entity_blocks} entity blocks")
    
    line_idx = elem_start + 1
    for block in range(num_entity_blocks):
        entity_header = lines[line_idx].strip().split()
        element_type = int(entity_header[2])
        num_elements_in_block = int(entity_header[3])
        line_idx += 1
        
        for i in range(num_elements_in_block):
            parts = lines[line_idx].strip().split()
            if element_type == 4 and len(parts) >= 5:  # Tetrahedron
                n1, n2, n3, n4 = int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1, int(parts[4]) - 1
                tetrahedra.append([n1, n2, n3, n4])
            line_idx += 1
    
    # Extract boundary triangles from tetrahedra (return actual coordinates)
    print("🔍 Extracting boundary triangles from tetrahedra...")
    all_faces = []
    face_to_coords = {}
    
    for tet in tetrahedra:
        n1, n2, n3, n4 = tet
        faces = [
            tuple(sorted([n1, n2, n3])),
            tuple(sorted([n1, n2, n4])),
            tuple(sorted([n1, n3, n4])),
            tuple(sorted([n2, n3, n4]))
        ]
        
        for face in faces:
            all_faces.append(face)
            if face not in face_to_coords:
                # Store coordinates for this face
                face_to_coords[face] = np.array([
                    vertices[face[0]],
                    vertices[face[1]],
                    vertices[face[2]]
                ])
    
    # Boundary faces appear only once
    face_counts = Counter(all_faces)
    boundary_triangles = []
    
    for face, count in face_counts.items():
        if count == 1:  # Boundary face
            boundary_triangles.append(face_to_coords[face])
    
    print(f"✅ Mesh processed: {len(vertices)} vertices → {len(boundary_triangles)} boundary triangles")
    print(f"📏 Bounds: [{vertices.min(axis=0)}] to [{vertices.max(axis=0)}]")
    
    return boundary_triangles


def compute_sdf(boundary_triangles, domain_bounds, resolution):
    """Compute SDF using pysdf with proper triangle processing"""
    print("🔧 Processing triangles for pysdf...")
    
    # Extract vertices and faces like the working professional version
    vertices = []
    faces = []
    vertex_map = {}
    
    # Build vertex list and map
    vertex_idx = 0
    for triangle in boundary_triangles:
        for vertex in triangle:
            vertex_tuple = tuple(vertex)
            if vertex_tuple not in vertex_map:
                vertices.append(vertex)
                vertex_map[vertex_tuple] = vertex_idx
                vertex_idx += 1
    
    # Build face list
    for triangle in boundary_triangles:
        face = []
        for vertex in triangle:
            vertex_tuple = tuple(vertex)
            face.append(vertex_map[vertex_tuple])
        faces.append(face)
    
    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.uint32)
    
    print(f"🎯 Processed mesh: {len(vertices):,} vertices, {len(faces):,} faces")
    
    # Create SDF object
    print("🚀 Initializing production-quality pysdf...")
    sdf = SDF(vertices, faces)
    print(f"📐 SDF surface area: {sdf.surface_area:.3f}")
    
    # Create grid
    xmin, ymin, zmin, xmax, ymax, zmax = domain_bounds
    nx, ny, nz = resolution
    
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny) 
    z = np.linspace(zmin, zmax, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)
    
    print(f"⚡ Computing SDF on {nx}×{ny}×{nz} = {len(grid_points):,} grid points...")
    
    # Compute SDF
    start_time = time.time()
    sdf_values = sdf(grid_points)
    elapsed = time.time() - start_time
    
    print(f"✅ SDF computation completed in {elapsed:.3f}s")
    print(f"📊 SDF range: [{sdf_values.min():.3f}, {sdf_values.max():.3f}]")
    
    # Reshape to grid
    sdf_grid = sdf_values.reshape(resolution)
    
    return X, Y, Z, sdf_grid


def visualize_sdf(X, Y, Z, sdf_grid, output_file):
    """Visualize the φ=0 contour with professional styling"""
    print("🎨 Extracting φ=0 boundary surface...")
    
    if MCUBES_AVAILABLE:
        vertices, triangles = mcubes.marching_cubes(sdf_grid, 0.0)
        print("🔥 Using PyMCubes for high-quality isosurface extraction")
    elif SKIMAGE_AVAILABLE:
        vertices, triangles, _, _ = measure.marching_cubes(sdf_grid, 0.0)
        print("⚙️ Using scikit-image for isosurface extraction")
    else:
        print("❌ No marching cubes library available")
        return None
    
    # Transform to world coordinates
    nx, ny, nz = sdf_grid.shape
    vertices[:, 0] = X.min() + (vertices[:, 0] / (nx - 1)) * (X.max() - X.min())
    vertices[:, 1] = Y.min() + (vertices[:, 1] / (ny - 1)) * (Y.max() - Y.min())
    vertices[:, 2] = Z.min() + (vertices[:, 2] / (nz - 1)) * (Z.max() - Z.min())
    
    print(f"🎯 Extracted {len(vertices):,} vertices, {len(triangles):,} triangles")
    
    # Create professional visualization
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface with professional styling
    surface = ax.plot_trisurf(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        triangles=triangles,
        alpha=0.85,
        color='steelblue',
        shade=True,
        linewidth=0.1,
        edgecolor='darkblue'
    )
    
    # Professional styling
    ax.set_xlabel('X [units]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y [units]', fontsize=12, fontweight='bold') 
    ax.set_zlabel('Z [units]', fontsize=12, fontweight='bold')
    ax.set_title('Professional Immersed Boundary Surface (φ = 0)\nNASA-Grade SDF using Production pysdf Library', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Set equal aspect ratio
    all_coords = vertices
    max_range = np.array([
        all_coords[:, 0].max() - all_coords[:, 0].min(),
        all_coords[:, 1].max() - all_coords[:, 1].min(),
        all_coords[:, 2].max() - all_coords[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (all_coords[:, 0].max() + all_coords[:, 0].min()) * 0.5
    mid_y = (all_coords[:, 1].max() + all_coords[:, 1].min()) * 0.5
    mid_z = (all_coords[:, 2].max() + all_coords[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Professional viewing angle
    ax.view_init(elev=25, azim=45)
    
    # Add grid and professional appearance
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Saved visualization to {output_file}")
    
    plt.show()
    return fig


def manage_history_window(base_path, max_runs=3):
    """Maintain a 3-slot history window of SDF runs."""
    if not base_path.exists():
        return
    
    # Get all run directories (timestamped folders)
    run_dirs = [d for d in base_path.iterdir() if d.is_dir() and len(d.name) == 15]  # YYYYMMDD_HHMMSS format
    
    # Sort by timestamp (newest first)
    run_dirs.sort(key=lambda x: x.name, reverse=True)
    
    # Remove oldest runs if we exceed max_runs
    if len(run_dirs) >= max_runs:
        for old_dir in run_dirs[max_runs-1:]:  # Keep max_runs-1, remove the rest
            print(f"🗑️  Removing old run: {old_dir.name}")
            import shutil
            shutil.rmtree(old_dir)


def store_sdf_data(X, Y, Z, sdf_grid, output_dir, base_name):
    """Store SDF data properly in sdf_files directory with 3-slot history window"""
    base_output_path = Path(output_dir)
    base_output_path.mkdir(exist_ok=True)
    
    # Manage history window BEFORE creating new run
    manage_history_window(base_output_path, max_runs=3)
    
    # Generate timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create timestamped subdirectory for this run
    run_path = base_output_path / timestamp
    run_path.mkdir(exist_ok=True)
    
    print(f"📁 Storing run in: sdf_files/{timestamp}/")
    
    # Store as numpy binary (fastest loading)
    numpy_file = run_path / f"{base_name}.npz"
    np.savez_compressed(numpy_file,
                       sdf_values=sdf_grid,
                       domain_bounds=[X.min(), Y.min(), Z.min(), X.max(), Y.max(), Z.max()],
                       resolution=sdf_grid.shape,
                       grid_spacing=[
                           (X.max() - X.min()) / (X.shape[0] - 1),
                           (Y.max() - Y.min()) / (Y.shape[1] - 1),
                           (Z.max() - Z.min()) / (Z.shape[2] - 1)
                       ])
    
    # Store SDF matrix as pure .npy file (perfect for PDE initial conditions)
    sdf_matrix_file = run_path / f"{base_name}_sdf_matrix.npy"
    np.save(sdf_matrix_file, sdf_grid)
    
    # Store metadata as JSON
    json_file = run_path / f"{base_name}_metadata.json"
    metadata = {
        'run_id': timestamp,
        'timestamp': datetime.now().isoformat(),
        'domain_bounds': [float(X.min()), float(Y.min()), float(Z.min()), 
                         float(X.max()), float(Y.max()), float(Z.max())],
        'resolution': list(sdf_grid.shape),
        'grid_spacing': [
            float((X.max() - X.min()) / (X.shape[0] - 1)),
            float((Y.max() - Y.min()) / (Y.shape[1] - 1)),
            float((Z.max() - Z.min()) / (Z.shape[2] - 1))
        ],
        'sdf_stats': {
            'min': float(sdf_grid.min()),
            'max': float(sdf_grid.max()),
            'mean': float(sdf_grid.mean()),
            'std': float(sdf_grid.std())
        },
        'total_points': int(sdf_grid.size),
        'inside_fraction': float(np.sum(sdf_grid < 0) / sdf_grid.size)
    }
    
    with open(json_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # JAX-Fluids compatible format
    jax_file = run_path / f"{base_name}_jaxfluids.json"
    jax_data = {
        'run_id': timestamp,
        'domain_bounds': metadata['domain_bounds'],
        'resolution': metadata['resolution'],
        'grid_spacing': metadata['grid_spacing'],
        'sdf_values': sdf_grid.flatten().tolist(),
        'format': 'JAX-Fluids_compatible',
        'timestamp': metadata['timestamp']
    }
    
    with open(jax_file, 'w') as f:
        json.dump(jax_data, f)
    
    print(f"💾 Stored SDF data in run {timestamp}:")
    print(f"   📦 Binary: {numpy_file}")
    print(f"   🔢 SDF Matrix: {sdf_matrix_file}")
    print(f"   📄 Metadata: {json_file}")
    print(f"   🚀 JAX-Fluids: {jax_file}")
    
    return {
        'run_id': timestamp,
        'run_path': str(run_path),
        'binary': str(numpy_file),
        'sdf_matrix': str(sdf_matrix_file),
        'metadata': str(json_file),
        'jaxfluids': str(jax_file)
    }


def main():
    parser = argparse.ArgumentParser(description='Professional Immersed Boundary SDF Endpoint')
    parser.add_argument('mesh_file', help='Path to Gmsh mesh file')
    parser.add_argument('--domain', required=True, help='Domain bounds as "(xmin,ymin,zmin,xmax,ymax,zmax)"')
    parser.add_argument('--resolution', required=True, help='Cartesian grid resolution as "(nx,ny,nz)"')
    parser.add_argument('--plot', action='store_true', help='Enable visualization plotting')
    parser.add_argument('--output-dir', default='sdf_files', help='Output directory for SDF files')
    
    args = parser.parse_args()
    
    # Parse arguments
    domain_bounds = eval(args.domain)
    resolution = eval(args.resolution)
    mesh_name = Path(args.mesh_file).stem
    
    print("=" * 70)
    print("🚀 PROFESSIONAL IMMERSED BOUNDARY SDF ENDPOINT")
    print("=" * 70)
    print(f"📁 Mesh: {args.mesh_file}")
    print(f"📐 Domain: {domain_bounds}")
    print(f"🔢 Resolution: {resolution}")
    print(f"🎨 Plotting: {'ON' if args.plot else 'OFF'}")
    print(f"💾 Output: {args.output_dir}/")
    print()
    
    try:
        # Parse mesh
        boundary_triangles = parse_gmsh_mesh(args.mesh_file)
        
        # Compute SDF
        X, Y, Z, sdf_grid = compute_sdf(boundary_triangles, domain_bounds, resolution)
        
        # Store SDF data
        stored_files = store_sdf_data(X, Y, Z, sdf_grid, args.output_dir, mesh_name)
        
        # Visualize if requested
        if args.plot:
            viz_file = Path(stored_files['run_path']) / f"{mesh_name}_visualization.png"
            visualize_sdf(X, Y, Z, sdf_grid, str(viz_file))
            stored_files['visualization'] = str(viz_file)
        
        print()
        print("=" * 70)
        print("🎉 SDF COMPUTATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"⏱️  Total processing time: {time.time():.1f}s")
        print(f"📊 SDF range: [{sdf_grid.min():.3f}, {sdf_grid.max():.3f}]")
        print(f"💾 Files stored in: {args.output_dir}/")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 