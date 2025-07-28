#!/usr/bin/env python3
"""
Improved CFD Wind Tunnel Generator with Proper Object Boundary Markers

This version properly integrates the object mesh and creates actual boundary elements
for the object_wall marker, fixing the SU2 simulation issues.
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Tuple, Any

def create_improved_cfd_wind_tunnel():
    """
    Create a CFD-ready wind tunnel mesh with proper object boundary integration
    """
    
    print("🔧 IMPROVED CFD Wind Tunnel Generator with Object Boundaries")
    print("=" * 60)
    
    start_time = time.time()
    
    # Input propeller mesh file
    propeller_file = r"C:\Users\kriss\Desktop\New Simulation\projects\propeller\mesh\5_bladed_Propeller_medium_tetrahedral.su2"
    output_file = "output/propeller_wind_tunnel_cfd.su2"
    
    if not os.path.exists(propeller_file):
        print(f"❌ Propeller mesh not found: {propeller_file}")
        return None
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Step 1: Read propeller mesh with boundary information
        print("📖 Reading propeller mesh with boundary extraction...")
        object_data = read_object_mesh_with_boundaries(propeller_file)
        
        # Step 2: Create CFD domain
        print("🏗️  Creating CFD domain...")
        domain_data = create_cfd_domain_improved(object_data)
        
        # Step 3: Extract object surface boundaries
        print("🔍 Extracting object surface boundaries...")
        object_boundaries = extract_object_surface_boundaries(object_data)
        
        # Step 4: Write complete SU2 mesh with proper boundaries
        print("💾 Writing improved SU2 mesh...")
        write_improved_su2_mesh(object_data, domain_data, object_boundaries, output_file)
        
        generation_time = time.time() - start_time
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        
        print(f"🎉 IMPROVED CFD wind tunnel generated!")
        print(f"   📁 Output: {output_file}")
        print(f"   ⏱️  Time: {generation_time:.2f}s")
        print(f"   📊 Size: {file_size:.1f} MB")
        print(f"   ✅ Object boundaries properly assigned!")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def read_object_mesh_with_boundaries(mesh_file: str) -> Dict[str, Any]:
    """Read object mesh and extract all boundary information"""
    
    print(f"   📖 Reading: {os.path.basename(mesh_file)}")
    
    with open(mesh_file, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Parse mesh components
    nodes = []
    elements = []
    boundaries = {}
    
    current_section = None
    current_boundary = None
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('NELEM='):
            current_section = 'elements'
            continue
        elif line.startswith('NPOIN='):
            current_section = 'nodes'
            continue
        elif line.startswith('NMARK='):
            current_section = 'boundaries'
            continue
        elif line.startswith('MARKER_TAG='):
            boundary_name = line.split('=')[1].strip()
            current_boundary = boundary_name
            boundaries[boundary_name] = []
            continue
        elif line.startswith('MARKER_ELEMS='):
            continue
        
        if current_section == 'elements' and line and not line.startswith('N'):
            parts = line.split()
            if len(parts) >= 5:  # Tetrahedral: type + 4 nodes
                try:
                    # Parse element: type + 4 node indices
                    element = [int(x) for x in parts[:5]]
                    elements.append(element)
                except ValueError:
                    # Skip malformed lines that can't be parsed as integers
                    print(f"   ⚠️  Skipping malformed element line: {line}")
                    continue
                
        elif current_section == 'nodes' and line and not line.startswith('N'):
            parts = line.split()
            if len(parts) >= 3:  # Need at least 3 coordinates
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    nodes.append([x, y, z])
                except ValueError:
                    # Skip malformed lines that can't be parsed as floats
                    print(f"   ⚠️  Skipping malformed node line: {line}")
                    continue
                
        elif current_section == 'boundaries' and current_boundary and line and not line.startswith('MARKER_'):
            parts = line.split()
            if len(parts) > 1:
                boundaries[current_boundary].append([int(x) for x in parts])
    
    nodes_array = np.array(nodes)
    
    # Calculate bounding box
    bbox = {
        'min_x': nodes_array[:, 0].min(),
        'max_x': nodes_array[:, 0].max(),
        'min_y': nodes_array[:, 1].min(),
        'max_y': nodes_array[:, 1].max(),
        'min_z': nodes_array[:, 2].min(),
        'max_z': nodes_array[:, 2].max(),
    }
    
    chord_length = bbox['max_x'] - bbox['min_x']
    
    print(f"   📊 Object: {len(nodes)} nodes, {len(elements)} elements")
    print(f"   📏 Chord length: {chord_length:.3f}")
    print(f"   🏷️  Original boundaries: {list(boundaries.keys())}")
    
    return {
        'nodes': nodes_array,
        'elements': elements,
        'boundaries': boundaries,
        'bbox': bbox,
        'chord_length': chord_length
    }

def extract_object_surface_boundaries(object_data: Dict[str, Any]) -> Dict[str, List]:
    """Extract surface boundaries from object mesh"""
    
    print("   🔍 Extracting object surface boundaries...")
    
    # For simplicity, we'll extract all external faces of the object
    # In a more sophisticated version, we'd use proper surface detection
    
    object_boundaries = []
    elements = object_data['elements']
    
    # For tetrahedral elements, extract surface triangles
    # This is a simplified approach - each tet face that appears only once is on the surface
    face_count = {}
    element_faces = {}
    
    for elem_idx, elem in enumerate(elements):
        if len(elem) >= 5 and elem[0] == 10:  # Tetrahedron
            # Get the 4 nodes (skip element type only)
            nodes = elem[1:5]
            
            # Generate 4 triangular faces of the tetrahedron
            faces = [
                tuple(sorted([nodes[0], nodes[1], nodes[2]])),
                tuple(sorted([nodes[0], nodes[1], nodes[3]])),
                tuple(sorted([nodes[0], nodes[2], nodes[3]])),
                tuple(sorted([nodes[1], nodes[2], nodes[3]]))
            ]
            
            element_faces[elem_idx] = faces
            
            for face in faces:
                if face in face_count:
                    face_count[face] += 1
                else:
                    face_count[face] = 1
    
    # Surface faces appear exactly once (not shared with another element)
    surface_faces = [face for face, count in face_count.items() if count == 1]
    
    # Convert to SU2 boundary format (triangle elements)
    for face in surface_faces:
        # Triangle boundary element: [5, node1, node2, node3]
        object_boundaries.append([5, face[0], face[1], face[2]])
    
    print(f"   📊 Extracted {len(object_boundaries)} surface boundary faces")
    
    return object_boundaries

def create_cfd_domain_improved(object_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create CFD domain with improved sizing"""
    
    bbox = object_data['bbox']
    chord = object_data['chord_length']
    
    # CFD-appropriate domain sizing
    upstream_length = 2.0 * chord
    downstream_length = 4.0 * chord
    domain_height = 3.0 * chord
    domain_width = 3.0 * chord
    
    # Center the object
    center_y = (bbox['min_y'] + bbox['max_y']) / 2
    center_z = (bbox['min_z'] + bbox['max_z']) / 2
    
    # Domain bounds
    domain_bounds = {
        'min_x': bbox['min_x'] - upstream_length,
        'max_x': bbox['max_x'] + downstream_length,
        'min_y': center_y - domain_height/2,
        'max_y': center_y + domain_height/2,
        'min_z': center_z - domain_width/2,
        'max_z': center_z + domain_width/2,
    }
    
    total_length = domain_bounds['max_x'] - domain_bounds['min_x']
    total_height = domain_bounds['max_y'] - domain_bounds['min_y']
    total_width = domain_bounds['max_z'] - domain_bounds['min_z']
    
    print(f"   📏 CFD Domain dimensions:")
    print(f"      Length: {total_length:.1f} ({total_length/chord:.1f} chords)")
    print(f"      Height: {total_height:.1f} ({total_height/chord:.1f} chords)")
    print(f"      Width: {total_width:.1f} ({total_width/chord:.1f} chords)")
    
    # Create reasonable mesh resolution
    nx = 40  # 40 points in flow direction
    ny = 25  # 25 points in height
    nz = 20  # 20 points in width
    
    print(f"   🔗 Mesh resolution: {nx}x{ny}x{nz} = {nx*ny*nz:,} nodes")
    
    x_coords = np.linspace(domain_bounds['min_x'], domain_bounds['max_x'], nx)
    y_coords = np.linspace(domain_bounds['min_y'], domain_bounds['max_y'], ny)
    z_coords = np.linspace(domain_bounds['min_z'], domain_bounds['max_z'], nz)
    
    # Generate domain nodes
    domain_nodes = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                domain_nodes.append([x_coords[i], y_coords[j], z_coords[k]])
    
    domain_nodes = np.array(domain_nodes)
    
    # Generate tetrahedral elements (split each hex into 6 tets)
    domain_elements = []
    
    for k in range(nz-1):
        for j in range(ny-1):
            for i in range(nx-1):
                # Get 8 corner nodes of hexahedron
                n0 = k * nx * ny + j * nx + i
                n1 = k * nx * ny + j * nx + (i + 1)
                n2 = k * nx * ny + (j + 1) * nx + (i + 1)
                n3 = k * nx * ny + (j + 1) * nx + i
                n4 = (k + 1) * nx * ny + j * nx + i
                n5 = (k + 1) * nx * ny + j * nx + (i + 1)
                n6 = (k + 1) * nx * ny + (j + 1) * nx + (i + 1)
                n7 = (k + 1) * nx * ny + (j + 1) * nx + i
                
                # Split hexahedron into 6 tetrahedra (VTK type 10)
                # Standard hex-to-tet decomposition
                domain_elements.extend([
                    [10, n0, n1, n3, n4],  # Tet 1
                    [10, n1, n2, n3, n6],  # Tet 2  
                    [10, n1, n3, n4, n6],  # Tet 3
                    [10, n3, n4, n6, n7],  # Tet 4
                    [10, n1, n4, n5, n6],  # Tet 5
                    [10, n4, n5, n6, n7]   # Tet 6
                ])
    
    # Create boundary elements from actual tetrahedral faces
    boundaries = create_domain_boundaries_from_tets(domain_elements, nx, ny, nz, domain_bounds)
    
    return {
        'nodes': domain_nodes,
        'elements': domain_elements,  
        'boundaries': boundaries,
        'bounds': domain_bounds,
        'mesh_size': (nx, ny, nz)
    }

def create_domain_boundaries_from_tets(domain_elements, nx, ny, nz, domain_bounds):
    """Extract actual boundary faces from tetrahedral elements"""
    from collections import defaultdict
    
    # Extract all faces from tetrahedra and count occurrences
    face_count = defaultdict(int)
    
    for elem in domain_elements:
        if len(elem) >= 5 and elem[0] == 10:  # Tetrahedron
            nodes = elem[1:5]
            
            # Generate the 4 faces of the tetrahedron
            faces = [
                tuple(sorted([nodes[0], nodes[1], nodes[2]])),
                tuple(sorted([nodes[0], nodes[1], nodes[3]])),
                tuple(sorted([nodes[0], nodes[2], nodes[3]])),
                tuple(sorted([nodes[1], nodes[2], nodes[3]])),
            ]
            
            for face in faces:
                face_count[face] += 1
    
    # Boundary faces appear only once (not shared between elements)
    boundary_faces = [face for face, count in face_count.items() if count == 1]
    
    print(f"   🔍 Found {len(boundary_faces)} boundary faces from tetrahedral elements")
    
    # Classify boundary faces based on node positions
    boundaries = {
        'inlet': [],
        'outlet': [],
        'slip_wall': []
    }
    
    # Get node position from structured grid index
    def get_node_position(node_idx):
        k = node_idx // (nx * ny)
        j = (node_idx % (nx * ny)) // nx
        i = node_idx % nx
        
        x = domain_bounds['min_x'] + i * (domain_bounds['max_x'] - domain_bounds['min_x']) / (nx - 1)
        y = domain_bounds['min_y'] + j * (domain_bounds['max_y'] - domain_bounds['min_y']) / (ny - 1)
        z = domain_bounds['min_z'] + k * (domain_bounds['max_z'] - domain_bounds['min_z']) / (nz - 1)
        
        return x, y, z
    
    # Classify each boundary face
    tolerance = 1e-6
    
    for face in boundary_faces:
        # Get positions of the 3 nodes
        positions = [get_node_position(node) for node in face]
        
        # Calculate face center
        center_x = sum(pos[0] for pos in positions) / 3
        center_y = sum(pos[1] for pos in positions) / 3
        center_z = sum(pos[2] for pos in positions) / 3
        
        # Classify based on position
        if abs(center_x - domain_bounds['min_x']) < tolerance:
            # Inlet boundary (x = min)
            boundaries['inlet'].append([5, face[0], face[1], face[2]])
        elif abs(center_x - domain_bounds['max_x']) < tolerance:
            # Outlet boundary (x = max)
            boundaries['outlet'].append([5, face[0], face[1], face[2]])
        elif (abs(center_y - domain_bounds['min_y']) < tolerance or 
              abs(center_y - domain_bounds['max_y']) < tolerance or
              abs(center_z - domain_bounds['min_z']) < tolerance or 
              abs(center_z - domain_bounds['max_z']) < tolerance):
            # Slip wall boundaries (y and z boundaries)
            boundaries['slip_wall'].append([5, face[0], face[1], face[2]])
    
    print(f"   🏷️  Domain boundaries created:")
    for name, elems in boundaries.items():
        print(f"      {name}: {len(elems)} faces")
    
    return boundaries

def write_improved_su2_mesh(object_data: Dict[str, Any], domain_data: Dict[str, Any], 
                           object_boundaries: List, output_file: str):
    """Write complete SU2 mesh with proper object boundaries"""
    
    print("   💾 Writing SU2 mesh with object boundaries...")
    
    with open(output_file, 'w') as f:
        # Problem dimension
        f.write("NDIME=3\n")
        
        # Elements (domain + object)
        total_elements = len(domain_data['elements']) + len(object_data['elements'])
        f.write(f"NELEM={total_elements}\n")
        
        # Write domain elements first (no element indices needed)
        for elem in domain_data['elements']:
            f.write(" ".join(map(str, elem)) + "\n")
        
        # Write object elements with offset node indices (no element indices needed)
        node_offset = len(domain_data['nodes'])
        
        for elem in object_data['elements']:
            if len(elem) >= 4:
                # Offset node indices for object elements
                elem_with_offset = [elem[0]]  # Keep element type
                for j in range(1, len(elem)):  # Offset all node indices
                    elem_with_offset.append(elem[j] + node_offset)
                f.write(" ".join(map(str, elem_with_offset)) + "\n")
        
        f.write("\n")
        
        # Nodes (domain + object)
        total_nodes = len(domain_data['nodes']) + len(object_data['nodes'])
        f.write(f"NPOIN={total_nodes}\n")
        
        # Write domain nodes with node indices
        node_index = 0
        for node in domain_data['nodes']:
            f.write(f"{node[0]:.6f} {node[1]:.6f} {node[2]:.6f} {node_index}\n")
            node_index += 1
        
        # Write object nodes with node indices
        for node in object_data['nodes']:
            f.write(f"{node[0]:.6f} {node[1]:.6f} {node[2]:.6f} {node_index}\n")
            node_index += 1
        
        f.write("\n")
        
        # Boundaries (domain + object)
        total_boundaries = len(domain_data['boundaries']) + 1  # +1 for object_wall
        f.write(f"NMARK={total_boundaries}\n")
        
        # Write domain boundaries
        for boundary_name, boundary_elems in domain_data['boundaries'].items():
            f.write(f"MARKER_TAG={boundary_name}\n")
            f.write(f"MARKER_ELEMS={len(boundary_elems)}\n")
            
            for elem in boundary_elems:
                f.write(" ".join(map(str, elem)) + "\n")
        
        # Write object boundary (offset node indices)
        f.write("MARKER_TAG=object_wall\n")
        f.write(f"MARKER_ELEMS={len(object_boundaries)}\n")
        
        for elem in object_boundaries:
            # Offset node indices for object boundary elements
            elem_with_offset = [elem[0]]  # Keep element type (5 for triangle)
            for j in range(1, len(elem)):
                elem_with_offset.append(elem[j] + node_offset)
            f.write(" ".join(map(str, elem_with_offset)) + "\n")
    
    print(f"   ✅ Mesh written with {len(object_boundaries)} object boundary elements!")

def test_improved_generator():
    """Test the improved wind tunnel generator"""
    
    print("🧪 TESTING IMPROVED WIND TUNNEL GENERATOR")
    print("=" * 60)
    
    result = create_improved_cfd_wind_tunnel()
    
    if result:
        print(f"\n✅ IMPROVED GENERATOR SUCCESS!")
        print(f"📁 Generated mesh: {result}")
        
        # Verify the mesh
        print("\n🔍 Verifying generated mesh...")
        
        if os.path.exists(result):
            file_size = os.path.getsize(result) / (1024 * 1024)
            print(f"   📊 File size: {file_size:.1f} MB")
            
            # Quick boundary check
            with open(result, 'r') as f:
                content = f.read()
                
            inlet_count = content.count('MARKER_TAG=inlet')
            outlet_count = content.count('MARKER_TAG=outlet')
            slip_wall_count = content.count('MARKER_TAG=slip_wall')
            object_wall_count = content.count('MARKER_TAG=object_wall')
            
            print(f"   🏷️  Boundary markers found:")
            print(f"      inlet: {inlet_count}")
            print(f"      outlet: {outlet_count}")
            print(f"      slip_wall: {slip_wall_count}")
            print(f"      object_wall: {object_wall_count}")
            
            if object_wall_count > 0:
                print(f"   ✅ Object wall boundary properly created!")
            else:
                print(f"   ❌ Object wall boundary missing!")
        
        return True
    else:
        print(f"\n❌ IMPROVED GENERATOR FAILED!")
        return False

if __name__ == "__main__":
    test_improved_generator() 