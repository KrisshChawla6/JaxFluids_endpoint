#!/usr/bin/env python3
"""
Generate CFD-Ready Wind Tunnel Mesh with Proper Dimensions
"""

import os
import sys
import time
import numpy as np

def create_cfd_wind_tunnel():
    """
    Create a CFD-ready wind tunnel mesh with proper dimensions and fine mesh
    """
    
    print("🔧 CFD-Ready Wind Tunnel Generator")
    print("=" * 50)
    
    start_time = time.time()
    
    # Input propeller mesh file
    propeller_file = r"C:\Users\kriss\Desktop\New Simulation\projects\propeller\mesh\5_bladed_Propeller_medium_tetrahedral.su2"
    output_file = "output/propeller_wind_tunnel_cfd.su2"
    
    if not os.path.exists(propeller_file):
        print(f"❌ Propeller mesh not found: {propeller_file}")
        return None
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Step 1: Read propeller mesh
        print("📖 Reading 5-bladed propeller mesh...")
        object_data = read_airfoil_mesh(propeller_file)
        
        # Step 2: Create CFD domain
        print("🏗️  Creating CFD-ready domain...")
        domain_data = create_cfd_domain(object_data)
        
        # Step 3: Write SU2 mesh
        print("💾 Writing CFD SU2 mesh...")
        write_cfd_su2_mesh(object_data, domain_data, output_file)
        
        generation_time = time.time() - start_time
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        
        print(f"🎉 CFD wind tunnel generated!")
        print(f"   📁 Output: {output_file}")
        print(f"   ⏱️  Time: {generation_time:.2f}s")
        print(f"   📊 Size: {file_size:.1f} MB")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def read_airfoil_mesh(mesh_file: str):
    """Read propeller mesh"""
    
    nodes = []
    elements = []
    
    with open(mesh_file, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    reading_elements = False
    reading_nodes = False
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('NELEM='):
            reading_elements = True
            reading_nodes = False
            continue
        elif line.startswith('NPOIN='):
            reading_elements = False
            reading_nodes = True
            continue
        elif line.startswith('NMARK='):
            break
        elif reading_elements and line:
            parts = line.split()
            if len(parts) > 1:
                elements.append([int(x) for x in parts])
        elif reading_nodes and line:
            parts = line.split()
            if len(parts) >= 4:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                nodes.append([x, y, z])
    
    nodes_array = np.array(nodes)
    
    # Calculate airfoil bounding box
    bbox = {
        'min_x': nodes_array[:, 0].min(),
        'max_x': nodes_array[:, 0].max(),
        'min_y': nodes_array[:, 1].min(),
        'max_y': nodes_array[:, 1].max(),
        'min_z': nodes_array[:, 2].min(),
        'max_z': nodes_array[:, 2].max(),
    }
    
    chord_length = bbox['max_x'] - bbox['min_x']
    
    print(f"   📊 Propeller: {len(nodes)} nodes, diameter: {chord_length:.3f}")
    
    return {
        'nodes': nodes_array,
        'elements': elements,
        'bbox': bbox,
        'chord_length': chord_length
    }

def create_cfd_domain(object_data):
    """Create CFD-ready domain with proper proportions"""
    
    bbox = object_data['bbox']
    chord = object_data['chord_length']
    
    # CFD-appropriate domain sizing (based on chord lengths) - Compact design
    upstream_length = 1.5 * chord    # 1.5 chord lengths upstream
    downstream_length = 2.5 * chord  # 2.5 chord lengths downstream  
    domain_height = 2 * chord        # 2 chord lengths total height
    domain_width = 2 * chord         # 2 chord lengths total width
    
    # Center the airfoil
    center_y = (bbox['min_y'] + bbox['max_y']) / 2
    center_z = (bbox['min_z'] + bbox['max_z']) / 2
    
    # Domain bounds (much smaller and more reasonable!)
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
    print(f"      Length: {total_length:.1f} ({total_length/chord:.0f} chords)")
    print(f"      Height: {total_height:.1f} ({total_height/chord:.0f} chords)")
    print(f"      Width: {total_width:.1f} ({total_width/chord:.0f} chords)")
    
    # Create fine mesh for CFD
    nx = 50  # 50 points in flow direction
    ny = 30  # 30 points in height
    nz = 25  # 25 points in width
    
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
    
    # Generate tetrahedra instead of hexahedra for better CFD
    domain_elements = []
    elem_id = 0
    
    # Convert hex grid to tetrahedra (6 tetrahedra per hex)
    for k in range(nz-1):
        for j in range(ny-1):
            for i in range(nx-1):
                # Get 8 corner nodes of the hex
                n0 = k * nx * ny + j * nx + i
                n1 = k * nx * ny + j * nx + (i + 1)
                n2 = k * nx * ny + (j + 1) * nx + (i + 1)
                n3 = k * nx * ny + (j + 1) * nx + i
                n4 = (k + 1) * nx * ny + j * nx + i
                n5 = (k + 1) * nx * ny + j * nx + (i + 1)
                n6 = (k + 1) * nx * ny + (j + 1) * nx + (i + 1)
                n7 = (k + 1) * nx * ny + (j + 1) * nx + i
                
                # Split hex into 6 tetrahedra
                tetrahedra = [
                    [10, n0, n1, n3, n4, elem_id],
                    [10, n1, n2, n3, n6, elem_id+1],
                    [10, n1, n3, n4, n6, elem_id+2],
                    [10, n3, n4, n6, n7, elem_id+3],
                    [10, n1, n4, n5, n6, elem_id+4],
                    [10, n4, n5, n6, n1, elem_id+5]
                ]
                
                domain_elements.extend(tetrahedra)
                elem_id += 6
    
    # Create boundary elements
    boundaries = create_cfd_boundaries(nx, ny, nz)
    
    return {
        'nodes': domain_nodes,
        'elements': domain_elements,  
        'boundaries': boundaries,
        'bounds': domain_bounds,
        'mesh_size': (nx, ny, nz)
    }

def create_cfd_boundaries(nx, ny, nz):
    """Create properly labeled CFD boundaries"""
    
    boundaries = {
        'inlet': [],        # X = min (upstream) - RED
        'outlet': [],       # X = max (downstream) - BLUE
        'slip_wall': [],    # Y and Z boundaries - GREEN
        'object_wall': []   # Airfoil surface - YELLOW
    }
    
    # Inlet face (X = min, i = 0) - RED
    for k in range(nz-1):
        for j in range(ny-1):
            n0 = k * nx * ny + j * nx + 0
            n1 = k * nx * ny + (j + 1) * nx + 0
            n2 = (k + 1) * nx * ny + (j + 1) * nx + 0
            n3 = (k + 1) * nx * ny + j * nx + 0
            boundaries['inlet'].append([5, n0, n1, n2, n3])
    
    # Outlet face (X = max, i = nx-1) - BLUE
    for k in range(nz-1):
        for j in range(ny-1):
            n0 = k * nx * ny + j * nx + (nx-1)
            n1 = (k + 1) * nx * ny + j * nx + (nx-1)
            n2 = (k + 1) * nx * ny + (j + 1) * nx + (nx-1)
            n3 = k * nx * ny + (j + 1) * nx + (nx-1)
            boundaries['outlet'].append([5, n0, n1, n2, n3])
    
    # All other walls as slip_wall - GREEN
    # Top wall (Y = max)
    for k in range(nz-1):
        for i in range(nx-1):
            n0 = k * nx * ny + (ny-1) * nx + i
            n1 = k * nx * ny + (ny-1) * nx + (i + 1)
            n2 = (k + 1) * nx * ny + (ny-1) * nx + (i + 1)
            n3 = (k + 1) * nx * ny + (ny-1) * nx + i
            boundaries['slip_wall'].append([5, n0, n1, n2, n3])
    
    # Bottom wall (Y = min)
    for k in range(nz-1):
        for i in range(nx-1):
            n0 = k * nx * ny + 0 * nx + i
            n1 = (k + 1) * nx * ny + 0 * nx + i
            n2 = (k + 1) * nx * ny + 0 * nx + (i + 1)
            n3 = k * nx * ny + 0 * nx + (i + 1)
            boundaries['slip_wall'].append([5, n0, n1, n2, n3])
    
    # Side walls (Z = min and Z = max)
    for j in range(ny-1):
        for i in range(nx-1):
            # Z = min
            n0 = 0 * nx * ny + j * nx + i
            n1 = 0 * nx * ny + j * nx + (i + 1)
            n2 = 0 * nx * ny + (j + 1) * nx + (i + 1)
            n3 = 0 * nx * ny + (j + 1) * nx + i
            boundaries['slip_wall'].append([5, n0, n1, n2, n3])
            
            # Z = max
            n0 = (nz-1) * nx * ny + j * nx + i
            n1 = (nz-1) * nx * ny + (j + 1) * nx + i
            n2 = (nz-1) * nx * ny + (j + 1) * nx + (i + 1)
            n3 = (nz-1) * nx * ny + j * nx + (i + 1)
            boundaries['slip_wall'].append([5, n0, n1, n2, n3])
    
    print(f"   🏷️  Created boundaries:")
    for name, elems in boundaries.items():
        print(f"      {name}: {len(elems)} faces")
    
    return boundaries

def write_cfd_su2_mesh(object_data, domain_data, output_file: str):
    """Write CFD SU2 mesh with proper boundary labels"""
    
    with open(output_file, 'w') as f:
        # Problem dimension
        f.write("NDIME=3\n\n")
        
        # Elements (domain + object)
        total_elements = len(domain_data['elements']) + len(object_data['elements'])
        f.write(f"NELEM={total_elements}\n")
        
        # Write domain elements
        for elem in domain_data['elements']:
            f.write(" ".join(map(str, elem)) + "\n")
        
        # Write object elements (offset node indices)
        node_offset = len(domain_data['nodes'])
        for i, elem in enumerate(object_data['elements']):
            if len(elem) >= 4:
                elem_with_offset = [elem[0]]
                for j in range(1, len(elem) - 1):
                    elem_with_offset.append(elem[j] + node_offset)
                elem_with_offset.append(len(domain_data['elements']) + i)
                f.write(" ".join(map(str, elem_with_offset)) + "\n")
        
        f.write("\n")
        
        # Nodes (domain + object)
        total_nodes = len(domain_data['nodes']) + len(object_data['nodes'])
        f.write(f"NPOIN={total_nodes}\n")
        
        # Write domain nodes
        for i, node in enumerate(domain_data['nodes']):
            f.write(f"{node[0]:.6f} {node[1]:.6f} {node[2]:.6f} {i}\n")
        
        # Write object nodes
        for i, node in enumerate(object_data['nodes']):
            f.write(f"{node[0]:.6f} {node[1]:.6f} {node[2]:.6f} {i + len(domain_data['nodes'])}\n")
        
        f.write("\n")
        
        # Boundaries
        f.write(f"NMARK={len(domain_data['boundaries'])}\n")
        
        for boundary_name, boundary_elems in domain_data['boundaries'].items():
            f.write(f"MARKER_TAG={boundary_name}\n")
            f.write(f"MARKER_ELEMS={len(boundary_elems)}\n")
            
            for elem in boundary_elems:
                f.write(" ".join(map(str, elem)) + "\n")
            
            f.write("\n")

if __name__ == "__main__":
    result = create_cfd_wind_tunnel()
    
    if result:
        print(f"\n✅ CFD Wind Tunnel Generation Complete!")
        print(f"📁 Ready for SU2 CFD simulation: {result}")
    else:
        print(f"\n💥 CFD Wind Tunnel Generation Failed!") 