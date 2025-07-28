#!/usr/bin/env python3
"""
Convert SU2 mesh to VTK format for visualization
"""

import numpy as np
import os
import sys
from pathlib import Path

def su2_to_vtk(su2_file: str, vtk_file: str = None):
    """
    Convert SU2 mesh file to VTK format
    """
    
    if vtk_file is None:
        vtk_file = su2_file.replace('.su2', '.vtk')
    
    print(f"🔄 Converting SU2 to VTK...")
    print(f"   📖 Input: {su2_file}")
    print(f"   📝 Output: {vtk_file}")
    
    try:
        # Read SU2 mesh
        nodes, elements, boundaries = read_su2_mesh(su2_file)
        
        # Write VTK mesh
        write_vtk_mesh(nodes, elements, boundaries, vtk_file)
        
        file_size = os.path.getsize(vtk_file) / (1024 * 1024)
        print(f"✅ Conversion successful!")
        print(f"   📊 {len(nodes)} nodes, {len(elements)} elements")
        print(f"   📁 Output: {vtk_file} ({file_size:.1f} MB)")
        
        return vtk_file
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return None

def read_su2_mesh(su2_file: str):
    """Read SU2 mesh file"""
    
    nodes = []
    elements = []
    boundaries = {}
    
    with open(su2_file, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('NELEM='):
            # Read elements
            n_elem = int(line.split('=')[1])
            i += 1
            for _ in range(n_elem):
                if i < len(lines):
                    parts = lines[i].strip().split()
                    if len(parts) > 1:
                        elements.append([int(x) for x in parts])
                    i += 1
                    
        elif line.startswith('NPOIN='):
            # Read nodes
            n_points = int(line.split('=')[1])
            i += 1
            for _ in range(n_points):
                if i < len(lines):
                    parts = lines[i].strip().split()
                    if len(parts) >= 4:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        nodes.append([x, y, z])
                    i += 1
                    
        elif line.startswith('NMARK='):
            # Read boundaries
            n_markers = int(line.split('=')[1])
            i += 1
            
            for _ in range(n_markers):
                # Read marker tag
                while i < len(lines) and not lines[i].strip().startswith('MARKER_TAG='):
                    i += 1
                if i >= len(lines):
                    break
                    
                marker_name = lines[i].strip().split('=')[1]
                i += 1
                
                # Read marker elements
                while i < len(lines) and not lines[i].strip().startswith('MARKER_ELEMS='):
                    i += 1
                if i >= len(lines):
                    break
                    
                n_marker_elems = int(lines[i].strip().split('=')[1])
                i += 1
                
                marker_elements = []
                for _ in range(n_marker_elems):
                    if i < len(lines):
                        parts = lines[i].strip().split()
                        if len(parts) > 1:
                            marker_elements.append([int(x) for x in parts])
                        i += 1
                
                boundaries[marker_name] = marker_elements
        else:
            i += 1
    
    print(f"   📊 Read: {len(nodes)} nodes, {len(elements)} elements, {len(boundaries)} boundaries")
    return np.array(nodes), elements, boundaries

def write_vtk_mesh(nodes, elements, boundaries, vtk_file: str):
    """Write VTK mesh file"""
    
    with open(vtk_file, 'w') as f:
        # VTK header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("SU2 Wind Tunnel Mesh\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n\n")
        
        # Points
        f.write(f"POINTS {len(nodes)} float\n")
        for node in nodes:
            f.write(f"{node[0]:.6f} {node[1]:.6f} {node[2]:.6f}\n")
        f.write("\n")
        
        # Cells - only write valid elements
        valid_elements = []
        for elem in elements:
            if len(elem) >= 4:  # At least element type + 3 nodes
                valid_elements.append(elem)
        
        if valid_elements:
            # Calculate total size for CELLS section
            total_size = sum(len(elem) for elem in valid_elements)
            
            f.write(f"CELLS {len(valid_elements)} {total_size}\n")
            for elem in valid_elements:
                # VTK format: number_of_points point1 point2 ... pointN
                if len(elem) >= 4:
                    n_points = len(elem) - 2  # Subtract element type and element ID
                    f.write(f"{n_points}")
                    for j in range(1, len(elem) - 1):  # Skip element type and element ID
                        f.write(f" {elem[j]}")
                    f.write("\n")
            f.write("\n")
            
            # Cell types
            f.write(f"CELL_TYPES {len(valid_elements)}\n")
            for elem in valid_elements:
                if len(elem) >= 4:
                    elem_type = elem[0]
                    # Convert SU2 element types to VTK
                    vtk_type = su2_to_vtk_element_type(elem_type)
                    f.write(f"{vtk_type}\n")
            f.write("\n")
        
        # Point data for boundaries
        if boundaries:
            f.write(f"POINT_DATA {len(nodes)}\n")
            f.write("SCALARS boundary_id int 1\n")
            f.write("LOOKUP_TABLE default\n")
            
            # Initialize all points as interior (0)
            point_boundary = np.zeros(len(nodes), dtype=int)
            
            # Mark boundary points
            boundary_id = 1
            for boundary_name, boundary_elements in boundaries.items():
                for elem in boundary_elements:
                    if len(elem) > 1:
                        for node_id in elem[1:]:  # Skip element type
                            if 0 <= node_id < len(nodes):
                                point_boundary[node_id] = boundary_id
                boundary_id += 1
            
            for bid in point_boundary:
                f.write(f"{bid}\n")

def su2_to_vtk_element_type(su2_type: int) -> int:
    """Convert SU2 element type to VTK element type"""
    
    # SU2 to VTK element type mapping
    mapping = {
        3: 5,   # Triangle -> VTK_TRIANGLE
        5: 9,   # Quadrilateral -> VTK_QUAD  
        10: 10, # Tetrahedron -> VTK_TETRA
        12: 12, # Hexahedron -> VTK_HEXAHEDRON
        13: 13, # Prism -> VTK_WEDGE
        14: 14, # Pyramid -> VTK_PYRAMID
    }
    
    return mapping.get(su2_type, 1)  # Default to VTK_VERTEX if unknown

def main():
    """Convert Eppler wind tunnel mesh to VTK"""
    
    # Path to our generated mesh
    su2_file = "../WindTunnel/output/eppler_wind_tunnel_complete.su2"
    vtk_file = "eppler_wind_tunnel_complete.vtk"
    
    if not os.path.exists(su2_file):
        print(f"❌ SU2 mesh not found at: {su2_file}")
        return False
    
    # Convert to VTK
    result = su2_to_vtk(su2_file, vtk_file)
    
    if result:
        print(f"\n🎉 VTK conversion successful!")
        print(f"📁 VTK file: {result}")
        return result
    else:
        print(f"\n❌ VTK conversion failed!")
        return False

if __name__ == "__main__":
    vtk_file = main()
    
    if vtk_file:
        print(f"\n✅ Ready for visualization!")
        print(f"📁 VTK file: {vtk_file}")
        
        # Also generate HTML visualization
        html_file = vtk_file.replace('.vtk', '_visualization.html')
        print(f"\n🌐 Generating HTML visualization...")
        
        try:
            import subprocess
            result = subprocess.run([
                'python', 'vtk_viewer.py', vtk_file, html_file, 
                'Eppler Wind Tunnel - Complete SU2 Mesh'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ HTML visualization created: {html_file}")
            else:
                print(f"❌ HTML generation failed: {result.stderr}")
        except Exception as e:
            print(f"❌ HTML generation error: {e}")
    else:
        print(f"\n💥 Conversion failed!") 