#!/usr/bin/env python3
"""
Simple SU2 to VTK Converter - Clean and reliable
Fixed to handle tetrahedral elements (type 10)
"""

import os
import sys

def simple_su2_to_vtk():
    """Simple SU2 to VTK conversion"""
    
    su2_file = "WindTunnel/output/eppler_wind_tunnel_complete.su2"
    vtk_file = "eppler_wind_tunnel_fixed.vtk"
    
    print("🎯 FIXED SU2 TO VTK CONVERTER")
    print("=" * 35)
    print("Now handles tetrahedral elements!")
    
    if not os.path.exists(su2_file):
        print(f"❌ SU2 file not found: {su2_file}")
        return False
    
    file_size = os.path.getsize(su2_file) / (1024 * 1024)
    print(f"📁 Input: {su2_file} ({file_size:.1f} MB)")
    print(f"📁 Output: {vtk_file}")
    
    try:
        print("📖 Reading SU2 file...")
        
        # Read SU2 file
        with open(su2_file, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Parse nodes
        nodes = []
        elements = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('NPOIN='):
                n_points = int(line.split('=')[1])
                print(f"   📊 Reading {n_points:,} nodes...")
                i += 1
                
                for _ in range(n_points):
                    if i < len(lines):
                        parts = lines[i].strip().split()
                        if len(parts) >= 4:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            nodes.append([x, y, z])
                        i += 1
                        
            elif line.startswith('NELEM='):
                n_elem = int(line.split('=')[1])
                print(f"   📊 Reading {n_elem:,} elements...")
                i += 1
                
                for _ in range(n_elem):
                    if i < len(lines):
                        parts = lines[i].strip().split()
                        if len(parts) > 1:
                            elements.append([int(x) for x in parts])
                        i += 1
            else:
                i += 1
        
        print(f"✅ Parsed: {len(nodes):,} nodes, {len(elements):,} elements")
        
        # Analyze element types
        element_types = {}
        for elem in elements:
            if len(elem) > 0:
                elem_type = elem[0]
                element_types[elem_type] = element_types.get(elem_type, 0) + 1
        
        print(f"📊 Element types found:")
        for elem_type, count in element_types.items():
            type_name = {5: "Triangle", 10: "Tetrahedron", 12: "Hexahedron"}.get(elem_type, f"Type-{elem_type}")
            print(f"   - {type_name} (type {elem_type}): {count:,}")
        
        # Write VTK file
        print("📝 Writing VTK file...")
        
        with open(vtk_file, 'w') as f:
            # VTK header - CLEAN formatting
            f.write("# vtk DataFile Version 3.0\n")
            f.write("SU2 Wind Tunnel Mesh - Fixed\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            f.write("\n")
            
            # Points
            f.write(f"POINTS {len(nodes)} float\n")
            for node in nodes:
                f.write(f"{node[0]:.6f} {node[1]:.6f} {node[2]:.6f}\n")
            f.write("\n")
            
            # Handle tetrahedral elements (type 10) - the main element type
            valid_elements = []
            for elem in elements:
                if len(elem) >= 5 and elem[0] == 10:  # Tetrahedral element type
                    valid_elements.append(elem)
            
            if valid_elements:
                print(f"   📊 Writing {len(valid_elements):,} tetrahedral elements...")
                
                # Calculate total size
                total_size = len(valid_elements) * 5  # 4 points + count per tetrahedron
                
                f.write(f"CELLS {len(valid_elements)} {total_size}\n")
                for elem in valid_elements:
                    if len(elem) >= 5:
                        # VTK format: 4 point1 point2 point3 point4
                        f.write(f"4 {elem[1]} {elem[2]} {elem[3]} {elem[4]}\n")
                f.write("\n")
                
                # Cell types (all tetrahedra = VTK type 10)
                f.write(f"CELL_TYPES {len(valid_elements)}\n")
                for _ in valid_elements:
                    f.write("10\n")
                f.write("\n")
            else:
                print(f"   ⚠️  No tetrahedral elements found!")
                return False
        
        # Check output
        if os.path.exists(vtk_file):
            output_size = os.path.getsize(vtk_file) / (1024 * 1024)
            print(f"✅ VTK file created: {vtk_file} ({output_size:.1f} MB)")
            
            # Verify header
            with open(vtk_file, 'r') as f:
                header = f.read(300)
                print(f"📋 VTK header preview:")
                for line in header.split('\n')[:8]:
                    print(f"   {line}")
            
            return vtk_file
        else:
            print(f"❌ VTK file creation failed!")
            return False
            
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return False

if __name__ == "__main__":
    result = simple_su2_to_vtk()
    
    if result:
        print(f"\n🎉 Fixed SU2 to VTK conversion successful!")
        print(f"📁 Fixed VTK file: {result}")
    else:
        print(f"\n💥 Fixed SU2 to VTK conversion failed!")
        
    input("Press Enter to exit...") 