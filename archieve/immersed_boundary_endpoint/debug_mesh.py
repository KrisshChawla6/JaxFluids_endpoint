#!/usr/bin/env python3
"""
Debug script to analyze the mesh file structure.
"""

import numpy as np
from pathlib import Path

def analyze_mesh_file():
    """Analyze the structure of the mesh file."""
    
    mesh_file = Path("../mesh/5_bladed_Propeller.STEP_medium_tetrahedral.msh")
    
    print("Analyzing mesh file structure...")
    print(f"File: {mesh_file}")
    print("="*60)
    
    with open(mesh_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Total lines: {len(lines)}")
    
    # Find all sections
    sections = []
    for i, line in enumerate(lines):
        if line.strip().startswith('$'):
            sections.append((i+1, line.strip()))
    
    print(f"\nSections found:")
    for line_num, section in sections:
        print(f"  Line {line_num}: {section}")
    
    # Look for nodes and elements sections
    in_nodes = False
    in_elements = False
    nodes_count = 0
    elements_count = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        if line == '$Nodes':
            in_nodes = True
            print(f"\nNodes section starts at line {i+1}")
            # Next line should contain node info
            if i+1 < len(lines):
                node_info = lines[i+1].strip()
                print(f"Node info line: {node_info}")
            continue
        elif line == '$EndNodes':
            in_nodes = False
            print(f"Nodes section ends at line {i+1}")
            continue
        elif line == '$Elements':
            in_elements = True
            print(f"\nElements section starts at line {i+1}")
            # Next line should contain element info
            if i+1 < len(lines):
                elem_info = lines[i+1].strip()
                print(f"Element info line: {elem_info}")
            continue
        elif line == '$EndElements':
            in_elements = False
            print(f"Elements section ends at line {i+1}")
            continue
        
        if in_nodes and line and not line.startswith('$'):
            nodes_count += 1
            if nodes_count <= 5:  # Show first few nodes
                print(f"  Node line {i+1}: {line}")
        
        if in_elements and line and not line.startswith('$'):
            elements_count += 1
            if elements_count <= 5:  # Show first few elements
                print(f"  Element line {i+1}: {line}")
    
    print(f"\nTotal node lines found: {nodes_count}")
    print(f"Total element lines found: {elements_count}")
    
    # Check if this is an old format mesh
    if '$Nodes' not in [s[1] for s in sections]:
        print("\nWARNING: This appears to be an old format mesh file!")
        print("Looking for alternative structure...")
        
        # Look for number patterns that might be coordinates
        coord_lines = []
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    # Try to parse as floats
                    coords = [float(x) for x in parts[:3]]
                    coord_lines.append((i+1, coords))
                    if len(coord_lines) <= 10:
                        print(f"  Potential coordinate line {i+1}: {coords}")
                except ValueError:
                    pass
        
        print(f"\nFound {len(coord_lines)} potential coordinate lines")
        
        # Look for integer patterns that might be elements
        elem_lines = []
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    # Try to parse as integers
                    if all(x.isdigit() for x in parts):
                        elems = [int(x) for x in parts]
                        elem_lines.append((i+1, elems))
                        if len(elem_lines) <= 10:
                            print(f"  Potential element line {i+1}: {elems}")
                except ValueError:
                    pass
        
        print(f"\nFound {len(elem_lines)} potential element lines")

if __name__ == "__main__":
    analyze_mesh_file() 