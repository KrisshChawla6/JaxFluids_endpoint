#!/usr/bin/env python3
"""
Gmsh Direct Mesh Viewer
Uses Gmsh Python API to load and visualize the rocket engine mesh
"""

import os
import sys

try:
    import gmsh
    print(f"✅ Gmsh Python API available")
except ImportError:
    print("❌ Gmsh not available. Install with: pip install gmsh")
    exit()

def main():
    """Load and visualize mesh with Gmsh"""
    
    print("🚀 GMSH DIRECT MESH VIEWER")
    print("=" * 40)
    
    # Mesh file path
    mesh_file = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\mesh\Rocket Engine.msh"
    
    if not os.path.exists(mesh_file):
        print(f"❌ Mesh file not found: {mesh_file}")
        return
    
    print(f"📂 Loading: {mesh_file}")
    
    try:
        # Initialize Gmsh
        gmsh.initialize()
        
        # Load the mesh file
        print("🔄 Loading mesh with Gmsh...")
        gmsh.open(mesh_file)
        
        # Get model info
        print("📊 Mesh information:")
        
        # Get entities
        entities = gmsh.model.getEntities()
        print(f"   Total entities: {len(entities)}")
        
        for dim, tag in entities:
            print(f"   Entity: dim={dim}, tag={tag}")
        
        # Get physical groups if any
        physical_groups = gmsh.model.getPhysicalGroups()
        if physical_groups:
            print(f"   Physical groups: {len(physical_groups)}")
            for dim, tag in physical_groups:
                name = gmsh.model.getPhysicalName(dim, tag)
                print(f"     Group: dim={dim}, tag={tag}, name='{name}'")
        
        # Get mesh statistics
        nodes = gmsh.model.mesh.getNodes()
        print(f"   Nodes: {len(nodes[0]):,}")
        
        # Get elements by type
        element_types = gmsh.model.mesh.getElementTypes()
        print(f"   Element types: {len(element_types)}")
        
        for elem_type in element_types:
            elem_name = gmsh.model.mesh.getElementProperties(elem_type)[0]
            elements = gmsh.model.mesh.getElementsByType(elem_type)
            print(f"     {elem_name}: {len(elements[1]):,} elements")
        
        # Get boundary elements (faces) for surface visualization
        print("\n🔍 Analyzing boundary faces...")
        
        # Get all 2D entities (surfaces)
        surfaces = [e for e in entities if e[0] == 2]
        print(f"   2D surfaces: {len(surfaces)}")
        
        # Get boundary faces from 3D elements
        volume_entities = [e for e in entities if e[0] == 3]
        if volume_entities:
            print(f"   3D volumes: {len(volume_entities)}")
            
            # For each volume, get its boundary faces
            total_boundary_faces = 0
            for dim, tag in volume_entities:
                boundary = gmsh.model.getBoundary([(dim, tag)], oriented=False)
                boundary_faces = [e for e in boundary if e[0] == 2]
                total_boundary_faces += len(boundary_faces)
                print(f"     Volume {tag}: {len(boundary_faces)} boundary faces")
            
            print(f"   Total boundary faces: {total_boundary_faces}")
        
        print("\n🎮 Launching Gmsh GUI...")
        print("   Use Gmsh interface to:")
        print("   • Rotate: Left mouse drag")
        print("   • Zoom: Mouse wheel or right mouse drag")
        print("   • Pan: Shift + left mouse drag")
        print("   • View options: Right-click menu")
        print("   • Close: File > Exit or close window")
        
        # Launch the GUI
        if '-nopopup' in sys.argv:
            print("   Running without GUI (nopopup mode)")
        else:
            gmsh.fltk.run()
        
        print("✅ Gmsh viewer closed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        gmsh.finalize()

if __name__ == "__main__":
    main() 