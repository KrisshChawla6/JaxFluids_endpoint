#!/usr/bin/env python3
"""
Gmsh Terminal Mesh Analysis
Terminal-only analysis of the rocket engine mesh using Gmsh API
"""

import os
import sys

try:
    import gmsh
    print(f"✅ Gmsh Python API available")
except ImportError:
    print("❌ Gmsh not available. Install with: pip install gmsh")
    exit()

def analyze_mesh(mesh_file):
    """Analyze mesh structure in detail"""
    
    print(f"🔍 DETAILED MESH ANALYSIS")
    print("=" * 50)
    
    try:
        # Initialize Gmsh
        gmsh.initialize()
        
        # Load the mesh file
        print(f"📂 Loading: {os.path.basename(mesh_file)}")
        gmsh.open(mesh_file)
        
        # Basic model info
        print(f"\n📊 MODEL INFORMATION:")
        model_name = gmsh.model.getCurrent()
        print(f"   Model name: {model_name}")
        
        # Get all entities
        entities = gmsh.model.getEntities()
        print(f"   Total entities: {len(entities)}")
        
        # Categorize entities by dimension
        entities_by_dim = {}
        for dim, tag in entities:
            if dim not in entities_by_dim:
                entities_by_dim[dim] = []
            entities_by_dim[dim].append(tag)
        
        for dim in sorted(entities_by_dim.keys()):
            dim_name = ["Points", "Curves", "Surfaces", "Volumes"][dim]
            print(f"   {dim_name} (dim {dim}): {len(entities_by_dim[dim])} entities")
        
        # Physical groups
        physical_groups = gmsh.model.getPhysicalGroups()
        if physical_groups:
            print(f"\n🏷️  PHYSICAL GROUPS:")
            for dim, tag in physical_groups:
                name = gmsh.model.getPhysicalName(dim, tag)
                entities_in_group = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
                print(f"   Group '{name}': dim={dim}, tag={tag}, entities={len(entities_in_group)}")
        else:
            print(f"\n🏷️  PHYSICAL GROUPS: None defined")
        
        # Mesh statistics
        print(f"\n🔢 MESH STATISTICS:")
        
        # Nodes
        nodes = gmsh.model.mesh.getNodes()
        node_count = len(nodes[0])
        print(f"   Nodes: {node_count:,}")
        
        # Elements by type
        element_types = gmsh.model.mesh.getElementTypes()
        print(f"   Element types: {len(element_types)}")
        
        total_elements = 0
        for elem_type in element_types:
            elem_name = gmsh.model.mesh.getElementProperties(elem_type)[0]
            elements = gmsh.model.mesh.getElementsByType(elem_type)
            elem_count = len(elements[1])
            total_elements += elem_count
            print(f"     {elem_name}: {elem_count:,}")
        
        print(f"   Total elements: {total_elements:,}")
        
        # Boundary analysis for 3D meshes
        if 3 in entities_by_dim:
            print(f"\n🔍 BOUNDARY ANALYSIS:")
            
            # Get boundary faces from volume entities
            volume_entities = [(3, tag) for tag in entities_by_dim[3]]
            all_boundary_faces = []
            
            for vol_entity in volume_entities:
                boundary = gmsh.model.getBoundary([vol_entity], oriented=False, recursive=False)
                boundary_faces = [e for e in boundary if e[0] == 2]
                all_boundary_faces.extend(boundary_faces)
                print(f"   Volume {vol_entity[1]}: {len(boundary_faces)} boundary faces")
            
            print(f"   Total boundary faces: {len(all_boundary_faces)}")
            
            # Get unique boundary faces (some might be shared)
            unique_faces = list(set(all_boundary_faces))
            print(f"   Unique boundary faces: {len(unique_faces)}")
            
            # Analyze face positions to identify potential inlet/outlet
            print(f"\n🎯 POTENTIAL INLET/OUTLET ANALYSIS:")
            
            face_positions = {}
            for face_dim, face_tag in unique_faces:
                # Get nodes of this face
                face_nodes = gmsh.model.mesh.getNodes(face_dim, face_tag)
                if len(face_nodes[1]) > 0:
                    # Get coordinates
                    coords = face_nodes[1].reshape(-1, 3)
                    # Calculate centroid
                    centroid = coords.mean(axis=0)
                    face_positions[face_tag] = centroid
            
            if face_positions:
                # Find extremes in X direction (flow direction)
                x_coords = [pos[0] for pos in face_positions.values()]
                min_x, max_x = min(x_coords), max(x_coords)
                
                print(f"   X-coordinate range: {min_x:.1f} to {max_x:.1f}")
                
                # Find faces at extremes
                tolerance = (max_x - min_x) * 0.01  # 1% tolerance
                
                inlet_faces = [tag for tag, pos in face_positions.items() 
                              if abs(pos[0] - min_x) < tolerance]
                outlet_faces = [tag for tag, pos in face_positions.items() 
                               if abs(pos[0] - max_x) < tolerance]
                
                print(f"   Potential inlet faces (min X): {len(inlet_faces)}")
                print(f"   Potential outlet faces (max X): {len(outlet_faces)}")
                
                # Calculate areas of potential inlet/outlet faces
                if inlet_faces:
                    inlet_area = 0
                    for face_tag in inlet_faces:
                        # Get face area (approximate)
                        face_nodes = gmsh.model.mesh.getNodes(2, face_tag)
                        if len(face_nodes[1]) > 0:
                            coords = face_nodes[1].reshape(-1, 3)
                            # Simple area estimation
                            inlet_area += len(coords) * 0.1  # Rough estimate
                    print(f"   Estimated inlet area: {inlet_area:.1f}")
                
                if outlet_faces:
                    outlet_area = 0
                    for face_tag in outlet_faces:
                        face_nodes = gmsh.model.mesh.getNodes(2, face_tag)
                        if len(face_nodes[1]) > 0:
                            coords = face_nodes[1].reshape(-1, 3)
                            outlet_area += len(coords) * 0.1  # Rough estimate
                    print(f"   Estimated outlet area: {outlet_area:.1f}")
        
        # Mesh quality info
        print(f"\n📏 MESH BOUNDS:")
        if node_count > 0:
            all_coords = nodes[1].reshape(-1, 3)
            bounds = {
                'X': (all_coords[:, 0].min(), all_coords[:, 0].max()),
                'Y': (all_coords[:, 1].min(), all_coords[:, 1].max()),
                'Z': (all_coords[:, 2].min(), all_coords[:, 2].max())
            }
            
            for axis, (min_val, max_val) in bounds.items():
                print(f"   {axis}: {min_val:.1f} to {max_val:.1f} (range: {max_val-min_val:.1f})")
        
        print(f"\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        gmsh.finalize()

def main():
    """Main function"""
    
    print("🚀 GMSH TERMINAL MESH ANALYZER")
    print("=" * 50)
    
    # Mesh file path
    mesh_file = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\mesh\Rocket Engine.msh"
    
    if not os.path.exists(mesh_file):
        print(f"❌ Mesh file not found: {mesh_file}")
        return
    
    analyze_mesh(mesh_file)

if __name__ == "__main__":
    main() 