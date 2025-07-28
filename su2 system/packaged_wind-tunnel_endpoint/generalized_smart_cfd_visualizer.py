#!/usr/bin/env python3
"""
Generalized Smart CFD Visualizer
Efficiently visualizes any wind tunnel VTK file by automatically detecting object vs domain regions
and rendering them appropriately without creating sub-VTKs
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse

def generalized_smart_visualization(su2_file: str, 
                                  flow_direction: str = "+X",
                                  object_color: str = "gold",
                                  domain_color: str = "lightblue",
                                  show_boundaries: bool = True,
                                  point_size: int = 3,
                                  domain_sampling_ratio: float = 0.3) -> bool:
    """
    Create smart CFD visualization from any SU2 wind tunnel file
    
    This function works exactly like the original smart_cfd_visualizer but generalized:
    1. Takes a SU2 file (full CFD domain)
    2. Extracts object elements to create temporary object.vtk
    3. Creates domain point cloud from all SU2 nodes
    4. Visualizes object as detailed mesh + domain as point cloud
    
    Args:
        su2_file: Path to SU2 wind tunnel file
        flow_direction: Flow direction (+X, -X, +Y, -Y, +Z, -Z)
        object_color: Color for the object mesh
        domain_color: Color for the domain point cloud
        show_boundaries: Whether to show boundary planes
        point_size: Size of domain points
        domain_sampling_ratio: Ratio of domain points to sample (0.1-1.0)
        
    Returns:
        Success status
    """
    
    print("🎯 GENERALIZED SMART CFD VISUALIZER")
    print("=" * 50)
    print(f"📁 SU2 file: {su2_file}")
    print(f"➡️ Flow direction: {flow_direction}")
    
    if not os.path.exists(su2_file):
        print(f"❌ SU2 file not found: {su2_file}")
        return False
    
    try:
        print("\n🔍 STEP 1: Analyzing CFD domain...")
        domain_info = analyze_cfd_domain(su2_file)
        
        print("\n🔍 STEP 2: Creating domain point cloud...")
        point_cloud = create_smart_point_cloud(domain_info, domain_sampling_ratio)
        
        print("\n🔍 STEP 3: Extracting object from SU2...")
        object_vtk_file = extract_object_from_su2(su2_file)
        
        print("\n🔍 STEP 4: Creating smart visualization...")
        success = create_generalized_smart_viz(
            object_vtk_file, point_cloud, domain_info, flow_direction,
            object_color, domain_color, show_boundaries, point_size
        )
        
        # Clean up temporary file
        cleanup_temp_files({'object_vtk_file': object_vtk_file})
        
        return success
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False



def analyze_cfd_domain(su2_file: str) -> Dict[str, Any]:
    """Analyze CFD domain to get bounds and structure (copied from original)"""
    
    print(f"   📖 Reading domain structure from: {os.path.basename(su2_file)}")
    
    nodes = []
    
    with open(su2_file, 'r') as f:
        lines = f.readlines()
    
    reading_nodes = False
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('NPOIN='):
            reading_nodes = True
            continue
        elif line.startswith('NMARK='):
            break
        elif reading_nodes and line:
            parts = line.split()
            if len(parts) >= 3:  # Updated for new SU2 format: nodes are just "x y z"
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                nodes.append([x, y, z])
    
    nodes = np.array(nodes)
    
    # Calculate domain bounds
    bounds = {
        'min_x': nodes[:, 0].min(), 'max_x': nodes[:, 0].max(),
        'min_y': nodes[:, 1].min(), 'max_y': nodes[:, 1].max(),
        'min_z': nodes[:, 2].min(), 'max_z': nodes[:, 2].max(),
    }
    
    length = bounds['max_x'] - bounds['min_x']
    height = bounds['max_y'] - bounds['min_y']
    width = bounds['max_z'] - bounds['min_z']
    
    print(f"   📏 Domain: {length:.0f} x {height:.0f} x {width:.0f}")
    print(f"   📊 Total nodes: {len(nodes):,}")
    
    return {
        'bounds': bounds,
        'dimensions': (length, height, width),
        'nodes': nodes,
        'center': [(bounds['min_x'] + bounds['max_x'])/2,
                  (bounds['min_y'] + bounds['max_y'])/2,
                  (bounds['min_z'] + bounds['max_z'])/2]
    }

def create_smart_point_cloud(domain_info: Dict[str, Any], sampling_ratio: float) -> np.ndarray:
    """Create point cloud with wind tunnel domain nodes (copied from original)"""
    
    nodes = domain_info['nodes']
    
    print(f"   🎯 Creating wind tunnel point cloud...")
    
    # Use ALL nodes for complete coverage - no filtering!
    # The mesh generation already properly separates wind tunnel from object
    wind_tunnel_points = nodes
    
    print(f"   📊 Complete point cloud: {len(wind_tunnel_points):,} points")
    print(f"      Using ALL mesh nodes for full domain coverage")
    
    # Sample for performance based on sampling ratio
    max_points = int(100000 * sampling_ratio)  # Scale max points by ratio
    
    if len(wind_tunnel_points) > max_points:
        print(f"      Sampling for performance...")
        indices = np.random.choice(len(wind_tunnel_points), 
                                 size=min(max_points, len(wind_tunnel_points)), 
                                 replace=False)
        wind_tunnel_points = wind_tunnel_points[indices]
        print(f"      Sampled to: {len(wind_tunnel_points):,} points")
    
    return wind_tunnel_points

def extract_object_from_su2(su2_file: str) -> str:
    """Extract object elements from SU2 file and create object VTK (like propeller_only.vtk)"""
    
    import tempfile
    import pyvista as pv
    
    print(f"   📄 Extracting object from: {os.path.basename(su2_file)}")
    
    # Create temporary file name
    temp_dir = tempfile.gettempdir()
    object_vtk_file = os.path.join(temp_dir, "generalized_object_temp.vtk")
    
    try:
        # Read SU2 file to extract object elements
        nodes = []
        elements = []
        object_elements = []
        
        with open(su2_file, 'r') as f:
            lines = f.readlines()
        
        # Parse SU2 file
        reading_elements = False
        reading_nodes = False
        reading_markers = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('NELEM='):
                reading_elements = True
                continue
            elif line.startswith('NPOIN='):
                reading_elements = False
                reading_nodes = True
                continue
            elif line.startswith('NMARK='):
                reading_nodes = False
                reading_markers = True
                continue
            elif reading_elements and line:
                # Parse element: type node1 node2 node3 [node4] element_id
                parts = line.split()
                if len(parts) >= 4:
                    elem_type = int(parts[0])
                    nodes_in_elem = parts[1:-1]  # All but first and last
                    elem_id = int(parts[-1])
                    elements.append({
                        'type': elem_type,
                        'nodes': [int(n) for n in nodes_in_elem],
                        'id': elem_id
                    })
            elif reading_nodes and line:
                # Parse node: x y z node_id
                parts = line.split()
                if len(parts) >= 4:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    node_id = int(parts[3])
                    nodes.append([x, y, z, node_id])
            elif reading_markers and line.startswith('TAG_BOUNDARY='):
                # Check if this is an object boundary (not inlet, outlet, walls)
                boundary_name = line.split('=')[1].strip()
                if boundary_name.lower() in ['object_wall', 'object', 'propeller', 'wing', 'body', 'airfoil']:
                    # This is likely the object boundary - read its elements
                    print(f"   🎯 Found object boundary: {boundary_name}")
                    # Note: We'll use geometric detection instead of boundary markers
                    # since boundary names vary
        
        print(f"   📊 Parsed: {len(nodes)} nodes, {len(elements)} elements")
        
        # Convert to numpy arrays
        nodes_array = np.array([[n[0], n[1], n[2]] for n in nodes])
        
        # Strategy: Identify object elements using geometric analysis
        # Object elements are typically in the center, away from domain boundaries
        
        # Calculate domain bounds
        bounds = {
            'min_x': nodes_array[:, 0].min(), 'max_x': nodes_array[:, 0].max(),
            'min_y': nodes_array[:, 1].min(), 'max_y': nodes_array[:, 1].max(),
            'min_z': nodes_array[:, 2].min(), 'max_z': nodes_array[:, 2].max(),
        }
        
        center = np.array([
            (bounds['min_x'] + bounds['max_x']) / 2,
            (bounds['min_y'] + bounds['max_y']) / 2,
            (bounds['min_z'] + bounds['max_z']) / 2
        ])
        
        # Identify object elements
        object_elements = []
        object_nodes_set = set()
        
        print(f"   🔍 Identifying object elements...")
        
        for elem in elements:
            # Get element center (average of node positions)
            elem_nodes = [nodes[i] for i in elem['nodes'] if i < len(nodes)]
            if len(elem_nodes) == 0:
                continue
                
            elem_center = np.mean([[n[0], n[1], n[2]] for n in elem_nodes], axis=0)
            
            # Distance from domain center
            dist_from_center = np.linalg.norm(elem_center - center)
            
            # Check if element is away from boundaries (likely object)
            boundary_tolerance = 0.2  # 20% tolerance from boundaries
            x_range = bounds['max_x'] - bounds['min_x']
            y_range = bounds['max_y'] - bounds['min_y']
            z_range = bounds['max_z'] - bounds['min_z']
            
            is_near_boundary = (
                elem_center[0] <= bounds['min_x'] + x_range * boundary_tolerance or
                elem_center[0] >= bounds['max_x'] - x_range * boundary_tolerance or
                elem_center[1] <= bounds['min_y'] + y_range * boundary_tolerance or
                elem_center[1] >= bounds['max_y'] - y_range * boundary_tolerance or
                elem_center[2] <= bounds['min_z'] + z_range * boundary_tolerance or
                elem_center[2] >= bounds['max_z'] - z_range * boundary_tolerance
            )
            
            # Object elements: not near boundaries and in inner region
            max_dist = max(x_range, y_range, z_range) * 0.3  # Inner 30% region
            
            if not is_near_boundary and dist_from_center <= max_dist:
                object_elements.append(elem)
                object_nodes_set.update(elem['nodes'])
        
        print(f"   🎯 Found {len(object_elements)} object elements using {len(object_nodes_set)} nodes")
        
        if len(object_elements) == 0:
            print(f"   ⚠️ No object elements found, creating empty object")
            empty_mesh = pv.PolyData()
            empty_mesh.save(object_vtk_file)
            return object_vtk_file
        
        # Create VTK mesh from object elements
        object_nodes_list = sorted(list(object_nodes_set))
        node_id_map = {old_id: new_id for new_id, old_id in enumerate(object_nodes_list)}
        
        # Extract object node coordinates
        object_points = []
        for node_id in object_nodes_list:
            if node_id < len(nodes):
                node = nodes[node_id]
                object_points.append([node[0], node[1], node[2]])
        
        object_points = np.array(object_points)
        
        # Convert elements to VTK format
        vtk_cells = []
        for elem in object_elements:
            # Map old node IDs to new node IDs
            mapped_nodes = []
            for node_id in elem['nodes']:
                if node_id in node_id_map:
                    mapped_nodes.append(node_id_map[node_id])
            
            if len(mapped_nodes) >= 3:  # Valid element
                vtk_cells.append(mapped_nodes)
        
        # Create PyVista mesh
        if len(vtk_cells) > 0 and len(object_points) > 0:
            try:
                # Create mesh with triangular/tetrahedral cells
                mesh = pv.PolyData()
                mesh.points = object_points
                
                # Convert cells to VTK format - be more careful with cell types
                valid_faces = []
                for cell in vtk_cells:
                    if len(cell) == 3:  # Triangular elements
                        # Validate indices are within range
                        if all(0 <= idx < len(object_points) for idx in cell):
                            valid_faces.extend([3] + cell)  # 3 = number of points in triangle
                
                if valid_faces:
                    mesh.faces = valid_faces
                    mesh = mesh.clean()
                    mesh.save(object_vtk_file)
                    print(f"   ✅ Object VTK created: {mesh.n_points:,} points, {mesh.n_cells:,} cells")
                else:
                    # Fallback to point cloud
                    mesh = pv.PolyData(object_points)
                    mesh.save(object_vtk_file)
                    print(f"   ✅ Object point cloud: {len(object_points):,} points")
                    
            except Exception as e:
                print(f"   ⚠️ VTK mesh creation failed: {e}")
                # Fallback to point cloud
                mesh = pv.PolyData(object_points)
                mesh.save(object_vtk_file)
                print(f"   ✅ Object point cloud: {len(object_points):,} points")
        else:
            print(f"   ⚠️ No valid object cells, creating point cloud")
            mesh = pv.PolyData(object_points)
            mesh.save(object_vtk_file)
            print(f"   ✅ Object point cloud: {len(object_points):,} points")
        
        return object_vtk_file
        
    except Exception as e:
        print(f"   ❌ Error extracting object: {e}")
        import traceback
        traceback.print_exc()
        # Create fallback empty file
        empty_mesh = pv.PolyData()
        empty_mesh.save(object_vtk_file)
        return object_vtk_file



def calculate_local_densities_simple(points: np.ndarray) -> np.ndarray:
    """Calculate local point density using simplified approach"""
    
    n_points = len(points)
    if n_points == 0:
        return np.array([])
    
    # Use adaptive radius based on mesh size
    mesh_size = np.max(np.ptp(points, axis=0))
    radius = mesh_size * 0.05  # 5% of mesh size
    
    densities = np.zeros(n_points)
    
    # Sample for efficiency if too many points
    if n_points > 5000:
        sample_indices = np.random.choice(n_points, 5000, replace=False)
        sample_points = points[sample_indices]
        
        # Calculate density for sampled points
        for i, point in enumerate(sample_points):
            distances = np.linalg.norm(sample_points - point, axis=1)
            neighbor_count = np.sum(distances <= radius)
            densities[sample_indices[i]] = neighbor_count
        
        # Interpolate for non-sampled points
        non_sample_indices = np.setdiff1d(np.arange(n_points), sample_indices)
        if len(non_sample_indices) > 0:
            non_sample_points = points[non_sample_indices]
            for i, point in enumerate(non_sample_points):
                distances = np.linalg.norm(sample_points - point, axis=1)
                nearest_idx = np.argmin(distances)
                densities[non_sample_indices[i]] = densities[sample_indices[nearest_idx]]
    else:
        # Calculate for all points if small enough
        for i, point in enumerate(points):
            distances = np.linalg.norm(points - point, axis=1)
            neighbor_count = np.sum(distances <= radius)
            densities[i] = neighbor_count
    
    return densities

def extract_object_to_vtk(mesh, object_indices: np.ndarray, original_vtk_file: str) -> str:
    """Extract object mesh to temporary VTK file"""
    
    import tempfile
    import pyvista as pv
    
    # Create temporary file name
    temp_dir = tempfile.gettempdir()
    object_vtk_file = os.path.join(temp_dir, "object_temp.vtk")
    
    print(f"   📄 Extracting object to: {object_vtk_file}")
    
    try:
        if len(object_indices) == 0:
            print(f"   ⚠️ No object points found, creating empty object file")
            # Create empty VTK file
            empty_mesh = pv.PolyData()
            empty_mesh.save(object_vtk_file)
            return object_vtk_file
        
        # Find cells that contain object points
        object_cells = []
        
        print(f"   🔍 Finding object cells...")
        max_cells_to_check = min(mesh.n_cells, 20000)  # Limit for performance
        
        for cell_id in range(max_cells_to_check):
            try:
                cell = mesh.get_cell(cell_id)
                cell_points = cell.point_ids
                if np.any(np.isin(cell_points, object_indices)):
                    object_cells.append(cell_id)
            except:
                continue
        
        if object_cells:
            print(f"   📊 Found {len(object_cells)} object cells")
            # Extract object mesh
            object_mesh = mesh.extract_cells(object_cells)
            
            # Clean up the mesh
            object_mesh = object_mesh.clean()
            
            # Save to temporary file
            object_mesh.save(object_vtk_file)
            print(f"   ✅ Object extracted successfully ({object_mesh.n_points:,} points, {object_mesh.n_cells:,} cells)")
        else:
            print(f"   ⚠️ No object cells found, creating point cloud")
            # Create point cloud from object points
            object_points = mesh.points[object_indices]
            object_cloud = pv.PolyData(object_points)
            object_cloud.save(object_vtk_file)
            print(f"   ✅ Object point cloud created ({len(object_points):,} points)")
        
        return object_vtk_file
        
    except Exception as e:
        print(f"   ❌ Error extracting object: {e}")
        # Create fallback empty file
        empty_mesh = pv.PolyData()
        empty_mesh.save(object_vtk_file)
        return object_vtk_file



def create_generalized_smart_viz(object_vtk_file: str,
                                point_cloud: np.ndarray,
                                domain_info: Dict[str, Any],
                                flow_direction: str,
                                object_color: str,
                                domain_color: str,
                                show_boundaries: bool,
                                point_size: int) -> bool:
    """Create the smart visualization (following original pattern)"""
    
    try:
        import pyvista as pv
        
        bounds = domain_info['bounds']
        center = domain_info['center']
        dimensions = domain_info['dimensions']
        
        print(f"   🎨 Setting up visualization...")
        
        # Create plotter with black background
        plotter = pv.Plotter(window_size=[1600, 1200])
        plotter.set_background('black')
        
        # 1. Add object mesh (detailed rendering from extracted VTK)
        if object_vtk_file and os.path.exists(object_vtk_file):
            print(f"      🎯 Loading object from: {os.path.basename(object_vtk_file)}")
            
            try:
                object_mesh = pv.read(object_vtk_file)
                
                if object_mesh.n_points > 0:
                    print(f"      🎯 Adding object mesh ({object_mesh.n_points:,} points, {object_mesh.n_cells:,} cells)...")
                    
                    if object_mesh.n_cells > 0:
                        # Render as mesh with edges
                        plotter.add_mesh(
                            object_mesh,
                            color=object_color,
                            show_edges=True,
                            edge_color='darkgoldenrod',
                            opacity=0.9,
                            label='Object',
                            smooth_shading=True
                        )
                    else:
                        # Render as point cloud if no cells
                        plotter.add_mesh(
                            object_mesh,
                            color=object_color,
                            point_size=point_size * 2,
                            render_points_as_spheres=True,
                            opacity=0.8,
                            label='Object'
                        )
                else:
                    print(f"      ⚠️ Object mesh is empty")
                    
            except Exception as e:
                print(f"      ❌ Error loading object mesh: {e}")
        else:
            print(f"      ⚠️ No object VTK file available")
        
        # 2. Add domain as point cloud (CFD domain from SU2)
        if len(point_cloud) > 0:
            print(f"      🌊 Adding CFD domain point cloud ({len(point_cloud):,} points)...")
            
            point_cloud_mesh = pv.PolyData(point_cloud)
            plotter.add_mesh(
                point_cloud_mesh,
                color=domain_color,
                point_size=point_size,
                opacity=0.6,
                label='CFD Domain'
            )
        
        # 3. Add boundary visualization
        if show_boundaries:
            print(f"      🔲 Adding boundary elements...")
            add_boundary_visualization_generalized(plotter, bounds, center, dimensions, flow_direction)
        
        # 4. Add flow direction indicator
        print(f"      ➡️ Adding flow direction indicator...")
        add_flow_direction_indicator_generalized(plotter, bounds, center, dimensions, flow_direction)
        
        # 5. Setup camera and view
        setup_optimal_camera_view_generalized(plotter, bounds, center, dimensions, flow_direction)
        
        # 6. Add UI elements
        add_ui_elements_generalized(plotter, domain_info, point_cloud, flow_direction)
        
        print(f"\n✨ Opening interactive 3D visualization...")
        print(f"   🖱️ Mouse controls:")
        print(f"      • Left click + drag: Rotate")
        print(f"      • Right click + drag: Pan")
        print(f"      • Scroll: Zoom")
        print(f"      • Press 'q' to quit")
        print(f"      • Press 's' to screenshot")
        
        # Show the plot
        plotter.show()
        
        print("✅ Visualization completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_temp_files(regions: Dict[str, Any]):
    """Clean up temporary files"""
    
    object_vtk_file = regions.get('object_vtk_file')
    if object_vtk_file and os.path.exists(object_vtk_file):
        try:
            os.remove(object_vtk_file)
            print(f"   🧹 Cleaned up temporary file: {object_vtk_file}")
        except Exception as e:
            print(f"   ⚠️ Could not remove temporary file: {e}")

def add_boundary_visualization_generalized(plotter, bounds: Dict[str, float], center: List[float], 
                                         dimensions: Tuple[float, float, float], flow_direction: str):
    """Add boundary box and planes"""
    
    import pyvista as pv
    
    # Convert bounds dict to list format for pyvista
    bounds_list = [
        bounds['min_x'], bounds['max_x'],  # X
        bounds['min_y'], bounds['max_y'],  # Y
        bounds['min_z'], bounds['max_z']   # Z
    ]
    
    # Boundary box
    boundary_box = pv.Box(bounds=bounds_list)
    plotter.add_mesh(
        boundary_box,
        style='wireframe',
        color='lime',
        line_width=3,
        opacity=0.8,
        label='Wind Tunnel'
    )
    
    # Inlet and outlet planes based on flow direction
    flow_info = get_flow_direction_info(flow_direction)
    
    if flow_info:
        axis_idx = flow_info['axis_idx']
        direction = flow_info['direction']
        
        # Convert dimensions tuple to dict for compatibility
        dim_dict = {
            'length': dimensions[0],
            'width': dimensions[1], 
            'height': dimensions[2]
        }
        
        # Inlet plane
        inlet_center = center.copy()
        if axis_idx == 0:  # X axis
            inlet_center[axis_idx] = bounds['min_x'] if direction > 0 else bounds['max_x']
        elif axis_idx == 1:  # Y axis
            inlet_center[axis_idx] = bounds['min_y'] if direction > 0 else bounds['max_y']
        else:  # Z axis
            inlet_center[axis_idx] = bounds['min_z'] if direction > 0 else bounds['max_z']
        
        inlet_plane = create_boundary_plane_generalized(inlet_center, flow_info['normal'], dim_dict)
        plotter.add_mesh(inlet_plane, color='red', opacity=0.3, label='Inlet')
        
        # Outlet plane  
        outlet_center = center.copy()
        if axis_idx == 0:  # X axis
            outlet_center[axis_idx] = bounds['max_x'] if direction > 0 else bounds['min_x']
        elif axis_idx == 1:  # Y axis
            outlet_center[axis_idx] = bounds['max_y'] if direction > 0 else bounds['min_y']
        else:  # Z axis
            outlet_center[axis_idx] = bounds['max_z'] if direction > 0 else bounds['min_z']
        
        outlet_plane = create_boundary_plane_generalized(outlet_center, flow_info['normal'], dim_dict)
        plotter.add_mesh(outlet_plane, color='blue', opacity=0.3, label='Outlet')

def create_boundary_plane_generalized(center: List[float], normal: np.ndarray, dimensions: Dict[str, float]):
    """Create a boundary plane"""
    
    import pyvista as pv
    
    # Determine plane size based on dimensions
    if abs(normal[0]) > 0.5:  # X-normal plane
        i_size = dimensions['width']
        j_size = dimensions['height']
    elif abs(normal[1]) > 0.5:  # Y-normal plane
        i_size = dimensions['length']
        j_size = dimensions['height']
    else:  # Z-normal plane
        i_size = dimensions['length']
        j_size = dimensions['width']
    
    return pv.Plane(center=center, direction=normal, i_size=i_size, j_size=j_size)

def add_flow_direction_indicator_generalized(plotter, bounds: Dict[str, float], center: List[float],
                                           dimensions: Tuple[float, float, float], flow_direction: str):
    """Add flow direction arrow"""
    
    import pyvista as pv
    
    flow_info = get_flow_direction_info(flow_direction)
    
    if flow_info:
        # Position arrow outside the domain
        arrow_start = center.copy()
        axis_idx = flow_info['axis_idx']
        direction = flow_info['direction']
        
        # Place arrow upstream
        offset = max(dimensions) * 0.3
        
        if axis_idx == 0:  # X axis
            arrow_start[axis_idx] = bounds['min_x'] - offset if direction > 0 else bounds['max_x'] + offset
        elif axis_idx == 1:  # Y axis
            arrow_start[axis_idx] = bounds['min_y'] - offset if direction > 0 else bounds['max_y'] + offset
        else:  # Z axis
            arrow_start[axis_idx] = bounds['min_z'] - offset if direction > 0 else bounds['max_z'] + offset
        
        # Create arrow
        arrow_scale = max(dimensions) * 0.15
        arrow = pv.Arrow(
            start=arrow_start,
            direction=flow_info['normal'] * direction,
            scale=arrow_scale
        )
        
        plotter.add_mesh(arrow, color='red', label=f'Flow {flow_direction}')

def get_flow_direction_info(flow_direction: str) -> Optional[Dict[str, Any]]:
    """Get flow direction information"""
    
    flow_map = {
        '+X': {'axis_idx': 0, 'direction': 1, 'normal': np.array([1, 0, 0])},
        '-X': {'axis_idx': 0, 'direction': -1, 'normal': np.array([1, 0, 0])},
        '+Y': {'axis_idx': 1, 'direction': 1, 'normal': np.array([0, 1, 0])},
        '-Y': {'axis_idx': 1, 'direction': -1, 'normal': np.array([0, 1, 0])},
        '+Z': {'axis_idx': 2, 'direction': 1, 'normal': np.array([0, 0, 1])},
        '-Z': {'axis_idx': 2, 'direction': -1, 'normal': np.array([0, 0, 1])},
    }
    
    return flow_map.get(flow_direction.upper())

def setup_optimal_camera_view_generalized(plotter, bounds: Dict[str, float], center: List[float],
                                         dimensions: Tuple[float, float, float], flow_direction: str):
    """Setup optimal camera position"""
    
    # Position camera for best view based on flow direction
    max_dim = max(dimensions)
    
    flow_info = get_flow_direction_info(flow_direction)
    
    if flow_info:
        axis_idx = flow_info['axis_idx']
        
        # Position camera perpendicular to flow direction
        camera_pos = center.copy()
        
        if axis_idx == 0:  # X flow - view from Y-Z perspective
            camera_pos[1] -= max_dim * 1.5
            camera_pos[2] += max_dim * 0.8
        elif axis_idx == 1:  # Y flow - view from X-Z perspective  
            camera_pos[0] -= max_dim * 1.5
            camera_pos[2] += max_dim * 0.8
        else:  # Z flow - view from X-Y perspective
            camera_pos[0] -= max_dim * 1.5
            camera_pos[1] += max_dim * 0.8
        
        plotter.camera_position = [camera_pos, center, [0, 0, 1]]
    
    # Add coordinate axes
    plotter.add_axes(
        xlabel='X (Flow)' if flow_direction.upper().startswith('+X') or flow_direction.upper().startswith('-X') else 'X',
        ylabel='Y (Flow)' if flow_direction.upper().startswith('+Y') or flow_direction.upper().startswith('-Y') else 'Y',
        zlabel='Z (Flow)' if flow_direction.upper().startswith('+Z') or flow_direction.upper().startswith('-Z') else 'Z',
        line_width=5, labels_off=False
    )

def add_ui_elements_generalized(plotter, domain_info: Dict[str, Any], point_cloud: np.ndarray, flow_direction: str):
    """Add UI elements like legend, title, and info text"""
    
    # Title
    plotter.add_title("Generalized Smart CFD Visualization", font_size=16)
    
    # Legend
    plotter.add_legend(bcolor='white', size=(0.25, 0.3))
    
    # Info text
    dimensions = domain_info['dimensions']
    info_text = f"""Flow Direction: {flow_direction}
Domain: {dimensions[0]:.0f} x {dimensions[1]:.0f} x {dimensions[2]:.0f}
Nodes: {len(domain_info['nodes']):,}
Point Cloud: {len(point_cloud):,}"""
    
    plotter.add_text(
        info_text,
        position='upper_left',
        font_size=10,
        color='white'
    )

def main():
    """Main function for command line usage"""
    
    parser = argparse.ArgumentParser(description='Generalized Smart CFD Visualizer')
    parser.add_argument('su2_file', help='Path to SU2 wind tunnel file')
    parser.add_argument('--flow-direction', default='+X', 
                       choices=['+X', '-X', '+Y', '-Y', '+Z', '-Z'],
                       help='Flow direction')
    parser.add_argument('--object-color', default='gold', help='Object color')
    parser.add_argument('--domain-color', default='lightblue', help='Domain color')
    parser.add_argument('--no-boundaries', action='store_true', help='Hide boundary planes')
    parser.add_argument('--point-size', type=int, default=3, help='Domain point size')
    parser.add_argument('--domain-sampling', type=float, default=0.3, 
                       help='Domain sampling ratio (0.1-1.0)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not 0.1 <= args.domain_sampling <= 1.0:
        print("❌ Domain sampling ratio must be between 0.1 and 1.0")
        return False
    
    # Run visualization
    success = generalized_smart_visualization(
        su2_file=args.su2_file,
        flow_direction=args.flow_direction,
        object_color=args.object_color,
        domain_color=args.domain_color,
        show_boundaries=not args.no_boundaries,
        point_size=args.point_size,
        domain_sampling_ratio=args.domain_sampling
    )
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 