#!/usr/bin/env python3
"""
Smart CFD Visualizer
Point cloud for domain + detailed object + boundary box
"""

import os
import sys
import numpy as np

def smart_cfd_visualization():
    """Create smart CFD visualization"""
    
    print("🎯 SMART CFD VISUALIZER")
    print("=" * 30)
    
    # Input files - Updated paths for packaged endpoint
    cfd_file = "output/propeller_wind_tunnel_cfd.su2"
    propeller_file = "propeller_only.vtk"
    
    if not os.path.exists(cfd_file):
        print(f"❌ CFD mesh not found: {cfd_file}")
        return False
    
    if not os.path.exists(propeller_file):
        print(f"❌ Propeller VTK not found: {propeller_file}")
        print("Run extract_propeller_only.py first")
        return False
    
    try:
        print("🔍 STEP 1: Analyzing CFD domain...")
        domain_info = analyze_cfd_domain(cfd_file)
        
        print("🔍 STEP 2: Creating point cloud...")
        point_cloud = create_smart_point_cloud(domain_info)
        
        print("🔍 STEP 3: Creating smart visualization...")
        success = create_smart_viz(propeller_file, point_cloud, domain_info)
        
        return success
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def analyze_cfd_domain(su2_file):
    """Analyze CFD domain to get bounds and structure"""
    
    print(f"   📖 Reading domain structure...")
    
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

def create_smart_point_cloud(domain_info):
    """Create point cloud with ALL wind tunnel elements"""
    
    bounds = domain_info['bounds']
    nodes = domain_info['nodes']
    center = domain_info['center']
    
    print(f"   🎯 Creating complete wind tunnel point cloud...")
    
    # Use ALL nodes for complete coverage - no filtering!
    # The mesh generation already properly separates wind tunnel from object
    wind_tunnel_points = nodes
    
    print(f"   📊 Complete point cloud: {len(wind_tunnel_points):,} points")
    print(f"      Using ALL mesh nodes for full domain coverage")
    
    # Optional: Sample for performance if too many points
    if len(wind_tunnel_points) > 100000:  # Only sample if >100k points
        print(f"      Sampling for performance...")
        indices = np.random.choice(len(wind_tunnel_points), 
                                 size=min(50000, len(wind_tunnel_points)), 
                                 replace=False)
        wind_tunnel_points = wind_tunnel_points[indices]
        print(f"      Sampled to: {len(wind_tunnel_points):,} points")
    
    return wind_tunnel_points

def create_smart_viz(propeller_vtk, point_cloud, domain_info):
    """Create smart CFD visualization"""
    
    try:
        import pyvista as pv
        
        print(f"   🎨 Loading propeller geometry...")
        propeller = pv.read(propeller_vtk)
        
        print(f"   🎨 Creating visualization...")
        
        # Create plotter
        plotter = pv.Plotter(window_size=[1400, 1000])
        plotter.set_background('darkslategray')
        
        # 1. Add detailed propeller (main object)
        print(f"      🟡 Adding detailed propeller...")
        plotter.add_mesh(
            propeller,
            color='gold',
            show_edges=True,
            edge_color='darkgoldenrod',
            opacity=0.9,
            label='Propeller'
        )
        
        # 2. Add point cloud (CFD domain)
        print(f"      🔵 Adding CFD domain point cloud...")
        if len(point_cloud) > 0:
            point_cloud_mesh = pv.PolyData(point_cloud)
            plotter.add_mesh(
                point_cloud_mesh,
                color='lightblue',
                point_size=3,
                opacity=0.6,
                label='CFD Domain'
            )
        
        # 3. Add boundary box (wind tunnel bounds) - Use ACTUAL mesh bounds
        print(f"      🟢 Adding wind tunnel boundaries...")
        bounds = domain_info['bounds']
        
        # Use actual mesh bounds for the bounding box (not calculated bounds)
        # This ensures the box matches exactly where nodes exist
        actual_nodes = domain_info['nodes']
        actual_bounds = [
            actual_nodes[:, 0].min(), actual_nodes[:, 0].max(),  # X
            actual_nodes[:, 1].min(), actual_nodes[:, 1].max(),  # Y
            actual_nodes[:, 2].min(), actual_nodes[:, 2].max()   # Z
        ]
        
        boundary_box = pv.Box(bounds=actual_bounds)
        plotter.add_mesh(
            boundary_box,
            style='wireframe',
            color='lime',
            line_width=3,
            opacity=0.8,
            label='Wind Tunnel'
        )
        
        # 4. Add boundary planes (inlet/outlet) - Use actual mesh bounds
        actual_center = [
            (actual_bounds[0] + actual_bounds[1]) / 2,  # X center
            (actual_bounds[2] + actual_bounds[3]) / 2,  # Y center  
            (actual_bounds[4] + actual_bounds[5]) / 2   # Z center
        ]
        
        actual_height = actual_bounds[3] - actual_bounds[2]  # Y range
        actual_width = actual_bounds[5] - actual_bounds[4]   # Z range
        
        # Inlet plane (red)
        inlet_plane = pv.Plane(
            center=[actual_bounds[0], actual_center[1], actual_center[2]],
            direction=[1, 0, 0],
            i_size=actual_height,
            j_size=actual_width
        )
        plotter.add_mesh(inlet_plane, color='red', opacity=0.3, label='Inlet')
        
        # Outlet plane (blue)
        outlet_plane = pv.Plane(
            center=[actual_bounds[1], actual_center[1], actual_center[2]],
            direction=[1, 0, 0],
            i_size=actual_height,
            j_size=actual_width
        )
        plotter.add_mesh(outlet_plane, color='blue', opacity=0.3, label='Outlet')
        
        # 5. Add flow direction arrow
        arrow_start = [bounds['min_x'] - domain_info['dimensions'][0]*0.2, 
                      domain_info['center'][1], 
                      domain_info['center'][2]]
        arrow = pv.Arrow(start=arrow_start, direction=[1, 0, 0], 
                        scale=domain_info['dimensions'][0]*0.15)
        plotter.add_mesh(arrow, color='red', label='Flow Direction')
        
        # 6. Add coordinate axes
        plotter.add_axes(
            xlabel='X (Flow)',
            ylabel='Y (Height)', 
            zlabel='Z (Width)',
            line_width=5,
            labels_off=False
        )
        
        # Camera setup for best view
        plotter.camera_position = [
            (bounds['min_x'] - domain_info['dimensions'][0], 
             domain_info['center'][1] - domain_info['dimensions'][1], 
             domain_info['center'][2] + domain_info['dimensions'][2]),
            domain_info['center'],
            [0, 1, 0]
        ]
        
        # Add legend
        plotter.add_legend(bcolor='white', size=(0.25, 0.3))
        
        # Add title
        plotter.add_title("Smart CFD Wind Tunnel Visualization", font_size=16)
        
        # Add info text
        info_text = f"""Domain: {domain_info['dimensions'][0]:.0f} x {domain_info['dimensions'][1]:.0f} x {domain_info['dimensions'][2]:.0f}
Nodes: {len(domain_info['nodes']):,}
Point Cloud: {len(point_cloud):,}"""
        
        plotter.add_text(
            info_text,
            position='upper_left',
            font_size=10,
            color='white'
        )
        
        print("\n✨ Opening interactive 3D visualization...")
        print("   🖱️  Mouse controls:")
        print("      • Left click + drag: Rotate")
        print("      • Right click + drag: Pan") 
        print("      • Scroll: Zoom")
        print("      • Press 'q' to quit")
        
        # Show the plot
        plotter.show()
        
        print("✅ Visualization completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = smart_cfd_visualization()
    
    if success:
        print("\n🎉 Smart CFD visualization completed!")
    else:
        print("\n💥 Visualization failed!") 