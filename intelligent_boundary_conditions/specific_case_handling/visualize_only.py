#!/usr/bin/env python3
"""
Simple visualization script for rocket engine mesh tagging
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add the parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

from geometry_parser import GeometryParser

def create_simple_visualization():
    """Create a simple 3D scatter plot visualization"""
    
    print("🎨 Creating simple visualization...")
    
    # Parse the geometry
    mesh_file = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\mesh\Rocket Engine.msh"
    parser = GeometryParser(mesh_file)
    geometry_data = parser.parse_geometry()
    
    print(f"✅ Loaded {len(parser.faces)} faces")
    
    # Run our custom tagger (simplified version)
    bounds_extent = parser.geometry_bounds['extent']
    primary_axis = np.argmax(bounds_extent)
    
    print(f"📏 Primary axis: {['X', 'Y', 'Z'][primary_axis]} (extent: {bounds_extent[primary_axis]:.1f})")
    
    # Get face coordinates along primary axis
    coords = [face.centroid[primary_axis] for face in parser.faces]
    min_coord = min(coords)
    max_coord = max(coords)
    coord_range = max_coord - min_coord
    
    # Define end regions
    end_tolerance = 0.03 * coord_range
    
    print(f"🎯 End tolerance: {end_tolerance:.1f}")
    
    # Tag faces
    inlet_faces = []
    outlet_faces = []
    wall_faces = []
    
    for i, face in enumerate(parser.faces):
        coord = face.centroid[primary_axis]
        
        if coord <= min_coord + end_tolerance:
            inlet_faces.append(i)
            face.tag = 'inlet'
        elif coord >= max_coord - end_tolerance:
            outlet_faces.append(i)
            face.tag = 'outlet'
        else:
            wall_faces.append(i)
            face.tag = 'wall'
    
    print(f"🏷️  Tagged: {len(inlet_faces)} inlet, {len(outlet_faces)} outlet, {len(wall_faces)} wall")
    
    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Sample faces for visualization (too many to show all)
    sample_size = 1000
    
    # Sample inlet faces
    if inlet_faces:
        inlet_sample = np.random.choice(inlet_faces, min(len(inlet_faces), sample_size//3), replace=False)
        inlet_centroids = np.array([parser.faces[i].centroid for i in inlet_sample])
        ax1.scatter(inlet_centroids[:, 0], inlet_centroids[:, 1], inlet_centroids[:, 2], 
                   c='red', s=50, alpha=0.8, label=f'Inlet ({len(inlet_faces)})')
    
    # Sample outlet faces
    if outlet_faces:
        outlet_sample = np.random.choice(outlet_faces, min(len(outlet_faces), sample_size//3), replace=False)
        outlet_centroids = np.array([parser.faces[i].centroid for i in outlet_sample])
        ax1.scatter(outlet_centroids[:, 0], outlet_centroids[:, 1], outlet_centroids[:, 2], 
                   c='green', s=50, alpha=0.8, label=f'Outlet ({len(outlet_faces)})')
    
    # Sample wall faces
    if wall_faces:
        wall_sample = np.random.choice(wall_faces, min(len(wall_faces), sample_size//3), replace=False)
        wall_centroids = np.array([parser.faces[i].centroid for i in wall_sample])
        ax1.scatter(wall_centroids[:, 0], wall_centroids[:, 1], wall_centroids[:, 2], 
                   c='gray', s=10, alpha=0.3, label=f'Wall ({len(wall_faces)})')
    
    ax1.set_xlabel('X (Flow Direction)')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Rocket Nozzle - Tagged Faces')
    ax1.legend()
    
    # Side view (X-Y plane)
    ax2 = fig.add_subplot(222)
    
    if inlet_faces:
        inlet_centroids = np.array([parser.faces[i].centroid for i in inlet_faces[:100]])
        ax2.scatter(inlet_centroids[:, 0], inlet_centroids[:, 1], c='red', s=20, alpha=0.8, label='Inlet')
    
    if outlet_faces:
        outlet_centroids = np.array([parser.faces[i].centroid for i in outlet_faces[:100]])
        ax2.scatter(outlet_centroids[:, 0], outlet_centroids[:, 1], c='green', s=20, alpha=0.8, label='Outlet')
    
    if wall_faces:
        wall_sample = np.random.choice(wall_faces, min(len(wall_faces), 500), replace=False)
        wall_centroids = np.array([parser.faces[i].centroid for i in wall_sample])
        ax2.scatter(wall_centroids[:, 0], wall_centroids[:, 1], c='gray', s=1, alpha=0.3, label='Wall')
    
    ax2.set_xlabel('X (Flow Direction)')
    ax2.set_ylabel('Y')
    ax2.set_title('Side View (X-Y)')
    ax2.legend()
    ax2.grid(True)
    
    # Top view (X-Z plane)
    ax3 = fig.add_subplot(223)
    
    if inlet_faces:
        inlet_centroids = np.array([parser.faces[i].centroid for i in inlet_faces[:100]])
        ax3.scatter(inlet_centroids[:, 0], inlet_centroids[:, 2], c='red', s=20, alpha=0.8, label='Inlet')
    
    if outlet_faces:
        outlet_centroids = np.array([parser.faces[i].centroid for i in outlet_faces[:100]])
        ax3.scatter(outlet_centroids[:, 0], outlet_centroids[:, 2], c='green', s=20, alpha=0.8, label='Outlet')
    
    if wall_faces:
        wall_sample = np.random.choice(wall_faces, min(len(wall_faces), 500), replace=False)
        wall_centroids = np.array([parser.faces[i].centroid for i in wall_sample])
        ax3.scatter(wall_centroids[:, 0], wall_centroids[:, 2], c='gray', s=1, alpha=0.3, label='Wall')
    
    ax3.set_xlabel('X (Flow Direction)')
    ax3.set_ylabel('Z')
    ax3.set_title('Top View (X-Z)')
    ax3.legend()
    ax3.grid(True)
    
    # Face distribution histogram
    ax4 = fig.add_subplot(224)
    
    coords_array = np.array(coords)
    ax4.hist(coords_array, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    ax4.axvline(min_coord + end_tolerance, color='red', linestyle='--', label='Inlet boundary')
    ax4.axvline(max_coord - end_tolerance, color='green', linestyle='--', label='Outlet boundary')
    ax4.set_xlabel('X Coordinate')
    ax4.set_ylabel('Number of Faces')
    ax4.set_title('Face Distribution Along Flow Axis')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'rocket_nozzle_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Saved visualization to: {output_file}")
    
    # Show the plot
    try:
        plt.show()
        print("📊 Visualization displayed!")
    except:
        print("📊 Interactive display not available, but image saved successfully!")

if __name__ == "__main__":
    create_simple_visualization() 