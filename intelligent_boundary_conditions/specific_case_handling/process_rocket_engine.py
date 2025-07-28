#!/usr/bin/env python3
"""
Rocket Engine Mesh Processor
=============================

This script processes the Rocket Engine.msh file and automatically tags
the surfaces based on area: bigger surface as outlet, smaller as inlet.
Includes visualization capabilities.
"""

import os
import sys
import numpy as np
from pathlib import Path
import logging
from typing import List

# Add the parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

from geometry_parser import GeometryParser
from face_tagger import FaceTagger, TaggingMethod, RocketNozzleType
from boundary_condition_generator import BoundaryConditionGenerator, RocketEngineConditions
from main_api import IntelligentBoundaryConditionsAPI, IntelligentBCRequest

# Optional visualization
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class AreaBasedFaceTagger:
    """
    Custom face tagger that uses surface area to determine inlet/outlet
    """
    
    def __init__(self, geometry_parser: GeometryParser):
        """Initialize with parsed geometry"""
        self.geometry_parser = geometry_parser
        self.faces = geometry_parser.faces
        self.bounds = geometry_parser.geometry_bounds
        self._virtual_inlet = None
        self._virtual_outlet = None
        
    def tag_by_surface_area(self, area_threshold_factor: float = 0.1) -> dict:
        """
        Tag faces based on identifying hollow opening regions (inlet/outlet)
        
        Args:
            area_threshold_factor: Factor to determine what constitutes an end face
            
        Returns:
            Dictionary with tagged face results
        """
        
        logger.info("Analyzing geometry for hollow opening detection...")
        
        # NEW APPROACH: Find hollow openings by analyzing end regions
        return self._find_hollow_openings()
    
    def _find_hollow_openings(self) -> dict:
        """
        Find hollow openings (inlet/outlet) by analyzing the end regions of the nozzle
        """
        
        # Step 1: Determine the primary flow axis (longest dimension)
        bounds_extent = self.bounds['extent']
        primary_axis = np.argmax(bounds_extent)
        axis_names = ['X', 'Y', 'Z']
        
        logger.info(f"Primary flow axis: {axis_names[primary_axis]} (extent: {bounds_extent[primary_axis]:.3f})")
        logger.info(f"Geometry bounds: min={self.bounds['min']}, max={self.bounds['max']}")
        
        # Step 2: Find faces at each end of the primary axis
        end_regions = self._identify_end_regions(primary_axis)
        
        # Step 3: Create virtual inlet/outlet faces to represent the hollow openings
        virtual_faces = self._create_virtual_opening_faces(end_regions, primary_axis)
        
        # Step 4: Tag all actual faces as walls, and create virtual boundary representations
        tagged_faces = {'inlet': [], 'outlet': [], 'wall': []}
        
        # Tag all existing faces as walls (since they're all solid surfaces)
        for i, face in enumerate(self.faces):
            face.tag = 'wall'
            tagged_faces['wall'].append(i)
        
        # Add virtual faces for inlet and outlet
        if len(virtual_faces) >= 2:
            # Sort by area (smaller = inlet, larger = outlet for rocket nozzles)
            virtual_faces.sort(key=lambda x: x['area'])
            
            inlet_virtual = virtual_faces[0]
            outlet_virtual = virtual_faces[-1]  # Largest
            
            logger.info(f"Created virtual inlet: area={inlet_virtual['area']:.6f}, center={inlet_virtual['center']}")
            logger.info(f"Created virtual outlet: area={outlet_virtual['area']:.6f}, center={outlet_virtual['center']}")
            
            # Store virtual face information for boundary condition generation
            self._virtual_inlet = inlet_virtual
            self._virtual_outlet = outlet_virtual
            
            # For the tagging system, we'll mark some nearby real faces as inlet/outlet representatives
            inlet_nearby = self._find_nearby_faces(inlet_virtual['center'], inlet_virtual['radius'])
            outlet_nearby = self._find_nearby_faces(outlet_virtual['center'], outlet_virtual['radius'])
            
            # Retag nearby faces
            for face_idx in inlet_nearby[:5]:  # Tag a few representative faces
                if face_idx < len(self.faces):
                    self.faces[face_idx].tag = 'inlet'
                    tagged_faces['inlet'].append(face_idx)
                    tagged_faces['wall'].remove(face_idx)
            
            for face_idx in outlet_nearby[:5]:  # Tag a few representative faces
                if face_idx < len(self.faces):
                    self.faces[face_idx].tag = 'outlet'
                    tagged_faces['outlet'].append(face_idx)
                    if face_idx in tagged_faces['wall']:
                        tagged_faces['wall'].remove(face_idx)
        
        logger.info(f"Tagged inlet: {len(tagged_faces['inlet'])} faces (virtual opening area: {self._virtual_inlet['area']:.6f})")
        logger.info(f"Tagged outlet: {len(tagged_faces['outlet'])} faces (virtual opening area: {self._virtual_outlet['area']:.6f})")
        logger.info(f"Tagged wall: {len(tagged_faces['wall'])} faces")
        
        return tagged_faces
    
    def _identify_end_regions(self, primary_axis: int) -> List[dict]:
        """
        Identify the end regions where hollow openings should be
        """
        
        # Get coordinates along primary axis
        coords = [face.centroid[primary_axis] for face in self.faces]
        min_coord = min(coords)
        max_coord = max(coords)
        coord_range = max_coord - min_coord
        
        # Define end regions (first and last 3% of the geometry)
        end_tolerance = 0.03 * coord_range
        
        start_region = {
            'center_coord': min_coord,
            'tolerance': end_tolerance,
            'type': 'start'
        }
        
        end_region = {
            'center_coord': max_coord,
            'tolerance': end_tolerance,
            'type': 'end'
        }
        
        logger.info(f"End regions: start at {min_coord:.6f} ± {end_tolerance:.6f}, end at {max_coord:.6f} ± {end_tolerance:.6f}")
        
        return [start_region, end_region]
    
    def _create_virtual_opening_faces(self, end_regions: List[dict], primary_axis: int) -> List[dict]:
        """
        Create virtual faces representing the hollow openings
        """
        
        virtual_faces = []
        
        for region in end_regions:
            # Find faces near this end region
            nearby_faces = []
            target_coord = region['center_coord']
            tolerance = region['tolerance']
            
            for i, face in enumerate(self.faces):
                face_coord = face.centroid[primary_axis]
                if abs(face_coord - target_coord) <= tolerance:
                    nearby_faces.append((i, face))
            
            if nearby_faces:
                # Estimate the opening area based on nearby face distribution
                face_coords = []
                for _, face in nearby_faces:
                    # Get coordinates in the plane perpendicular to primary axis
                    coord = face.centroid.copy()
                    coord[primary_axis] = 0  # Project to perpendicular plane
                    face_coords.append(coord)
                
                face_coords = np.array(face_coords)
                
                # Find the extent of the opening in the perpendicular plane
                if len(face_coords) > 0:
                    perp_axes = [i for i in range(3) if i != primary_axis]
                    
                    # Calculate opening dimensions
                    ranges = []
                    for axis in perp_axes:
                        axis_coords = face_coords[:, axis]
                        ranges.append(axis_coords.max() - axis_coords.min())
                    
                    # Estimate circular opening
                    avg_radius = np.mean(ranges) / 2
                    opening_area = np.pi * avg_radius**2
                    
                    # Get center position
                    center_pos = np.mean([face.centroid for _, face in nearby_faces], axis=0)
                    center_pos[primary_axis] = target_coord  # Place exactly at the end
                    
                    virtual_face = {
                        'center': center_pos,
                        'area': opening_area,
                        'radius': avg_radius,
                        'type': region['type'],
                        'normal_axis': primary_axis
                    }
                    
                    virtual_faces.append(virtual_face)
                    
                    logger.info(f"Virtual {region['type']} opening: area={opening_area:.6f}, radius={avg_radius:.6f}")
        
        return virtual_faces
    
    def _find_nearby_faces(self, center: np.ndarray, radius: float) -> List[int]:
        """
        Find face indices near a given center point
        """
        
        nearby_indices = []
        
        for i, face in enumerate(self.faces):
            distance = np.linalg.norm(face.centroid - center)
            if distance <= radius * 2:  # Within 2x the radius
                nearby_indices.append(i)
        
        # Sort by distance
        nearby_indices.sort(key=lambda i: np.linalg.norm(self.faces[i].centroid - center))
        
        return nearby_indices
    
    def _classify_end_faces_by_area(self, potential_faces) -> dict:
        """
        Classify end faces as inlet (smaller) or outlet (larger)
        Enhanced to handle hollow/open faces by analyzing spatial distribution
        """
        
        if len(potential_faces) < 2:
            logger.error("Not enough potential end faces found")
            return {'inlet': [], 'outlet': [], 'wall': []}
        
        # Sort by area
        sorted_by_area = sorted(potential_faces, key=lambda x: x[2], reverse=True)
        
        # NEW: Analyze spatial distribution to find hollow end faces
        end_faces = self._find_hollow_end_faces(potential_faces)
        
        logger.info(f"Spatial analysis found {len(end_faces)} end face groups")
        
        if len(end_faces) >= 2:
            # Use spatial analysis results
            inlet_faces, outlet_faces = end_faces[0], end_faces[1]
            
            # Determine which is inlet vs outlet based on position and area
            inlet_total_area = sum(face[2] for face in inlet_faces)
            outlet_total_area = sum(face[2] for face in outlet_faces)
            
            # Get average positions to determine flow direction
            inlet_avg_pos = np.mean([face[1].centroid for face in inlet_faces], axis=0)
            outlet_avg_pos = np.mean([face[1].centroid for face in outlet_faces], axis=0)
            
            # For rocket nozzles, typically outlet is larger and downstream
            if outlet_total_area < inlet_total_area:
                # Swap if needed
                inlet_faces, outlet_faces = outlet_faces, inlet_faces
                inlet_total_area, outlet_total_area = outlet_total_area, inlet_total_area
                inlet_avg_pos, outlet_avg_pos = outlet_avg_pos, inlet_avg_pos
            
            logger.info(f"Spatial analysis found:")
            logger.info(f"  Inlet: {len(inlet_faces)} faces, area={inlet_total_area:.6f}, pos={inlet_avg_pos}")
            logger.info(f"  Outlet: {len(outlet_faces)} faces, area={outlet_total_area:.6f}, pos={outlet_avg_pos}")
            
        else:
            # Fallback to area grouping
            logger.info("Falling back to area-based grouping")
            return self._fallback_area_grouping(potential_faces)
        
        # Tag the faces
        tagged_faces = {'inlet': [], 'outlet': [], 'wall': []}
        
        # Tag inlet faces
        for face_idx, face, area in inlet_faces:
            self.faces[face_idx].tag = 'inlet'
            tagged_faces['inlet'].append(face_idx)
        
        # Tag outlet faces
        for face_idx, face, area in outlet_faces:
            self.faces[face_idx].tag = 'outlet'
            tagged_faces['outlet'].append(face_idx)
        
        # Tag remaining faces as walls
        for i, face in enumerate(self.faces):
            if face.tag is None:
                face.tag = 'wall'
                tagged_faces['wall'].append(i)
        
        logger.info(f"Tagged inlet: {len(tagged_faces['inlet'])} faces")
        logger.info(f"Tagged outlet: {len(tagged_faces['outlet'])} faces")
        logger.info(f"Tagged wall: {len(tagged_faces['wall'])} faces")
        
        return tagged_faces
    
    def _find_hollow_end_faces(self, potential_faces) -> List[List]:
        """
        Find hollow/open end faces by spatial clustering
        """
        
        # Extract centroids for clustering
        centroids = np.array([face[1].centroid for face in potential_faces])
        
        # Determine primary flow axis (longest dimension)
        bounds_extent = self.bounds['extent']
        primary_axis = np.argmax(bounds_extent)
        
        logger.info(f"Primary flow axis: {['X', 'Y', 'Z'][primary_axis]} (extent: {bounds_extent})")
        
        # Project centroids onto primary axis
        primary_coords = centroids[:, primary_axis]
        
        # Find clusters at the extremes of the primary axis
        min_coord = primary_coords.min()
        max_coord = primary_coords.max()
        coord_range = max_coord - min_coord
        
        # Define end regions (e.g., first and last 10% of the length)
        end_tolerance = 0.15 * coord_range
        
        start_end_faces = []
        finish_end_faces = []
        
        for i, (face_idx, face, area) in enumerate(potential_faces):
            coord = primary_coords[i]
            
            if coord <= min_coord + end_tolerance:
                start_end_faces.append((face_idx, face, area))
            elif coord >= max_coord - end_tolerance:
                finish_end_faces.append((face_idx, face, area))
        
        logger.info(f"Found {len(start_end_faces)} faces at start end")
        logger.info(f"Found {len(finish_end_faces)} faces at finish end")
        
        # Filter out groups that are too small (likely not actual ends)
        end_face_groups = []
        
        if len(start_end_faces) >= 3:  # Need at least a few faces to form an opening
            end_face_groups.append(start_end_faces)
        
        if len(finish_end_faces) >= 3:
            end_face_groups.append(finish_end_faces)
        
        # Sort groups by total area (larger opening is typically outlet)
        if len(end_face_groups) >= 2:
            end_face_groups.sort(key=lambda group: sum(f[2] for f in group), reverse=True)
        
        return end_face_groups
    
    def _fallback_area_grouping(self, potential_faces) -> dict:
        """
        Fallback area grouping method
        """
        
        # Sort by area
        sorted_by_area = sorted(potential_faces, key=lambda x: x[2], reverse=True)
        
        # Take top 20% as potential end faces
        num_end_faces = max(2, len(sorted_by_area) // 5)
        top_faces = sorted_by_area[:num_end_faces]
        
        # Split into two groups based on position
        centroids = np.array([face[1].centroid for face in top_faces])
        bounds_extent = self.bounds['extent']
        primary_axis = np.argmax(bounds_extent)
        
        coords = centroids[:, primary_axis]
        median_coord = np.median(coords)
        
        group1 = [face for i, face in enumerate(top_faces) if coords[i] <= median_coord]
        group2 = [face for i, face in enumerate(top_faces) if coords[i] > median_coord]
        
        tagged_faces = {'inlet': [], 'outlet': [], 'wall': []}
        
        # Assign smaller area group as inlet, larger as outlet
        if group1 and group2:
            area1 = sum(f[2] for f in group1)
            area2 = sum(f[2] for f in group2)
            
            if area1 < area2:
                inlet_group, outlet_group = group1, group2
            else:
                inlet_group, outlet_group = group2, group1
            
            for face_idx, face, area in inlet_group:
                self.faces[face_idx].tag = 'inlet'
                tagged_faces['inlet'].append(face_idx)
                
            for face_idx, face, area in outlet_group:
                self.faces[face_idx].tag = 'outlet'
                tagged_faces['outlet'].append(face_idx)
        
        # Tag remaining as walls
        for i, face in enumerate(self.faces):
            if face.tag is None:
                face.tag = 'wall'
                tagged_faces['wall'].append(i)
        
        return tagged_faces

def visualize_tagged_faces(geometry_parser: GeometryParser, use_pyvista: bool = True):
    """
    Visualize the tagged faces
    """
    
    if use_pyvista and PYVISTA_AVAILABLE:
        visualize_with_pyvista(geometry_parser)
    elif MATPLOTLIB_AVAILABLE:
        visualize_with_matplotlib(geometry_parser)
    else:
        logger.warning("No visualization libraries available")
        print_face_summary(geometry_parser)

def visualize_with_pyvista(geometry_parser: GeometryParser):
    """Visualize using PyVista"""
    
    logger.info("Creating PyVista visualization...")
    
    try:
        # Create PyVista mesh
        vertices = []
        faces_pv = []
        face_colors = []
        
        color_map = {
            'inlet': [1.0, 0.0, 0.0],    # Red
            'outlet': [0.0, 1.0, 0.0],   # Green  
            'wall': [0.5, 0.5, 0.5],     # Gray
            None: [0.0, 0.0, 1.0]        # Blue for untagged
        }
        
        for face in geometry_parser.faces:
            start_idx = len(vertices)
            vertices.extend(face.vertices.tolist())
            
            # Create face connectivity
            if len(face.vertices) == 3:
                faces_pv.extend([3, start_idx, start_idx + 1, start_idx + 2])
            elif len(face.vertices) == 4:
                faces_pv.extend([4, start_idx, start_idx + 1, start_idx + 2, start_idx + 3])
            
            # Add color based on tag
            color = color_map.get(face.tag, color_map[None])
            face_colors.extend([color] * len(face.vertices))
        
        vertices = np.array(vertices)
        face_colors = np.array(face_colors)
        
        # Create mesh
        mesh = pv.PolyData(vertices, faces_pv)
        
        # Add colors
        mesh.point_data['colors'] = face_colors
        
        # Plot
        plotter = pv.Plotter(off_screen=True)  # Use off-screen rendering
        plotter.add_mesh(mesh, scalars='colors', rgb=True, show_edges=True)
        plotter.add_text("Rocket Engine Mesh\nRed=Inlet, Green=Outlet, Gray=Wall", 
                        position='upper_left', font_size=12)
        
        # Save screenshot instead of showing interactively
        screenshot_path = "rocket_engine_visualization.png"
        plotter.screenshot(screenshot_path)
        logger.info(f"Saved visualization to {screenshot_path}")
        
        # Also try to show if possible
        try:
            plotter.show()
        except:
            logger.info("Interactive display not available, screenshot saved instead")
        
    except Exception as e:
        logger.error(f"PyVista visualization failed: {e}")
        visualize_with_matplotlib(geometry_parser)

def visualize_with_matplotlib(geometry_parser: GeometryParser):
    """Visualize using Matplotlib"""
    
    logger.info("Creating Matplotlib visualization...")
    
    try:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot face centroids colored by tag
        color_map = {
            'inlet': 'red',
            'outlet': 'green', 
            'wall': 'gray',
            None: 'blue'
        }
        
        for tag in ['inlet', 'outlet', 'wall', None]:
            faces_with_tag = [f for f in geometry_parser.faces if f.tag == tag]
            if faces_with_tag:
                centroids = np.array([f.centroid for f in faces_with_tag])
                areas = np.array([f.area for f in faces_with_tag])
                
                # Size points by area
                sizes = (areas / areas.max()) * 100 + 10
                
                ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                          c=color_map[tag], s=sizes, alpha=0.7, 
                          label=f'{tag or "untagged"} ({len(faces_with_tag)} faces)')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        ax.set_title('Rocket Engine Mesh - Face Tagging by Area\n(Point size = face area)')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logger.error(f"Matplotlib visualization failed: {e}")
        print_face_summary(geometry_parser)

def print_face_summary(geometry_parser: GeometryParser):
    """Print a text summary of the tagging results"""
    
    print("\n" + "="*50)
    print("ROCKET ENGINE MESH TAGGING SUMMARY")
    print("="*50)
    
    # Count faces by tag
    tag_counts = {}
    tag_areas = {}
    
    for face in geometry_parser.faces:
        tag = face.tag or 'untagged'
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        tag_areas[tag] = tag_areas.get(tag, 0) + face.area
    
    print(f"{'Tag':<12} {'Count':<8} {'Total Area':<15} {'Avg Area':<12}")
    print("-" * 50)
    
    for tag in ['inlet', 'outlet', 'wall', 'untagged']:
        if tag in tag_counts:
            count = tag_counts[tag]
            total_area = tag_areas[tag]
            avg_area = total_area / count
            print(f"{tag:<12} {count:<8} {total_area:<15.6f} {avg_area:<12.6f}")
    
    print("-" * 50)
    print(f"Total faces: {len(geometry_parser.faces)}")
    print(f"Total area: {sum(f.area for f in geometry_parser.faces):.6f}")

def main():
    """Main processing function"""
    
    # File paths
    mesh_file = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\mesh\Rocket Engine.msh"
    output_dir = "./rocket_engine_bc_output"
    
    print("🚀 ROCKET ENGINE MESH PROCESSOR")
    print("="*50)
    
    # Check if file exists
    if not os.path.exists(mesh_file):
        logger.error(f"Mesh file not found: {mesh_file}")
        return
    
    try:
        # Step 1: Parse geometry
        print("\n📐 Step 1: Parsing geometry...")
        parser = GeometryParser(mesh_file)
        geometry_data = parser.parse_geometry()
        
        summary = parser.get_geometry_summary()
        print(f"✅ Parsed {summary['num_faces']} faces")
        print(f"   Format: {summary['format']}")
        print(f"   Total area: {summary['total_area']:.6f}")
        
        # Step 2: Tag faces by area
        print("\n🏷️  Step 2: Tagging faces by surface area...")
        area_tagger = AreaBasedFaceTagger(parser)
        tagged_faces = area_tagger.tag_by_surface_area()
        
        print(f"✅ Tagged faces:")
        print(f"   Inlet (smaller): {len(tagged_faces['inlet'])} faces")
        print(f"   Outlet (larger): {len(tagged_faces['outlet'])} faces") 
        print(f"   Wall: {len(tagged_faces['wall'])} faces")
        
        # Step 3: Visualize
        print("\n👁️  Step 3: Visualizing results...")
        visualize_tagged_faces(parser, use_pyvista=True)
        
        # Step 4: Generate boundary conditions
        print("\n⚙️  Step 4: Generating boundary conditions...")
        
        # Create a custom face tagger for the API
        face_tagger = FaceTagger(parser)
        # Copy the tagged faces to the face tagger
        for i, face in enumerate(parser.faces):
            face_tagger.faces[i].tag = face.tag
        
        # Generate boundary conditions
        bc_generator = BoundaryConditionGenerator(parser, face_tagger)
        
        # Use hydrogen rocket conditions
        flow_conditions = RocketEngineConditions.get_rocket_conditions(
            fuel_type="hydrogen",
            chamber_pressure=7e6,  # 7 MPa
            chamber_temperature=3600,  # K
            ambient_pressure=101325,  # Sea level
            gamma=1.3
        )
        
        # Domain configuration
        domain_config = {
            "x": {"cells": 200, "range": [-0.2, 0.4]},
            "y": {"cells": 100, "range": [-0.15, 0.15]}, 
            "z": {"cells": 1, "range": [0.0, 1.0]},
            "decomposition": {"split_x": 1, "split_y": 1, "split_z": 1}
        }
        
        # Generate configuration
        jaxfluids_config = bc_generator.generate_jaxfluids_config(
            flow_conditions=flow_conditions,
            domain_config=domain_config
        )
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        
        config_file = os.path.join(output_dir, "rocket_engine_config.json")
        bc_generator.save_configuration(jaxfluids_config, config_file)
        
        print(f"✅ Saved JAX-Fluids configuration to: {config_file}")
        
        # Generate and save masks
        masks = bc_generator.generate_boundary_masks(
            domain_resolution=(200, 100, 1),
            output_format="numpy"
        )
        
        mask_dir = os.path.join(output_dir, "boundary_masks")
        bc_generator.save_boundary_masks(masks, mask_dir)
        
        print(f"✅ Saved boundary masks to: {mask_dir}")
        
        # Print final summary
        print_face_summary(parser)
        
        print("\n🎉 PROCESSING COMPLETE!")
        print(f"📁 Output directory: {output_dir}")
        print("🔗 Ready for JAX-Fluids simulation!")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 