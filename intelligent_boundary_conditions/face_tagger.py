#!/usr/bin/env python3
"""
Face Tagger Module
==================

This module implements intelligent face tagging for rocket nozzle inlet/outlet detection.
Supports both automated heuristic-based tagging and manual interactive selection.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from enum import Enum

# Optional visualization libraries
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

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from .geometry_parser import GeometryParser, GeometryFace
except ImportError:
    from geometry_parser import GeometryParser, GeometryFace

logger = logging.getLogger(__name__)

class TaggingMethod(Enum):
    """Available tagging methods"""
    AUTOMATIC_Z_AXIS = "z_axis_heuristic"
    AUTOMATIC_X_AXIS = "x_axis_heuristic"
    AUTOMATIC_FLOW_DIRECTION = "flow_direction_heuristic"
    AUTOMATIC_CLUSTERING = "clustering_based"
    MANUAL_INTERACTIVE = "manual_interactive"
    MANUAL_SELECTION = "manual_selection"

class RocketNozzleType(Enum):
    """Types of rocket nozzle geometries"""
    CONVERGING_DIVERGING = "converging_diverging"
    SIMPLE_CONVERGING = "simple_converging"
    BELL_NOZZLE = "bell_nozzle"
    CONICAL_NOZZLE = "conical_nozzle"
    AEROSPIKE = "aerospike"

class FaceTagger:
    """
    Intelligent face tagging system for rocket nozzles and internal flow geometries
    """
    
    def __init__(self, geometry_parser: GeometryParser):
        """
        Initialize face tagger
        
        Args:
            geometry_parser: Parsed geometry object
        """
        self.geometry_parser = geometry_parser
        self.faces = geometry_parser.faces
        self.bounds = geometry_parser.geometry_bounds
        
        if not self.faces:
            raise ValueError("No faces found in geometry parser. Parse geometry first.")
        
        # Flow direction heuristics
        self.primary_flow_axis = 'x'  # Default assumption
        self.nozzle_type = RocketNozzleType.CONVERGING_DIVERGING
        
        # Manual selection state
        self.selected_faces = []
        self.interactive_session_active = False
        
    def auto_tag_faces(self, 
                      method: TaggingMethod = TaggingMethod.AUTOMATIC_Z_AXIS,
                      nozzle_type: RocketNozzleType = RocketNozzleType.CONVERGING_DIVERGING,
                      flow_axis: str = 'x',
                      **kwargs) -> Dict[str, List[int]]:
        """
        Automatically tag faces as inlet, outlet, or wall using heuristics
        
        Args:
            method: Tagging method to use
            nozzle_type: Type of rocket nozzle
            flow_axis: Primary flow direction ('x', 'y', or 'z')
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Dictionary mapping tags to lists of face IDs
        """
        
        self.nozzle_type = nozzle_type
        self.primary_flow_axis = flow_axis
        
        logger.info(f"Auto-tagging faces using method: {method.value}")
        logger.info(f"Nozzle type: {nozzle_type.value}, Flow axis: {flow_axis}")
        
        if method == TaggingMethod.AUTOMATIC_Z_AXIS:
            return self._tag_by_axis_extremes('z', **kwargs)
        elif method == TaggingMethod.AUTOMATIC_X_AXIS:
            return self._tag_by_axis_extremes('x', **kwargs)
        elif method == TaggingMethod.AUTOMATIC_FLOW_DIRECTION:
            return self._tag_by_flow_direction(**kwargs)
        elif method == TaggingMethod.AUTOMATIC_CLUSTERING:
            return self._tag_by_clustering(**kwargs)
        else:
            raise ValueError(f"Automatic tagging not supported for method: {method.value}")
    
    def _tag_by_axis_extremes(self, axis: str, inlet_tolerance: float = 0.05, 
                             outlet_tolerance: float = 0.05) -> Dict[str, List[int]]:
        """
        Tag faces based on axis extremes (e.g., inlet at min z, outlet at max z)
        
        Args:
            axis: Axis to use ('x', 'y', or 'z')
            inlet_tolerance: Tolerance for inlet face detection (fraction of domain extent)
            outlet_tolerance: Tolerance for outlet face detection (fraction of domain extent)
        """
        
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_idx = axis_map[axis.lower()]
        
        # Get domain bounds for this axis
        min_coord = self.bounds['min'][axis_idx]
        max_coord = self.bounds['max'][axis_idx]
        extent = max_coord - min_coord
        
        # Define thresholds
        inlet_threshold = min_coord + inlet_tolerance * extent
        outlet_threshold = max_coord - outlet_tolerance * extent
        
        tagged_faces = {'inlet': [], 'outlet': [], 'wall': []}
        
        for face in self.faces:
            face_coord = face.centroid[axis_idx]
            
            if face_coord <= inlet_threshold:
                face.tag = 'inlet'
                tagged_faces['inlet'].append(face.face_id)
            elif face_coord >= outlet_threshold:
                face.tag = 'outlet'
                tagged_faces['outlet'].append(face.face_id)
            else:
                face.tag = 'wall'
                tagged_faces['wall'].append(face.face_id)
        
        logger.info(f"Tagged faces by {axis}-axis: "
                   f"inlet={len(tagged_faces['inlet'])}, "
                   f"outlet={len(tagged_faces['outlet'])}, "
                   f"wall={len(tagged_faces['wall'])}")
        
        return tagged_faces
    
    def _tag_by_flow_direction(self, **kwargs) -> Dict[str, List[int]]:
        """
        Tag faces based on flow direction analysis and normal vectors
        """
        
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        flow_axis_idx = axis_map[self.primary_flow_axis.lower()]
        
        # Flow direction vector (positive along flow axis)
        flow_direction = np.zeros(3)
        flow_direction[flow_axis_idx] = 1.0
        
        tagged_faces = {'inlet': [], 'outlet': [], 'wall': []}
        
        # Analyze face normals relative to flow direction
        for face in self.faces:
            # Dot product of face normal with flow direction
            normal_flow_dot = np.dot(face.normal, flow_direction)
            
            # Face normal alignment threshold
            alignment_threshold = 0.8
            
            if normal_flow_dot < -alignment_threshold:
                # Face normal opposes flow (inlet)
                face.tag = 'inlet'
                tagged_faces['inlet'].append(face.face_id)
            elif normal_flow_dot > alignment_threshold:
                # Face normal aligns with flow (outlet)
                face.tag = 'outlet'
                tagged_faces['outlet'].append(face.face_id)
            else:
                # Face normal perpendicular to flow (wall)
                face.tag = 'wall'
                tagged_faces['wall'].append(face.face_id)
        
        # Refine based on geometric position
        return self._refine_by_position(tagged_faces)
    
    def _tag_by_clustering(self, n_clusters: int = 3, **kwargs) -> Dict[str, List[int]]:
        """
        Tag faces using clustering on face centroids and normals
        """
        
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Falling back to axis-based tagging.")
            return self._tag_by_axis_extremes(self.primary_flow_axis)
        
        # Prepare feature matrix (centroids + normals)
        features = []
        for face in self.faces:
            feature_vector = np.concatenate([face.centroid, face.normal])
            features.append(feature_vector)
        
        features = np.array(features)
        
        # Apply clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        # Analyze clusters to assign tags
        tagged_faces = {'inlet': [], 'outlet': [], 'wall': []}
        
        for cluster_id in range(n_clusters):
            cluster_faces = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            if not cluster_faces:
                continue
            
            # Analyze cluster characteristics
            cluster_centroids = [self.faces[i].centroid for i in cluster_faces]
            cluster_normals = [self.faces[i].normal for i in cluster_faces]
            
            # Determine cluster type based on position and orientation
            tag = self._classify_cluster(cluster_centroids, cluster_normals)
            
            for face_idx in cluster_faces:
                self.faces[face_idx].tag = tag
                tagged_faces[tag].append(self.faces[face_idx].face_id)
        
        return tagged_faces
    
    def _classify_cluster(self, centroids: List[np.ndarray], 
                         normals: List[np.ndarray]) -> str:
        """
        Classify a cluster of faces as inlet, outlet, or wall
        """
        
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        flow_axis_idx = axis_map[self.primary_flow_axis.lower()]
        
        # Calculate average position and normal
        avg_centroid = np.mean(centroids, axis=0)
        avg_normal = np.mean(normals, axis=0)
        
        # Normalize average normal
        avg_normal = avg_normal / np.linalg.norm(avg_normal)
        
        # Check position relative to domain
        min_coord = self.bounds['min'][flow_axis_idx]
        max_coord = self.bounds['max'][flow_axis_idx]
        relative_position = (avg_centroid[flow_axis_idx] - min_coord) / (max_coord - min_coord)
        
        # Check normal orientation
        flow_direction = np.zeros(3)
        flow_direction[flow_axis_idx] = 1.0
        normal_flow_dot = np.dot(avg_normal, flow_direction)
        
        # Classification logic
        if relative_position < 0.2 and normal_flow_dot < -0.5:
            return 'inlet'
        elif relative_position > 0.8 and normal_flow_dot > 0.5:
            return 'outlet'
        else:
            return 'wall'
    
    def _refine_by_position(self, tagged_faces: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """
        Refine tagging based on geometric position constraints
        """
        
        # For rocket nozzles, ensure inlet is upstream and outlet is downstream
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        flow_axis_idx = axis_map[self.primary_flow_axis.lower()]
        
        # Calculate average positions
        inlet_positions = []
        outlet_positions = []
        
        for face_id in tagged_faces['inlet']:
            face = self.geometry_parser.get_face_by_id(face_id)
            if face:
                inlet_positions.append(face.centroid[flow_axis_idx])
        
        for face_id in tagged_faces['outlet']:
            face = self.geometry_parser.get_face_by_id(face_id)
            if face:
                outlet_positions.append(face.centroid[flow_axis_idx])
        
        # Check if inlet is actually upstream of outlet
        if inlet_positions and outlet_positions:
            avg_inlet_pos = np.mean(inlet_positions)
            avg_outlet_pos = np.mean(outlet_positions)
            
            if avg_inlet_pos > avg_outlet_pos:
                # Swap inlet and outlet
                logger.warning("Swapping inlet and outlet based on position analysis")
                tagged_faces['inlet'], tagged_faces['outlet'] = tagged_faces['outlet'], tagged_faces['inlet']
                
                # Update face tags
                for face_id in tagged_faces['inlet']:
                    face = self.geometry_parser.get_face_by_id(face_id)
                    if face:
                        face.tag = 'inlet'
                
                for face_id in tagged_faces['outlet']:
                    face = self.geometry_parser.get_face_by_id(face_id)
                    if face:
                        face.tag = 'outlet'
        
        return tagged_faces
    
    def manual_tag_faces(self, visualization: bool = True) -> Dict[str, List[int]]:
        """
        Manually tag faces using interactive visualization
        
        Args:
            visualization: Whether to use visualization for selection
            
        Returns:
            Dictionary mapping tags to lists of face IDs
        """
        
        if visualization and PYVISTA_AVAILABLE:
            return self._interactive_tagging_pyvista()
        elif visualization and MATPLOTLIB_AVAILABLE:
            return self._interactive_tagging_matplotlib()
        else:
            return self._manual_tagging_console()
    
    def _interactive_tagging_pyvista(self) -> Dict[str, List[int]]:
        """
        Interactive face tagging using PyVista
        """
        
        logger.info("Starting interactive face tagging with PyVista")
        
        # Create PyVista mesh
        vertices = []
        faces_pv = []
        
        for face in self.faces:
            start_idx = len(vertices)
            vertices.extend(face.vertices.tolist())
            
            # Create face connectivity (assuming triangular faces)
            if len(face.vertices) == 3:
                faces_pv.extend([3, start_idx, start_idx + 1, start_idx + 2])
            elif len(face.vertices) == 4:
                faces_pv.extend([4, start_idx, start_idx + 1, start_idx + 2, start_idx + 3])
        
        vertices = np.array(vertices)
        mesh = pv.PolyData(vertices, faces_pv)
        
        # Interactive plotting
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightblue', show_edges=True)
        plotter.add_text("Select faces: \n1. Click faces for inlet (red)\n2. Press 'i' to switch to outlet mode\n3. Click faces for outlet (green)\n4. Press 'w' for wall mode\n5. Press 'q' to finish", 
                        position='upper_left')
        
        # Face selection callback
        selected_cells = []
        current_mode = 'inlet'
        
        def callback(picked):
            if picked.cell_id >= 0:
                selected_cells.append((picked.cell_id, current_mode))
                logger.info(f"Selected face {picked.cell_id} as {current_mode}")
        
        plotter.enable_cell_picking(callback=callback)
        plotter.show()
        
        # Process selections
        tagged_faces = {'inlet': [], 'outlet': [], 'wall': []}
        
        for cell_id, tag in selected_cells:
            if cell_id < len(self.faces):
                self.faces[cell_id].tag = tag
                tagged_faces[tag].append(cell_id)
        
        return tagged_faces
    
    def _interactive_tagging_matplotlib(self) -> Dict[str, List[int]]:
        """
        Interactive face tagging using Matplotlib (simpler 3D view)
        """
        
        logger.info("Starting interactive face tagging with Matplotlib")
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all face centroids
        centroids = np.array([face.centroid for face in self.faces])
        
        colors = ['blue'] * len(self.faces)
        scatter = ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
                           c=colors, s=50, picker=True)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Interactive Face Selection\nClick faces to select')
        
        # Selection state
        selected_faces = []
        current_mode = 'inlet'
        
        def on_pick(event):
            nonlocal current_mode
            ind = event.ind[0]
            
            if ind < len(self.faces):
                selected_faces.append((ind, current_mode))
                
                # Update color
                if current_mode == 'inlet':
                    colors[ind] = 'red'
                elif current_mode == 'outlet':
                    colors[ind] = 'green'
                else:
                    colors[ind] = 'orange'
                
                scatter._facecolors[ind] = colors[ind]
                fig.canvas.draw()
                
                logger.info(f"Selected face {ind} as {current_mode}")
        
        def on_key(event):
            nonlocal current_mode
            if event.key == 'i':
                current_mode = 'inlet'
                ax.set_title(f'Mode: {current_mode.upper()} - Click faces to select')
            elif event.key == 'o':
                current_mode = 'outlet' 
                ax.set_title(f'Mode: {current_mode.upper()} - Click faces to select')
            elif event.key == 'w':
                current_mode = 'wall'
                ax.set_title(f'Mode: {current_mode.upper()} - Click faces to select')
            fig.canvas.draw()
        
        fig.canvas.mpl_connect('pick_event', on_pick)
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        plt.show()
        
        # Process selections
        tagged_faces = {'inlet': [], 'outlet': [], 'wall': []}
        
        for face_idx, tag in selected_faces:
            self.faces[face_idx].tag = tag
            tagged_faces[tag].append(face_idx)
        
        return tagged_faces
    
    def _manual_tagging_console(self) -> Dict[str, List[int]]:
        """
        Manual face tagging via console interface
        """
        
        logger.info("Starting manual face tagging via console")
        
        # Display face information
        print("\nAvailable faces:")
        for i, face in enumerate(self.faces[:20]):  # Show first 20 faces
            print(f"Face {i}: centroid={face.centroid}, area={face.area:.4f}")
        
        if len(self.faces) > 20:
            print(f"... and {len(self.faces) - 20} more faces")
        
        tagged_faces = {'inlet': [], 'outlet': [], 'wall': []}
        
        # Get user input for each category
        for tag in ['inlet', 'outlet', 'wall']:
            while True:
                try:
                    face_ids_str = input(f"\nEnter {tag} face IDs (comma-separated, or 'done'): ")
                    if face_ids_str.lower() == 'done':
                        break
                    
                    face_ids = [int(x.strip()) for x in face_ids_str.split(',') if x.strip()]
                    
                    for face_id in face_ids:
                        if 0 <= face_id < len(self.faces):
                            self.faces[face_id].tag = tag
                            tagged_faces[tag].append(face_id)
                        else:
                            print(f"Warning: Face ID {face_id} out of range")
                    
                except ValueError:
                    print("Invalid input. Please enter comma-separated integers.")
                except KeyboardInterrupt:
                    break
        
        return tagged_faces
    
    def get_tagging_summary(self) -> Dict[str, Any]:
        """
        Get summary of current face tagging state
        """
        
        tagged_counts = {
            'inlet': len([f for f in self.faces if f.tag == 'inlet']),
            'outlet': len([f for f in self.faces if f.tag == 'outlet']),
            'wall': len([f for f in self.faces if f.tag == 'wall']),
            'untagged': len([f for f in self.faces if f.tag is None])
        }
        
        tagged_areas = {
            'inlet': sum(f.area for f in self.faces if f.tag == 'inlet'),
            'outlet': sum(f.area for f in self.faces if f.tag == 'outlet'),
            'wall': sum(f.area for f in self.faces if f.tag == 'wall'),
            'untagged': sum(f.area for f in self.faces if f.tag is None)
        }
        
        return {
            'total_faces': len(self.faces),
            'tagged_counts': tagged_counts,
            'tagged_areas': tagged_areas,
            'tagging_complete': tagged_counts['untagged'] == 0,
            'has_inlet': tagged_counts['inlet'] > 0,
            'has_outlet': tagged_counts['outlet'] > 0
        }
    
    def validate_tagging(self) -> Dict[str, Any]:
        """
        Validate the current face tagging for physical consistency
        """
        
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        summary = self.get_tagging_summary()
        
        # Check for essential boundary conditions
        if not summary['has_inlet']:
            validation_results['errors'].append("No inlet faces tagged")
            validation_results['valid'] = False
        
        if not summary['has_outlet']:
            validation_results['errors'].append("No outlet faces tagged")
            validation_results['valid'] = False
        
        # Check for reasonable area ratios
        if summary['has_inlet'] and summary['has_outlet']:
            inlet_area = summary['tagged_areas']['inlet']
            outlet_area = summary['tagged_areas']['outlet']
            
            if outlet_area > inlet_area * 10:
                validation_results['warnings'].append(
                    f"Outlet area ({outlet_area:.4f}) much larger than inlet area ({inlet_area:.4f})")
            
            if inlet_area > outlet_area * 10:
                validation_results['warnings'].append(
                    f"Inlet area ({inlet_area:.4f}) much larger than outlet area ({outlet_area:.4f})")
        
        # Check for untagged faces
        if summary['tagged_counts']['untagged'] > 0:
            validation_results['warnings'].append(
                f"{summary['tagged_counts']['untagged']} faces remain untagged")
        
        return validation_results 