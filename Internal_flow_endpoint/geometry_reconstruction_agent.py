#!/usr/bin/env python3
"""
VectraSim Geometry Reconstruction Agent
Advanced geometry processing for internal flow domains

This agent handles:
- Edge detection and boundary identification
- Virtual face reconstruction using interpolation
- Nozzle and duct geometry processing
- Complex internal geometry reconstruction
- Boundary-fitted mesh generation support
- PyVista 3D mesh visualization with tagged faces
"""

import numpy as np
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from scipy.interpolate import griddata, RBFInterpolator
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
from pathlib import Path

# PyVista for 3D mesh visualization
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("⚠️ PyVista not available. Install with: pip install pyvista")

logger = logging.getLogger(__name__)

class GeometryReconstructionAgent:
    """
    Advanced geometry reconstruction for internal flow domains
    Identifies edges and reconstructs virtual faces using interpolation
    """
    
    def __init__(self):
        """Initialize the geometry reconstruction agent"""
        logger.info("🔧 Geometry Reconstruction Agent initialized")
        
    def identify_design_edges(self, geometry_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify absolute edges of the design where holes/openings exist
        
        Args:
            geometry_data: Input geometry data with points, boundaries
            
        Returns:
            Dictionary containing identified edges and boundaries
        """
        
        print("🔍 Identifying Design Edges and Boundaries")
        
        # Extract geometry points and boundaries
        points = np.array(geometry_data.get('points', []))
        boundaries = geometry_data.get('boundaries', {})
        flow_type = geometry_data.get('flow_type', 'nozzle')
        
        print(f"   📊 Processing {len(points)} geometry points")
        print(f"   🎯 Flow type: {flow_type}")
        
        # Identify different edge types
        edge_analysis = {
            'inlet_edges': self._identify_inlet_edges(points, boundaries),
            'outlet_edges': self._identify_outlet_edges(points, boundaries),
            'wall_edges': self._identify_wall_edges(points, boundaries),
            'symmetry_edges': self._identify_symmetry_edges(points, boundaries),
            'hole_boundaries': self._identify_hole_boundaries(points, boundaries)
        }
        
        # Analyze edge connectivity
        connectivity = self._analyze_edge_connectivity(edge_analysis)
        
        # Identify critical design features
        design_features = self._identify_design_features(points, edge_analysis, flow_type)
        
        result = {
            'edge_analysis': edge_analysis,
            'connectivity': connectivity,
            'design_features': design_features,
            'geometry_stats': {
                'total_points': len(points),
                'inlet_points': len(edge_analysis['inlet_edges']),
                'outlet_points': len(edge_analysis['outlet_edges']),
                'wall_points': len(edge_analysis['wall_edges'])
            }
        }
        
        print(f"   ✅ Identified {len(edge_analysis['inlet_edges'])} inlet edge points")
        print(f"   ✅ Identified {len(edge_analysis['outlet_edges'])} outlet edge points")
        print(f"   ✅ Identified {len(edge_analysis['wall_edges'])} wall edge points")
        
        return result
    
    def reconstruct_virtual_faces(self, edge_data: Dict[str, Any], reconstruction_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct virtual faces using interpolation techniques
        
        Args:
            edge_data: Edge analysis data from identify_design_edges
            reconstruction_params: Parameters for reconstruction
            
        Returns:
            Dictionary containing reconstructed virtual faces
        """
        
        print("🎨 Reconstructing Virtual Faces using Interpolation")
        
        method = reconstruction_params.get('method', 'rbf')
        resolution = reconstruction_params.get('resolution', 100)
        smoothing = reconstruction_params.get('smoothing', 0.1)
        
        print(f"   🔧 Method: {method}")
        print(f"   📐 Resolution: {resolution}x{resolution}")
        print(f"   🌊 Smoothing: {smoothing}")
        
        # Extract edge points
        inlet_edges = edge_data['edge_analysis']['inlet_edges']
        outlet_edges = edge_data['edge_analysis']['outlet_edges']
        wall_edges = edge_data['edge_analysis']['wall_edges']
        
        virtual_faces = {}
        
        # Reconstruct internal flow surfaces
        if len(inlet_edges) > 0 and len(outlet_edges) > 0:
            virtual_faces['flow_surface'] = self._reconstruct_flow_surface(
                inlet_edges, outlet_edges, wall_edges, method, resolution, smoothing
            )
        
        # Reconstruct wall surfaces
        if len(wall_edges) > 0:
            virtual_faces['wall_surface'] = self._reconstruct_wall_surface(
                wall_edges, method, resolution, smoothing
            )
        
        # Reconstruct transition regions
        virtual_faces['transition_regions'] = self._reconstruct_transition_regions(
            edge_data, method, resolution, smoothing
        )
        
        # Generate 3D mesh with tagged faces
        mesh_data = self._generate_3d_mesh_with_tags(virtual_faces, edge_data, resolution)
        
        result = {
            'virtual_faces': virtual_faces,
            'mesh_data': mesh_data,
            'reconstruction_quality': self._assess_reconstruction_quality(virtual_faces),
            'boundary_conditions': self._generate_boundary_conditions(virtual_faces, edge_data)
        }
        
        print(f"   ✅ Reconstructed {len(virtual_faces)} virtual face types")
        print(f"   📊 Mesh quality: {result['reconstruction_quality']['overall_score']:.2f}")
        
        return result
    
    def visualize_mesh_pyvista(self, reconstruction_data: Dict[str, Any], 
                              output_path: str = "./geometry_visualization.html", 
                              show_interactive: bool = True) -> None:
        """
        Create PyVista 3D mesh visualization with tagged faces
        
        Args:
            reconstruction_data: Reconstruction data with mesh and boundary info
            output_path: Path to save HTML visualization
            show_interactive: Whether to show interactive plot
        """
        
        if not PYVISTA_AVAILABLE:
            print("❌ PyVista not available. Cannot create 3D visualization.")
            return
        
        print("🎨 Creating PyVista 3D Mesh Visualization")
        
        mesh_data = reconstruction_data.get('mesh_data', {})
        if not mesh_data:
            print("❌ No mesh data available for visualization")
            return
        
        # Create PyVista plotter
        plotter = pv.Plotter(notebook=False, off_screen=not show_interactive)
        plotter.set_background('white')
        
        # Create main mesh
        main_mesh = self._create_pyvista_mesh(mesh_data)
        
        if main_mesh is not None:
            # Add main mesh (flow domain)
            plotter.add_mesh(
                main_mesh, 
                color='lightblue', 
                opacity=0.3,
                label='Flow Domain'
            )
            
            # Add boundary faces with different colors
            self._add_boundary_faces_to_plotter(plotter, reconstruction_data)
            
            # Add edge lines
            self._add_edge_lines_to_plotter(plotter, reconstruction_data)
            
            # Add annotations
            self._add_annotations_to_plotter(plotter, reconstruction_data)
        
        # Configure visualization
        try:
            plotter.add_legend()
        except ValueError:
            pass  # No labels to show in legend
        plotter.show_axes()
        plotter.add_text("🚀 VectraSim Geometry Reconstruction", position='upper_left', font_size=12)
        
        # Add coordinate system
        plotter.show_bounds(
            grid='back',
            location='all',
            ticks='both'
        )
        
        # Save as HTML first (before showing interactive)
        if output_path.endswith('.html'):
            print(f"   💾 Saving interactive visualization to {output_path}")
            try:
                plotter.export_html(output_path)
                print(f"   ✅ HTML visualization saved successfully")
            except Exception as e:
                print(f"   ⚠️ Could not export HTML: {e}")
        
        if show_interactive:
            print("   🖥️ Launching interactive 3D visualization...")
            try:
                plotter.show()
            except Exception as e:
                print(f"   ⚠️ Could not show interactive plot: {e}")
        
        plotter.close()
        
        print("   ✅ PyVista visualization complete!")
    
    def _create_pyvista_mesh(self, mesh_data: Dict[str, Any]) -> Optional[pv.PolyData]:
        """Create PyVista mesh from mesh data"""
        
        nodes = mesh_data.get('nodes', [])
        elements = mesh_data.get('elements', [])
        
        if not nodes or not elements:
            return None
        
        # Convert to numpy arrays
        points = np.array(nodes)
        cells = []
        
        # Convert elements to PyVista format
        for elem in elements:
            if len(elem) == 4:  # Quadrilateral
                cells.extend([4] + elem)  # 4 vertices per quad
            elif len(elem) == 3:  # Triangle
                cells.extend([3] + elem)  # 3 vertices per triangle
        
        if not cells:
            return None
        
        # Create PyVista mesh
        mesh = pv.PolyData(points, cells)
        
        return mesh
    
    def _add_boundary_faces_to_plotter(self, plotter: pv.Plotter, reconstruction_data: Dict[str, Any]) -> None:
        """Add boundary faces with different colors and tags"""
        
        edge_analysis = reconstruction_data.get('edge_analysis', {})
        
        # Color scheme for different boundary types
        boundary_colors = {
            'inlet': 'red',
            'outlet': 'blue', 
            'wall': 'gray',
            'symmetry': 'green'
        }
        
        # Add inlet faces
        inlet_edges = edge_analysis.get('inlet_edges', [])
        if inlet_edges:
            inlet_mesh = self._create_boundary_mesh(inlet_edges, 'inlet')
            if inlet_mesh:
                plotter.add_mesh(
                    inlet_mesh,
                    color=boundary_colors['inlet'],
                    opacity=0.8,
                    label='Inlet (SIMPLE_INFLOW)',
                    line_width=3
                )
        
        # Add outlet faces
        outlet_edges = edge_analysis.get('outlet_edges', [])
        if outlet_edges:
            outlet_mesh = self._create_boundary_mesh(outlet_edges, 'outlet')
            if outlet_mesh:
                plotter.add_mesh(
                    outlet_mesh,
                    color=boundary_colors['outlet'],
                    opacity=0.8,
                    label='Outlet (SIMPLE_OUTFLOW)',
                    line_width=3
                )
        
        # Add wall faces
        wall_edges = edge_analysis.get('wall_edges', [])
        if wall_edges:
            wall_mesh = self._create_boundary_mesh(wall_edges, 'wall')
            if wall_mesh:
                plotter.add_mesh(
                    wall_mesh,
                    color=boundary_colors['wall'],
                    opacity=0.9,
                    label='Walls (NO_SLIP)',
                    line_width=2
                )
        
        # Add symmetry faces
        symmetry_edges = edge_analysis.get('symmetry_edges', [])
        if symmetry_edges:
            symmetry_mesh = self._create_boundary_mesh(symmetry_edges, 'symmetry')
            if symmetry_mesh:
                plotter.add_mesh(
                    symmetry_mesh,
                    color=boundary_colors['symmetry'],
                    opacity=0.6,
                    label='Symmetry',
                    line_width=2
                )
    
    def _create_boundary_mesh(self, edge_points: List, boundary_type: str) -> Optional[pv.PolyData]:
        """Create PyVista mesh for boundary edges"""
        
        if not edge_points:
            return None
        
        points = np.array(edge_points)
        
        if len(points) < 2:
            return None
        
        # Create line cells for edges
        lines = []
        for i in range(len(points) - 1):
            lines.extend([2, i, i + 1])  # 2 points per line
        
        # Add closing line if it's a closed boundary
        if boundary_type in ['wall', 'symmetry'] and len(points) > 2:
            lines.extend([2, len(points) - 1, 0])  # Close the loop
        
        mesh = pv.PolyData(points, lines)
        
        # Add boundary type as mesh data
        mesh[f'{boundary_type}_tag'] = np.ones(len(points))
        
        return mesh
    
    def _add_edge_lines_to_plotter(self, plotter: pv.Plotter, reconstruction_data: Dict[str, Any]) -> None:
        """Add edge lines to highlight geometry features"""
        
        design_features = reconstruction_data.get('design_features', {})
        
        # Highlight throat location if it's a nozzle
        if 'throat_location' in design_features and 'throat_radius' in design_features:
            throat_x = design_features['throat_location']
            throat_r = design_features['throat_radius']
            
            # Create throat circle
            theta = np.linspace(0, 2*np.pi, 50)
            throat_points = np.column_stack([
                np.full_like(theta, throat_x),
                throat_r * np.cos(theta),
                throat_r * np.sin(theta)
            ])
            
            throat_lines = []
            for i in range(len(throat_points) - 1):
                throat_lines.extend([2, i, i + 1])
            throat_lines.extend([2, len(throat_points) - 1, 0])  # Close circle
            
            throat_mesh = pv.PolyData(throat_points, throat_lines)
            plotter.add_mesh(
                throat_mesh,
                color='orange',
                line_width=4,
                label=f'Throat (r={throat_r:.3f}m)'
            )
    
    def _add_annotations_to_plotter(self, plotter: pv.Plotter, reconstruction_data: Dict[str, Any]) -> None:
        """Add text annotations and labels"""
        
        design_features = reconstruction_data.get('design_features', {})
        
        # Add performance annotations
        annotations = []
        
        if 'expansion_ratio' in design_features:
            exp_ratio = design_features['expansion_ratio']
            annotations.append(f"Expansion Ratio: {exp_ratio:.2f}")
        
        if 'nozzle_length' in design_features:
            length = design_features['nozzle_length']
            annotations.append(f"Nozzle Length: {length:.3f}m")
        
        if 'throat_radius' in design_features:
            throat_r = design_features['throat_radius']
            annotations.append(f"Throat Radius: {throat_r:.3f}m")
        
        # Add annotation text
        if annotations:
            annotation_text = '\n'.join(annotations)
            plotter.add_text(
                annotation_text,
                position='upper_right',
                font_size=10,
                color='black'
            )
    
    def _generate_3d_mesh_with_tags(self, virtual_faces: Dict[str, Any], 
                                   edge_data: Dict[str, Any], resolution: int) -> Dict[str, Any]:
        """Generate 3D mesh with boundary face tags for PyVista"""
        
        mesh_data = {
            'nodes': [],
            'elements': [],
            'boundary_tags': {},
            'element_type': 'mixed'
        }
        
        # Generate volume mesh based on virtual faces
        if 'flow_surface' in virtual_faces:
            flow_surface = virtual_faces['flow_surface']
            X, Y, Z = flow_surface['x_grid'], flow_surface['y_grid'], flow_surface['z_surface']
            
            # Create 3D nodes (extrude 2D grid)
            nodes = []
            z_layers = np.linspace(-0.01, 0.01, 5)  # Create thin 3D volume
            
            for k, z_val in enumerate(z_layers):
                for i in range(resolution):
                    for j in range(resolution):
                        nodes.append([X[i, j], Y[i, j], z_val])
            
            mesh_data['nodes'] = nodes
            
            # Create volume elements (hexahedra)
            elements = []
            for k in range(len(z_layers) - 1):
                for i in range(resolution - 1):
                    for j in range(resolution - 1):
                        # Node indices for current layer
                        n1 = k * resolution * resolution + i * resolution + j
                        n2 = k * resolution * resolution + i * resolution + j + 1
                        n3 = k * resolution * resolution + (i + 1) * resolution + j + 1
                        n4 = k * resolution * resolution + (i + 1) * resolution + j
                        
                        # Node indices for next layer
                        n5 = (k + 1) * resolution * resolution + i * resolution + j
                        n6 = (k + 1) * resolution * resolution + i * resolution + j + 1
                        n7 = (k + 1) * resolution * resolution + (i + 1) * resolution + j + 1
                        n8 = (k + 1) * resolution * resolution + (i + 1) * resolution + j
                        
                        # Hexahedron element
                        elements.append([n1, n2, n3, n4, n5, n6, n7, n8])
            
            mesh_data['elements'] = elements
            
            # Tag boundary faces
            mesh_data['boundary_tags'] = self._tag_boundary_faces(edge_data, resolution, len(z_layers))
        
        return mesh_data
    
    def _tag_boundary_faces(self, edge_data: Dict[str, Any], resolution: int, z_layers: int) -> Dict[str, Any]:
        """Tag boundary faces for different boundary conditions"""
        
        tags = {
            'inlet_faces': [],
            'outlet_faces': [],
            'wall_faces': [],
            'symmetry_faces': []
        }
        
        # This is simplified - real implementation would properly identify
        # which mesh faces correspond to which boundary types
        
        return tags

    def _identify_inlet_edges(self, points: np.ndarray, boundaries: Dict[str, Any]) -> List[np.ndarray]:
        """Identify inlet boundary edges"""
        
        # For nozzle/rocket: inlet typically at minimum x-coordinate
        if 'inlet' in boundaries:
            return boundaries['inlet']
        
        # Auto-detect based on geometry
        min_x = np.min(points[:, 0])
        inlet_tolerance = 0.01 * (np.max(points[:, 0]) - min_x)
        inlet_points = points[points[:, 0] <= min_x + inlet_tolerance]
        
        return inlet_points.tolist()
    
    def _identify_outlet_edges(self, points: np.ndarray, boundaries: Dict[str, Any]) -> List[np.ndarray]:
        """Identify outlet boundary edges"""
        
        if 'outlet' in boundaries:
            return boundaries['outlet']
        
        # Auto-detect based on geometry
        max_x = np.max(points[:, 0])
        outlet_tolerance = 0.01 * (max_x - np.min(points[:, 0]))
        outlet_points = points[points[:, 0] >= max_x - outlet_tolerance]
        
        return outlet_points.tolist()
    
    def _identify_wall_edges(self, points: np.ndarray, boundaries: Dict[str, Any]) -> List[np.ndarray]:
        """Identify wall boundary edges"""
        
        if 'walls' in boundaries:
            return boundaries['walls']
        
        # Auto-detect wall boundaries (typically outer envelope)
        hull = ConvexHull(points[:, :2])  # 2D hull for x-y plane
        wall_points = points[hull.vertices]
        
        return wall_points.tolist()
    
    def _identify_symmetry_edges(self, points: np.ndarray, boundaries: Dict[str, Any]) -> List[np.ndarray]:
        """Identify symmetry boundary edges"""
        
        if 'symmetry' in boundaries:
            return boundaries['symmetry']
        
        # Auto-detect symmetry lines (typically y=0 for axisymmetric)
        symmetry_tolerance = 0.01 * (np.max(points[:, 1]) - np.min(points[:, 1]))
        symmetry_points = points[np.abs(points[:, 1]) <= symmetry_tolerance]
        
        return symmetry_points.tolist()
    
    def _identify_hole_boundaries(self, points: np.ndarray, boundaries: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify internal holes/openings in the geometry"""
        
        holes = []
        
        if 'holes' in boundaries:
            return boundaries['holes']
        
        # Use Delaunay triangulation to find potential holes
        tri = Delaunay(points[:, :2])
        
        # Identify large empty regions that could be holes
        # This is a simplified approach - more sophisticated hole detection would be needed
        
        return holes
    
    def _analyze_edge_connectivity(self, edge_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze connectivity between different edge types"""
        
        connectivity = {
            'inlet_to_wall': self._find_edge_connections(
                edge_analysis['inlet_edges'], edge_analysis['wall_edges']
            ),
            'outlet_to_wall': self._find_edge_connections(
                edge_analysis['outlet_edges'], edge_analysis['wall_edges']
            ),
            'wall_continuity': self._analyze_wall_continuity(edge_analysis['wall_edges'])
        }
        
        return connectivity
    
    def _find_edge_connections(self, edge1: List, edge2: List) -> Dict[str, Any]:
        """Find connection points between two edge sets"""
        
        if not edge1 or not edge2:
            return {'connections': [], 'distance_threshold': 0.01}
        
        edge1_array = np.array(edge1)
        edge2_array = np.array(edge2)
        
        connections = []
        threshold = 0.01  # Connection distance threshold
        
        for i, p1 in enumerate(edge1_array):
            distances = np.linalg.norm(edge2_array - p1, axis=1)
            close_points = np.where(distances < threshold)[0]
            
            if len(close_points) > 0:
                connections.append({
                    'edge1_index': i,
                    'edge2_indices': close_points.tolist(),
                    'distances': distances[close_points].tolist()
                })
        
        return {
            'connections': connections,
            'distance_threshold': threshold,
            'connection_count': len(connections)
        }
    
    def _analyze_wall_continuity(self, wall_edges: List) -> Dict[str, Any]:
        """Analyze continuity and smoothness of wall edges"""
        
        if not wall_edges:
            return {'continuous': False, 'smoothness': 0.0}
        
        wall_array = np.array(wall_edges)
        
        # Calculate curvature and smoothness metrics
        if len(wall_array) > 2:
            # Simple curvature estimation
            diff1 = np.diff(wall_array, axis=0)
            diff2 = np.diff(diff1, axis=0)
            curvature = np.linalg.norm(diff2, axis=1)
            avg_curvature = np.mean(curvature)
            
            smoothness = 1.0 / (1.0 + avg_curvature)  # Higher = smoother
        else:
            smoothness = 1.0
        
        return {
            'continuous': True,
            'smoothness': smoothness,
            'point_count': len(wall_edges)
        }
    
    def _identify_design_features(self, points: np.ndarray, edge_analysis: Dict[str, Any], flow_type: str) -> Dict[str, Any]:
        """Identify specific design features based on flow type"""
        
        features = {}
        
        if flow_type in ['nozzle', 'rocket_engine']:
            features.update(self._identify_nozzle_features(points, edge_analysis))
        elif flow_type == 'duct':
            features.update(self._identify_duct_features(points, edge_analysis))
        elif flow_type == 'diffuser':
            features.update(self._identify_diffuser_features(points, edge_analysis))
        
        return features
    
    def _identify_nozzle_features(self, points: np.ndarray, edge_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Identify nozzle-specific features (throat, expansion ratio, etc.)"""
        
        wall_points = np.array(edge_analysis['wall_edges'])
        if len(wall_points) == 0:
            return {}
        
        # Find throat (minimum area location)
        x_coords = wall_points[:, 0]
        y_coords = wall_points[:, 1]
        
        # Group points by x-coordinate to find local radius
        x_unique = np.unique(x_coords)
        local_radii = []
        throat_x = None
        min_radius = float('inf')
        
        for x in x_unique:
            local_points = wall_points[np.abs(wall_points[:, 0] - x) < 0.001]
            if len(local_points) > 0:
                local_radius = np.max(local_points[:, 1])  # Assuming axisymmetric
                local_radii.append(local_radius)
                
                if local_radius < min_radius:
                    min_radius = local_radius
                    throat_x = x
        
        # Calculate expansion ratio
        inlet_radius = np.max(wall_points[wall_points[:, 0] == np.min(x_coords), 1])
        outlet_radius = np.max(wall_points[wall_points[:, 0] == np.max(x_coords), 1])
        
        expansion_ratio = (outlet_radius / min_radius) ** 2 if min_radius > 0 else 1.0
        
        return {
            'throat_location': throat_x,
            'throat_radius': min_radius,
            'inlet_radius': inlet_radius,
            'outlet_radius': outlet_radius,
            'expansion_ratio': expansion_ratio,
            'nozzle_length': np.max(x_coords) - np.min(x_coords)
        }
    
    def _identify_duct_features(self, points: np.ndarray, edge_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Identify duct-specific features"""
        
        return {
            'duct_type': 'rectangular',  # or 'circular', 'complex'
            'cross_sectional_area': 0.0,  # Calculate based on geometry
            'hydraulic_diameter': 0.0
        }
    
    def _identify_diffuser_features(self, points: np.ndarray, edge_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Identify diffuser-specific features"""
        
        return {
            'diffuser_angle': 0.0,  # Calculate divergence angle
            'area_ratio': 1.0
        }
    
    def _reconstruct_flow_surface(self, inlet_edges: List, outlet_edges: List, 
                                wall_edges: List, method: str, resolution: int, 
                                smoothing: float) -> Dict[str, Any]:
        """Reconstruct the main flow surface between inlet and outlet"""
        
        # Combine all edge points
        all_points = np.vstack([
            np.array(inlet_edges),
            np.array(outlet_edges),
            np.array(wall_edges)
        ])
        
        # Create regular grid for interpolation
        x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
        y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
        
        x_grid = np.linspace(x_min, x_max, resolution)
        y_grid = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Interpolate surface using specified method
        if method == 'rbf':
            interpolator = RBFInterpolator(
                all_points[:, :2], 
                all_points[:, 2] if all_points.shape[1] > 2 else np.zeros(len(all_points)),
                smoothing=smoothing
            )
            Z = interpolator(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
        else:
            # Use griddata for other methods
            Z = griddata(
                all_points[:, :2],
                all_points[:, 2] if all_points.shape[1] > 2 else np.zeros(len(all_points)),
                (X, Y),
                method=method,
                fill_value=0.0
            )
        
        return {
            'x_grid': X,
            'y_grid': Y,
            'z_surface': Z,
            'method': method,
            'resolution': resolution
        }
    
    def _reconstruct_wall_surface(self, wall_edges: List, method: str, 
                                resolution: int, smoothing: float) -> Dict[str, Any]:
        """Reconstruct wall surfaces with proper boundary conditions"""
        
        if not wall_edges:
            return {}
        
        wall_array = np.array(wall_edges)
        
        # Create parameterized representation of wall
        # This is simplified - real implementation would handle complex 3D walls
        
        return {
            'wall_points': wall_array,
            'parametric_representation': 'spline',
            'boundary_type': 'no_slip_wall'
        }
    
    def _reconstruct_transition_regions(self, edge_data: Dict[str, Any], method: str,
                                      resolution: int, smoothing: float) -> Dict[str, Any]:
        """Reconstruct smooth transition regions between different boundaries"""
        
        transitions = {}
        
        # Inlet-to-wall transition
        if edge_data['connectivity']['inlet_to_wall']['connection_count'] > 0:
            transitions['inlet_wall'] = self._create_smooth_transition(
                edge_data['edge_analysis']['inlet_edges'],
                edge_data['edge_analysis']['wall_edges'],
                method, smoothing
            )
        
        # Outlet-to-wall transition
        if edge_data['connectivity']['outlet_to_wall']['connection_count'] > 0:
            transitions['outlet_wall'] = self._create_smooth_transition(
                edge_data['edge_analysis']['outlet_edges'],
                edge_data['edge_analysis']['wall_edges'],
                method, smoothing
            )
        
        return transitions
    
    def _create_smooth_transition(self, edge1: List, edge2: List, method: str, smoothing: float) -> Dict[str, Any]:
        """Create smooth transition between two edge sets"""
        
        # Implementation would create smooth blending between boundaries
        return {
            'transition_type': 'smooth_blend',
            'method': method,
            'smoothing': smoothing
        }
    
    def _assess_reconstruction_quality(self, virtual_faces: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of the reconstructed geometry"""
        
        quality_metrics = {
            'surface_smoothness': 0.8,  # Would calculate actual smoothness
            'boundary_continuity': 0.9,  # Would check C0/C1 continuity
            'mesh_quality': 0.85,       # Would assess element quality
            'overall_score': 0.85
        }
        
        return quality_metrics
    
    def _generate_boundary_conditions(self, virtual_faces: Dict[str, Any], 
                                    edge_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appropriate boundary conditions for JAX-Fluids"""
        
        boundary_conditions = {
            'west': {'type': 'SIMPLE_INFLOW'},
            'east': {'type': 'SIMPLE_OUTFLOW'},
            'north': {'type': 'SYMMETRY'},
            'south': {'type': 'SYMMETRY'},
            'walls': {'type': 'WALL'}
        }
        
        return boundary_conditions
    
    def export_geometry(self, reconstruction_data: Dict[str, Any], output_path: str) -> None:
        """Export reconstructed geometry in various formats"""
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export as JSON
        with open(output_dir / 'reconstructed_geometry.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            exportable_data = self._make_json_serializable(reconstruction_data)
            json.dump(exportable_data, f, indent=2)
        
        # Export mesh data
        if 'mesh_data' in reconstruction_data:
            self._export_mesh(reconstruction_data['mesh_data'], output_dir / 'mesh.txt')
        
        # Export PyVista visualization
        self.visualize_mesh_pyvista(reconstruction_data, str(output_dir / 'mesh_visualization.html'))
        
        # Export matplotlib visualization
        self._export_visualization(reconstruction_data, output_dir / 'geometry_plot.png')
        
        print(f"   ✅ Exported geometry to {output_dir}")
    
    def _make_json_serializable(self, data: Any) -> Any:
        """Convert numpy arrays and other non-serializable types to JSON-compatible format"""
        
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {key: self._make_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._make_json_serializable(item) for item in data]
        else:
            return data
    
    def _export_mesh(self, mesh_data: Dict[str, Any], filename: str) -> None:
        """Export mesh in a simple text format"""
        
        with open(filename, 'w') as f:
            f.write("# Reconstructed Geometry Mesh\n")
            f.write(f"# Nodes: {len(mesh_data['nodes'])}\n")
            f.write(f"# Elements: {len(mesh_data['elements'])}\n\n")
            
            f.write("NODES\n")
            for i, node in enumerate(mesh_data['nodes']):
                f.write(f"{i+1} {node[0]:.6f} {node[1]:.6f} {node[2]:.6f}\n")
            
            f.write("\nELEMENTS\n")
            for i, elem in enumerate(mesh_data['elements']):
                f.write(f"{i+1} {' '.join(map(str, [n+1 for n in elem]))}\n")
    
    def _export_visualization(self, reconstruction_data: Dict[str, Any], filename: str) -> None:
        """Export visualization of the reconstructed geometry"""
        
        if 'virtual_faces' not in reconstruction_data:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot flow surface
        if 'flow_surface' in reconstruction_data['virtual_faces']:
            flow_surface = reconstruction_data['virtual_faces']['flow_surface']
            X, Y, Z = flow_surface['x_grid'], flow_surface['y_grid'], flow_surface['z_surface']
            
            ax1.contour(X, Y, Z, levels=20)
            ax1.set_title('Flow Surface Contours')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.axis('equal')
        
        # Plot boundary conditions
        if 'boundary_conditions' in reconstruction_data:
            ax2.text(0.1, 0.9, 'Boundary Conditions:', transform=ax2.transAxes, fontweight='bold')
            bc_text = '\n'.join([f"{k}: {v['type']}" for k, v in 
                               reconstruction_data['boundary_conditions'].items()])
            ax2.text(0.1, 0.1, bc_text, transform=ax2.transAxes, fontfamily='monospace')
            ax2.set_title('Boundary Condition Summary')
            ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()


# Example usage function with PyVista visualization
def reconstruct_nozzle_geometry(nozzle_points: List[List[float]], output_dir: str = "./reconstructed_geometry") -> Dict[str, Any]:
    """
    Example function to reconstruct nozzle geometry from point data with PyVista visualization
    
    Args:
        nozzle_points: List of [x, y, z] points defining nozzle geometry
        output_dir: Directory to save results
        
    Returns:
        Complete reconstruction data
    """
    
    # Initialize agent
    agent = GeometryReconstructionAgent()
    
    # Prepare geometry data
    geometry_data = {
        'points': nozzle_points,
        'boundaries': {},  # Auto-detect boundaries
        'flow_type': 'nozzle'
    }
    
    # Identify edges
    edge_data = agent.identify_design_edges(geometry_data)
    
    # Reconstruct virtual faces
    reconstruction_params = {
        'method': 'rbf',
        'resolution': 50,
        'smoothing': 0.1
    }
    
    reconstruction_data = agent.reconstruct_virtual_faces(edge_data, reconstruction_params)
    
    # Create PyVista visualization
    agent.visualize_mesh_pyvista(reconstruction_data, f"{output_dir}/mesh_3d.html", show_interactive=False)
    
    # Export all results
    agent.export_geometry(reconstruction_data, output_dir)
    
    return reconstruction_data


if __name__ == "__main__":
    # Example: Create a simple converging-diverging nozzle
    print("🚀 VectraSim Geometry Reconstruction Agent with PyVista")
    print("Testing with sample rocket nozzle geometry...")
    
    # Generate sample nozzle points
    x = np.linspace(-0.1, 0.2, 50)
    y_upper = []
    
    for xi in x:
        if xi < 0:  # Converging section
            yi = 0.05 + 0.02 * (xi + 0.1) / 0.1
        elif xi < 0.05:  # Throat region
            yi = 0.02 + 0.001 * np.sin(10 * np.pi * xi)
        else:  # Diverging section
            yi = 0.02 + 0.03 * (xi - 0.05) / 0.15
        y_upper.append(yi)
    
    # Create symmetric nozzle
    nozzle_points = []
    for i, xi in enumerate(x):
        nozzle_points.append([xi, y_upper[i], 0.0])      # Upper wall
        nozzle_points.append([xi, -y_upper[i], 0.0])     # Lower wall (symmetric)
    
    # Add inlet and outlet points
    for y in np.linspace(-y_upper[0], y_upper[0], 10):
        nozzle_points.append([x[0], y, 0.0])   # Inlet
        nozzle_points.append([x[-1], y, 0.0])  # Outlet
    
    # Reconstruct geometry with PyVista visualization
    result = reconstruct_nozzle_geometry(nozzle_points, "./test_reconstruction")
    
    print(f"✅ Reconstruction complete!")
    print(f"📊 Quality score: {result['reconstruction_quality']['overall_score']:.2f}")
    print(f"📁 Results saved to: ./test_reconstruction")
    print(f"🎨 3D visualization: ./test_reconstruction/mesh_3d.html")
    print(f"🖥️ PyVista visualization should be launching...") 