#!/usr/bin/env python3
"""
Circular Face Creator
====================

Find Circular Openings and Create Virtual Faces
Specifically targets the circular edges of hollow openings in rocket nozzles

Copied from the working implementation: intelligent_boundary_conditions/working/circular_face_creator.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
from typing import Tuple, Optional, Dict, Any

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False

try:
    from scipy.spatial import ConvexHull
    from sklearn.cluster import DBSCAN
    SCIPY_SKLEARN_AVAILABLE = True
except ImportError:
    SCIPY_SKLEARN_AVAILABLE = False

class CircularFaceCreator:
    """
    Professional circular face creator for internal flow geometries
    """
    
    def __init__(self, verbose: bool = True):
        """Initialize circular face creator"""
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Check dependencies
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista not available")
        if not MESHIO_AVAILABLE:
            raise ImportError("Meshio not available")
        if not SCIPY_SKLEARN_AVAILABLE:
            self.logger.warning("SciPy/Sklearn not available - some features may be limited")

    def find_circular_boundary_edges(self, mesh_file: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Find the actual circular edges at the inlet and outlet openings"""
        
        if self.verbose:
            self.logger.info("🔍 Finding circular boundary edges...")
        
        # Load with meshio first
        mesh = meshio.read(mesh_file)
        if self.verbose:
            self.logger.info(f"   Original mesh: {len(mesh.points)} points, {len(mesh.cells)} cell blocks")
        
        pv_mesh = pv.from_meshio(mesh)
        surface = pv_mesh.extract_surface()
        if self.verbose:
            self.logger.info(f"   Surface mesh: {surface.n_points} points, {surface.n_cells} cells")
        
        # Method 1: Try standard boundary edge extraction
        boundary_edges = surface.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=False,
            feature_edges=False,
            manifold_edges=False
        )
        
        if self.verbose:
            self.logger.info(f"   Method 1 - Standard boundary edges: {boundary_edges.n_points} points")
        
        # Method 2: Try with different feature edge settings
        if boundary_edges.n_points == 0:
            if self.verbose:
                self.logger.info("   Trying method 2 - Non-manifold edges...")
            boundary_edges = surface.extract_feature_edges(
                boundary_edges=True,
                non_manifold_edges=True,
                feature_edges=True,
                manifold_edges=False
            )
            if self.verbose:
                self.logger.info(f"   Method 2 - Non-manifold edges: {boundary_edges.n_points} points")
        
        # Method 3: Try to find "holes" by analyzing mesh connectivity
        if boundary_edges.n_points == 0:
            if self.verbose:
                self.logger.info("   Trying method 3 - Analyzing mesh topology...")
            
            # Get all points and analyze X-coordinate distribution
            all_points = surface.points
            x_coords = all_points[:, 0]
            
            # Find points at extreme X positions (likely inlet/outlet regions)
            x_min, x_max = x_coords.min(), x_coords.max()
            x_range = x_max - x_min
            
            # Points near the ends
            inlet_threshold = x_min + 0.05 * x_range
            outlet_threshold = x_max - 0.05 * x_range
            
            inlet_candidates = all_points[x_coords < inlet_threshold]
            outlet_candidates = all_points[x_coords > outlet_threshold]
            
            if self.verbose:
                self.logger.info(f"   Found {len(inlet_candidates)} inlet candidate points")
                self.logger.info(f"   Found {len(outlet_candidates)} outlet candidate points")
            
            # For each region, find the points that form a circle
            inlet_points = self._find_circular_pattern(inlet_candidates, "Inlet")
            outlet_points = self._find_circular_pattern(outlet_candidates, "Outlet")
            
            if inlet_points is not None and outlet_points is not None:
                if self.verbose:
                    self.logger.info(f"   Successfully found circular patterns!")
                return inlet_points, outlet_points
        
        # Method 4: If we found boundary edges, process them
        if boundary_edges.n_points > 0:
            edge_points = boundary_edges.points
            if self.verbose:
                self.logger.info(f"   Found {len(edge_points)} boundary edge points")
            
            # Cluster points by position to find the two circular openings
            x_coords = edge_points[:, 0]
            
            # Find the two extreme X regions (inlet and outlet)
            x_min, x_max = x_coords.min(), x_coords.max()
            x_range = x_max - x_min
            
            # Points near inlet (low X)
            inlet_mask = x_coords < (x_min + 0.1 * x_range)
            inlet_points = edge_points[inlet_mask]
            
            # Points near outlet (high X) 
            outlet_mask = x_coords > (x_max - 0.1 * x_range)
            outlet_points = edge_points[outlet_mask]
            
            if self.verbose:
                self.logger.info(f"   Inlet region: {len(inlet_points)} points at X≈{inlet_points[:, 0].mean():.1f}")
                self.logger.info(f"   Outlet region: {len(outlet_points)} points at X≈{outlet_points[:, 0].mean():.1f}")
            
            return inlet_points, outlet_points
        
        if self.verbose:
            self.logger.error("❌ No boundary edges found with any method")
        return None, None

    def _find_circular_pattern(self, points: np.ndarray, region_name: str) -> Optional[np.ndarray]:
        """Find points that form a circular pattern"""
        if len(points) < 10:
            return None
                
        # Project to Y-Z plane
        yz_points = points[:, 1:]
        center_y = yz_points[:, 0].mean()
        center_z = yz_points[:, 1].mean()
        center = np.array([center_y, center_z])
        
        # Calculate distances from center
        distances = np.linalg.norm(yz_points - center, axis=1)
        
        # Find points that are roughly the same distance from center (forming a circle)
        median_dist = np.median(distances)
        tolerance = median_dist * 0.2  # 20% tolerance
        
        circle_mask = np.abs(distances - median_dist) < tolerance
        circle_points = points[circle_mask]
        
        if self.verbose:
            self.logger.info(f"   {region_name}: {len(circle_points)} points form circle pattern")
            self.logger.info(f"   {region_name}: radius ≈ {median_dist:.1f}")
        
        return circle_points if len(circle_points) > 20 else None

    def fit_circle_and_create_face(self, boundary_points: np.ndarray, face_type: str = "inlet") -> Optional[Dict[str, Any]]:
        """Fit a circle to boundary points and create a triangulated face"""
        
        if self.verbose:
            self.logger.info(f"🔧 Creating circular {face_type} face...")
        
        if len(boundary_points) < 10:
            if self.verbose:
                self.logger.error(f"❌ Not enough boundary points for {face_type}")
            return None
        
        # Get the average X position for the face plane
        x_pos = boundary_points[:, 0].mean()
        
        # Project to Y-Z plane for circle fitting
        yz_points = boundary_points[:, 1:]  # Y,Z coordinates
        
        # More robust circle fitting using least squares
        center_y, center_z, radius = self._fit_circle_least_squares(yz_points)
        
        # Verify the fit quality
        distances = np.sqrt((yz_points[:, 0] - center_y)**2 + (yz_points[:, 1] - center_z)**2)
        fit_error = np.std(distances - radius)
        
        if self.verbose:
            self.logger.info(f"   Circle center: Y={center_y:.1f}, Z={center_z:.1f}")
            self.logger.info(f"   Circle radius: {radius:.1f}")
            self.logger.info(f"   Fit quality: σ={fit_error:.2f} (lower is better)")
        
        # Create a high-quality circular face with more triangles for smoothness
        n_segments = 64  # More segments for smoother circle
        angles = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
        
        # Create circle points on the exact fitted circle
        circle_points = []
        for angle in angles:
            y = center_y + radius * np.cos(angle)
            z = center_z + radius * np.sin(angle)
            circle_points.append([x_pos, y, z])
        
        circle_points = np.array(circle_points)
        
        # Create triangular faces from center to circle edge (fan triangulation)
        triangles = []
        center_point = np.array([x_pos, center_y, center_z])
        
        for i in range(n_segments):
            j = (i + 1) % n_segments
            triangle = np.array([
                center_point,
                circle_points[i],
                circle_points[j]
            ])
            triangles.append(triangle)
        
        face_data = {
            'triangles': np.array(triangles),
            'center': center_point,
            'radius': radius,
            'x_position': x_pos,
            'boundary_points': boundary_points,
            'fit_error': fit_error,
            'n_triangles': len(triangles),
            'face_type': face_type,
            'normal': np.array([1.0, 0.0, 0.0])  # X-normal for typical rocket nozzle
        }
        
        if self.verbose:
            self.logger.info(f"   Created {len(triangles)} triangular faces")
        
        return face_data

    def _fit_circle_least_squares(self, points_2d: np.ndarray) -> Tuple[float, float, float]:
        """Fit circle using least squares method"""
        x, y = points_2d[:, 0], points_2d[:, 1]
        
        # Set up least squares problem: (x-cx)² + (y-cy)² = r²
        # Rearranged: x² + y² - 2*cx*x - 2*cy*y + cx² + cy² - r² = 0
        # Linear form: -2*cx*x - 2*cy*y + (cx² + cy² - r²) = -(x² + y²)
        A = np.column_stack([x, y, np.ones(len(x))])
        b = x**2 + y**2
        
        # Solve for [2*cx, 2*cy, -(cx² + cy² - r²)]
        try:
            coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
            cx = coeffs[0] / 2
            cy = coeffs[1] / 2
            r = np.sqrt(coeffs[2] + cx**2 + cy**2)
            return cx, cy, r
        except:
            # Fallback to simple center calculation
            cx = np.mean(x)
            cy = np.mean(y)
            r = np.mean(np.sqrt((x - cx)**2 + (y - cy)**2))
            return cx, cy, r

    def create_mesh_with_virtual_faces_only(self, original_mesh_file: str, inlet_face_data: Dict, outlet_face_data: Dict):
        """Create mesh with ONLY the original surface + the 2 virtual circular faces"""
        
        if self.verbose:
            self.logger.info("🔄 Creating mesh with virtual circular faces...")
        
        # Load original surface
        mesh = meshio.read(original_mesh_file)
        pv_mesh = pv.from_meshio(mesh)
        surface = pv_mesh.extract_surface()
        
        # All original faces are walls
        wall_faces = surface.n_cells
        face_types = np.ones(wall_faces)  # All walls = 1
        
        combined_meshes = [surface]
        
        # Add inlet face if available
        if inlet_face_data is not None:
            inlet_triangles = inlet_face_data['triangles']
            inlet_points = inlet_triangles.reshape(-1, 3)
            inlet_faces = []
            
            for i in range(0, len(inlet_points), 3):
                inlet_faces.append([3, i, i+1, i+2])
            
            inlet_mesh = pv.PolyData(inlet_points, np.array(inlet_faces))
            combined_meshes.append(inlet_mesh)
            
            # Add inlet face types
            inlet_face_types = np.zeros(inlet_mesh.n_cells)  # Inlet = 0
            face_types = np.concatenate([face_types, inlet_face_types])
        
        # Add outlet face if available
        if outlet_face_data is not None:
            outlet_triangles = outlet_face_data['triangles']
            outlet_points = outlet_triangles.reshape(-1, 3)
            outlet_faces = []
            
            for i in range(0, len(outlet_points), 3):
                outlet_faces.append([3, i, i+1, i+2])
            
            outlet_mesh = pv.PolyData(outlet_points, np.array(outlet_faces))
            combined_meshes.append(outlet_mesh)
            
            # Add outlet face types
            outlet_face_types = np.full(outlet_mesh.n_cells, 2)  # Outlet = 2
            face_types = np.concatenate([face_types, outlet_face_types])
        
        # Combine all meshes
        combined_mesh = combined_meshes[0]
        for mesh_part in combined_meshes[1:]:
            combined_mesh = combined_mesh.merge(mesh_part)
        
        # Add face type data
        combined_mesh['face_type'] = face_types
        
        return combined_mesh

    def visualize_circular_faces(self, combined_mesh, inlet_data=None, outlet_data=None):
        """Visualize with focus on the circular virtual faces"""
        
        if self.verbose:
            self.logger.info("🎮 Creating visualization...")
        
        plotter = pv.Plotter(window_size=[1600, 1000])
        
        # Add mesh with face type coloring
        plotter.add_mesh(
            combined_mesh,
            scalars='face_type',
            cmap=['red', 'lightgray', 'green'],  # 0=red, 1=gray, 2=green
            show_edges=False,  # Cleaner look without edges
            opacity=0.9
        )
        
        # Add the circular virtual faces with enhanced highlighting
        if inlet_data and 'triangles' in inlet_data:
            # Create inlet face mesh for highlighting
            inlet_triangles = inlet_data['triangles']
            inlet_points = inlet_triangles.reshape(-1, 3)
            inlet_faces = []
            
            for i in range(0, len(inlet_points), 3):
                inlet_faces.append([3, i, i+1, i+2])
            
            inlet_mesh = pv.PolyData(inlet_points, np.array(inlet_faces))
            
            # Add inlet with bright red and edge highlighting
            plotter.add_mesh(
                inlet_mesh,
                color='red',
                opacity=0.95,
                show_edges=True,
                edge_color='darkred',
                line_width=2
            )
            
            # Add inlet center point
            plotter.add_points(
                np.array([inlet_data['center']]),
                color='darkred',
                point_size=15,
                render_points_as_spheres=True
            )
        
        if outlet_data and 'triangles' in outlet_data:
            # Create outlet face mesh for highlighting
            outlet_triangles = outlet_data['triangles']
            outlet_points = outlet_triangles.reshape(-1, 3)
            outlet_faces = []
            
            for i in range(0, len(outlet_points), 3):
                outlet_faces.append([3, i, i+1, i+2])
            
            outlet_mesh = pv.PolyData(outlet_points, np.array(outlet_faces))
            
            # Add outlet with bright green and edge highlighting
            plotter.add_mesh(
                outlet_mesh,
                color='green',
                opacity=0.95,
                show_edges=True,
                edge_color='darkgreen',
                line_width=2
            )
            
            # Add outlet center point
            plotter.add_points(
                np.array([outlet_data['center']]),
                color='darkgreen',
                point_size=15,
                render_points_as_spheres=True
            )
        
        # Set optimal camera position to see both inlet and outlet
        plotter.camera_position = 'iso'
        plotter.add_axes()
        plotter.set_background('navy')
        
        if self.verbose:
            self.logger.info("✅ Visualization ready!")
        
        plotter.show()

    def process_mesh(self, mesh_file: str) -> Dict[str, Any]:
        """
        Main processing method - finds circular faces in mesh
        
        Args:
            mesh_file: Path to mesh file
            
        Returns:
            Dictionary with inlet/outlet face data
        """
        if not os.path.exists(mesh_file):
            raise FileNotFoundError(f"Mesh file not found: {mesh_file}")
        
        # Step 1: Find circular boundary edges
        inlet_points, outlet_points = self.find_circular_boundary_edges(mesh_file)
        
        if inlet_points is None or outlet_points is None:
            raise RuntimeError("Failed to detect inlet/outlet boundaries")
        
        # Step 2: Create virtual faces
        inlet_face_data = self.fit_circle_and_create_face(inlet_points, "inlet")
        outlet_face_data = self.fit_circle_and_create_face(outlet_points, "outlet")
        
        if inlet_face_data is None or outlet_face_data is None:
            raise RuntimeError("Failed to create virtual boundary faces")
        
        result = {
            "inlet_face": inlet_face_data,
            "outlet_face": outlet_face_data,
            "inlet_points": inlet_points,
            "outlet_points": outlet_points,
            "mesh_file": mesh_file,
            "processing_success": True
        }
        
        if self.verbose:
            self.logger.info("✅ Circular face processing completed successfully")
        
        return result

def main():
    """Main function for standalone execution"""
    
    print("🚀 CIRCULAR OPENING FACE CREATOR")
    print("=" * 60)
    
    mesh_file = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\mesh\Rocket Engine.msh"
    
    if not os.path.exists(mesh_file):
        print(f"❌ Mesh file not found: {mesh_file}")
        return
    
    try:
        creator = CircularFaceCreator()
        result = creator.process_mesh(mesh_file)
        
        print("✅ Circular virtual faces created successfully!")
        print(f"Inlet: center={result['inlet_face']['center']}, radius={result['inlet_face']['radius']:.1f}")
        print(f"Outlet: center={result['outlet_face']['center']}, radius={result['outlet_face']['radius']:.1f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 