#!/usr/bin/env python3
"""
Virtual Face Detector
Detects circular inlet/outlet openings and creates virtual boundary faces
Based on the proven circular_face_creator.py implementation
"""

import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import sys

try:
    import pyvista as pv
    import meshio
    import gmsh
    from scipy.spatial import ConvexHull
    from sklearn.cluster import DBSCAN
except ImportError as e:
    raise ImportError(f"Required dependencies missing: {e}")

class VirtualFaceDetector:
    """
    Detects virtual inlet/outlet faces from rocket nozzle mesh geometry
    """
    
    def __init__(self, mesh_file: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the virtual face detector
        
        Args:
            mesh_file: Path to mesh file (.msh format)
            logger: Optional logger instance
        """
        self.mesh_file = Path(mesh_file)
        self.logger = logger or logging.getLogger(__name__)
        
        if not self.mesh_file.exists():
            raise FileNotFoundError(f"Mesh file not found: {self.mesh_file}")
            
        self.inlet_points = None
        self.outlet_points = None
        self.inlet_face = None
        self.outlet_face = None
        
    def detect_virtual_faces(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect circular boundary edges and create virtual faces
        
        Returns:
            Tuple of (inlet_points, outlet_points)
        """
        self.logger.info("Detecting circular boundary edges...")
        
        # Load mesh with meshio
        mesh = meshio.read(str(self.mesh_file))
        self.logger.info(f"Loaded mesh: {len(mesh.points)} points, {len(mesh.cells)} cell blocks")
        
        # Convert to PyVista for analysis
        pv_mesh = pv.read(str(self.mesh_file))
        self.logger.info(f"PyVista mesh: {pv_mesh.n_points} points, {pv_mesh.n_cells} cells")
        
        # Extract boundary edges
        boundary_edges = self._extract_boundary_edges(pv_mesh)
        
        # Cluster boundary points into inlet/outlet
        inlet_points, outlet_points = self._cluster_boundary_points(boundary_edges)
        
        if inlet_points is None or outlet_points is None:
            raise RuntimeError("Failed to detect inlet/outlet boundary points")
            
        self.inlet_points = inlet_points
        self.outlet_points = outlet_points
        
        self.logger.info(f"Detected inlet: {len(inlet_points)} points")
        self.logger.info(f"Detected outlet: {len(outlet_points)} points")
        
        return inlet_points, outlet_points
        
    def create_virtual_faces(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Create virtual circular faces from detected boundary points
        
        Returns:
            Tuple of (inlet_face_data, outlet_face_data)
        """
        if self.inlet_points is None or self.outlet_points is None:
            raise RuntimeError("Must detect virtual faces first")
            
        self.logger.info("Creating virtual circular faces...")
        
        # Fit circles and create faces
        self.inlet_face = self._fit_circle_and_create_face(self.inlet_points, "inlet")
        self.outlet_face = self._fit_circle_and_create_face(self.outlet_points, "outlet")
        
        return self.inlet_face, self.outlet_face
        
    def _extract_boundary_edges(self, mesh: pv.PolyData) -> np.ndarray:
        """Extract boundary edges from mesh"""
        self.logger.debug("Extracting boundary edges...")
        
        # Get mesh edges
        edges = mesh.extract_all_edges()
        
        # Find boundary edges (edges that belong to only one face)
        boundary_points = []
        
        # Simple approach: find points on the boundary
        # For rocket nozzles, boundary points are typically at the extremes
        points = mesh.points
        
        # Find min/max X coordinates (inlet/outlet typically at X extremes)
        x_coords = points[:, 0]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        
        # Points near the extremes are likely boundary points
        tolerance = (x_max - x_min) * 0.05  # 5% tolerance
        
        inlet_candidates = points[x_coords < (x_min + tolerance)]
        outlet_candidates = points[x_coords > (x_max - tolerance)]
        
        return np.vstack([inlet_candidates, outlet_candidates])
        
    def _cluster_boundary_points(self, boundary_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster boundary points into inlet and outlet groups"""
        self.logger.debug("Clustering boundary points...")
        
        # Use X-coordinate to separate inlet (low X) from outlet (high X)
        x_coords = boundary_points[:, 0]
        x_median = np.median(x_coords)
        
        inlet_mask = x_coords < x_median
        outlet_mask = x_coords >= x_median
        
        inlet_points = boundary_points[inlet_mask]
        outlet_points = boundary_points[outlet_mask]
        
        # Use DBSCAN to refine clusters and remove outliers
        if len(inlet_points) > 3:
            inlet_points = self._refine_cluster_with_dbscan(inlet_points)
        if len(outlet_points) > 3:
            outlet_points = self._refine_cluster_with_dbscan(outlet_points)
            
        return inlet_points, outlet_points
        
    def _refine_cluster_with_dbscan(self, points: np.ndarray) -> np.ndarray:
        """Refine point cluster using DBSCAN to remove outliers"""
        if len(points) < 4:
            return points
            
        # Use DBSCAN clustering
        clustering = DBSCAN(eps=50.0, min_samples=3).fit(points)
        labels = clustering.labels_
        
        # Find the largest cluster (main boundary)
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        
        if len(unique_labels) > 0:
            main_cluster_label = unique_labels[np.argmax(counts)]
            main_cluster_points = points[labels == main_cluster_label]
            return main_cluster_points
        else:
            return points
            
    def _fit_circle_and_create_face(self, points: np.ndarray, face_type: str) -> Dict[str, Any]:
        """Fit circle to points and create virtual face data"""
        self.logger.debug(f"Creating {face_type} face...")
        
        # Project points to best-fit plane
        center = np.mean(points, axis=0)
        
        # For rocket nozzles, assume faces are roughly perpendicular to X-axis
        # Find the principal plane
        centered_points = points - center
        
        # SVD to find principal components
        U, S, Vt = np.linalg.svd(centered_points)
        normal = Vt[-1]  # Smallest singular vector = normal to best-fit plane
        
        # Project points to plane
        projected_points = points - np.outer(np.dot(centered_points, normal), normal)
        
        # Fit circle in 2D projection
        # Simple circle fitting using least squares
        circle_center_2d, radius = self._fit_circle_2d(projected_points, normal)
        
        # Create virtual face data
        face_data = {
            "center": circle_center_2d,
            "radius": radius,
            "normal": normal,
            "points": points,
            "projected_points": projected_points,
            "type": face_type
        }
        
        self.logger.info(f"{face_type.capitalize()} face: center={circle_center_2d}, radius={radius:.2f}")
        
        return face_data
        
    def _fit_circle_2d(self, points: np.ndarray, normal: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fit circle to 3D points using 2D projection"""
        # Create orthonormal basis for the plane
        v1 = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
        v1 = v1 - np.dot(v1, normal) * normal
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        
        # Project to 2D
        center_3d = np.mean(points, axis=0)
        points_2d = np.array([
            [np.dot(p - center_3d, v1), np.dot(p - center_3d, v2)]
            for p in points
        ])
        
        # Fit circle in 2D using algebraic method
        x, y = points_2d[:, 0], points_2d[:, 1]
        
        # Set up system: (x-cx)^2 + (y-cy)^2 = r^2
        # Expand: x^2 + y^2 - 2*cx*x - 2*cy*y + cx^2 + cy^2 - r^2 = 0
        # Linear system: Ax = b where x = [cx, cy, cx^2+cy^2-r^2]
        
        A = np.column_stack([2*x, 2*y, np.ones(len(x))])
        b = x**2 + y**2
        
        try:
            result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            cx_2d, cy_2d = result[0], result[1]
            
            # Convert back to 3D
            center_3d = center_3d + cx_2d * v1 + cy_2d * v2
            
            # Calculate radius
            distances = np.linalg.norm(points - center_3d, axis=1)
            radius = np.mean(distances)
            
        except np.linalg.LinAlgError:
            # Fallback: use centroid and average distance
            center_3d = np.mean(points, axis=0)
            distances = np.linalg.norm(points - center_3d, axis=1)
            radius = np.mean(distances)
            
        return center_3d, radius
        
    def get_face_data(self) -> Dict[str, Dict[str, Any]]:
        """Get complete face data for both inlet and outlet"""
        if self.inlet_face is None or self.outlet_face is None:
            raise RuntimeError("Must create virtual faces first")
            
        return {
            "inlet": self.inlet_face,
            "outlet": self.outlet_face
        }
        
    def save_face_data(self, output_dir: str):
        """Save virtual face data to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.inlet_points is not None:
            np.save(output_path / "inlet_points.npy", self.inlet_points)
        if self.outlet_points is not None:
            np.save(output_path / "outlet_points.npy", self.outlet_points)
            
        self.logger.info(f"Virtual face data saved to {output_path}") 