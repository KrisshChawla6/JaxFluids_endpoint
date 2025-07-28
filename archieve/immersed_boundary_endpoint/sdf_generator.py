"""
Signed Distance Function Generator

This module computes signed distance functions from mesh geometry
using efficient algorithms suitable for immersed boundary methods.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Callable
from scipy.spatial import cKDTree
import logging
try:
    from .mesh_processor import GmshProcessor
except ImportError:
    from mesh_processor import GmshProcessor

logger = logging.getLogger(__name__)


class SignedDistanceFunction:
    """
    Computes signed distance functions from triangulated surface meshes.
    
    Uses efficient algorithms including:
    - KDTree-based nearest neighbor search
    - Triangle-point distance computation
    - Normal-based inside/outside determination
    - JAX-compatible implementations for GPU acceleration
    """
    
    def __init__(self, mesh_processor: GmshProcessor):
        """
        Initialize SDF generator.
        
        Args:
            mesh_processor: Processed mesh data
        """
        self.mesh_processor = mesh_processor
        self.surface_triangles = None
        self.triangle_normals = None
        self.triangle_centers = None
        self.kdtree = None
        
        if not mesh_processor.surface_triangles:
            raise ValueError("Mesh processor must have surface triangles. Call read_mesh() first.")
        
        self._prepare_geometry()
        logger.info("Initialized SignedDistanceFunction")
    
    def _prepare_geometry(self) -> None:
        """Prepare geometric data for SDF computation."""
        self.surface_triangles = np.array(self.mesh_processor.surface_triangles)
        
        # Compute triangle normals and centers
        self.triangle_normals = []
        self.triangle_centers = []
        
        for triangle in self.surface_triangles:
            # Compute normal using cross product
            v1 = triangle[1] - triangle[0]
            v2 = triangle[2] - triangle[0]
            normal = np.cross(v1, v2)
            normal = normal / (np.linalg.norm(normal) + 1e-12)  # Normalize
            
            # Compute center
            center = np.mean(triangle, axis=0)
            
            self.triangle_normals.append(normal)
            self.triangle_centers.append(center)
        
        self.triangle_normals = np.array(self.triangle_normals)
        self.triangle_centers = np.array(self.triangle_centers)
        
        # Build KDTree for fast nearest neighbor queries
        all_vertices = self.surface_triangles.reshape(-1, 3)
        self.kdtree = cKDTree(all_vertices)
        
        logger.info(f"Prepared geometry with {len(self.surface_triangles)} triangles")
    
    def point_to_triangle_distance(self, point: np.ndarray, triangle: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute distance from point to triangle.
        
        Args:
            point: Query point (3,)
            triangle: Triangle vertices (3, 3)
            
        Returns:
            Tuple of (distance, closest_point)
        """
        # Triangle vertices
        a, b, c = triangle[0], triangle[1], triangle[2]
        
        # Vector from a to point
        ap = point - a
        
        # Triangle edges
        ab = b - a
        ac = c - a
        
        # Compute barycentric coordinates
        d1 = np.dot(ab, ap)
        d2 = np.dot(ac, ap)
        
        # Check if point is in vertex region outside A
        if d1 <= 0.0 and d2 <= 0.0:
            closest = a
            return np.linalg.norm(point - closest), closest
        
        # Check if point is in vertex region outside B
        bp = point - b
        d3 = np.dot(ab, bp)
        d4 = np.dot(ac, bp)
        if d3 >= 0.0 and d4 <= d3:
            closest = b
            return np.linalg.norm(point - closest), closest
        
        # Check if point is in edge region of AB
        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / (d1 - d3)
            closest = a + v * ab
            return np.linalg.norm(point - closest), closest
        
        # Check if point is in vertex region outside C
        cp = point - c
        d5 = np.dot(ab, cp)
        d6 = np.dot(ac, cp)
        if d6 >= 0.0 and d5 <= d6:
            closest = c
            return np.linalg.norm(point - closest), closest
        
        # Check if point is in edge region of AC
        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = d2 / (d2 - d6)
            closest = a + w * ac
            return np.linalg.norm(point - closest), closest
        
        # Check if point is in edge region of BC
        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            closest = b + w * (c - b)
            return np.linalg.norm(point - closest), closest
        
        # Point is inside triangle - project onto plane
        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        closest = a + ab * v + ac * w
        
        return np.linalg.norm(point - closest), closest
    
    def compute_sdf_point(self, point: np.ndarray) -> float:
        """
        Compute signed distance for a single point using improved 3D method.
        
        Args:
            point: Query point (3,)
            
        Returns:
            Signed distance value (negative inside, positive outside)
        """
        # Use KDTree to find nearest vertices first (much faster)
        k_nearest = min(50, len(self.surface_triangles))  # Check more triangles for accuracy
        
        # Query multiple nearest points
        all_vertices = np.array(self.surface_triangles).reshape(-1, 3)
        distances, indices = self.kdtree.query(point, k=min(k_nearest*3, len(all_vertices)))
        
        # Get candidate triangles that contain these vertices
        triangle_candidates = set()
        for vertex_idx in indices:
            # Each triangle has 3 vertices, find which triangles this vertex belongs to
            triangle_idx = vertex_idx // 3
            # Add this triangle and nearby ones
            for i in range(max(0, triangle_idx-1), min(len(self.surface_triangles), triangle_idx+2)):
                triangle_candidates.add(i)
        
        # Find closest point on surface
        min_distance = float('inf')
        closest_triangle_idx = 0
        closest_point_on_surface = None
        
        for i in triangle_candidates:
            if i < len(self.surface_triangles):
                triangle = self.surface_triangles[i]
                distance, closest_point = self.point_to_triangle_distance(point, triangle)
                if distance < min_distance:
                    min_distance = distance
                    closest_triangle_idx = i
                    closest_point_on_surface = closest_point
        
        # Improved sign determination using ray casting
        sign = self._determine_sign_robust(point, closest_point_on_surface, closest_triangle_idx)
        
        return sign * min_distance
    
    def _determine_sign_robust(self, point: np.ndarray, closest_surface_point: np.ndarray, triangle_idx: int) -> float:
        """
        Robustly determine if point is inside or outside using multiple methods.
        
        Args:
            point: Query point
            closest_surface_point: Closest point on surface
            triangle_idx: Index of closest triangle
            
        Returns:
            Sign: -1.0 for inside, +1.0 for outside
        """
        # Method 1: Normal direction (primary)
        triangle_normal = self.triangle_normals[triangle_idx]
        to_point = point - closest_surface_point
        normal_sign = 1.0 if np.dot(to_point, triangle_normal) >= 0 else -1.0
        
        # Method 2: Ray casting for verification (more robust)
        try:
            # Cast ray in random directions and count intersections
            ray_directions = [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]), 
                np.array([0.0, 0.0, 1.0]),
                np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
            ]
            
            inside_votes = 0
            total_votes = 0
            
            for ray_dir in ray_directions:
                intersections = self._count_ray_intersections(point, ray_dir)
                total_votes += 1
                if intersections % 2 == 1:  # Odd number = inside
                    inside_votes += 1
            
            # If majority of rays agree point is inside
            if inside_votes > total_votes // 2:
                ray_sign = -1.0
            else:
                ray_sign = 1.0
            
            # Use ray casting result if it's confident, otherwise use normal
            if abs(inside_votes - total_votes//2) > 1:  # Clear majority
                return ray_sign
            else:
                return normal_sign
                
        except:
            # Fallback to normal method
            return normal_sign
    
    def _count_ray_intersections(self, point: np.ndarray, ray_direction: np.ndarray, max_triangles: int = 100) -> int:
        """
        Count intersections of ray with surface triangles.
        
        Args:
            point: Ray origin
            ray_direction: Ray direction (normalized)
            max_triangles: Maximum triangles to check
            
        Returns:
            Number of intersections
        """
        intersections = 0
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        
        # Check subset of triangles for performance
        triangle_subset = self.surface_triangles[:max_triangles] if len(self.surface_triangles) > max_triangles else self.surface_triangles
        
        for triangle in triangle_subset:
            if self._ray_triangle_intersection(point, ray_direction, triangle):
                intersections += 1
        
        return intersections
    
    def _ray_triangle_intersection(self, ray_origin: np.ndarray, ray_direction: np.ndarray, triangle: np.ndarray) -> bool:
        """
        Check if ray intersects triangle using Möller-Trumbore algorithm.
        
        Args:
            ray_origin: Ray origin point
            ray_direction: Ray direction (normalized)
            triangle: Triangle vertices (3, 3)
            
        Returns:
            True if intersection exists
        """
        v0, v1, v2 = triangle[0], triangle[1], triangle[2]
        
        # Edges
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Cross product
        h = np.cross(ray_direction, edge2)
        a = np.dot(edge1, h)
        
        # Ray parallel to triangle
        if abs(a) < 1e-8:
            return False
        
        f = 1.0 / a
        s = ray_origin - v0
        u = f * np.dot(s, h)
        
        if u < 0.0 or u > 1.0:
            return False
        
        q = np.cross(s, edge1)
        v = f * np.dot(ray_direction, q)
        
        if v < 0.0 or u + v > 1.0:
            return False
        
        # Intersection distance
        t = f * np.dot(edge2, q)
        
        return t > 1e-8  # Ray intersects triangle in forward direction
    
    def compute_sdf_grid(self, 
                        grid_points: np.ndarray,
                        batch_size: int = 50000) -> np.ndarray:
        """
        Compute signed distance function on a grid of points with optimizations.
        
        Args:
            grid_points: Grid points array of shape (N, 3)
            batch_size: Process points in batches to manage memory
            
        Returns:
            SDF values array of shape (N,)
        """
        num_points = grid_points.shape[0]
        sdf_values = np.zeros(num_points)
        
        logger.info(f"Computing SDF for {num_points:,} grid points using optimized method")
        
        # Get mesh bounds for quick inside/outside estimation
        all_vertices = self.surface_triangles.reshape(-1, 3)
        mesh_min = np.min(all_vertices, axis=0)
        mesh_max = np.max(all_vertices, axis=0)
        
        # Process in larger batches with vectorized operations where possible
        for i in range(0, num_points, batch_size):
            end_idx = min(i + batch_size, num_points)
            batch_points = grid_points[i:end_idx]
            batch_size_actual = len(batch_points)
            
            # Quick bounding box check - points far outside get positive SDF
            outside_bbox = np.any((batch_points < mesh_min - 1.0) | (batch_points > mesh_max + 1.0), axis=1)
            
            # For points outside bounding box, compute approximate distance
            for j, point in enumerate(batch_points):
                if outside_bbox[j]:
                    # Quick approximation for far points
                    bbox_center = (mesh_min + mesh_max) / 2
                    sdf_values[i + j] = np.linalg.norm(point - bbox_center)
                else:
                    # Accurate computation for points near the mesh
                    sdf_values[i + j] = self.compute_sdf_point(point)
            
            # Progress reporting
            progress = (i + batch_size_actual) / num_points * 100
            if i == 0 or (i // batch_size + 1) % 5 == 0:
                logger.info(f"Progress: {progress:.1f}% ({i + batch_size_actual:,}/{num_points:,} points)")
        
        logger.info("SDF computation completed")
        return sdf_values
    
    def compute_sdf_cartesian_grid(self,
                                  bounds: Tuple[np.ndarray, np.ndarray],
                                  resolution: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute SDF on a regular Cartesian grid.
        
        Args:
            bounds: Tuple of (min_coords, max_coords) each of shape (3,)
            resolution: Grid resolution (nx, ny, nz)
            
        Returns:
            Tuple of (grid_coordinates, sdf_values)
            grid_coordinates: meshgrid tuple (X, Y, Z)
            sdf_values: SDF array of shape resolution
        """
        min_coords, max_coords = bounds
        nx, ny, nz = resolution
        
        # Create coordinate arrays
        x = np.linspace(min_coords[0], max_coords[0], nx)
        y = np.linspace(min_coords[1], max_coords[1], ny)
        z = np.linspace(min_coords[2], max_coords[2], nz)
        
        # Create meshgrid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Flatten for SDF computation
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # Compute SDF
        sdf_values = self.compute_sdf_grid(grid_points)
        
        # Reshape back to grid
        sdf_grid = sdf_values.reshape(resolution)
        
        return (X, Y, Z), sdf_grid
    
    @staticmethod
    @jax.jit
    def jax_sdf_interpolation(sdf_grid: jax.Array,
                             bounds: Tuple[jax.Array, jax.Array],
                             query_points: jax.Array) -> jax.Array:
        """
        JAX-compatible SDF interpolation for GPU acceleration.
        
        Args:
            sdf_grid: Precomputed SDF grid
            bounds: Grid bounds (min_coords, max_coords)
            query_points: Points to interpolate SDF at
            
        Returns:
            Interpolated SDF values
        """
        min_coords, max_coords = bounds
        grid_shape = sdf_grid.shape
        
        # Normalize coordinates to grid indices
        normalized_coords = (query_points - min_coords) / (max_coords - min_coords)
        grid_coords = normalized_coords * jnp.array([grid_shape[0]-1, grid_shape[1]-1, grid_shape[2]-1])
        
        # Clamp coordinates to valid range
        grid_coords = jnp.clip(grid_coords, 0, jnp.array([grid_shape[0]-1, grid_shape[1]-1, grid_shape[2]-1]))
        
        # Trilinear interpolation
        i0 = jnp.floor(grid_coords).astype(int)
        i1 = jnp.minimum(i0 + 1, jnp.array([grid_shape[0]-1, grid_shape[1]-1, grid_shape[2]-1]))
        
        # Interpolation weights
        w = grid_coords - i0
        
        # Extract values at grid corners
        v000 = sdf_grid[i0[..., 0], i0[..., 1], i0[..., 2]]
        v001 = sdf_grid[i0[..., 0], i0[..., 1], i1[..., 2]]
        v010 = sdf_grid[i0[..., 0], i1[..., 1], i0[..., 2]]
        v011 = sdf_grid[i0[..., 0], i1[..., 1], i1[..., 2]]
        v100 = sdf_grid[i1[..., 0], i0[..., 1], i0[..., 2]]
        v101 = sdf_grid[i1[..., 0], i0[..., 1], i1[..., 2]]
        v110 = sdf_grid[i1[..., 0], i1[..., 1], i0[..., 2]]
        v111 = sdf_grid[i1[..., 0], i1[..., 1], i1[..., 2]]
        
        # Trilinear interpolation
        v00 = v000 * (1 - w[..., 2]) + v001 * w[..., 2]
        v01 = v010 * (1 - w[..., 2]) + v011 * w[..., 2]
        v10 = v100 * (1 - w[..., 2]) + v101 * w[..., 2]
        v11 = v110 * (1 - w[..., 2]) + v111 * w[..., 2]
        
        v0 = v00 * (1 - w[..., 1]) + v01 * w[..., 1]
        v1 = v10 * (1 - w[..., 1]) + v11 * w[..., 1]
        
        return v0 * (1 - w[..., 0]) + v1 * w[..., 0]
    
    def create_jax_levelset_function(self, 
                                   bounds: Tuple[np.ndarray, np.ndarray],
                                   resolution: Tuple[int, int, int]) -> Callable:
        """
        Create a JAX-compatible levelset function for use in JAX-Fluids.
        
        Args:
            bounds: Grid bounds for SDF computation
            resolution: Grid resolution for SDF precomputation
            
        Returns:
            JAX-compatible function that can be used as levelset initializer
        """
        # Precompute SDF on grid
        _, sdf_grid = self.compute_sdf_cartesian_grid(bounds, resolution)
        
        # Convert to JAX arrays
        sdf_grid_jax = jnp.array(sdf_grid)
        min_coords_jax = jnp.array(bounds[0])
        max_coords_jax = jnp.array(bounds[1])
        
        def levelset_function(x, y, z):
            """JAX-compatible levelset function."""
            # Stack coordinates
            points = jnp.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
            
            # Interpolate SDF
            sdf_values = SignedDistanceFunction.jax_sdf_interpolation(
                sdf_grid_jax, (min_coords_jax, max_coords_jax), points
            )
            
            # Reshape to match input grid shape
            return sdf_values.reshape(x.shape)
        
        return levelset_function 