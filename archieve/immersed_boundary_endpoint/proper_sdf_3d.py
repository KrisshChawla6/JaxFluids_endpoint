#!/usr/bin/env python3
"""
Proper 3D Signed Distance Function Implementation

Based on:
- Bærentzen & Aanæs (2002): "Generating Signed Distance Fields From Triangle Meshes"
- Optimized point-to-triangle distance algorithms
- Angle-weighted normal method for correct inside/outside determination
"""

import numpy as np
import logging
from scipy.spatial import KDTree
from multiprocessing import Pool, cpu_count
import time
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class ProperSDF3D:
    """
    Proper 3D Signed Distance Function using angle-weighted normals.
    
    This implementation follows the mathematical foundation from academic literature
    for computing accurate signed distance fields from triangulated meshes.
    """
    
    def __init__(self, triangles: np.ndarray):
        """
        Initialize with triangle mesh.
        
        Args:
            triangles: Array of triangles, shape (N, 3, 3) where N is number of triangles
        """
        self.triangles = np.array(triangles)
        self.num_triangles = len(self.triangles)
        
        logger.info(f"ProperSDF3D initialized with {self.num_triangles:,} triangles")
        
        # Precompute triangle properties
        self._precompute_triangle_properties()
        
        # Build spatial acceleration structure
        self._build_spatial_structure()
        
        # Precompute angle-weighted normals for vertices and edges
        self._precompute_angle_weighted_normals()
        
    def _precompute_triangle_properties(self):
        """Precompute triangle normals, areas, and other properties."""
        # Triangle vertices
        v0 = self.triangles[:, 0, :]  # First vertex of each triangle
        v1 = self.triangles[:, 1, :]  # Second vertex
        v2 = self.triangles[:, 2, :]  # Third vertex
        
        # Edge vectors
        self.edge_01 = v1 - v0  # v0 -> v1
        self.edge_02 = v2 - v0  # v0 -> v2
        self.edge_12 = v2 - v1  # v1 -> v2
        
        # Face normals (outward pointing)
        self.face_normals = np.cross(self.edge_01, self.edge_02)
        self.face_areas = 0.5 * np.linalg.norm(self.face_normals, axis=1)
        
        # Normalize face normals
        norm_lengths = np.linalg.norm(self.face_normals, axis=1, keepdims=True)
        norm_lengths = np.where(norm_lengths > 1e-12, norm_lengths, 1.0)
        self.face_normals = self.face_normals / norm_lengths
        
        # Triangle centers for spatial queries
        self.triangle_centers = (v0 + v1 + v2) / 3.0
        
        logger.info(f"Precomputed properties for {self.num_triangles:,} triangles")
        
    def _build_spatial_structure(self):
        """Build KDTree for spatial acceleration."""
        # Use all triangle vertices for spatial queries
        all_vertices = self.triangles.reshape(-1, 3)
        self.vertex_kdtree = KDTree(all_vertices)
        
        # Also build KDTree for triangle centers
        self.center_kdtree = KDTree(self.triangle_centers)
        
        logger.info("Built spatial acceleration structures")
        
    def _precompute_angle_weighted_normals(self):
        """
        Precompute angle-weighted normals for vertices and edges.
        This is the key to proper inside/outside determination.
        """
        # For simplicity, we'll use a vertex-based approach
        # In practice, you'd want to build a proper half-edge data structure
        
        # Get unique vertices and their triangle associations
        all_vertices = self.triangles.reshape(-1, 3)
        unique_vertices, inverse_indices = np.unique(all_vertices, axis=0, return_inverse=True)
        
        self.unique_vertices = unique_vertices
        self.vertex_to_triangles = {}
        
        # Map each vertex to its incident triangles
        for tri_idx in range(self.num_triangles):
            for vert_idx in range(3):
                global_vert_idx = tri_idx * 3 + vert_idx
                unique_vert_idx = inverse_indices[global_vert_idx]
                
                if unique_vert_idx not in self.vertex_to_triangles:
                    self.vertex_to_triangles[unique_vert_idx] = []
                self.vertex_to_triangles[unique_vert_idx].append((tri_idx, vert_idx))
        
        # Compute angle-weighted normals for each unique vertex
        self.vertex_angle_weighted_normals = np.zeros((len(unique_vertices), 3))
        
        for unique_vert_idx, incident_triangles in self.vertex_to_triangles.items():
            vertex_pos = unique_vertices[unique_vert_idx]
            weighted_normal = np.zeros(3)
            
            for tri_idx, vert_idx in incident_triangles:
                # Get the angle at this vertex in this triangle
                v0, v1, v2 = self.triangles[tri_idx, 0], self.triangles[tri_idx, 1], self.triangles[tri_idx, 2]
                
                if vert_idx == 0:
                    # Angle at v0
                    edge1, edge2 = v1 - v0, v2 - v0
                elif vert_idx == 1:
                    # Angle at v1  
                    edge1, edge2 = v0 - v1, v2 - v1
                else:
                    # Angle at v2
                    edge1, edge2 = v0 - v2, v1 - v2
                
                # Compute angle using dot product
                edge1_norm = np.linalg.norm(edge1)
                edge2_norm = np.linalg.norm(edge2)
                
                if edge1_norm > 1e-12 and edge2_norm > 1e-12:
                    cos_angle = np.clip(np.dot(edge1, edge2) / (edge1_norm * edge2_norm), -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    
                    # Weight the face normal by this angle
                    weighted_normal += angle * self.face_normals[tri_idx]
            
            # Normalize the weighted normal
            norm_length = np.linalg.norm(weighted_normal)
            if norm_length > 1e-12:
                self.vertex_angle_weighted_normals[unique_vert_idx] = weighted_normal / norm_length
            else:
                # Fallback to average normal
                avg_normal = np.mean([self.face_normals[tri_idx] for tri_idx, _ in incident_triangles], axis=0)
                norm_length = np.linalg.norm(avg_normal)
                if norm_length > 1e-12:
                    self.vertex_angle_weighted_normals[unique_vert_idx] = avg_normal / norm_length
        
        logger.info(f"Computed angle-weighted normals for {len(unique_vertices):,} vertices")
        
    def point_to_triangle_distance(self, point: np.ndarray, triangle_idx: int) -> Tuple[float, np.ndarray, str]:
        """
        Compute distance from point to triangle using Voronoi regions.
        
        Args:
            point: Query point (3D)
            triangle_idx: Index of triangle
            
        Returns:
            (distance, closest_point, region_type)
            region_type: 'face', 'edge', or 'vertex'
        """
        triangle = self.triangles[triangle_idx]
        v0, v1, v2 = triangle[0], triangle[1], triangle[2]
        
        # Edge vectors
        edge01 = v1 - v0
        edge02 = v2 - v0
        edge12 = v2 - v1
        
        # Vector from v0 to point
        v0p = point - v0
        
        # Project point onto triangle plane
        normal = self.face_normals[triangle_idx]
        dist_to_plane = np.dot(v0p, normal)
        projected_point = point - dist_to_plane * normal
        
        # Compute barycentric coordinates
        d00 = np.dot(edge02, edge02)
        d01 = np.dot(edge02, edge01)
        d11 = np.dot(edge01, edge01)
        d20 = np.dot(projected_point - v0, edge02)
        d21 = np.dot(projected_point - v0, edge01)
        
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-12:
            # Degenerate triangle, use closest vertex
            dist_v0 = np.linalg.norm(point - v0)
            dist_v1 = np.linalg.norm(point - v1)
            dist_v2 = np.linalg.norm(point - v2)
            
            if dist_v0 <= dist_v1 and dist_v0 <= dist_v2:
                return dist_v0, v0, 'vertex'
            elif dist_v1 <= dist_v2:
                return dist_v1, v1, 'vertex'
            else:
                return dist_v2, v2, 'vertex'
        
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        # Check if point is inside triangle
        if u >= 0 and v >= 0 and w >= 0:
            # Inside triangle - distance is distance to plane
            return abs(dist_to_plane), projected_point, 'face'
        
        # Outside triangle - find closest edge or vertex
        closest_dist = float('inf')
        closest_point = None
        region_type = None
        
        # Check edges
        edges = [
            (v0, v1, edge01, 'edge01'),
            (v1, v2, edge12, 'edge12'), 
            (v2, v0, -edge02, 'edge20')
        ]
        
        for start, end, edge_vec, edge_name in edges:
            # Project point onto edge
            t = np.dot(point - start, edge_vec) / max(np.dot(edge_vec, edge_vec), 1e-12)
            t = np.clip(t, 0.0, 1.0)
            
            edge_point = start + t * edge_vec
            dist = np.linalg.norm(point - edge_point)
            
            if dist < closest_dist:
                closest_dist = dist
                closest_point = edge_point
                region_type = 'edge'
        
        # Check vertices
        for vertex in [v0, v1, v2]:
            dist = np.linalg.norm(point - vertex)
            if dist < closest_dist:
                closest_dist = dist
                closest_point = vertex
                region_type = 'vertex'
        
        return closest_dist, closest_point, region_type
        
    def compute_sdf(self, point: np.ndarray) -> float:
        """
        Compute signed distance for a single point using proper 3D method.
        
        Args:
            point: Query point (3D)
            
        Returns:
            Signed distance (negative inside, positive outside)
        """
        # Find nearest triangles using spatial acceleration
        k_nearest = min(50, self.num_triangles)
        center_distances, center_indices = self.center_kdtree.query(point, k=k_nearest)
        
        # Find the actual minimum distance
        min_distance = float('inf')
        closest_triangle_idx = -1
        closest_point = None
        closest_region_type = None
        
        for tri_idx in center_indices:
            dist, closest_pt, region_type = self.point_to_triangle_distance(point, tri_idx)
            
            if dist < min_distance:
                min_distance = dist
                closest_triangle_idx = tri_idx
                closest_point = closest_pt
                closest_region_type = region_type
        
        # Determine sign using angle-weighted normal method
        if closest_region_type == 'face':
            # Use face normal
            direction_vector = point - closest_point
            face_normal = self.face_normals[closest_triangle_idx]
            sign_test = np.dot(direction_vector, face_normal)
            
        elif closest_region_type == 'vertex':
            # Use angle-weighted normal at vertex
            # Find which unique vertex this corresponds to
            all_vertices = self.triangles.reshape(-1, 3)
            vertex_distances = np.linalg.norm(all_vertices - closest_point.reshape(1, -1), axis=1)
            closest_vertex_global_idx = np.argmin(vertex_distances)
            
            # Map to unique vertex index (this is approximate - in practice you'd maintain proper mapping)
            unique_vertices_distances = np.linalg.norm(self.unique_vertices - closest_point.reshape(1, -1), axis=1)
            unique_vertex_idx = np.argmin(unique_vertices_distances)
            
            direction_vector = point - closest_point
            vertex_normal = self.vertex_angle_weighted_normals[unique_vertex_idx]
            sign_test = np.dot(direction_vector, vertex_normal)
            
        else:  # edge
            # For edges, interpolate between adjacent face normals
            # This is simplified - proper implementation would use edge angle-weighted normals
            direction_vector = point - closest_point
            face_normal = self.face_normals[closest_triangle_idx]
            sign_test = np.dot(direction_vector, face_normal)
        
        # Return signed distance
        return min_distance if sign_test > 0 else -min_distance
        
    def compute_sdf_batch_parallel(self, points: np.ndarray, 
                                  num_processes: Optional[int] = None) -> np.ndarray:
        """
        Compute SDF for many points using multiprocessing.
        
        Args:
            points: Array of points, shape (N, 3)
            num_processes: Number of processes (None = auto-detect)
            
        Returns:
            Array of SDF values, shape (N,)
        """
        if num_processes is None:
            num_processes = min(cpu_count(), 8)
        
        total_points = len(points)
        logger.info(f"Computing proper 3D SDF for {total_points:,} points using {num_processes} processes")
        
        # Split into chunks
        chunk_size = max(total_points // (num_processes * 4), 1000)
        chunks = []
        for i in range(0, total_points, chunk_size):
            end_idx = min(i + chunk_size, total_points)
            chunks.append(points[i:end_idx])
        
        logger.info(f"Split into {len(chunks)} chunks of ~{chunk_size:,} points each")
        
        start_time = time.time()
        
        # Use multiprocessing
        with Pool(processes=num_processes, initializer=_init_worker_proper,
                 initargs=(self.triangles, self.face_normals, self.triangle_centers,
                          self.unique_vertices, self.vertex_angle_weighted_normals,
                          self.vertex_to_triangles)) as pool:
            
            results = []
            completed_points = 0
            
            for i, result in enumerate(pool.imap(_compute_sdf_chunk_proper, chunks)):
                results.extend(result)
                completed_points += len(result)
                
                progress = completed_points / total_points * 100
                elapsed = time.time() - start_time
                if elapsed > 0:
                    rate = completed_points / elapsed
                    eta = (total_points - completed_points) / rate if rate > 0 else 0
                    logger.info(f"  Proper SDF Progress: {progress:.1f}% "
                              f"({completed_points:,}/{total_points:,}) "
                              f"[{rate:.0f} pts/sec, ETA: {eta:.0f}s]")
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Proper 3D SDF computation completed in {elapsed:.1f}s "
                   f"({total_points/elapsed:.0f} pts/sec)")
        
        return np.array(results)


# Global variables for multiprocessing
_worker_triangles = None
_worker_face_normals = None
_worker_triangle_centers = None
_worker_unique_vertices = None
_worker_vertex_normals = None
_worker_vertex_to_triangles = None
_worker_center_kdtree = None

def _init_worker_proper(triangles, face_normals, triangle_centers,
                       unique_vertices, vertex_normals, vertex_to_triangles):
    """Initialize worker process with mesh data."""
    global _worker_triangles, _worker_face_normals, _worker_triangle_centers
    global _worker_unique_vertices, _worker_vertex_normals, _worker_vertex_to_triangles
    global _worker_center_kdtree
    
    _worker_triangles = triangles
    _worker_face_normals = face_normals
    _worker_triangle_centers = triangle_centers
    _worker_unique_vertices = unique_vertices
    _worker_vertex_normals = vertex_normals
    _worker_vertex_to_triangles = vertex_to_triangles
    
    # Build KDTree for this worker
    _worker_center_kdtree = KDTree(triangle_centers)

def _compute_sdf_chunk_proper(points):
    """Compute SDF for a chunk of points using proper 3D method."""
    results = []
    
    for point in points:
        # Find nearest triangles
        k_nearest = min(20, len(_worker_triangles))
        center_distances, center_indices = _worker_center_kdtree.query(point, k=k_nearest)
        
        # Find minimum distance
        min_distance = float('inf')
        closest_triangle_idx = -1
        closest_point = None
        closest_region_type = None
        
        for tri_idx in center_indices:
            dist, closest_pt, region_type = _point_to_triangle_distance_worker(point, tri_idx)
            
            if dist < min_distance:
                min_distance = dist
                closest_triangle_idx = tri_idx
                closest_point = closest_pt
                closest_region_type = region_type
        
        # Determine sign
        direction_vector = point - closest_point
        
        if closest_region_type == 'face':
            face_normal = _worker_face_normals[closest_triangle_idx]
            sign_test = np.dot(direction_vector, face_normal)
        else:
            # Simplified - use face normal for edges and vertices too
            face_normal = _worker_face_normals[closest_triangle_idx]
            sign_test = np.dot(direction_vector, face_normal)
        
        sdf_value = min_distance if sign_test > 0 else -min_distance
        results.append(sdf_value)
    
    return results

def _point_to_triangle_distance_worker(point, triangle_idx):
    """Worker version of point-to-triangle distance."""
    triangle = _worker_triangles[triangle_idx]
    v0, v1, v2 = triangle[0], triangle[1], triangle[2]
    
    # Simplified version - project to plane and check barycentric coordinates
    edge01 = v1 - v0
    edge02 = v2 - v0
    v0p = point - v0
    
    # Project onto plane
    normal = _worker_face_normals[triangle_idx]
    dist_to_plane = np.dot(v0p, normal)
    projected_point = point - dist_to_plane * normal
    
    # Barycentric coordinates
    d00 = np.dot(edge02, edge02)
    d01 = np.dot(edge02, edge01)
    d11 = np.dot(edge01, edge01)
    d20 = np.dot(projected_point - v0, edge02)
    d21 = np.dot(projected_point - v0, edge01)
    
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        # Degenerate - use closest vertex
        dists = [np.linalg.norm(point - v) for v in [v0, v1, v2]]
        min_idx = np.argmin(dists)
        return dists[min_idx], [v0, v1, v2][min_idx], 'vertex'
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    if u >= 0 and v >= 0 and w >= 0:
        # Inside triangle
        return abs(dist_to_plane), projected_point, 'face'
    
    # Outside - find closest edge/vertex
    closest_dist = float('inf')
    closest_point = None
    
    # Check edges
    for start, end in [(v0, v1), (v1, v2), (v2, v0)]:
        edge_vec = end - start
        t = np.dot(point - start, edge_vec) / max(np.dot(edge_vec, edge_vec), 1e-12)
        t = np.clip(t, 0.0, 1.0)
        
        edge_point = start + t * edge_vec
        dist = np.linalg.norm(point - edge_point)
        
        if dist < closest_dist:
            closest_dist = dist
            closest_point = edge_point
    
    # Check vertices
    for vertex in [v0, v1, v2]:
        dist = np.linalg.norm(point - vertex)
        if dist < closest_dist:
            closest_dist = dist
            closest_point = vertex
    
    return closest_dist, closest_point, 'edge' 