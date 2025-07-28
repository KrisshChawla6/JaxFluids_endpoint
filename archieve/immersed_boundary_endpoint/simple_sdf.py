import numpy as np
import logging
from scipy.spatial import KDTree
from multiprocessing import Pool, cpu_count
import time

logger = logging.getLogger(__name__)

class SimpleSDF:
    """
    Simple but robust SDF implementation for non-watertight surface meshes.
    This approach calculates the distance to the nearest surface point and uses a
    heuristic based on proximity to the mesh center to determine the sign.
    """

    def __init__(self, surface_triangles):
        """
        Initializes the SimpleSDF computer.
        Args:
            surface_triangles: A list or numpy array of triangles, shape (n, 3, 3).
        """
        if not isinstance(surface_triangles, np.ndarray):
            self.surface_triangles = np.array(surface_triangles)
        else:
            self.surface_triangles = surface_triangles
            
        self.num_triangles = len(self.surface_triangles)
        
        # Get all vertices and create a KDTree for fast nearest-neighbor searches.
        all_vertices = self.surface_triangles.reshape(-1, 3)
        self.kdtree = KDTree(all_vertices)
        
        # Compute mesh statistics for the inside/outside heuristic.
        self.mesh_center = np.mean(all_vertices, axis=0)
        self.mesh_bounds = (np.min(all_vertices, axis=0), np.max(all_vertices, axis=0))
        self.mesh_size = self.mesh_bounds[1] - self.mesh_bounds[0]
        self.mesh_radius = np.linalg.norm(self.mesh_size) / 2
        
        logger.info(f"SimpleSDF initialized with {self.num_triangles} triangles.")
        logger.info(f"Mesh Center: {self.mesh_center}, Radius: {self.mesh_radius:.2f}")

    def distance_to_surface(self, point: np.ndarray) -> float:
        """Computes the minimum distance from a point to the mesh surface using optimized search."""
        # Use KDTree to find closest vertices - this is the key optimization
        k_nearest = min(30, len(self.kdtree.data))  # Reduced from 50 for speed
        distances, indices = self.kdtree.query(point, k=k_nearest)
        
        # Get candidate triangles more efficiently
        triangle_candidates = set()
        for vertex_idx in indices:
            triangle_idx = vertex_idx // 3
            # Add nearby triangles
            for i in range(max(0, triangle_idx-1), min(self.num_triangles, triangle_idx+2)):
                triangle_candidates.add(i)
        
        # Find minimum distance to candidate triangles
        min_dist = float('inf')
        for tri_idx in triangle_candidates:
            triangle = self.surface_triangles[tri_idx]
            dist = self._point_to_triangle_distance(point, triangle)
            if dist < min_dist:
                min_dist = dist
        
        return min_dist

    def _point_to_triangle_distance(self, point: np.ndarray, triangle: np.ndarray) -> float:
        """Calculates the exact distance from a point to a single 3D triangle."""
        a, b, c = triangle[0], triangle[1], triangle[2]
        ab, ac = b - a, c - a
        
        # Project point onto triangle plane to find the closest point
        normal = np.cross(ab, ac)
        if np.linalg.norm(normal) < 1e-12: # Handle degenerate triangles
            return min(np.linalg.norm(point - a), np.linalg.norm(point - b), np.linalg.norm(point - c))
        
        normal /= np.linalg.norm(normal)
        dist_to_plane = np.dot(point - a, normal)
        projected_point = point - dist_to_plane * normal
        
        # Check if the projected point is inside the triangle using barycentric coordinates
        v0, v1, v2 = c - a, b - a, projected_point - a
        dot00, dot01, dot02 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v0, v2)
        dot11, dot12 = np.dot(v1, v1), np.dot(v1, v2)
        
        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        
        # If inside, the distance is the distance to the plane
        if (u >= 0) and (v >= 0) and (u + v <= 1):
            return abs(dist_to_plane)
            
        # If outside, find the closest point on one of the triangle's edges
        dists = [
            np.linalg.norm(point - (a + np.clip(np.dot(point - a, ab) / np.dot(ab, ab), 0, 1) * ab)),
            np.linalg.norm(point - (b + np.clip(np.dot(point - b, c - b) / np.dot(c-b, c-b), 0, 1) * (c - b))),
            np.linalg.norm(point - (c + np.clip(np.dot(point - c, a - c) / np.dot(a-c, a-c), 0, 1) * (a - c)))
        ]
        return min(dists)

    def is_inside_heuristic(self, point: np.ndarray, surface_distance: float = None) -> bool:
        """
        Optimized inside/outside test - reuses surface distance if already computed.
        """
        dist_to_center = np.linalg.norm(point - self.mesh_center)
        
        # Reuse surface distance if provided (avoids double computation)
        if surface_distance is None:
            surface_distance = self.distance_to_surface(point)
        
        # Optimized thresholds
        center_threshold = self.mesh_radius * 0.75  # Slightly tighter
        surface_threshold = self.mesh_radius * 0.12  # Closer to surface
        
        return dist_to_center < center_threshold and surface_distance < surface_threshold

    def compute_sdf(self, point: np.ndarray) -> float:
        """
        Optimized SDF computation - computes distance once and reuses it.
        """
        distance = self.distance_to_surface(point)
        is_inside = self.is_inside_heuristic(point, surface_distance=distance)
        return -distance if is_inside else distance
    
    def compute_sdf_batch_parallel(self, grid_points: np.ndarray, 
                                  num_processes: int = None,
                                  batch_size: int = 10000) -> np.ndarray:
        """
        Compute SDF for many points using multiprocessing for maximum speed.
        
        Args:
            grid_points: Array of points to compute SDF for, shape (N, 3)
            num_processes: Number of processes to use (None = auto-detect)
            batch_size: Size of batches to process
            
        Returns:
            Array of SDF values, shape (N,)
        """
        if num_processes is None:
            num_processes = min(cpu_count(), 8)  # Cap at 8 to avoid memory issues
        
        total_points = len(grid_points)
        logger.info(f"Computing SDF for {total_points:,} points using {num_processes} processes")
        
        # Split into chunks for multiprocessing
        chunk_size = max(batch_size // num_processes, 1000)  # Ensure reasonable chunk size
        chunks = []
        for i in range(0, total_points, chunk_size):
            end_idx = min(i + chunk_size, total_points)
            chunks.append(grid_points[i:end_idx])
        
        logger.info(f"Split into {len(chunks)} chunks of ~{chunk_size:,} points each")
        
        # Process chunks in parallel
        start_time = time.time()
        
        with Pool(processes=num_processes, initializer=_init_worker, 
                 initargs=(self.surface_triangles, self.mesh_center, 
                          self.mesh_radius, self.num_triangles)) as pool:
            
            # Process all chunks
            results = []
            completed_points = 0
            
            for i, result in enumerate(pool.imap(_compute_sdf_chunk, chunks)):
                results.extend(result)
                completed_points += len(result)
                
                # Progress reporting
                progress = completed_points / total_points * 100
                elapsed = time.time() - start_time
                if elapsed > 0:
                    rate = completed_points / elapsed
                    eta = (total_points - completed_points) / rate if rate > 0 else 0
                    logger.info(f"  Parallel SDF Progress: {progress:.1f}% "
                              f"({completed_points:,}/{total_points:,}) "
                              f"[{rate:.0f} pts/sec, ETA: {eta:.0f}s]")
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Parallel SDF computation completed in {elapsed:.1f}s "
                   f"({total_points/elapsed:.0f} pts/sec)")
        
        return np.array(results)


# Global variables for multiprocessing workers
_worker_triangles = None
_worker_center = None
_worker_radius = None
_worker_num_triangles = None
_worker_kdtree = None

def _init_worker(triangles, center, radius, num_triangles):
    """Initialize worker process with shared data."""
    global _worker_triangles, _worker_center, _worker_radius, _worker_num_triangles, _worker_kdtree
    _worker_triangles = triangles
    _worker_center = center
    _worker_radius = radius
    _worker_num_triangles = num_triangles
    
    # Create KDTree for this worker
    all_vertices = triangles.reshape(-1, 3)
    _worker_kdtree = KDTree(all_vertices)

def _compute_sdf_chunk(points):
    """Compute SDF for a chunk of points in a worker process."""
    results = []
    
    for point in points:
        # Optimized distance computation using worker's KDTree
        k_nearest = min(30, len(_worker_kdtree.data))
        distances, indices = _worker_kdtree.query(point, k=k_nearest)
        
        # Get candidate triangles
        triangle_candidates = set()
        for vertex_idx in indices:
            triangle_idx = vertex_idx // 3
            for i in range(max(0, triangle_idx-1), min(_worker_num_triangles, triangle_idx+2)):
                triangle_candidates.add(i)
        
        # Find minimum distance
        min_dist = float('inf')
        for tri_idx in triangle_candidates:
            triangle = _worker_triangles[tri_idx]
            dist = _point_to_triangle_distance_worker(point, triangle)
            if dist < min_dist:
                min_dist = dist
        
        # Inside/outside test
        dist_to_center = np.linalg.norm(point - _worker_center)
        center_threshold = _worker_radius * 0.75
        surface_threshold = _worker_radius * 0.12
        
        is_inside = (dist_to_center < center_threshold and min_dist < surface_threshold)
        sdf_value = -min_dist if is_inside else min_dist
        
        results.append(sdf_value)
    
    return results

def _point_to_triangle_distance_worker(point, triangle):
    """Optimized point-to-triangle distance for worker processes."""
    a, b, c = triangle[0], triangle[1], triangle[2]
    ab, ac = b - a, c - a
    
    # Project point onto triangle plane
    normal = np.cross(ab, ac)
    normal_len = np.linalg.norm(normal)
    if normal_len < 1e-12:
        return min(np.linalg.norm(point - a), np.linalg.norm(point - b), np.linalg.norm(point - c))
    
    normal /= normal_len
    dist_to_plane = np.dot(point - a, normal)
    projected_point = point - dist_to_plane * normal
    
    # Barycentric coordinates
    v0, v1, v2 = c - a, b - a, projected_point - a
    dot00, dot01, dot02 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v0, v2)
    dot11, dot12 = np.dot(v1, v1), np.dot(v1, v2)
    
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return min(np.linalg.norm(point - a), np.linalg.norm(point - b), np.linalg.norm(point - c))
    
    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    # Inside triangle
    if (u >= 0) and (v >= 0) and (u + v <= 1):
        return abs(dist_to_plane)
    
    # Closest edge
    ab_dot = np.dot(ab, ab)
    bc_dot = np.dot(c-b, c-b)  
    ca_dot = np.dot(a-c, a-c)
    
    if ab_dot > 1e-12:
        t_ab = np.clip(np.dot(point - a, ab) / ab_dot, 0, 1)
        dist_ab = np.linalg.norm(point - (a + t_ab * ab))
    else:
        dist_ab = np.linalg.norm(point - a)
        
    if bc_dot > 1e-12:
        t_bc = np.clip(np.dot(point - b, c - b) / bc_dot, 0, 1)
        dist_bc = np.linalg.norm(point - (b + t_bc * (c - b)))
    else:
        dist_bc = np.linalg.norm(point - b)
        
    if ca_dot > 1e-12:
        t_ca = np.clip(np.dot(point - c, a - c) / ca_dot, 0, 1)
        dist_ca = np.linalg.norm(point - (c + t_ca * (a - c)))
    else:
        dist_ca = np.linalg.norm(point - c)
    
    return min(dist_ab, dist_bc, dist_ca) 