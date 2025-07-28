#!/usr/bin/env python3
"""
Simple Immersed Boundary Method

Usage:
    python simple_immersed_boundary.py --mesh ../mesh/propeller.msh --domain_bounds "(-100,-100,-100,100,100,100)" --resolution "(80,80,80)"
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
import argparse
import time
from pathlib import Path
from typing import Tuple, List
from scipy.spatial import cKDTree

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    import mcubes
    MCUBES_AVAILABLE = True
except ImportError:
    try:
        from skimage import measure
        MCUBES_AVAILABLE = False
        logger.info("Using scikit-image for marching cubes (install PyMCubes for better performance)")
    except ImportError:
        logger.error("Neither PyMCubes nor scikit-image available!")
        raise

from mesh_processor import GmshProcessor

class SimpleImmersedBoundary:
    """
    Simple immersed boundary method inspired by sxyu/sdf.
    
    Just does: Mesh -> SDF -> Visualize φ=0
    """
    
    def __init__(self, mesh_file: str):
        """Initialize with mesh file."""
        self.mesh_processor = GmshProcessor(mesh_file)
        self.mesh_processor.read_mesh()
        
        # Get triangles as simple array
        self.triangles = np.array(self.mesh_processor.surface_triangles)
        self.num_triangles = len(self.triangles)
        
        logger.info(f"Loaded mesh: {self.num_triangles:,} triangles")
        
        # Build spatial acceleration (KDTree on triangle centers)
        triangle_centers = self.triangles.mean(axis=1)
        self.kdtree = cKDTree(triangle_centers)
        
        logger.info("Built spatial acceleration structure")
    
    def point_to_triangle_distance(self, point: np.ndarray, triangle: np.ndarray) -> float:
        """
        Fast point-to-triangle distance (inspired by sxyu/sdf approach).
        
        Args:
            point: Query point (3,)
            triangle: Triangle vertices (3, 3)
            
        Returns:
            Unsigned distance
        """
        v0, v1, v2 = triangle[0], triangle[1], triangle[2]
        
        # Edge vectors
        edge0 = v1 - v0
        edge1 = v2 - v0
        
        # Vector from v0 to point
        v0_to_point = point - v0
        
        # Compute dot products
        a = np.dot(edge0, edge0)
        b = np.dot(edge0, edge1)
        c = np.dot(edge1, edge1)
        d = np.dot(edge0, v0_to_point)
        e = np.dot(edge1, v0_to_point)
        
        det = a * c - b * b
        s = b * e - c * d
        t = b * d - a * e
        
        if s + t <= det:
            if s < 0:
                if t < 0:
                    # Region 4
                    if d < 0:
                        s = np.clip(-d / a, 0, 1)
                        t = 0
                    else:
                        s = 0
                        t = np.clip(-e / c, 0, 1)
                else:
                    # Region 3
                    s = 0
                    t = np.clip(-e / c, 0, 1)
            else:
                if t < 0:
                    # Region 5
                    s = np.clip(-d / a, 0, 1)
                    t = 0
                else:
                    # Region 0 (inside triangle)
                    inv_det = 1 / det
                    s *= inv_det
                    t *= inv_det
        else:
            if s < 0:
                # Region 2
                tmp0 = b + d
                tmp1 = c + e
                if tmp1 > tmp0:
                    numer = tmp1 - tmp0
                    denom = a - 2 * b + c
                    s = np.clip(numer / denom, 0, 1)
                    t = 1 - s
                else:
                    t = np.clip(-e / c, 0, 1)
                    s = 0
            else:
                if t < 0:
                    # Region 6
                    tmp0 = b + e
                    tmp1 = a + d
                    if tmp1 > tmp0:
                        numer = tmp1 - tmp0
                        denom = a - 2 * b + c
                        t = np.clip(numer / denom, 0, 1)
                        s = 1 - t
                    else:
                        s = np.clip(-d / a, 0, 1)
                        t = 0
                else:
                    # Region 1
                    numer = c + e - b - d
                    if numer <= 0:
                        s = 0
                    else:
                        denom = a - 2 * b + c
                        s = np.clip(numer / denom, 0, 1)
                    t = 1 - s
        
        # Compute closest point
        closest_point = v0 + s * edge0 + t * edge1
        
        # Return distance
        return np.linalg.norm(point - closest_point)
    
    def compute_sdf_point(self, point: np.ndarray) -> float:
        """
        Compute signed distance for a single point.
        
        Args:
            point: Query point (3,)
            
        Returns:
            Signed distance (negative inside, positive outside)
        """
        # Find nearest triangles using KDTree
        k_nearest = min(20, self.num_triangles)
        distances, indices = self.kdtree.query(point, k=k_nearest)
        
        # Find minimum distance to any triangle
        min_distance = float('inf')
        closest_triangle_idx = -1
        
        for idx in indices:
            dist = self.point_to_triangle_distance(point, self.triangles[idx])
            if dist < min_distance:
                min_distance = dist
                closest_triangle_idx = idx
        
        # Determine sign using triangle normal
        triangle = self.triangles[closest_triangle_idx]
        v0, v1, v2 = triangle[0], triangle[1], triangle[2]
        
        # Compute triangle normal (outward pointing)
        edge0 = v1 - v0
        edge1 = v2 - v0
        normal = np.cross(edge0, edge1)
        normal = normal / np.linalg.norm(normal)
        
        # Vector from triangle center to point
        triangle_center = (v0 + v1 + v2) / 3.0
        to_point = point - triangle_center
        
        # Sign test
        sign = np.dot(normal, to_point)
        
        return min_distance if sign > 0 else -min_distance
    
    def compute_sdf_grid(self, domain_bounds: Tuple[float, float, float, float, float, float],
                        resolution: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute SDF on a regular grid.
        
        Args:
            domain_bounds: (xmin, ymin, zmin, xmax, ymax, zmax)
            resolution: (nx, ny, nz)
            
        Returns:
            (X, Y, Z, sdf_values) grid coordinates and SDF values
        """
        xmin, ymin, zmin, xmax, ymax, zmax = domain_bounds
        nx, ny, nz = resolution
        
        logger.info(f"Computing SDF on {nx}×{ny}×{nz} = {nx*ny*nz:,} grid")
        logger.info(f"Domain: [{xmin:.1f}, {xmax:.1f}] × [{ymin:.1f}, {ymax:.1f}] × [{zmin:.1f}, {zmax:.1f}]")
        
        # Create grid
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        z = np.linspace(zmin, zmax, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Compute SDF at each grid point
        sdf_values = np.zeros_like(X)
        total_points = nx * ny * nz
        
        start_time = time.time()
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    point = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                    sdf_values[i, j, k] = self.compute_sdf_point(point)
            
            # Progress reporting
            progress = (i + 1) / nx * 100
            elapsed = time.time() - start_time
            if elapsed > 0:
                rate = (i + 1) * ny * nz / elapsed
                eta = (total_points - (i + 1) * ny * nz) / rate
                logger.info(f"Progress: {progress:.1f}% [{rate:.0f} pts/sec, ETA: {eta:.0f}s]")
        
        elapsed = time.time() - start_time
        logger.info(f"✓ SDF computation completed in {elapsed:.1f}s ({total_points/elapsed:.0f} pts/sec)")
        
        return X, Y, Z, sdf_values
    
    def visualize_boundary(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                          sdf_values: np.ndarray, output_file: str = None) -> plt.Figure:
        """
        Visualize only the φ=0 boundary surface.
        
        Args:
            X, Y, Z: Grid coordinates
            sdf_values: SDF values
            output_file: Optional output file path
            
        Returns:
            Matplotlib figure
        """
        logger.info("Extracting φ=0 boundary surface...")
        
        # Extract isosurface using marching cubes
        if MCUBES_AVAILABLE:
            vertices, triangles = mcubes.marching_cubes(sdf_values, 0.0)
            
            # Transform to actual coordinates
            nx, ny, nz = sdf_values.shape
            vertices[:, 0] = vertices[:, 0] / (nx - 1) * (X.max() - X.min()) + X.min()
            vertices[:, 1] = vertices[:, 1] / (ny - 1) * (Y.max() - Y.min()) + Y.min()
            vertices[:, 2] = vertices[:, 2] / (nz - 1) * (Z.max() - Z.min()) + Z.min()
            
            logger.info(f"PyMCubes: {len(vertices):,} vertices, {len(triangles):,} triangles")
        else:
            # Fallback to scikit-image
            dx = (X.max() - X.min()) / (X.shape[0] - 1)
            dy = (Y.max() - Y.min()) / (Y.shape[1] - 1)
            dz = (Z.max() - Z.min()) / (Z.shape[2] - 1)
            
            vertices, triangles, _, _ = measure.marching_cubes(
                sdf_values, level=0.0, spacing=(dx, dy, dz)
            )
            
            vertices[:, 0] += X.min()
            vertices[:, 1] += Y.min()
            vertices[:, 2] += Z.min()
            
            logger.info(f"scikit-image: {len(vertices):,} vertices, {len(triangles):,} triangles")
        
        # Create visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the boundary surface
        ax.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            triangles=triangles,
            alpha=0.8,
            color='blue',
            shade=True,
            lightsource=plt.matplotlib.colors.LightSource(azdeg=45, altdeg=60)
        )
        
        # Styling
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12) 
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title('Immersed Boundary Surface (φ = 0)', fontsize=14, pad=20)
        
        # Set equal aspect ratio
        all_coords = vertices
        max_range = np.array([
            all_coords[:, 0].max() - all_coords[:, 0].min(),
            all_coords[:, 1].max() - all_coords[:, 1].min(),
            all_coords[:, 2].max() - all_coords[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (all_coords[:, 0].max() + all_coords[:, 0].min()) * 0.5
        mid_y = (all_coords[:, 1].max() + all_coords[:, 1].min()) * 0.5
        mid_z = (all_coords[:, 2].max() + all_coords[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Good viewing angle
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
        # Save if requested
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {output_file}")
        
        return fig

def main():
    parser = argparse.ArgumentParser(description='Simple Immersed Boundary Method')
    parser.add_argument('--mesh', required=True, help='Path to mesh file (.msh)')
    parser.add_argument('--domain_bounds', required=True, 
                       help='Domain bounds as "(xmin,ymin,zmin,xmax,ymax,zmax)"')
    parser.add_argument('--resolution', default='(50,50,50)',
                       help='Grid resolution as "(nx,ny,nz)"')
    parser.add_argument('--output', default='immersed_boundary.png',
                       help='Output image file')
    
    args = parser.parse_args()
    
    # Parse domain bounds
    domain_bounds = eval(args.domain_bounds)
    resolution = eval(args.resolution)
    
    logger.info("="*60)
    logger.info("SIMPLE IMMERSED BOUNDARY METHOD")
    logger.info("="*60)
    logger.info(f"Mesh: {args.mesh}")
    logger.info(f"Domain: {domain_bounds}")
    logger.info(f"Resolution: {resolution}")
    
    # Create immersed boundary
    ib = SimpleImmersedBoundary(args.mesh)
    
    # Compute SDF
    X, Y, Z, sdf_values = ib.compute_sdf_grid(domain_bounds, resolution)
    
    # Report SDF statistics
    logger.info(f"SDF range: [{sdf_values.min():.3f}, {sdf_values.max():.3f}]")
    logger.info(f"Inside fraction: {np.sum(sdf_values < 0) / sdf_values.size * 100:.2f}%")
    
    # Visualize boundary
    fig = ib.visualize_boundary(X, Y, Z, sdf_values, args.output)
    plt.show()
    
    logger.info("✓ Complete!")

if __name__ == "__main__":
    main() 