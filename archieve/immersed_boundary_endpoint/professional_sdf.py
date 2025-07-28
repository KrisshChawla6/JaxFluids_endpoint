#!/usr/bin/env python3
"""
Professional Immersed Boundary Method using pysdf

Uses the production-quality pysdf library (https://github.com/sxyu/sdf)
for NASA-grade signed distance function computation.

Usage:
    python professional_sdf.py --mesh ../mesh/propeller.msh --domain_bounds "(-100,-100,-100,100,100,100)" --resolution "(50,50,50)"
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
import argparse
import time
from pathlib import Path
from typing import Tuple, List, Optional
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    from pysdf import SDF
    logger.info("Using production-quality pysdf library")
except ImportError:
    logger.error("pysdf library not found! Install with: pip install pysdf")
    raise

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

class ProfessionalImmersedBoundary:
    """
    Professional immersed boundary method using production-quality pysdf library.
    
    This provides NASA-grade signed distance function computation with:
    - Parallelized triangle-to-point distance computation
    - Robust inside/outside determination
    - Spatial acceleration structures
    - Proper handling of self-intersections and non-watertight meshes
    """
    
    def __init__(self, mesh_file: str):
        """Initialize with mesh file."""
        self.mesh_processor = GmshProcessor(mesh_file)
        self.mesh_processor.read_mesh()
        
        # Extract vertices and faces for pysdf
        vertices = []
        faces = []
        vertex_map = {}
        
        logger.info("Processing mesh for pysdf...")
        
        # Build vertex list and map
        vertex_idx = 0
        for triangle in self.mesh_processor.surface_triangles:
            for vertex in triangle:
                vertex_tuple = tuple(vertex)
                if vertex_tuple not in vertex_map:
                    vertices.append(vertex)
                    vertex_map[vertex_tuple] = vertex_idx
                    vertex_idx += 1
        
        # Build face list
        for triangle in self.mesh_processor.surface_triangles:
            face = []
            for vertex in triangle:
                vertex_tuple = tuple(vertex)
                face.append(vertex_map[vertex_tuple])
            faces.append(face)
        
        self.vertices = np.array(vertices, dtype=np.float32)
        self.faces = np.array(faces, dtype=np.uint32)
        
        logger.info(f"Mesh processed: {len(self.vertices):,} vertices, {len(self.faces):,} faces")
        
        # Initialize pysdf (production-quality SDF computation)
        logger.info("Initializing production-quality SDF computation...")
        start_time = time.time()
        self.sdf = SDF(self.vertices, self.faces)
        init_time = time.time() - start_time
        
        logger.info(f"✓ SDF initialized in {init_time:.3f}s")
        logger.info(f"Surface area: {self.sdf.surface_area:.3f}")
        
        # Get mesh bounds for reference
        self.mesh_bounds = {
            'min': self.vertices.min(axis=0),
            'max': self.vertices.max(axis=0),
            'center': self.vertices.mean(axis=0),
            'size': self.vertices.max(axis=0) - self.vertices.min(axis=0)
        }
        
        logger.info(f"Mesh bounds: [{self.mesh_bounds['min']}, {self.mesh_bounds['max']}]")
        logger.info(f"Mesh size: {self.mesh_bounds['size']}")
    
    def compute_sdf_grid(self, domain_bounds: Tuple[float, float, float, float, float, float],
                        resolution: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute SDF on a regular grid using production-quality pysdf library.
        
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
        
        # Flatten grid points for batch computation
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)
        
        logger.info("Computing SDF values using production-quality pysdf...")
        start_time = time.time()
        
        # Use pysdf for fast, parallelized SDF computation
        sdf_flat = self.sdf(points)
        
        elapsed = time.time() - start_time
        rate = len(points) / elapsed
        
        logger.info(f"✓ SDF computation completed in {elapsed:.3f}s ({rate:.0f} pts/sec)")
        
        # Reshape back to grid
        sdf_values = sdf_flat.reshape(X.shape)
        
        return X, Y, Z, sdf_values
    
    def visualize_boundary(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                          sdf_values: np.ndarray, output_file: str = None) -> plt.Figure:
        """
        Visualize the φ=0 boundary surface using high-quality marching cubes.
        
        Args:
            X, Y, Z: Grid coordinates
            sdf_values: SDF values
            output_file: Optional output file path
            
        Returns:
            Matplotlib figure
        """
        logger.info("Extracting φ=0 boundary surface using marching cubes...")
        
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
        
        # Create professional visualization
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the boundary surface with professional styling
        surface = ax.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            triangles=triangles,
            alpha=0.85,
            color='steelblue',
            shade=True,
            lightsource=plt.matplotlib.colors.LightSource(azdeg=45, altdeg=60),
            linewidth=0.1,
            edgecolor='darkblue'
        )
        
        # Professional styling
        ax.set_xlabel('X [units]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y [units]', fontsize=12, fontweight='bold') 
        ax.set_zlabel('Z [units]', fontsize=12, fontweight='bold')
        ax.set_title('Professional Immersed Boundary Surface (φ = 0)\nUsing Production-Quality pysdf Library', 
                    fontsize=14, fontweight='bold', pad=20)
        
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
        
        # Professional viewing angle
        ax.view_init(elev=25, azim=45)
        
        # Add grid and professional appearance
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        
        plt.tight_layout()
        
        # Save if requested
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved professional visualization to {output_file}")
        
        return fig
    
    def export_for_jaxfluids(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                           sdf_values: np.ndarray, output_file: str):
        """
        Export SDF data in a format suitable for JAX-Fluids.
        
        Args:
            X, Y, Z: Grid coordinates
            sdf_values: SDF values
            output_file: Output file path (JSON format)
        """
        logger.info("Exporting SDF data for JAX-Fluids...")
        
        # Prepare data structure
        export_data = {
            'metadata': {
                'description': 'Professional SDF for JAX-Fluids immersed boundary',
                'library': 'pysdf (https://github.com/sxyu/sdf)',
                'mesh_vertices': int(len(self.vertices)),
                'mesh_faces': int(len(self.faces)),
                'surface_area': float(self.sdf.surface_area),
                'grid_resolution': list(sdf_values.shape),
                'domain_bounds': [
                    float(X.min()), float(Y.min()), float(Z.min()),
                    float(X.max()), float(Y.max()), float(Z.max())
                ],
                'sdf_range': [float(sdf_values.min()), float(sdf_values.max())],
                'inside_fraction': float(np.sum(sdf_values < 0) / sdf_values.size),
            },
            'grid': {
                'x': X[:, 0, 0].astype(float).tolist(),
                'y': Y[0, :, 0].astype(float).tolist(),
                'z': Z[0, 0, :].astype(float).tolist(),
            },
            'sdf_values': sdf_values.astype(float).tolist(),
            'mesh_bounds': {
                'min': self.mesh_bounds['min'].astype(float).tolist(),
                'max': self.mesh_bounds['max'].astype(float).tolist(),
                'center': self.mesh_bounds['center'].astype(float).tolist(),
                'size': self.mesh_bounds['size'].astype(float).tolist(),
            }
        }
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"✓ JAX-Fluids data exported to {output_file}")
        logger.info(f"  Grid: {sdf_values.shape}")
        logger.info(f"  SDF range: [{sdf_values.min():.3f}, {sdf_values.max():.3f}]")
        logger.info(f"  Inside fraction: {np.sum(sdf_values < 0) / sdf_values.size * 100:.2f}%")
    
    def generate_offset_surfaces(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                               sdf_values: np.ndarray, offsets: List[float], 
                               output_dir: str):
        """
        Generate offset surfaces at specified distances.
        
        Args:
            X, Y, Z: Grid coordinates
            sdf_values: SDF values
            offsets: List of offset distances
            output_dir: Output directory for .obj files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Generating {len(offsets)} offset surfaces...")
        
        for i, offset in enumerate(offsets):
            logger.info(f"Generating offset surface at distance {offset:.3f}...")
            
            if MCUBES_AVAILABLE:
                vertices, triangles = mcubes.marching_cubes(sdf_values, offset)
                
                # Transform to actual coordinates
                nx, ny, nz = sdf_values.shape
                vertices[:, 0] = vertices[:, 0] / (nx - 1) * (X.max() - X.min()) + X.min()
                vertices[:, 1] = vertices[:, 1] / (ny - 1) * (Y.max() - Y.min()) + Y.min()
                vertices[:, 2] = vertices[:, 2] / (nz - 1) * (Z.max() - Z.min()) + Z.min()
                
                # Export as OBJ
                obj_file = os.path.join(output_dir, f"offset_{offset:+.3f}.obj")
                with open(obj_file, 'w') as f:
                    f.write(f"# Offset surface at distance {offset:.3f}\n")
                    f.write(f"# Generated using professional pysdf library\n")
                    f.write(f"# Vertices: {len(vertices):,}, Triangles: {len(triangles):,}\n\n")
                    
                    # Write vertices
                    for v in vertices:
                        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                    
                    # Write faces (OBJ uses 1-based indexing)
                    for t in triangles:
                        f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
                
                logger.info(f"  ✓ Saved {obj_file} ({len(vertices):,} vertices, {len(triangles):,} triangles)")
        
        logger.info(f"✓ All offset surfaces saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Professional Immersed Boundary Method using pysdf')
    parser.add_argument('--mesh', required=True, help='Path to mesh file (.msh)')
    parser.add_argument('--domain_bounds', required=True, 
                       help='Domain bounds as "(xmin,ymin,zmin,xmax,ymax,zmax)"')
    parser.add_argument('--resolution', default='(50,50,50)',
                       help='Grid resolution as "(nx,ny,nz)"')
    parser.add_argument('--output', default='professional_boundary.png',
                       help='Output image file')
    parser.add_argument('--export_jaxfluids', default='sdf_data.json',
                       help='Export file for JAX-Fluids')
    parser.add_argument('--offset_surfaces', default='',
                       help='Comma-separated offset distances for surface generation')
    parser.add_argument('--offset_dir', default='offset_surfaces',
                       help='Directory for offset surface files')
    
    args = parser.parse_args()
    
    # Parse arguments
    domain_bounds = eval(args.domain_bounds)
    resolution = eval(args.resolution)
    
    logger.info("="*70)
    logger.info("PROFESSIONAL IMMERSED BOUNDARY METHOD")
    logger.info("Using Production-Quality pysdf Library")
    logger.info("="*70)
    logger.info(f"Mesh: {args.mesh}")
    logger.info(f"Domain: {domain_bounds}")
    logger.info(f"Resolution: {resolution}")
    
    # Create professional immersed boundary
    ib = ProfessionalImmersedBoundary(args.mesh)
    
    # Compute SDF using production-quality pysdf
    X, Y, Z, sdf_values = ib.compute_sdf_grid(domain_bounds, resolution)
    
    # Report SDF statistics
    logger.info(f"SDF range: [{sdf_values.min():.6f}, {sdf_values.max():.6f}]")
    logger.info(f"Inside fraction: {np.sum(sdf_values < 0) / sdf_values.size * 100:.2f}%")
    logger.info(f"Zero-crossings: {np.sum(np.abs(sdf_values) < 1e-6):,}")
    
    # Visualize boundary
    fig = ib.visualize_boundary(X, Y, Z, sdf_values, args.output)
    
    # Export for JAX-Fluids
    ib.export_for_jaxfluids(X, Y, Z, sdf_values, args.export_jaxfluids)
    
    # Generate offset surfaces if requested
    if args.offset_surfaces:
        offsets = [float(x.strip()) for x in args.offset_surfaces.split(',')]
        ib.generate_offset_surfaces(X, Y, Z, sdf_values, offsets, args.offset_dir)
    
    # Show visualization
    plt.show()
    
    logger.info("✓ Professional immersed boundary computation complete!")

if __name__ == "__main__":
    main() 