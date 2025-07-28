#!/usr/bin/env python3
"""
Quick visualization of existing SDF data with improved plotting
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from mesh_processor import GmshProcessor
from wind_tunnel_domain import WindTunnelDomain
from simple_sdf import SimpleSDF
from visualization import SDFVisualizer

def visualize_existing_sdf_data():
    """Load and visualize the most recent SDF computation with improved plotting."""
    
    # Look for the most recent output directory
    output_base = Path("output")
    if not output_base.exists():
        logger.error("No output directory found!")
        return
    
    # Find the most recent directory
    output_dirs = [d for d in output_base.iterdir() if d.is_dir()]
    if not output_dirs:
        logger.error("No output directories found!")
        return
    
    latest_dir = max(output_dirs, key=lambda x: x.stat().st_mtime)
    logger.info(f"Using latest output directory: {latest_dir}")
    
    # Try to find saved data files
    possible_files = [
        "sdf_data.npz",
        "grid_data.npz", 
        "domain_info.pkl",
        "sdf_values.npy",
        "grid_coords.npy"
    ]
    
    found_files = {}
    for filename in possible_files:
        filepath = latest_dir / filename
        if filepath.exists():
            found_files[filename] = filepath
            logger.info(f"Found: {filename}")
    
    if not found_files:
        logger.warning("No saved data files found. Let me recreate the data...")
        # Recreate the computation setup
        mesh_file = Path("../mesh/5_bladed_Propeller.STEP_medium_tetrahedral.msh")
        
        # Load mesh
        processor = GmshProcessor(str(mesh_file))
        processor.read_mesh()
        
        # Create wind tunnel domain  
        wind_tunnel = WindTunnelDomain()
        
        # Get object bounds from triangles
        all_vertices = np.array(processor.surface_triangles).reshape(-1, 3)
        object_bounds = (all_vertices.min(axis=0), all_vertices.max(axis=0))
        
        tunnel_config = {
            'upstream_factor': 3.0,
            'downstream_factor': 5.0,
            'width_factor': 4.0,
            'height_factor': 4.0
        }
        
        domain_info = wind_tunnel.create_wind_tunnel_around_object(
            object_bounds,
            tunnel_config
        )
        
        # Use production-like resolution
        resolution = wind_tunnel.suggest_grid_resolution(
            domain_info,
            target_cells_per_diameter=15,
            max_cells_total=1000000,
            sdf_refinement_factor=1.5
        )
        
        logger.info(f"Grid resolution: {resolution} ({np.prod(resolution):,} cells)")
        
        # Create grid
        domain_min, domain_max = domain_info['domain_bounds']
        x = np.linspace(domain_min[0], domain_max[0], resolution[0])
        y = np.linspace(domain_min[1], domain_max[1], resolution[1]) 
        z = np.linspace(domain_min[2], domain_max[2], resolution[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Use a smaller subset for quick visualization
        logger.info("Creating subset for quick visualization...")
        step = 2  # Use every 2nd point in each dimension
        X_sub = X[::step, ::step, ::step]
        Y_sub = Y[::step, ::step, ::step]
        Z_sub = Z[::step, ::step, ::step]
        
        grid_points = np.column_stack([X_sub.ravel(), Y_sub.ravel(), Z_sub.ravel()])
        logger.info(f"Computing SDF on {len(grid_points):,} grid points (subset)...")
        
        # Compute SDF
        simple_sdf = SimpleSDF(processor.surface_triangles)
        sdf_values = simple_sdf.compute_sdf_batch_parallel(grid_points)
        
        # Reshape for visualization
        sdf_3d = sdf_values.reshape(X_sub.shape)
        
        X, Y, Z = X_sub, Y_sub, Z_sub
        
    else:
        logger.info("Loading existing data...")
        # Load the mesh for visualization
        mesh_file = Path("../mesh/5_bladed_Propeller.STEP_medium_tetrahedral.msh")
        processor = GmshProcessor(str(mesh_file))
        processor.read_mesh()
        
        # Try to reconstruct the grid and SDF data
        # This is a simplified approach - in practice you'd save/load this properly
        logger.warning("Using simplified data reconstruction...")
        
        # Create a reasonable grid for visualization
        wind_tunnel = WindTunnelDomain()
        
        # Get object bounds from triangles
        all_vertices = np.array(processor.surface_triangles).reshape(-1, 3)
        object_bounds = (all_vertices.min(axis=0), all_vertices.max(axis=0))
        
        tunnel_config = {
            'upstream_factor': 3.0,
            'downstream_factor': 5.0,
            'width_factor': 4.0,
            'height_factor': 4.0
        }
        
        domain_info = wind_tunnel.create_wind_tunnel_around_object(
            object_bounds,
            tunnel_config
        )
        
        # Use moderate resolution for visualization
        resolution = (80, 60, 60)  # ~288k points
        logger.info(f"Using visualization grid: {resolution} ({np.prod(resolution):,} cells)")
        
        domain_min, domain_max = domain_info['domain_bounds']
        x = np.linspace(domain_min[0], domain_max[0], resolution[0])
        y = np.linspace(domain_min[1], domain_max[1], resolution[1])
        z = np.linspace(domain_min[2], domain_max[2], resolution[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # Compute SDF with parallel processing
        simple_sdf = SimpleSDF(processor.surface_triangles)
        sdf_values = simple_sdf.compute_sdf_batch_parallel(grid_points)
    
    logger.info(f"SDF range: [{sdf_values.min():.3f}, {sdf_values.max():.3f}]")
    logger.info(f"Inside fraction: {np.sum(sdf_values < 0) / len(sdf_values) * 100:.2f}%")
    
    # Create high-quality visualizations
    logger.info("Creating high-quality visualizations...")
    visualizer = SDFVisualizer(processor, None)
    
    # High-quality single contour
    logger.info("1. High-quality φ=0 contour...")
    try:
        fig1 = visualizer.plot_sdf_3d_contour(
            (X, Y, Z), sdf_values,
            contour_level=0.0,
            figsize=(16, 12),
            alpha=0.8,
            show_mesh=True,
            smooth_surface=True,
            high_quality=True
        )
        fig1.suptitle("High-Quality SDF Boundary (φ=0)", fontsize=16)
        plt.show()
        
        # Save it
        output_path = latest_dir / "hq_sdf_boundary.png"
        fig1.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to: {output_path}")
        plt.close(fig1)
    except Exception as e:
        logger.error(f"High-quality visualization failed: {e}")
    
    # Advanced multi-level visualization
    logger.info("2. Advanced multi-level contours...")
    try:
        fig2 = visualizer.plot_sdf_advanced_contour(
            (X, Y, Z), sdf_values,
            main_level=0.0,
            additional_levels=None,
            figsize=(18, 12),
            show_mesh=True
        )
        fig2.suptitle("Advanced Multi-Level SDF Visualization", fontsize=16)
        plt.show()
        
        # Save it
        output_path2 = latest_dir / "advanced_multilevel_sdf.png"
        fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to: {output_path2}")
        plt.close(fig2)
    except Exception as e:
        logger.error(f"Advanced visualization failed: {e}")
    
    logger.info("✓ Visualization complete!")

if __name__ == "__main__":
    visualize_existing_sdf_data() 