#!/usr/bin/env python3
"""
Main SDF Generation Workflow

This script encapsulates the full process from mesh to JAX-Fluids config.
It is designed to be called from the main `run.py` script.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import sys

# Add package directory to path to ensure imports work
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from mesh_processor import GmshProcessor
from wind_tunnel_domain import WindTunnelDomain
from visualization import SDFVisualizer
from simple_sdf import SimpleSDF

logger = logging.getLogger(__name__)

def run_sdf_workflow(mesh_file: str,
                     output_dir_base: str,
                     run_name: str,
                     target_cells_per_diameter: int,
                     max_cells_total: int,
                     sdf_refinement_factor: float):
    """
    Execute the complete SDF generation workflow.
    
    Args:
        mesh_file: Path to the input Gmsh mesh file.
        output_dir_base: Base directory for output.
        run_name: Unique name for this run (e.g., 'fast_run').
        target_cells_per_diameter: Base grid resolution.
        max_cells_total: Maximum allowed grid cells.
        sdf_refinement_factor: Multiplier for grid fineness.
    """
    
    output_dir = Path(output_dir_base) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info(f"Starting SDF Workflow: {run_name}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("="*80)
    
    try:
        # 1. Load Mesh
        logger.info("\n--- Step 1: Loading Mesh ---")
        mesh_path = Path(mesh_file)
        if not mesh_path.exists():
            logger.error(f"Mesh file not found at: {mesh_path.resolve()}")
            return
            
        processor = GmshProcessor(str(mesh_path))
        processor.read_mesh()
        
        object_bounds = processor.get_mesh_bounds()
        logger.info(f"✓ Loaded {len(processor.surface_triangles)} triangles")
        logger.info(f"  - Object Bounds: {object_bounds}")
        
        # 2. Create Wind Tunnel
        logger.info("\n--- Step 2: Creating Wind Tunnel Domain ---")
        wind_tunnel = WindTunnelDomain()
        
        tunnel_config = {
            'upstream_length_factor': 4.0, 'downstream_length_factor': 8.0,
            'width_factor': 4.0, 'height_factor': 4.0, 'flow_direction': 'x'
        }
        
        domain_info = wind_tunnel.create_wind_tunnel_around_object(
            object_bounds, tunnel_config
        )
        logger.info(f"✓ Wind tunnel created. Domain size: {domain_info['domain_size']}")
        
        # 3. Configure Grid
        logger.info("\n--- Step 3: Configuring Grid Resolution ---")
        resolution = wind_tunnel.suggest_grid_resolution(
            domain_info,
            target_cells_per_diameter=target_cells_per_diameter,
            max_cells_total=max_cells_total,
            sdf_refinement_factor=sdf_refinement_factor
        )
        logger.info(f"✓ Grid resolution set to: {resolution} ({np.prod(resolution):,} cells)")

        # 4. Compute SDF
        logger.info("\n--- Step 4: Computing Signed Distance Function ---")
        simple_sdf = SimpleSDF(processor.surface_triangles)
        
        domain_min, domain_max = domain_info['domain_bounds']
        x = np.linspace(domain_min[0], domain_max[0], resolution[0])
        y = np.linspace(domain_min[1], domain_max[1], resolution[1])
        z = np.linspace(domain_min[2], domain_max[2], resolution[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        logger.info(f"Computing SDF on {len(grid_points):,} grid points...")
        
        # Use parallel processing for large grids, single-threaded for small ones
        if len(grid_points) > 50000:
            logger.info("Using parallel processing for large grid...")
            sdf_values = simple_sdf.compute_sdf_batch_parallel(grid_points)
        else:
            logger.info("Using single-threaded processing for small grid...")
            # Batch processing with progress tracking for smaller grids
            sdf_values = []
            batch_size = 5000
            total_points = len(grid_points)
            
            for i in range(0, total_points, batch_size):
                batch_end = min(i + batch_size, total_points)
                batch_points = grid_points[i:batch_end]
                
                batch_sdf = [simple_sdf.compute_sdf(p) for p in batch_points]
                sdf_values.extend(batch_sdf)
                
                progress = (i + len(batch_points)) / total_points * 100
                logger.info(f"  SDF Progress: {progress:.1f}% ({i + len(batch_points):,}/{total_points:,})")
            
            sdf_values = np.array(sdf_values)
        
        logger.info("✓ SDF computation completed.")
        logger.info(f"  - SDF Range: [{sdf_values.min():.3f}, {sdf_values.max():.3f}]")
        logger.info(f"  - Inside Fraction: {np.sum(sdf_values < 0) / len(sdf_values) * 100:.2f}%")
        
        # 5. Visualize Results with Enhanced Plotting
        logger.info("\n--- Step 5: Generating High-Quality Visualizations ---")
        visualizer = SDFVisualizer(processor, None)
        
        # High-quality 3D SDF contour surface (φ=0)
        logger.info("  - Creating high-quality 3D SDF contour surface...")
        try:
            fig_contour = visualizer.plot_sdf_3d_contour(
                (X, Y, Z), sdf_values, contour_level=0.0,
                figsize=(16, 12), alpha=0.8, show_mesh=True,
                smooth_surface=True, high_quality=True
            )
            contour_path = output_dir / "sdf_3d_contour_hq.png"
            fig_contour.savefig(contour_path, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig_contour)
            logger.info(f"    ✓ High-quality 3D SDF contour saved to {contour_path}")
        except Exception as e:
            logger.error(f"    ✗ Could not create high-quality 3D contour: {e}", exc_info=True)
        
        # Advanced multi-level SDF visualization
        logger.info("  - Creating advanced multi-level SDF visualization...")
        try:
            fig_advanced = visualizer.plot_sdf_advanced_contour(
                (X, Y, Z), sdf_values,
                main_level=0.0,
                additional_levels=None,  # Auto-select meaningful levels
                figsize=(18, 12),
                show_mesh=True
            )
            advanced_path = output_dir / "sdf_advanced_multilevel.png"
            fig_advanced.savefig(advanced_path, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig_advanced)
            logger.info(f"    ✓ Advanced multi-level SDF visualization saved to {advanced_path}")
        except Exception as e:
            logger.error(f"    ✗ Could not create advanced visualization: {e}", exc_info=True)

        # Cross-sections
        logger.info("  - Creating cross-sections...")
        # ... (visualization code for cross-sections can be added here)

        # 6. Export for JAX-Fluids
        logger.info("\n--- Step 6: Exporting JAX-Fluids Configuration ---")
        # ... (exporting code can be added here)
        
        logger.info("\n" + "="*80)
        logger.info(f"Workflow '{run_name}' Completed Successfully!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)

if __name__ == '__main__':
    # This allows running the test directly for debugging
    run_sdf_workflow(
        mesh_file='../../mesh/5_bladed_Propeller.STEP_medium_tetrahedral.msh',
        output_dir_base='../output',
        run_name='direct_test_run',
        target_cells_per_diameter=15,
        max_cells_total=100000,
        sdf_refinement_factor=1.5
    ) 