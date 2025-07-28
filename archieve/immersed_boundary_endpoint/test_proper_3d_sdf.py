#!/usr/bin/env python3
"""
Test Script for Proper 3D SDF Implementation

This script demonstrates:
1. Proper 3D SDF computation using angle-weighted normals
2. Professional 3D levelset contour visualization
3. Fast parallel processing
4. Accurate propeller shape representation
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from mesh_processor import GmshProcessor
from wind_tunnel_domain import WindTunnelDomain
from proper_sdf_3d import ProperSDF3D
from proper_contour_viz import ProperContourVisualizer

def test_proper_3d_sdf():
    """Test the proper 3D SDF implementation with professional visualization."""
    
    logger.info("="*80)
    logger.info("Testing Proper 3D SDF Implementation")
    logger.info("="*80)
    
    # 1. Load mesh
    logger.info("\n--- Step 1: Loading Propeller Mesh ---")
    mesh_file = Path("../mesh/5_bladed_Propeller.STEP_medium_tetrahedral.msh")
    
    processor = GmshProcessor(str(mesh_file))
    processor.read_mesh()
    
    logger.info(f"✓ Loaded mesh with {len(processor.surface_triangles):,} triangles")
    
    # 2. Create wind tunnel domain
    logger.info("\n--- Step 2: Creating Wind Tunnel Domain ---")
    wind_tunnel = WindTunnelDomain()
    
    # Get object bounds
    all_vertices = np.array(processor.surface_triangles).reshape(-1, 3)
    object_bounds = (all_vertices.min(axis=0), all_vertices.max(axis=0))
    
    tunnel_config = {
        'upstream_factor': 2.0,
        'downstream_factor': 4.0,
        'width_factor': 3.0,
        'height_factor': 3.0
    }
    
    domain_info = wind_tunnel.create_wind_tunnel_around_object(object_bounds, tunnel_config)
    
    logger.info(f"✓ Created wind tunnel domain")
    logger.info(f"  Object size: {domain_info['object_size']}")
    logger.info(f"  Domain size: {domain_info['domain_size']}")
    
    # 3. Configure grid for proper testing
    logger.info("\n--- Step 3: Configuring Test Grid ---")
    
    # Use moderate resolution for testing (can be increased for production)
    resolution = (60, 40, 40)  # ~96k points for reasonable speed
    logger.info(f"Grid resolution: {resolution} ({np.prod(resolution):,} cells)")
    
    domain_min, domain_max = domain_info['domain_bounds']
    x = np.linspace(domain_min[0], domain_max[0], resolution[0])
    y = np.linspace(domain_min[1], domain_max[1], resolution[1])
    z = np.linspace(domain_min[2], domain_max[2], resolution[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # 4. Compute SDF using proper 3D method
    logger.info("\n--- Step 4: Computing Proper 3D SDF ---")
    
    proper_sdf = ProperSDF3D(processor.surface_triangles)
    
    start_time = time.time()
    sdf_values = proper_sdf.compute_sdf_batch_parallel(grid_points)
    elapsed = time.time() - start_time
    
    logger.info(f"✓ Proper 3D SDF computation completed in {elapsed:.1f}s")
    logger.info(f"  - SDF Range: [{sdf_values.min():.3f}, {sdf_values.max():.3f}]")
    logger.info(f"  - Inside Fraction: {np.sum(sdf_values < 0) / len(sdf_values) * 100:.2f}%")
    logger.info(f"  - Processing Rate: {len(grid_points)/elapsed:.0f} pts/sec")
    
    # 5. Professional 3D Visualization
    logger.info("\n--- Step 5: Creating Professional Visualizations ---")
    
    visualizer = ProperContourVisualizer(processor)
    
    # Single φ=0 levelset (main boundary)
    logger.info("  - Creating main boundary visualization (φ=0)...")
    try:
        fig1 = visualizer.plot_professional_3d_contour(
            (X, Y, Z), sdf_values,
            levels=[0.0],
            figsize=(16, 12),
            show_mesh=True,
            alpha=0.8
        )
        fig1.suptitle("Proper 3D SDF: Propeller Boundary (φ=0)", fontsize=16)
        
        # Save and show
        output_dir = Path("output/proper_3d_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        boundary_path = output_dir / "proper_sdf_boundary.png"
        fig1.savefig(boundary_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig1)
        
        logger.info(f"    ✓ Saved to {boundary_path}")
        
    except Exception as e:
        logger.error(f"    ✗ Main boundary visualization failed: {e}")
    
    # Multiple levelsets to show field structure
    logger.info("  - Creating multi-level field visualization...")
    try:
        fig2 = visualizer.plot_multiple_levelsets(
            (X, Y, Z), sdf_values,
            num_levels=5,
            figsize=(18, 12)
        )
        fig2.suptitle("Proper 3D SDF: Multi-Level Field Structure", fontsize=16)
        
        multilevel_path = output_dir / "proper_sdf_multilevel.png"
        fig2.savefig(multilevel_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig2)
        
        logger.info(f"    ✓ Saved to {multilevel_path}")
        
    except Exception as e:
        logger.error(f"    ✗ Multi-level visualization failed: {e}")
    
    # Export isosurface as mesh
    logger.info("  - Exporting isosurface mesh...")
    try:
        mesh_path = output_dir / "propeller_isosurface.obj"
        success = visualizer.export_isosurface_mesh(
            (X, Y, Z), sdf_values,
            level=0.0,
            filename=str(mesh_path)
        )
        
        if success:
            logger.info(f"    ✓ Exported isosurface mesh to {mesh_path}")
        else:
            logger.warning("    ✗ Mesh export failed")
            
    except Exception as e:
        logger.error(f"    ✗ Mesh export failed: {e}")
    
    # 6. Performance and Quality Analysis
    logger.info("\n--- Step 6: Analysis Results ---")
    
    # Check SDF quality
    zero_crossings = np.sum(np.abs(sdf_values) < 0.1)  # Points very close to boundary
    logger.info(f"  - Points near boundary (|φ| < 0.1): {zero_crossings:,} ({zero_crossings/len(sdf_values)*100:.2f}%)")
    
    # Check inside/outside ratio
    inside_points = np.sum(sdf_values < 0)
    outside_points = np.sum(sdf_values > 0)
    logger.info(f"  - Inside points: {inside_points:,} ({inside_points/len(sdf_values)*100:.2f}%)")
    logger.info(f"  - Outside points: {outside_points:,} ({outside_points/len(sdf_values)*100:.2f}%)")
    
    # Performance metrics
    points_per_second = len(grid_points) / elapsed
    logger.info(f"  - Performance: {points_per_second:.0f} points/second")
    logger.info(f"  - Grid density: {np.prod(resolution)/1e6:.2f}M points")
    
    logger.info("\n" + "="*80)
    logger.info("✓ Proper 3D SDF Test Completed Successfully!")
    logger.info("="*80)
    
    return {
        'sdf_values': sdf_values,
        'grid_coords': (X, Y, Z),
        'performance': points_per_second,
        'inside_fraction': inside_points / len(sdf_values),
        'output_dir': output_dir
    }

def compare_sdf_methods():
    """Compare the proper 3D SDF with the simple heuristic method."""
    
    logger.info("\n" + "="*80)
    logger.info("Comparing SDF Methods")
    logger.info("="*80)
    
    # Load mesh
    mesh_file = Path("../mesh/5_bladed_Propeller.STEP_medium_tetrahedral.msh")
    processor = GmshProcessor(str(mesh_file))
    processor.read_mesh()
    
    # Create small test grid for comparison
    all_vertices = np.array(processor.surface_triangles).reshape(-1, 3)
    bounds_min, bounds_max = all_vertices.min(axis=0), all_vertices.max(axis=0)
    
    # Small grid around the propeller
    margin = (bounds_max - bounds_min) * 0.2
    test_min = bounds_min - margin
    test_max = bounds_max + margin
    
    resolution = (30, 20, 20)  # Small for comparison
    x = np.linspace(test_min[0], test_max[0], resolution[0])
    y = np.linspace(test_min[1], test_max[1], resolution[1])
    z = np.linspace(test_min[2], test_max[2], resolution[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    logger.info(f"Comparing methods on {len(grid_points):,} test points")
    
    # Method 1: Simple heuristic (from simple_sdf.py)
    logger.info("\n--- Method 1: Simple Heuristic SDF ---")
    try:
        from simple_sdf import SimpleSDF
        simple_sdf = SimpleSDF(processor.surface_triangles)
        
        start_time = time.time()
        simple_values = simple_sdf.compute_sdf_batch_parallel(grid_points)
        simple_time = time.time() - start_time
        
        logger.info(f"Simple method: {simple_time:.2f}s, {len(grid_points)/simple_time:.0f} pts/sec")
        logger.info(f"Simple SDF range: [{simple_values.min():.3f}, {simple_values.max():.3f}]")
        logger.info(f"Simple inside fraction: {np.sum(simple_values < 0)/len(simple_values)*100:.2f}%")
        
    except Exception as e:
        logger.error(f"Simple method failed: {e}")
        simple_values = None
    
    # Method 2: Proper 3D SDF
    logger.info("\n--- Method 2: Proper 3D SDF ---")
    try:
        proper_sdf = ProperSDF3D(processor.surface_triangles)
        
        start_time = time.time()
        proper_values = proper_sdf.compute_sdf_batch_parallel(grid_points)
        proper_time = time.time() - start_time
        
        logger.info(f"Proper method: {proper_time:.2f}s, {len(grid_points)/proper_time:.0f} pts/sec")
        logger.info(f"Proper SDF range: [{proper_values.min():.3f}, {proper_values.max():.3f}]")
        logger.info(f"Proper inside fraction: {np.sum(proper_values < 0)/len(proper_values)*100:.2f}%")
        
    except Exception as e:
        logger.error(f"Proper method failed: {e}")
        proper_values = None
    
    # Compare results
    if simple_values is not None and proper_values is not None:
        logger.info("\n--- Comparison Results ---")
        
        # Correlation
        correlation = np.corrcoef(simple_values, proper_values)[0, 1]
        logger.info(f"Correlation coefficient: {correlation:.4f}")
        
        # Difference statistics
        diff = np.abs(simple_values - proper_values)
        logger.info(f"Mean absolute difference: {diff.mean():.4f}")
        logger.info(f"Max absolute difference: {diff.max():.4f}")
        
        # Sign agreement
        simple_signs = np.sign(simple_values)
        proper_signs = np.sign(proper_values)
        agreement = np.sum(simple_signs == proper_signs) / len(simple_signs)
        logger.info(f"Sign agreement: {agreement*100:.2f}%")
        
        # Speed comparison
        speed_ratio = simple_time / proper_time
        logger.info(f"Speed ratio (simple/proper): {speed_ratio:.2f}x")
    
    logger.info("="*80)

if __name__ == "__main__":
    # Install PyMCubes if not available
    try:
        import mcubes
    except ImportError:
        logger.warning("PyMCubes not found. Install with: pip install PyMCubes")
        logger.info("Falling back to scikit-image for marching cubes")
    
    # Run main test
    results = test_proper_3d_sdf()
    
    # Run comparison (optional)
    try:
        compare_sdf_methods()
    except Exception as e:
        logger.warning(f"Comparison test failed: {e}")
    
    logger.info("\n🎉 All tests completed! Check the output directory for results.") 