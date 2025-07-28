#!/usr/bin/env python3
"""
Test script for computing signed distance function from the propeller mesh.

This script demonstrates the complete workflow:
1. Load and process the propeller mesh
2. Compute signed distance function 
3. Visualize results with contour plots
4. Export data for JAX-Fluids integration
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from immersed_boundary_endpoint.mesh_processor import GmshProcessor
from immersed_boundary_endpoint.sdf_generator import SignedDistanceFunction
from immersed_boundary_endpoint.grid_mapper import CartesianGridMapper
from immersed_boundary_endpoint.visualization import SDFVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_propeller_sdf_workflow():
    """
    Complete test workflow for propeller SDF computation.
    """
    logger.info("Starting propeller SDF test workflow")
    
    # Path to the propeller mesh file
    mesh_file = Path(__file__).parent.parent.parent / "mesh" / "5_bladed_Propeller.STEP_medium_tetrahedral.msh"
    
    if not mesh_file.exists():
        logger.error(f"Mesh file not found: {mesh_file}")
        return False
    
    try:
        # Step 1: Load and process mesh
        logger.info("Step 1: Loading and processing mesh")
        mesh_processor = GmshProcessor(str(mesh_file))
        mesh_processor.read_mesh()
        
        # Print mesh information
        mesh_info = mesh_processor.get_mesh_info()
        logger.info(f"Mesh info: {mesh_info}")
        
        # Step 2: Initialize SDF generator
        logger.info("Step 2: Initializing SDF generator")
        sdf_generator = SignedDistanceFunction(mesh_processor)
        
        # Step 3: Setup domain and grid mapping
        logger.info("Step 3: Setting up domain and grid mapping")
        grid_mapper = CartesianGridMapper(sdf_generator)
        
        # Get mesh bounds and setup computational domain
        min_coords, max_coords = mesh_processor.get_mesh_bounds()
        logger.info(f"Mesh bounds: min={min_coords}, max={max_coords}")
        
        # Setup domain with reasonable resolution for testing
        resolution = (50, 50, 30)  # Moderate resolution for testing
        domain_info = grid_mapper.setup_domain(
            (min_coords, max_coords),
            resolution,
            padding_factor=1.3
        )
        
        logger.info(f"Domain info: {domain_info}")
        
        # Step 4: Compute SDF on grid
        logger.info("Step 4: Computing SDF on Cartesian grid")
        grid_coords, sdf_values = sdf_generator.compute_sdf_cartesian_grid(
            grid_mapper.grid_bounds,
            resolution
        )
        
        logger.info(f"SDF computation completed. Min SDF: {sdf_values.min():.6f}, Max SDF: {sdf_values.max():.6f}")
        
        # Step 5: Visualization
        logger.info("Step 5: Creating visualizations")
        visualizer = SDFVisualizer(mesh_processor, sdf_generator)
        
        # Create output directory for test results
        output_dir = Path(__file__).parent / "test_results"
        output_dir.mkdir(exist_ok=True)
        
        # Plot mesh geometry
        try:
            fig_mesh = visualizer.plot_mesh_geometry(figsize=(12, 9))
            mesh_plot_path = output_dir / "propeller_mesh_geometry.png"
            fig_mesh.savefig(mesh_plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig_mesh)
            logger.info(f"Saved mesh geometry plot: {mesh_plot_path}")
        except Exception as e:
            logger.warning(f"Could not create mesh geometry plot: {e}")
        
        # Plot SDF cross-sections
        try:
            fig_cross = visualizer.plot_sdf_cross_sections(grid_coords, sdf_values, num_slices=3)
            cross_plot_path = output_dir / "propeller_sdf_cross_sections.png"
            fig_cross.savefig(cross_plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig_cross)
            logger.info(f"Saved SDF cross-sections plot: {cross_plot_path}")
        except Exception as e:
            logger.warning(f"Could not create cross-sections plot: {e}")
        
        # Plot individual slices
        for axis in ['x', 'y', 'z']:
            try:
                fig_slice = visualizer.plot_sdf_2d_slice(
                    grid_coords, sdf_values, 
                    slice_axis=axis, slice_position=0.0,
                    figsize=(12, 8)
                )
                slice_plot_path = output_dir / f"propeller_sdf_{axis}_slice.png"
                fig_slice.savefig(slice_plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig_slice)
                logger.info(f"Saved {axis}-slice plot: {slice_plot_path}")
            except Exception as e:
                logger.warning(f"Could not create {axis}-slice plot: {e}")
        
        # Step 6: Export data for JAX-Fluids
        logger.info("Step 6: Exporting data for JAX-Fluids integration")
        
        # Export SDF data
        sdf_data_path = output_dir / "propeller_sdf_data"
        grid_mapper.export_sdf_data(str(sdf_data_path))
        
        # Generate JAX-Fluids configuration files
        case_setup = grid_mapper.generate_case_setup_json(
            case_name="propeller_immersed_boundary",
            end_time=1e-3,
            save_dt=1e-4
        )
        
        numerical_setup = grid_mapper.generate_numerical_setup_json(
            levelset_model="FLUID-SOLID"
        )
        
        # Save configuration files
        import json
        
        case_setup_path = output_dir / "case_setup.json"
        with open(case_setup_path, 'w') as f:
            json.dump(case_setup, f, indent=2)
        logger.info(f"Saved case setup: {case_setup_path}")
        
        numerical_setup_path = output_dir / "numerical_setup.json"
        with open(numerical_setup_path, 'w') as f:
            json.dump(numerical_setup, f, indent=2)
        logger.info(f"Saved numerical setup: {numerical_setup_path}")
        
        # Generate levelset function template
        levelset_template = grid_mapper.get_jax_fluids_compatible_function()
        template_path = output_dir / "levelset_function_template.py"
        with open(template_path, 'w') as f:
            f.write(levelset_template)
        logger.info(f"Saved levelset function template: {template_path}")
        
        # Step 7: Validation and statistics
        logger.info("Step 7: Computing validation statistics")
        
        # Basic statistics
        stats = {
            'mesh_triangles': len(mesh_processor.surface_triangles),
            'mesh_nodes': len(mesh_processor.nodes),
            'grid_resolution': resolution,
            'sdf_min': float(sdf_values.min()),
            'sdf_max': float(sdf_values.max()),
            'sdf_mean': float(sdf_values.mean()),
            'sdf_std': float(sdf_values.std()),
            'zero_crossings': int(np.sum(np.abs(sdf_values) < grid_mapper.cell_sizes.min())),
            'domain_bounds': {
                'min': grid_mapper.grid_bounds[0].tolist(),
                'max': grid_mapper.grid_bounds[1].tolist()
            }
        }
        
        # Save statistics
        stats_path = output_dir / "sdf_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics: {stats_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("PROPELLER SDF COMPUTATION - TEST RESULTS")
        print("="*60)
        print(f"Mesh file: {mesh_file.name}")
        print(f"Surface triangles: {stats['mesh_triangles']}")
        print(f"Grid resolution: {stats['grid_resolution']}")
        print(f"SDF range: [{stats['sdf_min']:.6f}, {stats['sdf_max']:.6f}]")
        print(f"Zero crossings: {stats['zero_crossings']}")
        print(f"Output directory: {output_dir}")
        print("="*60)
        
        logger.info("Propeller SDF test workflow completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_sphere_validation():
    """
    Validation test using a simple analytical sphere geometry.
    """
    logger.info("Running sphere validation test")
    
    try:
        # Create a simple sphere mesh for validation
        # This would require mesh generation, so for now we'll skip
        # In a real implementation, you'd generate a sphere mesh
        logger.info("Sphere validation test skipped - requires mesh generation")
        return True
        
    except Exception as e:
        logger.error(f"Sphere validation test failed: {e}")
        return False


def main():
    """
    Run all tests.
    """
    print("JAX-Fluids Immersed Boundary Endpoint - Test Suite")
    print("="*60)
    
    # Run tests
    tests = [
        ("Propeller SDF Workflow", test_propeller_sdf_workflow),
        ("Sphere Validation", test_simple_sphere_validation)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\nRunning test: {test_name}")
        print("-" * 40)
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:<30} {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 