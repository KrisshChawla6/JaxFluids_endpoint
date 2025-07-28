#!/usr/bin/env python3
"""
Main Runner for Immersed Boundary SDF Generation

This script provides a clean, command-line interface to generate
SDF data for JAX-Fluids simulations.

Usage:
  - For a quick, low-quality preview:
    python run.py --quality fast

  - For a good-quality result:
    python run.py --quality standard --refinement 2.0

  - For a production-quality result (slow):
    python run.py --quality production --refinement 3.0
"""

import argparse
import logging
from pathlib import Path
import sys

# Add package directory to path to ensure imports work
sys.path.append(str(Path(__file__).parent.resolve()))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to run the SDF generation workflow."""
    
    parser = argparse.ArgumentParser(description="JAX-Fluids Immersed Boundary SDF Generator")
    
    parser.add_argument(
        '--quality',
        type=str,
        default='standard',
        choices=['fast', 'standard', 'production'],
        help="Quality of the SDF generation. Affects resolution and speed."
    )
    
    parser.add_argument(
        '--refinement',
        type=float,
        default=1.5,
        help="SDF grid refinement factor. Higher values mean finer grids."
    )
    
    parser.add_argument(
        '--mesh-file',
        type=str,
        default='../mesh/5_bladed_Propeller.STEP_medium_tetrahedral.msh',
        help="Path to the input Gmsh .msh file."
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help="Directory to save the results."
    )
    
    args = parser.parse_args()
    
    # Set parameters based on quality setting
    if args.quality == 'fast':
        logger.info("Setting configuration for FAST quality")
        cells_per_diameter = 10
        max_cells = 50000
        refinement = 1.0
    elif args.quality == 'standard':
        logger.info("Setting configuration for STANDARD quality")
        cells_per_diameter = 20
        max_cells = 150000
        refinement = args.refinement
    else:  # production
        logger.info("Setting configuration for PRODUCTION quality")
        logger.warning("Production quality may take a significant amount of time and memory!")
        cells_per_diameter = 40  # High base resolution
        max_cells = 1000000      # High cell limit
        refinement = args.refinement
        
    try:
        from tests.test_workflow import run_sdf_workflow
    except ImportError:
        logger.error("Could not import the SDF workflow. Make sure you are running from the `immersed_boundary_endpoint` directory.")
        sys.exit(1)
        
    # Run the main workflow
    run_sdf_workflow(
        mesh_file=args.mesh_file,
        output_dir_base=args.output_dir,
        run_name=f"{args.quality}_q{refinement:.1f}x",
        target_cells_per_diameter=cells_per_diameter,
        max_cells_total=max_cells,
        sdf_refinement_factor=refinement
    )

if __name__ == "__main__":
    main() 