"""
Command-line interface for JAX-Fluids post-processing.
Provides a clean, user-friendly CLI for common operations.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import (
    process_simulation,
    create_visualization,
    export_vtk,
    create_animation,
    quick_visualization
)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        prog='jax-fluids-postprocess',
        description='Professional post-processing toolkit for JAX-Fluids simulation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick visualization
  jax-fluids-postprocess quick-viz simulation/domain --variable pressure

  # Full processing workflow  
  jax-fluids-postprocess process simulation/domain output --plot --export-vtk

  # Create animation
  jax-fluids-postprocess animate simulation/domain output --variable velocity_magnitude

  # Batch export multiple variables
  jax-fluids-postprocess export simulation/domain output --variables velocity_magnitude pressure density
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Quick visualization command
    quick_parser = subparsers.add_parser(
        'quick-viz',
        help='Quick visualization (process + visualize)'
    )
    quick_parser.add_argument(
        'results_path',
        type=str,
        help='Path to simulation results directory'
    )
    quick_parser.add_argument(
        '--variable', '-v',
        type=str,
        default='velocity_magnitude',
        help='Variable to visualize (default: velocity_magnitude)'
    )
    quick_parser.add_argument(
        '--mesh-path', '-m',
        type=str,
        help='Path to mesh file (.msh format)'
    )
    
    # Full process command
    process_parser = subparsers.add_parser(
        'process',
        help='Full processing workflow'
    )
    process_parser.add_argument(
        'results_path',
        type=str,
        help='Path to simulation results directory'
    )
    process_parser.add_argument(
        'output_path',
        type=str,
        help='Path for output files'
    )
    process_parser.add_argument(
        '--variable', '-v',
        type=str,
        default='velocity_magnitude',
        help='Variable to process (default: velocity_magnitude)'
    )
    process_parser.add_argument(
        '--time-index', '-t',
        type=int,
        default=-1,
        help='Time step index (-1 for last step)'
    )
    process_parser.add_argument(
        '--mesh-path', '-m',
        type=str,
        help='Path to mesh file (.msh format)'
    )
    process_parser.add_argument(
        '--plot', '-p',
        action='store_true',
        help='Show interactive visualization'
    )
    process_parser.add_argument(
        '--export-vtk',
        action='store_true',
        help='Export to VTK format'
    )
    process_parser.add_argument(
        '--save-screenshot',
        type=str,
        help='Save screenshot to specified path'
    )
    
    # Animation command
    anim_parser = subparsers.add_parser(
        'animate',
        help='Create 2D animation'
    )
    anim_parser.add_argument(
        'results_path',
        type=str,
        help='Path to simulation results directory'
    )
    anim_parser.add_argument(
        'output_path', 
        type=str,
        help='Path for output files'
    )
    anim_parser.add_argument(
        '--variable', '-v',
        type=str,
        default='velocity_magnitude',
        help='Variable to animate (default: velocity_magnitude)'
    )
    anim_parser.add_argument(
        '--plane',
        type=str,
        choices=['xy', 'xz', 'yz'],
        default='xy',
        help='Plane orientation (default: xy)'
    )
    anim_parser.add_argument(
        '--plane-value',
        type=float,
        default=0.5,
        help='Plane position 0.0-1.0 (default: 0.5)'
    )
    anim_parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Frames per second (default: 10)'
    )
    anim_parser.add_argument(
        '--format',
        type=str,
        choices=['gif', 'mp4'],
        default='gif',
        help='Output format (default: gif)'
    )
    
    # Export command
    export_parser = subparsers.add_parser(
        'export',
        help='Export to VTK format'
    )
    export_parser.add_argument(
        'results_path',
        type=str,
        help='Path to simulation results directory'
    )
    export_parser.add_argument(
        'output_path',
        type=str,
        help='Path for output files'
    )
    export_parser.add_argument(
        '--variables',
        type=str,
        nargs='+',
        help='Variables to export (space-separated list)'
    )
    export_parser.add_argument(
        '--time-indices',
        type=int,
        nargs='+',
        help='Time indices to export (space-separated list)'
    )
    export_parser.add_argument(
        '--mesh-path', '-m',
        type=str,
        help='Path to mesh file (.msh format)'
    )
    
    return parser


def cmd_quick_viz(args) -> int:
    """Handle quick visualization command."""
    try:
        quick_visualization(
            results_path=args.results_path,
            variable=args.variable,
            mesh_path=args.mesh_path
        )
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_process(args) -> int:
    """Handle full process command."""
    try:
        # Process simulation
        print("🔄 Processing simulation data...")
        results = process_simulation(
            results_path=args.results_path,
            output_path=args.output_path,
            time_index=args.time_index
        )
        
        # Export VTK if requested
        if args.export_vtk:
            print("📤 Exporting to VTK format...")
            export_vtk(
                flow_data=results['flow_data'],
                output_path=args.output_path,
                mesh_path=args.mesh_path
            )
        
        # Create visualization if requested
        if args.plot or args.save_screenshot:
            print("🎨 Creating visualization...")
            create_visualization(
                flow_data=results['flow_data'],
                variable=args.variable,
                mesh_path=args.mesh_path,
                interactive=args.plot,
                save_path=args.save_screenshot
            )
        
        print("✅ Processing complete!")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_animate(args) -> int:
    """Handle animation command."""
    try:
        animation_path = create_animation(
            results_path=args.results_path,
            output_path=args.output_path,
            variable=args.variable,
            plane=args.plane,
            plane_value=args.plane_value,
            fps=args.fps,
            format=args.format
        )
        
        print(f"✅ Animation created: {animation_path}")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_export(args) -> int:
    """Handle export command."""
    try:
        from .api import batch_export
        
        results = batch_export(
            results_path=args.results_path,
            output_path=args.output_path,
            variables=args.variables or ["velocity_magnitude", "pressure"],
            mesh_path=args.mesh_path,
            time_indices=args.time_indices
        )
        
        print(f"✅ Exported {len(results)} time steps")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    args = parser.parse_args()
    
    if args.command == 'quick-viz':
        return cmd_quick_viz(args)
    elif args.command == 'process':
        return cmd_process(args)
    elif args.command == 'animate':
        return cmd_animate(args)
    elif args.command == 'export':
        return cmd_export(args)
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())