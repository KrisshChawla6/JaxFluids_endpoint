"""
Public API for JAX-Fluids Post-Processing
=========================================

High-level functions for common post-processing tasks.
These provide a clean, functional interface to the package capabilities.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np

from .core.processor import FluidProcessor
from .visualization.interactive import InteractiveVisualizer
from .visualization.animation import AnimationCreator
from .io.vtk_exporter import VTKExporter


def process_simulation(
    results_path: Union[str, Path],
    output_path: Union[str, Path],
    variables: Optional[List[str]] = None,
    time_index: int = -1
) -> Dict[str, Any]:
    """
    Process JAX-Fluids simulation results and extract flow variables.
    
    Args:
        results_path: Path to simulation results directory containing .h5 files
        output_path: Path where outputs will be saved
        variables: List of variables to extract (default: all available)
        time_index: Time step to process (-1 for last step)
        
    Returns:
        Dictionary containing:
        - 'flow_data': Extracted flow variables
        - 'metadata': Simulation metadata
        - 'grid_info': Grid dimensions and spacing
        - 'time_info': Time step information
        
    Example:
        >>> results = process_simulation(
        ...     "simulation/domain", 
        ...     "output",
        ...     variables=["velocity", "pressure"]
        ... )
        >>> print(f"Grid shape: {results['grid_info']['shape']}")
    """
    processor = FluidProcessor(
        results_path=results_path,
        output_path=output_path
    )
    
    # Extract flow data
    flow_data = processor.extract_flow_variables(time_index=time_index)
    
    # Get metadata
    metadata = processor.get_simulation_metadata()
    grid_info = processor.get_grid_info()
    time_info = processor.get_time_info()
    
    # Save summary
    processor.save_summary(flow_data)
    
    return {
        'flow_data': flow_data,
        'metadata': metadata,
        'grid_info': grid_info,
        'time_info': time_info,
        'processor': processor  # For advanced usage
    }


def create_visualization(
    flow_data: Dict[str, np.ndarray],
    variable: str = "velocity_magnitude",
    mesh_path: Optional[Union[str, Path]] = None,
    interactive: bool = True,
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Create 3D visualization of flow data.
    
    Args:
        flow_data: Dictionary of flow variables from process_simulation()
        variable: Variable to visualize
        mesh_path: Optional path to mesh file (.msh format)
        interactive: Whether to show interactive viewer
        save_path: Optional path to save screenshot
        
    Example:
        >>> results = process_simulation("simulation/domain", "output")
        >>> create_visualization(
        ...     results['flow_data'],
        ...     variable="pressure",
        ...     mesh_path="mesh/propeller.msh"
        ... )
    """
    visualizer = InteractiveVisualizer()
    
    if interactive:
        visualizer.show_interactive(
            flow_data=flow_data,
            variable=variable,
            mesh_path=mesh_path
        )
    
    if save_path:
        visualizer.save_screenshot(
            flow_data=flow_data,
            variable=variable,
            mesh_path=mesh_path,
            output_path=save_path
        )


def export_vtk(
    flow_data: Dict[str, np.ndarray],
    output_path: Union[str, Path],
    mesh_path: Optional[Union[str, Path]] = None
) -> Dict[str, Path]:
    """
    Export flow data and mesh to VTK formats.
    
    Args:
        flow_data: Dictionary of flow variables
        output_path: Directory to save VTK files
        mesh_path: Optional path to mesh file
        
    Returns:
        Dictionary with paths to created VTK files:
        - 'flow_grid': Path to flow data VTK file (.vts)
        - 'mesh': Path to mesh VTK file (.vtp) if mesh provided
        
    Example:
        >>> results = process_simulation("simulation/domain", "output") 
        >>> vtk_files = export_vtk(
        ...     results['flow_data'],
        ...     "vtk_output",
        ...     mesh_path="mesh/propeller.msh"
        ... )
        >>> print(f"Flow data: {vtk_files['flow_grid']}")
    """
    exporter = VTKExporter()
    
    return exporter.export_all(
        flow_data=flow_data,
        output_path=output_path,
        mesh_path=mesh_path
    )


def create_animation(
    results_path: Union[str, Path],
    output_path: Union[str, Path],
    variable: str = "velocity_magnitude",
    plane: str = "xy",
    plane_value: float = 0.5,
    time_range: Optional[Tuple[int, int]] = None,
    fps: int = 10,
    format: str = "gif"
) -> Path:
    """
    Create 2D animation of flow variable over time.
    
    Args:
        results_path: Path to simulation results directory
        output_path: Directory to save animation
        variable: Variable to animate
        plane: Plane orientation ("xy", "xz", "yz")
        plane_value: Position along plane normal (0.0-1.0)
        time_range: Optional tuple (start_idx, end_idx) for time range
        fps: Frames per second
        format: Output format ("gif" or "mp4")
        
    Returns:
        Path to created animation file
        
    Example:
        >>> animation_path = create_animation(
        ...     "simulation/domain",
        ...     "animations",
        ...     variable="pressure",
        ...     plane="xy",
        ...     plane_value=0.3
        ... )
        >>> print(f"Animation saved: {animation_path}")
    """
    # Create processor to access time series data
    processor = FluidProcessor(
        results_path=results_path,
        output_path=output_path
    )
    
    # Create animator
    animator = AnimationCreator()
    
    # Generate animation
    animation_path = animator.create_time_series_animation(
        processor=processor,
        variable=variable,
        plane=plane,
        plane_value=plane_value,
        time_range=time_range,
        fps=fps,
        format=format,
        output_path=output_path
    )
    
    return animation_path


# Convenience functions for common workflows
def quick_visualization(
    results_path: Union[str, Path],
    variable: str = "velocity_magnitude",
    mesh_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Quick visualization workflow: process + visualize in one call.
    
    Args:
        results_path: Path to simulation results directory
        variable: Variable to visualize  
        mesh_path: Optional path to mesh file
        
    Example:
        >>> quick_visualization("simulation/domain", "pressure", "mesh.msh")
    """
    # Process simulation
    results = process_simulation(results_path, "temp_output")
    
    # Create visualization
    create_visualization(
        results['flow_data'],
        variable=variable,
        mesh_path=mesh_path
    )


def batch_export(
    results_path: Union[str, Path],
    output_path: Union[str, Path],
    variables: List[str],
    mesh_path: Optional[Union[str, Path]] = None,
    time_indices: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Batch process multiple variables and time steps.
    
    Args:
        results_path: Path to simulation results directory
        output_path: Directory for outputs
        variables: List of variables to process
        mesh_path: Optional path to mesh file
        time_indices: List of time indices (default: all)
        
    Returns:
        Dictionary with processing results for each time step
        
    Example:
        >>> results = batch_export(
        ...     "simulation/domain",
        ...     "batch_output", 
        ...     ["velocity_magnitude", "pressure", "density"],
        ...     time_indices=[0, 5, 10]
        ... )
    """
    processor = FluidProcessor(
        results_path=results_path,
        output_path=output_path,
        mesh_path=mesh_path
    )
    
    if time_indices is None:
        time_indices = list(range(processor.reader.get_num_time_steps()))
    
    results = {}
    
    for time_idx in time_indices:
        print(f"Processing time step {time_idx}...")
        
        # Extract flow data for this time step
        flow_data = processor.extract_flow_variables(time_index=time_idx)
        
        # Export VTK files
        vtk_files = export_vtk(
            flow_data,
            output_path / f"time_{time_idx:04d}",
            mesh_path=mesh_path
        )
        
        results[time_idx] = {
            'flow_data': flow_data,
            'vtk_files': vtk_files
        }
    
    return results