"""
JAX-Fluids Post-Processing Package
=================================

A professional toolkit for post-processing JAX-Fluids simulation results with
interactive 3D visualization, mesh embedding, and animation capabilities.

Main Components:
- FluidProcessor: Core processing engine
- InteractiveVisualizer: 3D visualization with PyVista
- AnimationCreator: 2D animations over time
- DataReader: HDF5 file reading utilities
- VTKExporter: Export to VTK formats

Example Usage:
    >>> from jax_fluids_postprocess import FluidProcessor
    >>> processor = FluidProcessor("path/to/results")
    >>> flow_data = processor.extract_flow_data()
    >>> processor.visualize_interactive(flow_data, mesh_path="mesh.msh")
"""

__version__ = "1.0.0"
__author__ = "JAX-Fluids Post-Processing Team"
__email__ = "contact@jaxfluids.com"

# Core functionality
from .core.processor import FluidProcessor
from .core.data_reader import DataReader

# Visualization
from .visualization.interactive import InteractiveVisualizer
from .visualization.animation import AnimationCreator

# I/O utilities
from .io.h5_reader import JAXFluidsReader as H5Reader
from .io.vtk_exporter import VTKExporter

# Main API functions
from .api import (
    process_simulation,
    create_visualization,
    export_vtk,
    create_animation,
    quick_visualization
)

__all__ = [
    # Core classes
    "FluidProcessor",
    "DataReader", 
    "InteractiveVisualizer",
    "AnimationCreator",
    "H5Reader",
    "VTKExporter",
    
    # API functions
    "process_simulation",
    "create_visualization", 
    "export_vtk",
    "create_animation",
    "quick_visualization",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]