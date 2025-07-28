"""
JAX-Fluids Immersed Boundary Endpoint

This package provides tools for handling immersed boundary conditions in JAX-Fluids
using signed distance functions computed from mesh geometries.

Main modules:
- mesh_processor: Handles gmsh file reading and processing
- sdf_generator: Computes signed distance functions from mesh geometry
- grid_mapper: Maps SDFs onto JAX-Fluids Cartesian grids
- visualization: Plotting and visualization utilities
"""

from .mesh_processor import GmshProcessor
from .sdf_generator import SignedDistanceFunction
from .grid_mapper import CartesianGridMapper
from .visualization import SDFVisualizer
from .wind_tunnel_domain import WindTunnelDomain

__version__ = "0.1.0"
__author__ = "JAX-Fluids Immersed Boundary Team"

__all__ = [
    "GmshProcessor",
    "SignedDistanceFunction", 
    "CartesianGridMapper",
    "SDFVisualizer",
    "WindTunnelDomain"
] 