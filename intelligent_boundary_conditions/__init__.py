"""
Intelligent Boundary Conditions Endpoint
=========================================

This endpoint provides intelligent tagging of inlet and outlet faces for rocket nozzles
and other complex 3D internal flow geometries in JAX-Fluids simulations.

Features:
- Automated geometry parsing (STL, MSH, CAD files)
- Intelligent face detection and tagging using heuristics
- Manual face selection interface with visualization
- JAX-Fluids compatible boundary condition generation
- Integration with SDF/level-set methods
"""

from .geometry_parser import GeometryParser
from .face_tagger import FaceTagger
from .boundary_condition_generator import BoundaryConditionGenerator
from .main_api import IntelligentBoundaryConditionsAPI

__version__ = "1.0.0"
__author__ = "JAX-Fluids Agentic System"

__all__ = [
    "GeometryParser",
    "FaceTagger", 
    "BoundaryConditionGenerator",
    "IntelligentBoundaryConditionsAPI"
] 