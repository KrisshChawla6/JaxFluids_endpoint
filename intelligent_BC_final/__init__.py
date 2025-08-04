"""
Intelligent Boundary Condition Final Endpoint
============================================

A simplified endpoint for generating intelligent boundary conditions
for internal flow CFD simulations using JAX-Fluids.

This endpoint automatically:
1. Detects circular inlet/outlet openings in mesh files
2. Generates JAX-Fluids compatible masks for boundary conditions
3. Outputs all necessary files for simulation setup

Usage:
    from intelligent_BC_final import IntelligentBoundaryEndpoint
    
    endpoint = IntelligentBoundaryEndpoint()
    result = endpoint.process_mesh("path/to/mesh.msh", "output/directory")

Author: AI Assistant
Version: 1.0.0
"""

from .intelligent_boundary_endpoint import IntelligentBoundaryEndpoint

__version__ = "1.0.0"
__author__ = "AI Assistant"

__all__ = ["IntelligentBoundaryEndpoint"] 