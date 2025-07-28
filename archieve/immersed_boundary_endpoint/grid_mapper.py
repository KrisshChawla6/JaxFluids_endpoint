"""
Cartesian Grid Mapper

This module provides functionality to map signed distance functions
onto JAX-Fluids compatible Cartesian grid structures.
"""

import numpy as np
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Optional
import logging
try:
    from .sdf_generator import SignedDistanceFunction
except ImportError:
    from sdf_generator import SignedDistanceFunction

logger = logging.getLogger(__name__)


class CartesianGridMapper:
    """
    Maps signed distance functions onto JAX-Fluids Cartesian grids.
    
    Handles:
    - Grid generation compatible with JAX-Fluids domain setup
    - SDF mapping with proper boundary conditions
    - Integration with JAX-Fluids levelset initialization
    """
    
    def __init__(self, sdf_generator: SignedDistanceFunction):
        """
        Initialize the grid mapper.
        
        Args:
            sdf_generator: Configured SDF generator
        """
        self.sdf_generator = sdf_generator
        self.grid_bounds = None
        self.grid_resolution = None
        self.cell_sizes = None
        self.levelset_function = None
        
        logger.info("Initialized CartesianGridMapper")
    
    def setup_domain(self,
                    domain_bounds: Tuple[np.ndarray, np.ndarray],
                    resolution: Tuple[int, int, int],
                    padding_factor: float = 1.2) -> Dict[str, Any]:
        """
        Setup computational domain for JAX-Fluids.
        
        Args:
            domain_bounds: Physical domain bounds (min_coords, max_coords)
            resolution: Grid resolution (nx, ny, nz)
            padding_factor: Factor to expand domain around geometry
            
        Returns:
            Dictionary containing domain setup information
        """
        min_coords, max_coords = domain_bounds
        
        # Calculate domain size with padding
        domain_size = max_coords - min_coords
        padded_size = domain_size * padding_factor
        center = (min_coords + max_coords) / 2
        
        # New bounds with padding
        self.grid_bounds = (
            center - padded_size / 2,
            center + padded_size / 2
        )
        
        self.grid_resolution = resolution
        
        # Calculate cell sizes
        self.cell_sizes = padded_size / np.array(resolution)
        
        domain_info = {
            'bounds': self.grid_bounds,
            'resolution': resolution,
            'cell_sizes': self.cell_sizes,
            'domain_size': padded_size,
            'center': center
        }
        
        logger.info(f"Domain setup: bounds={self.grid_bounds}, resolution={resolution}")
        return domain_info
    
    def create_jax_fluids_domain_config(self,
                                      case_name: str = "immersed_boundary",
                                      inactive_axes: Optional[Tuple[str]] = None) -> Dict[str, Any]:
        """
        Create JAX-Fluids compatible domain configuration.
        
        Args:
            case_name: Name for the case
            inactive_axes: Axes to make inactive (e.g., ('z',) for 2D)
            
        Returns:
            Domain configuration dictionary
        """
        if self.grid_bounds is None or self.grid_resolution is None:
            raise ValueError("Domain not setup. Call setup_domain() first.")
        
        min_coords, max_coords = self.grid_bounds
        nx, ny, nz = self.grid_resolution
        
        # Handle 2D cases
        if inactive_axes and 'z' in inactive_axes:
            nz = 1
            max_coords[2] = min_coords[2] + self.cell_sizes[2]
        
        domain_config = {
            "x": {
                "cells": int(nx),
                "range": [float(min_coords[0]), float(max_coords[0])]
            },
            "y": {
                "cells": int(ny),
                "range": [float(min_coords[1]), float(max_coords[1])]
            },
            "z": {
                "cells": int(nz),
                "range": [float(min_coords[2]), float(max_coords[2])]
            },
            "decomposition": {
                "split_x": 1,
                "split_y": 1,
                "split_z": 1
            }
        }
        
        return domain_config
    
    def create_levelset_initializer(self) -> str:
        """
        Create levelset initialization string for JAX-Fluids.
        
        Returns:
            Lambda function string for levelset initialization
        """
        if self.grid_bounds is None:
            raise ValueError("Domain not setup. Call setup_domain() first.")
        
        # Create the JAX levelset function
        self.levelset_function = self.sdf_generator.create_jax_levelset_function(
            self.grid_bounds, self.grid_resolution
        )
        
        # For JAX-Fluids, we need to return a lambda string that can be evaluated
        # Since we can't serialize the actual function, we'll need to handle this differently
        # For now, return instructions for manual setup
        
        levelset_info = {
            'bounds': self.grid_bounds,
            'resolution': self.grid_resolution,
            'instructions': """
            To use this levelset function in JAX-Fluids:
            1. Use the precomputed SDF grid from the mapper
            2. Create interpolation function in your case setup
            3. Apply as levelset initialization lambda
            """
        }
        
        return levelset_info
    
    def generate_case_setup_json(self,
                                case_name: str = "immersed_boundary",
                                end_time: float = 1e-3,
                                save_dt: float = 1e-4,
                                fluid_properties: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate complete JAX-Fluids case setup JSON.
        
        Args:
            case_name: Case name
            end_time: Simulation end time
            save_dt: Save interval
            fluid_properties: Fluid property overrides
            
        Returns:
            Complete case setup dictionary
        """
        if self.grid_bounds is None:
            raise ValueError("Domain not setup. Call setup_domain() first.")
        
        # Default fluid properties
        default_fluid = {
            "rho": 1.0,
            "u": 0.0,
            "v": 0.0,
            "w": 0.0,
            "p": 101325.0
        }
        
        if fluid_properties:
            default_fluid.update(fluid_properties)
        
        domain_config = self.create_jax_fluids_domain_config(case_name)
        
        case_setup = {
            "general": {
                "case_name": case_name,
                "end_time": end_time,
                "save_path": "./results",
                "save_dt": save_dt
            },
            "restart": {
                "flag": False,
                "file_path": ""
            },
            "domain": domain_config,
            "boundary_conditions": {
                "primitives": {
                    "east": {"type": "ZEROGRADIENT"},
                    "west": {"type": "ZEROGRADIENT"},
                    "north": {"type": "ZEROGRADIENT"},
                    "south": {"type": "ZEROGRADIENT"},
                    "top": {"type": "ZEROGRADIENT"},
                    "bottom": {"type": "ZEROGRADIENT"}
                },
                "levelset": {
                    "east": {"type": "ZEROGRADIENT"},
                    "west": {"type": "ZEROGRADIENT"},
                    "north": {"type": "ZEROGRADIENT"},
                    "south": {"type": "ZEROGRADIENT"},
                    "top": {"type": "ZEROGRADIENT"},
                    "bottom": {"type": "ZEROGRADIENT"}
                }
            },
            "initial_condition": {
                "primitives": {
                    "positive": default_fluid.copy(),
                    "negative": default_fluid.copy()
                },
                "levelset": "PLACEHOLDER_FOR_SDF_FUNCTION"
            }
        }
        
        logger.info(f"Generated case setup for {case_name}")
        return case_setup
    
    def generate_numerical_setup_json(self,
                                    levelset_model: str = "FLUID-SOLID") -> Dict[str, Any]:
        """
        Generate JAX-Fluids numerical setup JSON for immersed boundary.
        
        Args:
            levelset_model: Type of levelset model ("FLUID-SOLID" or "FLUID-FLUID")
            
        Returns:
            Numerical setup dictionary
        """
        numerical_setup = {
            "conservatives": {
                "halo_cells": 2,
                "time_integration": {
                    "time_integrator": "RK3",
                    "CFL": 0.8
                }
            },
            "levelset": {
                "model": levelset_model,
                "halo_cells": 2,
                "geometry": {
                    "curvature": {
                        "curvature_calculator": "CENTRAL_DIFFERENCES_2"
                    }
                },
                "reinitialization_runtime": {
                    "is_active": True,
                    "interval": 5,
                    "steps": 10
                },
                "narrowband": {
                    "computation_width": 6,
                    "interface_width": 3
                },
                "extension": {
                    "primitives": {
                        "iterative": {
                            "steps": 10,
                            "CFL": 0.5
                        }
                    }
                },
                "interface_flux": {
                    "method": "CELLCENTER",
                    "derivative_stencil": "CENTRAL_DIFFERENCES_2"
                }
            },
            "solvers": {
                "convective_fluxes": {
                    "convective_solver": "GODUNOV",
                    "riemann_solver": "HLL"
                },
                "space_solver": "FINITE_VOLUME",
                "reconstruction_stencil": "WENO5"
            },
            "precision": {
                "is_double_precision": False
            }
        }
        
        return numerical_setup
    
    def export_sdf_data(self, filename: str) -> None:
        """
        Export precomputed SDF data for external use.
        
        Args:
            filename: Output filename (without extension)
        """
        if self.grid_bounds is None:
            raise ValueError("Domain not setup. Call setup_domain() first.")
        
        # Compute SDF on grid
        grid_coords, sdf_values = self.sdf_generator.compute_sdf_cartesian_grid(
            self.grid_bounds, self.grid_resolution
        )
        
        # Save as numpy arrays
        np.savez(
            f"{filename}.npz",
            sdf_values=sdf_values,
            grid_x=grid_coords[0],
            grid_y=grid_coords[1], 
            grid_z=grid_coords[2],
            bounds_min=self.grid_bounds[0],
            bounds_max=self.grid_bounds[1],
            resolution=self.grid_resolution,
            cell_sizes=self.cell_sizes
        )
        
        logger.info(f"Exported SDF data to {filename}.npz")
    
    def get_jax_fluids_compatible_function(self) -> str:
        """
        Generate JAX-Fluids compatible levelset function string.
        
        Returns:
            String representation of levelset function
        """
        if self.grid_bounds is None:
            raise ValueError("Domain not setup. Call setup_domain() first.")
        
        # This would need to be customized based on specific geometry
        # For now, return a template that can be modified
        
        function_template = f"""
# JAX-Fluids Levelset Function for Immersed Boundary
# Generated from mesh: {self.sdf_generator.mesh_processor.mesh_file}
# Domain bounds: {self.grid_bounds}
# Resolution: {self.grid_resolution}

# You will need to:
# 1. Load the precomputed SDF data
# 2. Create interpolation function
# 3. Use in case setup as:
# "levelset": "lambda x,y,z: your_sdf_interpolation_function(x,y,z)"

def create_levelset_function():
    # Load precomputed SDF data
    import numpy as np
    data = np.load('sdf_data.npz')
    sdf_values = data['sdf_values']
    bounds_min = data['bounds_min']
    bounds_max = data['bounds_max']
    
    # Create interpolation function here
    # Return lambda function compatible with JAX-Fluids
    pass
"""
        
        return function_template 