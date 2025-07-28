"""
Wind Tunnel Domain Generator

This module creates Cartesian wind tunnel domains around immersed objects
for proper levelset computation in JAX-Fluids simulations.
"""

import numpy as np
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class WindTunnelDomain:
    """
    Creates wind tunnel domains around immersed objects.
    
    Handles:
    - Automatic domain sizing based on object dimensions
    - Wind tunnel geometry considerations
    - Proper boundary layer spacing
    - Integration with JAX-Fluids domain setup
    """
    
    def __init__(self):
        """Initialize wind tunnel domain generator."""
        self.object_bounds = None
        self.domain_bounds = None
        self.tunnel_config = None
        
        logger.info("Initialized WindTunnelDomain")
    
    def create_wind_tunnel_around_object(self,
                                       object_bounds: Tuple[np.ndarray, np.ndarray],
                                       tunnel_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create a wind tunnel domain around an object.
        
        Args:
            object_bounds: Tuple of (min_coords, max_coords) of the object
            tunnel_config: Configuration for tunnel dimensions
            
        Returns:
            Dictionary containing domain information
        """
        min_coords, max_coords = object_bounds
        object_size = max_coords - min_coords
        object_center = (min_coords + max_coords) / 2
        
        # Default tunnel configuration - MUCH LARGER for proper SDF computation
        default_config = {
            'upstream_length_factor': 8.0,     # 8x object length upstream (increased)
            'downstream_length_factor': 15.0,  # 15x object length downstream (increased)
            'width_factor': 6.0,               # 6x object width on each side (increased)
            'height_factor': 6.0,              # 6x object height on each side (increased)
            'min_upstream_length': 50.0,       # Minimum upstream length (increased)
            'min_downstream_length': 100.0,    # Minimum downstream length (increased)
            'min_width': 50.0,                 # Minimum tunnel width (increased)
            'min_height': 50.0,                # Minimum tunnel height (increased)
            'flow_direction': 'x'              # Primary flow direction
        }
        
        if tunnel_config:
            default_config.update(tunnel_config)
        
        self.tunnel_config = default_config
        
        # Determine primary dimensions
        flow_dir = default_config['flow_direction']
        if flow_dir == 'x':
            length_dim = 0
            width_dim = 1
            height_dim = 2
        elif flow_dir == 'y':
            length_dim = 1
            width_dim = 0
            height_dim = 2
        else:  # 'z'
            length_dim = 2
            width_dim = 0
            height_dim = 1
        
        object_length = object_size[length_dim]
        object_width = object_size[width_dim]
        object_height = object_size[height_dim]
        
        # Calculate tunnel dimensions
        upstream_length = max(
            default_config['upstream_length_factor'] * object_length,
            default_config['min_upstream_length']
        )
        downstream_length = max(
            default_config['downstream_length_factor'] * object_length,
            default_config['min_downstream_length']
        )
        tunnel_width = max(
            default_config['width_factor'] * object_width,
            default_config['min_width']
        )
        tunnel_height = max(
            default_config['height_factor'] * object_height,
            default_config['min_height']
        )
        
        # Create domain bounds
        domain_min = np.copy(object_center)
        domain_max = np.copy(object_center)
        
        # Set dimensions based on flow direction
        domain_min[length_dim] = min_coords[length_dim] - upstream_length
        domain_max[length_dim] = max_coords[length_dim] + downstream_length
        
        domain_min[width_dim] = object_center[width_dim] - tunnel_width / 2
        domain_max[width_dim] = object_center[width_dim] + tunnel_width / 2
        
        domain_min[height_dim] = object_center[height_dim] - tunnel_height / 2
        domain_max[height_dim] = object_center[height_dim] + tunnel_height / 2
        
        self.object_bounds = object_bounds
        self.domain_bounds = (domain_min, domain_max)
        
        domain_info = {
            'object_bounds': object_bounds,
            'domain_bounds': self.domain_bounds,
            'object_size': object_size,
            'object_center': object_center,
            'tunnel_dimensions': {
                'upstream_length': upstream_length,
                'downstream_length': downstream_length,
                'tunnel_width': tunnel_width,
                'tunnel_height': tunnel_height,
                'total_length': upstream_length + downstream_length + object_length,
                'flow_direction': flow_dir
            },
            'domain_size': domain_max - domain_min,
            'config': default_config
        }
        
        logger.info(f"Created wind tunnel domain:")
        logger.info(f"  Object size: {object_size}")
        logger.info(f"  Domain size: {domain_max - domain_min}")
        logger.info(f"  Upstream length: {upstream_length:.3f}")
        logger.info(f"  Downstream length: {downstream_length:.3f}")
        
        return domain_info
    
    def suggest_grid_resolution(self,
                              domain_info: Dict[str, Any],
                              target_cells_per_diameter: int = 20,
                              max_cells_total: int = 2000000,
                              sdf_refinement_factor: float = 1.0) -> Tuple[int, int, int]:
        """
        Suggest appropriate grid resolution for the wind tunnel.
        
        Args:
            domain_info: Domain information from create_wind_tunnel_around_object
            target_cells_per_diameter: Target cells across object diameter
            max_cells_total: Maximum total cells
            sdf_refinement_factor: Additional refinement factor for SDF accuracy (1.0 = default, 2.0 = twice as fine)
            
        Returns:
            Suggested grid resolution (nx, ny, nz)
        """
        object_size = domain_info['object_size']
        domain_size = domain_info['domain_size']
        flow_dir = domain_info['tunnel_dimensions']['flow_direction']
        
        # Determine characteristic object diameter
        if flow_dir == 'x':
            char_diameter = max(object_size[1], object_size[2])  # Cross-flow diameter
            streamwise_dim = 0
        elif flow_dir == 'y':
            char_diameter = max(object_size[0], object_size[2])
            streamwise_dim = 1
        else:  # 'z'
            char_diameter = max(object_size[0], object_size[1])
            streamwise_dim = 2
        
        # Calculate base cell size from object diameter with refinement factor
        base_cell_size = char_diameter / (target_cells_per_diameter * sdf_refinement_factor)
        
        # Calculate grid resolution for each direction
        resolution = []
        for i in range(3):
            n_cells = int(np.ceil(domain_size[i] / base_cell_size))
            
            # Refine streamwise direction
            if i == streamwise_dim:
                n_cells = int(n_cells * 1.5)  # More cells in flow direction
            
            # Apply additional SDF refinement
            n_cells = int(n_cells * sdf_refinement_factor)
            
            # Ensure minimum resolution
            n_cells = max(n_cells, 10)
            resolution.append(n_cells)
        
        # Check total cell count and adjust if necessary
        total_cells = np.prod(resolution)
        if total_cells > max_cells_total:
            scale_factor = (max_cells_total / total_cells) ** (1/3)
            resolution = [int(n * scale_factor) for n in resolution]
            resolution = [max(n, 10) for n in resolution]  # Ensure minimum
        
        logger.info(f"Suggested grid resolution: {tuple(resolution)}")
        logger.info(f"Total cells: {np.prod(resolution):,}")
        logger.info(f"SDF refinement factor: {sdf_refinement_factor:.1f}")
        logger.info(f"Effective cells per diameter: {resolution[1 if flow_dir == 'x' else 0] * char_diameter / domain_size[1 if flow_dir == 'x' else 0]:.1f}")
        
        return tuple(resolution)
    
    def create_jax_fluids_config(self,
                               domain_info: Dict[str, Any],
                               resolution: Tuple[int, int, int],
                               case_name: str = "wind_tunnel_immersed_boundary") -> Dict[str, Any]:
        """
        Create JAX-Fluids configuration for wind tunnel simulation.
        
        Args:
            domain_info: Domain information
            resolution: Grid resolution
            case_name: Case name
            
        Returns:
            Complete JAX-Fluids case configuration
        """
        domain_min, domain_max = domain_info['domain_bounds']
        flow_dir = domain_info['tunnel_dimensions']['flow_direction']
        
        # Set up flow conditions based on flow direction
        if flow_dir == 'x':
            inlet_face = "west"
            outlet_face = "east"
            wall_faces = ["north", "south", "top", "bottom"]
            flow_velocity = {"u": 50.0, "v": 0.0, "w": 0.0}  # 50 m/s in x-direction
        elif flow_dir == 'y':
            inlet_face = "south"
            outlet_face = "north"
            wall_faces = ["east", "west", "top", "bottom"]
            flow_velocity = {"u": 0.0, "v": 50.0, "w": 0.0}  # 50 m/s in y-direction
        else:  # 'z'
            inlet_face = "bottom"
            outlet_face = "top"
            wall_faces = ["east", "west", "north", "south"]
            flow_velocity = {"u": 0.0, "v": 0.0, "w": 50.0}  # 50 m/s in z-direction
        
        # Create domain configuration
        domain_config = {
            "x": {
                "cells": int(resolution[0]),
                "range": [float(domain_min[0]), float(domain_max[0])]
            },
            "y": {
                "cells": int(resolution[1]),
                "range": [float(domain_min[1]), float(domain_max[1])]
            },
            "z": {
                "cells": int(resolution[2]),
                "range": [float(domain_min[2]), float(domain_max[2])]
            },
            "decomposition": {
                "split_x": 1,
                "split_y": 1,
                "split_z": 1
            }
        }
        
        # Create boundary conditions
        boundary_conditions = {
            "primitives": {},
            "levelset": {}
        }
        
        # Set inlet condition
        boundary_conditions["primitives"][inlet_face] = {
            "type": "DIRICHLET",
            "primitives_callable": {
                "rho": 1.225,  # Air density at sea level
                "u": flow_velocity["u"],
                "v": flow_velocity["v"], 
                "w": flow_velocity["w"],
                "p": 101325.0  # Atmospheric pressure
            }
        }
        boundary_conditions["levelset"][inlet_face] = {"type": "DIRICHLET", "levelset": 1.0}
        
        # Set outlet condition
        boundary_conditions["primitives"][outlet_face] = {"type": "ZEROGRADIENT"}
        boundary_conditions["levelset"][outlet_face] = {"type": "ZEROGRADIENT"}
        
        # Set wall conditions
        for wall_face in wall_faces:
            boundary_conditions["primitives"][wall_face] = {"type": "SYMMETRY"}
            boundary_conditions["levelset"][wall_face] = {"type": "SYMMETRY"}
        
        # Create case setup
        case_setup = {
            "general": {
                "case_name": case_name,
                "end_time": 0.01,  # 10ms simulation
                "save_path": "./results",
                "save_dt": 0.001   # Save every 1ms
            },
            "restart": {
                "flag": False,
                "file_path": ""
            },
            "domain": domain_config,
            "boundary_conditions": boundary_conditions,
            "initial_condition": {
                "primitives": {
                    "positive": {  # Fluid region (outside object)
                        "rho": 1.225,
                        "u": flow_velocity["u"],
                        "v": flow_velocity["v"],
                        "w": flow_velocity["w"],
                        "p": 101325.0
                    },
                    "negative": {  # Inside object (should not be used)
                        "rho": 1.225,
                        "u": 0.0,
                        "v": 0.0,
                        "w": 0.0,
                        "p": 101325.0
                    }
                },
                "levelset": "PLACEHOLDER_FOR_SDF_FUNCTION"
            }
        }
        
        logger.info(f"Created wind tunnel configuration for {case_name}")
        logger.info(f"  Flow direction: {flow_dir}")
        logger.info(f"  Inlet: {inlet_face}, Outlet: {outlet_face}")
        logger.info(f"  Flow velocity: {flow_velocity}")
        
        return case_setup 