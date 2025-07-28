#!/usr/bin/env python3
"""
Boundary Condition Generator Module
===================================

This module generates JAX-Fluids compatible boundary condition configurations
and masks from tagged geometry faces for rocket nozzles and internal flows.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

# JAX imports (optional)
try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    from .geometry_parser import GeometryParser, GeometryFace
    from .face_tagger import FaceTagger
except ImportError:
    from geometry_parser import GeometryParser, GeometryFace
    from face_tagger import FaceTagger

logger = logging.getLogger(__name__)

class BoundaryConditionType:
    """Standard JAX-Fluids boundary condition types"""
    
    # Inlet boundary conditions
    DIRICHLET = "DIRICHLET"
    SIMPLE_INFLOW = "SIMPLE_INFLOW"
    CHARACTERISTIC_INFLOW = "CHARACTERISTIC_INFLOW"
    
    # Outlet boundary conditions
    ZEROGRADIENT = "ZEROGRADIENT"
    SIMPLE_OUTFLOW = "SIMPLE_OUTFLOW"
    CHARACTERISTIC_OUTFLOW = "CHARACTERISTIC_OUTFLOW"
    
    # Wall boundary conditions
    NOSLIP = "NOSLIP"
    SLIP = "SLIP"
    SYMMETRY = "SYMMETRY"
    
    # Inactive boundaries
    INACTIVE = "INACTIVE"

class RocketEngineConditions:
    """Standard rocket engine operating conditions"""
    
    @staticmethod
    def get_rocket_conditions(fuel_type: str = "hydrogen", 
                             chamber_pressure: float = 6.9e6,  # Pa
                             chamber_temperature: float = 3580,  # K
                             ambient_pressure: float = 101325,  # Pa
                             gamma: float = 1.3) -> Dict[str, float]:
        """
        Get standard rocket engine conditions
        
        Args:
            fuel_type: Type of fuel ("hydrogen", "kerosene", "methane")
            chamber_pressure: Chamber pressure in Pa
            chamber_temperature: Chamber temperature in K
            ambient_pressure: Ambient pressure in Pa
            gamma: Specific heat ratio
            
        Returns:
            Dictionary of flow conditions
        """
        
        # Gas constants for different fuels
        gas_constants = {
            "hydrogen": 4124.0,    # J/(kg·K) for H2/O2 combustion products
            "kerosene": 287.0,     # J/(kg·K) for RP-1/O2 combustion products
            "methane": 518.0       # J/(kg·K) for CH4/O2 combustion products
        }
        
        R = gas_constants.get(fuel_type, 287.0)
        
        # Calculate derived quantities
        chamber_density = chamber_pressure / (R * chamber_temperature)
        sound_speed = np.sqrt(gamma * R * chamber_temperature)
        
        return {
            'chamber_pressure': chamber_pressure,
            'chamber_temperature': chamber_temperature,
            'chamber_density': chamber_density,
            'ambient_pressure': ambient_pressure,
            'gas_constant': R,
            'gamma': gamma,
            'sound_speed': sound_speed,
            'fuel_type': fuel_type
        }

class BoundaryConditionGenerator:
    """
    Generates JAX-Fluids compatible boundary condition configurations
    """
    
    def __init__(self, geometry_parser: GeometryParser, face_tagger: FaceTagger):
        """
        Initialize boundary condition generator
        
        Args:
            geometry_parser: Parsed geometry object
            face_tagger: Face tagger with tagged faces
        """
        self.geometry_parser = geometry_parser
        self.face_tagger = face_tagger
        self.bounds = geometry_parser.geometry_bounds
        
        # Validate that faces are tagged
        summary = face_tagger.get_tagging_summary()
        if not summary['has_inlet'] or not summary['has_outlet']:
            logger.warning("Faces are not properly tagged. Some boundary conditions may be missing.")
    
    def generate_jaxfluids_config(self, 
                                 flow_conditions: Dict[str, float],
                                 domain_config: Dict[str, Any],
                                 physics_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate complete JAX-Fluids configuration with intelligent boundary conditions
        
        Args:
            flow_conditions: Flow conditions (pressure, temperature, etc.)
            domain_config: Domain discretization configuration
            physics_config: Physics and numerical configuration
            
        Returns:
            Complete JAX-Fluids configuration dictionary
        """
        
        logger.info("Generating JAX-Fluids configuration with intelligent boundary conditions")
        
        # Generate boundary conditions
        boundary_conditions = self.generate_boundary_conditions(flow_conditions)
        
        # Create base configuration
        config = {
            "general": {
                "case_name": "intelligent_boundary_rocket_nozzle",
                "end_time": 0.01,
                "save_path": "./results_intelligent_bc",
                "save_dt": 0.001
            },
            "domain": domain_config,
            "boundary_conditions": boundary_conditions,
            "initial_condition": self._generate_initial_conditions(flow_conditions),
            "material_properties": self._generate_material_properties(flow_conditions)
        }
        
        # Add physics configuration if provided
        if physics_config:
            config.update(physics_config)
        
        return config
    
    def generate_boundary_conditions(self, 
                                   flow_conditions: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate boundary conditions based on tagged faces
        
        Args:
            flow_conditions: Flow conditions dictionary
            
        Returns:
            JAX-Fluids boundary conditions configuration
        """
        
        # Get tagged face summary
        summary = self.face_tagger.get_tagging_summary()
        
        # Generate inlet conditions
        inlet_bc = self._generate_inlet_boundary_condition(flow_conditions)
        
        # Generate outlet conditions
        outlet_bc = self._generate_outlet_boundary_condition(flow_conditions)
        
        # Generate wall conditions
        wall_bc = self._generate_wall_boundary_condition()
        
        # Map to JAX-Fluids directional boundaries
        boundary_conditions = self._map_to_jaxfluids_boundaries(
            inlet_bc, outlet_bc, wall_bc
        )
        
        logger.info(f"Generated boundary conditions: "
                   f"inlet faces={summary['tagged_counts']['inlet']}, "
                   f"outlet faces={summary['tagged_counts']['outlet']}, "
                   f"wall faces={summary['tagged_counts']['wall']}")
        
        return boundary_conditions
    
    def _generate_inlet_boundary_condition(self, 
                                         flow_conditions: Dict[str, float]) -> Dict[str, Any]:
        """Generate inlet boundary condition"""
        
        # Extract conditions
        chamber_pressure = flow_conditions['chamber_pressure']
        chamber_temperature = flow_conditions['chamber_temperature']
        chamber_density = flow_conditions['chamber_density']
        gamma = flow_conditions['gamma']
        R = flow_conditions['gas_constant']
        
        # Calculate inlet velocity (assuming choked flow at throat)
        sound_speed = flow_conditions['sound_speed']
        inlet_velocity = sound_speed  # Mach 1 at throat
        
        return {
            "type": BoundaryConditionType.SIMPLE_INFLOW,
            "primitives_callable": {
                "rho": chamber_density,
                "u": inlet_velocity,
                "v": 0.0,
                "w": 0.0,
                "p": chamber_pressure
            },
            "description": "Rocket chamber conditions at inlet"
        }
    
    def _generate_outlet_boundary_condition(self, 
                                          flow_conditions: Dict[str, float]) -> Dict[str, Any]:
        """Generate outlet boundary condition"""
        
        ambient_pressure = flow_conditions['ambient_pressure']
        
        return {
            "type": BoundaryConditionType.SIMPLE_OUTFLOW,
            "primitives_callable": {
                "p": ambient_pressure
            },
            "description": "Ambient pressure at nozzle exit"
        }
    
    def _generate_wall_boundary_condition(self) -> Dict[str, Any]:
        """Generate wall boundary condition"""
        
        return {
            "type": BoundaryConditionType.NOSLIP,
            "description": "No-slip wall condition for nozzle surfaces"
        }
    
    def _map_to_jaxfluids_boundaries(self, 
                                   inlet_bc: Dict[str, Any],
                                   outlet_bc: Dict[str, Any], 
                                   wall_bc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map tagged faces to JAX-Fluids directional boundaries
        """
        
        # Determine primary flow direction
        flow_axis = self.face_tagger.primary_flow_axis.lower()
        
        # Get domain bounds
        bounds = self.bounds
        
        # Default boundary assignment
        boundary_config = {
            "west": {"type": BoundaryConditionType.SYMMETRY},
            "east": {"type": BoundaryConditionType.SYMMETRY},
            "north": {"type": BoundaryConditionType.SYMMETRY},
            "south": {"type": BoundaryConditionType.SYMMETRY},
            "top": {"type": BoundaryConditionType.INACTIVE},
            "bottom": {"type": BoundaryConditionType.INACTIVE}
        }
        
        # Assign inlet and outlet based on flow direction
        if flow_axis == 'x':
            boundary_config["west"] = inlet_bc
            boundary_config["east"] = outlet_bc
            boundary_config["north"] = wall_bc
            boundary_config["south"] = wall_bc
        elif flow_axis == 'y':
            boundary_config["south"] = inlet_bc  
            boundary_config["north"] = outlet_bc
            boundary_config["west"] = wall_bc
            boundary_config["east"] = wall_bc
        elif flow_axis == 'z':
            boundary_config["bottom"] = inlet_bc
            boundary_config["top"] = outlet_bc
            boundary_config["west"] = wall_bc
            boundary_config["east"] = wall_bc
            boundary_config["north"] = wall_bc
            boundary_config["south"] = wall_bc
        
        return boundary_config
    
    def _generate_initial_conditions(self, 
                                   flow_conditions: Dict[str, float]) -> Dict[str, Any]:
        """Generate initial conditions"""
        
        # Use chamber conditions scaled down for initialization
        chamber_density = flow_conditions['chamber_density']
        chamber_pressure = flow_conditions['chamber_pressure']
        sound_speed = flow_conditions['sound_speed']
        
        return {
            "rho": chamber_density * 0.5,
            "u": sound_speed * 0.1,
            "v": 0.0,
            "w": 0.0,
            "p": chamber_pressure * 0.3
        }
    
    def _generate_material_properties(self, 
                                    flow_conditions: Dict[str, float]) -> Dict[str, Any]:
        """Generate material properties"""
        
        gamma = flow_conditions['gamma']
        R = flow_conditions['gas_constant']
        
        return {
            "equation_of_state": {
                "model": "IdealGas",
                "specific_heat_ratio": gamma,
                "specific_gas_constant": R
            },
            "transport": {
                "dynamic_viscosity": {
                    "model": "CUSTOM",
                    "value": 5e-5
                },
                "bulk_viscosity": 0.0,
                "thermal_conductivity": {
                    "model": "CUSTOM",
                    "value": 0.1
                }
            }
        }
    
    def generate_boundary_masks(self, 
                              domain_resolution: Tuple[int, int, int],
                              output_format: str = "numpy") -> Dict[str, Any]:
        """
        Generate 3D boundary condition masks for direct application in JAX-Fluids
        
        Args:
            domain_resolution: Grid resolution (nx, ny, nz)
            output_format: Output format ("numpy", "jax", "dict")
            
        Returns:
            Dictionary containing boundary masks
        """
        
        logger.info(f"Generating boundary masks for resolution {domain_resolution}")
        
        nx, ny, nz = domain_resolution
        
        # Initialize masks
        if output_format == "jax" and JAX_AVAILABLE:
            inlet_mask = jnp.zeros((nx, ny, nz), dtype=bool)
            outlet_mask = jnp.zeros((nx, ny, nz), dtype=bool)
            wall_mask = jnp.zeros((nx, ny, nz), dtype=bool)
        else:
            inlet_mask = np.zeros((nx, ny, nz), dtype=bool)
            outlet_mask = np.zeros((nx, ny, nz), dtype=bool)
            wall_mask = np.zeros((nx, ny, nz), dtype=bool)
        
        # Get domain bounds
        x_min, y_min, z_min = self.bounds['min']
        x_max, y_max, z_max = self.bounds['max']
        
        # Create coordinate arrays
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        z = np.linspace(z_min, z_max, nz)
        
        # Generate masks based on tagged faces
        for face in self.geometry_parser.faces:
            if face.tag is None:
                continue
            
            # Find grid cells that intersect with this face
            mask_indices = self._face_to_grid_mask(face, x, y, z)
            
            if face.tag == 'inlet':
                inlet_mask[mask_indices] = True
            elif face.tag == 'outlet':
                outlet_mask[mask_indices] = True
            elif face.tag == 'wall':
                wall_mask[mask_indices] = True
        
        masks = {
            'inlet_mask': inlet_mask,
            'outlet_mask': outlet_mask,
            'wall_mask': wall_mask,
            'resolution': domain_resolution,
            'bounds': self.bounds,
            'format': output_format
        }
        
        logger.info(f"Generated masks: "
                   f"inlet points={np.sum(inlet_mask)}, "
                   f"outlet points={np.sum(outlet_mask)}, "
                   f"wall points={np.sum(wall_mask)}")
        
        return masks
    
    def _face_to_grid_mask(self, 
                          face: GeometryFace, 
                          x: np.ndarray, 
                          y: np.ndarray, 
                          z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert a geometry face to grid cell mask indices
        """
        
        # Get face bounding box
        vertices = face.vertices
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        
        # Find grid cells within bounding box
        x_indices = np.where((x >= min_coords[0]) & (x <= max_coords[0]))[0]
        y_indices = np.where((y >= min_coords[1]) & (y <= max_coords[1]))[0]
        z_indices = np.where((z >= min_coords[2]) & (z <= max_coords[2]))[0]
        
        # Create meshgrid for these indices
        xi, yi, zi = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
        
        return (xi.flatten(), yi.flatten(), zi.flatten())
    
    def save_configuration(self, 
                          config: Dict[str, Any], 
                          output_file: str,
                          format: str = "json") -> None:
        """
        Save the generated configuration to file
        
        Args:
            config: Configuration dictionary
            output_file: Output file path
            format: Output format ("json", "yaml")
        """
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2, default=self._json_serializer)
        elif format.lower() == "yaml":
            try:
                import yaml
                with open(output_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            except ImportError:
                logger.warning("PyYAML not available. Saving as JSON instead.")
                with open(output_path.with_suffix('.json'), 'w') as f:
                    json.dump(config, f, indent=2, default=self._json_serializer)
        
        logger.info(f"Saved configuration to {output_path}")
    
    def save_boundary_masks(self, 
                           masks: Dict[str, Any], 
                           output_directory: str) -> None:
        """
        Save boundary masks to files
        
        Args:
            masks: Boundary masks dictionary
            output_directory: Output directory path
        """
        
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each mask
        for mask_name, mask_data in masks.items():
            if mask_name.endswith('_mask'):
                mask_file = output_dir / f"{mask_name}.npy"
                np.save(mask_file, mask_data)
                logger.info(f"Saved {mask_name} to {mask_file}")
        
        # Save metadata
        metadata = {
            'resolution': masks['resolution'],
            'bounds': masks['bounds'],
            'format': masks['format']
        }
        
        metadata_file = output_dir / "mask_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=self._json_serializer)
        
        logger.info(f"Saved mask metadata to {metadata_file}")
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy arrays"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def generate_levelset_integration(self, 
                                    sdf_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate level-set integration configuration for SDF-based geometries
        
        Args:
            sdf_file: Path to SDF file (optional)
            
        Returns:
            Level-set configuration for JAX-Fluids
        """
        
        levelset_config = {
            "levelset": {
                "model": "FLUID_SOLID_DYNAMIC",
                "solid_velocity": {
                    "u": 0.0,
                    "v": 0.0, 
                    "w": 0.0
                }
            }
        }
        
        if sdf_file:
            levelset_config["initial_condition"] = {
                "levelset": f"CUSTOM_SDF({sdf_file})"
            }
        
        return levelset_config
    
    def get_generation_summary(self) -> Dict[str, Any]:
        """Get summary of boundary condition generation"""
        
        face_summary = self.face_tagger.get_tagging_summary()
        validation = self.face_tagger.validate_tagging()
        
        return {
            'geometry_file': str(self.geometry_parser.geometry_file),
            'geometry_format': self.geometry_parser.file_format,
            'face_summary': face_summary,
            'validation': validation,
            'flow_axis': self.face_tagger.primary_flow_axis,
            'nozzle_type': self.face_tagger.nozzle_type.value if self.face_tagger.nozzle_type else None,
            'generation_timestamp': np.datetime64('now').astype(str)
        } 