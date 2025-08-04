#!/usr/bin/env python3
"""
JAX-Fluids Configuration Generator
Generates complete JAX-Fluids setup.json and numerical.json with forcing terms
Based on the working rocket_simulation_final configuration
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

class JAXConfigGenerator:
    """
    Generates JAX-Fluids configuration files for internal flow simulations
    with virtual boundary conditions using the forcing system
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize configuration generator
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.case_config = None
        self.numerical_config = None
        
    def generate_case_setup(self,
                           case_name: str,
                           domain_bounds: list,
                           grid_shape: tuple,
                           sdf_path: str,
                           end_time: float = 0.05,
                           save_dt: float = 0.005) -> Dict[str, Any]:
        """
        Generate JAX-Fluids case setup configuration
        
        Args:
            case_name: Name of the simulation case
            domain_bounds: [x_min, y_min, z_min, x_max, y_max, z_max] 
            grid_shape: (nx, ny, nz)
            sdf_path: Path to SDF file for levelset
            end_time: Simulation end time
            save_dt: Save interval
            
        Returns:
            Configuration dictionary
        """
        self.logger.info("Generating case setup configuration...")
        
        x_min, y_min, z_min, x_max, y_max, z_max = domain_bounds
        nx, ny, nz = grid_shape
        
        self.case_config = {
            "general": {
                "case_name": case_name,
                "end_time": end_time,
                "save_path": "./output/",
                "save_dt": save_dt
            },
            "restart": {
                "flag": False,
                "file_path": ""
            },
            "domain": {
                "x": {
                    "cells": nx,
                    "range": [x_min, x_max]
                },
                "y": {
                    "cells": ny,
                    "range": [y_min, y_max]
                },
                "z": {
                    "cells": nz,
                    "range": [z_min, z_max]
                },
                "decomposition": {
                    "split_x": 1,
                    "split_y": 1,
                    "split_z": 1
                }
            },
            "boundary_conditions": {
                "east": {"type": "SYMMETRY"},
                "west": {"type": "SYMMETRY"},
                "north": {"type": "SYMMETRY"},
                "south": {"type": "SYMMETRY"},
                "top": {"type": "SYMMETRY"},
                "bottom": {"type": "SYMMETRY"}
            },
            "initial_condition": {
                "primitives": {
                    "rho": 1.0,
                    "u": 10.0,
                    "v": 0.0,
                    "w": 0.0,
                    "p": 1000000.0
                },
                "levelset": sdf_path
            },
            "forcings": {
                "mass_flow": {
                    "direction": "x",
                    "target_value": 15.0
                },
                "temperature": {
                    "target_value": 1500.0
                }
            },
            "material_properties": {
                "equation_of_state": {
                    "model": "IdealGas",
                    "specific_heat_ratio": 1.4,
                    "specific_gas_constant": 287.0
                },
                "transport": {
                    "dynamic_viscosity": {
                        "model": "CUSTOM",
                        "value": 1.8e-05
                    },
                    "bulk_viscosity": 0.0,
                    "thermal_conductivity": {
                        "model": "PRANDTL",
                        "prandtl_number": 0.72
                    }
                }
            },
            "nondimensionalization_parameters": {
                "density_reference": 1.0,
                "length_reference": 1.0,
                "velocity_reference": 50.0,
                "temperature_reference": 288.15
            },
            "output": {
                "primitives": [
                    "density",
                    "velocity",
                    "pressure", 
                    "temperature"
                ],
                "miscellaneous": [
                    "mach_number"
                ],
                "quantities": {
                    "levelset": ["levelset"]
                }
            }
        }
        
        self.logger.debug("Case setup configuration generated")
        return self.case_config
        
    def generate_numerical_setup(self) -> Dict[str, Any]:
        """
        Generate JAX-Fluids numerical setup configuration
        
        Returns:
            Numerical configuration dictionary
        """
        self.logger.info("Generating numerical setup configuration...")
        
        self.numerical_config = {
            "conservatives": {
                "convective_fluxes": {
                    "split_scheme": "GODUNOV",
                    "riemann_solver": "HLLC",
                    "reconstruction": "WENO5",
                    "reconstruction_variables": "CONSERVATIVE"
                },
                "diffusive_fluxes": {
                    "temperature": "CENTRAL",
                    "velocity": "CENTRAL"
                },
                "time_integration": {
                    "integrator": "RK3",
                    "CFL": 0.9
                }
            },
            "active_physics": {
                "NAVIERSTOKES": True
            },
            "levelset": {
                "narrowband": {
                    "computation_width": 8,
                    "reinitialization_width": 12
                },
                "interface_interaction": {
                    "levelset_model": "FLUID_FLUID"
                }
            },
            "precision": {
                "active_forcings": "DOUBLE",
                "inactive_forcings": "DOUBLE",
                "computation_dof": "DOUBLE"
            },
            "output": {
                "format": "HDF5"
            }
        }
        
        self.logger.debug("Numerical setup configuration generated")
        return self.numerical_config
        
    def generate_simulation_parameters(self,
                                     max_iterations: int = 100,
                                     save_interval: int = 10,
                                     convergence_tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Generate simulation runtime parameters
        
        Args:
            max_iterations: Maximum simulation iterations
            save_interval: Output save interval
            convergence_tolerance: Convergence tolerance
            
        Returns:
            Parameters dictionary
        """
        self.logger.info("Generating simulation parameters...")
        
        params = {
            "simulation": {
                "max_iterations": max_iterations,
                "convergence_tolerance": convergence_tolerance,
                "max_simulation_time": 0.01
            },
            "execution": {
                "max_iterations": max_iterations,
                "save_interval": save_interval,
                "monitoring_interval": 1,
                "convergence_tolerance": convergence_tolerance,
                "max_simulation_time": 0.01,
                "output_fields": ["density", "velocity", "pressure", "temperature", "mach_number"],
                "checkpoint_interval": 50
            },
            "physics": {
                "compressible_flow": True,
                "supersonic": True,
                "viscous": True,
                "heat_transfer": True
            },
            "grid": {
                "type": "cartesian",
                "uniform": True
            },
            "output": {
                "format": "hdf5",
                "precision": "double",
                "compression": True
            }
        }
        
        return params
        
    def customize_forcing_parameters(self,
                                   inlet_mass_flow: float = 15.0,
                                   inlet_temperature: float = 1500.0,
                                   outlet_pressure: float = None) -> Dict[str, Any]:
        """
        Customize forcing parameters for specific flow conditions
        
        Args:
            inlet_mass_flow: Target mass flow rate
            inlet_temperature: Target inlet temperature
            outlet_pressure: Target outlet pressure (if None, uses outflow)
            
        Returns:
            Updated forcing configuration
        """
        if self.case_config is None:
            raise RuntimeError("Must generate case setup first")
            
        forcings = {
            "mass_flow": {
                "direction": "x",
                "target_value": inlet_mass_flow
            },
            "temperature": {
                "target_value": inlet_temperature
            }
        }
        
        if outlet_pressure is not None:
            forcings["pressure"] = {
                "target_value": outlet_pressure
            }
            
        self.case_config["forcings"] = forcings
        
        self.logger.info(f"Updated forcing parameters:")
        self.logger.info(f"  Mass flow: {inlet_mass_flow}")
        self.logger.info(f"  Temperature: {inlet_temperature}")
        if outlet_pressure:
            self.logger.info(f"  Outlet pressure: {outlet_pressure}")
            
        return forcings
        
    def save_configurations(self, output_dir: str):
        """
        Save configuration files to directory
        
        Args:
            output_dir: Output directory path
        """
        if self.case_config is None or self.numerical_config is None:
            raise RuntimeError("Must generate configurations first")
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save case setup
        case_file = output_path / "rocket_setup.json"
        with open(case_file, 'w', encoding='utf-8') as f:
            json.dump(self.case_config, f, indent=2)
            
        # Save numerical setup
        numerical_file = output_path / "numerical_setup.json"
        with open(numerical_file, 'w', encoding='utf-8') as f:
            json.dump(self.numerical_config, f, indent=2)
            
        # Save simulation parameters
        params = self.generate_simulation_parameters()
        params_file = output_path / "simulation_parameters.json"
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2)
            
        self.logger.info(f"Configuration files saved:")
        self.logger.info(f"  Case setup: {case_file}")
        self.logger.info(f"  Numerical setup: {numerical_file}")
        self.logger.info(f"  Parameters: {params_file}")
        
    def validate_configuration(self) -> bool:
        """
        Validate generated configurations
        
        Returns:
            True if valid, False otherwise
        """
        if self.case_config is None or self.numerical_config is None:
            self.logger.error("Configurations not generated")
            return False
            
        # Check required sections in case config
        required_case_sections = [
            "general", "domain", "boundary_conditions", 
            "initial_condition", "material_properties", "output"
        ]
        
        for section in required_case_sections:
            if section not in self.case_config:
                self.logger.error(f"Missing required section: {section}")
                return False
                
        # Check required sections in numerical config
        required_numerical_sections = [
            "conservatives", "active_physics", "levelset", "output"
        ]
        
        for section in required_numerical_sections:
            if section not in self.numerical_config:
                self.logger.error(f"Missing required numerical section: {section}")
                return False
                
        # Validate domain consistency
        domain = self.case_config["domain"]
        for axis in ["x", "y", "z"]:
            if "cells" not in domain[axis] or "range" not in domain[axis]:
                self.logger.error(f"Invalid domain specification for axis {axis}")
                return False
                
        # Check forcing section exists (key for internal BCs)
        if "forcings" not in self.case_config:
            self.logger.warning("No forcing terms defined - internal BCs may not work")
            
        self.logger.info("Configuration validation passed")
        return True
        
    def create_jax_runner_template(self, output_dir: str):
        """
        Create template JAX-Fluids runner script
        
        Args:
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        
        runner_template = '''#!/usr/bin/env python3
"""
JAX-Fluids Runner Script
Auto-generated for internal flow simulation with virtual boundary conditions
"""

import logging
import numpy as np
from pathlib import Path
from jaxfluids import InputManager, InitializationManager, SimulationManager

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simulation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_boundary_masks():
    """Load boundary condition masks"""
    logger = logging.getLogger(__name__)
    
    mask_dir = Path("masks")
    inlet_file = mask_dir / "inlet_boundary_mask.npy"
    outlet_file = mask_dir / "outlet_boundary_mask.npy"
    
    if not inlet_file.exists() or not outlet_file.exists():
        raise FileNotFoundError("Boundary masks not found")
        
    inlet_mask = np.load(inlet_file)
    outlet_mask = np.load(outlet_file)
    
    logger.info(f"Loaded masks: inlet={np.sum(inlet_mask)}, outlet={np.sum(outlet_mask)}")
    
    return inlet_mask, outlet_mask

def run_simulation():
    """Run JAX-Fluids simulation"""
    logger = setup_logging()
    logger.info("Starting JAX-Fluids simulation...")
    
    # Load configuration
    case_setup = "config/rocket_setup.json"
    numerical_setup = "config/numerical_setup.json"
    
    # Initialize JAX-Fluids
    input_manager = InputManager(case_setup, numerical_setup)
    initialization_manager = InitializationManager(input_manager)
    simulation_manager = SimulationManager(input_manager)
    
    # Initialize simulation
    buffer_dict = initialization_manager.initialization()
    
    # Load boundary masks (for monitoring)
    inlet_mask, outlet_mask = load_boundary_masks()
    
    # Run simulation
    logger.info("Running simulation loop...")
    buffer_dict = simulation_manager.simulate(buffer_dict)
    
    logger.info("Simulation completed successfully!")

if __name__ == "__main__":
    run_simulation()
'''
        
        runner_file = output_path / "run_simulation.py"
        with open(runner_file, 'w', encoding='utf-8') as f:
            f.write(runner_template)
            
        self.logger.info(f"Runner template created: {runner_file}")
        
    def get_configuration_summary(self) -> str:
        """Get human-readable configuration summary"""
        if self.case_config is None:
            return "No configuration generated"
            
        domain = self.case_config["domain"]
        grid_info = f"{domain['x']['cells']}x{domain['y']['cells']}x{domain['z']['cells']}"
        
        x_range = domain['x']['range']
        y_range = domain['y']['range'] 
        z_range = domain['z']['range']
        
        forcings = self.case_config.get("forcings", {})
        forcing_info = []
        for key, value in forcings.items():
            if isinstance(value, dict) and "target_value" in value:
                forcing_info.append(f"{key}={value['target_value']}")
                
        summary = f"""
JAX-Fluids Configuration Summary:
================================
Case: {self.case_config['general']['case_name']}
Grid: {grid_info}
Domain: X=[{x_range[0]}, {x_range[1]}]
        Y=[{y_range[0]}, {y_range[1]}]
        Z=[{z_range[0]}, {z_range[1]}]
End Time: {self.case_config['general']['end_time']}
Forcing: {', '.join(forcing_info) if forcing_info else 'None'}
Physics: Navier-Stokes + Levelset
Output: HDF5 format
"""
        
        return summary 