#!/usr/bin/env python3
"""
Wind Tunnel Configuration Generator for SU2 CFD

This module provides automated configuration generation for SU2 CFD wind tunnel simulations.
It includes marker extraction from mesh files, proper boundary condition handling, and 
wind tunnel orientation support with SU2 compatibility fixes.
"""

import os
import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WindTunnelOrientation(Enum):
    """Wind tunnel flow direction orientations"""
    POSITIVE_X = "+X"  # Flow in +X direction (standard)
    NEGATIVE_X = "-X"  # Flow in -X direction
    POSITIVE_Y = "+Y"  # Flow in +Y direction
    NEGATIVE_Y = "-Y"  # Flow in -Y direction
    POSITIVE_Z = "+Z"  # Flow in +Z direction
    NEGATIVE_Z = "-Z"  # Flow in -Z direction

def get_flow_direction_vector(orientation: WindTunnelOrientation) -> Tuple[float, float, float]:
    """Get the flow direction vector for the given orientation"""
    direction_map = {
        WindTunnelOrientation.POSITIVE_X: (1.0, 0.0, 0.0),
        WindTunnelOrientation.NEGATIVE_X: (-1.0, 0.0, 0.0),
        WindTunnelOrientation.POSITIVE_Y: (0.0, 1.0, 0.0),
        WindTunnelOrientation.NEGATIVE_Y: (0.0, -1.0, 0.0),
        WindTunnelOrientation.POSITIVE_Z: (0.0, 0.0, 1.0),
        WindTunnelOrientation.NEGATIVE_Z: (0.0, 0.0, -1.0),
    }
    return direction_map[orientation]

def extract_mesh_markers(mesh_file_path: str) -> Dict[str, List[str]]:
    """Extract boundary markers from SU2 mesh file"""
    logger.info(f"Extracting markers from mesh file: {mesh_file_path}")
    
    markers = {
        'wall_markers': [],
        'farfield_markers': [],
        'inlet_markers': [],
        'outlet_markers': [],
        'symmetry_markers': []
    }
    
    try:
        if not os.path.exists(mesh_file_path):
            logger.warning(f"Mesh file not found: {mesh_file_path}")
            # Return default markers for meshes without boundary info
            markers['wall_markers'] = ['wall']
            markers['farfield_markers'] = ['farfield']
            return markers
        
        with open(mesh_file_path, 'r') as f:
            content = f.read()
        
        # Look for boundary marker definitions
        nmark_pattern = r'NMARK=\s*(\d+)'
        nmark_match = re.search(nmark_pattern, content)
        
        if nmark_match:
            nmark = int(nmark_match.group(1))
            logger.info(f"Found {nmark} boundary markers")
            
            if nmark == 0:
                logger.warning("NMARK=0, no boundary markers defined in mesh")
                # Use default markers for meshes without explicit boundary definitions
                markers['wall_markers'] = ['wall']
                markers['farfield_markers'] = ['farfield']
                return markers
            
            # Extract marker names and types
            marker_pattern = r'MARKER_TAG=\s*([^\s\n]+)'
            marker_matches = re.findall(marker_pattern, content)
            
            for marker_name in marker_matches:
                marker_lower = marker_name.lower()
                
                # Categorize markers based on name patterns
                if any(wall_term in marker_lower for wall_term in ['wall', 'airfoil', 'blade', 'surface']):
                    markers['wall_markers'].append(marker_name)
                elif any(far_term in marker_lower for far_term in ['far', 'inlet', 'outlet', 'pressure']):
                    markers['farfield_markers'].append(marker_name)
                elif 'symmetry' in marker_lower or 'sym' in marker_lower:
                    markers['symmetry_markers'].append(marker_name)
                else:
                    # Default unknown markers to farfield
                    markers['farfield_markers'].append(marker_name)
        
        else:
            logger.warning("Could not find NMARK in mesh file")
            # Use default markers
            markers['wall_markers'] = ['wall']
            markers['farfield_markers'] = ['farfield']
        
        # Ensure we have at least default markers
        if not markers['wall_markers']:
            markers['wall_markers'] = ['wall']
        if not markers['farfield_markers']:
            markers['farfield_markers'] = ['farfield']
        
        logger.info(f"Extracted markers: {markers}")
        return markers
        
    except Exception as e:
        logger.error(f"Error extracting markers: {e}")
        # Return safe defaults
        return {
            'wall_markers': ['wall'],
            'farfield_markers': ['farfield'],
            'inlet_markers': [],
            'outlet_markers': [],
            'symmetry_markers': []
                }

@dataclass
class FlowConditions:
    """Flow conditions for wind tunnel simulation"""
    mach_number: float = 0.2  # Updated from convergence tests: validated up to 0.2
    reynolds_number: float = 1e5  # Updated: conservative value that works well
    angle_of_attack: float = 0.0
    sideslip_angle: float = 0.0
    temperature: float = 288.15  # Kelvin
    pressure: float = 101325.0  # Pascal
    velocity_magnitude: Optional[float] = None
    flow_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    wind_tunnel_orientation: WindTunnelOrientation = WindTunnelOrientation.POSITIVE_X

    def __post_init__(self):
        # Update flow direction based on orientation
        self.flow_direction = get_flow_direction_vector(self.wind_tunnel_orientation)

@dataclass
class TurbulenceModel:
    """Turbulence model configuration"""
    model_type: str = "SA"  # SA, SST, NONE
    nu_factor: float = 3.0
    turbulence_intensity: float = 0.05
    turb2lam_ratio: float = 10.0

@dataclass
class BoundaryConditions:
    """Boundary conditions for the simulation"""
    wall_markers: List[str] = None
    farfield_markers: List[str] = None
    inlet_markers: List[str] = None
    outlet_markers: List[str] = None
    symmetry_markers: List[str] = None
    wall_type: str = "EULER"  # HEATFLUX, ISOTHERMAL, EULER
    wall_value: float = 0.0  # Heat flux or temperature value

    def __post_init__(self):
        if self.wall_markers is None:
            self.wall_markers = []
        if self.farfield_markers is None:
            self.farfield_markers = ["farfield"]
        if self.inlet_markers is None:
            self.inlet_markers = []
        if self.outlet_markers is None:
            self.outlet_markers = []
        if self.symmetry_markers is None:
            self.symmetry_markers = []

@dataclass
class SolverSettings:
    """Solver configuration settings with convergence-tested parameters"""
    solver_type: str = "EULER"  # EULER proven to work well from convergence tests
    max_iterations: int = 200
    convergence_tolerance: float = 1e-6
    cfl_number: float = 0.1  # Validated in convergence tests up to 0.1
    cfl_adapt: bool = False  # Disabled for stability as used in convergence tests
    cfl_adapt_param: Tuple[float, float, float, float] = (0.1, 2.0, 10.0, 1e10)
    linear_solver: str = "FGMRES"
    linear_solver_prec: str = "ILU"
    convective_scheme: str = "ROE"  # ROE scheme validated in convergence tests
    time_discretization: str = "EULER_IMPLICIT"
    multigrid_levels: int = 3
    multigrid: bool = True
    
    # Additional stability settings
    venkat_limiter_coeff: float = 0.05  # More limiting for stability
    jst_sensor_coeff: Tuple[float, float] = (0.5, 0.02)

@dataclass
class OutputSettings:
    """Output configuration settings"""
    output_format: str = "PARAVIEW"
    history_frequency: int = 10  # More frequent output for monitoring
    screen_frequency: int = 1
    solution_frequency: int = 250  # More frequent solution output
    surface_output: bool = True
    volume_output: bool = True
    convergence_history: bool = True

@dataclass
class WindTunnelConfig:
    """Complete wind tunnel configuration"""
    mesh_file: str
    flow_conditions: FlowConditions
    boundary_conditions: BoundaryConditions
    turbulence: TurbulenceModel
    solver: SolverSettings
    output: OutputSettings
    reference_length: float = 1.0
    reference_area: float = 1.0
    moment_reference_point: Tuple[float, float, float] = (0.25, 0.0, 0.0)

class SU2ConfigGenerator:
    """Generates SU2 configuration files from WindTunnelConfig objects"""
    
    def __init__(self, su2_executable: str = None):
        self.su2_executable = su2_executable
    
    def _load_base_template(self) -> str:
        """Load base configuration template with improved convergence settings"""
        return """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
% SU2 configuration file for wind tunnel simulation                           %
% Generated by Wind Tunnel Configuration Generator                            %
% Generated on: {timestamp}                                                   %
% Configuration ID: {config_id}                                               %
% Wind Tunnel Orientation: {orientation}                                      %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ------------- DIRECT, ADJOINT, AND LINEARIZED PROBLEM DEFINITION ------------%
SOLVER= {solver_type}

% ----------- COMPRESSIBLE AND INCOMPRESSIBLE FREE-STREAM DEFINITION ----------%
MACH_NUMBER= {mach_number}
AOA= {angle_of_attack}
SIDESLIP_ANGLE= {sideslip_angle}
REYNOLDS_NUMBER= {reynolds_number}
REYNOLDS_LENGTH= {reference_length}

% ----------------------- FREESTREAM SPECIFICATION ----------------------------%
FREESTREAM_TEMPERATURE= {temperature}
FREESTREAM_PRESSURE= {pressure}

% ---- IDEAL GAS, POLYTROPIC, VAN DER WAALS AND PENG ROBINSON CONSTANTS -------%
GAMMA_VALUE= 1.4
GAS_CONSTANT= 287.058

% --------------------------- VISCOSITY MODEL ---------------------------------%
VISCOSITY_MODEL= SUTHERLAND
MU_REF= 1.716E-5
MU_T_REF= 273.15
SUTHERLAND_CONSTANT= 110.4

% --------------------------- THERMAL CONDUCTIVITY MODEL ----------------------%
CONDUCTIVITY_MODEL= CONSTANT_PRANDTL
PRANDTL_LAM= 0.72
PRANDTL_TURB= 0.90

% ---------------------- REFERENCE VALUE DEFINITION ---------------------------%
REF_ORIGIN_MOMENT_X= {moment_x}
REF_ORIGIN_MOMENT_Y= {moment_y}
REF_ORIGIN_MOMENT_Z= {moment_z}
REF_LENGTH= {reference_length}
REF_AREA= {reference_area}

% ----------------------- BOUNDARY CONDITION DEFINITION -----------------------%
{boundary_conditions}

% ------------------------ SURFACES IDENTIFICATION ----------------------------%
{surface_markers}

% ------------- COMMON PARAMETERS DEFINING THE NUMERICAL METHOD ---------------%
NUM_METHOD_GRAD= {gradient_method}
CFL_NUMBER= {cfl_number}
CFL_ADAPT= {cfl_adapt}
CFL_ADAPT_PARAM= {cfl_adapt_param}

% ------------------------ LINEAR SOLVER DEFINITION ---------------------------%
LINEAR_SOLVER= {linear_solver}
LINEAR_SOLVER_PREC= {linear_solver_prec}
LINEAR_SOLVER_ERROR= 1E-8
LINEAR_SOLVER_ITER= 10

% ----------------------- MULTIGRID PARAMETERS -------------------------------%
MGLEVEL= {multigrid_levels}
MGCYCLE= W_CYCLE
MG_PRE_SMOOTH= ( 1, 2, 3, 3 )
MG_POST_SMOOTH= ( 0, 0, 0, 0 )
MG_CORRECTION_SMOOTH= ( 0, 0, 0, 0 )
MG_DAMP_RESTRICTION= 0.9
MG_DAMP_PROLONGATION= 0.9

% -------------------- FLOW NUMERICAL METHOD DEFINITION -----------------------%
CONV_NUM_METHOD_FLOW= {convective_scheme}
MUSCL_FLOW= {muscl_flow}
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= {venkat_limiter_coeff}
JST_SENSOR_COEFF= {jst_sensor_coeff}
TIME_DISCRE_FLOW= {time_discretization}

{turbulence_settings}

% --------------------------- CONVERGENCE PARAMETERS --------------------------%
ITER= {max_iterations}
CONV_RESIDUAL_MINVAL= {convergence_tolerance}
CONV_STARTITER= 10
CONV_CAUCHY_ELEMS= 100
CONV_CAUCHY_EPS= 1E-10
CONV_FIELD= (RMS_DENSITY, RMS_MOMENTUM-X, RMS_MOMENTUM-Y, RMS_ENERGY, LIFT, DRAG)

% ------------------------- INPUT/OUTPUT INFORMATION --------------------------%
MESH_FILENAME= {mesh_file}
MESH_FORMAT= SU2
MESH_OUT_FILENAME= mesh_out.su2
SOLUTION_FILENAME= solution_flow.dat
SOLUTION_ADJ_FILENAME= solution_adj.dat
TABULAR_FORMAT= CSV
CONV_FILENAME= history
RESTART_FILENAME= restart_flow.dat
RESTART_ADJ_FILENAME= restart_adj.dat
VOLUME_FILENAME= flow
VOLUME_ADJ_FILENAME= adjoint
GRAD_OBJFUNC_FILENAME= of_grad.dat
SURFACE_FILENAME= surface_flow
SURFACE_ADJ_FILENAME= surface_adjoint
SCREEN_OUTPUT= (INNER_ITER, RMS_DENSITY, RMS_MOMENTUM-X, RMS_MOMENTUM-Y, RMS_ENERGY, LIFT, DRAG)
SCREEN_WRT_FREQ_INNER= {screen_frequency}

% ------------------------- OUTPUT CONFIGURATION -----------------------------%
% Output file format and frequency
OUTPUT_FILES= (RESTART, PARAVIEW, SURFACE_PARAVIEW)
OUTPUT_WRT_FREQ= {history_frequency}
%
% Volume output fields for visualization and analysis
VOLUME_OUTPUT= (COORDINATES, SOLUTION, PRIMITIVE)
%
% History output fields for convergence monitoring and graph generation
HISTORY_OUTPUT= (ITER, RMS_RES, AERO_COEFF)

% ----------------------- DESIGN VARIABLE PARAMETERS -------------------------%
DV_KIND= HICKS_HENNE
DV_MARKER= {design_markers}
DV_PARAM= ( 1, 0.5 )
DV_VALUE= 0.01

% ------------------------ GRID DEFORMATION PARAMETERS -----------------------%
DEFORM_LINEAR_SOLVER= FGMRES
DEFORM_LINEAR_SOLVER_ITER= 500
DEFORM_NONLINEAR_ITER= 1
DEFORM_CONSOLE_OUTPUT= YES
DEFORM_STIFFNESS_TYPE= INVERSE_VOLUME

% -------------------- FREE-FORM DEFORMATION PARAMETERS ----------------------%
FFD_TOLERANCE= 1E-10
FFD_ITERATIONS= 500
"""

    def generate_boundary_conditions(self, bc: BoundaryConditions) -> str:
        """Generate boundary condition section"""
        conditions = []
        
        # Wall boundaries
        if bc.wall_markers:
            wall_marker_str = ", ".join(bc.wall_markers)
            if bc.wall_type.upper() == "HEATFLUX":
                conditions.append(f"MARKER_HEATFLUX= ( {wall_marker_str}, {bc.wall_value} )")
            elif bc.wall_type.upper() == "ISOTHERMAL":
                conditions.append(f"MARKER_ISOTHERMAL= ( {wall_marker_str}, {bc.wall_value} )")
            else:  # EULER wall
                conditions.append(f"MARKER_EULER= ( {wall_marker_str} )")
        
        # Farfield boundaries
        if bc.farfield_markers:
            farfield_marker_str = ", ".join(bc.farfield_markers)
            conditions.append(f"MARKER_FAR= ( {farfield_marker_str} )")
        
        # Inlet boundaries
        if bc.inlet_markers:
            inlet_marker_str = ", ".join(bc.inlet_markers)
            conditions.append(f"MARKER_INLET= ( {inlet_marker_str} )")
        
        # Outlet boundaries  
        if bc.outlet_markers:
            outlet_marker_str = ", ".join(bc.outlet_markers)
            conditions.append(f"MARKER_OUTLET= ( {outlet_marker_str} )")
        
        # Symmetry boundaries
        if bc.symmetry_markers:
            sym_marker_str = ", ".join(bc.symmetry_markers)
            conditions.append(f"MARKER_SYM= ( {sym_marker_str} )")
        else:
            conditions.append("MARKER_SYM= ( NONE )")
        
        return "\n".join(conditions)

    def generate_surface_markers(self, bc: BoundaryConditions) -> str:
        """Generate surface marker definitions"""
        all_markers = []
        all_markers.extend(bc.wall_markers)
        all_markers.extend(bc.farfield_markers)
        
        if all_markers:
            marker_str = ", ".join(all_markers)
            plotting_line = f"MARKER_PLOTTING= ( {marker_str} )"
            monitoring_line = f"MARKER_MONITORING= ( {marker_str} )"
            designing_line = f"MARKER_DESIGNING = ( {marker_str} )"
            
            return f"{plotting_line}\n{monitoring_line}\n{designing_line}"
        else:
            return "MARKER_PLOTTING= ( NONE )\nMARKER_MONITORING= ( NONE )\nMARKER_DESIGNING = ( NONE )"

    def generate_turbulence_settings(self, turb: TurbulenceModel, solver: SolverSettings) -> str:
        """Generate turbulence model settings without invalid options"""
        if turb.model_type.upper() == "NONE":
            return "% No turbulence model"
        
        elif turb.model_type.upper() == "SA":
            return """
% ----------------------- TURBULENCE MODEL DEFINITION -------------------------%
KIND_TURB_MODEL= SA
SA_OPTIONS= BCM

% --------------------------- TURBULENCE NUMERICS -----------------------------%
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO
SLOPE_LIMITER_TURB= VENKATAKRISHNAN
TIME_DISCRE_TURB= EULER_IMPLICIT
CFL_REDUCTION_TURB= 0.8
"""
        
        elif turb.model_type.upper() == "SST":
            return """
% ----------------------- TURBULENCE MODEL DEFINITION -------------------------%
KIND_TURB_MODEL= SST
SST_OPTIONS= BCM

% --------------------------- TURBULENCE NUMERICS -----------------------------%
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO
SLOPE_LIMITER_TURB= VENKATAKRISHNAN
TIME_DISCRE_TURB= EULER_IMPLICIT
CFL_REDUCTION_TURB= 0.8
"""
        
        return "% Unknown turbulence model"

    def generate_config(self, config: WindTunnelConfig) -> str:
        """Generate complete configuration file"""
        boundary_conditions = self.generate_boundary_conditions(config.boundary_conditions)
        surface_markers = self.generate_surface_markers(config.boundary_conditions)
        turbulence_settings = self.generate_turbulence_settings(config.turbulence, config.solver)
        
        # Get design markers (typically wall markers)
        design_markers = ", ".join(config.boundary_conditions.wall_markers) if config.boundary_conditions.wall_markers else "NONE"
        
        # Format CFL adapt parameters
        cfl_adapt_param_str = f"( {config.solver.cfl_adapt_param[0]}, {config.solver.cfl_adapt_param[1]}, {config.solver.cfl_adapt_param[2]}, {config.solver.cfl_adapt_param[3]} )"
        jst_sensor_coeff_str = f"( {config.solver.jst_sensor_coeff[0]}, {config.solver.jst_sensor_coeff[1]} )"
        
        # Set MUSCL_FLOW to NO for maximum numerical stability and compatibility
        # This prevents "Centered schemes do not use MUSCL reconstruction" errors
        muscl_flow = "NO"  # Force NO for all schemes to ensure stability
        
        template = self._load_base_template()
        
        return template.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            config_id=str(uuid.uuid4())[:8],
            orientation=config.flow_conditions.wind_tunnel_orientation.value,
            solver_type=config.solver.solver_type,
            mach_number=config.flow_conditions.mach_number,
            angle_of_attack=config.flow_conditions.angle_of_attack,
            sideslip_angle=config.flow_conditions.sideslip_angle,
            reynolds_number=config.flow_conditions.reynolds_number,
            temperature=config.flow_conditions.temperature,
            pressure=config.flow_conditions.pressure,
            reference_length=config.reference_length,
            reference_area=config.reference_area,
            moment_x=config.moment_reference_point[0],
            moment_y=config.moment_reference_point[1],
            moment_z=config.moment_reference_point[2],
            boundary_conditions=boundary_conditions,
            surface_markers=surface_markers,
            gradient_method="WEIGHTED_LEAST_SQUARES" if config.solver.convective_scheme == "JST" else "GREEN_GAUSS",
            cfl_number=config.solver.cfl_number,
            cfl_adapt="YES" if config.solver.cfl_adapt else "NO",
            cfl_adapt_param=cfl_adapt_param_str,
            linear_solver=config.solver.linear_solver,
            linear_solver_prec=config.solver.linear_solver_prec,
            multigrid_levels=config.solver.multigrid_levels,
            convective_scheme=config.solver.convective_scheme,
            muscl_flow=muscl_flow,
            venkat_limiter_coeff=config.solver.venkat_limiter_coeff,
            jst_sensor_coeff=jst_sensor_coeff_str,
            time_discretization=config.solver.time_discretization,
            turbulence_settings=turbulence_settings,
            max_iterations=config.solver.max_iterations,
            convergence_tolerance=config.solver.convergence_tolerance,
            mesh_file=config.mesh_file,
            history_frequency=config.output.history_frequency,
            screen_frequency=config.output.screen_frequency,
            design_markers=design_markers
        )

class WindTunnelSimulation:
    """Main class for managing wind tunnel simulations"""
    
    def __init__(self, su2_executable: str = None, workspace_dir: str = None):
        self.su2_executable = su2_executable or "C:\\Users\\kriss\\Desktop\\CAE AI\\win64-mpi\\bin\\SU2_CFD.exe"
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path("simulations")
        self.config_generator = SU2ConfigGenerator(su2_executable)

    def create_simulation(self, config: WindTunnelConfig, simulation_name: str = None) -> str:
        """Create a simulation directory with config file"""
        if simulation_name is None:
            simulation_name = f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        sim_dir = self.workspace_dir / simulation_name
        sim_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy mesh file to simulation directory
        mesh_source = Path(config.mesh_file)
        if mesh_source.exists():
            mesh_dest = sim_dir / mesh_source.name
            shutil.copy2(mesh_source, mesh_dest)
            logger.info(f"Copied mesh file: {mesh_source.name}")
            # Update config to use local filename
            config.mesh_file = mesh_source.name
        else:
            logger.warning(f"Mesh file not found: {config.mesh_file}")
        
        # Generate and save configuration file
        config_content = self.config_generator.generate_config(config)
        config_file = sim_dir / f"{simulation_name}.cfg"
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        # Save metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "simulation_name": simulation_name,
            "config_id": str(uuid.uuid4())[:8],
            "mesh_file": config.mesh_file,
            "wind_tunnel_orientation": config.flow_conditions.wind_tunnel_orientation.value,
            "status": "created"
        }
        
        metadata_file = sim_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created simulation: {simulation_name}")
        return str(sim_dir)
    
    def run_simulation(self, simulation_dir: str, max_iterations: int = None) -> bool:
        """Run SU2 simulation"""
        sim_path = Path(simulation_dir)
        
        # Find configuration file
        config_files = list(sim_path.glob("*.cfg"))
        if not config_files:
            logger.error(f"No configuration file found in {sim_path}")
            return False
        
        config_file = config_files[0]
        
        # Check if SU2 executable exists
        if not os.path.exists(self.su2_executable):
            logger.error(f"SU2 executable not found: {self.su2_executable}")
            return False
        
        # Prepare command (use just the filename since we cd to the directory)
        config_filename = config_file.name
        cmd = [self.su2_executable, config_filename]
        
        logger.info(f"Running simulation: {cmd}")
        
        try:
            # Change to simulation directory
            original_dir = os.getcwd()
            os.chdir(sim_path)
            
            # Run simulation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            # Save output
            with open("su2_output.log", "w") as f:
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)
            
            # Update metadata
            metadata_file = sim_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                metadata["status"] = "completed" if result.returncode == 0 else "failed"
                metadata["completed_at"] = datetime.now().isoformat()
                metadata["return_code"] = result.returncode
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            if result.returncode == 0:
                logger.info(f"Simulation completed successfully: {simulation_dir}")
                return True
            else:
                logger.error(f"Simulation failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Simulation timed out after 1 hour")
            return False
        except Exception as e:
            logger.error(f"Error running simulation: {e}")
            return False
        finally:
            os.chdir(original_dir)
    
    def batch_run_simulations(self, simulation_dirs: List[str], max_parallel: int = 1) -> Dict[str, bool]:
        """Run multiple simulations in batch"""
        results = {}
        
        for sim_dir in simulation_dirs:
            logger.info(f"Running simulation: {sim_dir}")
            success = self.run_simulation(sim_dir)
            results[sim_dir] = success
            
            if not success:
                logger.warning(f"Simulation failed: {sim_dir}")
        
        return results

def create_config_with_extracted_markers(mesh_file_path: str, **kwargs) -> WindTunnelConfig:
    """Create configuration with automatically extracted mesh markers and improved settings"""
    
    # Extract markers from mesh file
    markers = extract_mesh_markers(mesh_file_path)
    
    # Default values with convergence-tested settings
    defaults = {
        'mach_number': 0.15,
        'reynolds_number': 1e5,  # Conservative value that works
        'angle_of_attack': 0.0,
        'solver_type': 'EULER',  # Proven to work well in tests
        'turbulence_model': 'NONE',  # For EULER solver
        'max_iterations': 200,   # Reasonable convergence target
        'wind_tunnel_orientation': WindTunnelOrientation.POSITIVE_X
    }
    
    # Update defaults with provided kwargs
    params = {**defaults, **kwargs}
    
    flow_conditions = FlowConditions(
        mach_number=params['mach_number'],
        reynolds_number=params['reynolds_number'],
        angle_of_attack=params['angle_of_attack'],
        wind_tunnel_orientation=params['wind_tunnel_orientation']
    )
    
    boundary_conditions = BoundaryConditions(
        wall_markers=markers['wall_markers'],
        farfield_markers=markers['farfield_markers'],
        inlet_markers=markers['inlet_markers'],
        outlet_markers=markers['outlet_markers'],
        symmetry_markers=markers['symmetry_markers']
    )
    
    turbulence = TurbulenceModel(
        model_type=params['turbulence_model']
    )
    
    # Convergence-tested solver settings
    solver = SolverSettings(
        solver_type=params['solver_type'],
        max_iterations=params['max_iterations'],
        cfl_number=0.1,  # Validated CFL value
        cfl_adapt=False,  # Disabled as used in convergence tests
        convective_scheme="ROE",  # Validated convective scheme
        linear_solver_prec="ILU"  # Better preconditioning
    )
    
    output = OutputSettings(
        history_frequency=10,  # More frequent monitoring
        solution_frequency=250
    )
    
    return WindTunnelConfig(
        mesh_file=mesh_file_path,
        flow_conditions=flow_conditions,
        boundary_conditions=boundary_conditions,
        turbulence=turbulence,
        solver=solver,
        output=output
    )

if __name__ == "__main__":
    # Example usage
    logger.info("Enhanced Wind Tunnel Configuration Generator initialized")
    logger.info("Available orientations: +X, -X, +Y, -Y, +Z, -Z") 