#!/usr/bin/env python3
"""
Complete Wind Tunnel Endpoint
Combines packaged wind tunnel generation with production CFD simulation configuration
"""

import os
import sys
import json
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

# Add paths to import from both endpoints - Updated for VectraSim integration
current_dir = os.path.dirname(os.path.abspath(__file__))
packaged_endpoint_path = os.path.join(current_dir, '..', 'packaged_wind-tunnel_endpoint')
production_endpoint_path = os.path.join(current_dir, '..', 'production_endpoint')

# Add to sys.path if not already there
if packaged_endpoint_path not in sys.path:
    sys.path.append(packaged_endpoint_path)
if production_endpoint_path not in sys.path:
    sys.path.append(production_endpoint_path)

# Import from packaged wind tunnel endpoint
try:
    from wind_tunnel_api import (
        WindTunnelRequest, 
        WindTunnelResponse, 
        GeneralizedWindTunnelGenerator,
        create_wind_tunnel
    )
    print("✅ Successfully imported wind tunnel components")
except ImportError as e:
    print(f"❌ Failed to import wind tunnel components: {e}")
    # Fallback - try direct import
    sys.path.insert(0, packaged_endpoint_path)
    from wind_tunnel_api import (
        WindTunnelRequest, 
        WindTunnelResponse, 
        GeneralizedWindTunnelGenerator,
        create_wind_tunnel
    )

# Note: We don't import CFDParameterAgent here anymore
# The AI parameter generation is handled externally by VectraSim backend
# This endpoint only processes pre-generated JSON configurations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_endpoint.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CompleteWindTunnelRequest:
    """Complete wind tunnel request combining both endpoints"""
    
    # Object mesh file
    object_mesh_file: str
    
    # Wind tunnel generation parameters (packaged endpoint)
    tunnel_type: str = "standard"
    flow_direction: str = "+X"
    mesh_quality: str = "medium"
    domain_scale_factor: float = 1.0
    generate_vtk: bool = True
    
    # CFD simulation parameters (production endpoint)
    flow_type: str = "EULER"  # EULER, RANS, NAVIER_STOKES
    mach_number: float = 0.3
    reynolds_number: float = 1000000.0
    angle_of_attack: float = 0.0
    sideslip_angle: float = 0.0
    max_iterations: int = 100
    turbulence_model: str = "NONE"  # NONE, SA, SST, KE
    turbulence_intensity: float = 0.05
    viscosity_ratio: float = 10.0
    
    # Advanced CFD parameters
    freestream_pressure: float = 101325.0
    freestream_temperature: float = 288.15
    cfl_number: float = 1.0
    convergence_residual: float = 1e-8
    convective_scheme: str = "JST"
    
    # Output configuration
    output_directory: str = "complete_output"
    simulation_name: Optional[str] = None
    
    # Natural language prompt (alternative to explicit parameters)
    prompt: Optional[str] = None

@dataclass
class CompleteWindTunnelResponse:
    """Complete wind tunnel response"""
    success: bool
    message: str
    
    # Wind tunnel generation results
    wind_tunnel_file: Optional[str] = None
    vtk_file: Optional[str] = None
    mesh_stats: Optional[Dict] = None
    
    # CFD configuration results
    config_file: Optional[str] = None
    simulation_directory: Optional[str] = None
    
    # Timing and metadata
    total_time: Optional[float] = None
    wind_tunnel_time: Optional[float] = None
    config_generation_time: Optional[float] = None
    
    # Output files
    output_files: Optional[List[str]] = None
    error_details: Optional[str] = None

class CompleteWindTunnelEndpoint:
    """Complete wind tunnel endpoint combining both systems"""
    
    def __init__(self):
        self.logger = logger
        self.wind_tunnel_generator = GeneralizedWindTunnelGenerator()
        
        print("🚀 Complete Wind Tunnel Endpoint Initialized")
        print("✅ Packaged Wind Tunnel Generator Ready")
        print("✅ Ready to process pre-generated JSON configurations")
    
    def process_complete_request(self, request: CompleteWindTunnelRequest) -> CompleteWindTunnelResponse:
        """
        Process complete wind tunnel request
        1. Generate wind tunnel mesh from object
        2. Create CFD simulation configuration
        3. Organize all outputs
        """
        
        start_time = time.time()
        
        try:
            print("\n" + "="*60)
            print("🌪️ COMPLETE WIND TUNNEL ENDPOINT")
            print("="*60)
            print(f"📁 Object mesh: {request.object_mesh_file}")
            print(f"📂 Output directory: {request.output_directory}")
            
            # Create output directory
            output_dir = Path(request.output_directory)
            output_dir.mkdir(exist_ok=True)
            
            # Step 1: Generate wind tunnel mesh
            print(f"\n🏗️ STEP 1: Generating Wind Tunnel Mesh")
            print("-" * 40)
            
            wind_tunnel_start = time.time()
            wind_tunnel_result = self._generate_wind_tunnel_mesh(request, output_dir)
            wind_tunnel_time = time.time() - wind_tunnel_start
            
            if not wind_tunnel_result.success:
                return CompleteWindTunnelResponse(
                    success=False,
                    message=f"Wind tunnel generation failed: {wind_tunnel_result.message}",
                    error_details=wind_tunnel_result.error_details,
                    total_time=time.time() - start_time
                )
            
            # Step 2: Generate CFD configuration
            print(f"\n⚙️ STEP 2: Generating CFD Configuration")
            print("-" * 40)
            
            config_start = time.time()
            config_result = self._generate_cfd_configuration(request, wind_tunnel_result, output_dir)
            config_time = time.time() - config_start
            
            if not config_result['success']:
                return CompleteWindTunnelResponse(
                    success=False,
                    message=f"CFD configuration failed: {config_result['message']}",
                    wind_tunnel_file=wind_tunnel_result.output_file,
                    vtk_file=wind_tunnel_result.vtk_file,
                    total_time=time.time() - start_time
                )
            
            # Step 3: Organize outputs
            print(f"\n📁 STEP 3: Organizing Outputs")
            print("-" * 40)
            
            output_files = self._organize_outputs(request, wind_tunnel_result, config_result, output_dir)
            
            total_time = time.time() - start_time
            
            print(f"\n🎉 Complete Wind Tunnel Processing Successful!")
            print(f"   📁 Output directory: {output_dir}")
            print(f"   🌪️ Wind tunnel mesh: {wind_tunnel_result.output_file}")
            print(f"   ⚙️ CFD configuration: {config_result['config_file']}")
            print(f"   ⏱️ Total time: {total_time:.2f}s")
            print(f"   📊 Files generated: {len(output_files)}")
            
            return CompleteWindTunnelResponse(
                success=True,
                message="Complete wind tunnel processing successful",
                wind_tunnel_file=wind_tunnel_result.output_file,
                vtk_file=wind_tunnel_result.vtk_file,
                mesh_stats=wind_tunnel_result.mesh_stats,
                config_file=config_result['config_file'],
                simulation_directory=config_result['simulation_directory'],
                total_time=total_time,
                wind_tunnel_time=wind_tunnel_time,
                config_generation_time=config_time,
                output_files=output_files
            )
            
        except Exception as e:
            error_msg = f"Complete wind tunnel processing failed: {str(e)}"
            self.logger.error(error_msg)
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(error_details)
            
            return CompleteWindTunnelResponse(
                success=False,
                message=error_msg,
                error_details=error_details,
                total_time=time.time() - start_time
            )
    
    def _generate_wind_tunnel_mesh(self, request: CompleteWindTunnelRequest, output_dir: Path) -> WindTunnelResponse:
        """Generate wind tunnel mesh using packaged endpoint"""
        
        # Resolve mesh file path - handle both absolute and relative paths
        mesh_file = request.object_mesh_file
        if not os.path.isabs(mesh_file):
            # If relative path, check if it exists as-is first
            if not os.path.exists(mesh_file):
                # Try to resolve relative to the complete endpoint directory
                mesh_file = os.path.abspath(mesh_file)
        
        print(f"   📁 Resolved mesh file: {mesh_file}")
        print(f"   📋 Mesh exists: {os.path.exists(mesh_file)}")
        
        # Create wind tunnel request
        wind_tunnel_request = WindTunnelRequest(
            object_mesh_file=mesh_file,
            tunnel_type=request.tunnel_type,
            flow_direction=request.flow_direction,
            output_file=str(output_dir / "wind_tunnel.su2"),
            output_directory=str(output_dir),
            mesh_quality=request.mesh_quality,
            domain_scale_factor=request.domain_scale_factor,
            generate_vtk=request.generate_vtk,
            flow_velocity=request.mach_number * 343.0  # Convert Mach to m/s approximately
        )
        
        # Generate wind tunnel
        result = self.wind_tunnel_generator.generate_wind_tunnel(wind_tunnel_request)
        
        print(f"   ✅ Wind tunnel mesh: {result.output_file}")
        if result.vtk_file:
            print(f"   🎨 VTK visualization: {result.vtk_file}")
        if result.mesh_stats:
            print(f"   📊 Nodes: {result.mesh_stats['total_nodes']:,}")
            print(f"   🔷 Elements: {result.mesh_stats['total_elements']:,}")
        
        return result
    
    def _generate_cfd_configuration(self, request: CompleteWindTunnelRequest, 
                                  wind_tunnel_result: WindTunnelResponse, 
                                  output_dir: Path) -> Dict[str, Any]:
        """Generate simple SU2 configuration file using pre-generated AI parameters"""
        
        try:
            # Create simulation name if not provided
            sim_name = request.simulation_name or f"simulation_{int(time.time())}"
            sim_dir = output_dir / sim_name
            sim_dir.mkdir(exist_ok=True)
            
            # Use the parameters directly from the JSON (already processed by AI)
            print(f"   ⚙️ Using AI-generated parameters from JSON")
            print(f"   🎯 Flow type: {request.flow_type}")
            print(f"   🎯 Mach: {request.mach_number}, AOA: {request.angle_of_attack}°")
            print(f"   🎯 Iterations: {request.max_iterations}")
            
            # Generate simple SU2 configuration file directly
            config_file = sim_dir / "config.cfg"
            mesh_filename = os.path.basename(wind_tunnel_result.output_file)
            
            # Create basic SU2 config content
            config_content = f"""% SU2 Configuration - Generated by VectraSim AI
% Based on AI-extracted parameters from user prompt
% Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

% Problem definition
SOLVER= {request.flow_type}
MATH_PROBLEM= DIRECT
RESTART_SOL= NO

% Flow conditions
MACH_NUMBER= {request.mach_number}
AOA= {request.angle_of_attack}
SIDESLIP_ANGLE= {request.sideslip_angle}
FREESTREAM_PRESSURE= {request.freestream_pressure}
FREESTREAM_TEMPERATURE= {request.freestream_temperature}

% Reynolds number (for viscous flows)
REYNOLDS_NUMBER= {request.reynolds_number}
REYNOLDS_LENGTH= 1.0

% Turbulence (if RANS)
FREESTREAM_TURBULENCEINTENSITY= {request.turbulence_intensity}
FREESTREAM_TURB2LAMVISCRATIO= {request.viscosity_ratio}

% Numerical methods
NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
CONV_NUM_METHOD_FLOW= {request.convective_scheme}
MUSCL_FLOW= YES
TIME_DISCRE_FLOW= EULER_IMPLICIT

% CFL number
CFL_NUMBER= {request.cfl_number}

% Convergence criteria
CONV_RESIDUAL_MINVAL= {request.convergence_residual}
CONV_STARTITER= 10
CONV_CAUCHY_ELEMS= 100
CONV_CAUCHY_EPS= 1E-6

% Iteration limit
ITER= {request.max_iterations}

% Mesh and I/O
MESH_FILENAME= {mesh_filename}
MESH_FORMAT= SU2

% Output
OUTPUT_FORMAT= TECPLOT, PARAVIEW
VOLUME_OUTPUT= COORDINATES, SOLUTION, PRIMITIVE
OUTPUT_FILES= RESTART, PARAVIEW, SURFACE_PARAVIEW
CONV_FILENAME= history
RESTART_FILENAME= restart_flow.dat
VOLUME_FILENAME= flow
SURFACE_FILENAME= surface_flow

% Boundary conditions (using standard wind tunnel markers)
MARKER_FAR= ( inlet )
MARKER_EULER= ( slip_wall, object_wall )
MARKER_OUTLET= ( outlet, {request.freestream_pressure} )
MARKER_MONITORING= ( object_wall )

% Reference values
REF_ORIGIN_MOMENT_X = 0.0
REF_ORIGIN_MOMENT_Y = 0.0
REF_ORIGIN_MOMENT_Z = 0.0
REF_LENGTH= 1.0
REF_AREA= 1.0
"""
            
            # Write configuration file
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            # Also create a copy in the main output directory for easier access
            main_config_file = output_dir / "config.cfg"
            with open(main_config_file, 'w') as f:
                f.write(config_content)
            
            # Copy wind tunnel mesh to simulation directory
            mesh_dest = sim_dir / mesh_filename
            shutil.copy2(wind_tunnel_result.output_file, mesh_dest)
            
            print(f"   ✅ SU2 configuration: {config_file}")
            print(f"   📋 Main config file: {main_config_file}")
            print(f"   📁 Simulation directory: {sim_dir}")
            print(f"   🌪️ Mesh file copied: {mesh_dest}")
            
            return {
                'success': True,
                'message': 'SU2 configuration generated successfully',
                'config_file': str(main_config_file),  # Return the main config file path
                'simulation_directory': str(sim_dir),
                'mesh_file': str(mesh_dest),
                'sim_config_file': str(config_file)  # Also include the simulation directory config
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'SU2 configuration failed: {str(e)}',
                'error': str(e)
            }
    
    def _organize_outputs(self, request: CompleteWindTunnelRequest,
                         wind_tunnel_result: WindTunnelResponse,
                         config_result: Dict[str, Any],
                         output_dir: Path) -> List[str]:
        """Organize all output files"""
        
        output_files = []
        
        # Create summary file
        summary = {
            'request_parameters': asdict(request),
            'wind_tunnel_results': {
                'success': wind_tunnel_result.success,
                'output_file': wind_tunnel_result.output_file,
                'vtk_file': wind_tunnel_result.vtk_file,
                'mesh_stats': wind_tunnel_result.mesh_stats,
                'generation_time': wind_tunnel_result.generation_time
            },
            'cfd_configuration': {
                'success': config_result['success'],
                'config_file': config_result.get('config_file'),
                'simulation_directory': config_result.get('simulation_directory'),
                'mesh_file': config_result.get('mesh_file')
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'output_directory': str(output_dir)
        }
        
        summary_file = output_dir / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        output_files.append(str(summary_file))
        
        # List all generated files
        if wind_tunnel_result.output_file:
            output_files.append(wind_tunnel_result.output_file)
        if wind_tunnel_result.vtk_file:
            output_files.append(wind_tunnel_result.vtk_file)
        if config_result.get('config_file'):
            output_files.append(config_result['config_file'])
        if config_result.get('mesh_file'):
            output_files.append(config_result['mesh_file'])
        
        print(f"   📄 Summary file: {summary_file}")
        print(f"   📊 Total output files: {len(output_files)}")
        
        return output_files

def process_json_request(json_file: str) -> CompleteWindTunnelResponse:
    """
    Process a complete wind tunnel request from JSON file
    
    Args:
        json_file: Path to JSON parameter file
        
    Returns:
        CompleteWindTunnelResponse with results
    """
    
    try:
        # Load JSON parameters
        with open(json_file, 'r') as f:
            params = json.load(f)
        
        # Create request object
        request = CompleteWindTunnelRequest(**params)
        
        # Process request
        endpoint = CompleteWindTunnelEndpoint()
        return endpoint.process_complete_request(request)
        
    except Exception as e:
        return CompleteWindTunnelResponse(
            success=False,
            message=f"Failed to process JSON request: {str(e)}",
            error_details=str(e)
        )

def main():
    """Main entry point for command line usage"""
    
    if len(sys.argv) != 2:
        print("Usage: python complete_wind_tunnel_endpoint.py <json_parameter_file>")
        print("\nExample JSON structure:")
        print(json.dumps({
            "object_mesh_file": "propeller.su2",
            "tunnel_type": "standard",
            "flow_direction": "+X",
            "mesh_quality": "medium",
            "flow_type": "EULER",
            "mach_number": 0.3,
            "angle_of_attack": 5.0,
            "max_iterations": 100,
            "output_directory": "my_simulation_output",
            "simulation_name": "propeller_analysis"
        }, indent=2))
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not os.path.exists(json_file):
        print(f"❌ JSON parameter file not found: {json_file}")
        sys.exit(1)
    
    print(f"🚀 Processing complete wind tunnel request from: {json_file}")
    
    result = process_json_request(json_file)
    
    if result.success:
        print(f"\n✅ Processing completed successfully!")
        print(f"📁 Output directory: {result.output_files[0] if result.output_files else 'N/A'}")
        sys.exit(0)
    else:
        print(f"\n❌ Processing failed: {result.message}")
        if result.error_details:
            print(f"Error details: {result.error_details}")
        sys.exit(1)

if __name__ == "__main__":
    main() 