#!/usr/bin/env python3
"""
VectraSim Internal Flow Endpoint - Main API
Specialized for supersonic internal flows and rocket propulsion test cases

This endpoint handles:
- Supersonic nozzle flows
- Combustion chamber simulations  
- Rocket propulsion test cases
- Internal duct flows with complex boundary conditions
- Intelligent inlet/outlet boundary condition generation
- Forcing-based virtual boundaries with masks

Integration with intelligent_BC_final for automatic boundary condition detection
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Add intelligent_BC_final to path for boundary condition generation
sys.path.append(str(Path(__file__).parent.parent / "intelligent_BC_final"))
from intelligent_boundary_endpoint import IntelligentBoundaryEndpoint

from internal_flow_orchestrator import InternalFlowOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InternalFlowResponse:
    """Response from internal flow simulation generation"""
    success: bool
    simulation_directory: str
    case_file: str
    numerical_file: str
    run_script: str
    simulation_summary: Dict[str, Any]
    boundary_conditions: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

def create_internal_flow_simulation(
    user_prompt: str,
    mesh_file: str,
    output_directory: str = None,
    flow_type: str = "supersonic_nozzle",
    mach_number: float = None,
    pressure_ratio: float = None,
    temperature_inlet: float = None,
    geometry_type: str = "converging_diverging",
    advanced_config: Dict[str, Any] = None
) -> InternalFlowResponse:
    """
    Create internal flow simulation with intelligent boundary conditions
    
    Args:
        user_prompt: User description of the simulation
        mesh_file: Path to mesh file for boundary condition analysis
        output_directory: Where to save simulation files
        flow_type: Type of internal flow (supersonic_nozzle, rocket_engine, etc.)
        mach_number: Target Mach number
        pressure_ratio: Pressure ratio across the domain
        temperature_inlet: Inlet temperature
        geometry_type: Geometry type (converging_diverging, bell_nozzle, etc.)
        advanced_config: Advanced configuration options
        
    Returns:
        InternalFlowResponse with simulation files and boundary condition data
    """
    
    try:
        print("🚀 VectraSim Internal Flow Endpoint - Enhanced with Intelligent Boundary Conditions")
        print(f"🎯 Request: {user_prompt}")
        print(f"🌪️ Flow Type: {flow_type}")
        print(f"📐 Mesh File: {mesh_file}")
        print("=" * 80)
        
        # Validate inputs
        if not os.path.exists(mesh_file):
            raise FileNotFoundError(f"Mesh file not found: {mesh_file}")
        
        # Validate API key
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("❌ GEMINI_API_KEY environment variable not set. Required for AI agents.")
        
        # Set default output directory
        if output_directory is None:
            output_directory = Path.cwd() / "internal_flow_simulations"
        
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create bc_processed subdirectory for boundary condition data
        bc_processed_dir = output_dir / "bc_processed"
        bc_processed_dir.mkdir(exist_ok=True)
        
        # Find next available subdirectory
        subdirectory_index = 1
        while (bc_processed_dir / f"subdirectory_{subdirectory_index}").exists():
            subdirectory_index += 1
        
        bc_storage_dir = bc_processed_dir / f"subdirectory_{subdirectory_index}"
        bc_storage_dir.mkdir(exist_ok=True)
        
        print("\n🧠 STEP 1: INTELLIGENT BOUNDARY CONDITION GENERATION")
        print("=" * 60)
        print(f"📁 BC Storage: {bc_storage_dir}")
        
        # Initialize intelligent boundary condition endpoint
        bc_endpoint = IntelligentBoundaryEndpoint(verbose=True)
        
        # Generate boundary conditions and masks
        bc_result = bc_endpoint.process_mesh(
            mesh_file=mesh_file,
            output_dir=str(bc_storage_dir),
            simulation_name="internal_flow_bc"
        )
        
        print(f"✅ Boundary conditions generated:")
        print(f"   🔴 Inlet mask: {bc_result['inlet_points']:,} points")
        print(f"   🟢 Outlet mask: {bc_result['outlet_points']:,} points")
        print(f"   📦 VTK visualization: {bc_result.get('vtk_visualization', 'N/A')}")
        
        # Create advanced configuration
        if advanced_config is None:
            advanced_config = {}
            
        # Add flow parameters and boundary condition data to config
        flow_config = {
            "flow_type": flow_type,
            "mach_number": mach_number,
            "pressure_ratio": pressure_ratio,
            "temperature_inlet": temperature_inlet,
            "geometry_type": geometry_type,
            "boundary_conditions": bc_result,
            "bc_storage_dir": str(bc_storage_dir),
            **advanced_config
        }
        
        print("\n🤖 STEP 2: AI-DRIVEN SIMULATION GENERATION")
        print("=" * 60)
        
        # Initialize orchestrator with boundary condition data
        orchestrator = InternalFlowOrchestrator(
            gemini_api_key=gemini_api_key,
            flow_config=flow_config
        )
        
        print("🤖 Initializing Enhanced Internal Flow AI Agents...")
        print("   • Supersonic Case Setup Expert (with intelligent BCs)")
        print("   • Internal Flow Numerical Expert (mask-aware)")
        print("   • Adaptive Execution Agent (forcing-enhanced)")
        
        # Generate simulation with boundary condition integration
        response = orchestrator.create_internal_flow_simulation(
            user_prompt=user_prompt,
            output_directory=str(output_dir)
        )
        
        # Add boundary condition data to response
        response.boundary_conditions = bc_result
        
        print("\n🎉 Enhanced Internal Flow Simulation Generated Successfully!")
        print(f"📁 Simulation Directory: {response.simulation_directory}")
        print(f"🧠 Boundary Conditions: {bc_storage_dir}")
        print(f"🔴 Inlet Points: {bc_result['inlet_points']:,}")
        print(f"🟢 Outlet Points: {bc_result['outlet_points']:,}")
        
        return response
        
    except Exception as e:
        logger.error(f"Internal flow simulation creation failed: {e}")
        return InternalFlowResponse(
            success=False,
            simulation_directory="",
            case_file="",
            numerical_file="",
            run_script="",
            simulation_summary={},
            boundary_conditions=None,
            error_message=str(e)
        )

def main():
    """Example usage of the enhanced internal flow endpoint"""
    
    # Example configuration
    mesh_file = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\mesh\Rocket Engine.msh"
    
    response = create_internal_flow_simulation(
        user_prompt="Create a supersonic rocket nozzle simulation with intelligent boundary conditions",
        mesh_file=mesh_file,
        flow_type="rocket_engine",
        mach_number=3.0,
        pressure_ratio=50.0,
        temperature_inlet=3580.0
    )
    
    if response.success:
        print("\n🎉 Success!")
        print(f"Simulation: {response.simulation_directory}")
        print(f"Case file: {response.case_file}")
        print(f"Run script: {response.run_script}")
        if response.boundary_conditions:
            print(f"Inlet points: {response.boundary_conditions['inlet_points']:,}")
            print(f"Outlet points: {response.boundary_conditions['outlet_points']:,}")
    else:
        print(f"\n❌ Failed: {response.error_message}")

if __name__ == "__main__":
    main() 