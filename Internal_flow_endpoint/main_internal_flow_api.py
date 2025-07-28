#!/usr/bin/env python3
"""
VectraSim Internal Flow Endpoint - Main API
Specialized for supersonic internal flows and rocket propulsion test cases

This endpoint handles:
- Supersonic nozzle flows
- Combustion chamber simulations  
- Rocket propulsion test cases
- Internal duct flows with complex boundary conditions
- Inlet/outlet management for high-speed flows

Based on JAX-Fluids capabilities with SIMPLE_INFLOW/SIMPLE_OUTFLOW boundary conditions
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

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
    error_message: Optional[str] = None

def create_internal_flow_simulation(
    user_prompt: str,
    output_directory: str = None,
    flow_type: str = "supersonic_nozzle",
    mach_number: float = None,
    pressure_ratio: float = None,
    temperature_inlet: float = None,
    geometry_type: str = "converging_diverging",
    advanced_config: Dict[str, Any] = None
) -> InternalFlowResponse:
    """
    Create a JAX-Fluids internal flow simulation from user requirements
    
    Args:
        user_prompt: Description of the internal flow simulation needed
        output_directory: Where to save simulation files
        flow_type: Type of internal flow ('supersonic_nozzle', 'combustion_chamber', 
                  'rocket_engine', 'shock_tube', 'duct_flow')
        mach_number: Inlet or reference Mach number
        pressure_ratio: Pressure ratio across domain (inlet/outlet)
        temperature_inlet: Inlet temperature [K]
        geometry_type: Geometry type ('converging_diverging', 'straight_duct', 
                      'combustor', 'custom')
        advanced_config: Advanced configuration options
        
    Returns:
        InternalFlowResponse with simulation files and metadata
    """
    
    try:
        print("🚀 VectraSim Internal Flow Endpoint - Supersonic Specialist")
        print(f"🎯 Request: {user_prompt}")
        print(f"🌪️ Flow Type: {flow_type}")
        print("=" * 80)
        
        # Validate API key
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("❌ GEMINI_API_KEY environment variable not set. Required for AI agents.")
        
        # Set default output directory
        if output_directory is None:
            output_directory = Path.cwd() / "internal_flow_simulations"
        
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create advanced configuration
        if advanced_config is None:
            advanced_config = {}
            
        # Add flow parameters to config
        flow_config = {
            "flow_type": flow_type,
            "mach_number": mach_number,
            "pressure_ratio": pressure_ratio,
            "temperature_inlet": temperature_inlet,
            "geometry_type": geometry_type,
            **advanced_config
        }
        
        # Initialize orchestrator
        orchestrator = InternalFlowOrchestrator(
            gemini_api_key=gemini_api_key,
            flow_config=flow_config
        )
        
        print("🤖 Initializing Internal Flow AI Agents...")
        print("   • Supersonic Case Setup Expert")
        print("   • Internal Flow Numerical Expert") 
        print("   • Adaptive Execution Agent")
        
        # Generate simulation
        response = orchestrator.create_internal_flow_simulation(
            user_prompt=user_prompt,
            output_directory=str(output_dir)
        )
        
        print("\n🎉 Internal Flow Simulation Generated Successfully!")
        print(f"📁 Simulation Directory: {response.simulation_directory}")
        print(f"📄 Case File: {response.case_file}")
        print(f"🔢 Numerical File: {response.numerical_file}")
        print(f"🚀 Run Script: {response.run_script}")
        
        return response
        
    except Exception as e:
        error_msg = f"Internal flow simulation generation failed: {str(e)}"
        logger.error(error_msg)
        
        return InternalFlowResponse(
            success=False,
            simulation_directory="",
            case_file="",
            numerical_file="",
            run_script="",
            simulation_summary={},
            error_message=error_msg
        )

def create_rocket_propulsion_test(
    nozzle_type: str = "bell_nozzle",
    chamber_pressure: float = 3.0e6,  # 30 bar
    chamber_temperature: float = 3000.0,  # 3000 K
    ambient_pressure: float = 101325.0,  # 1 atm
    expansion_ratio: float = 16.0,
    fuel_type: str = "hot_gas",
    output_directory: str = None
) -> InternalFlowResponse:
    """
    Create a specialized rocket propulsion test case
    
    Args:
        nozzle_type: Type of nozzle ('bell_nozzle', 'conical', 'dual_bell')
        chamber_pressure: Combustion chamber pressure [Pa]
        chamber_temperature: Combustion chamber temperature [K]
        ambient_pressure: Ambient pressure [Pa]
        expansion_ratio: Nozzle area expansion ratio
        fuel_type: Fuel type ('hot_gas', 'hydrogen', 'methane', 'rp1')
        output_directory: Where to save simulation files
        
    Returns:
        InternalFlowResponse with rocket propulsion simulation
    """
    
    # Calculate pressure ratio
    pressure_ratio = chamber_pressure / ambient_pressure
    
    # Estimate Mach number based on expansion ratio (isentropic relation)
    gamma = 1.3  # Typical for hot combustion gases
    mach_exit = ((2/(gamma-1)) * ((pressure_ratio)**((gamma-1)/gamma) - 1))**0.5
    
    user_prompt = f"""
    Design a rocket propulsion test case with the following specifications:
    - Nozzle type: {nozzle_type}
    - Chamber pressure: {chamber_pressure/1e6:.1f} MPa
    - Chamber temperature: {chamber_temperature:.0f} K
    - Ambient pressure: {ambient_pressure/1000:.1f} kPa
    - Expansion ratio: {expansion_ratio}
    - Estimated exit Mach: {mach_exit:.1f}
    - Fuel type: {fuel_type}
    
    This is a high-fidelity supersonic internal flow simulation for rocket engine performance analysis.
    Include proper inlet conditions, nozzle geometry, and outlet boundary conditions.
    """
    
    return create_internal_flow_simulation(
        user_prompt=user_prompt,
        output_directory=output_directory,
        flow_type="rocket_engine",
        mach_number=mach_exit,
        pressure_ratio=pressure_ratio,
        temperature_inlet=chamber_temperature,
        geometry_type="converging_diverging",
        advanced_config={
            "nozzle_type": nozzle_type,
            "chamber_pressure": chamber_pressure,
            "ambient_pressure": ambient_pressure,
            "expansion_ratio": expansion_ratio,
            "fuel_type": fuel_type,
            "gamma": gamma
        }
    )

if __name__ == "__main__":
    # Example usage
    print("🚀 VectraSim Internal Flow Endpoint")
    print("Specialized for rocket propulsion and supersonic internal flows")
    
    # Test with rocket propulsion case
    response = create_rocket_propulsion_test(
        nozzle_type="bell_nozzle",
        chamber_pressure=2.0e6,  # 20 bar
        chamber_temperature=2800.0,  # 2800 K
        expansion_ratio=12.0,
        output_directory="./test_rocket_simulation"
    )
    
    if response.success:
        print(f"✅ Rocket simulation created: {response.simulation_directory}")
    else:
        print(f"❌ Failed: {response.error_message}") 