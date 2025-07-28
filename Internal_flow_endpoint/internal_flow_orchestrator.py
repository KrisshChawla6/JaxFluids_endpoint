#!/usr/bin/env python3
"""
VectraSim Internal Flow Orchestrator
Coordinates specialized AI agents for supersonic internal flows and rocket propulsion

This orchestrator manages:
- Supersonic Case Setup Expert (inlet/outlet conditions, nozzle geometry)
- Internal Flow Numerical Expert (shock-capturing schemes, supersonic numerics)  
- Adaptive Execution Agent (reusing from external flow with internal flow adaptations)
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from supersonic_internal_flow.case_setup_expert import SupersonicCaseSetupExpert
from supersonic_internal_flow.numerical_setup_expert import InternalFlowNumericalExpert
from supersonic_internal_flow.execution_agent import InternalFlowExecutionAgent
from adaptive_jaxfluids_agent import create_adaptive_jaxfluids_script

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

class InternalFlowOrchestrator:
    """
    Master orchestrator for internal flow simulations
    Coordinates specialized agents for rocket propulsion and supersonic flows
    """
    
    def __init__(self, gemini_api_key: str, flow_config: Dict[str, Any]):
        """
        Initialize the internal flow orchestrator
        
        Args:
            gemini_api_key: Gemini API key for AI agents
            flow_config: Flow configuration parameters
        """
        self.gemini_api_key = gemini_api_key
        self.flow_config = flow_config
        
        # Initialize specialized agents
        self.case_setup_expert = SupersonicCaseSetupExpert(gemini_api_key)
        self.numerical_expert = InternalFlowNumericalExpert(gemini_api_key)
        self.execution_agent = InternalFlowExecutionAgent(gemini_api_key)
        
        logger.info("🚀 Internal Flow Orchestrator initialized")
        logger.info(f"🎯 Flow Type: {flow_config.get('flow_type', 'unknown')}")
        
    def create_internal_flow_simulation(
        self,
        user_prompt: str,
        output_directory: str
    ) -> InternalFlowResponse:
        """
        Create a complete internal flow simulation using specialized agents
        
        Args:
            user_prompt: User description of the simulation needed
            output_directory: Directory to save simulation files
            
        Returns:
            InternalFlowResponse with all generated files and metadata
        """
        
        try:
            print("🚀 VectraSim Internal Flow Orchestrator")
            print("=" * 70)
            print(f"🎯 User Request: {user_prompt}")
            print(f"📁 Output Directory: {output_directory}")
            print(f"🌪️ Flow Configuration: {self.flow_config}")
            
            # Create unique simulation directory
            timestamp = str(int(time.time()))
            simulation_name = f"jaxfluids_internal_flow_{timestamp}"
            simulation_directory = Path(output_directory) / simulation_name
            simulation_directory.mkdir(parents=True, exist_ok=True)
            
            print(f"📂 Created simulation directory: {simulation_directory}")
            
            # Enhanced context for internal flows
            internal_flow_context = {
                "user_prompt": user_prompt,
                "simulation_name": simulation_name,
                "simulation_directory": str(simulation_directory),
                "flow_type": self.flow_config.get("flow_type", "supersonic_nozzle"),
                "mach_number": self.flow_config.get("mach_number"),
                "pressure_ratio": self.flow_config.get("pressure_ratio"),
                "temperature_inlet": self.flow_config.get("temperature_inlet"),
                "geometry_type": self.flow_config.get("geometry_type", "converging_diverging"),
                "advanced_physics": {
                    "compressible": True,
                    "supersonic": True,
                    "viscous": True,
                    "heat_transfer": True,
                    "shock_capturing": True
                }
            }
            
            # Add rocket-specific parameters if applicable
            if self.flow_config.get("flow_type") == "rocket_engine":
                internal_flow_context.update({
                    "nozzle_type": self.flow_config.get("nozzle_type"),
                    "chamber_pressure": self.flow_config.get("chamber_pressure"),
                    "ambient_pressure": self.flow_config.get("ambient_pressure"),
                    "expansion_ratio": self.flow_config.get("expansion_ratio"),
                    "fuel_type": self.flow_config.get("fuel_type"),
                    "gamma": self.flow_config.get("gamma", 1.3)
                })
            
            print("\n🤖 Phase 1: Supersonic Case Setup Expert")
            print("   Generating internal flow boundary conditions and geometry...")
            
            case_setup = self.case_setup_expert.generate_case_setup(internal_flow_context)
            case_file = simulation_directory / f"{simulation_name}.json"
            
            with open(case_file, 'w', encoding='utf-8') as f:
                json.dump(case_setup, f, indent=2)
            
            print(f"   ✅ Case setup saved: {case_file.name}")
            
            print("\n🔢 Phase 2: Internal Flow Numerical Expert")
            print("   Configuring shock-capturing schemes and supersonic numerics...")
            
            numerical_setup = self.numerical_expert.generate_numerical_setup(internal_flow_context)
            numerical_file = simulation_directory / "numerical_setup.json"
            
            with open(numerical_file, 'w', encoding='utf-8') as f:
                json.dump(numerical_setup, f, indent=2)
                
            print(f"   ✅ Numerical setup saved: {numerical_file.name}")
            
            print("\n🚀 Phase 3: Adaptive Execution Agent")
            print("   Generating intelligent JAX-Fluids run script...")
            
            # Create simulation intent for adaptive agent
            simulation_intent = self._create_simulation_intent(internal_flow_context)
            
            # Generate adaptive run script
            run_script_path = create_adaptive_jaxfluids_script(
                case_setup_path=str(case_file),
                numerical_setup_path=str(numerical_file),
                output_directory=str(simulation_directory),
                simulation_intent=simulation_intent,
                plotting_mode="standard",
                gemini_api_key=self.gemini_api_key
            )
            
            print(f"   ✅ Adaptive run script generated: {Path(run_script_path).name}")
            
            # Create simulation summary
            simulation_summary = {
                "simulation_name": simulation_name,
                "timestamp": timestamp,
                "user_prompt": user_prompt,
                "flow_configuration": self.flow_config,
                "physics_summary": {
                    "flow_regime": "supersonic_internal",
                    "compressible": True,
                    "viscous": True,
                    "heat_transfer": True,
                    "shock_capturing": True
                },
                "files_generated": {
                    "case_setup": case_file.name,
                    "numerical_setup": numerical_file.name,
                    "run_script": Path(run_script_path).name
                }
            }
            
            # Save simulation summary
            summary_file = simulation_directory / "simulation_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(simulation_summary, f, indent=2)
            
            print("\n🎉 Internal Flow Simulation Generation Complete!")
            print("=" * 70)
            print(f"📂 Simulation Directory: {simulation_directory}")
            print(f"📄 Case File: {case_file.name}")
            print(f"🔢 Numerical File: {numerical_file.name}")
            print(f"🚀 Run Script: {Path(run_script_path).name}")
            print(f"📊 Summary: {summary_file.name}")
            
            return InternalFlowResponse(
                success=True,
                simulation_directory=str(simulation_directory),
                case_file=str(case_file),
                numerical_file=str(numerical_file),
                run_script=run_script_path,
                simulation_summary=simulation_summary
            )
            
        except Exception as e:
            error_message = f"Internal flow orchestration failed: {str(e)}"
            logger.error(error_message)
            
            return InternalFlowResponse(
                success=False,
                simulation_directory="",
                case_file="",
                numerical_file="",
                run_script="",
                simulation_summary={},
                error_message=error_message
            )
    
    def _create_simulation_intent(self, context: Dict[str, Any]) -> str:
        """
        Create intelligent simulation intent for the adaptive agent
        
        Args:
            context: Internal flow context
            
        Returns:
            Simulation intent string for adaptive agent
        """
        
        flow_type = context.get("flow_type", "supersonic_nozzle")
        mach_number = context.get("mach_number")
        pressure_ratio = context.get("pressure_ratio")
        
        if flow_type == "rocket_engine":
            nozzle_type = context.get("nozzle_type", "bell_nozzle")
            chamber_pressure = context.get("chamber_pressure", 0)
            expansion_ratio = context.get("expansion_ratio", 1)
            
            intent = f"""Rocket propulsion {nozzle_type} simulation with supersonic internal flow.
            Chamber pressure: {chamber_pressure/1e6:.1f} MPa, expansion ratio: {expansion_ratio:.1f}.
            High-temperature combustion gases with shock waves and viscous effects.
            Critical for rocket engine performance analysis and nozzle optimization."""
            
        elif flow_type == "supersonic_nozzle":
            intent = f"""Supersonic nozzle flow simulation with Mach {mach_number:.1f} conditions.
            Pressure ratio: {pressure_ratio:.1f}, includes shock waves and expansion fans.
            Internal flow with converging-diverging geometry for compressible flow analysis."""
            
        elif flow_type == "combustion_chamber":
            intent = f"""Combustion chamber internal flow with high-temperature gas dynamics.
            Supersonic conditions with complex thermochemistry and heat transfer.
            Internal geometry with inlet/outlet boundary management."""
            
        elif flow_type == "shock_tube":
            intent = f"""Shock tube internal flow with supersonic wave propagation.
            High-speed compressible flow with shock-boundary layer interactions.
            1D/2D internal geometry for fundamental shock physics studies."""
            
        else:
            intent = f"""General supersonic internal flow simulation.
            Compressible flow with shock capturing and viscous effects.
            Internal duct geometry with inlet/outlet boundary conditions."""
        
        return intent
    
    def get_flow_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of the internal flow endpoint
        
        Returns:
            Dictionary describing supported flow types and features
        """
        
        return {
            "supported_flow_types": [
                "supersonic_nozzle",
                "rocket_engine", 
                "combustion_chamber",
                "shock_tube",
                "duct_flow"
            ],
            "geometry_types": [
                "converging_diverging",
                "straight_duct",
                "combustor",
                "custom"
            ],
            "boundary_conditions": [
                "SIMPLE_INFLOW",
                "SIMPLE_OUTFLOW", 
                "DIRICHLET",
                "WALL",
                "SYMMETRY"
            ],
            "physics_capabilities": {
                "compressible_flow": True,
                "supersonic_flow": True,
                "shock_capturing": True,
                "viscous_effects": True,
                "heat_transfer": True,
                "high_temperature": True,
                "multi_species": False  # Future capability
            },
            "numerical_methods": {
                "reconstruction": ["WENO5-Z", "WENO7", "WENO-CU6"],
                "riemann_solvers": ["HLLC", "ROE", "AUSM"],
                "time_integration": ["RK3", "RK2", "EULER"]
            }
        } 