#!/usr/bin/env python3
"""
VectraSim Internal Flow Execution Agent
Coordinates with the adaptive JAX-Fluids agent for supersonic internal flows

This agent handles:
- Supersonic internal flow execution coordination
- Integration with adaptive JAX-Fluids agent
- Rocket propulsion simulation management
- Error handling for high-speed flows
- Production-ready script generation
"""

import json
import logging
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

class InternalFlowExecutionAgent:
    """
    AI agent for coordinating internal flow simulation execution
    Works with adaptive JAX-Fluids agent for supersonic flows
    """
    
    def __init__(self, gemini_api_key: str):
        """Initialize the internal flow execution agent"""
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=gemini_api_key,
            temperature=0.1,
            max_tokens=4096
        )
        logger.info("⚡ Internal Flow Execution Agent initialized")
    
    def coordinate_execution(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate execution for internal flow simulations
        
        Args:
            context: Internal flow context with execution parameters
            
        Returns:
            Execution coordination metadata
        """
        
        print("⚡ Internal Flow Execution Agent - Coordinating Execution")
        
        # Extract context parameters
        flow_type = context.get("flow_type", "supersonic_nozzle")
        simulation_name = context.get("simulation_name", "internal_flow")
        
        execution_metadata = {
            "execution_strategy": self._determine_execution_strategy(context),
            "monitoring_parameters": self._get_monitoring_parameters(context),
            "optimization_flags": self._get_optimization_flags(context),
            "error_handling": self._get_error_handling_config(context)
        }
        
        print(f"   ✅ Coordinated execution for {flow_type}")
        print(f"   🎯 Strategy: {execution_metadata['execution_strategy']}")
        print(f"   📊 Monitoring: {len(execution_metadata['monitoring_parameters'])} parameters")
        
        return execution_metadata
    
    def _determine_execution_strategy(self, context: Dict[str, Any]) -> str:
        """Determine optimal execution strategy for the flow type"""
        
        flow_type = context.get("flow_type", "supersonic_nozzle")
        mach_number = context.get("mach_number", 1.0)
        
        if flow_type == "rocket_engine":
            if mach_number > 3.0:
                return "high_mach_rocket_execution"
            else:
                return "standard_rocket_execution"
        elif flow_type == "supersonic_nozzle":
            return "nozzle_flow_execution"
        elif flow_type == "shock_tube":
            return "shock_tube_execution"
        else:
            return "general_supersonic_execution"
    
    def _get_monitoring_parameters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get monitoring parameters for supersonic internal flows"""
        
        flow_type = context.get("flow_type", "supersonic_nozzle")
        
        base_parameters = {
            "mach_number": "Monitor local Mach number distribution",
            "pressure_ratio": "Track pressure ratio across domain",
            "temperature": "Monitor temperature for high-temp gases",
            "shock_detection": "Detect shock waves and discontinuities"
        }
        
        if flow_type == "rocket_engine":
            base_parameters.update({
                "thrust_coefficient": "Calculate thrust performance",
                "nozzle_efficiency": "Monitor nozzle expansion efficiency",
                "chamber_conditions": "Track combustion chamber state"
            })
        
        return base_parameters
    
    def _get_optimization_flags(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimization flags for supersonic performance"""
        
        mach_number = context.get("mach_number", 1.0)
        
        return {
            "use_shock_capturing": True,
            "enable_positivity_limiter": True,
            "conservative_time_stepping": mach_number > 2.0,
            "high_order_reconstruction": True,
            "adaptive_cfl": mach_number > 1.5,
            "viscous_heating": True,
            "compressible_corrections": True
        }
    
    def _get_error_handling_config(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get error handling configuration for robust execution"""
        
        return {
            "shock_detection_tolerance": 1e-6,
            "pressure_positivity_threshold": 1e-10,
            "temperature_bounds": [50.0, 5000.0],  # K
            "velocity_bounds": [0.0, 3000.0],      # m/s
            "recovery_strategies": [
                "reduce_time_step",
                "increase_diffusion",
                "apply_filters"
            ],
            "critical_failure_actions": [
                "save_state",
                "generate_diagnostics", 
                "graceful_shutdown"
            ]
        }
    
    def generate_execution_summary(self, context: Dict[str, Any]) -> str:
        """
        Generate execution summary for the adaptive JAX-Fluids agent
        
        Args:
            context: Internal flow context
            
        Returns:
            Execution summary string
        """
        
        flow_type = context.get("flow_type", "supersonic_nozzle")
        mach_number = context.get("mach_number", 1.0)
        chamber_pressure = context.get("chamber_pressure", 0)
        
        if flow_type == "rocket_engine":
            summary = f"""
ROCKET PROPULSION EXECUTION SUMMARY:
- Chamber pressure: {chamber_pressure/1e6:.1f} MPa
- Expected exit Mach: {mach_number:.1f}
- Supersonic nozzle expansion with shock capturing
- High-temperature combustion gas physics
- WENO5-Z + HLLC for robust shock resolution
- Production-ready rocket engine simulation

CRITICAL MONITORING:
- Thrust coefficient and nozzle efficiency
- Shock wave formation and expansion fans
- Temperature distribution in nozzle
- Pressure recovery and losses

EXECUTION STRATEGY: Production rocket propulsion analysis
"""
        elif flow_type == "supersonic_nozzle":
            summary = f"""
SUPERSONIC NOZZLE EXECUTION SUMMARY:
- Mach number: {mach_number:.1f}
- Converging-diverging nozzle geometry
- Shock wave and expansion fan capture
- Viscous boundary layer effects
- Heat transfer in high-speed flow

CRITICAL MONITORING:
- Mach number distribution
- Shock positioning accuracy
- Boundary layer development
- Pressure recovery efficiency

EXECUTION STRATEGY: High-fidelity nozzle flow analysis
"""
        else:
            summary = f"""
SUPERSONIC INTERNAL FLOW EXECUTION SUMMARY:
- Flow type: {flow_type}
- Mach number: {mach_number:.1f}
- Internal geometry with inlet/outlet
- Compressible flow with shock capture
- Viscous and heat transfer effects

EXECUTION STRATEGY: General supersonic internal flow
"""
        
        return summary.strip() 