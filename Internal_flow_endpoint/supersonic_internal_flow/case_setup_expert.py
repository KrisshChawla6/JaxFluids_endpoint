#!/usr/bin/env python3
"""
VectraSim Supersonic Case Setup Expert
Specialized AI agent for internal flow case configurations

This expert handles:
- SIMPLE_INFLOW/SIMPLE_OUTFLOW boundary conditions
- Supersonic nozzle geometries (converging-diverging)
- Rocket propulsion chamber conditions
- High-temperature gas properties
- Internal flow domain setup
"""

import json
import logging
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

class SupersonicCaseSetupExpert:
    """
    AI expert specialized in supersonic internal flow case setup
    Generates JAX-Fluids configurations for rocket propulsion and nozzle flows
    """
    
    def __init__(self, gemini_api_key: str):
        """Initialize the supersonic case setup expert"""
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=gemini_api_key,
            temperature=0.1,
            max_tokens=4096
        )
        logger.info("🌪️ Supersonic Case Setup Expert initialized")
    
    def generate_case_setup(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate JAX-Fluids case setup for supersonic internal flows
        
        Args:
            context: Internal flow context with flow parameters
            
        Returns:
            JAX-Fluids case setup dictionary
        """
        
        print("🌪️ Supersonic Case Setup Expert - Generating Configuration")
        
        # Extract context parameters
        flow_type = context.get("flow_type", "supersonic_nozzle")
        geometry_type = context.get("geometry_type", "converging_diverging")
        simulation_name = context.get("simulation_name", "supersonic_internal_flow")
        
        # Create AI prompt for supersonic internal flow
        prompt = self._create_supersonic_prompt(context)
        
        try:
            print("   🤖 AI generating supersonic internal flow configuration...")
            
            # Invoke AI model
            messages = [HumanMessage(content=prompt)]
            response = self.model.invoke(messages)
            
            # Parse AI response
            case_setup = self._parse_ai_response(response.content, context)
            
            print(f"   ✅ Generated configuration for {flow_type}")
            print(f"   🎯 Geometry: {geometry_type}")
            print(f"   🌪️ Flow regime: supersonic internal")
            
            return case_setup
            
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            print(f"   ⚠️ AI generation failed, using template: {str(e)}")
            return self._create_template_case_setup(context)
    
    def _create_supersonic_prompt(self, context: Dict[str, Any]) -> str:
        """Create specialized AI prompt for supersonic internal flows"""
        
        flow_type = context.get("flow_type", "supersonic_nozzle")
        mach_number = context.get("mach_number", 2.0)
        pressure_ratio = context.get("pressure_ratio", 10.0)
        temperature_inlet = context.get("temperature_inlet", 2800.0)
        simulation_name = context.get("simulation_name", "supersonic_internal")
        
        prompt = f"""
You are a supersonic internal flow expert generating JAX-Fluids case setup for rocket propulsion applications.

SIMULATION CONTEXT:
- Flow Type: {flow_type}
- Mach Number: {mach_number}
- Pressure Ratio: {pressure_ratio}
- Inlet Temperature: {temperature_inlet} K
- Simulation: {simulation_name}

CRITICAL REQUIREMENTS FOR SUPERSONIC INTERNAL FLOWS:
1. Use SIMPLE_INFLOW for inlet (west) - specify density, velocity, temperature
2. Use SIMPLE_OUTFLOW for outlet (east) - specify pressure only
3. Use SYMMETRY for top/bottom boundaries (axisymmetric nozzle)
4. Configure converging-diverging nozzle geometry with proper domain
5. High-temperature gas properties for rocket propulsion
6. Proper nondimensionalization for supersonic conditions

MANDATORY JAX-FLUIDS INTERNAL FLOW STRUCTURE:
{{
  "case_name": "{simulation_name}",
  "general": {{
    "case_name": "{simulation_name}",
    "end_time": 0.01,
    "save_path": "./results_supersonic",
    "save_dt": 0.001
  }},
  "domain": {{
    "x": {{
      "cells": 400,
      "range": [-0.1, 0.3],
      "stretching": {{
        "type": "BOUNDARY_LAYER",
        "parameters": {{
          "lower_bound": -0.1,
          "upper_bound": 0.3,
          "cells": 400
        }}
      }}
    }},
    "y": {{
      "cells": 200, 
      "range": [-0.05, 0.05],
      "stretching": {{
        "type": "BOUNDARY_LAYER",
        "parameters": {{
          "lower_bound": -0.05,
          "upper_bound": 0.05,
          "cells": 200
        }}
      }}
    }},
    "z": {{
      "cells": 1,
      "range": [0.0, 1.0]
    }},
    "decomposition": {{
      "split_x": 1,
      "split_y": 1,
      "split_z": 1
    }}
  }},
  "boundary_conditions": {{
    "west": {{
      "type": "SIMPLE_INFLOW",
      "primitives_callable": {{
        "rho": <calculate from inlet conditions>,
        "u": <calculate supersonic velocity>,
        "v": 0.0,
        "w": 0.0
      }}
    }},
    "east": {{
      "type": "SIMPLE_OUTFLOW", 
      "primitives_callable": {{
        "p": <ambient pressure>
      }}
    }},
    "north": {{"type": "SYMMETRY"}},
    "south": {{"type": "SYMMETRY"}},
    "top": {{"type": "INACTIVE"}},
    "bottom": {{"type": "INACTIVE"}}
  }},
  "initial_condition": {{
    "rho": <initial density>,
    "u": <initial velocity>,
    "v": 0.0,
    "w": 0.0,
    "p": <initial pressure>
  }},
  "material_properties": {{
    "equation_of_state": {{
      "model": "IdealGas",
      "specific_heat_ratio": 1.3,
      "specific_gas_constant": 287.0
    }},
    "transport": {{
      "dynamic_viscosity": {{
        "model": "CUSTOM",
        "value": 5e-5
      }},
      "bulk_viscosity": 0.0,
      "thermal_conductivity": {{
        "model": "CUSTOM", 
        "value": 0.1
      }}
    }}
  }},
  "nondimensionalization_parameters": {{
    "density_reference": <reference density>,
    "length_reference": 0.1,
    "velocity_reference": <reference velocity>,
    "temperature_reference": <reference temperature>
  }},
  "forcings": {{
    "gravity": [0.0, 0.0, 0.0]
  }},
  "output": {{
    "primitives": ["density", "velocity", "pressure", "temperature"],
    "miscellaneous": ["mach_number", "schlieren"],
    "levelset": []
  }}
}}

SPECIFIC CALCULATIONS FOR ROCKET PROPULSION:
- For chamber conditions: P₀ = {pressure_ratio * 101325:.0f} Pa, T₀ = {temperature_inlet:.0f} K
- Use isentropic relations for nozzle flow
- Calculate density from ideal gas law: ρ = P/(R*T)
- Calculate sonic velocity: a = sqrt(γ*R*T)
- For supersonic flow: u = M * a
- Reference conditions for nondimensionalization

Generate ONLY the JSON configuration. Ensure all numerical values are properly calculated for supersonic rocket propulsion conditions.
"""
        
        return prompt
    
    def _parse_ai_response(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse AI response and extract JSON configuration"""
        
        try:
            # Find JSON in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                case_setup = json.loads(json_str)
                
                # Validate critical fields
                self._validate_internal_flow_config(case_setup)
                
                return case_setup
            else:
                raise ValueError("No valid JSON found in AI response")
                
        except Exception as e:
            logger.warning(f"Failed to parse AI response: {e}")
            return self._create_template_case_setup(context)
    
    def _validate_internal_flow_config(self, config: Dict[str, Any]) -> None:
        """Validate that configuration is suitable for internal flows"""
        
        # Check required sections
        required_sections = ["general", "domain", "boundary_conditions", 
                           "initial_condition", "material_properties"]
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        # Check boundary conditions for internal flow
        bc = config.get("boundary_conditions", {})
        
        # Must have inlet (SIMPLE_INFLOW or DIRICHLET)
        west_bc = bc.get("west", {}).get("type")
        if west_bc not in ["SIMPLE_INFLOW", "DIRICHLET"]:
            logger.warning(f"Unusual inlet boundary condition: {west_bc}")
        
        # Must have outlet (SIMPLE_OUTFLOW or ZEROGRADIENT)
        east_bc = bc.get("east", {}).get("type")
        if east_bc not in ["SIMPLE_OUTFLOW", "ZEROGRADIENT"]:
            logger.warning(f"Unusual outlet boundary condition: {east_bc}")
        
        print(f"   ✅ Validated internal flow configuration")
        print(f"   🔄 Inlet BC: {west_bc}")
        print(f"   🔄 Outlet BC: {east_bc}")
    
    def _create_template_case_setup(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create template case setup for supersonic internal flows"""
        
        simulation_name = context.get("simulation_name", "supersonic_internal_flow")
        flow_type = context.get("flow_type", "supersonic_nozzle")
        
        # Default supersonic rocket conditions
        chamber_pressure = context.get("chamber_pressure", 2.0e6)  # 20 bar
        chamber_temperature = context.get("temperature_inlet", 2800.0)  # 2800 K
        ambient_pressure = context.get("ambient_pressure", 101325.0)  # 1 atm
        gamma = context.get("gamma", 1.3)
        R = 287.0  # J/kg/K for hot gases
        
        # Calculate flow properties
        chamber_density = chamber_pressure / (R * chamber_temperature)
        sonic_velocity = (gamma * R * chamber_temperature) ** 0.5
        inlet_velocity = 0.5 * sonic_velocity  # Subsonic inlet
        
        # Reference conditions
        rho_ref = chamber_density
        u_ref = sonic_velocity
        T_ref = chamber_temperature
        L_ref = 0.1  # 10 cm characteristic length
        
        template = {
            "case_name": simulation_name,
            "general": {
                "case_name": simulation_name,
                "end_time": 0.005,
                "save_path": "./results_supersonic",
                "save_dt": 0.0005
            },
            "domain": {
                "x": {
                    "cells": 320,
                    "range": [-0.05, 0.15]
                },
                "y": {
                    "cells": 160,
                    "range": [-0.04, 0.04] 
                },
                "z": {
                    "cells": 1,
                    "range": [0.0, 1.0]
                },
                "decomposition": {
                    "split_x": 1,
                    "split_y": 1,
                    "split_z": 1
                }
            },
            "boundary_conditions": {
                "west": {
                    "type": "SIMPLE_INFLOW",
                    "primitives_callable": {
                        "rho": chamber_density,
                        "u": inlet_velocity,
                        "v": 0.0,
                        "w": 0.0
                    }
                },
                "east": {
                    "type": "SIMPLE_OUTFLOW",
                    "primitives_callable": {
                        "p": ambient_pressure
                    }
                },
                "north": {"type": "SYMMETRY"},
                "south": {"type": "SYMMETRY"},
                "top": {"type": "INACTIVE"},
                "bottom": {"type": "INACTIVE"}
            },
            "initial_condition": {
                "rho": chamber_density * 0.8,
                "u": inlet_velocity * 0.5,
                "v": 0.0,
                "w": 0.0,
                "p": chamber_pressure * 0.6
            },
            "material_properties": {
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
            },
            "nondimensionalization_parameters": {
                "density_reference": rho_ref,
                "length_reference": L_ref,
                "velocity_reference": u_ref,
                "temperature_reference": T_ref
            },
            "forcings": {
                "gravity": [0.0, 0.0, 0.0]
            },
            "output": {
                "primitives": ["density", "velocity", "pressure", "temperature"],
                "miscellaneous": ["mach_number", "schlieren"],
                "levelset": []
            }
        }
        
        print(f"   📐 Generated template for {flow_type}")
        print(f"   🔥 Chamber: {chamber_pressure/1e6:.1f} MPa, {chamber_temperature:.0f} K")
        print(f"   🌪️ Sonic velocity: {sonic_velocity:.0f} m/s")
        
        return template 