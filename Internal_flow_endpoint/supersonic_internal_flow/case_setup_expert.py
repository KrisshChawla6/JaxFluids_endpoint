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
        """Create specialized AI prompt for supersonic internal flows with intelligent boundary conditions"""
        
        flow_type = context.get("flow_type", "supersonic_nozzle")
        mach_number = context.get("mach_number", 2.0)
        pressure_ratio = context.get("pressure_ratio", 10.0)
        temperature_inlet = context.get("temperature_inlet", 2800.0)
        simulation_name = context.get("simulation_name", "supersonic_internal")
        
        # Check for intelligent boundary conditions
        has_intelligent_bcs = context.get("has_intelligent_bcs", False)
        inlet_mask_file = context.get("inlet_mask_file", "")
        outlet_mask_file = context.get("outlet_mask_file", "")
        inlet_points = context.get("inlet_points", 0)
        outlet_points = context.get("outlet_points", 0)
        bc_storage_dir = context.get("bc_storage_directory", "")
        
        if has_intelligent_bcs:
            bc_integration_text = f"""
INTELLIGENT BOUNDARY CONDITIONS DETECTED:
✅ Inlet mask: {inlet_points:,} points - {inlet_mask_file}
✅ Outlet mask: {outlet_points:,} points - {outlet_mask_file}
✅ BC storage: {bc_storage_dir}

CRITICAL: Use JAX-Fluids FORCING SYSTEM instead of traditional SIMPLE_INFLOW/OUTFLOW!
- Do NOT use SIMPLE_INFLOW or SIMPLE_OUTFLOW boundary conditions
- ALL domain boundaries must be SYMMETRY (this is internal flow with immersed boundaries)
- Use "forcings" section to apply inlet/outlet conditions via the generated masks
- The forcing system will handle the virtual inlet/outlet boundary conditions
- This follows the proven rocket_simulation_final configuration pattern

FORCING CONFIGURATION REQUIREMENTS:
1. Add "forcings" section with mass_flow and temperature targets
2. Reference the generated inlet/outlet mask files in forcing configuration
3. Use high-pressure inlet conditions (6-7 MPa) for rocket propulsion
4. Use atmospheric outlet conditions (101 kPa)
5. Include proper immersed boundary levelset for nozzle walls
"""
        else:
            bc_integration_text = """
NO INTELLIGENT BOUNDARY CONDITIONS DETECTED:
⚠️ Falling back to traditional SIMPLE_INFLOW/OUTFLOW boundary conditions
- Use SIMPLE_INFLOW for inlet (west) - specify density, velocity, temperature
- Use SIMPLE_OUTFLOW for outlet (east) - specify pressure only
- Use SYMMETRY for top/bottom boundaries (axisymmetric nozzle)
"""

        prompt = f"""
You are a supersonic internal flow expert generating JAX-Fluids case setup for rocket propulsion applications.

SIMULATION CONTEXT:
- Flow Type: {flow_type}
- Mach Number: {mach_number}
- Pressure Ratio: {pressure_ratio}
- Inlet Temperature: {temperature_inlet} K
- Simulation: {simulation_name}

{bc_integration_text}

MANDATORY JAX-FLUIDS INTERNAL FLOW STRUCTURE (Enhanced with Intelligent BCs):
{{
  "general": {{
    "case_name": "{simulation_name}",
    "end_time": 0.05,
    "save_path": "./output/",
    "save_dt": 0.005
  }},
  "restart": {{
    "flag": false,
    "file_path": ""
  }},
  "domain": {{
    "x": {{
      "cells": 128,
      "range": [-200.0, 1800.0]
    }},
    "y": {{
      "cells": 64,
      "range": [-800.0, 800.0]
    }},
    "z": {{
      "cells": 64,
      "range": [-800.0, 800.0]
    }},
    "decomposition": {{
      "split_x": 1,
      "split_y": 1,
      "split_z": 1
    }}
  }},"""

        if has_intelligent_bcs:
            prompt += f"""
  "boundary_conditions": {{
    "east": {{"type": "SYMMETRY"}},
    "west": {{"type": "SYMMETRY"}},
    "north": {{"type": "SYMMETRY"}},
    "south": {{"type": "SYMMETRY"}},
    "top": {{"type": "SYMMETRY"}},
    "bottom": {{"type": "SYMMETRY"}}
  }},
  "initial_condition": {{
    "primitives": {{
      "rho": 1.0,
      "u": 10.0,
      "v": 0.0,
      "w": 0.0,
      "p": 1000000.0
    }},
    "levelset": "{bc_storage_dir}/internal_flow_bc_sdf_matrix.npy"
  }},
  "forcings": {{
    "mass_flow": {{
      "direction": "x",
      "target_value": 15.0,
      "inlet_mask_file": "{inlet_mask_file}",
      "outlet_mask_file": "{outlet_mask_file}"
    }},
    "temperature": {{
      "target_value": {temperature_inlet},
      "inlet_conditions": {{
        "pressure": 6900000.0,
        "temperature": {temperature_inlet},
        "velocity": 50.0
      }},
      "outlet_conditions": {{
        "pressure": 101325.0,
        "temperature": 288.15
      }}
    }}
  }},"""
        else:
            prompt += f"""
  "boundary_conditions": {{
    "east": {{
      "type": "SIMPLE_OUTFLOW",
      "primitives": {{
        "p": 101325.0
      }}
    }},
    "west": {{
      "type": "SIMPLE_INFLOW", 
      "primitives": {{
        "rho": 3.5,
        "u": {mach_number * 300.0},
        "v": 0.0,
        "w": 0.0,
        "p": {pressure_ratio * 101325.0},
        "T": {temperature_inlet}
      }}
    }},
    "north": {{"type": "SYMMETRY"}},
    "south": {{"type": "SYMMETRY"}},
    "top": {{"type": "SYMMETRY"}},
    "bottom": {{"type": "SYMMETRY"}}
  }},
  "initial_condition": {{
    "primitives": {{
      "rho": 1.0,
      "u": 50.0,
      "v": 0.0,
      "w": 0.0,
      "p": 500000.0
    }}
  }},"""

        prompt += f"""
  "material_properties": {{
    "equation_of_state": {{
      "model": "IdealGas",
      "specific_heat_ratio": 1.4,
      "specific_gas_constant": 287.0
    }},
    "transport": {{
      "dynamic_viscosity": {{
        "model": "CUSTOM",
        "value": 1.8e-05
      }},
      "bulk_viscosity": 0.0,
      "thermal_conductivity": {{
        "model": "PRANDTL",
        "prandtl_number": 0.72
      }}
    }}
  }},
  "nondimensionalization_parameters": {{
    "density_reference": 1.0,
    "length_reference": 1.0,
    "velocity_reference": 50.0,
    "temperature_reference": 288.15
  }},
  "output": {{
    "domain": {{
      "write_interval": 10,
      "write_start": 0,
      "primitives": ["density", "velocity", "pressure", "temperature"],
      "derived": ["mach_number", "speed_of_sound"]
    }}
  }}
}}

CRITICAL SUCCESS FACTORS:
1. {"✅ INTELLIGENT BCs: Use forcing system with generated masks" if has_intelligent_bcs else "⚠️ FALLBACK: Use traditional SIMPLE_INFLOW/OUTFLOW"}
2. ✅ High-temperature gas properties for rocket propulsion
3. ✅ Proper domain dimensions for internal rocket nozzle flow
4. ✅ Appropriate time stepping and output configuration
5. ✅ Supersonic-capable numerical schemes (will be handled by numerical expert)

Generate a complete JAX-Fluids case setup that implements the above requirements.
Return ONLY the JSON configuration, no explanatory text.
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
        """Create template case setup for supersonic internal flows with intelligent BC support"""
        
        simulation_name = context.get("simulation_name", "supersonic_internal_flow")
        flow_type = context.get("flow_type", "supersonic_nozzle")
        
        # Check for intelligent boundary conditions
        has_intelligent_bcs = context.get("has_intelligent_bcs", False)
        inlet_mask_file = context.get("inlet_mask_file", "")
        outlet_mask_file = context.get("outlet_mask_file", "")
        bc_storage_dir = context.get("bc_storage_directory", "")
        temperature_inlet = context.get("temperature_inlet", 3580.0)
        
        print(f"   📐 Generated template for {flow_type}")
        
        if has_intelligent_bcs:
            print(f"   🧠 Using intelligent boundary conditions with forcing system")
            print(f"   🔴 Inlet mask: {inlet_mask_file}")
            print(f"   🟢 Outlet mask: {outlet_mask_file}")
            
            # Enhanced template with intelligent boundary conditions (forcing system)
            template = {
                "general": {
                    "case_name": simulation_name,
                    "end_time": 0.05,
                    "save_path": "./output/",
                    "save_dt": 0.005
                },
                "restart": {
                    "flag": False,
                    "file_path": ""
                },
                "domain": {
                    "x": {
                        "cells": 128,
                        "range": [-200.0, 1800.0]
                    },
                    "y": {
                        "cells": 64,
                        "range": [-800.0, 800.0]
                    },
                    "z": {
                        "cells": 64,
                        "range": [-800.0, 800.0]
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
                    "levelset": f"{bc_storage_dir}/internal_flow_bc_sdf_matrix.npy"
                },
                "forcings": {
                    "mass_flow": {
                        "direction": "x",
                        "target_value": 15.0,
                        "inlet_mask_file": inlet_mask_file,
                        "outlet_mask_file": outlet_mask_file
                    },
                    "temperature": {
                        "target_value": temperature_inlet,
                        "inlet_conditions": {
                            "pressure": 6900000.0,
                            "temperature": temperature_inlet,
                            "velocity": 50.0
                        },
                        "outlet_conditions": {
                            "pressure": 101325.0,
                            "temperature": 288.15
                        }
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
                    "domain": {
                        "write_interval": 10,
                        "write_start": 0,
                        "primitives": ["density", "velocity", "pressure", "temperature"],
                        "derived": ["mach_number", "speed_of_sound"]
                    }
                }
            }
            
        else:
            print(f"   ⚠️ Falling back to traditional SIMPLE_INFLOW/OUTFLOW")
            
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
            
            print(f"   🔥 Chamber: {chamber_pressure/1e6:.1f} MPa, {chamber_temperature:.0f} K")
            print(f"   🌪️ Sonic velocity: {sonic_velocity:.0f} m/s")
            
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
                    "primitives_callable": {
                        "rho": chamber_density * 0.5,
                        "u": inlet_velocity * 0.1,
                        "v": 0.0,
                        "w": 0.0,
                        "p": ambient_pressure * 5.0
                    }
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
                    "density_reference": chamber_density,
                    "length_reference": 0.1,
                    "velocity_reference": sonic_velocity,
                    "temperature_reference": chamber_temperature
                },
                "output": {
                    "primitives": ["density", "velocity", "pressure", "temperature"],
                    "miscellaneous": ["mach_number", "schlieren"]
                }
            }
        
        return template 