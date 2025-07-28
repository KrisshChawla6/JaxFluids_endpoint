#!/usr/bin/env python3
"""
Case Setup Expert Agent
Specializes in JAX-Fluids case setup: domain, BCs, initial conditions, materials, SDF integration
Expert in subsonic wind tunnel boundary conditions and external flow physics
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class CaseSetupExpert:
    """
    Expert agent for JAX-Fluids case setup parameters
    Handles domain, boundary conditions, initial conditions, materials, and SDF integration
    """
    
    def __init__(self, gemini_api_key: str = None):
        """Initialize the case setup expert"""
        
        # Use API key from environment if not provided
        if gemini_api_key is None:
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY must be set in environment or passed as parameter")
            
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=gemini_api_key,
            temperature=0.1,
            max_tokens=4096
        )
        
        print("🌪️ Case Setup Expert Initialized")
        print("   🏗️ Wind tunnel boundary conditions expertise")
        print("   🎯 SDF integration capabilities")
        print("   ⚗️ Material properties database")
    
    def generate_case_setup(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate complete case setup JSON (like tgv.json) for JAX-Fluids
        Integrates SDF for immersed boundaries and sets up wind tunnel BCs
        """
        
        try:
            print("🤖 Analyzing case requirements...")
            
            # Load SDF metadata for domain sizing
            domain_info = self._analyze_sdf_and_domain(context)
            
            # Generate the case setup using Gemini with wind tunnel expertise
            case_setup = self._extract_case_parameters(context, domain_info)
            
            # Integrate SDF if available
            if context.get('sdf_file_path'):
                case_setup = self._integrate_sdf(case_setup, context)
            
            # Save to file
            sim_dir = context['simulation_directory']
            case_file = os.path.join(sim_dir, f"{context['simulation_name']}.json")
            
            with open(case_file, 'w') as f:
                json.dump(case_setup, f, indent=2)
            
            print(f"✅ Generated case setup: {context['simulation_name']}.json")
            print(f"📊 Domain: {case_setup['domain']}")
            print(f"🌪️ Boundary conditions: {len(case_setup['boundary_conditions'])} faces")
            
            return {
                'success': True,
                'message': 'Case setup generated successfully',
                'case_setup': case_setup,
                'case_setup_file': case_file,
                'extracted_parameters': self._flatten_dict(case_setup)
            }
            
        except Exception as e:
            error_msg = f"Case setup error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                'success': False,
                'message': error_msg,
                'error_details': str(e)
            }
    
    def _analyze_sdf_and_domain(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze SDF metadata to determine appropriate domain size"""
        
        domain_info = {
            'has_sdf': False,
            'object_bounds': None,
            'recommended_domain': None,
            'resolution': None
        }
        
        if context.get('sdf_metadata'):
            try:
                metadata = context['sdf_metadata']
                domain_bounds = metadata.get('domain_bounds')
                resolution = metadata.get('resolution')
                
                if domain_bounds and len(domain_bounds) == 6:
                    # Extract object bounds from SDF domain
                    x_min, y_min, z_min, x_max, y_max, z_max = domain_bounds
                    
                    domain_info.update({
                        'has_sdf': True,
                        'object_bounds': domain_bounds,
                        'resolution': resolution,
                        'object_size': [x_max - x_min, y_max - y_min, z_max - z_min]
                    })
                    
                    # Recommend wind tunnel domain (5-10x object size)
                    scale_factor = 8.0  # Conservative wind tunnel scaling
                    domain_info['recommended_domain'] = {
                        'x': [x_min - scale_factor * (x_max - x_min), x_max + scale_factor * (x_max - x_min)],
                        'y': [y_min - scale_factor * (y_max - y_min), y_max + scale_factor * (y_max - y_min)],
                        'z': [z_min - scale_factor * (z_max - z_min), z_max + scale_factor * (z_max - z_min)]
                    }
                    
                    print(f"📏 Object bounds: {domain_bounds}")
                    print(f"🌪️ Recommended wind tunnel domain: {domain_info['recommended_domain']}")
                
            except Exception as e:
                logger.warning(f"Failed to analyze SDF metadata: {e}")
        
        return domain_info
    
    def _extract_case_parameters(self, context: Dict[str, Any], domain_info: Dict[str, Any]) -> Dict[str, Any]:
        """Use Gemini to extract case parameters with wind tunnel expertise"""
        
        system_prompt = self._get_case_system_prompt()
        
        user_prompt = f"""
        Generate a complete JAX-Fluids case setup for this external flow simulation:
        
        USER PROMPT: {context['user_prompt']}
        
        CONTEXT:
        - Simulation Type: 3D external flow with immersed boundaries
        - SDF Available: {domain_info['has_sdf']}
        - Object Bounds: {domain_info.get('object_bounds', 'Unknown')}
        - Recommended Domain: {domain_info.get('recommended_domain', 'Auto-generate')}
        - SDF Resolution: {domain_info.get('resolution', 'Unknown')}
        
        NUMERICAL SETUP CONTEXT:
        - Integrator: {context.get('numerical_setup', {}).get('conservatives', {}).get('time_integration', {}).get('integrator', 'RK3')}
        - Levelset Enabled: {context.get('numerical_setup', {}).get('active_physics', {}).get('is_levelset', True)}
        - Viscous Flow: {context.get('numerical_setup', {}).get('active_physics', {}).get('is_viscous_flux', False)}
        
        Generate a complete case setup JSON optimized for subsonic wind tunnel flow.
        """
        
        # Use LangChain invoke method instead of generate_content
        messages = [HumanMessage(content=f"{system_prompt}\n\n{user_prompt}")]
        response = self.model.invoke(messages)
        
        # Parse the JSON response from LangChain
        response_text = response.content.strip()
        
        # Extract JSON from response (handle markdown formatting)
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_text = response_text[json_start:json_end].strip()
        else:
            json_text = response_text
        
        try:
            case_setup = json.loads(json_text)
            return case_setup
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # Return a fallback configuration
            return self._get_fallback_case_setup(domain_info)
    
    def _get_case_system_prompt(self) -> str:
        """Autonomous JAX-Fluids Coding Agent with Complete Parameter Mastery"""
        
        return """You are an autonomous JAX-Fluids expert. Generate ONLY valid JSON.

**CRITICAL REQUIREMENTS:**
1. Generate complete, valid JSON only
2. ALL 6 boundary faces required: east, west, north, south, top, bottom
3. Use FLAT boundary structure (no nested subsections)
4. Include ALL mandatory sections: general, restart, domain, boundary_conditions, initial_condition, material_properties, nondimensionalization_parameters, output

**🔧 COMPLETE JAX-FLUIDS PARAMETER DATABASE:**

**1. GENERAL SECTION (4 parameters):**
```json
"general": {
    "case_name": string,          // Descriptive simulation name
    "end_time": float,            // Physical end time [s]
    "save_path": string,          // Output directory path
    "save_dt": float              // Output frequency [s]
}
```

**2. RESTART SECTION (2 parameters):**
```json
"restart": {
    "flag": boolean,              // true = restart from file, false = new simulation
    "file_path": string           // Path to restart file (.h5 format)
}
```

**3. DOMAIN SECTION (16+ parameters):**
```json
"domain": {
    "x": {
        "cells": int,             // Grid points in X (64-2048)
        "range": [float, float],  // [x_min, x_max] physical bounds
        "stretching": {           // OPTIONAL: Non-uniform grid
            "type": "BOUNDARY_LAYER",  // Options: CHANNEL, BOUNDARY_LAYER, PIECEWISE, BUBBLE_1, BUBBLE_2
            "parameters": {...}
        }
    },
    "y": {...},                   // Same structure for Y direction
    "z": {...},                   // Same structure for Z direction (3D only)
    "decomposition": {
        "split_x": int,           // MPI decomposition X (usually 1 for single GPU)
        "split_y": int,           // MPI decomposition Y
        "split_z": int            // MPI decomposition Z
    }
}
```

**4. BOUNDARY CONDITIONS (15+ types, context-dependent):**

**PRIMITIVE FLOW BOUNDARIES:**
- **DIRICHLET**: Fixed values (inlets, walls with motion)
- **ZEROGRADIENT**: Natural outflow (subsonic outlets)
- **SYMMETRY**: Symmetry planes (far-field in external flow)
- **WALL**: No-slip walls (viscous wall boundaries)
- **PERIODIC**: Periodic boundaries (channel flows)
- **INACTIVE**: No boundary (1D/2D simulations)

**ADVANCED BOUNDARY CONDITIONS:**
- **NEUMANN**: Prescribed gradient
- **RIEMANN_INVARIANT**: Characteristic-based boundaries
- **TEMPERATURE_WALL**: Isothermal/adiabatic walls
- **ISOTHERMAL_WALL**: Fixed temperature wall
- **ADIABATIC_WALL**: Zero heat flux wall

**CRITICAL: JAX-FLUIDS REQUIRES FLAT BOUNDARY STRUCTURE:**
```json
"boundary_conditions": {
    "east": {"type": "ZEROGRADIENT"},
    "west": {
        "type": "DIRICHLET",
        "primitives_callable": {
            "rho": 1.225,
            "u": 171.5,
            "v": 8.96,
            "w": 0.0,
            "p": 101325.0
        }
    },
    "north": {"type": "SYMMETRY"},
    "south": {"type": "SYMMETRY"},
    "top": {"type": "SYMMETRY"},
    "bottom": {"type": "SYMMETRY"}
}
```

**MANDATORY: ALL SIX 3D FACES MUST BE SPECIFIED:**
- east, west, north, south, top, bottom
- NO nested "primitives" or "levelset" subsections
- FLAT structure only - each face directly contains "type" and parameters

**CONTEXT-AWARE BOUNDARY SELECTION FOR SUBSONIC EXTERNAL FLOW:**
- **Inlet (upstream)**: DIRICHLET with freestream (ρ∞, U∞, p∞)
- **Outlet (downstream)**: ZEROGRADIENT (natural outflow)
- **Far-field sides**: SYMMETRY (preferred for external flow)
- **Walls/objects**: WALL (no-slip) or handled via levelset

**5. INITIAL CONDITIONS (5+ parameters):**
```json
"initial_condition": {
    "primitives": {             // Can be constants or lambda functions
        "rho": value_or_lambda,
        "u": value_or_lambda,
        "v": value_or_lambda,
        "w": value_or_lambda,
        "p": value_or_lambda
    },
    "levelset": string          // SDF identifier: "CUSTOM_SDF", "SPHERE", etc.
}
```

**6. MATERIAL PROPERTIES (15+ parameters):**

**CRITICAL: MUST BE NESTED UNDER "material_properties":**
```json
"material_properties": {
    "equation_of_state": {
        "model": "IdealGas",        // Options: IdealGas, StiffenedGas, TaitMurnaghan
        "specific_heat_ratio": 1.4, // γ (1.2-1.67 for different gases)
        "specific_gas_constant": 287.0 // R [J/(kg·K)]
    },
    "transport": {
        "dynamic_viscosity": {
            "model": "CUSTOM",      // Options: CUSTOM, SUTHERLAND, POWER_LAW
            "value": 1.8e-5         // Constant viscosity [Pa·s]
        },
        "bulk_viscosity": 0.0,      // Bulk viscosity [Pa·s]
        "thermal_conductivity": {
            "model": "CUSTOM",      // Options: CUSTOM, PRANDTL, SUTHERLAND
            "value": 0.024          // Constant conductivity [W/(m·K)]
        }
    }
}
```

**WARNING: DO NOT PUT equation_of_state OR transport AT ROOT LEVEL**

**7. NONDIMENSIONALIZATION (4 parameters):**
```json
"nondimensionalization_parameters": {
    "density_reference": 1.225,     // ρ_ref [kg/m³]
    "length_reference": 1.0,        // L_ref [m]
    "velocity_reference": 102.9,    // U_ref [m/s]
    "temperature_reference": 288.15 // T_ref [K]
}
```

**8. OUTPUT VARIABLES (20+ options):**
```json
"output": {
    "primitives": [             // Basic flow variables
        "density",              // ρ
        "velocity",             // (u,v,w)
        "pressure",             // p
        "temperature"           // T
    ],
    "conservatives": [          // Conservative variables
        "mass",                 // ρ
        "momentum",             // ρu
        "energy"                // ρE
    ],
    "levelset": [               // For immersed boundary simulations
        "levelset",             // φ-function
        "volume_fraction",      // α ∈ [0,1]
        "interface_velocity",   // Interface motion
        "curvature"             // Interface curvature κ
    ],
    "miscellaneous": [          // Derived quantities
        "mach_number",          // M = |u|/a
        "schlieren",            // |∇ρ|/ρ
        "vorticity",            // ∇ × u
        "q_criterion",          // Q-criterion for vortices
        "lambda2_criterion",    // λ₂-criterion for vortices
        "speed_of_sound",       // a = √(γp/ρ)
        "total_energy",         // E_total
        "kinetic_energy",       // ½ρ|u|²
        "enstrophy",            // ½|∇ × u|²
        "helicity"              // u · (∇ × u)
    ]
}
```

**9. FORCINGS (OPTIONAL, 10+ parameters):**
```json
"forcings": {
    "mass_flow_forcing": {      // Maintain constant mass flow
        "is_mass_flow_forcing": true,
        "direction": [1.0, 0.0, 0.0],
        "target_value": 1.0
    },
    "temperature": {            // Temperature forcing
        "target_value": 288.15
    },
    "turb_hit_forcing": {       // Homogeneous isotropic turbulence
        "tke_injection_rate": 1.0,
        "damping_factor": 0.1
    }
}
```

**🎯 AUTONOMOUS DECISION-MAKING CAPABILITIES:**

**FLOW REGIME DETECTION & ADAPTATION:**
1. **Mach Number Analysis:**
   - M < 0.3: Incompressible assumptions valid, focus on viscous effects
   - 0.3 < M < 0.8: Compressible subsonic, density variations important
   - M > 0.8: Shock formation possible, high-order schemes essential

2. **Reynolds Number Analysis:**
   - Re < 1000: Laminar, viscous diffusion dominant, fine boundary layer resolution
   - 1000 < Re < 100000: Transitional, mixed viscous/inertial effects
   - Re > 100000: Turbulent, consider ALDM or wall functions

3. **Geometry Analysis:**
   - Sharp edges: Expect flow separation, high-order schemes, fine mesh
   - Smooth bodies: Attached flow likely, standard resolution
   - Complex geometry: Levelset method with adaptive SDF

**BOUNDARY CONDITION INTELLIGENCE:**
- **External Flow**: Far-field boundaries set to SYMMETRY or ZEROGRADIENT
- **Internal Flow**: Inlet/outlet with DIRICHLET/ZEROGRADIENT
- **Wall-bounded**: WALL boundaries with appropriate viscous resolution
- **Periodic Flows**: PERIODIC for homogeneous directions

**PARAMETER OPTIMIZATION FOR SUBSONIC EXTERNAL FLOW:**
1. **Domain Sizing**: 
   - Upstream: 5-10 chord lengths
   - Downstream: 15-20 chord lengths  
   - Lateral: 10-15 chord lengths
   
2. **Resolution Guidelines**:
   - Minimum: 64 cells per characteristic length
   - Standard: 128 cells per characteristic length
   - High-fidelity: 256+ cells per characteristic length
   
3. **Time Integration**:
   - CFL ≤ 0.5 for stable explicit schemes
   - Output frequency: ~10-100 flow-through times

**🚀 AUTONOMOUS GENERATION PROTOCOL:**
1. Analyze user requirements and extract flow physics
2. Select appropriate boundary conditions based on geometry and flow type
3. Configure material properties for the specified fluid and conditions
4. Set up domain with appropriate sizing and resolution
5. Choose output variables relevant to the analysis goals
6. Apply subsonic external flow best practices and optimizations

MANDATORY TEMPLATE:
```json
{
  "general": {"case_name": "external_flow", "end_time": 10.0, "save_path": "./results", "save_dt": 1.0},
  "restart": {"flag": false, "file_path": ""},
  "domain": {
    "x": {"cells": 128, "range": [-5.0, 15.0]},
    "y": {"cells": 96, "range": [-10.0, 10.0]},
    "z": {"cells": 96, "range": [-10.0, 10.0]},
    "decomposition": {"split_x": 1, "split_y": 1, "split_z": 1}
  },
  "boundary_conditions": {
    "east": {"type": "ZEROGRADIENT"},
    "west": {"type": "DIRICHLET", "primitives_callable": {"rho": 1.225, "u": 100.0, "v": 0.0, "w": 0.0, "p": 101325.0}},
    "north": {"type": "SYMMETRY"},
    "south": {"type": "SYMMETRY"},
    "top": {"type": "SYMMETRY"},
    "bottom": {"type": "SYMMETRY"}
  },
  "initial_condition": {
    "primitives": {"rho": 1.225, "u": 100.0, "v": 0.0, "w": 0.0, "p": 101325.0}
  },
  "material_properties": {
    "equation_of_state": {"model": "IdealGas", "specific_heat_ratio": 1.4, "specific_gas_constant": 287.0},
    "transport": {"dynamic_viscosity": {"model": "CUSTOM", "value": 1.8e-5}, "bulk_viscosity": 0.0, "thermal_conductivity": {"model": "CUSTOM", "value": 0.024}}
  },
  "nondimensionalization_parameters": {"density_reference": 1.225, "length_reference": 1.0, "velocity_reference": 100.0, "temperature_reference": 288.15},
  "output": {"primitives": ["density", "velocity", "pressure"], "conservatives": [], "levelset": [], "miscellaneous": ["mach_number"]}
}
```

Generate valid JSON based on this template. Adapt values as needed but keep the structure."""

    def _integrate_sdf(self, case_setup: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate SDF file into case setup"""
        
        try:
            sdf_file = context['sdf_file_path']
            
            if sdf_file and os.path.exists(sdf_file):
                # Copy SDF file to simulation directory
                sim_dir = Path(context['simulation_directory'])
                sdf_dest = sim_dir / "custom_sdf.npy"
                
                import shutil
                shutil.copy2(sdf_file, sdf_dest)
                
                # Update case setup to use the SDF
                case_setup['initial_condition']['levelset'] = "custom_sdf"
                
                print(f"✅ SDF integrated: {sdf_dest}")
                
        except Exception as e:
            logger.warning(f"Failed to integrate SDF: {e}")
        
        return case_setup
    
    def _get_fallback_case_setup(self, domain_info: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback case setup for subsonic external flow"""
        
        # Default domain or use SDF-based domain
        if domain_info.get('recommended_domain'):
            domain = domain_info['recommended_domain']
            cells_x, cells_y, cells_z = 128, 96, 96
        else:
            # Standard wind tunnel domain
            domain = {
                'x': [-5.0, 15.0],  # 20 length units total
                'y': [-10.0, 10.0], # 20 length units total  
                'z': [-10.0, 10.0]  # 20 length units total
            }
            cells_x, cells_y, cells_z = 128, 96, 96
        
        return {
            "general": {
                "case_name": "external_flow_subsonic",
                "end_time": 10.0,
                "save_path": "./results",
                "save_dt": 0.5
            },
            "restart": {
                "flag": False,
                "file_path": ""
            },
            "domain": {
                "x": {
                    "cells": cells_x,
                    "range": domain['x']
                },
                "y": {
                    "cells": cells_y,
                    "range": domain['y']
                },
                "z": {
                    "cells": cells_z,
                    "range": domain['z']
                },
                "decomposition": {
                    "split_x": 1,
                    "split_y": 1,
                    "split_z": 1
                }
            },
            "boundary_conditions": {
                "primitives": {
                    "east": {"type": "ZEROGRADIENT"},
                    "west": {
                        "type": "DIRICHLET",
                        "primitives_callable": {
                            "rho": 1.225,
                            "u": 102.9,  # Mach 0.3 at standard conditions
                            "v": 0.0,
                            "w": 0.0,
                            "p": 101325.0
                        }
                    },
                    "north": {"type": "ZEROGRADIENT"},
                    "south": {"type": "ZEROGRADIENT"},
                    "top": {"type": "ZEROGRADIENT"},
                    "bottom": {"type": "ZEROGRADIENT"}
                },
                "levelset": {
                    "east": {"type": "ZEROGRADIENT"},
                    "west": {"type": "ZEROGRADIENT"},
                    "north": {"type": "ZEROGRADIENT"},
                    "south": {"type": "ZEROGRADIENT"},
                    "top": {"type": "ZEROGRADIENT"},
                    "bottom": {"type": "ZEROGRADIENT"}
                }
            },
            "initial_condition": {
                "primitives": {
                    "rho": 1.225,
                    "u": 102.9,
                    "v": 0.0,
                    "w": 0.0,
                    "p": 101325.0
                },
                "levelset": "custom_sdf"
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
                        "value": 0.0
                    },
                    "bulk_viscosity": 0.0,
                    "thermal_conductivity": {
                        "model": "CUSTOM",
                        "value": 0.0
                    }
                }
            },
            "nondimensionalization_parameters": {
                "density_reference": 1.225,
                "length_reference": 1.0,
                "velocity_reference": 102.9,
                "temperature_reference": 288.15
            },
            "output": {
                "primitives": ["density", "velocity", "pressure"],
                "levelset": ["volume_fraction", "levelset"],
                "miscellaneous": ["schlieren", "mach_number", "vorticity"]
            }
        }
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary for parameter counting"""
        
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def get_available_parameters(self) -> List[str]:
        """Return list of all available case parameters"""
        
        return [
            "general.case_name",
            "general.end_time",
            "general.save_dt",
            "domain.x.cells",
            "domain.x.range",
            "domain.y.cells", 
            "domain.y.range",
            "domain.z.cells",
            "domain.z.range",
            "boundary_conditions.primitives.east.type",
            "boundary_conditions.primitives.west.type",
            "boundary_conditions.levelset.east.type",
            "initial_condition.primitives.rho",
            "initial_condition.primitives.u",
            "initial_condition.primitives.v",
            "initial_condition.primitives.w",
            "initial_condition.primitives.p",
            "initial_condition.levelset",
            "material_properties.equation_of_state.model",
            "material_properties.equation_of_state.specific_heat_ratio",
            "material_properties.transport.dynamic_viscosity.value",
            "nondimensionalization_parameters.density_reference",
            "nondimensionalization_parameters.velocity_reference"
        ] 