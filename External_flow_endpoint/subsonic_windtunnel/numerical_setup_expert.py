#!/usr/bin/env python3
"""
Numerical Setup Expert Agent
Specializes in all JAX-Fluids numerical parameters (30+ options)
Based on comprehensive JAX-Fluids documentation analysis
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class NumericalSetupExpert:
    """
    Expert agent for JAX-Fluids numerical setup parameters
    Handles 30+ numerical options with deep framework knowledge
    """
    
    def __init__(self, gemini_api_key: str = None):
        """Initialize the numerical setup expert"""
        
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
        
        print("Numerical Setup Expert Initialized")
        print("   JAX-Fluids parameter database loaded")
        print("   30+ numerical parameters expertise ready")
    
    def get_template_config(self) -> Dict[str, Any]:
        """Get a template configuration for testing"""
        return {
            "space_discretization": {
                "derivative_stencil": "CENTRAL_4",
                "reconstruction_stencil": "WENO5-Z",
                "convective_flux": "GODUNOV"
            },
            "time_discretization": {
                "time_integrator": "RK3",
                "time_step_kind": "CFL",
                "CFL": 0.5
            },
            "output": {
                "output_period": 100,
                "output_timestamps": [0.0, 1.0]
            },
            "material": {
                "equation_of_state": "IDEAL_GAS"
            }
        }

    def generate_numerical_setup(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate complete numerical_setup.json for JAX-Fluids
        Analyzes user prompt and expertly selects all numerical parameters
        """
        
        try:
            print("Analyzing prompt for numerical parameters...")
            
            # Generate the numerical setup using Gemini with comprehensive JAX-Fluids knowledge
            numerical_setup = self._extract_numerical_parameters(context)
            
            # Save to file
            sim_dir = context['simulation_directory']
            numerical_file = os.path.join(sim_dir, 'numerical_setup.json')
            
            with open(numerical_file, 'w') as f:
                json.dump(numerical_setup, f, indent=2)
            
            print(f"✅ Generated numerical_setup.json")
            print(f"📊 Parameters configured: {len(self._flatten_dict(numerical_setup))}")
            
            return {
                'success': True,
                'message': 'Numerical setup generated successfully',
                'numerical_setup': numerical_setup,
                'numerical_setup_file': numerical_file,
                'extracted_parameters': self._flatten_dict(numerical_setup)
            }
            
        except Exception as e:
            error_msg = f"Numerical setup error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                'success': False,
                'message': error_msg,
                'error_details': str(e)
            }
    
    def _extract_numerical_parameters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use Gemini to extract numerical parameters with JAX-Fluids expertise"""
        
        system_prompt = self._get_numerical_system_prompt()
        user_prompt = f"""
        Analyze this external flow simulation request and generate a complete JAX-Fluids numerical_setup.json:
        
        USER PROMPT: {context['user_prompt']}
        
        CONTEXT:
        - Simulation Type: External flow around 3D object with immersed boundaries
        - SDF Available: {'Yes' if context.get('sdf_file_path') else 'No'}
        - Domain: {context.get('domain_bounds', 'Auto-detect from SDF')}
        - Resolution: {context.get('resolution', 'Auto-detect from SDF')}
        
        Generate a complete numerical_setup.json with ALL required parameters optimized for subsonic external flow.
        """
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = self.model.invoke([HumanMessage(content=full_prompt)])
        
        # Parse the JSON response
        response_text = response.content.strip()
        
        # Extract JSON from response (handle markdown formatting)
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_text = response_text[json_start:json_end].strip()
        else:
            json_text = response_text
        
        try:
            numerical_setup = json.loads(json_text)
            return numerical_setup
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # Return a fallback configuration
            return self._get_fallback_numerical_setup()
    
    def _get_numerical_system_prompt(self) -> str:
        """Comprehensive system prompt with all JAX-Fluids numerical parameters"""
        
        return """You are a JAX-Fluids numerical setup expert with comprehensive knowledge of all 30+ numerical parameters.

**REQUIRED OUTPUT FORMAT:**
Respond with a complete JSON object for numerical_setup.json:

```json
{
    "conservatives": {
        "halo_cells": int (3-7, typical: 5 for WENO5),
        "time_integration": {
            "integrator": string ("EULER", "RK2", "RK3"),
            "CFL": float (0.1-0.9, typical: 0.5)
        },
        "convective_fluxes": {
            "convective_solver": string ("GODUNOV", "AUSM", "LAX_FRIEDRICHS", "CENTRAL"),
            "godunov": {
                "riemann_solver": string ("HLLC", "HLL", "ROE", "RUSANOV", "LAX_FRIEDRICHS"),
                "signal_speed": string ("EINFELDT", "ROE", "DAVIS"),
                "reconstruction_stencil": string ("WENO3", "WENO5", "WENO5-Z", "WENO7", "CENTRAL2", "CENTRAL4", "CENTRAL6"),
                "reconstruction_variable": string ("PRIMITIVE", "CONSERVATIVE", "CHAR-PRIMITIVE", "CHAR-CONSERVATIVE")
            },
            "ausm": {
                "flux_splitting": string ("AUSM", "AUSMPLUS", "AUSMPLUSUP"),
                "signal_speed": string ("EINFELDT", "ROE")
            }
        },
        "dissipative_fluxes": {
            "reconstruction_stencil": string ("CENTRAL2", "CENTRAL4", "CENTRAL6"),
            "derivative_stencil_center": string ("CENTRAL2", "CENTRAL4", "CENTRAL6", "CENTRAL8"),
            "derivative_stencil_face": string ("CENTRAL2", "CENTRAL4", "CENTRAL6")
        }
    },
    "active_physics": {
        "is_convective_flux": bool (always true),
        "is_viscous_flux": bool (true for viscous flows),
        "is_heat_flux": bool (true for heat transfer),
        "is_volume_force": bool (true for gravity/body forces),
        "is_surface_tension": bool (false for single phase),
        "is_levelset": bool (true for immersed boundaries)
    },
    "levelset": {
        "narrowband_computation": {
            "is_narrowband": bool (true for efficiency),
            "narrowband_size": int (5-15, typical: 10)
        },
        "reinitialization": {
            "is_reinitialization": bool (true),
            "reinitialization_interval": int (1-5, typical: 2),
            "reinitialization_steps": int (5-20, typical: 10)
        },
        "interface_interaction": {
            "levelset_model": string ("FLUID-SOLID-STATIC", "FLUID-SOLID-DYNAMIC"),
            "surface_tension_coefficient": float (0.0 for no surface tension)
        }
    },
    "precision": {
        "is_double_precision_compute": bool (true for accuracy),
        "is_double_precision_output": bool (true for accuracy)
    },
    "output": {
        "is_xdmf": bool (false, use HDF5),
        "derivative_stencil": string ("CENTRAL2", "CENTRAL4")
    }
}
```

**PARAMETER EXPERTISE:**

1. **Time Integration** (JAX-Fluids specific):
   - EULER: First-order, most stable, CFL ≤ 0.5
   - RK2: Second-order, good balance, CFL ≤ 0.7
   - RK3: Third-order, most accurate, CFL ≤ 0.9
   - For external flow: RK3 preferred for accuracy

2. **Convective Schemes** (Based on JAX-Fluids capabilities):
   - GODUNOV: Most accurate upwind method
     * HLLC: Best for external flow, handles contact discontinuities
     * WENO5-Z: State-of-the-art reconstruction for smooth flows
     * CHAR-PRIMITIVE: Best for external flow applications
   - AUSM: Good for subsonic flows
   - CENTRAL: Simple, robust, more dissipative

3. **Reconstruction Stencils** (JAX-Fluids options):
   - WENO3: 3rd order, robust, less accurate
   - WENO5: 5th order, standard choice
   - WENO5-Z: 5th order with improved weights, best for smooth flows
   - WENO7: 7th order, highest accuracy, more expensive
   - CENTRAL: Non-oscillatory, good for viscous flows

4. **Riemann Solvers** (JAX-Fluids implementations):
   - HLLC: Best overall for external flow
   - HLL: Robust, slightly more dissipative
   - ROE: Classical, good for smooth flows
   - RUSANOV: Most robust, most dissipative

5. **Levelset Parameters** (Immersed boundary specific):
   - Narrowband: Essential for efficiency with immersed boundaries
   - Reinitialization: Critical for levelset quality
   - FLUID-SOLID-STATIC: For fixed objects like propellers

6. **Halo Cells**: Must match reconstruction stencil
   - WENO3: 3 halo cells
   - WENO5: 5 halo cells
   - WENO7: 7 halo cells

**SUBSONIC EXTERNAL FLOW BEST PRACTICES:**

1. **Accuracy Priority**:
   - integrator: "RK3"
   - convective_solver: "GODUNOV"
   - riemann_solver: "HLLC"
   - reconstruction_stencil: "WENO5-Z"
   - reconstruction_variable: "CHAR-PRIMITIVE"

2. **Stability Priority**:
   - integrator: "RK2"
   - CFL: 0.3-0.5
   - riemann_solver: "HLLC"
   - reconstruction_stencil: "WENO5"

3. **Efficiency Priority**:
   - integrator: "RK2"
   - convective_solver: "AUSM"
   - reconstruction_stencil: "WENO3"

**PARAMETER SELECTION LOGIC:**

- Extract Mach number from prompt → affects solver choice
- Extract accuracy requirements → affects reconstruction order
- Extract object complexity → affects levelset parameters
- Extract time constraints → affects efficiency vs accuracy tradeoff
- External flow → always enable levelset for immersed boundaries
- Subsonic → HLLC riemann solver preferred
- Viscous effects mentioned → enable viscous_flux and heat_flux
- High-fidelity → WENO5-Z, RK3, double precision
- Quick analysis → WENO3, RK2, single precision

Always respond with the complete JSON only, no additional text."""

    def _get_fallback_numerical_setup(self) -> Dict[str, Any]:
        """Fallback numerical setup for subsonic external flow"""
        
        return {
            "conservatives": {
                "halo_cells": 5,
                "time_integration": {
                    "integrator": "RK3",
                    "CFL": 0.5
                },
                "convective_fluxes": {
                    "convective_solver": "GODUNOV",
                    "godunov": {
                        "riemann_solver": "HLLC",
                        "signal_speed": "EINFELDT",
                        "reconstruction_stencil": "WENO5-Z",
                        "reconstruction_variable": "CHAR-PRIMITIVE"
                    }
                },
                "dissipative_fluxes": {
                    "reconstruction_stencil": "CENTRAL4",
                    "derivative_stencil_center": "CENTRAL4",
                    "derivative_stencil_face": "CENTRAL4"
                }
            },
            "active_physics": {
                "is_convective_flux": True,
                "is_viscous_flux": False,
                "is_heat_flux": False,
                "is_volume_force": False,
                "is_levelset": True
            },
            "levelset": {
                "narrowband_computation": {
                    "is_narrowband": True,
                    "narrowband_size": 10
                },
                "reinitialization": {
                    "is_reinitialization": True,
                    "reinitialization_interval": 2,
                    "reinitialization_steps": 10
                },
                "interface_interaction": {
                    "levelset_model": "FLUID-SOLID-STATIC",
                    "surface_tension_coefficient": 0.0
                }
            },
            "precision": {
                "is_double_precision_compute": True,
                "is_double_precision_output": True
            },
            "output": {
                "is_xdmf": False,
                "derivative_stencil": "CENTRAL4"
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
        """Return list of all available numerical parameters"""
        
        return [
            "conservatives.halo_cells",
            "conservatives.time_integration.integrator",
            "conservatives.time_integration.CFL",
            "conservatives.convective_fluxes.convective_solver",
            "conservatives.convective_fluxes.godunov.riemann_solver",
            "conservatives.convective_fluxes.godunov.signal_speed",
            "conservatives.convective_fluxes.godunov.reconstruction_stencil",
            "conservatives.convective_fluxes.godunov.reconstruction_variable",
            "conservatives.dissipative_fluxes.reconstruction_stencil",
            "conservatives.dissipative_fluxes.derivative_stencil_center",
            "conservatives.dissipative_fluxes.derivative_stencil_face",
            "active_physics.is_convective_flux",
            "active_physics.is_viscous_flux",
            "active_physics.is_heat_flux",
            "active_physics.is_volume_force",
            "active_physics.is_levelset",
            "levelset.narrowband_computation.is_narrowband",
            "levelset.narrowband_computation.narrowband_size",
            "levelset.reinitialization.is_reinitialization",
            "levelset.reinitialization.reinitialization_interval",
            "levelset.reinitialization.reinitialization_steps",
            "levelset.interface_interaction.levelset_model",
            "levelset.interface_interaction.surface_tension_coefficient",
            "precision.is_double_precision_compute",
            "precision.is_double_precision_output",
            "output.is_xdmf",
            "output.derivative_stencil"
        ] 