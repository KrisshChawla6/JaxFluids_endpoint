#!/usr/bin/env python3
"""
VectraSim Internal Flow Numerical Setup Expert
Specialized AI agent for supersonic internal flow numerical configurations

This expert handles:
- Shock-capturing schemes (WENO5-Z, WENO7)
- Supersonic Riemann solvers (HLLC, ROE)
- High-speed compressible flow numerics
- Heat transfer and viscous flow settings
- Robust time integration for rocket propulsion
"""

import json
import logging
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

class InternalFlowNumericalExpert:
    """
    AI expert specialized in supersonic internal flow numerical setup
    Generates JAX-Fluids numerical configurations optimized for rocket propulsion
    """
    
    def __init__(self, gemini_api_key: str):
        """Initialize the internal flow numerical expert"""
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=gemini_api_key,
            temperature=0.1,
            max_tokens=4096
        )
        logger.info("🔢 Internal Flow Numerical Expert initialized")
    
    def generate_numerical_setup(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate JAX-Fluids numerical setup for supersonic internal flows
        
        Args:
            context: Internal flow context with flow parameters
            
        Returns:
            JAX-Fluids numerical setup dictionary
        """
        
        print("🔢 Internal Flow Numerical Expert - Generating Configuration")
        
        # Extract context parameters
        flow_type = context.get("flow_type", "supersonic_nozzle")
        mach_number = context.get("mach_number", 2.0)
        
        # Create AI prompt for supersonic numerics
        prompt = self._create_numerical_prompt(context)
        
        try:
            print("   🤖 AI generating supersonic numerical configuration...")
            
            # Invoke AI model
            messages = [HumanMessage(content=prompt)]
            response = self.model.invoke(messages)
            
            # Parse AI response
            numerical_setup = self._parse_ai_response(response.content, context)
            
            print(f"   ✅ Generated numerical config for {flow_type}")
            print(f"   🌪️ Mach number: {mach_number}")
            print(f"   🔢 Shock-capturing: WENO5-Z + HLLC")
            
            return numerical_setup
            
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            print(f"   ⚠️ AI generation failed, using template: {str(e)}")
            return self._create_template_numerical_setup(context)
    
    def _create_numerical_prompt(self, context: Dict[str, Any]) -> str:
        """Create specialized AI prompt for supersonic numerical setup"""
        
        flow_type = context.get("flow_type", "supersonic_nozzle")
        mach_number = context.get("mach_number", 2.0)
        
        prompt = f"""
You are a supersonic CFD expert generating JAX-Fluids numerical setup for rocket propulsion applications.

SIMULATION CONTEXT:
- Flow Type: {flow_type}
- Mach Number: {mach_number}
- Application: Rocket propulsion and supersonic internal flows

CRITICAL REQUIREMENTS FOR SUPERSONIC INTERNAL FLOWS:
1. Use WENO5-Z reconstruction for shock capturing
2. Use HLLC Riemann solver for robust supersonic flow
3. Enable viscous and heat flux for realistic rocket conditions
4. Use RK3 time integration for stability
5. Conservative CFL for supersonic stability
6. Double precision for high-temperature accuracy

MANDATORY JAX-FLUIDS NUMERICAL STRUCTURE FOR SUPERSONIC FLOWS:
{{
  "conservatives": {{
    "halo_cells": 5,
    "time_integration": {{
      "integrator": "RK3",
      "CFL": 0.4
    }},
    "convective_fluxes": {{
      "convective_solver": "GODUNOV",
      "godunov": {{
        "riemann_solver": "HLLC",
        "signal_speed": "EINFELDT",
        "reconstruction_stencil": "WENO5-Z",
        "reconstruction_variable": "CHAR-PRIMITIVE"
      }}
    }},
    "dissipative_fluxes": {{
      "reconstruction_stencil": "CENTRAL4",
      "derivative_stencil_center": "CENTRAL4",
      "derivative_stencil_face": "CENTRAL4"
    }},
    "positivity": {{
      "is_interpolation_limiter": true
    }}
  }},
  "active_physics": {{
    "is_convective_flux": true,
    "is_viscous_flux": true,
    "is_heat_flux": true,
    "is_volume_force": false,
    "is_surface_tension": false,
    "is_levelset": false
  }},
  "precision": {{
    "is_double_precision_compute": true,
    "is_double_precision_output": true
  }},
  "output": {{
    "is_xdmf": false,
    "derivative_stencil": "CENTRAL4"
  }}
}}

SPECIFIC CONSIDERATIONS FOR ROCKET PROPULSION:
- WENO5-Z provides excellent shock resolution for nozzle flows
- HLLC solver handles contact discontinuities and rarefactions
- Viscous effects important for boundary layers in nozzles  
- Heat flux critical for high-temperature combustion gases
- Conservative CFL (0.4) for supersonic stability
- Double precision for accurate thermodynamic calculations

Generate ONLY the JSON configuration optimized for supersonic rocket propulsion flows.
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
                numerical_setup = json.loads(json_str)
                
                # Validate critical fields
                self._validate_numerical_config(numerical_setup)
                
                return numerical_setup
            else:
                raise ValueError("No valid JSON found in AI response")
                
        except Exception as e:
            logger.warning(f"Failed to parse AI response: {e}")
            return self._create_template_numerical_setup(context)
    
    def _validate_numerical_config(self, config: Dict[str, Any]) -> None:
        """Validate that configuration is suitable for supersonic flows"""
        
        # Check required sections
        required_sections = ["conservatives", "active_physics", "precision"]
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        # Check for shock-capturing scheme
        convective = config.get("conservatives", {}).get("convective_fluxes", {})
        godunov = convective.get("godunov", {})
        reconstruction = godunov.get("reconstruction_stencil", "")
        
        if "WENO" not in reconstruction:
            logger.warning(f"Non-WENO reconstruction for supersonic flow: {reconstruction}")
        
        # Check Riemann solver
        riemann_solver = godunov.get("riemann_solver", "")
        if riemann_solver not in ["HLLC", "ROE", "AUSM"]:
            logger.warning(f"Unusual Riemann solver for supersonic: {riemann_solver}")
        
        print(f"   ✅ Validated supersonic numerical configuration")
        print(f"   🔧 Reconstruction: {reconstruction}")
        print(f"   ⚡ Riemann solver: {riemann_solver}")
    
    def _create_template_numerical_setup(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create template numerical setup for supersonic internal flows"""
        
        flow_type = context.get("flow_type", "supersonic_nozzle")
        mach_number = context.get("mach_number", 2.0)
        
        # Conservative CFL for supersonic flows
        cfl = 0.4 if mach_number > 1.5 else 0.5
        
        template = {
            "conservatives": {
                "halo_cells": 5,
                "time_integration": {
                    "integrator": "RK3",
                    "CFL": cfl
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
                },
                "positivity": {
                    "is_interpolation_limiter": True
                }
            },
            "active_physics": {
                "is_convective_flux": True,
                "is_viscous_flux": True,
                "is_heat_flux": True,
                "is_volume_force": False,
                "is_surface_tension": False,
                "is_levelset": False
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
        
        print(f"   📐 Generated template for {flow_type}")
        print(f"   🌪️ Mach: {mach_number}, CFL: {cfl}")
        print(f"   🔧 WENO5-Z + HLLC for shock capturing")
        
        return template 