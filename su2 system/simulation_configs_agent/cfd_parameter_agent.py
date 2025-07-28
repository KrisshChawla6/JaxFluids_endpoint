#!/usr/bin/env python3
"""
CFD Parameter Agent

Intelligent agent that converts user prompts into CFD simulation parameters
using Gemini 2.0 Flash and generates SU2 configuration files.
"""

import os
import json
import logging
import google.generativeai as genai
from typing import Dict, Any, Optional
from wind_tunnel_generator import (
    WindTunnelSimulation,
    create_config_with_extracted_markers,
    WindTunnelOrientation,
    FlowConditions,
    SolverSettings,
    TurbulenceModel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CFDParameterAgent:
    """Intelligent agent for converting user prompts to CFD parameters"""
    
    def __init__(self, api_key: str):
        """Initialize the agent with Gemini API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.wind_tunnel_sim = WindTunnelSimulation()
        
    def get_system_prompt(self) -> str:
        """System prompt defining the agent's role and output format"""
        return """You are a CFD (Computational Fluid Dynamics) parameter extraction agent. Your job is to analyze user prompts and extract COMPREHENSIVE simulation parameters for wind tunnel CFD simulations using SU2 solver.

**REQUIRED OUTPUT FORMAT:**
You must respond with a valid JSON object containing these exact parameters:

```json
{
    "mach_number": float (0.05 to 0.2, typical: 0.15 for subsonic),
    "reynolds_number": float (1e4 to 1e6, typical: 1e5),
    "angle_of_attack": float (-10 to 10 degrees, typical: 0-8),
    "max_iterations": int (SMART: based on flow complexity - see guidelines below),
    "wind_tunnel_orientation": string ("+X", "-X", "+Y", "-Y", "+Z", "-Z"),
    "solver_type": string ("EULER", "NAVIER_STOKES", "RANS"),
    "turbulence_model": string ("NONE", "SA", "SST"),
    "mesh_file_name": string (name of mesh file if mentioned, or "auto"),
    "simulation_description": string (brief description of what's being simulated),
    
    "temperature": float (273.15 to 373.15 K, typical: 288.15 for standard conditions),
    "pressure": float (80000 to 120000 Pa, typical: 101325 for sea level),
    "cfl_number": float (0.05 to 1.0, typical: 0.1 for stability),
    "convective_scheme": string ("ROE", "LAX-FRIEDRICH", "AUSM", "HLLC", "JST"),
    "linear_solver": string ("FGMRES", "BCGSTAB", "GMRES", "RESTARTED_FGMRES"),
    "wall_type": string ("HEATFLUX", "ISOTHERMAL", "ADIABATIC"),
    "turbulence_intensity": float (0.01 to 0.15, typical: 0.05),
    "reference_length": float (0.1 to 10.0 m, typical: 1.0),
    "reference_area": float (0.01 to 100.0 m², typical: 1.0),
    "wall_markers": array of strings (e.g., ["wall", "airfoil", "body"]),
    "farfield_markers": array of strings (e.g., ["farfield", "inlet", "outlet"]),
    "inlet_markers": array of strings (optional, e.g., ["inlet"]),
    "outlet_markers": array of strings (optional, e.g., ["outlet"]),
    "symmetry_markers": array of strings (optional, e.g., ["symmetry"])
}
```

**COMPREHENSIVE PARAMETER GUIDELINES:**

1. **Flow Conditions:**
   - **Mach Number**: 0.05-0.2 (validated range for convergence)
   - **Reynolds Number**: 1e4-1e5 (conservative values)
   - **Angle of Attack**: 0-10° (tested successfully)
   - **Temperature**: Analyze context:
     * Standard conditions: 288.15 K (15°C)
     * Cold conditions/high altitude: 250-280 K
     * Hot conditions/ground level: 295-310 K
     * Explicit temperature mentioned: use that value
   - **Pressure**: Analyze context:
     * Sea level standard: 101325 Pa
     * High altitude: 80000-90000 Pa
     * Pressurized conditions: 110000-120000 Pa
     * Explicit pressure mentioned: use that value

2. **Solver Settings (INTELLIGENT SELECTION):**
   - **CFL Number**: Base on stability requirements:
     * EULER + simple geometry: 0.5-1.0
     * RANS + complex flow: 0.1-0.3
     * High Mach (>0.15): 0.05-0.2
     * Default: 0.1 for stability
   - **Convective Scheme**: Base on solver and flow:
     * EULER solver: "ROE" or "LAX-FRIEDRICH"
     * RANS solver: "ROE" or "AUSM"
     * High-speed flow: "HLLC"
     * Centered schemes: "JST" (CRITICAL: JST is incompatible with MUSCL reconstruction)
     * Default: "ROE"
   - **Linear Solver**: Base on problem size:
     * Standard problems: "FGMRES"
     * Large problems: "RESTARTED_FGMRES"
     * Difficult convergence: "BCGSTAB"
     * Default: "FGMRES"

   **CRITICAL CONSTRAINT - NUMERICAL SCHEME COMPATIBILITY:**
   - **NEVER** combine JST with MUSCL reconstruction (causes "Centered schemes do not use MUSCL reconstruction" error)
   - **UPWIND schemes** (ROE, AUSM, HLLC, LAX-FRIEDRICH) work with MUSCL reconstruction
   - **CENTERED schemes** (JST) must NOT use MUSCL reconstruction
   - If choosing JST: convective_scheme="JST" (MUSCL will be automatically set to NO by the generator)
   - If choosing ROE/AUSM/HLLC: convective_scheme="ROE" (MUSCL will be automatically set to YES by the generator)
   - **DEFAULT SAFE CHOICE**: "ROE" (works with all configurations)

3. **Boundary Conditions (INTELLIGENT EXTRACTION):**
   - **Wall Type**: Analyze thermal requirements:
     * Heat transfer studies: "ISOTHERMAL"
     * Adiabatic flow: "ADIABATIC"
     * Standard CFD: "HEATFLUX"
     * Default: "HEATFLUX"
   - **Boundary Markers**: Extract from geometry context:
     * Airfoil: wall_markers=["wall", "airfoil"], farfield_markers=["farfield"]
     * Car/vehicle: wall_markers=["wall", "body", "wheels"], farfield_markers=["farfield", "inlet", "outlet"]
     * Propeller: wall_markers=["wall", "blade"], farfield_markers=["farfield"]
     * Generic: wall_markers=["wall"], farfield_markers=["farfield"]

4. **Reference Values (GEOMETRY-BASED):**
   - **Reference Length**: Extract from context:
     * Airfoil chord: 1.0 m (typical)
     * Car length: 4.0-5.0 m
     * Propeller diameter: 2.0-3.0 m
     * Small model: 0.5-1.0 m
     * Large model: 2.0-10.0 m
     * Default: 1.0 m
   - **Reference Area**: Extract from context:
     * Airfoil: 1.0 m² (chord × span)
     * Car frontal area: 2.0-2.5 m²
     * Propeller disk area: calculate from diameter
     * Default: 1.0 m²

5. **Turbulence Settings:**
   - **Turbulence Intensity**: Base on flow conditions:
     * Low turbulence (wind tunnel): 0.01-0.02
     * Typical external flow: 0.05
     * High turbulence environment: 0.10-0.15
     * Default: 0.05

6. **Max Iterations (CRITICAL - EXTRACT FROM USER PROMPT)**:
   - **ALWAYS EXTRACT EXPLICIT NUMBERS FROM USER PROMPT FIRST**
   - Look for keywords: "iterations", "iter", "steps", "max", "maximum"
   - Examples: "20 iterations" → max_iterations: 20, "run for 50 steps" → max_iterations: 50
   - **RESPECT USER'S EXPLICIT VALUES** - do not override with defaults
   - Only use intelligent defaults if NO specific number mentioned:
     * EULER solver: 150-200 iterations
     * RANS solver: 200-300 iterations
     * Complex geometry: +100 iterations
     * High angle of attack (>5°): +50 iterations

**NUMERICAL SCHEME SAFETY RULES:**
1. **SAFE DEFAULT**: Always use "ROE" unless specifically requested otherwise
2. **JST WARNING**: Only use "JST" for special centered scheme requirements (rare)
3. **COMPATIBILITY**: ROE/AUSM/HLLC work with all solver types and MUSCL settings

**CONTEXT ANALYSIS EXAMPLES:**

- "Airfoil analysis at cruise conditions":
  * temperature: 288.15, pressure: 101325, reference_length: 1.0, reference_area: 1.0
  * wall_markers: ["wall", "airfoil"], farfield_markers: ["farfield"]
  * cfl_number: 0.1, convective_scheme: "ROE" (SAFE CHOICE)

- "F1 car aerodynamics at high speed":
  * reference_length: 5.0, reference_area: 2.2, turbulence_intensity: 0.08
  * wall_markers: ["wall", "body", "wheels"], farfield_markers: ["farfield", "inlet", "outlet"]
  * cfl_number: 0.1, convective_scheme: "ROE" (SAFE CHOICE)

- "High altitude propeller analysis":
  * temperature: 250.0, pressure: 85000, reference_length: 2.5, reference_area: 4.9
  * wall_markers: ["wall", "blade"], farfield_markers: ["farfield"]
  * turbulence_intensity: 0.03, convective_scheme: "ROE" (SAFE CHOICE)

- "Hot weather aircraft performance":
  * temperature: 305.0, pressure: 101325, turbulence_intensity: 0.06
  * cfl_number: 0.05 (for stability in hot conditions), convective_scheme: "ROE" (SAFE CHOICE)

- "Special centered scheme analysis" (RARE):
  * convective_scheme: "JST" (ONLY when specifically requested - MUSCL will be auto-disabled)

- "Quick test with 20 iterations":
  * max_iterations: 20 (EXTRACT EXPLICIT NUMBER FROM USER)

- "Run simulation for 50 steps maximum":
  * max_iterations: 50 (EXTRACT EXPLICIT NUMBER FROM USER)

- "Use maximum of 100 iterations for convergence":
  * max_iterations: 100 (EXTRACT EXPLICIT NUMBER FROM USER)

**IMPORTANT**: Respond ONLY with the JSON object, no additional text or explanation. Extract ALL parameters intelligently based on the context, don't just use defaults."""

    def extract_parameters(self, user_prompt: str) -> Dict[str, Any]:
        """Extract CFD parameters from user prompt using Gemini"""
        
        # Handle minimal prompts by expanding them with context
        if user_prompt.strip().lower() in ['run', 'start', 'go', 'simulate']:
            user_prompt = "external airfoil flow analysis at moderate angle of attack with standard atmospheric conditions"
            logger.info(f"Expanded minimal prompt to: '{user_prompt}'")
        elif len(user_prompt.strip()) < 10:
            # Very short prompts get expanded with airfoil context
            user_prompt = f"airfoil aerodynamic analysis: {user_prompt}"
            logger.info(f"Expanded short prompt to: '{user_prompt}'")
        
        system_prompt = self.get_system_prompt()
        
        full_prompt = f"{system_prompt}\n\nUser Request: {user_prompt}\n\nExtract CFD parameters:"
        
        try:
            response = self.model.generate_content(full_prompt)
            response_text = response.text.strip()
            
            # Remove markdown code block if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]  # Remove '```json'
            if response_text.endswith('```'):
                response_text = response_text[:-3]  # Remove '```'
            
            # Parse JSON
            parameters = json.loads(response_text)
            
            # Required fields for simulation (made more flexible for boundary markers)
            required_fields = [
                'mach_number', 'reynolds_number', 'angle_of_attack', 'solver_type',
                'turbulence_model', 'wind_tunnel_orientation', 'max_iterations',
                'simulation_description', 'mesh_file_name', 
                'temperature', 'pressure', 'cfl_number', 'convective_scheme',
                'linear_solver', 'wall_type', 'turbulence_intensity',
                'reference_length', 'reference_area', 'wall_markers',
                'farfield_markers'
                # Note: inlet_markers, outlet_markers, symmetry_markers are optional
            ]
            
            for field in required_fields:
                if field not in parameters:
                    raise ValueError(f"Missing required field: {field}")
            
            # Ensure optional boundary markers exist (can be empty lists)
            optional_marker_fields = ['inlet_markers', 'outlet_markers', 'symmetry_markers']
            for field in optional_marker_fields:
                if field not in parameters:
                    parameters[field] = []  # Default to empty list
            
            logger.info(f"Extracted parameters: {parameters}")
            return parameters
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response was: {response_text}")
            raise ValueError(f"Invalid JSON response from Gemini: {e}")
            
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            raise
    
    def validate_and_normalize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize extracted parameters"""
        
        normalized = params.copy()
        
        # Validate Mach number (convergence-tested range)
        if not (0.05 <= params['mach_number'] <= 0.2):
            logger.warning(f"Mach number {params['mach_number']} out of range, clamping to 0.05-0.2")
            normalized['mach_number'] = max(0.05, min(0.2, params['mach_number']))
        
        # Validate Reynolds number (convergence-tested range)
        if not (1e4 <= params['reynolds_number'] <= 1e6):
            logger.warning(f"Reynolds number {params['reynolds_number']} out of range, clamping to 1e4-1e6")
            normalized['reynolds_number'] = max(1e4, min(1e6, params['reynolds_number']))
        
        # Validate angle of attack (convergence-tested range)
        if not (-10 <= params['angle_of_attack'] <= 10):
            logger.warning(f"Angle of attack {params['angle_of_attack']} out of range, clamping to -10 to 10")
            normalized['angle_of_attack'] = max(-10, min(10, params['angle_of_attack']))
        
        # Validate temperature
        if not (273.15 <= params.get('temperature', 288.15) <= 373.15):
            logger.warning(f"Temperature {params.get('temperature')} out of range, clamping to 273.15-373.15 K")
            normalized['temperature'] = max(273.15, min(373.15, params.get('temperature', 288.15)))
        
        # Validate pressure
        if not (80000 <= params.get('pressure', 101325) <= 120000):
            logger.warning(f"Pressure {params.get('pressure')} out of range, clamping to 80000-120000 Pa")
            normalized['pressure'] = max(80000, min(120000, params.get('pressure', 101325)))
        
        # Validate CFL number
        if not (0.05 <= params.get('cfl_number', 0.1) <= 1.0):
            logger.warning(f"CFL number {params.get('cfl_number')} out of range, clamping to 0.05-1.0")
            normalized['cfl_number'] = max(0.05, min(1.0, params.get('cfl_number', 0.1)))
        
        # Validate convective scheme with safety bias toward ROE
        valid_schemes = ["ROE", "LAX-FRIEDRICH", "AUSM", "HLLC", "JST"]
        scheme = params.get('convective_scheme', 'ROE')
        
        if scheme not in valid_schemes:
            logger.warning(f"Invalid convective scheme {scheme}, using safe default ROE")
            normalized['convective_scheme'] = 'ROE'
        elif scheme == 'JST':
            # JST is valid but warn about MUSCL incompatibility
            logger.info(f"JST scheme selected - MUSCL reconstruction will be automatically disabled for compatibility")
            normalized['convective_scheme'] = 'JST'
        else:
            # For all upwind schemes, prefer ROE as the most stable default
            normalized['convective_scheme'] = scheme
            logger.info(f"Upwind scheme {scheme} selected - MUSCL reconstruction will be automatically enabled")
        
        # Validate linear solver
        valid_solvers = ["FGMRES", "BCGSTAB", "GMRES", "RESTARTED_FGMRES"]
        if params.get('linear_solver', 'FGMRES') not in valid_solvers:
            logger.warning(f"Invalid linear solver {params.get('linear_solver')}, using FGMRES")
            normalized['linear_solver'] = 'FGMRES'
        
        # Validate wall type
        valid_wall_types = ["HEATFLUX", "ISOTHERMAL", "ADIABATIC"]
        if params.get('wall_type', 'HEATFLUX') not in valid_wall_types:
            logger.warning(f"Invalid wall type {params.get('wall_type')}, using HEATFLUX")
            normalized['wall_type'] = 'HEATFLUX'
        
        # Validate turbulence intensity
        if not (0.01 <= params.get('turbulence_intensity', 0.05) <= 0.15):
            logger.warning(f"Turbulence intensity {params.get('turbulence_intensity')} out of range, clamping to 0.01-0.15")
            normalized['turbulence_intensity'] = max(0.01, min(0.15, params.get('turbulence_intensity', 0.05)))
        
        # Validate reference length
        if not (0.1 <= params.get('reference_length', 1.0) <= 10.0):
            logger.warning(f"Reference length {params.get('reference_length')} out of range, clamping to 0.1-10.0 m")
            normalized['reference_length'] = max(0.1, min(10.0, params.get('reference_length', 1.0)))
        
        # Validate reference area
        if not (0.01 <= params.get('reference_area', 1.0) <= 100.0):
            logger.warning(f"Reference area {params.get('reference_area')} out of range, clamping to 0.01-100.0 m²")
            normalized['reference_area'] = max(0.01, min(100.0, params.get('reference_area', 1.0)))
        
        # Validate boundary markers (ensure they are lists)
        if not isinstance(params.get('wall_markers', []), list):
            logger.warning("Wall markers not a list, using default ['wall']")
            normalized['wall_markers'] = ['wall']
        
        if not isinstance(params.get('farfield_markers', []), list):
            logger.warning("Farfield markers not a list, using default ['farfield']")
            normalized['farfield_markers'] = ['farfield']
        
        # Ensure optional markers are lists (can be empty)
        for marker_type in ['inlet_markers', 'outlet_markers', 'symmetry_markers']:
            if not isinstance(params.get(marker_type, []), list):
                logger.warning(f"{marker_type} not a list, using empty list")
                normalized[marker_type] = []
        
        # Validate and intelligently adjust max_iterations based on flow complexity
        max_iter = params.get('max_iterations', 200)
        solver_type = params.get('solver_type', 'RANS')
        aoa = abs(params.get('angle_of_attack', 0))
        mach = params.get('mach_number', 0.15)
        reynolds = params.get('reynolds_number', 1e5)
        
        # Calculate intelligent iteration count
        if solver_type == "EULER":
            base_iterations = 150  # EULER converges faster
            if aoa <= 3:
                base_iterations = 120
            elif aoa > 5:
                base_iterations = 180
        else:  # RANS or NAVIER_STOKES
            base_iterations = 250  # RANS needs more iterations for turbulence
            if aoa <= 3:
                base_iterations = 200
            elif aoa > 5:
                base_iterations = 300
        
        # Add complexity factors
        complexity_bonus = 0
        if mach > 0.15:
            complexity_bonus += 75  # Higher Mach
        if reynolds > 5e4:
            complexity_bonus += 50  # Higher Reynolds
        if aoa > 5:
            complexity_bonus += 100  # High angle of attack
        
        # Check for complex geometry keywords in simulation description
        sim_description = params.get('simulation_description', '').lower()
        geometry_keywords = ['propeller', 'blade', 'car', 'f1', 'complex', 'vehicle', 'turbine']
        if any(keyword in sim_description for keyword in geometry_keywords):
            complexity_bonus += 150  # Complex geometry needs more iterations
            logger.info(f"Complex geometry detected in description: '{sim_description}' - adding 150 iterations")
        
        intelligent_iterations = base_iterations + complexity_bonus
        
        # RESPECT USER'S EXPLICIT REQUESTS: Always use the AI-extracted value (which should come from user prompt)
        # Only use intelligent calculation as fallback if no specific value was requested
        if 'max_iterations' in params:
            # User/AI provided a specific value - always respect it
            normalized['max_iterations'] = max_iter
            logger.info(f"Using user-specified/AI-extracted max_iterations: {max_iter}")
        else:
            # No specific value provided - use intelligent calculation
            normalized['max_iterations'] = intelligent_iterations
            logger.info(f"No specific max_iterations provided, using intelligent calculation: {intelligent_iterations}")
        
        # Ensure max_iterations is within bounds (50-1000)
        normalized['max_iterations'] = max(50, min(1000, normalized['max_iterations']))
        
        # Validate wind tunnel orientation
        valid_orientations = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
        if params.get('wind_tunnel_orientation', '+X') not in valid_orientations:
            logger.warning(f"Invalid wind tunnel orientation {params.get('wind_tunnel_orientation')}, using +X")
            normalized['wind_tunnel_orientation'] = '+X'
        
        # Validate solver type
        valid_solvers = ["EULER", "NAVIER_STOKES", "RANS"]
        if params.get('solver_type', 'RANS') not in valid_solvers:
            logger.warning(f"Invalid solver type {params.get('solver_type')}, using RANS")
            normalized['solver_type'] = 'RANS'
        
        # Validate turbulence model
        valid_turb_models = ["NONE", "SA", "SST"]
        if params.get('turbulence_model', 'SA') not in valid_turb_models:
            logger.warning(f"Invalid turbulence model {params.get('turbulence_model')}, using SA")
            normalized['turbulence_model'] = 'SA'
        
        # Ensure turbulence model consistency with solver
        if normalized['solver_type'] == 'EULER' and normalized['turbulence_model'] != 'NONE':
            logger.warning("EULER solver requires NONE turbulence model, correcting")
            normalized['turbulence_model'] = 'NONE'
        elif normalized['solver_type'] in ['RANS', 'NAVIER_STOKES'] and normalized['turbulence_model'] == 'NONE':
            logger.warning("RANS/NAVIER_STOKES solver should use turbulence model, setting to SA")
            normalized['turbulence_model'] = 'SA'
        
        logger.info(f"Parameter validation completed")
        logger.info(f"Temperature: {normalized['temperature']} K, Pressure: {normalized['pressure']} Pa")
        logger.info(f"CFL: {normalized['cfl_number']}, Scheme: {normalized['convective_scheme']}")
        logger.info(f"Reference Length: {normalized['reference_length']} m, Area: {normalized['reference_area']} m²")
        logger.info(f"Wall markers: {normalized['wall_markers']}, Farfield: {normalized['farfield_markers']}")
        
        return normalized
    
    def determine_mesh_file(self, mesh_file_name: str) -> str:
        """Determine which mesh file to use"""
        
        if mesh_file_name == "auto":
            # Default to Project3 for propeller simulations since that's our validated mesh
            return "Project3/5_bladed_Propeller_medium_tetrahedral.su2"
        
        # Check if the mesh file name matches any of our available meshes
        mesh_mappings = {
            "original": "project1/original_medium_tetrahedral.su2",
            "airfoil": "project1/original_medium_tetrahedral.su2",
            "eppler": "Project2/Eppler 1230_medium_tetrahedral.su2",
            "propeller": "Project3/5_bladed_Propeller_medium_tetrahedral.su2",
            "blade": "Project3/5_bladed_Propeller_medium_tetrahedral.su2", 
            "5_bladed": "Project3/5_bladed_Propeller_medium_tetrahedral.su2",
            "project3": "Project3/5_bladed_Propeller_medium_tetrahedral.su2",
            "project1": "project1/original_medium_tetrahedral.su2",
            "project2": "Project2/Eppler 1230_medium_tetrahedral.su2"
        }
        
        mesh_lower = mesh_file_name.lower()
        for key, path in mesh_mappings.items():
            if key in mesh_lower:
                return path
        
        # If no match found, use default
        logger.warning(f"Mesh file '{mesh_file_name}' not recognized, using default")
        return "project1/original_medium_tetrahedral.su2"
    
    def create_simulation_from_prompt(self, user_prompt: str) -> str:
        """Complete workflow: prompt -> parameters -> simulation"""
        
        logger.info(f"Processing user prompt: {user_prompt}")
        
        # Step 1: Extract parameters using Gemini
        print("🤖 Analyzing prompt with Gemini 2.0 Flash...")
        raw_params = self.extract_parameters(user_prompt)
        
        # Step 2: Validate and normalize parameters
        print("✅ Validating parameters...")
        params = self.validate_and_normalize_parameters(raw_params)
        
        # Step 3: Determine mesh file
        mesh_file = self.determine_mesh_file(params['mesh_file_name'])
        print(f"📄 Using mesh file: {mesh_file}")
        
        # Step 4: Convert orientation string to enum
        orientation_map = {
            "+X": WindTunnelOrientation.POSITIVE_X,
            "-X": WindTunnelOrientation.NEGATIVE_X,
            "+Y": WindTunnelOrientation.POSITIVE_Y,
            "-Y": WindTunnelOrientation.NEGATIVE_Y,
            "+Z": WindTunnelOrientation.POSITIVE_Z,
            "-Z": WindTunnelOrientation.NEGATIVE_Z
        }
        orientation = orientation_map[params['wind_tunnel_orientation']]
        
        # Step 5: Create configuration
        print("⚙️ Creating CFD configuration...")
        config = create_config_with_extracted_markers(
            mesh_file_path=mesh_file,
            mach_number=params['mach_number'],
            reynolds_number=params['reynolds_number'],
            angle_of_attack=params['angle_of_attack'],
            max_iterations=params['max_iterations'],
            wind_tunnel_orientation=orientation,
            solver_type=params['solver_type'],
            turbulence_model=params['turbulence_model']
        )
        
        # Step 6: Create simulation
        print("📁 Creating simulation...")
        simulation_name = f"ai_generated_{params['simulation_description'].replace(' ', '_').lower()}"
        sim_dir = self.wind_tunnel_sim.create_simulation(config, simulation_name)
        
        # Step 7: Save AI metadata
        ai_metadata = {
            "user_prompt": user_prompt,
            "extracted_parameters": raw_params,
            "normalized_parameters": params,
            "mesh_file": mesh_file,
            "generated_by": "CFD Parameter Agent",
            "model": "gemini-2.0-flash-exp"
        }
        
        ai_metadata_file = os.path.join(sim_dir, "ai_metadata.json")
        with open(ai_metadata_file, 'w') as f:
            json.dump(ai_metadata, f, indent=2)
        
        return sim_dir
    
    def run_simulation_from_prompt(self, user_prompt: str) -> bool:
        """Create and run simulation from user prompt"""
        
        sim_dir = self.create_simulation_from_prompt(user_prompt)
        
        print("🚀 Running SU2 simulation...")
        success = self.wind_tunnel_sim.run_simulation(sim_dir)
        
        if success:
            print("✅ Simulation completed successfully!")
        else:
            print("❌ Simulation failed!")
        
        return success

def main():
    """Example usage of the CFD Parameter Agent"""
    
    # Initialize agent with API key
    api_key = "AIzaSyB2mzctXXTAK8RRc5IHaKJ87b9inm4x9A4"
    agent = CFDParameterAgent(api_key)
    
    # Example prompts
    example_prompts = [
        "Analyze a propeller at 10 degrees angle of attack with high Reynolds number",
        "Low speed airfoil simulation for wind tunnel testing",
        "High Mach number flow over an airfoil at cruise conditions",
        "Eppler airfoil performance at 5 degrees with moderate Reynolds number"
    ]
    
    print("🤖 CFD Parameter Agent Ready!")
    print("=" * 60)
    
    # Interactive mode
    while True:
        print("\nOptions:")
        print("1. Enter custom prompt")
        print("2. Use example prompt") 
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            user_prompt = input("\nEnter your CFD simulation request: ").strip()
            if user_prompt:
                try:
                    sim_dir = agent.create_simulation_from_prompt(user_prompt)
                    print(f"\n✅ Simulation created: {sim_dir}")
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    
        elif choice == "2":
            print("\nExample prompts:")
            for i, prompt in enumerate(example_prompts, 1):
                print(f"{i}. {prompt}")
            
            try:
                example_choice = int(input("\nSelect example (1-4): ")) - 1
                if 0 <= example_choice < len(example_prompts):
                    user_prompt = example_prompts[example_choice]
                    print(f"\nUsing prompt: {user_prompt}")
                    sim_dir = agent.create_simulation_from_prompt(user_prompt)
                    print(f"\n✅ Simulation created: {sim_dir}")
                else:
                    print("Invalid selection")
            except Exception as e:
                print(f"\n❌ Error: {e}")
                
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid option")

if __name__ == "__main__":
    main() 