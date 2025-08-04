#!/usr/bin/env python3
"""
JAX-Fluids Agent

Simple, clean agent that converts user prompts into JAX-Fluids simulation configurations
using Gemini 2.5 Pro and generates complete simulation setups.
"""

import os
import json
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import google.generativeai as genai

class JAXFluidsAgent:
    """Intelligent agent for converting user prompts to JAX-Fluids configurations"""
    
    def __init__(self, api_key: str = None):
        """Initialize the agent with Gemini API key"""
        if not api_key:
            api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be provided or set in environment")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        
        print("JAX-Fluids Agent Initialized")
        print("   Gemini 2.5 Pro ready")
        print("   External flow expertise loaded")
    
    def get_system_prompt(self) -> str:
        """System prompt for JAX-Fluids parameter extraction"""
        return """You are a JAX-Fluids CFD expert agent. Convert user prompts into complete JAX-Fluids simulation configurations for external flow around 3D objects with immersed boundaries.

**REQUIRED OUTPUT:**
Respond with TWO separate JSON objects: numerical_setup and case_setup.

**NUMERICAL_SETUP JSON:**
```json
{
    "general": {
        "setup_type": "incompressible" | "compressible",
        "equation_system": "NAVIER_STOKES" | "EULER",
        "precision": "DOUBLE_PRECISION"
    },
    "space_discretization": {
        "derivative_stencil": "CENTRAL_4" | "CENTRAL_6",
        "reconstruction_stencil": "WENO5-JS" | "WENO7-JS",
        "convective_flux": "GODUNOV",
        "riemann_solver": "HLL" | "HLLC" | "ROE"
    },
    "time_discretization": {
        "time_integrator": "RK3" | "RK4",
        "time_step_kind": "CFL",
        "CFL": 0.3-0.5,
        "max_timesteps": 50000-200000,
        "time_end": 5.0-10.0
    },
    "output": {
        "output_period": 100-1000,
        "output_timestamps": [0.0, 1.0, 5.0],
        "output_quantities": ["velocity", "pressure", "density"],
        "output_format": "HDF5"
    }
}
```

**CASE_SETUP JSON:**
```json
{
    "domain": {
        "x": [-10.0, 30.0],
        "y": [-10.0, 10.0], 
        "z": [-10.0, 10.0]
    },
    "resolution": [200, 100, 100],
    "boundary_conditions": {
        "x_minus": {
            "type": "DIRICHLET",
            "primitive_variables": {
                "velocity": [50.0, 0.0, 0.0],
                "pressure": 101325.0
            }
        },
        "x_plus": {"type": "NEUMANN_ZERO_GRADIENT"},
        "y_minus": {"type": "SYMMETRY"},
        "y_plus": {"type": "SYMMETRY"},
        "z_minus": {"type": "SYMMETRY"},
        "z_plus": {"type": "SYMMETRY"}
    },
    "initial_conditions": {
        "velocity": [50.0, 0.0, 0.0],
        "pressure": 101325.0
    },
    "materials": {
        "positive_levelset": {
            "type": "FLUID",
            "equation_of_state": "INCOMPRESSIBLE",
            "density": 1.225,
            "dynamic_viscosity": 1.81e-5
        }
    },
    "levelset": {
        "model": "FLUID_SOLID_LEVELSET",
        "initialization": "FROM_SDF_FILE"
    }
}
```

**GUIDELINES:**
- For subsonic (Mach < 0.3): use incompressible, NAVIER_STOKES
- For compressible (Mach > 0.3): use compressible, EULER
- High accuracy: CENTRAL_6 + WENO7 + RK4
- Standard accuracy: CENTRAL_4 + WENO5 + RK3
- Extract velocity from Mach number: v = Mach * 343 m/s
- Adjust domain size based on object scale
- Wind tunnel: inlet at x_minus, outlet at x_plus, symmetry on sides

Analyze the user prompt and output both JSON objects separated by "---CASE_SETUP---".
"""
    
    def extract_parameters(self, user_prompt: str) -> Dict[str, Any]:
        """Extract JAX-Fluids parameters from user prompt"""
        
        print("Analyzing user prompt for JAX-Fluids parameters...")
        
        full_prompt = f"{self.get_system_prompt()}\n\nUser Request: {user_prompt}"
        
        try:
            response = self.model.generate_content(full_prompt)
            response_text = response.text.strip()
            
            # Split numerical and case setup
            if "---CASE_SETUP---" in response_text:
                parts = response_text.split("---CASE_SETUP---")
                numerical_text = parts[0].strip()
                case_text = parts[1].strip()
            else:
                # Fallback: try to find JSON blocks
                import re
                json_blocks = re.findall(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if len(json_blocks) >= 2:
                    numerical_text = json_blocks[0]
                    case_text = json_blocks[1]
                else:
                    raise ValueError("Could not parse two JSON configurations from response")
            
            # Clean and parse JSON
            numerical_text = self._clean_json_text(numerical_text)
            case_text = self._clean_json_text(case_text)
            
            numerical_setup = json.loads(numerical_text)
            case_setup = json.loads(case_text)
            
            print("Successfully extracted JAX-Fluids parameters")
            return {
                'numerical_setup': numerical_setup,
                'case_setup': case_setup,
                'user_prompt': user_prompt
            }
            
        except Exception as e:
            print(f"Error extracting parameters: {e}")
            return self._get_fallback_config(user_prompt)
    
    def _clean_json_text(self, text: str) -> str:
        """Clean JSON text from markdown formatting"""
        # Remove markdown code blocks
        text = text.replace('```json', '').replace('```', '')
        # Find JSON object boundaries
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            return text[start:end]
        return text.strip()
    
    def _get_fallback_config(self, user_prompt: str) -> Dict[str, Any]:
        """Fallback configuration if AI extraction fails"""
        print("Using fallback configuration")
        
        # Determine if compressible based on prompt
        is_compressible = any(term in user_prompt.lower() 
                            for term in ['mach', 'supersonic', 'compressible', 'shock'])
        
        numerical_setup = {
            "general": {
                "setup_type": "compressible" if is_compressible else "incompressible",
                "equation_system": "EULER" if is_compressible else "NAVIER_STOKES",
                "precision": "DOUBLE_PRECISION"
            },
            "space_discretization": {
                "derivative_stencil": "CENTRAL_4",
                "reconstruction_stencil": "WENO5-JS",
                "convective_flux": "GODUNOV",
                "riemann_solver": "HLLC"
            },
            "time_discretization": {
                "time_integrator": "RK3",
                "time_step_kind": "CFL",
                "CFL": 0.4,
                "max_timesteps": 100000,
                "time_end": 5.0
            },
            "output": {
                "output_period": 500,
                "output_timestamps": [0.0, 1.0, 5.0],
                "output_quantities": ["velocity", "pressure"],
                "output_format": "HDF5"
            }
        }
        
        case_setup = {
            "domain": {"x": [-10.0, 30.0], "y": [-10.0, 10.0], "z": [-10.0, 10.0]},
            "resolution": [200, 100, 100],
            "boundary_conditions": {
                "x_minus": {
                    "type": "DIRICHLET",
                    "primitive_variables": {"velocity": [50.0, 0.0, 0.0], "pressure": 101325.0}
                },
                "x_plus": {"type": "NEUMANN_ZERO_GRADIENT"},
                "y_minus": {"type": "SYMMETRY"},
                "y_plus": {"type": "SYMMETRY"},
                "z_minus": {"type": "SYMMETRY"},
                "z_plus": {"type": "SYMMETRY"}
            },
            "initial_conditions": {"velocity": [50.0, 0.0, 0.0], "pressure": 101325.0},
            "materials": {
                "positive_levelset": {
                    "type": "FLUID",
                    "equation_of_state": "INCOMPRESSIBLE",
                    "density": 1.225,
                    "dynamic_viscosity": 1.81e-5
                }
            },
            "levelset": {"model": "FLUID_SOLID_LEVELSET", "initialization": "FROM_SDF_FILE"}
        }
        
        return {
            'numerical_setup': numerical_setup,
            'case_setup': case_setup,
            'user_prompt': user_prompt
        }
    
    def generate_run_script(self, simulation_name: str) -> str:
        """Generate JAX-Fluids run script"""
        return f'''#!/usr/bin/env python3
"""
JAX-Fluids External Flow Simulation
Generated by JAX-Fluids Agent

Simulation: {simulation_name}
"""

import jax
import jax.numpy as jnp
from jaxfluids import InputManager, Initializer, SimulationManager
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run JAX-Fluids simulation"""
    
    print("Starting JAX-Fluids External Flow Simulation")
    print(f"Simulation: {simulation_name}")
    
    try:
        # Initialize JAX-Fluids
        logger.info("Initializing simulation...")
        
        input_manager = InputManager(
            case_setup_file="{simulation_name}.json",
            numerical_setup_file="numerical_setup.json"
        )
        
        initializer = Initializer(input_manager)
        simulation_manager = SimulationManager(input_manager)
        
        # Run simulation
        logger.info("Starting simulation...")
        start_time = time.time()
        
        simulation_manager.simulate()
        
        end_time = time.time()
        simulation_time = end_time - start_time
        
        logger.info(f"Simulation completed in {{simulation_time:.2f}} seconds")
        print("Simulation completed successfully!")
        print(f"Total time: {{simulation_time:.2f}} seconds")
        
    except Exception as e:
        logger.error(f"Simulation failed: {{e}}")
        print(f"Simulation failed: {{e}}")
        raise

if __name__ == "__main__":
    main()
'''

def generate_simulation(user_prompt: str, sdf_file: str = None, output_dir: str = None) -> Dict[str, Any]:
    """
    Main function to generate complete JAX-Fluids simulation
    
    Args:
        user_prompt: Natural language description of simulation
        sdf_file: Path to SDF file for immersed boundary
        output_dir: Directory to save simulation (default: creates new)
    
    Returns:
        Dict with simulation results
    """
    
    try:
        # Initialize agent
        agent = JAXFluidsAgent()
        
        # Extract parameters
        params = agent.extract_parameters(user_prompt)
        
        # Create simulation directory
        if not output_dir:
            timestamp = int(time.time())
            safe_name = "".join(c for c in user_prompt.lower().replace(" ", "_") 
                              if c.isalnum() or c in "_-")[:30]
            sim_name = f"jaxfluids_{timestamp}_{safe_name}"
            output_dir = f"../testing_externalflow/{sim_name}"
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save configurations
        numerical_file = Path(output_dir) / "numerical_setup.json"
        with open(numerical_file, 'w') as f:
            json.dump(params['numerical_setup'], f, indent=2)
        
        sim_name = Path(output_dir).name
        case_file = Path(output_dir) / f"{sim_name}.json"
        case_setup = params['case_setup']
        
        # Integrate SDF if provided
        if sdf_file:
            case_setup['levelset']['sdf_file'] = sdf_file
            print(f"Integrated SDF: {Path(sdf_file).name}")
        
        with open(case_file, 'w') as f:
            json.dump(case_setup, f, indent=2)
        
        # Generate run script
        run_script = agent.generate_run_script(sim_name)
        run_file = Path(output_dir) / "run.py"
        with open(run_file, 'w', encoding='utf-8') as f:
            f.write(run_script)
        
        # Generate README
        readme_content = f"""# {sim_name.title()}

## Generated JAX-Fluids External Flow Simulation

**User Request:** {user_prompt}

## Files
- `numerical_setup.json` - Numerical parameters
- `{sim_name}.json` - Case setup with domain and BCs
- `run.py` - Simulation runner script

## Run Simulation
```bash
cd {output_dir}
python run.py
```

## Configuration
- **Setup Type:** {params['numerical_setup']['general']['setup_type']}
- **Equation System:** {params['numerical_setup']['general']['equation_system']}
- **Resolution:** {params['case_setup']['resolution']}
- **SDF Integrated:** {'Yes' if sdf_file else 'No'}
"""
        
        readme_file = Path(output_dir) / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"Generated simulation: {output_dir}")
        print(f"Files created: {len(list(Path(output_dir).glob('*')))}")
        
        return {
            'success': True,
            'simulation_directory': str(output_dir),
            'simulation_name': sim_name,
            'files_generated': [
                str(numerical_file),
                str(case_file),
                str(run_file),
                str(readme_file)
            ],
            'sdf_integrated': sdf_file is not None,
            'parameters': params
        }
        
    except Exception as e:
        print(f"Simulation generation failed: {e}")
        return {
            'success': False,
            'error': str(e)
        } 