#!/usr/bin/env python3
"""
Adaptive JAX-Fluids Coding Agent
A true agentic system that understands JAX-Fluids physics and generates 
context-aware run.py scripts based on simulation requirements.

This agent can:
- Handle 1D/2D/3D simulations differently
- Adapt plotting based on physics (shock tubes, viscous flows, turbulence, etc.)
- Enable/disable visualization components
- Generate physics-specific post-processing
- Understand JAX-Fluids examples and adapt patterns
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

class SimulationType(Enum):
    """Different types of JAX-Fluids simulations"""
    SHOCK_TUBE = "shock_tube"
    VISCOUS_FLOW = "viscous_flow" 
    EXTERNAL_FLOW = "external_flow"
    TURBULENCE = "turbulence"
    HEAT_TRANSFER = "heat_transfer"
    ADVECTION = "advection"
    RIEMANN_PROBLEM = "riemann_problem"
    MULTIPHASE = "multiphase"

class PlottingMode(Enum):
    """Different plotting approaches"""
    MINIMAL = "minimal"           # Basic density/pressure only
    STANDARD = "standard"         # Standard flow quantities
    ADVANCED = "advanced"         # Full physics visualization
    RESEARCH = "research"         # Research-grade with custom analysis
    OFF = "off"                  # No plotting

@dataclass
class AgenticConfig:
    """Configuration for the adaptive agent"""
    
    # Simulation files
    case_file: str
    numerical_file: str
    
    # Agent instructions
    simulation_intent: str = ""  # e.g., "shock tube analysis", "airfoil drag study"
    plotting_mode: PlottingMode = PlottingMode.STANDARD
    enable_postprocess: bool = True
    
    # Dimensional preferences
    force_dimension: Optional[str] = None  # "1D", "2D", "3D" or None for auto-detect
    
    # Hardware settings
    cuda_device: str = "0"
    
    # Advanced settings
    custom_analysis: Optional[str] = None
    reference_data_path: Optional[str] = None
    
class AdaptiveJAXFluidsAgent:
    """
    Truly agentic JAX-Fluids code generator that understands physics
    and adapts scripts based on simulation requirements
    """
    
    def __init__(self, gemini_api_key: str = None):
        """Initialize the adaptive agent"""
        
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if self.gemini_api_key:
            self.model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=self.gemini_api_key,
                temperature=0.1,
                max_tokens=4096
            )
        else:
            self.model = None
            logger.warning("No Gemini API key found - using rule-based generation")
            
        # Load JAX-Fluids knowledge base
        self.examples_knowledge = self._build_examples_knowledge()
    
    def _build_examples_knowledge(self) -> Dict[str, Any]:
        """Build knowledge base from JAX-Fluids examples"""
        
        knowledge = {
            "1D": {
                "shock_tubes": {
                    "pattern": "Single quantity focus, 1D plots, shock tracking",
                    "quantities": ["density", "velocity", "pressure"],
                    "visualization": "create_1D_animation, create_1D_figure",
                    "analysis": "shock_position, compression_ratio"
                },
                "advection": {
                    "pattern": "Single scalar transport, periodic boundaries",
                    "quantities": ["density"],
                    "visualization": "create_1D_animation only"
                },
                "heat_transfer": {
                    "pattern": "Diffusion-dominated, temperature focus",
                    "quantities": ["density", "temperature", "pressure"],
                    "visualization": "create_1D_figure with temperature profile"
                }
            },
            "2D": {
                "external_flow": {
                    "pattern": "Immersed boundaries, masking, multiple quantities",
                    "quantities": ["density", "velocity", "pressure", "mach_number", "schlieren"],
                    "visualization": "create_2D_animation, create_2D_figure with masking"
                },
                "viscous_flow": {
                    "pattern": "Boundary layer focus, velocity profiles",
                    "quantities": ["density", "velocity", "pressure"],
                    "visualization": "1D line plots of velocity profiles"
                },
                "riemann_problems": {
                    "pattern": "Discontinuity evolution, contour plots",
                    "quantities": ["density", "velocity", "pressure"],
                    "visualization": "create_2D_animation with contours"
                }
            },
            "3D": {
                "turbulence": {
                    "pattern": "Energy analysis, velocity field focus, 2D slices",
                    "quantities": ["velocity"],
                    "visualization": "create_2D_animation of slices, energy analysis",
                    "analysis": "TKE, energy_dissipation, custom_energy_plots"
                },
                "external_flow": {
                    "pattern": "3D flow around objects, multiple planes",
                    "quantities": ["density", "velocity", "pressure", "mach_number"],
                    "visualization": "multiple plane animations"
                }
            }
        }
        
        return knowledge
    
    def analyze_simulation(
        self, 
        case_config: Dict[str, Any], 
        numerical_config: Dict[str, Any],
        config: AgenticConfig
    ) -> Dict[str, Any]:
        """Intelligently analyze simulation to determine characteristics"""
        
        # Determine dimension
        dimension = self._determine_dimension(case_config, config.force_dimension)
        
        # Classify simulation type
        sim_type = self._classify_simulation_type(case_config, numerical_config, config.simulation_intent)
        
        # Determine physics
        physics = self._analyze_physics(numerical_config)
        
        # Select appropriate quantities based on physics and intent
        quantities = self._select_quantities(sim_type, physics, dimension, config)
        
        # Determine visualization strategy
        viz_strategy = self._plan_visualization(sim_type, dimension, config.plotting_mode, physics)
        
        # Plan post-processing based on simulation type
        postprocess_plan = self._plan_postprocessing(sim_type, dimension, config, case_config)
        
        analysis = {
            'dimension': dimension,
            'simulation_type': sim_type,
            'physics': physics,
            'quantities': quantities,
            'visualization_strategy': viz_strategy,
            'postprocess_plan': postprocess_plan,
            'requires_masking': physics.get('levelset', False),
            'requires_custom_analysis': sim_type in [SimulationType.TURBULENCE, SimulationType.SHOCK_TUBE],
            'hardware_config': self._plan_hardware_config(dimension, config)
        }
        
        return analysis
    
    def _determine_dimension(self, case_config: Dict[str, Any], force_dim: Optional[str]) -> str:
        """Determine simulation dimension"""
        
        if force_dim:
            return force_dim
            
        domain = case_config.get('domain', {})
        active_dims = 0
        
        for dim in ['x', 'y', 'z']:
            if dim in domain:
                cells = domain[dim].get('cells', 1)
                if cells > 1:
                    active_dims += 1
        
        if active_dims <= 1:
            return "1D"
        elif active_dims <= 2:
            return "2D"
        else:
            return "3D"
    
    def _classify_simulation_type(
        self, 
        case_config: Dict[str, Any], 
        numerical_config: Dict[str, Any],
        intent: str
    ) -> SimulationType:
        """Intelligently classify the simulation type"""
        
        # Check for explicit intent keywords
        intent_lower = intent.lower()
        
        if any(word in intent_lower for word in ['shock', 'tube', 'riemann']):
            if 'riemann' in intent_lower or '2d' in intent_lower:
                return SimulationType.RIEMANN_PROBLEM
            return SimulationType.SHOCK_TUBE
            
        if any(word in intent_lower for word in ['turbulence', 'tgv', 'energy']):
            return SimulationType.TURBULENCE
            
        if any(word in intent_lower for word in ['external', 'airfoil', 'cylinder', 'flow around']):
            return SimulationType.EXTERNAL_FLOW
            
        if any(word in intent_lower for word in ['heat', 'temperature', 'thermal']):
            return SimulationType.HEAT_TRANSFER
            
        if any(word in intent_lower for word in ['advection', 'transport']):
            return SimulationType.ADVECTION
            
        # Analyze configuration for clues
        physics = numerical_config.get('active_physics', {})
        bc = case_config.get('boundary_conditions', {})
        
        # Check for levelset (external flow)
        if physics.get('is_levelset', False):
            return SimulationType.EXTERNAL_FLOW
            
        # Check for viscous flow
        if physics.get('is_viscous_flux', False):
            # Check boundary conditions for wall flows
            if any(bc_type.get('type') == 'NOSLIP' for bc_type in bc.values()):
                return SimulationType.VISCOUS_FLOW
            return SimulationType.TURBULENCE
            
        # Check for heat transfer
        if physics.get('is_heat_flux', False):
            return SimulationType.HEAT_TRANSFER
            
        # Check for shock-like initial conditions
        initial = case_config.get('initial_condition', {})
        if any('lambda' in str(val) and ('<=') in str(val) for val in initial.values()):
            return SimulationType.SHOCK_TUBE
            
        # Default fallback
        return SimulationType.EXTERNAL_FLOW
    
    def _analyze_physics(self, numerical_config: Dict[str, Any]) -> Dict[str, bool]:
        """Analyze active physics"""
        
        active_physics = numerical_config.get('active_physics', {})
        
        return {
            'viscous': active_physics.get('is_viscous_flux', False),
            'heat': active_physics.get('is_heat_flux', False),
            'levelset': active_physics.get('is_levelset', False),
            'convective': active_physics.get('is_convective_flux', True),
            'multiphase': active_physics.get('is_multiphase', False),
        }
    
    def _select_quantities(
        self, 
        sim_type: SimulationType, 
        physics: Dict[str, bool], 
        dimension: str,
        config: AgenticConfig
    ) -> List[str]:
        """Intelligently select output quantities based on simulation type"""
        
        base_quantities = ["density", "velocity", "pressure"]
        
        if sim_type == SimulationType.SHOCK_TUBE:
            # Shock tubes focus on basic quantities
            return base_quantities
            
        elif sim_type == SimulationType.TURBULENCE:
            # Turbulence needs velocity components
            return ["velocity", "pressure"]
            
        elif sim_type == SimulationType.EXTERNAL_FLOW:
            # External flow needs comprehensive set
            quantities = base_quantities.copy()
            if physics['levelset']:
                quantities.extend(["levelset", "volume_fraction"])
            quantities.extend(["mach_number", "schlieren"])
            return quantities
            
        elif sim_type == SimulationType.HEAT_TRANSFER:
            # Heat transfer needs temperature
            quantities = base_quantities + ["temperature"]
            return quantities
            
        elif sim_type == SimulationType.ADVECTION:
            # Advection often just needs density
            return ["density"]
            
        elif sim_type == SimulationType.VISCOUS_FLOW:
            # Viscous flow needs velocity focus
            return base_quantities
            
        else:
            return base_quantities
    
    def _plan_visualization(
        self, 
        sim_type: SimulationType, 
        dimension: str, 
        plotting_mode: PlottingMode,
        physics: Dict[str, bool]
    ) -> Dict[str, Any]:
        """Plan visualization strategy based on simulation type and requirements"""
        
        if plotting_mode == PlottingMode.OFF:
            return {"enabled": False}
            
        strategy = {
            "enabled": True,
            "functions": [],
            "plots": {},
            "custom_analysis": False
        }
        
        # Dimensional strategy
        if dimension == "1D":
            strategy["functions"] = ["create_1D_animation", "create_1D_figure"]
            
            if sim_type == SimulationType.SHOCK_TUBE:
                strategy["plots"] = {
                    "density": "shock profile",
                    "velocity": "particle velocity", 
                    "pressure": "pressure jump"
                }
                strategy["layout"] = "(1,3)"
                
        elif dimension == "2D":
            if sim_type == SimulationType.EXTERNAL_FLOW and physics.get('levelset'):
                strategy["functions"] = ["create_2D_animation", "create_2D_figure"]
                strategy["plots"] = {
                    "density": "masked density field",
                    "pressure": "masked pressure field",
                    "mach_number": "clipped mach field",
                    "schlieren": "schlieren visualization"
                }
                strategy["layout"] = "(2,2)"
                strategy["masking"] = True
                
            elif sim_type == SimulationType.VISCOUS_FLOW:
                strategy["functions"] = ["create_1D_animation"]  # Profile plots
                strategy["plots"] = {
                    "velocity_profile": "boundary layer profile"
                }
                strategy["custom_analysis"] = True
                
        elif dimension == "3D":
            strategy["functions"] = ["create_2D_animation"]  # 2D slices
            
            if sim_type == SimulationType.TURBULENCE:
                strategy["plots"] = {
                    "u": "x-velocity component",
                    "v": "y-velocity component", 
                    "w": "z-velocity component"
                }
                strategy["layout"] = "(1,3)"
                strategy["custom_analysis"] = True
                strategy["energy_analysis"] = True
        
        # Adjust for plotting mode
        if plotting_mode == PlottingMode.MINIMAL:
            strategy["plots"] = {"density": "density field", "pressure": "pressure field"}
            strategy["layout"] = "(1,2)"
            
        return strategy
    
    def _plan_postprocessing(
        self, 
        sim_type: SimulationType, 
        dimension: str,
        config: AgenticConfig,
        case_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Plan post-processing based on simulation type"""
        
        if not config.enable_postprocess:
            return {"enabled": False}
            
        plan = {
            "enabled": True,
            "load_quantities": True,
            "custom_calculations": [],
            "save_data": False
        }
        
        if sim_type == SimulationType.TURBULENCE:
            plan["custom_calculations"] = [
                "kinetic_energy_calculation",
                "energy_dissipation_rate",
                "reference_comparison"
            ]
            
        elif sim_type == SimulationType.SHOCK_TUBE:
            plan["custom_calculations"] = [
                "shock_position_tracking",
                "compression_ratio"
            ]
            
        elif sim_type == SimulationType.EXTERNAL_FLOW:
            plan["custom_calculations"] = [
                "masking_setup",
                "force_calculation"
            ]
            
        return plan
    
    def _plan_hardware_config(self, dimension: str, config: AgenticConfig) -> Dict[str, str]:
        """Plan hardware configuration based on simulation size"""
        
        if dimension == "3D":
            return {"cuda_device": config.cuda_device or "0"}
        elif dimension == "2D":
            return {"cuda_device": config.cuda_device or "0"}
        else:
            return {"cuda_device": ""}  # CPU for 1D
    
    def generate_adaptive_script(
        self,
        config: AgenticConfig,
        case_config: Dict[str, Any],
        numerical_config: Dict[str, Any],
        output_path: str
    ) -> str:
        """Generate an adaptive JAX-Fluids script based on intelligent analysis"""
        
        print(f"🤖 ADAPTIVE JAX-FLUIDS AGENT")
        print(f"🎯 Intent: {config.simulation_intent}")
        print(f"📊 Plotting mode: {config.plotting_mode.value}")
        
        # Intelligent analysis
        analysis = self.analyze_simulation(case_config, numerical_config, config)
        
        print(f"🔍 Detected: {analysis['dimension']} {analysis['simulation_type'].value}")
        print(f"🧮 Physics: {analysis['physics']}")
        print(f"📈 Quantities: {analysis['quantities']}")
        
        # Generate script content using AI ONLY - no fallbacks for high-stakes
        if not self.model or not self.gemini_api_key:
            raise RuntimeError("❌ MISSION CRITICAL ERROR: No AI model available. High-stakes applications like SpaceX/Tesla require full AI reasoning. Gemini API key is mandatory.")
            
        script_content = self._generate_with_ai(config, analysis, case_config, numerical_config)
        
        # Write to file
        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
            
        # Make executable
        try:
            os.chmod(output_file, 0o755)
        except:
            pass
            
        print(f"✅ Generated adaptive script: {output_file}")
        return str(output_file)
    
    def _generate_with_ai(
        self, 
        config: AgenticConfig, 
        analysis: Dict[str, Any],
        case_config: Dict[str, Any],
        numerical_config: Dict[str, Any]
    ) -> str:
        """Generate script using AI with JAX-Fluids knowledge"""
        
        prompt = f"""
You are a JAX-Fluids expert with comprehensive knowledge of the complete library API and capabilities.

**COMPREHENSIVE JAX-FLUIDS API KNOWLEDGE:**

 **CORE CLASSES:**
 - InputManager(case_file, numerical_file): Loads and validates configurations
 - InitializationManager(input_manager): Handles simulation initialization
 - SimulationManager(input_manager): Manages simulation execution
 
 **REQUIRED IMPORTS:**
 ```python
 import os
 import json
 import glob
 from jaxfluids import InputManager, InitializationManager, SimulationManager
 ```

**SIMULATION EXECUTION PATTERNS:**
```python
from jaxfluids import InputManager, InitializationManager, SimulationManager

# Standard pattern
input_manager = InputManager(case_file, numerical_file)
initialization_manager = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)
buffers = initialization_manager.initialization()
sim_manager.simulate(buffers)
```

**ADVANCED FEATURES:**
- 1D/2D/3D simulations with automatic dimension detection
- Multiple boundary condition types (DIRICHLET, ZEROGRADIENT, SYMMETRY, WALL, etc.)
- Immersed boundary method with levelset
- Multiple equation of state models
- Viscous and inviscid flows
- Heat transfer capabilities
- Advanced numerical schemes (WENO, Riemann solvers)
- Parallel execution support

**OUTPUT HANDLING:**
- Primitive variables: density, velocity, pressure, temperature
- Derived quantities: mach_number, schlieren, vorticity
- Levelset quantities: volume_fraction, levelset, normal
- Custom post-processing and visualization

 **CONFIGURATION OPTIMIZATION:**
 JAX-Fluids configurations may need optimization for stability:
 - Single device decomposition for development
 - Proper output configuration to avoid internal bugs
 - Correct nondimensionalization parameters
 - Appropriate boundary condition setup
 
 **SIMULATION TIME INTELLIGENCE:**
 Choose simulation parameters based on physics and user intent:
 - Quick tests/debugging: 0.001-0.1 time units (hundreds of iterations)
 - Development/validation: 1-10 time units (thousands of iterations)  
 - Production CFD: 10-1000+ time units (tens of thousands to millions of iterations)
 - Research/high-fidelity: 100-10000+ time units (millions+ iterations)
 
 Consider flow physics:
 - Transient phenomena: Longer times to capture evolution
 - Steady-state convergence: Time for flow to develop and stabilize
 - External flow: Time for wake development and force convergence
 - Shock interactions: Time for wave propagation across domain

SIMULATION CONTEXT:
- Type: {analysis['simulation_type'].value} ({analysis['dimension']})
- Intent: {config.simulation_intent}
- Plotting Mode: {config.plotting_mode.value}
- Physics: {analysis['physics']}

Generate a complete JAX-Fluids script with:
- VectraSim header and proper error handling  
- Production optimization function for configuration stability
- Proper JAX-Fluids API usage patterns
- Configuration files: {config.case_file} and {config.numerical_file}

Include this optimization function:
```python
def modify_config_for_production(case_file: str) -> str:
    \"\"\"Apply VectraSim's production optimizations for stable JAX-Fluids execution\"\"\"
    
    print("🔧 Applying VectraSim production optimizations...")
    
    # Load original config
    with open(case_file, 'r', encoding='utf-8') as f:
        case_config = json.load(f)
    
    # INTELLIGENT OUTPUT CONFIGURATION (based on proven working solution)
    if 'output' in case_config:
        # Fix common field name errors that cause crashes
        if 'miscellaneous' in case_config['output']:
            misc_fields = case_config['output']['miscellaneous']
            if 'q_criterion' in misc_fields:
                misc_fields[misc_fields.index('q_criterion')] = 'qcriterion'
                print("🔧 Fixed field name: q_criterion -> qcriterion")
        
        # PROVEN WORKING OUTPUT STRATEGY (from successful 116-step run)
        # Key insight: levelset output fields cause NoneType error even with complex SDF
        # Always use essential fields only and remove levelset output for stability
        case_config['output']['primitives'] = ['density', 'velocity', 'pressure', 'temperature']
        case_config['output']['miscellaneous'] = ['mach_number']  # Fixed field name above
        case_config['output']['levelset'] = []  # CRITICAL: Remove to avoid NoneType error
        case_config['output']['conservatives'] = []  # Keep minimal
        
        print("🔧 Applied proven working output strategy (essential fields, no levelset output)")
    else:
        # Fallback if no output section exists
        case_config['output'] = {{
            "primitives": ['density', 'velocity', 'pressure', 'temperature'],
            "miscellaneous": ['mach_number'],
            "levelset": []
        }}
        print("🔧 Created essential output configuration")
    
    # Ensure single device decomposition for development/testing
    if 'domain' in case_config and 'decomposition' in case_config['domain']:
        case_config['domain']['decomposition'] = {{
            'split_x': 1,
            'split_y': 1, 
            'split_z': 1
        }}
    
    # Ensure proper nondimensionalization for stable numerics
    if 'nondimensionalization_parameters' in case_config:
        nondim = case_config['nondimensionalization_parameters']
        if 'length_reference' not in nondim or nondim['length_reference'] == 0:
            nondim['length_reference'] = 1.0  # Proper length reference
    
    # Add gravity if missing (required by JAX-Fluids)
    if 'forcings' not in case_config:
        case_config['forcings'] = {{'gravity': [0.0, 0.0, 0.0]}}
    
    # INTELLIGENT SIMULATION TIMING (based on proven working solution)
    if 'general' in case_config:
        # Create results directory
        os.makedirs("./results", exist_ok=True)
        case_config['general']['save_path'] = "./results"
        
        # Detect execution environment
        is_hpcc = any(env_var in os.environ for env_var in ['SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID'])
        
        if is_hpcc:
            # HPCC: Preserve AI agent's intelligent timing decisions
            current_end_time = case_config['general'].get('end_time', 1.0)
            if current_end_time > 1000.0:
                case_config['general']['save_dt'] = current_end_time / 100
            elif current_end_time > 10.0:
                case_config['general']['save_dt'] = current_end_time / 50
            else:
                case_config['general']['save_dt'] = current_end_time / 10
            print("🔧 HPCC environment: Preserved intelligent timing")
        else:
            # Local environment: Use proven 100+ timestep approach
            case_config['general']['end_time'] = 20.0    # Proven: ~116 timesteps with dt ≈ 0.173
            case_config['general']['save_dt'] = 2.0      # Proven: ~10 snapshots for good resolution
            print("🔧 Local environment: Applied proven 100+ timestep timing (20.0 end_time)")
        
        # Apply proven stable mesh size for local development
        if not is_hpcc and 'domain' in case_config:
            # Check if mesh is very large and scale down for local stability
            x_cells = case_config['domain'].get('x', {{}}).get('cells', 64)
            y_cells = case_config['domain'].get('y', {{}}).get('cells', 64)
            z_cells = case_config['domain'].get('z', {{}}).get('cells', 64)
            total_cells = x_cells * y_cells * z_cells
            
            if total_cells > 300000:  # > 0.3M cells might be unstable locally
                case_config['domain']['x']['cells'] = 64
                case_config['domain']['y']['cells'] = 64
                case_config['domain']['z']['cells'] = 64
                print("🔧 Local environment: Scaled mesh to proven stable 64x64x64")
    
    # PRESERVE INTELLIGENT SDF GEOMETRY (based on actual working solution)
    # The working approach was to KEEP the complex SDF and just fix output configuration
    if 'initial_condition' in case_config and 'levelset' in case_config['initial_condition']:
        levelset_value = case_config['initial_condition']['levelset']
        if 'sdf' in str(levelset_value).lower() or 'CUSTOM_SDF' in str(levelset_value):
            # Always preserve the intelligent SDF setup - this is what actually worked
            config_dir = os.path.dirname(case_file)
            sdf_pattern = os.path.join(config_dir, 'sdf_data', '*', '*_sdf_matrix.npy')
            sdf_files = glob.glob(sdf_pattern)
            
            if sdf_files:
                sdf_file = sdf_files[0]
                rel_sdf_path = os.path.relpath(sdf_file, config_dir)
                case_config['initial_condition']['levelset'] = f'CUSTOM_SDF({{rel_sdf_path}})'
                print(f"🔧 Preserved intelligent SDF geometry: {{{{rel_sdf_path}}}}")
            else:
                # Keep the existing SDF reference - don't force sphere geometry
                print("🔧 Preserved existing SDF configuration (no external file needed)")
    
    # Write optimized config
    optimized_file = case_file.replace('.json', '_production_optimized.json')
    with open(optimized_file, 'w', encoding='utf-8') as f:
        json.dump(case_config, f, indent=2)
    
    print(f"✅ VectraSim production optimizations applied: {{os.path.basename(optimized_file)}}")
    return optimized_file
```

MANDATORY HEADER FORMAT:
```python
#!/usr/bin/env python3
\"\"\"
VectraSim Intelligent Simulation Suite
AI-Generated JAX-Fluids Script - Mission Critical

Generated for: {analysis['simulation_type'].value} ({analysis['dimension']})
Simulation Intent: {config.simulation_intent}
Plotting Mode: {config.plotting_mode.value}

This script was automatically generated by VectraSim's AI agent
based on JAX-Fluids (Apache 2.0 licensed) for computational fluid dynamics.

VectraSim - Advanced Computational Physics Platform
\"\"\"
```

 Use EXACTLY this proven JAX-Fluids pattern - no modifications:
 ```python
def run_simulation(case_file: str, numerical_file: str):
    \"\"\"Runs the JAX-Fluids simulation with enhanced error handling.\"\"\"
    try:
        optimized_case_file = modify_config_for_production(case_file)
        
        input_manager = InputManager(optimized_case_file, numerical_file)
        initialization_manager = InitializationManager(input_manager)
        sim_manager = SimulationManager(input_manager)
        buffers = initialization_manager.initialization()
        sim_manager.simulate(buffers)  # PROVEN WORKING - do NOT use advance()
        print("🎉 Simulation completed successfully! Check ./results/ for .h5 files")

    except FileNotFoundError as e:
        print(f"❌ Error: Configuration file not found: {{e}}")
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format in configuration file: {{e}}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {{e}}")

if __name__ == "__main__":
    case_file = "{config.case_file}"
    numerical_file = "{config.numerical_file}"
    run_simulation(case_file, numerical_file)
 ```

Create a production-quality script that handles errors gracefully and will execute successfully with the provided configuration files."""
        
        try:
            # Use LangChain invoke method - MISSION CRITICAL AI ONLY
            messages = [HumanMessage(content=prompt)]
            response = self.model.invoke(messages)
            
            script_content = self._extract_code_from_response(response.content)
            if not script_content:
                raise ValueError("AI failed to generate code block - no fallbacks allowed for high-stakes applications")
                
            return script_content
        except Exception as e:
            # NO FALLBACKS - High-stakes applications require AI reasoning
            raise RuntimeError(f"❌ MISSION CRITICAL ERROR: AI generation failed: {e}. High-stakes applications like SpaceX/Tesla require full AI reasoning, not templates. Check API key and connection.") from e
    
    def _get_relevant_examples(self, analysis: Dict[str, Any]) -> str:
        """Get relevant example patterns for the simulation type"""
        
        dimension = analysis['dimension']
        sim_type = analysis['simulation_type']
        
        examples = []
        
        if dimension == "1D" and sim_type == SimulationType.SHOCK_TUBE:
            examples.append("SOD SHOCK TUBE: Basic setup, density/velocity/pressure, 1D plots")
            
        elif dimension == "2D" and sim_type == SimulationType.EXTERNAL_FLOW:
            examples.append("NACA AIRFOIL: Levelset masking, 2D plots, external flow quantities")
            
        elif dimension == "3D" and sim_type == SimulationType.TURBULENCE:
            examples.append("TGV: Velocity focus, energy analysis, 2D slice visualization")
            
        return "\n".join(examples)
    
    def _extract_code_from_response(self, response_text: str) -> str:
        """Extract Python code from AI response - Mission Critical"""
        
        # Look for code blocks
        if "```python" in response_text:
            start = response_text.find("```python") + 9
            end = response_text.find("```", start)
            if end > start:
                return response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            if end > start:
                return response_text[start:end].strip()
        
        # For high-stakes applications, we need the full response if no code blocks
        # The AI should generate the complete script
        return response_text.strip()
    
    def _generate_with_rules(self, config: AgenticConfig, analysis: Dict[str, Any]) -> str:
        """Generate script using VectraSim's proven JAX-Fluids approach"""
        
        hardware_config = analysis['hardware_config']
        cuda_device = hardware_config.get('cuda_device', '0')
        
        script = f'''#!/usr/bin/env python3
"""
VectraSim Intelligent Simulation Suite
Adaptive JAX-Fluids Script Generator - Production Ready

Generated for: {analysis['simulation_type'].value} ({analysis['dimension']})
Simulation Intent: {config.simulation_intent}
Plotting Mode: {config.plotting_mode.value}

This script was automatically generated by VectraSim's adaptive agent
based on JAX-Fluids (Apache 2.0 licensed) for computational fluid dynamics.

VectraSim - Advanced Computational Physics Platform
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "{cuda_device}"

import sys
import time
import json
import numpy as np
from typing import Dict, Any

def modify_config_for_production(case_file: str) -> str:
    """Apply VectraSim's proven optimizations for stable JAX-Fluids execution"""
    
    print("🔧 Applying VectraSim optimizations...")
    
    # Load original config
    with open(case_file, 'r', encoding='utf-8') as f:
        case_config = json.load(f)
    
    # VectraSim's proven optimizations
    # Disable levelset output to avoid JAX-Fluids output writing bug
    case_config['output'] = {{
        "primitives": [],     # Minimal output for stability
        "miscellaneous": [], 
        "levelset": []        # Disable levelset output (avoids known JAX-Fluids bug)
    }}
    
    # Ensure single device decomposition
    if 'domain' in case_config and 'decomposition' in case_config['domain']:
        case_config['domain']['decomposition'] = {{
            'split_x': 1,
            'split_y': 1, 
            'split_z': 1
        }}
    
    # Add gravity if missing (required by JAX-Fluids)
    if 'forcings' not in case_config:
        case_config['forcings'] = {{'gravity': [0.0, 0.0, 0.0]}}
    
    # Write optimized config
    optimized_file = case_file.replace('.json', '_adaptive_optimized.json')
    with open(optimized_file, 'w', encoding='utf-8') as f:
        json.dump(case_config, f, indent=2)
    
    print(f"✅ VectraSim optimizations applied: {{os.path.basename(optimized_file)}}")
    return optimized_file

def main():
    """Adaptive JAX-Fluids simulation using VectraSim's proven approach"""
    
    print("🤖 VectraSim Adaptive JAX-Fluids Agent")
    print(f"🎯 Type: {analysis['simulation_type'].value} ({analysis['dimension']})")
    print(f"📊 Intent: {config.simulation_intent}")
    print(f"📈 Plotting: {config.plotting_mode.value}")
    print("=" * 70)
    
    try:
        # Import JAX-Fluids
        from jaxfluids import InputManager, InitializationManager, SimulationManager
        print("✅ JAX-Fluids imported successfully")
        
        # Apply VectraSim's proven optimizations
        case_file = "{config.case_file}"
        numerical_file = "{config.numerical_file}"
        
        optimized_case_file = modify_config_for_production(case_file)
        
        # SETUP SIMULATION with optimized config
        print("🔧 Setting up simulation with VectraSim optimizations...")
        input_manager = InputManager(optimized_case_file, numerical_file)
        initialization_manager = InitializationManager(input_manager)
        sim_manager = SimulationManager(input_manager)
        
        print("✅ JAX-Fluids managers created successfully")
        
        # INITIALIZE
        print("🚀 Initializing simulation...")
        init_start = time.time()
        
        jxf_buffers = initialization_manager.initialization()
        init_time = time.time() - init_start
        
        print(f"✅ Initialization completed in {{init_time:.2f}} seconds")
        
        # RUN SIMULATION using VectraSim's proven method
        print("⏰ Running simulation...")
        print("   Using VectraSim's proven stable configuration")
        
        sim_start = time.time()
        sim_manager.simulate(jxf_buffers)
        sim_time = time.time() - sim_start
        
        # Clean up temporary files
        try:
            os.remove(optimized_case_file)
            print("🧹 Temporary files cleaned up")
        except:
            pass
        
        # Success summary
        print("=" * 70)
        print(f"🎉 VectraSim Adaptive Simulation Completed Successfully!")
        print(f"📊 Summary:")
        print(f"   • Type: {analysis['simulation_type'].value} ({analysis['dimension']})")
        print(f"   • Intent: {config.simulation_intent}")
        print(f"   • Plotting Mode: {config.plotting_mode.value}")
        print(f"   • Initialization: {{init_time:.2f}}s")
        print(f"   • Simulation: {{sim_time:.2f}}s")
        print(f"   • Total: {{init_time + sim_time:.2f}}s")
        print(f"   • Status: ✅ ADAPTIVE SUCCESS")
        print("=" * 70)
        print("🤖 VectraSim Adaptive Agent: MISSION COMPLETE!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Adaptive simulation failed: {{e}}")
        print("🔍 Detailed traceback:")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    if exit_code == 0:
        print("\\n🏁 VectraSim Adaptive JAX-Fluids: COMPLETE!")
        print("✅ Intelligent simulation pipeline successful!")
    sys.exit(exit_code)
'''
        
        return script

def create_adaptive_jaxfluids_script(
    case_setup_path: str,
    numerical_setup_path: str,
    output_directory: str,
    simulation_intent: str,
    plotting_mode: str = "standard",
    gemini_api_key: str = None
) -> str:
    """
    Create an adaptive JAX-Fluids script that understands physics and requirements
    
    Args:
        case_setup_path: Path to case JSON
        numerical_setup_path: Path to numerical JSON
        output_directory: Where to save run.py
        simulation_intent: What you're trying to simulate (e.g., "shock tube analysis")
        plotting_mode: "minimal", "standard", "advanced", "research", or "off"
        gemini_api_key: Optional API key for enhanced AI generation
        
    Returns:
        str: Path to generated adaptive script
    """
    
    # Load configurations - handle relative paths correctly
    # If paths are just filenames, they should be in the output directory
    if not os.path.isabs(case_setup_path) and os.sep not in case_setup_path:
        case_setup_path = os.path.join(output_directory, case_setup_path)
    if not os.path.isabs(numerical_setup_path) and os.sep not in numerical_setup_path:
        numerical_setup_path = os.path.join(output_directory, numerical_setup_path)
        
    with open(case_setup_path, 'r', encoding='utf-8') as f:
        case_config = json.load(f)
    
    with open(numerical_setup_path, 'r', encoding='utf-8') as f:
        numerical_config = json.load(f)
    
    # Create configuration
    config = AgenticConfig(
        case_file=Path(case_setup_path).name,
        numerical_file=Path(numerical_setup_path).name,
        simulation_intent=simulation_intent,
        plotting_mode=PlottingMode(plotting_mode)
    )
    
    # Create agent
    agent = AdaptiveJAXFluidsAgent(gemini_api_key)
    
    # Generate script
    output_path = Path(output_directory) / "run.py"
    return agent.generate_adaptive_script(config, case_config, numerical_config, str(output_path)) 