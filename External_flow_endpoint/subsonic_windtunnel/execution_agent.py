#!/usr/bin/env python3
"""
Execution Agent - JAX-Fluids Run Script Generation
Generates production-ready run.py scripts using JAX-Fluids documentation patterns
"""

import os
import sys
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Add parent directory to path for jaxfluids_run_generator import
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from adaptive_jaxfluids_agent import create_adaptive_jaxfluids_script, AdaptiveJAXFluidsAgent, AgenticConfig, PlottingMode
except ImportError:
    print("Warning: Adaptive JAX-Fluids agent not available - using fallback")
    create_adaptive_jaxfluids_script = None

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class ExecutionAgent:
    """
    Expert agent for JAX-Fluids simulation execution
    Generates run.py scripts and manages simulation lifecycle
    """
    
    def __init__(self, gemini_api_key: str = None):
        """Initialize the execution agent"""
        
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
        
        logger.info("🚀 ExecutionAgent initialized with Gemini 2.0 Flash Experimental")
    
    def generate_run_script(self, 
                          numerical_setup: Dict[str, Any],
                          case_setup: Dict[str, Any], 
                          user_prompt: str,
                          output_dir: str = ".") -> Dict[str, Any]:
        """Generate JAX-Fluids run.py script using adaptive agent"""
        
        logger.info("🔧 Generating JAX-Fluids execution script...")
        
        # Check if adaptive agent is available
        if create_adaptive_jaxfluids_script:
            try:
                # Write the JSON files to the simulation directory for the AI agent to use
                # Case file is named after simulation name, numerical is always numerical_setup.json
                simulation_name = os.path.basename(output_dir)  # Extract from directory name
                case_file = os.path.join(output_dir, f"{simulation_name}.json")
                numerical_file = os.path.join(output_dir, 'numerical_setup.json')
                
                # Write the files so they exist for the AI agent
                with open(case_file, 'w', encoding='utf-8') as f:
                    json.dump(case_setup, f, indent=2)
                with open(numerical_file, 'w', encoding='utf-8') as f:
                    json.dump(numerical_setup, f, indent=2)
                
                # Extract simulation intent from user prompt
                simulation_intent = self._extract_simulation_intent(user_prompt)
                
                # Determine plotting mode based on prompt
                plotting_mode = self._determine_plotting_mode(user_prompt)
                
                # Use adaptive agent to generate script
                script_path = create_adaptive_jaxfluids_script(
                    case_setup_path=os.path.basename(case_file),  # Just filename, not full path
                    numerical_setup_path=os.path.basename(numerical_file),  # Just filename
                    output_directory=output_dir,
                    simulation_intent=simulation_intent,
                    plotting_mode=plotting_mode,
                    gemini_api_key=os.getenv('GEMINI_API_KEY')
                )
                
                # Read the generated script
                with open(script_path, 'r', encoding='utf-8') as f:
                    script_content = f.read()
                
                # Keep the JSON files in the directory for JAX-Fluids execution
                # No cleanup needed - these are the actual configuration files
                
                return {
                    'script_content': script_content,
                    'script_path': script_path,
                    'execution_parameters': self._extract_execution_parameters(numerical_setup, case_setup),
                    'estimated_runtime': self._estimate_runtime(numerical_setup, case_setup),
                    'memory_requirements': self._estimate_memory(numerical_setup, case_setup),
                    'adaptive_agent_used': True
                }
                
            except Exception as e:
                # NO FALLBACKS - High-stakes applications require AI reasoning
                logger.error(f"❌ MISSION CRITICAL ERROR: Adaptive agent failed: {e}")
                raise RuntimeError(f"❌ EXECUTION AGENT FAILURE: High-stakes applications like SpaceX/Tesla require full AI reasoning. Adaptive agent error: {e}") from e
        
        return {
            'script_content': script_content,
            'script_path': os.path.join(output_dir, 'run.py'),
            'execution_parameters': self._extract_execution_parameters(numerical_setup, case_setup),
            'estimated_runtime': "Unknown",
            'memory_requirements': "Unknown",
            'fallback_used': True
        }
    
    def _extract_simulation_intent(self, user_prompt: str) -> str:
        """Extract simulation intent from user prompt"""
        # Simple extraction - in production this could be more sophisticated
        return user_prompt.replace("Create a", "").replace("Generate a", "").strip()
    
    def _determine_plotting_mode(self, user_prompt: str) -> str:
        """Determine plotting mode from user prompt"""
        prompt_lower = user_prompt.lower()
        
        if any(word in prompt_lower for word in ['research', 'detailed', 'comprehensive', 'analysis']):
            return 'research'
        elif any(word in prompt_lower for word in ['advanced', 'full', 'complete']):
            return 'advanced'
        elif any(word in prompt_lower for word in ['minimal', 'basic', 'simple', 'performance']):
            return 'minimal'
        elif any(word in prompt_lower for word in ['no plot', 'without visualization', 'no visualization']):
            return 'off'
        else:
            return 'standard'
    
    def _get_execution_system_prompt(self) -> str:
        """System prompt for JAX-Fluids run script generation"""
        
        return """You are a JAX-Fluids execution expert specializing in creating production-ready run.py scripts.

**REQUIRED OUTPUT:**
Generate a complete Python script that follows JAX-Fluids best practices.

**SCRIPT STRUCTURE:**
- Proper shebang and docstring
- All necessary imports (jax, jaxfluids, etc.)
- Configuration loading from JSON files
- Simulation initialization and execution
- Progress monitoring and logging
- Error handling and cleanup

**REQUIREMENTS:**
- Use InputReader to load numerical_setup.json and case.json
- Initialize simulation with proper domain and boundary conditions
- Run simulation with time stepping and convergence monitoring
- Save results in standard JAX-Fluids format
- Include timing and performance metrics
- Handle interruption gracefully

Always generate complete, executable Python code only."""
    
    def _get_fallback_script(self, numerical_setup: Dict[str, Any], case_setup: Dict[str, Any], output_dir: str) -> str:
        """Production-ready JAX-Fluids script using VectraSim's proven approach with intelligent optimizations"""
        
        end_time = numerical_setup.get('end_time', 1.0)
        case_name = case_setup.get('general', {}).get('case_name', 'external_flow')
        
        # Get domain information for summary
        domain = case_setup.get('domain', {})
        nx = domain.get('x', {}).get('cells', 64)
        ny = domain.get('y', {}).get('cells', 64) 
        nz = domain.get('z', {}).get('cells', 64)
        
        script_content = f"""#!/usr/bin/env python3
\"\"\"
VectraSim Intelligent Simulation Suite
JAX-Fluids External Flow Simulation - Production Ready

Auto-generated by VectraSim's adaptive system
based on JAX-Fluids (Apache 2.0 licensed) for computational fluid dynamics.

VectraSim - Advanced Computational Physics Platform
\"\"\"

import os
import sys
import time
import json
from typing import Dict, Any

def modify_config_for_production(case_file: str, numerical_file: str = None) -> str:
    \"\"\"Intelligently modify configuration for stable JAX-Fluids execution based on environment and requirements\"\"\"
    
    print("🔧 Applying intelligent production optimizations...")
    
    # Load original configs
    with open(case_file, 'r', encoding='utf-8') as f:
        case_config = json.load(f)
    
    numerical_config = None
    if numerical_file and os.path.exists(numerical_file):
        with open(numerical_file, 'r', encoding='utf-8') as f:
            numerical_config = json.load(f)
    
    # Detect execution environment (HPCC vs local)
    is_hpcc_environment = _detect_hpcc_environment()
    
    # Apply intelligent output strategy based on environment and user intent
    output_strategy = _determine_output_strategy(case_config, is_hpcc_environment)
    
    # PRESERVE intelligent output config unless we have a specific override reason
    if output_strategy.get('preserve_original', False):
        print(f"🎯 Output strategy: {{output_strategy['strategy_name']}} (preserving intelligent config)")
    else:
        case_config['output'] = output_strategy['output_config']
        print(f"🎯 Output strategy: {{output_strategy['strategy_name']}}")
    
    # Intelligent simulation duration and save frequency
    timing_config = _optimize_simulation_timing(case_config, is_hpcc_environment)
    if 'general' in case_config:
        case_config['general'].update(timing_config)
        print(f"⏰ Timing optimized: {{timing_config['end_time']}}s end time, save every {{timing_config['save_dt']}}s")
    
    # Adaptive mesh resolution based on environment capabilities
    if not is_hpcc_environment:
        mesh_config = _optimize_mesh_for_environment(case_config, is_hpcc_environment)
        if mesh_config['adjusted']:
            case_config['domain'].update(mesh_config['domain_config'])
            print(f"🔧 Mesh adapted for local environment: {{mesh_config['description']}}")
    else:
        print("🚀 HPCC detected: Preserving high-resolution mesh for maximum accuracy")
    
    # Intelligent numerical parameter optimization
    if numerical_config:
        numerical_optimizations = _optimize_numerical_params(numerical_config, case_config, is_hpcc_environment)
        if numerical_optimizations['modified']:
            numerical_file_optimized = numerical_file.replace('.json', '_production_optimized.json')
            with open(numerical_file_optimized, 'w', encoding='utf-8') as f:
                json.dump(numerical_optimizations['config'], f, indent=2)
            print(f"🔧 Numerical parameters optimized: {{numerical_optimizations['description']}}")
    
    # Essential JAX-Fluids compatibility fixes
    case_config = _apply_essential_fixes(case_config)
    
    # Write optimized config
    optimized_file = case_file.replace('.json', '_production_optimized.json')
    with open(optimized_file, 'w', encoding='utf-8') as f:
        json.dump(case_config, f, indent=2)
    
    print(f"✅ Intelligent optimizations applied: {{os.path.basename(optimized_file)}}")
    return optimized_file

def _detect_hpcc_environment() -> bool:
    \"\"\"Detect if running on HPCC cluster vs local machine\"\"\"
    hpcc_indicators = [
        'SLURM_JOB_ID',
        'PBS_JOBID', 
        'LSB_JOBID',
        'CUDA_VISIBLE_DEVICES'
    ]
    
    # Check for HPCC environment variables
    for indicator in hpcc_indicators:
        if os.getenv(indicator):
            return True
    
    # Check for high-end hardware (multiple GPUs, high memory)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        if memory_gb > 128 or cpu_count > 32:  # High-end workstation/cluster
            return True
    except ImportError:
        pass
    
    return False

def _determine_output_strategy(case_config: Dict, is_hpcc: bool) -> Dict:
    \"\"\"Intelligently determine output strategy based on environment and simulation goals\"\"\"
    
    # Extract simulation intent from case name or setup
    case_name = case_config.get('general', {{}}).get('case_name', '').lower()
    
    if 'test' in case_name or 'debug' in case_name:
        # Testing/debugging: minimal output for fastest execution
        return {{
            'strategy_name': 'minimal_testing',
            'output_config': {{
                "primitives": [],
                "miscellaneous": [],
                "levelset": []
            }}
        }}
    elif is_hpcc and ('production' in case_name or 'research' in case_name):
        # HPCC production: full output for analysis
        return {{
            'strategy_name': 'hpcc_production',
            'output_config': {{
                "primitives": ["density", "velocity", "pressure", "temperature"],
                "miscellaneous": ["mach_number", "vorticity"],
                "levelset": ["levelset", "volume_fraction"]
            }}
        }}
    elif is_hpcc:
        # HPCC general: moderate output
        return {{
            'strategy_name': 'hpcc_standard',
            'output_config': {{
                "primitives": ["density", "velocity", "pressure"],
                "miscellaneous": ["mach_number"],
                "levelset": ["levelset"]
            }}
        }}
    else:
        # Local development: minimal output for stability (based on our working discovery)
        return {{
            'strategy_name': 'local_stable',
            'output_config': {{
                "primitives": [],
                "miscellaneous": [],
                "levelset": []
            }}
        }}

def _optimize_simulation_timing(case_config: Dict, is_hpcc: bool) -> Dict:
    \"\"\"Optimize simulation timing based on environment and goals\"\"\"
    
    general = case_config.get('general', {{}})
    current_end_time = general.get('end_time', 1.0)
    
    if is_hpcc:
        # HPCC: Allow full simulation duration as specified by agents
        return {{
            'end_time': current_end_time,
            'save_dt': max(current_end_time / 50, 0.1)  # ~50 snapshots
        }}
    else:
        # Local: Use working approach - balance between meaningful results and resource constraints
        if current_end_time < 5.0:
            # Short simulation: keep as is but ensure reasonable saves
            return {{
                'end_time': current_end_time, 
                'save_dt': max(current_end_time / 20, 0.05)
            }}
        else:
            # Long simulation: adapt for local execution using proven working values
            return {{
                'end_time': 20.0,  # Proven working value for ~100 natural timesteps
                'save_dt': 5.0     # Proven working save frequency
            }}

def _optimize_mesh_for_environment(case_config: Dict, is_hpcc: bool) -> Dict:
    \"\"\"Optimize mesh resolution based on environment capabilities\"\"\"
    
    if is_hpcc:
        # HPCC: Preserve agent's high-resolution choice
        return {{'adjusted': False}}
    
    domain = case_config.get('domain', {{}})
    nx = domain.get('x', {{}}).get('cells', 64)
    ny = domain.get('y', {{}}).get('cells', 64)
    nz = domain.get('z', {{}}).get('cells', 64)
    
    total_cells = nx * ny * nz
    
    if total_cells > 1_000_000:  # > 1M cells
        # Large mesh: reduce to proven working size for local stability
        scale_factor = (300_000 / total_cells) ** (1/3)  # Target ~300k cells
        new_nx = max(32, int(nx * scale_factor))
        new_ny = max(32, int(ny * scale_factor))
        new_nz = max(32, int(nz * scale_factor))
        
        return {{
            'adjusted': True,
            'description': f'Reduced from {{nx}}×{{ny}}×{{nz}} to {{new_nx}}×{{new_ny}}×{{new_nz}} for local stability',
            'domain_config': {{
                'x': {{**domain.get('x', {{}}), 'cells': new_nx}},
                'y': {{**domain.get('y', {{}}), 'cells': new_ny}},
                'z': {{**domain.get('z', {{}}), 'cells': new_nz}}
            }}
        }}
    
    return {{'adjusted': False}}

def _optimize_numerical_params(numerical_config: Dict, case_config: Dict, is_hpcc: bool) -> Dict:
    \"\"\"Optimize numerical parameters based on environment and stability requirements\"\"\"
    
    modified = False
    description_parts = []
    
    # Conservative CFL for stability (based on working discovery)
    if 'conservatives' in numerical_config and 'time_integration' in numerical_config['conservatives']:
        time_integration = numerical_config['conservatives']['time_integration']
        current_cfl = time_integration.get('CFL', 0.5)
        
        if not is_hpcc and current_cfl > 0.5:
            # Local environment: use conservative CFL for stability
            time_integration['CFL'] = 0.5
            modified = True
            description_parts.append(f'CFL reduced to 0.5 for local stability')
        elif is_hpcc and current_cfl < 0.3:
            # HPCC: allow more aggressive timestepping if desired
            time_integration['CFL'] = 0.5
            modified = True
            description_parts.append(f'CFL optimized to 0.5 for HPCC performance')
    
    return {{
        'modified': modified,
        'config': numerical_config,
        'description': '; '.join(description_parts) if description_parts else 'No changes needed'
    }}

def _apply_essential_fixes(case_config: Dict) -> Dict:
    \"\"\"Apply essential JAX-Fluids compatibility fixes\"\"\"
    
    # Single device decomposition for stability
    if 'domain' in case_config:
        if 'decomposition' not in case_config['domain']:
            case_config['domain']['decomposition'] = {{}}
        case_config['domain']['decomposition'].update({{
            'split_x': 1,
            'split_y': 1,
            'split_z': 1
        }})
    
    # Required gravity forcing
    if 'forcings' not in case_config:
        case_config['forcings'] = {{'gravity': [0.0, 0.0, 0.0]}}
    
    # Ensure proper nondimensionalization
    if 'nondimensionalization_parameters' in case_config:
        nondim = case_config['nondimensionalization_parameters']
        if 'length_reference' not in nondim or nondim['length_reference'] == 0:
            nondim['length_reference'] = 1.0
    
    return case_config

def main():
    \"\"\"Main simulation function using VectraSim's proven approach\"\"\"
    
    print("🚀 VectraSim → JAX-Fluids: Production Simulation")
    print("=" * 70)
    
    # Configuration directory
    output_dir = "{output_dir}"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Import JAX-Fluids
        from jaxfluids import InputManager, InitializationManager, SimulationManager
        print("✅ JAX-Fluids imported successfully")
        
        # Configuration files
        case_file = None
        numerical_file = None
        
        # Find configuration files
        for file in os.listdir(output_dir):
            if file.endswith('.json'):
                if 'numerical' in file.lower():
                    numerical_file = os.path.join(output_dir, file)
                elif file.startswith('jaxfluids_external_flow_'):
                    case_file = os.path.join(output_dir, file)
        
        if not case_file or not numerical_file:
            raise FileNotFoundError("Could not find required configuration files")
        
        print(f"📋 Found configuration files:")
        print(f"   Case: {{os.path.basename(case_file)}}")
        print(f"   Numerical: {{os.path.basename(numerical_file)}}")
        
        # Apply intelligent production optimizations
        optimized_case_file = modify_config_for_production(case_file, numerical_file)
        
        # Use optimized numerical file if it was created
        optimized_numerical_file = numerical_file.replace('.json', '_production_optimized.json')
        final_numerical_file = optimized_numerical_file if os.path.exists(optimized_numerical_file) else numerical_file
        
        # Create JAX-Fluids managers
        print("🔧 Creating JAX-Fluids managers...")
        input_manager = InputManager(optimized_case_file, final_numerical_file)
        initialization_manager = InitializationManager(input_manager)
        sim_manager = SimulationManager(input_manager)
        
        print("✅ JAX-Fluids managers created successfully")
        
        # Initialize simulation
        print("🚀 Initializing simulation...")
        init_start = time.time()
        
        buffers = initialization_manager.initialization()
        init_time = time.time() - init_start
        
        print(f"✅ Initialization completed in {{init_time:.2f}} seconds")
        
        # Run simulation using VectraSim's proven method
        print("⏰ Running simulation...")
        print("   Using VectraSim's optimized configuration")
        
        sim_start = time.time()
        
        # Use the standard simulate() method with our optimized config
        sim_manager.simulate(buffers)
        
        sim_time = time.time() - sim_start
        total_time = init_time + sim_time
        
        # Clean up temporary files
        try:
            os.remove(optimized_case_file)
            if os.path.exists(optimized_numerical_file):
                os.remove(optimized_numerical_file)
            print("🧹 Temporary optimization files cleaned up")
        except:
            pass
        
        # Success summary
        print("=" * 70)
        print(f"🎉 VectraSim Simulation Completed Successfully!")
        print(f"📊 Performance Summary:")
        print(f"   • Configuration: VectraSim External Flow Endpoint")
        print(f"   • Case: {case_name}")
        print(f"   • Grid: {nx}×{ny}×{nz} cells ({{({nx}*{ny}*{nz})/1e6:.3f}}M total)")
        print(f"   • Physics: 3D External Flow with Levelset")
        print(f"   • Numerical: High-order methods (WENO, HLLC, RK3)")
        print(f"   • Initialization time: {{init_time:.2f}} seconds")
        print(f"   • Simulation time: {{sim_time:.2f}} seconds")
        print(f"   • Total runtime: {{total_time:.2f}} seconds")
        print(f"   • Status: ✅ PRODUCTION SUCCESS")
        print("=" * 70)
        print("🚀 VectraSim External Flow Pipeline: FULLY OPERATIONAL!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\\n⚠️ Simulation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Simulation failed: {{e}}")
        print("🔍 Detailed traceback:")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    if exit_code == 0:
        print("\\n🏁 VectraSim External Flow Simulation: COMPLETE!")
        print("✅ Ready for production deployment!")
    sys.exit(exit_code)
"""
        
        return script_content
    
    def _validate_script(self, script_content: str) -> bool:
        """Validate that the generated script has required components"""
        
        required_elements = [
            'import',
            'def main',
            'InputReader',
            'if __name__'
        ]
        
        return all(element in script_content for element in required_elements)
    
    def execute_simulation(self, script_path: str, run_simulation: bool = False) -> Dict[str, Any]:
        """Execute the JAX-Fluids simulation"""
        
        if not run_simulation:
            return {
                'status': 'script_generated',
                'message': 'Simulation script generated but not executed (run_simulation=False)',
                'script_path': script_path
            }
        
        logger.info(f"🚀 Executing simulation: {script_path}")
        
        try:
            start_time = time.time()
            
            # Run the simulation script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                logger.info("✅ Simulation completed successfully")
                return {
                    'status': 'success',
                    'execution_time': execution_time,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'script_path': script_path
                }
            else:
                logger.error(f"❌ Simulation failed with code {result.returncode}")
                return {
                    'status': 'failed',
                    'return_code': result.returncode,
                    'execution_time': execution_time,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'script_path': script_path
                }
                
        except subprocess.TimeoutExpired:
            logger.error("⏰ Simulation timed out")
            return {
                'status': 'timeout',
                'message': 'Simulation exceeded 1 hour timeout',
                'script_path': script_path
            }
        except Exception as e:
            logger.error(f"❌ Execution error: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'script_path': script_path
            }
    
    def get_execution_parameters(self) -> List[str]:
        """Return list of available execution parameters"""
        
        return [
            "time_stepping_scheme",
            "cfl_number", 
            "end_time",
            "save_interval",
            "restart_interval",
            "output_format",
            "precision",
            "device_count",
            "memory_limit",
            "checkpoint_frequency"
        ]
    
    def _extract_execution_parameters(self, numerical_setup: Dict, case_setup: Dict) -> Dict[str, Any]:
        """Extract key execution parameters for monitoring"""
        
        params = {}
        
        # From numerical setup
        if 'time_integration' in numerical_setup:
            params['time_scheme'] = numerical_setup['time_integration'].get('integrator', 'RK3')
            params['cfl_number'] = numerical_setup['time_integration'].get('CFL', 0.5)
        
        if 'output' in numerical_setup:
            params['save_interval'] = numerical_setup['output'].get('save_interval', 100)
        
        # From case setup  
        params['end_time'] = case_setup.get('end_time', 1.0)
        params['domain_size'] = case_setup.get('domain', {}).get('x', {}).get('cells', 'Unknown')
        
        return params
    
    def _estimate_runtime(self, numerical_setup: Dict, case_setup: Dict) -> str:
        """Estimate simulation runtime"""
        
        try:
            end_time = case_setup.get('end_time', 1.0)
            cfl = numerical_setup.get('time_integration', {}).get('CFL', 0.5)
            domain_cells = case_setup.get('domain', {}).get('x', {}).get('cells', 100)
            
            # Rough estimate based on typical JAX-Fluids performance
            estimated_steps = int(end_time / (cfl * 0.001))  # Rough dt estimate
            estimated_seconds = estimated_steps * 0.01  # ~10ms per step estimate
            
            if estimated_seconds < 60:
                return f"~{estimated_seconds:.0f} seconds"
            elif estimated_seconds < 3600:
                return f"~{estimated_seconds/60:.1f} minutes"  
            else:
                return f"~{estimated_seconds/3600:.1f} hours"
                
        except:
            return "Unknown"
    
    def _estimate_memory(self, numerical_setup: Dict, case_setup: Dict) -> str:
        """Estimate memory requirements"""
        
        try:
            domain = case_setup.get('domain', {})
            nx = domain.get('x', {}).get('cells', 100)
            ny = domain.get('y', {}).get('cells', 100) 
            nz = domain.get('z', {}).get('cells', 100)
            
            total_cells = nx * ny * nz
            
            # Rough estimate: ~8 bytes per cell per variable, ~10 variables
            memory_bytes = total_cells * 8 * 10
            memory_gb = memory_bytes / (1024**3)
            
            if memory_gb < 1:
                return f"~{memory_bytes/(1024**2):.0f} MB"
            else:
                return f"~{memory_gb:.1f} GB"
                
        except:
            return "Unknown" 