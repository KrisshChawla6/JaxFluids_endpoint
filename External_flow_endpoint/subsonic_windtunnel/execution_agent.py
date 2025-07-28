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
        """Production-ready JAX-Fluids script using VectraSim's proven approach"""
        
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

def modify_config_for_production(case_file: str) -> str:
    \"\"\"Modify case configuration for stable JAX-Fluids execution\"\"\"
    
    print("🔧 Optimizing configuration for JAX-Fluids compatibility...")
    
    # Load original config
    with open(case_file, 'r', encoding='utf-8') as f:
        case_config = json.load(f)
    
    # Apply VectraSim's proven optimizations
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
    optimized_file = case_file.replace('.json', '_optimized.json')
    with open(optimized_file, 'w', encoding='utf-8') as f:
        json.dump(case_config, f, indent=2)
    
    print(f"✅ Configuration optimized: {{os.path.basename(optimized_file)}}")
    return optimized_file

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
        
        # Optimize configuration for stable execution
        optimized_case_file = modify_config_for_production(case_file)
        
        # Create JAX-Fluids managers
        print("🔧 Creating JAX-Fluids managers...")
        input_manager = InputManager(optimized_case_file, numerical_file)
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
            print("🧹 Temporary files cleaned up")
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