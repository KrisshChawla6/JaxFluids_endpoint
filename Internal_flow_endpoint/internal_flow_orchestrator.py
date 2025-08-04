#!/usr/bin/env python3
"""
VectraSim Internal Flow Orchestrator
Coordinates specialized AI agents for supersonic internal flows and rocket propulsion

This orchestrator manages:
- Supersonic Case Setup Expert (with intelligent boundary conditions)
- Internal Flow Numerical Expert (mask-aware numerical schemes)  
- Adaptive Execution Agent (forcing system integration)

Enhanced with intelligent_BC_final integration for automatic boundary condition generation
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from supersonic_internal_flow.case_setup_expert import SupersonicCaseSetupExpert
from supersonic_internal_flow.numerical_setup_expert import InternalFlowNumericalExpert
from supersonic_internal_flow.execution_agent import InternalFlowExecutionAgent
from adaptive_jaxfluids_agent import create_adaptive_jaxfluids_script

logger = logging.getLogger(__name__)

@dataclass
class InternalFlowResponse:
    """Response from internal flow simulation generation"""
    success: bool
    simulation_directory: str
    case_file: str
    numerical_file: str
    run_script: str
    simulation_summary: Dict[str, Any]
    boundary_conditions: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class InternalFlowOrchestrator:
    """
    Master orchestrator for internal flow simulations
    Coordinates specialized agents for rocket propulsion and supersonic flows
    Enhanced with intelligent boundary condition integration
    """
    
    def __init__(self, gemini_api_key: str, flow_config: Dict[str, Any]):
        """
        Initialize the internal flow orchestrator
        
        Args:
            gemini_api_key: Gemini API key for AI agents
            flow_config: Flow configuration parameters including boundary conditions
        """
        self.gemini_api_key = gemini_api_key
        self.flow_config = flow_config
        
        # Initialize specialized agents
        self.case_setup_expert = SupersonicCaseSetupExpert(gemini_api_key)
        self.numerical_expert = InternalFlowNumericalExpert(gemini_api_key)
        self.execution_agent = InternalFlowExecutionAgent(gemini_api_key)
        
        # Extract boundary condition information
        self.boundary_conditions = flow_config.get('boundary_conditions', {})
        self.bc_storage_dir = flow_config.get('bc_storage_dir', '')
        
        logger.info("🚀 Enhanced Internal Flow Orchestrator initialized")
        logger.info(f"🎯 Flow Type: {flow_config.get('flow_type', 'unknown')}")
        
        if self.boundary_conditions:
            logger.info(f"🧠 BC Integration: {self.boundary_conditions.get('inlet_points', 0):,} inlet points, {self.boundary_conditions.get('outlet_points', 0):,} outlet points")

    def create_internal_flow_simulation(
        self,
        user_prompt: str,
        output_directory: str
    ) -> InternalFlowResponse:
        """
        Create complete internal flow simulation with intelligent boundary conditions
        
        Args:
            user_prompt: User description of simulation
            output_directory: Where to save simulation files
            
        Returns:
            InternalFlowResponse with all simulation files and boundary condition data
        """
        
        print("🌪️ INTERNAL FLOW ORCHESTRATOR - Enhanced with Intelligent BCs")
        
        # Create unique simulation ID
        simulation_id = int(time.time())
        simulation_name = f"jaxfluids_internal_flow_{simulation_id}"
        
        # Setup output directory
        output_path = Path(output_directory)
        simulation_dir = output_path / simulation_name
        simulation_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Simulation Directory: {simulation_dir}")
        print(f"🧠 Boundary Conditions: {self.bc_storage_dir}")
        
        # Create enhanced context with boundary condition data
        context = {
            "user_prompt": user_prompt,
            "simulation_name": simulation_name,
            "simulation_directory": str(simulation_dir),
            "output_directory": output_directory,
            **self.flow_config
        }
        
        # Add detailed boundary condition context
        if self.boundary_conditions:
            context.update({
                "has_intelligent_bcs": True,
                "inlet_mask_file": self.boundary_conditions.get('inlet_mask_file', ''),
                "outlet_mask_file": self.boundary_conditions.get('outlet_mask_file', ''),
                "boundary_data_file": self.boundary_conditions.get('boundary_data_file', ''),
                "inlet_points": self.boundary_conditions.get('inlet_points', 0),
                "outlet_points": self.boundary_conditions.get('outlet_points', 0),
                "bc_storage_directory": self.bc_storage_dir
            })
        else:
            context["has_intelligent_bcs"] = False
        
        try:
            print("\n🌪️ PHASE 1: SUPERSONIC CASE SETUP (with Intelligent BCs)")
            print("=" * 60)
            
            # Generate case setup with boundary condition integration
            case_setup = self.case_setup_expert.generate_case_setup(context)
            
            print("\n🔢 PHASE 2: NUMERICAL SETUP (Mask-Aware)")
            print("=" * 60)
            
            # Generate numerical setup aware of mask-based BCs
            numerical_setup = self.numerical_expert.generate_numerical_setup(context)
            
            print("\n⚡ PHASE 3: EXECUTION SCRIPT GENERATION (Forcing-Enhanced)")
            print("=" * 60)
            
            # Generate execution summary and create proper JAX-Fluids run script
            execution_summary = self.execution_agent.generate_execution_summary(context)
            
            # Create comprehensive Python run script
            run_script_content = self._create_jax_fluids_run_script(simulation_name, context)
            
            execution_result = {
                'script': run_script_content,
                'summary': execution_summary,
                'forcing_system_details': 'Integrated with intelligent boundary conditions'
            }
            
            print("\n🔗 PHASE 4: INTEGRATION & FINALIZATION")
            print("=" * 60)
            
            # Save all files
            case_file = self._save_case_setup(case_setup, simulation_dir, simulation_name)
            numerical_file = self._save_numerical_setup(numerical_setup, simulation_dir)
            run_script = self._save_execution_script(execution_result, simulation_dir)
            
            # Create simulation summary with boundary condition info
            simulation_summary = self._create_simulation_summary(
                context, case_setup, numerical_setup, execution_result
            )
            
            print(f"✅ Case Setup: {case_file}")
            print(f"✅ Numerical Setup: {numerical_file}")
            print(f"✅ Run Script: {run_script}")
            
            if self.boundary_conditions:
                print(f"✅ Boundary Conditions: Integrated from {self.bc_storage_dir}")
                print(f"   🔴 Inlet mask: {self.boundary_conditions['inlet_points']:,} points")
                print(f"   🟢 Outlet mask: {self.boundary_conditions['outlet_points']:,} points")
            
            return InternalFlowResponse(
                success=True,
                simulation_directory=str(simulation_dir),
                case_file=case_file,
                numerical_file=numerical_file,
                run_script=run_script,
                simulation_summary=simulation_summary,
                boundary_conditions=self.boundary_conditions
            )
            
        except Exception as e:
            error_msg = f"Orchestrator failed: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            
            return InternalFlowResponse(
                success=False,
                simulation_directory=str(simulation_dir),
                case_file="",
                numerical_file="",
                run_script="",
                simulation_summary={},
                boundary_conditions=self.boundary_conditions,
                error_message=error_msg
            )
    
    def _save_case_setup(self, case_setup: Dict[str, Any], simulation_dir: Path, simulation_name: str) -> str:
        """
        Save the case setup file to the simulation directory.
        """
        case_file = simulation_dir / f"{simulation_name}.json"
        with open(case_file, 'w', encoding='utf-8') as f:
            json.dump(case_setup, f, indent=2)
        return str(case_file)
    
    def _save_numerical_setup(self, numerical_setup: Dict[str, Any], simulation_dir: Path) -> str:
        """
        Save the numerical setup file to the simulation directory.
        """
        numerical_file = simulation_dir / "numerical_setup.json"
        with open(numerical_file, 'w', encoding='utf-8') as f:
            json.dump(numerical_setup, f, indent=2)
        return str(numerical_file)
    
    def _save_execution_script(self, execution_result: Dict[str, Any], simulation_dir: Path) -> str:
        """
        Save the execution script to the simulation directory as a Python file.
        """
        run_script = simulation_dir / "run_simulation.py"
        with open(run_script, 'w', encoding='utf-8') as f:
            f.write(execution_result['script'])
        
        # Make it executable on Unix systems
        try:
            import stat
            run_script.chmod(run_script.stat().st_mode | stat.S_IEXEC)
        except:
            pass  # Windows doesn't need this
            
        return str(run_script)
    
    def _create_jax_fluids_run_script(self, simulation_name: str, context: Dict[str, Any]) -> str:
        """
        Creates a comprehensive JAX-Fluids run script based on the working rocket_simulation_final pattern.
        """
        bc_storage_dir = context.get("bc_storage_directory", "")
        inlet_points = context.get("inlet_points", 0)
        outlet_points = context.get("outlet_points", 0)
        
        script_content = f'''#!/usr/bin/env python3
"""
JAX-Fluids Internal Flow Simulation Runner
Enhanced with Intelligent Boundary Conditions
Generated by VectraSim Internal Flow Endpoint
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    import jax.numpy as jnp
    from jaxfluids import InputManager, InitializationManager, SimulationManager
    JAX_FLUIDS_AVAILABLE = True
except ImportError:
    print("⚠️ JAX-Fluids not available. This is a template script.")
    JAX_FLUIDS_AVAILABLE = False

class InternalFlowSimulationRunner:
    """Internal flow simulation runner with intelligent boundary conditions"""
    
    def __init__(self, config_file: str = "{simulation_name}.json", 
                 numerical_file: str = "numerical_setup.json"):
        """Initialize simulation runner"""
        self.config_file = Path(config_file)
        self.numerical_file = Path(numerical_file)
        self.output_dir = Path("output")
        self.log_dir = Path("logs")
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Simulation info
        self.simulation_name = "{simulation_name}"
        self.bc_storage_dir = r"{bc_storage_dir}"
        self.inlet_points = {inlet_points}
        self.outlet_points = {outlet_points}
        
    def setup_logging(self):
        """Setup simulation logging"""
        log_file = self.log_dir / f"simulation_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def validate_setup(self):
        """Validate simulation setup"""
        self.logger.info("🔍 VALIDATING SIMULATION SETUP")
        self.logger.info("=" * 60)
        
        # Check files
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {{self.config_file}}")
        if not self.numerical_file.exists():
            raise FileNotFoundError(f"Numerical file not found: {{self.numerical_file}}")
            
        # Validate boundary conditions
        if Path(self.bc_storage_dir).exists():
            self.logger.info(f"✅ Boundary conditions: {{self.bc_storage_dir}}")
            self.logger.info(f"   🔴 Inlet points: {{self.inlet_points:,}}")
            self.logger.info(f"   🟢 Outlet points: {{self.outlet_points:,}}")
        else:
            self.logger.warning("⚠️ Boundary condition directory not found")
            
        self.logger.info("✅ Setup validation completed")
        
    def run_simulation(self, max_iterations: int = 100):
        """Run the simulation"""
        if not JAX_FLUIDS_AVAILABLE:
            self.logger.error("❌ JAX-Fluids not available. Cannot run simulation.")
            return
            
        self.logger.info("🚀 STARTING INTERNAL FLOW SIMULATION")
        self.logger.info("=" * 80)
        self.logger.info(f"   Simulation: {{self.simulation_name}}")
        self.logger.info(f"   Config: {{self.config_file}}")
        self.logger.info(f"   Numerical: {{self.numerical_file}}")
        self.logger.info(f"   Max iterations: {{max_iterations}}")
        self.logger.info(f"   Intelligent BCs: {{self.inlet_points:,}} inlet, {{self.outlet_points:,}} outlet")
        self.logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Validate setup
            self.validate_setup()
            
            # Initialize JAX-Fluids
            self.logger.info("🔧 Initializing JAX-Fluids...")
            
            # Load configuration
            input_manager = InputManager(str(self.config_file), str(self.numerical_file))
            initialization_manager = InitializationManager(input_manager)
            sim_manager = SimulationManager(input_manager)
            
            # Initialize simulation
            self.logger.info("🔧 Initializing simulation state...")
            buffers = initialization_manager.initialization()
            
            # Run simulation
            self.logger.info("🚀 Running simulation...")
            sim_manager.simulate(buffers)
            
            elapsed_time = time.time() - start_time
            self.logger.info("🎉 SIMULATION COMPLETED SUCCESSFULLY!")
            self.logger.info(f"   Total time: {{elapsed_time:.2f}} seconds")
            self.logger.info(f"   Output directory: {{self.output_dir}}")
            
        except Exception as e:
            self.logger.error(f"❌ Simulation failed: {{e}}")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='JAX-Fluids Internal Flow Simulation')
    parser.add_argument('--iterations', '-i', type=int, default=100,
                       help='Maximum number of iterations')
    parser.add_argument('--config', '-c', type=str, default='{simulation_name}.json',
                       help='Configuration file')
    parser.add_argument('--numerical', '-n', type=str, default='numerical_setup.json',
                       help='Numerical setup file')
    
    args = parser.parse_args()
    
    print("🚀 JAX-Fluids Internal Flow Simulation Runner")
    print("Enhanced with Intelligent Boundary Conditions")
    print("=" * 60)
    
    try:
        runner = InternalFlowSimulationRunner(args.config, args.numerical)
        runner.run_simulation(args.iterations)
        
    except Exception as e:
        print(f"❌ Simulation failed: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        return script_content
    
    def _create_simulation_summary(self, context: Dict[str, Any], case_setup: Dict[str, Any], numerical_setup: Dict[str, Any], execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a comprehensive simulation summary.
        """
        simulation_id = int(time.time())
        simulation_name = context.get("simulation_name")
        
        summary = {
            "simulation_id": simulation_id,
            "simulation_name": simulation_name,
            "timestamp": simulation_id,
            "user_prompt": context.get("user_prompt"),
            "flow_configuration": context.get("flow_config"),
            "physics_summary": {
                "flow_regime": "supersonic_internal",
                "compressible": True,
                "viscous": True,
                "heat_transfer": True,
                "shock_capturing": True
            },
            "files_generated": {
                "case_setup": context.get("case_file"),
                "numerical_setup": context.get("numerical_file"),
                "run_script": context.get("run_script")
            },
            "boundary_conditions": self.boundary_conditions
        }
        
        # Add more specific details if available from agents
        if case_setup.get("boundary_conditions"):
            summary["case_setup_boundary_conditions"] = case_setup["boundary_conditions"]
        if numerical_setup.get("mask_aware_schemes"):
            summary["numerical_setup_mask_aware"] = numerical_setup["mask_aware_schemes"]
        if execution_result.get("forcing_system_details"):
            summary["execution_script_forcing"] = execution_result["forcing_system_details"]
            
        return summary
    
    def get_flow_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of the internal flow endpoint
        
        Returns:
            Dictionary describing supported flow types and features
        """
        
        return {
            "supported_flow_types": [
                "supersonic_nozzle",
                "rocket_engine", 
                "combustion_chamber",
                "shock_tube",
                "duct_flow"
            ],
            "geometry_types": [
                "converging_diverging",
                "straight_duct",
                "combustor",
                "custom"
            ],
            "boundary_conditions": [
                "SIMPLE_INFLOW",
                "SIMPLE_OUTFLOW", 
                "DIRICHLET",
                "WALL",
                "SYMMETRY"
            ],
            "physics_capabilities": {
                "compressible_flow": True,
                "supersonic_flow": True,
                "shock_capturing": True,
                "viscous_effects": True,
                "heat_transfer": True,
                "high_temperature": True,
                "multi_species": False  # Future capability
            },
            "numerical_methods": {
                "reconstruction": ["WENO5-Z", "WENO7", "WENO-CU6"],
                "riemann_solvers": ["HLLC", "ROE", "AUSM"],
                "time_integration": ["RK3", "RK2", "EULER"]
            }
        } 