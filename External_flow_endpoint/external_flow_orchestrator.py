#!/usr/bin/env python3
"""
External Flow Orchestrator - Master Agent
Coordinates the 3-agent subsonic wind tunnel system for JAX-Fluids
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import the 3 specialized agents
from subsonic_windtunnel.numerical_setup_expert import NumericalSetupExpert
from subsonic_windtunnel.case_setup_expert import CaseSetupExpert
from subsonic_windtunnel.execution_agent import ExecutionAgent

logger = logging.getLogger(__name__)

class ExternalFlowOrchestrator:
    """
    Master orchestrator for JAX-Fluids external flow simulations
    Coordinates 3 specialized agents for subsonic wind tunnel setups
    """
    
    def __init__(self, gemini_api_key: str = None):
        """Initialize the orchestrator with the 3 specialized agents"""
        
        self.gemini_api_key = gemini_api_key
        
        # Initialize the 3 specialized agents (they will use env var if key not provided)
        self.numerical_expert = NumericalSetupExpert(gemini_api_key)
        self.case_expert = CaseSetupExpert(gemini_api_key)
        self.execution_agent = ExecutionAgent(gemini_api_key)
        
        print("🎯 External Flow Orchestrator Initialized")
        print("   ✅ Numerical Setup Expert Ready")
        print("   ✅ Case Setup Expert Ready")
        print("   ✅ Execution Agent Ready")
    
    def orchestrate_simulation(
        self,
        user_prompt: str,
        multimodal_context: Optional[Dict[str, Any]] = None,
        sdf_file_path: Optional[str] = None,
        sdf_metadata: Optional[Dict[str, Any]] = None,
        output_directory: str = "simulation",
        simulation_name: Optional[str] = None,
        domain_bounds: Optional[List[float]] = None,
        resolution: Optional[List[int]] = None,
        flow_conditions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Orchestrate the complete simulation setup using 3 specialized agents
        
        Returns:
            Dict with simulation setup results
        """
        
        try:
            # Create simulation name if not provided
            if not simulation_name:
                timestamp = int(time.time())
                simulation_name = f"jaxfluids_external_flow_{timestamp}"
            
            sim_dir = Path(output_directory) / simulation_name
            sim_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"📁 Creating simulation: {simulation_name}")
            print(f"📂 Directory: {sim_dir}")
            
            # Prepare context for all agents
            agent_context = {
                'user_prompt': user_prompt,
                'multimodal_context': multimodal_context,
                'sdf_file_path': sdf_file_path,
                'sdf_metadata': sdf_metadata,
                'simulation_directory': str(sim_dir),
                'simulation_name': simulation_name,
                'domain_bounds': domain_bounds,
                'resolution': resolution,
                'flow_conditions': flow_conditions
            }
            
            # Step 1: Numerical Setup Expert
            print(f"\n🔢 AGENT 1: NUMERICAL SETUP EXPERT")
            print("-" * 40)
            
            numerical_result = self.numerical_expert.generate_numerical_setup(agent_context)
            
            if not numerical_result['success']:
                return {
                    'success': False,
                    'message': f"Numerical setup failed: {numerical_result['message']}",
                    'error_details': numerical_result.get('error_details')
                }
            
            # Step 2: Case Setup Expert  
            print(f"\n🌪️ AGENT 2: CASE SETUP EXPERT")
            print("-" * 40)
            
            # Add numerical setup results to context for case expert
            agent_context['numerical_setup'] = numerical_result['numerical_setup']
            agent_context['numerical_setup_file'] = numerical_result['numerical_setup_file']
            
            case_result = self.case_expert.generate_case_setup(agent_context)
            
            if not case_result['success']:
                return {
                    'success': False,
                    'message': f"Case setup failed: {case_result['message']}",
                    'error_details': case_result.get('error_details')
                }
            
            # Step 2.5: Generate SDF if mesh file is available
            sdf_result = None
            if self._should_generate_sdf(agent_context):
                print(f"\n🔧 SDF GENERATION: IMMERSED BOUNDARY")
                print("-" * 40)
                sdf_result = self._generate_sdf_for_case(case_result['case_setup'], agent_context)
                if sdf_result and sdf_result['success']:
                    agent_context['sdf_file_path'] = sdf_result['sdf_file_path']
                    agent_context['sdf_metadata'] = sdf_result
                    print(f"✅ SDF generated: {sdf_result['sdf_file_path']}")
                else:
                    print(f"⚠️ SDF generation failed, continuing without SDF")

            # Step 3: Execution Agent
            print(f"\n🚀 AGENT 3: EXECUTION AGENT")
            print("-" * 40)
            
            # Add case setup results to context for execution agent
            agent_context['case_setup'] = case_result['case_setup']
            agent_context['case_setup_file'] = case_result['case_setup_file']
            
            # Generate run script using the new method
            execution_result = self.execution_agent.generate_run_script(
                numerical_setup=numerical_result['numerical_setup'],
                case_setup=case_result['case_setup'],
                user_prompt=user_prompt,
                output_dir=agent_context['simulation_directory']
            )
            
            # Save the script to file
            script_path = execution_result['script_path']
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(execution_result['script_content'])
            
            # Make script executable
            import stat
            os.chmod(script_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
            
            # Convert to old format for compatibility
            execution_result = {
                'success': True,
                'message': 'Execution script generated successfully',
                'run_script_file': script_path,
                'execution_started': False,
                'execution_completed': False,
                'results_directory': None,
                'extracted_parameters': execution_result.get('execution_parameters', {}),
                'estimated_runtime': execution_result.get('estimated_runtime'),
                'memory_requirements': execution_result.get('memory_requirements')
            }
            
            if not execution_result['success']:
                return {
                    'success': False,
                    'message': f"Execution setup failed: {execution_result['message']}",
                    'error_details': execution_result.get('error_details')
                }
            
            # Step 4: Create comprehensive summary
            self._create_simulation_summary(
                sim_dir,
                agent_context,
                numerical_result,
                case_result,
                execution_result
            )
            
            print(f"\n✅ ORCHESTRATION COMPLETED SUCCESSFULLY!")
            print(f"📁 Simulation directory: {sim_dir}")
            print(f"📊 All 3 agents completed their tasks")
            
            return {
                'success': True,
                'message': 'External flow simulation setup completed',
                'simulation_directory': str(sim_dir),
                'numerical_setup_file': numerical_result['numerical_setup_file'],
                'case_setup_file': case_result['case_setup_file'],
                'run_script_file': execution_result['run_script_file'],
                'execution_started': execution_result.get('execution_started', False),
                'execution_completed': execution_result.get('execution_completed', False),
                'results_directory': execution_result.get('results_directory'),
                'extracted_parameters': {
                    'numerical_parameters': numerical_result.get('extracted_parameters'),
                    'case_parameters': case_result.get('extracted_parameters'),
                    'execution_parameters': execution_result.get('extracted_parameters')
                }
            }
            
        except Exception as e:
            error_msg = f"Orchestration error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                'success': False,
                'message': error_msg,
                'error_details': str(e)
            }
    
    def _create_simulation_summary(
        self,
        sim_dir: Path,
        context: Dict[str, Any],
        numerical_result: Dict[str, Any],
        case_result: Dict[str, Any],
        execution_result: Dict[str, Any]
    ):
        """Create a comprehensive simulation summary"""
        
        summary = {
            'simulation_info': {
                'name': context['simulation_name'],
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'user_prompt': context['user_prompt'],
                'sdf_integrated': context['sdf_file_path'] is not None
            },
            'agent_results': {
                'numerical_setup_expert': {
                    'success': numerical_result['success'],
                    'file': numerical_result.get('numerical_setup_file'),
                    'parameters_count': len(numerical_result.get('extracted_parameters', {}))
                },
                'case_setup_expert': {
                    'success': case_result['success'],
                    'file': case_result.get('case_setup_file'),
                    'parameters_count': len(case_result.get('extracted_parameters', {}))
                },
                'execution_agent': {
                    'success': execution_result['success'],
                    'file': execution_result.get('run_script_file'),
                    'execution_started': execution_result.get('execution_started', False)
                }
            },
            'generated_files': {
                'numerical_setup.json': numerical_result.get('numerical_setup_file'),
                'case_setup.json': case_result.get('case_setup_file'),
                'run.py': execution_result.get('run_script_file')
            },
            'sdf_integration': {
                'sdf_file': context['sdf_file_path'],
                'sdf_metadata': context['sdf_metadata']
            } if context['sdf_file_path'] else None
        }
        
        summary_file = sim_dir / 'simulation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Simulation summary: {summary_file}")

    def get_available_parameters(self) -> Dict[str, List[str]]:
        """
        Get all available parameters from the 3 agents
        Useful for documentation and debugging
        """
        
        return {
            'numerical_parameters': self.numerical_expert.get_available_parameters(),
            'case_parameters': self.case_expert.get_available_parameters(),
            'execution_parameters': self.execution_agent.get_execution_parameters()
        }
    
    def _should_generate_sdf(self, context: Dict[str, Any]) -> bool:
        """Check if SDF should be generated"""
        
        # Check if mesh file is available
        mesh_file = self._get_propeller_mesh_file()
        if not mesh_file or not os.path.exists(mesh_file):
            return False
            
        # Check if SDF is already provided
        if context.get('sdf_file_path'):
            return False
            
        return True
    
    def _get_propeller_mesh_file(self) -> Optional[str]:
        """Get the propeller mesh file path"""
        
        # Default propeller mesh path as specified by user
        mesh_file = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\mesh\5_bladed_Propeller.STEP_medium_tetrahedral.msh"
        
        if os.path.exists(mesh_file):
            return mesh_file
            
        # Fallback: look for any .msh file in the mesh directory
        mesh_dir = Path(r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\mesh")
        if mesh_dir.exists():
            msh_files = list(mesh_dir.glob("*.msh"))
            if msh_files:
                return str(msh_files[0])
        
        return None
    
    def _generate_sdf_for_case(self, case_setup: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate SDF using the case setup parameters"""
        
        try:
            # Import SDF integration module
            from sdf_integration import integrate_sdf_with_case_setup
            
            mesh_file = self._get_propeller_mesh_file()
            if not mesh_file:
                logger.warning("No mesh file found for SDF generation")
                return None
                
            print(f"🔧 Generating SDF for mesh: {os.path.basename(mesh_file)}")
            
            # Create SDF output directory within simulation directory
            sdf_output_dir = os.path.join(context['simulation_directory'], 'sdf_data')
            os.makedirs(sdf_output_dir, exist_ok=True)
            
            # Generate SDF using case setup parameters
            sdf_result = integrate_sdf_with_case_setup(
                mesh_file=mesh_file,
                case_setup=case_setup,
                output_directory=sdf_output_dir,
                mesh_name="propeller"
            )
            
            return sdf_result
            
        except Exception as e:
            logger.error(f"SDF generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            } 