#!/usr/bin/env python3
"""
JAX-Fluids External Flow API - Main Entry Point
Multimodal AI-driven system for 3D external flow simulations with immersed boundaries
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from external_flow_orchestrator import ExternalFlowOrchestrator
from sdf_integration import integrate_sdf_with_case_setup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExternalFlowRequest:
    """Main request for external flow simulation"""
    
    # User input
    user_prompt: str
    multimodal_context: Optional[Dict[str, Any]] = None
    
    # SDF integration (auto-populated by backend or custom path)
    sdf_file_path: Optional[str] = None
    sdf_metadata: Optional[Dict[str, Any]] = None
    custom_sdf_directory: Optional[str] = None  # Custom SDF search directory
    
    # Optional overrides
    domain_bounds: Optional[List[float]] = None
    resolution: Optional[List[int]] = None
    flow_conditions: Optional[Dict[str, Any]] = None
    
    # Output configuration
    output_directory: str = "external_flow_simulation"
    simulation_name: Optional[str] = None
    
@dataclass
class ExternalFlowResponse:
    """Response from external flow simulation"""
    
    success: bool
    message: str
    
    # Generated files
    simulation_directory: Optional[str] = None
    numerical_setup_file: Optional[str] = None
    case_setup_file: Optional[str] = None
    run_script_file: Optional[str] = None
    
    # Execution results
    execution_started: bool = False
    execution_completed: bool = False
    results_directory: Optional[str] = None
    
    # Metadata
    processing_time: float = 0.0
    extracted_parameters: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None

class ExternalFlowAPI:
    """
    Main API for JAX-Fluids External Flow Simulations
    Handles multimodal context, SDF integration, and orchestrates the 3-agent system
    """
    
    def __init__(self, gemini_api_key: str = None):
        """Initialize the External Flow API"""
        
        # Load API key
        if gemini_api_key is None:
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            if not gemini_api_key:
                raise ValueError("Gemini API key required. Set GEMINI_API_KEY environment variable.")
        
        self.orchestrator = ExternalFlowOrchestrator(gemini_api_key)
        self.logger = logger
        
        print("🌪️ JAX-Fluids External Flow API Initialized")
        print("✅ AI Orchestrator Ready")
        print("✅ Subsonic Wind Tunnel Experts Ready")
    
    def process_external_flow_request(self, request: ExternalFlowRequest) -> ExternalFlowResponse:
        """
        Main entry point for external flow simulation
        1. Auto-detect and load SDF from immersed boundary endpoint
        2. Parse user prompt and multimodal context
        3. Orchestrate 3-agent system for JAX-Fluids setup
        4. Generate and execute simulation
        """
        
        start_time = time.time()
        
        try:
            print("\n" + "="*80)
            print("🚀 JAX-FLUIDS EXTERNAL FLOW SIMULATION")
            print("="*80)
            print(f"📝 User Prompt: {request.user_prompt}")
            if request.multimodal_context:
                print(f"🖼️ Multimodal Context: {len(request.multimodal_context)} elements")
            
            # Step 1: Auto-detect SDF file if not provided
            if not request.sdf_file_path:
                request.sdf_file_path, request.sdf_metadata = self._auto_detect_sdf(request.custom_sdf_directory)
                print(f"🎯 Auto-detected SDF: {request.sdf_file_path}")
            
            # Step 2: Create output directory
            output_dir = Path(request.output_directory)
            output_dir.mkdir(exist_ok=True)
            
            # Step 3: Orchestrate the 3-agent system
            print(f"\n🤖 ORCHESTRATING 3-AGENT SUBSONIC WIND TUNNEL SYSTEM")
            print("-" * 60)
            
            orchestration_result = self.orchestrator.orchestrate_simulation(
                user_prompt=request.user_prompt,
                multimodal_context=request.multimodal_context,
                sdf_file_path=request.sdf_file_path,
                sdf_metadata=request.sdf_metadata,
                output_directory=str(output_dir),
                simulation_name=request.simulation_name,
                domain_bounds=request.domain_bounds,
                resolution=request.resolution,
                flow_conditions=request.flow_conditions
            )
            
            processing_time = time.time() - start_time
            
            if orchestration_result['success']:
                return ExternalFlowResponse(
                    success=True,
                    message="External flow simulation setup completed successfully",
                    simulation_directory=orchestration_result['simulation_directory'],
                    numerical_setup_file=orchestration_result['numerical_setup_file'],
                    case_setup_file=orchestration_result['case_setup_file'],
                    run_script_file=orchestration_result['run_script_file'],
                    execution_started=orchestration_result.get('execution_started', False),
                    execution_completed=orchestration_result.get('execution_completed', False),
                    results_directory=orchestration_result.get('results_directory'),
                    processing_time=processing_time,
                    extracted_parameters=orchestration_result.get('extracted_parameters')
                )
            else:
                return ExternalFlowResponse(
                    success=False,
                    message=f"Simulation setup failed: {orchestration_result['message']}",
                    processing_time=processing_time,
                    error_details=orchestration_result.get('error_details')
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"External flow API error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return ExternalFlowResponse(
                success=False,
                message=error_msg,
                processing_time=processing_time,
                error_details=str(e)
            )
    
    def _auto_detect_sdf(self, custom_sdf_directory: Optional[str] = None) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Auto-detect the latest SDF file from the immersed boundary endpoint
        Uses the 3-slot history window system
        """
        
        # Look for SDF files in the immersed boundary endpoint
        sdf_base_path = Path("../immersed_boundary_endpoint_final/sdf_files")
        
        if custom_sdf_directory:
            sdf_base_path = Path(custom_sdf_directory)
        
        if not sdf_base_path.exists():
            # Fallback: check current directory
            sdf_base_path = Path("sdf_files")
        
        if not sdf_base_path.exists():
            self.logger.warning("No SDF files directory found. SDF integration disabled.")
            return None, None
        
        try:
            # Find the most recent run directory (3-slot history window)
            run_dirs = [d for d in sdf_base_path.iterdir() if d.is_dir() and len(d.name) == 15]
            
            if not run_dirs:
                self.logger.warning("No SDF run directories found.")
                return None, None
            
            # Sort by timestamp (newest first)
            latest_run = max(run_dirs, key=lambda x: x.name)
            
            # Look for SDF matrix file (best for JAX-Fluids)
            sdf_matrix_files = list(latest_run.glob("*_sdf_matrix.npy"))
            
            if sdf_matrix_files:
                sdf_file = str(sdf_matrix_files[0])
                
                # Load metadata if available
                metadata_files = list(latest_run.glob("*_metadata.json"))
                metadata = None
                
                if metadata_files:
                    with open(metadata_files[0], 'r') as f:
                        metadata = json.load(f)
                
                print(f"✅ Found latest SDF: {latest_run.name}")
                return sdf_file, metadata
            
            else:
                self.logger.warning(f"No SDF matrix files found in {latest_run}")
                return None, None
                
        except Exception as e:
            self.logger.error(f"Error auto-detecting SDF: {e}")
            return None, None

# Convenience functions for direct usage
def create_external_flow_simulation(
    user_prompt: str,
    multimodal_context: Optional[Dict[str, Any]] = None,
    output_directory: str = "external_flow_simulation",
    custom_sdf_directory: Optional[str] = None,
    gemini_api_key: str = None
) -> ExternalFlowResponse:
    """
    Convenience function to create an external flow simulation
    
    Args:
        user_prompt: Natural language description of the simulation
        multimodal_context: Optional multimodal context (images, data, etc.)
        output_directory: Where to save the simulation
        custom_sdf_directory: Custom directory to search for SDF files
        gemini_api_key: Gemini API key (uses env var if not provided)
    
    Returns:
        ExternalFlowResponse with simulation results
    """
    
    api = ExternalFlowAPI(gemini_api_key)
    
    request = ExternalFlowRequest(
        user_prompt=user_prompt,
        multimodal_context=multimodal_context,
        output_directory=output_directory,
        custom_sdf_directory=custom_sdf_directory
    )
    
    return api.process_external_flow_request(request)

def main():
    """Example usage and testing"""
    
    # Example 1: Simple external flow
    response = create_external_flow_simulation(
        user_prompt="Create a subsonic external flow simulation around a propeller at Mach 0.3, 5 degrees angle of attack, using WENO5 reconstruction and HLLC Riemann solver",
        output_directory="propeller_external_flow"
    )
    
    if response.success:
        print(f"✅ Simulation created: {response.simulation_directory}")
        print(f"📁 Files generated:")
        print(f"   • Numerical setup: {response.numerical_setup_file}")
        print(f"   • Case setup: {response.case_setup_file}")
        print(f"   • Run script: {response.run_script_file}")
    else:
        print(f"❌ Simulation failed: {response.message}")

if __name__ == "__main__":
    main() 