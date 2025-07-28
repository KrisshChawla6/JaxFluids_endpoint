#!/usr/bin/env python3
"""
CFD Parameter Agent - Production Ready
Maintains compatibility with existing interface while using the working wind tunnel solution
"""

import os
import sys
import json
import subprocess
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

# Import our working components
from wind_tunnel_generator import (
    WindTunnelConfigGenerator, 
    WindTunnelConfig, 
    FlowType, 
    TurbulenceModel,
    create_preset_configs
)

class CFDParameterAgent:
    """
    Production-ready CFD Parameter Agent
    Maintains compatibility with existing interface while using working solutions
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the CFD Parameter Agent"""
        self.api_key = api_key
        self.config_generator = WindTunnelConfigGenerator()
        self.presets = create_preset_configs()
        self.working_directory = os.getcwd()
        
        print("🚀 CFD Parameter Agent - Production Ready")
        print("✅ Using validated wind tunnel solution")
    
    def create_simulation_from_prompt(self, prompt: str, output_dir: str = None) -> str:
        """
        Create simulation from natural language prompt
        Maintains compatibility with existing interface
        """
        
        print(f"🗣️  Processing prompt: '{prompt}'")
        
        # Parse the prompt to extract parameters
        config_params = self._parse_prompt_to_parameters(prompt)
        
        # Generate configuration
        config = self._create_config_from_parameters(config_params)
        
        # Create output directory
        if output_dir is None:
            timestamp = int(time.time())
            output_dir = f"simulation_{timestamp}"
        
        sim_dir = os.path.join(self.working_directory, "simulations", output_dir)
        os.makedirs(sim_dir, exist_ok=True)
        
        # Generate config file
        config_file = os.path.join(sim_dir, "config.cfg")
        self.config_generator.generate_config(config, config_file)
        
        print(f"✅ Created simulation directory: {sim_dir}")
        print(f"📁 Configuration file: {config_file}")
        
        return sim_dir
    
    def run_simulation_from_prompt(self, prompt: str, mesh_file: str = None) -> bool:
        """
        Run simulation from natural language prompt
        Maintains compatibility with existing interface
        """
        
        # Create simulation
        sim_dir = self.create_simulation_from_prompt(prompt)
        
        # Use default mesh if not specified
        if mesh_file is None:
            mesh_file = "propeller_wind_tunnel_cfd.su2"
        
        # Run simulation
        return self.run_simulation(sim_dir, mesh_file)
    
    def run_simulation(self, sim_dir: str, mesh_file: str = None) -> bool:
        """Run SU2 simulation in the specified directory"""
        
        config_file = os.path.join(sim_dir, "config.cfg")
        
        if not os.path.exists(config_file):
            print(f"❌ Config file not found: {config_file}")
            return False
        
        # Validate mesh file
        if mesh_file and not os.path.exists(mesh_file):
            print(f"❌ Mesh file not found: {mesh_file}")
            return False
        
        print(f"🚀 Running SU2 simulation in: {sim_dir}")
        
        try:
            # Run SU2_CFD
            result = subprocess.run(
                ['SU2_CFD', 'config.cfg'],
                cwd=sim_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            success = result.returncode == 0
            
            if success:
                print(f"✅ Simulation completed successfully!")
                
                # Check output files
                output_files = self._check_simulation_outputs(sim_dir)
                if output_files:
                    print(f"📁 Output files generated:")
                    for output_file in output_files:
                        print(f"   • {output_file}")
                
            else:
                print(f"❌ Simulation failed with return code: {result.returncode}")
                if result.stderr:
                    print(f"Error details: {result.stderr[:500]}...")
            
            return success
            
        except subprocess.TimeoutExpired:
            print(f"❌ Simulation timed out after 1 hour")
            return False
        except FileNotFoundError:
            print(f"❌ SU2_CFD not found. Please ensure SU2 is installed and in PATH.")
            return False
        except Exception as e:
            print(f"❌ Error running simulation: {e}")
            return False
    
    def _parse_prompt_to_parameters(self, prompt: str) -> Dict[str, Any]:
        """Parse natural language prompt to extract CFD parameters"""
        
        prompt_lower = prompt.lower()
        params = {}
        
        # Flow type detection
        if any(word in prompt_lower for word in ['euler', 'inviscid', 'potential']):
            params['flow_type'] = FlowType.EULER
        elif any(word in prompt_lower for word in ['rans', 'turbulent']):
            params['flow_type'] = FlowType.RANS
            params['turbulence_model'] = TurbulenceModel.SA
        elif any(word in prompt_lower for word in ['navier', 'viscous']):
            params['flow_type'] = FlowType.NAVIER_STOKES
        else:
            params['flow_type'] = FlowType.EULER  # Default
        
        # Mach number detection
        import re
        mach_match = re.search(r'mach\s*(?:number\s*)?(?:of\s*)?(\d+\.?\d*)', prompt_lower)
        if mach_match:
            params['mach_number'] = float(mach_match.group(1))
        elif 'subsonic' in prompt_lower:
            params['mach_number'] = 0.3
        elif 'transonic' in prompt_lower:
            params['mach_number'] = 0.8
        elif 'supersonic' in prompt_lower:
            params['mach_number'] = 1.5
        else:
            params['mach_number'] = 0.3  # Default
        
        # Angle of attack detection
        aoa_match = re.search(r'(?:angle\s*of\s*attack|aoa)\s*(?:of\s*)?(\d+\.?\d*)', prompt_lower)
        if aoa_match:
            params['angle_of_attack'] = float(aoa_match.group(1))
        elif 'zero' in prompt_lower and 'angle' in prompt_lower:
            params['angle_of_attack'] = 0.0
        else:
            params['angle_of_attack'] = 0.0  # Default
        
        # Reynolds number detection
        re_match = re.search(r'reynolds\s*(?:number\s*)?(?:of\s*)?(\d+(?:\.\d+)?(?:e[+-]?\d+)?)', prompt_lower)
        if re_match:
            params['reynolds_number'] = float(re_match.group(1))
        elif 'low reynolds' in prompt_lower:
            params['reynolds_number'] = 100000.0
        elif 'high reynolds' in prompt_lower:
            params['reynolds_number'] = 1000000.0
        else:
            params['reynolds_number'] = 1000000.0  # Default
        
        # Iterations detection
        iter_match = re.search(r'(\d+)\s*iterations?', prompt_lower)
        if iter_match:
            params['max_iterations'] = int(iter_match.group(1))
        elif 'quick' in prompt_lower or 'fast' in prompt_lower:
            params['max_iterations'] = 50
        elif 'detailed' in prompt_lower or 'accurate' in prompt_lower:
            params['max_iterations'] = 500
        else:
            params['max_iterations'] = 100  # Default
        
        # Project detection (for compatibility)
        if 'project 3' in prompt_lower or 'propeller' in prompt_lower:
            params['mesh_filename'] = 'propeller_wind_tunnel_cfd.su2'
        elif 'project 2' in prompt_lower or 'eppler' in prompt_lower:
            params['mesh_filename'] = 'eppler_mesh.su2'
        elif 'project 1' in prompt_lower or 'airfoil' in prompt_lower:
            params['mesh_filename'] = 'airfoil_mesh.su2'
        else:
            params['mesh_filename'] = 'propeller_wind_tunnel_cfd.su2'  # Default
        
        return params
    
    def _create_config_from_parameters(self, params: Dict[str, Any]) -> WindTunnelConfig:
        """Create WindTunnelConfig from parsed parameters"""
        
        return WindTunnelConfig(
            flow_type=params.get('flow_type', FlowType.EULER),
            mach_number=params.get('mach_number', 0.3),
            reynolds_number=params.get('reynolds_number', 1000000.0),
            angle_of_attack=params.get('angle_of_attack', 0.0),
            max_iterations=params.get('max_iterations', 100),
            turbulence_model=params.get('turbulence_model', TurbulenceModel.NONE),
            mesh_filename=params.get('mesh_filename', 'propeller_wind_tunnel_cfd.su2'),
            convergence_residual=1e-8,
            cfl_number=1.0
        )
    
    def _check_simulation_outputs(self, sim_dir: str) -> List[str]:
        """Check what output files were generated"""
        
        output_files = []
        
        # Common SU2 output files
        possible_outputs = [
            'flow.vtu',
            'history.csv',
            'restart_flow.dat',
            'surface_flow.csv',
            'forces_breakdown.dat'
        ]
        
        for output_file in possible_outputs:
            file_path = os.path.join(sim_dir, output_file)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                output_files.append(f"{output_file} ({file_size} bytes)")
        
        return output_files
    
    def get_available_presets(self) -> Dict[str, str]:
        """Get available preset configurations"""
        
        descriptions = {
            "euler_low_speed": "Inviscid flow at low Mach number (0.15), good for basic aerodynamics",
            "euler_transonic": "Inviscid transonic flow (Mach 0.8), for compressibility effects",
            "rans_low_reynolds": "Turbulent flow at low Reynolds number (100k), with SA turbulence model",
            "rans_high_reynolds": "Turbulent flow at high Reynolds number (1M), with SST turbulence model",
            "propeller_analysis": "Optimized for propeller analysis in wind tunnel"
        }
        
        return descriptions
    
    def create_preset_simulation(self, preset_name: str, output_dir: str = None) -> str:
        """Create simulation from preset configuration"""
        
        if preset_name not in self.presets:
            available = list(self.presets.keys())
            raise ValueError(f"Preset '{preset_name}' not found. Available: {available}")
        
        config = self.presets[preset_name]
        
        # Create output directory
        if output_dir is None:
            timestamp = int(time.time())
            output_dir = f"preset_{preset_name}_{timestamp}"
        
        sim_dir = os.path.join(self.working_directory, "simulations", output_dir)
        os.makedirs(sim_dir, exist_ok=True)
        
        # Generate config file
        config_file = os.path.join(sim_dir, "config.cfg")
        self.config_generator.generate_config(config, config_file)
        
        print(f"✅ Created preset simulation: {preset_name}")
        print(f"📁 Directory: {sim_dir}")
        
        return sim_dir

# Compatibility functions for existing code
def create_config_with_extracted_markers(mesh_file_path: str, **kwargs) -> WindTunnelConfig:
    """
    Compatibility function for existing code
    Creates config with extracted markers (now using working solution)
    """
    
    return WindTunnelConfig(
        mesh_filename=os.path.basename(mesh_file_path),
        flow_type=FlowType(kwargs.get('solver_type', 'EULER')),
        mach_number=kwargs.get('mach_number', 0.3),
        angle_of_attack=kwargs.get('angle_of_attack', 0.0),
        reynolds_number=kwargs.get('reynolds_number', 1000000.0),
        max_iterations=kwargs.get('max_iterations', 100),
        cfl_number=kwargs.get('cfl_number', 1.0),
        convergence_residual=kwargs.get('convergence_residual', 1e-8)
    )

class WindTunnelSimulation:
    """
    Compatibility class for existing code
    Wraps the new functionality with the old interface
    """
    
    def __init__(self):
        self.config_generator = WindTunnelConfigGenerator()
    
    def create_simulation(self, config: WindTunnelConfig, sim_name: str) -> str:
        """Create simulation directory and config file"""
        
        sim_dir = os.path.join("simulations", sim_name)
        os.makedirs(sim_dir, exist_ok=True)
        
        config_file = os.path.join(sim_dir, "config.cfg")
        self.config_generator.generate_config(config, config_file)
        
        return sim_dir
    
    def run_simulation(self, sim_dir: str) -> bool:
        """Run simulation in the specified directory"""
        
        agent = CFDParameterAgent()
        return agent.run_simulation(sim_dir)

def main():
    """Main function for testing"""
    
    print("🚀 CFD Parameter Agent - Production Ready")
    print("=" * 50)
    
    # Initialize agent
    agent = CFDParameterAgent()
    
    # Test with example prompts
    test_prompts = [
        "Project 3 propeller analysis at 8 degrees",
        "High speed propeller simulation with Mach 0.2",
        "Low Reynolds number flow analysis for Project 3",
        "Quick Euler analysis for propeller at Mach 0.3"
    ]
    
    print("\n🧪 Testing with example prompts:")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}] Testing: '{prompt}'")
        try:
            sim_dir = agent.create_simulation_from_prompt(prompt, f"test_{i}")
            print(f"   ✅ Created: {sim_dir}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ Production endpoint ready!")

if __name__ == "__main__":
    main() 