#!/usr/bin/env python3
"""
JAX-Fluids Run Script Generator Agent
Based on official JAX-Fluids documentation and examples

Generates production-ready run.py scripts that follow exact JAX-Fluids patterns
from the examples and documentation.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import google.generativeai as genai

logger = logging.getLogger(__name__)

@dataclass
class RunScriptConfig:
    """Configuration for run.py script generation"""
    
    # Input files (required)
    case_setup_file: str  # e.g., "case.json"
    numerical_setup_file: str  # e.g., "numerical_setup.json"
    
    # Simulation settings
    cuda_device: str = "0"  # GPU device or "" for CPU
    enable_postprocess: bool = True
    enable_visualization: bool = True
    
    # Output settings
    output_quantities: List[str] = None  # e.g., ["density", "velocity", "pressure"]
    animation_plane: str = "xy"  # "xy", "xz", "yz"
    figure_dpi: int = 200
    animation_interval: int = 100
    
    # Advanced settings
    custom_imports: List[str] = None
    custom_postprocess: str = None

class JAXFluidsRunGenerator:
    """
    Generates JAX-Fluids run.py scripts based on official documentation patterns
    """
    
    def __init__(self, gemini_api_key: str = None):
        """Initialize the run script generator"""
        
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        else:
            self.model = None
            logger.warning("No Gemini API key found - using template-based generation")
    
    def analyze_simulation_config(
        self, 
        case_config: Dict[str, Any],
        numerical_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze configuration files to determine simulation characteristics"""
        
        analysis = {
            'dimension': self._determine_dimension(case_config),
            'physics': self._analyze_physics(numerical_config),
            'output_quantities': self._determine_output_quantities(numerical_config),
            'levelset_enabled': numerical_config.get('active_physics', {}).get('is_levelset', False),
            'visualization_type': 'external_flow'  # Our specific use case
        }
        
        return analysis
    
    def _determine_dimension(self, case_config: Dict[str, Any]) -> str:
        """Determine if this is 1D, 2D, or 3D simulation"""
        
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
    
    def _analyze_physics(self, numerical_config: Dict[str, Any]) -> Dict[str, bool]:
        """Analyze active physics from numerical setup"""
        
        active_physics = numerical_config.get('active_physics', {})
        
        return {
            'viscous': active_physics.get('is_viscous_flux', False),
            'heat': active_physics.get('is_heat_flux', False),
            'levelset': active_physics.get('is_levelset', False),
            'convective': active_physics.get('is_convective_flux', True),
        }
    
    def _determine_output_quantities(self, numerical_config: Dict[str, Any]) -> List[str]:
        """Determine appropriate output quantities based on physics"""
        
        quantities = ["density", "velocity", "pressure"]
        
        physics = self._analyze_physics(numerical_config)
        
        if physics['levelset']:
            quantities.extend(["levelset", "volume_fraction"])
            
        # Add quantities for external flow visualization
        quantities.extend(["mach_number", "schlieren"])
        
        return quantities
    
    def generate_run_script(
        self,
        config: RunScriptConfig,
        case_config: Dict[str, Any],
        numerical_config: Dict[str, Any],
        output_path: str
    ) -> str:
        """Generate a complete JAX-Fluids run.py script"""
        
        print(f"🔧 Generating run script for: {config.case_setup_file}")
        
        # Analyze the simulation
        analysis = self.analyze_simulation_config(case_config, numerical_config)
        print(f"📊 Analysis: {analysis['dimension']} simulation with {len(analysis['output_quantities'])} quantities")
        
        # Generate the script content
        script_content = self._generate_script_content(config, analysis)
        print(f"📝 Generated script content length: {len(script_content)} characters")
        
        if len(script_content) < 100:
            print("⚠️ WARNING: Generated script content seems too short!")
            print(f"Content preview: {repr(script_content[:200])}")
        
        # Write to file
        output_file = Path(output_path)
        print(f"💾 Writing to: {output_file}")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(script_content)
            print(f"✅ File written successfully")
        except Exception as e:
            print(f"❌ Failed to write file: {e}")
            raise
        
        # Make executable
        try:
            os.chmod(output_file, 0o755)
        except Exception as e:
            print(f"⚠️ Could not make file executable: {e}")
        
        logger.info(f"Generated JAX-Fluids run script: {output_file}")
        return str(output_file)
    
    def _generate_script_content(
        self, 
        config: RunScriptConfig, 
        analysis: Dict[str, Any]
    ) -> str:
        """Generate the actual Python script content"""
        
        dimension = analysis['dimension']
        physics = analysis['physics']
        quantities = analysis['output_quantities']
        
        print(f"🔍 Generating content for {dimension} simulation")
        
        # Generate sections first
        postprocess_section = self._generate_postprocess_section(config, analysis)
        visualization_section = self._generate_visualization_section(config, analysis)
        custom_imports = self._generate_custom_imports(config)
        
        print(f"📦 Generated postprocess section: {len(postprocess_section)} chars")
        print(f"📦 Generated visualization section: {len(visualization_section)} chars")
        
        # Base template following JAX-Fluids examples
        script_template = self._get_base_template()
        
        print(f"📄 Base template length: {len(script_template)} chars")
        
        try:
            # Customize based on analysis
            script_content = script_template.format(
                cuda_device=config.cuda_device,
                case_setup_file=config.case_setup_file,
                numerical_setup_file=config.numerical_setup_file,
                output_quantities=self._format_quantities_list(quantities),
                postprocess_section=postprocess_section,
                visualization_section=visualization_section,
                custom_imports=custom_imports,
            )
            print(f"✅ Template formatting successful")
        except Exception as e:
            print(f"❌ Template formatting failed: {e}")
            # Return a basic working script as fallback
            script_content = self._get_fallback_script(config)
        
        return script_content
    
    def _get_base_template(self) -> str:
        """Get the base JAX-Fluids script template from examples"""
        
        return '''#!/usr/bin/env python3
"""
JAX-Fluids External Flow Simulation
Generated by External Flow Endpoint

This script follows the standard JAX-Fluids simulation pattern
as documented in the examples and documentation.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "{cuda_device}"

import numpy as np
import matplotlib.pyplot as plt
{custom_imports}

from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids_postprocess import load_data, create_2D_animation, create_2D_figure

def main():
    """Main simulation function following JAX-Fluids standard pattern"""
    
    print("🚀 Starting JAX-Fluids External Flow Simulation")
    print("=" * 60)
    
    try:
        # SETUP SIMULATION - Standard JAX-Fluids pattern
        print("📋 Setting up simulation managers...")
        input_manager = InputManager("{case_setup_file}", "{numerical_setup_file}")
        initialization_manager = InitializationManager(input_manager)
        sim_manager = SimulationManager(input_manager)
        print("✅ Simulation managers initialized")
        
        # RUN SIMULATION - Standard JAX-Fluids execution
        print("🔄 Running simulation...")
        jxf_buffers = initialization_manager.initialization()
        sim_manager.simulate(jxf_buffers)
        print("✅ Simulation completed successfully!")
        
{postprocess_section}
        
{visualization_section}
        
        print("\\n" + "=" * 60)
        print("🎉 JAX-Fluids simulation completed successfully!")
        print("📁 Results saved to:", sim_manager.output_writer.save_path_domain)
        
    except Exception as e:
        print(f"❌ Simulation failed: {{e}}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
'''
    
    def _generate_postprocess_section(
        self, 
        config: RunScriptConfig, 
        analysis: Dict[str, Any]
    ) -> str:
        """Generate post-processing section based on simulation type"""
        
        if not config.enable_postprocess:
            return "        # Post-processing disabled"
        
        quantities = analysis['output_quantities']
        
        section = f'''        # LOAD DATA - Standard JAX-Fluids post-processing
        print("📊 Loading simulation data...")
        path = sim_manager.output_writer.save_path_domain
        quantities = {self._format_quantities_list(quantities)}
        jxf_data = load_data(path, quantities)
        
        cell_centers = jxf_data.cell_centers
        data = jxf_data.data
        times = jxf_data.times
        print(f"✅ Loaded {{len(quantities)}} quantities over {{len(times)}} time steps")'''
        
        # Add levelset masking for external flow
        if analysis['physics'].get('levelset', False):
            section += '''
        
        # PREPARE DATA - Mask solid regions for external flow visualization
        if "volume_fraction" in data:
            mask_fluid = data["volume_fraction"] > 0.0
            mask_solid = 1.0 - mask_fluid
        else:
            mask_fluid = None
            mask_solid = None'''
        
        return section
    
    def _generate_visualization_section(
        self, 
        config: RunScriptConfig, 
        analysis: Dict[str, Any]
    ) -> str:
        """Generate visualization section based on simulation characteristics"""
        
        if not config.enable_visualization:
            return "        # Visualization disabled"
        
        dimension = analysis['dimension']
        physics = analysis['physics']
        
        if dimension == "1D":
            return self._generate_1d_visualization(config, analysis)
        elif dimension == "2D":
            return self._generate_2d_visualization(config, analysis)
        else:
            return self._generate_3d_visualization(config, analysis)
    
    def _generate_2d_visualization(
        self, 
        config: RunScriptConfig, 
        analysis: Dict[str, Any]
    ) -> str:
        """Generate 2D visualization following NACA airfoil example pattern"""
        
        levelset_enabled = analysis['physics'].get('levelset', False)
        
        # Base plot dictionary for external flow
        plot_setup = '''        # PLOT SETUP - External flow visualization
        nrows_ncols = (2,2)
        plot_dict = {{'''
        
        if levelset_enabled:
            plot_setup += '''
            "density": np.ma.masked_where(mask_solid, data["density"]),
            "pressure": np.ma.masked_where(mask_solid, data["pressure"]),
            "mach_number": np.clip(np.ma.masked_where(mask_solid, data["mach_number"]), 0.0, 3.0),
            "schlieren": np.clip(np.ma.masked_where(mask_solid, data["schlieren"]), 1e0, 5e2)'''
        else:
            plot_setup += '''
            "density": data["density"],
            "pressure": data["pressure"],
            "mach_number": np.clip(data["mach_number"], 0.0, 3.0),
            "schlieren": np.clip(data["schlieren"], 1e0, 5e2)'''
        
        plot_setup += '''
        }'''
        
        visualization = plot_setup + f'''
        
        # CREATE ANIMATION - Following JAX-Fluids examples
        print("🎬 Creating animation...")
        create_2D_animation(
            plot_dict, 
            cell_centers, 
            times, 
            nrows_ncols=nrows_ncols, 
            plane="{config.animation_plane}", plane_value=0.0,
            interval={config.animation_interval})
        print("✅ Animation saved")
        
        # CREATE FIGURE - Final state visualization
        print("📸 Creating final state figure...")
        create_2D_figure(
            plot_dict,
            nrows_ncols=nrows_ncols,
            cell_centers=cell_centers, 
            plane="{config.animation_plane}", plane_value=0.0,
            dpi={config.figure_dpi}, save_fig="external_flow_result.png")
        print("✅ Figure saved as external_flow_result.png")'''
        
        return visualization
    
    def _generate_1d_visualization(
        self, 
        config: RunScriptConfig, 
        analysis: Dict[str, Any]
    ) -> str:
        """Generate 1D visualization following shock tube examples"""
        
        return '''        # CREATE 1D VISUALIZATION
        print("📊 Creating 1D plots...")
        from jaxfluids_postprocess import create_1D_animation, create_1D_figure
        
        plot_dict = {
            "density": data["density"], 
            "velocityX": data["velocity"][:,0],
            "pressure": data["pressure"]
        }
        nrows_ncols = (1,3)
        
        # CREATE ANIMATION
        create_1D_animation(
            plot_dict,
            cell_centers,
            times,
            nrows_ncols=nrows_ncols,
            interval=200)
        
        # CREATE FIGURE
        create_1D_figure(
            plot_dict,
            cell_centers=cell_centers,
            nrows_ncols=nrows_ncols,
            axis="x", axis_values=(0,0), 
            save_fig="external_flow_1d.png")
        print("✅ 1D visualization completed")'''
    
    def _generate_3d_visualization(
        self, 
        config: RunScriptConfig, 
        analysis: Dict[str, Any]
    ) -> str:
        """Generate 3D visualization following TGV example"""
        
        return f'''        # CREATE 3D VISUALIZATION - 2D slices
        print("📊 Creating 3D visualization (2D slices)...")
        
        plot_dict = {{
            "u": data["velocity"][:,0],
            "v": data["velocity"][:,1],
            "w": data["velocity"][:,2],
        }}
        
        nrows_ncols = (1,3)
        os.makedirs("images", exist_ok=True)
        create_2D_animation(
            plot_dict, 
            cell_centers, 
            times, 
            nrows_ncols=nrows_ncols, 
            plane="{config.animation_plane}",
            plane_value=0.0, 
            interval={config.animation_interval},
            save_png="images", 
            fig_args={{"figsize": (15,5)}}, 
            dpi={config.figure_dpi})
        print("✅ 3D visualization completed")'''
    
    def _format_quantities_list(self, quantities: List[str]) -> str:
        """Format quantities list for Python code"""
        return repr(quantities)
    
    def _generate_custom_imports(self, config: RunScriptConfig) -> str:
        """Generate any custom imports needed"""
        
        if config.custom_imports:
            return '\n'.join(config.custom_imports)
        return ""

    def _get_fallback_script(self, config: RunScriptConfig) -> str:
        """Generate a basic fallback JAX-Fluids script when template formatting fails"""
        
        return f'''#!/usr/bin/env python3
"""
JAX-Fluids External Flow Simulation
Generated by External Flow Endpoint (Fallback Mode)

This script follows the standard JAX-Fluids simulation pattern.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "{config.cuda_device}"

import numpy as np
import matplotlib.pyplot as plt

from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids_postprocess import load_data, create_2D_animation, create_2D_figure

def main():
    """Main simulation function following JAX-Fluids standard pattern"""
    
    print("🚀 Starting JAX-Fluids External Flow Simulation")
    print("=" * 60)
    
    try:
        # SETUP SIMULATION - Standard JAX-Fluids pattern
        print("📋 Setting up simulation managers...")
        input_manager = InputManager("{config.case_setup_file}", "{config.numerical_setup_file}")
        initialization_manager = InitializationManager(input_manager)
        sim_manager = SimulationManager(input_manager)
        print("✅ Simulation managers initialized")
        
        # RUN SIMULATION - Standard JAX-Fluids execution
        print("🔄 Running simulation...")
        jxf_buffers = initialization_manager.initialization()
        sim_manager.simulate(jxf_buffers)
        print("✅ Simulation completed successfully!")
        
        # BASIC POST-PROCESSING
        print("📊 Loading simulation data...")
        path = sim_manager.output_writer.save_path_domain
        quantities = ["density", "velocity", "pressure"]
        jxf_data = load_data(path, quantities)
        
        cell_centers = jxf_data.cell_centers
        data = jxf_data.data
        times = jxf_data.times
        print(f"✅ Loaded {{len(quantities)}} quantities over {{len(times)}} time steps")
        
        # BASIC VISUALIZATION
        print("📸 Creating visualization...")
        plot_dict = {{
            "density": data["density"],
            "pressure": data["pressure"],
        }}
        nrows_ncols = (1,2)
        
        create_2D_figure(
            plot_dict,
            nrows_ncols=nrows_ncols,
            cell_centers=cell_centers, 
            plane="xy", plane_value=0.0,
            dpi=200, save_fig="external_flow_result.png")
        print("✅ Figure saved as external_flow_result.png")
        
        print("\\n" + "=" * 60)
        print("🎉 JAX-Fluids simulation completed successfully!")
        print("📁 Results saved to:", sim_manager.output_writer.save_path_domain)
        
    except Exception as e:
        print(f"❌ Simulation failed: {{e}}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
'''

def create_jaxfluids_run_script(
    case_setup_path: str,
    numerical_setup_path: str,
    output_directory: str,
    gemini_api_key: str = None
) -> str:
    """
    Convenience function to generate a JAX-Fluids run script
    
    Args:
        case_setup_path: Path to case setup JSON file
        numerical_setup_path: Path to numerical setup JSON file  
        output_directory: Directory where run.py will be saved
        gemini_api_key: Optional Gemini API key for enhanced generation
        
    Returns:
        str: Path to generated run.py file
    """
    
    # Load configuration files
    with open(case_setup_path, 'r', encoding='utf-8') as f:
        case_config = json.load(f)
    
    with open(numerical_setup_path, 'r', encoding='utf-8') as f:
        numerical_config = json.load(f)
    
    # Determine case setup filename for the script
    case_filename = Path(case_setup_path).name
    numerical_filename = Path(numerical_setup_path).name
    
    # Create configuration
    config = RunScriptConfig(
        case_setup_file=case_filename,
        numerical_setup_file=numerical_filename,
        cuda_device="0",  # Default to GPU
        enable_postprocess=True,
        enable_visualization=True,
        animation_plane="xy",
        figure_dpi=200,
        animation_interval=100
    )
    
    # Generate the script
    generator = JAXFluidsRunGenerator(gemini_api_key)
    
    output_path = Path(output_directory) / "run.py"
    
    script_path = generator.generate_run_script(
        config=config,
        case_config=case_config,
        numerical_config=numerical_config,
        output_path=str(output_path)
    )
    
    return script_path 