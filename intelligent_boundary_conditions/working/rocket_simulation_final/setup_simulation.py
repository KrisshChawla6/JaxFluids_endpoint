#!/usr/bin/env python3
"""
JAX-Fluids Rocket Simulation Setup
Complete setup script for production-ready rocket nozzle simulation
"""

import sys
import json
import shutil
from pathlib import Path
import logging

# Add helpers to path
sys.path.append(str(Path(__file__).parent / "helpers"))

class RocketSimulationSetup:
    """Complete setup for rocket nozzle simulation"""
    
    def __init__(self, base_dir: str = "."):
        """Initialize simulation setup"""
        self.base_dir = Path(base_dir)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for setup process"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def create_directory_structure(self):
        """Create the complete directory structure"""
        self.logger.info("📁 Creating directory structure...")
        
        directories = [
            "helpers",
            "config", 
            "masks",
            "output",
            "output/checkpoints",
            "logs"
        ]
        
        for dir_name in directories:
            dir_path = self.base_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"   ✅ {dir_name}")
            
    def copy_helper_files(self):
        """Copy necessary helper files"""
        self.logger.info("📋 Setting up helper files...")
        
        # Copy circular face creator from parent directory
        source_file = self.base_dir.parent / "circular_face_creator.py"
        if source_file.exists():
            target_file = self.base_dir / "helpers" / "circular_face_detector.py"
            shutil.copy2(source_file, target_file)
            self.logger.info(f"   ✅ Copied {source_file.name}")
        else:
            self.logger.warning(f"   ⚠️  Source file not found: {source_file}")
            
    def generate_simulation_parameters(self):
        """Generate simulation parameters configuration"""
        self.logger.info("⚙️ Generating simulation parameters...")
        
        params = {
            "simulation": {
                "name": "Rocket Nozzle Internal Supersonic Flow",
                "description": "Production simulation with virtual boundary conditions",
                "version": "1.0.0"
            },
            "execution": {
                "max_iterations": 100,
                "save_interval": 10,
                "monitoring_interval": 5,
                "checkpoint_interval": 25,
                "convergence_tolerance": 1e-6
            },
            "physics": {
                "flow_type": "internal_supersonic",
                "inlet_conditions": {
                    "pressure_pa": 6900000.0,
                    "temperature_k": 3580.0,
                    "velocity_ms": 50.0
                },
                "outlet_conditions": {
                    "pressure_pa": 101325.0,
                    "temperature_k": 288.15
                }
            },
            "grid": {
                "domain_bounds": [-200.0, -800.0, -800.0, 1800.0, 800.0, 800.0],
                "grid_shape": [128, 64, 64],
                "description": "Cartesian grid covering rocket nozzle domain"
            },
            "output": {
                "fields": ["density", "velocity", "pressure", "temperature", "mach_number"],
                "format": "numpy",
                "compression": True
            }
        }
        
        params_file = self.base_dir / "config" / "simulation_parameters.json"
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
            
        self.logger.info(f"   ✅ Parameters saved: {params_file}")
        
    def generate_masks(self):
        """Generate virtual boundary masks"""
        self.logger.info("🎯 Using existing masks...")
        
        # Check if masks already exist
        inlet_file = self.base_dir / "masks" / "inlet_boundary_mask.npy"
        outlet_file = self.base_dir / "masks" / "outlet_boundary_mask.npy"
        
        if inlet_file.exists() and outlet_file.exists():
            self.logger.info(f"   ✅ Found existing masks")
            
            # Create mask info from existing masks
            import numpy as np
            inlet_mask = np.load(inlet_file)
            outlet_mask = np.load(outlet_file)
            
            mask_info = {
                "inlet": {
                    "center": [0.0, 0.0, 0.0],
                    "radius": 313.6,
                    "active_points": int(inlet_mask.sum())
                },
                "outlet": {
                    "center": [1717.2, 0.0, 0.0], 
                    "radius": 602.7,
                    "active_points": int(outlet_mask.sum())
                },
                "domain_bounds": [-200.0, -800.0, -800.0, 1800.0, 800.0, 800.0],
                "grid_shape": [128, 64, 64]
            }
            
            info_file = self.base_dir / "masks" / "mask_info.json"
            with open(info_file, 'w') as f:
                json.dump(mask_info, f, indent=2)
                
            self.logger.info(f"   ✅ Mask info saved: {info_file}")
            return True
        else:
            self.logger.error("   ❌ No existing masks found")
            return False
            
    def validate_configuration(self):
        """Validate that all configuration files are present and valid"""
        self.logger.info("🔬 Validating configuration...")
        
        required_files = [
            "config/rocket_setup.json",
            "config/numerical_setup.json", 
            "config/simulation_parameters.json",
            "masks/inlet_boundary_mask.npy",
            "masks/outlet_boundary_mask.npy",
            "masks/mask_info.json"
        ]
        
        all_valid = True
        for file_path in required_files:
            full_path = self.base_dir / file_path
            if full_path.exists():
                self.logger.info(f"   ✅ {file_path}")
            else:
                self.logger.error(f"   ❌ Missing: {file_path}")
                all_valid = False
                
        return all_valid
        
    def create_quick_start_script(self):
        """Create a quick start script"""
        self.logger.info("📝 Creating quick start script...")
        
        script_content = '''#!/usr/bin/env python3
"""
Quick Start Script for Rocket Nozzle Simulation
Run this script to start the simulation with default parameters
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the rocket simulation with default settings"""
    print("STARTING ROCKET NOZZLE SIMULATION")
    print("=" * 50)
    
    # Run simulation with 100 iterations
    cmd = [sys.executable, "run_rocket_simulation.py", "--iterations", "100"]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\\nSimulation completed successfully!")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\\nSimulation failed with exit code {e.returncode}")
        return 1
        
    except KeyboardInterrupt:
        print("\\nSimulation interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''
        
        script_file = self.base_dir / "quick_start.py"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
            
        # Make executable on Unix systems
        try:
            script_file.chmod(0o755)
        except:
            pass  # Windows doesn't support chmod
            
        self.logger.info(f"   ✅ Quick start script: {script_file}")
        
    def run_complete_setup(self):
        """Run the complete simulation setup"""
        self.logger.info("🚀 STARTING COMPLETE ROCKET SIMULATION SETUP")
        self.logger.info("=" * 70)
        
        setup_steps = [
            ("Create directory structure", self.create_directory_structure),
            ("Copy helper files", self.copy_helper_files),
            ("Generate simulation parameters", self.generate_simulation_parameters),
            ("Setup boundary masks", self.generate_masks),
            ("Validate configuration", self.validate_configuration),
            ("Create quick start script", self.create_quick_start_script)
        ]
        
        for step_name, step_func in setup_steps:
            self.logger.info(f"🔧 {step_name}...")
            try:
                result = step_func()
                if result is False:
                    raise RuntimeError(f"Setup step failed: {step_name}")
                    
            except Exception as e:
                self.logger.error(f"❌ Setup failed at step '{step_name}': {e}")
                return False
                
        self.logger.info("=" * 70)
        self.logger.info("🎉 ROCKET SIMULATION SETUP COMPLETED SUCCESSFULLY!")
        self.logger.info("=" * 70)
        self.logger.info("📋 Next steps:")
        self.logger.info("   1. Run: python run_rocket_simulation.py --iterations 100")
        self.logger.info("   2. Or:  python quick_start.py")
        self.logger.info("   3. Monitor logs in: logs/")
        self.logger.info("   4. Check output in: output/")
        self.logger.info("=" * 70)
        
        return True

def main():
    """Main function"""
    print("🚀 JAX-FLUIDS ROCKET SIMULATION SETUP")
    print("=" * 50)
    print("Setting up production-ready rocket nozzle simulation...")
    print("=" * 50)
    
    setup = RocketSimulationSetup()
    success = setup.run_complete_setup()
    
    if success:
        print("\n✅ SUCCESS: Setup completed!")
        return 0
    else:
        print("\n❌ FAILED: Setup failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 