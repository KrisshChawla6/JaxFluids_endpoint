#!/usr/bin/env python3
"""
Intelligent Boundary Processor
Main endpoint for processing mesh files and generating JAX-Fluids internal flow simulations
Orchestrates the complete workflow from mesh to running simulation
"""

import logging
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from .core.virtual_face_detector import VirtualFaceDetector
from .core.sdf_generator import SDFGenerator
from .core.mask_generator import BoundaryMaskGenerator
from .core.jax_config_generator import JAXConfigGenerator

class IntelligentBoundaryProcessor:
    """
    Complete pipeline for converting mesh files to JAX-Fluids internal flow simulations
    
    Workflow:
    1. Load and analyze mesh file
    2. Detect virtual inlet/outlet faces
    3. Generate SDF for immersed boundaries
    4. Create 3D masks for JAX-Fluids grid
    5. Generate JAX-Fluids configuration files
    6. Set up complete simulation directory
    """
    
    def __init__(self, 
                 mesh_file: str,
                 output_dir: str = "intelligent_BC_simulation",
                 domain_bounds: list = None,
                 grid_shape: tuple = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the intelligent boundary processor
        
        Args:
            mesh_file: Path to input mesh file (.msh format)
            output_dir: Output directory for simulation setup
            domain_bounds: [x_min, y_min, z_min, x_max, y_max, z_max] (auto-detected if None)
            grid_shape: (nx, ny, nz) (default: (128, 64, 64))
            logger: Optional logger instance
        """
        self.mesh_file = Path(mesh_file)
        self.output_dir = Path(output_dir)
        self.logger = logger or self._setup_default_logger()
        
        if not self.mesh_file.exists():
            raise FileNotFoundError(f"Mesh file not found: {self.mesh_file}")
            
        # Default parameters
        self.domain_bounds = domain_bounds or [-200.0, -800.0, -800.0, 1800.0, 800.0, 800.0]
        self.grid_shape = grid_shape or (128, 64, 64)
        
        # Component instances
        self.face_detector = None
        self.sdf_generator = None
        self.mask_generator = None
        self.config_generator = None
        
        # Results storage
        self.virtual_faces = None
        self.sdf_matrix = None
        self.boundary_masks = None
        self.configurations = None
        
        self.logger.info(f"Initialized processor for mesh: {self.mesh_file}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
    def _setup_default_logger(self) -> logging.Logger:
        """Setup default logging configuration"""
        logger = logging.getLogger(self.__class__.__name__)
        
        if not logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            
        return logger
        
    def process_mesh(self, auto_detect_domain: bool = True) -> Dict[str, Any]:
        """
        Complete mesh processing pipeline
        
        Args:
            auto_detect_domain: Whether to auto-detect domain bounds from mesh
            
        Returns:
            Processing results summary
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING INTELLIGENT BOUNDARY PROCESSING")
        self.logger.info("=" * 60)
        
        results = {}
        
        try:
            # Step 1: Detect virtual faces
            self.logger.info("Step 1: Detecting virtual inlet/outlet faces...")
            self._detect_virtual_faces()
            results["virtual_faces"] = "Success"
            
            # Step 2: Generate SDF
            self.logger.info("Step 2: Generating signed distance function...")
            self._generate_sdf()
            results["sdf_generation"] = "Success"
            
            # Step 3: Create boundary masks
            self.logger.info("Step 3: Creating boundary masks...")
            self._create_boundary_masks()
            results["mask_generation"] = "Success"
            
            # Step 4: Generate configurations
            self.logger.info("Step 4: Generating JAX-Fluids configurations...")
            self._generate_configurations()
            results["config_generation"] = "Success"
            
            # Step 5: Setup simulation directory
            self.logger.info("Step 5: Setting up simulation directory...")
            self._setup_simulation_directory()
            results["directory_setup"] = "Success"
            
            self.logger.info("=" * 60)
            self.logger.info("PROCESSING COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 60)
            
            results["status"] = "Success"
            results["output_directory"] = str(self.output_dir)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            results["status"] = "Failed"
            results["error"] = str(e)
            raise
            
    def _detect_virtual_faces(self):
        """Detect virtual inlet/outlet faces from mesh"""
        self.face_detector = VirtualFaceDetector(str(self.mesh_file), self.logger)
        
        # Detect boundary points
        inlet_points, outlet_points = self.face_detector.detect_virtual_faces()
        
        # Create virtual faces
        inlet_face, outlet_face = self.face_detector.create_virtual_faces()
        
        self.virtual_faces = {
            "inlet": inlet_face,
            "outlet": outlet_face
        }
        
        self.logger.info(f"Virtual faces detected:")
        self.logger.info(f"  Inlet: center={inlet_face['center']}, radius={inlet_face['radius']:.2f}")
        self.logger.info(f"  Outlet: center={outlet_face['center']}, radius={outlet_face['radius']:.2f}")
        
    def _generate_sdf(self):
        """Generate signed distance function from mesh"""
        self.sdf_generator = SDFGenerator(str(self.mesh_file), self.logger)
        
        # Generate SDF on grid
        self.sdf_matrix = self.sdf_generator.generate_sdf(
            self.domain_bounds,
            self.grid_shape
        )
        
        # Get statistics
        stats = self.sdf_generator.get_sdf_stats()
        self.logger.info(f"SDF generated: {stats['shape']}, range=[{stats['min']:.2f}, {stats['max']:.2f}]")
        self.logger.info(f"  Inside points: {stats['inside_points']}, Outside: {stats['outside_points']}")
        
    def _create_boundary_masks(self):
        """Create 3D boundary masks for JAX-Fluids"""
        if self.virtual_faces is None:
            raise RuntimeError("Virtual faces not detected")
            
        self.mask_generator = BoundaryMaskGenerator(
            self.domain_bounds,
            self.grid_shape,
            self.logger
        )
        
        # Generate masks from virtual faces
        inlet_mask, outlet_mask = self.mask_generator.generate_masks_from_faces(
            self.virtual_faces["inlet"],
            self.virtual_faces["outlet"],
            thickness=50.0
        )
        
        self.boundary_masks = {
            "inlet": inlet_mask,
            "outlet": outlet_mask
        }
        
        # Get centroids for validation
        centroids = self.mask_generator.get_mask_centroids()
        self.logger.info(f"Boundary masks created:")
        self.logger.info(f"  Inlet: {inlet_mask.sum()} points, centroid={centroids['inlet']}")
        self.logger.info(f"  Outlet: {outlet_mask.sum()} points, centroid={centroids['outlet']}")
        
    def _generate_configurations(self):
        """Generate JAX-Fluids configuration files"""
        self.config_generator = JAXConfigGenerator(self.logger)
        
        # Generate case setup (will need SDF path)
        sdf_relative_path = "sdf/rocket_sdf.npy"  # Relative to simulation directory
        
        case_config = self.config_generator.generate_case_setup(
            case_name="intelligent_BC_internal_flow",
            domain_bounds=self.domain_bounds,
            grid_shape=self.grid_shape,
            sdf_path=sdf_relative_path,
            end_time=0.05,
            save_dt=0.005
        )
        
        # Generate numerical setup
        numerical_config = self.config_generator.generate_numerical_setup()
        
        # Validate configurations
        if not self.config_generator.validate_configuration():
            raise RuntimeError("Configuration validation failed")
            
        self.configurations = {
            "case": case_config,
            "numerical": numerical_config
        }
        
        self.logger.info("JAX-Fluids configurations generated and validated")
        
    def _setup_simulation_directory(self):
        """Setup complete simulation directory structure"""
        # Create directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        subdirs = ["config", "masks", "sdf", "output", "logs", "scripts"]
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(exist_ok=True)
            
        # Save configurations
        self.config_generator.save_configurations(self.output_dir / "config")
        
        # Save SDF
        sdf_file = self.output_dir / "sdf" / "rocket_sdf.npy"
        self.sdf_generator._save_sdf(str(sdf_file))
        
        # Save masks
        self.mask_generator.save_masks(self.output_dir / "masks")
        
        # Save virtual face data
        self.face_detector.save_face_data(self.output_dir / "virtual_faces")
        
        # Create runner script
        self.config_generator.create_jax_runner_template(self.output_dir)
        
        # Create README
        self._create_readme()
        
        # Create quick start script
        self._create_quick_start_script()
        
        self.logger.info(f"Simulation directory setup complete: {self.output_dir}")
        
    def _create_readme(self):
        """Create comprehensive README file"""
        readme_content = f"""# Intelligent Boundary Conditions - JAX-Fluids Simulation

## Overview
This directory contains a complete JAX-Fluids simulation setup for internal supersonic flow through a rocket nozzle, automatically generated from mesh file: `{self.mesh_file.name}`

## Directory Structure
```
{self.output_dir.name}/
├── config/                   # JAX-Fluids configuration files
│   ├── rocket_setup.json     # Case setup with forcing terms
│   ├── numerical_setup.json  # Numerical methods configuration  
│   └── simulation_parameters.json # Runtime parameters
├── masks/                    # Virtual boundary condition masks
│   ├── inlet_boundary_mask.npy   # 3D inlet mask
│   ├── outlet_boundary_mask.npy  # 3D outlet mask
│   └── mask_metadata.json        # Mask generation metadata
├── sdf/                      # Signed distance function
│   └── rocket_sdf.npy        # SDF for immersed boundaries
├── virtual_faces/            # Virtual face detection results
│   ├── inlet_points.npy      # Detected inlet boundary points
│   └── outlet_points.npy     # Detected outlet boundary points
├── output/                   # Simulation output (will be created)
├── logs/                     # Simulation logs
├── scripts/                  # Utility scripts
├── run_simulation.py         # Main simulation runner
├── quick_start.py           # Quick start script
└── README.md                # This file
```

## Technical Approach

### Virtual Boundary Conditions
The simulation implements virtual inlet/outlet boundary conditions using JAX-Fluids' native forcing system:

1. **Virtual Face Detection**: Automatically detects circular inlet/outlet openings from mesh geometry
2. **3D Mask Generation**: Maps virtual faces to Cartesian grid points for forcing application  
3. **Forcing Integration**: Uses JAX-Fluids' `forcings` system to apply mass flow and temperature conditions

### Key Features
- **Domain**: {self.domain_bounds} 
- **Grid**: {self.grid_shape[0]}×{self.grid_shape[1]}×{self.grid_shape[2]} Cartesian cells
- **Physics**: Compressible Navier-Stokes with immersed boundaries
- **Boundary Conditions**: Symmetry on domain faces, forcing on virtual faces
- **Output**: HDF5 format with density, velocity, pressure, temperature, Mach number

## Quick Start

### Option 1: Use Quick Start Script
```bash
python quick_start.py
```

### Option 2: Manual Execution  
```bash
python run_simulation.py
```

## Configuration

### Simulation Parameters
Edit `config/simulation_parameters.json` to modify:
- Maximum iterations
- Save intervals
- Convergence tolerance
- Output fields

### Flow Conditions
Edit the `forcings` section in `config/rocket_setup.json`:
```json
"forcings": {{
  "mass_flow": {{
    "direction": "x", 
    "target_value": 15.0
  }},
  "temperature": {{
    "target_value": 1500.0
  }}
}}
```

## Expected Results
- **Supersonic acceleration** through the rocket nozzle
- **Flow expansion** from inlet to outlet  
- **Pressure drop** along nozzle length
- **Mach number increase** toward outlet
- **Temperature variation** due to expansion

## Validation
- Inlet mask: {self.boundary_masks['inlet'].sum() if self.boundary_masks else 'N/A'} active grid points
- Outlet mask: {self.boundary_masks['outlet'].sum() if self.boundary_masks else 'N/A'} active grid points
- SDF range: [{self.sdf_matrix.min():.2f}, {self.sdf_matrix.max():.2f}] if generated

## Troubleshooting
1. **Missing masks**: Run `python masks/regenerate_masks.py`
2. **SDF issues**: Check mesh file integrity and domain bounds
3. **JAX-Fluids errors**: Verify configuration files syntax
4. **Convergence issues**: Adjust CFL number in numerical setup

Generated by Intelligent Boundary Conditions Processor v1.0
"""
        
        readme_file = self.output_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
            
        self.logger.info(f"README created: {readme_file}")
        
    def _create_quick_start_script(self):
        """Create quick start script for easy execution"""
        quick_start_content = '''#!/usr/bin/env python3
"""
Quick Start Script for Intelligent BC JAX-Fluids Simulation
"""

import sys
import logging
from pathlib import Path

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/quick_start.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_dependencies():
    """Check if required files exist"""
    logger = logging.getLogger(__name__)
    
    required_files = [
        "config/rocket_setup.json",
        "config/numerical_setup.json", 
        "masks/inlet_boundary_mask.npy",
        "masks/outlet_boundary_mask.npy",
        "sdf/rocket_sdf.npy"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
        
    logger.info("All required files found")
    return True

def run_quick_simulation():
    """Run quick simulation with basic monitoring"""
    logger = setup_logging()
    logger.info("Starting Quick JAX-Fluids Simulation...")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed")
        return False
        
    try:
        # Import JAX-Fluids (check if available)
        from jaxfluids import InputManager, InitializationManager, SimulationManager
        import numpy as np
        
        logger.info("JAX-Fluids imported successfully")
        
        # Load configuration
        case_setup = "config/rocket_setup.json"
        numerical_setup = "config/numerical_setup.json"
        
        logger.info("Loading configuration files...")
        input_manager = InputManager(case_setup, numerical_setup)
        initialization_manager = InitializationManager(input_manager)
        simulation_manager = SimulationManager(input_manager)
        
        # Initialize
        logger.info("Initializing simulation...")
        buffer_dict = initialization_manager.initialization()
        
        # Quick verification of masks
        inlet_mask = np.load("masks/inlet_boundary_mask.npy")
        outlet_mask = np.load("masks/outlet_boundary_mask.npy")
        logger.info(f"Boundary masks loaded: inlet={inlet_mask.sum()}, outlet={outlet_mask.sum()}")
        
        # Run simulation
        logger.info("Running simulation...")
        logger.info("This may take several minutes...")
        
        buffer_dict = simulation_manager.simulate(buffer_dict)
        
        logger.info("Simulation completed successfully!")
        logger.info("Check output/ directory for results")
        logger.info("Use visualization scripts to view results")
        
        return True
        
    except ImportError as e:
        logger.error(f"JAX-Fluids not available: {e}")
        logger.error("Please install JAX-Fluids to run simulation")
        return False
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return False

if __name__ == "__main__":
    success = run_quick_simulation()
    if success:
        print("\\n✅ Quick start completed successfully!")
        print("📊 Check output/ directory for simulation results")
    else:
        print("\\n❌ Quick start failed - check logs/quick_start.log")
        sys.exit(1)
'''
        
        quick_start_file = self.output_dir / "quick_start.py"
        with open(quick_start_file, 'w', encoding='utf-8') as f:
            f.write(quick_start_content)
            
        # Make executable (on Unix systems)
        try:
            quick_start_file.chmod(0o755)
        except:
            pass  # Windows doesn't support chmod
            
        self.logger.info(f"Quick start script created: {quick_start_file}")
        
    def get_processing_summary(self) -> str:
        """Get human-readable processing summary"""
        if self.virtual_faces is None:
            return "Processing not completed"
            
        inlet_face = self.virtual_faces["inlet"]
        outlet_face = self.virtual_faces["outlet"]
        
        summary = f"""
Intelligent Boundary Processing Summary:
======================================
Input Mesh: {self.mesh_file.name}
Output Directory: {self.output_dir}

Domain Configuration:
- Bounds: {self.domain_bounds}
- Grid: {self.grid_shape[0]}×{self.grid_shape[1]}×{self.grid_shape[2]}

Virtual Faces Detected:
- Inlet: center={inlet_face['center']}, radius={inlet_face['radius']:.2f}
- Outlet: center={outlet_face['center']}, radius={outlet_face['radius']:.2f}

Generated Components:
- SDF matrix: {self.sdf_matrix.shape if self.sdf_matrix is not None else 'Not generated'}
- Inlet mask: {self.boundary_masks['inlet'].sum() if self.boundary_masks else 'N/A'} points
- Outlet mask: {self.boundary_masks['outlet'].sum() if self.boundary_masks else 'N/A'} points
- JAX-Fluids configs: Generated
- Simulation directory: Setup complete

Ready for JAX-Fluids execution!
Run: python {self.output_dir}/quick_start.py
"""
        
        return summary
        
    def visualize_results(self, show_plots: bool = True):
        """Create visualizations of processing results"""
        if not show_plots:
            return
            
        try:
            # Visualize SDF
            if self.sdf_generator:
                sdf_viz_path = self.output_dir / "sdf_visualization.png"
                self.sdf_generator.visualize_sdf(str(sdf_viz_path))
                
            # Visualize masks
            if self.mask_generator:
                mask_viz_path = self.output_dir / "mask_visualization.png"
                self.mask_generator.visualize_masks(str(mask_viz_path))
                
            self.logger.info("Visualizations created in output directory")
            
        except Exception as e:
            self.logger.warning(f"Visualization failed: {e}")
            
    def cleanup_intermediate_files(self):
        """Clean up intermediate processing files (optional)"""
        self.logger.info("Cleaning up intermediate files...")
        
        # Could remove large intermediate files if needed
        # For now, keep everything for debugging
        pass 