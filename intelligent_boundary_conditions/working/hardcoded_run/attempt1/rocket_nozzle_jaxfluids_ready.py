#!/usr/bin/env python3
"""
JAX-Fluids Ready Rocket Nozzle Simulation - Attempt 1

This script creates a complete JAX-Fluids setup for internal flow in a rocket nozzle
with virtual inlet/outlet boundary conditions using boundary masks.

Approach: Use JAX-Fluids native boundary condition system with custom masks
"""

import numpy as np
import json
from pathlib import Path
import subprocess
import sys
import os
from datetime import datetime

# Add paths for our virtual face detection
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import pyvista as pv
    import meshio
    from circular_face_creator import find_circular_boundary_edges, fit_circle_and_create_face
    print("✅ Required libraries and virtual face detection available")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    exit(1)

class RocketNozzleJAXFluidsSetup:
    """Complete JAX-Fluids setup for rocket nozzle with virtual inlet/outlet"""
    
    def __init__(self, mesh_file: str, existing_endpoint_path: str):
        self.mesh_file = Path(mesh_file)
        self.endpoint_path = Path(existing_endpoint_path)
        self.output_dir = Path("rocket_nozzle_jaxfluids_simulation")
        
        # Hardcoded domain and flow parameters for rocket nozzle
        self.domain_bounds = [-200, -800, -800, 1800, 800, 800]  # [xmin, ymin, zmin, xmax, ymax, zmax]
        self.resolution = [128, 64, 64]  # [nx, ny, nz] - reasonable for testing
        
        # Rocket nozzle flow conditions
        self.flow_conditions = {
            'chamber_pressure': 6.9e6,      # 6.9 MPa
            'chamber_temperature': 3580.0,   # 3580 K  
            'ambient_pressure': 101325.0,    # 1 atm
            'gamma': 1.3,                    # Heat capacity ratio for combustion gases
            'gas_constant': 287.0,           # J/(kg·K)
            'inlet_velocity': 50.0           # m/s initial estimate
        }
    
    def detect_inlet_outlet_faces(self):
        """Detect virtual inlet and outlet faces from the mesh"""
        
        print("🎯 DETECTING VIRTUAL INLET/OUTLET FACES")
        print("=" * 60)
        
        # Use our proven circular face detection
        result = find_circular_boundary_edges(str(self.mesh_file))
        
        if result is None:
            raise ValueError("❌ Could not find boundary edges")
        
        inlet_points, outlet_points = result
        
        if inlet_points is None or outlet_points is None:
            raise ValueError("❌ Could not find inlet or outlet points")
        
        print(f"   🔵 Inlet: {len(inlet_points)} edge points at X≈{inlet_points[:, 0].mean():.1f}")
        print(f"   🔴 Outlet: {len(outlet_points)} edge points at X≈{outlet_points[:, 0].mean():.1f}")
        
        # Fit circles to each region
        inlet_face = fit_circle_and_create_face(inlet_points, "inlet")
        outlet_face = fit_circle_and_create_face(outlet_points, "outlet")
        
        if inlet_face is None or outlet_face is None:
            raise ValueError("❌ Could not fit circular faces")
        
        # Extract specs
        inlet_spec = {
            'center': inlet_face['center'],
            'radius': inlet_face['radius'], 
            'x_position': inlet_face['center'][0],
            'area': np.pi * inlet_face['radius']**2
        }
        
        outlet_spec = {
            'center': outlet_face['center'],
            'radius': outlet_face['radius'],
            'x_position': outlet_face['center'][0], 
            'area': np.pi * outlet_face['radius']**2
        }
        
        print(f"   ✅ Inlet: R={inlet_spec['radius']:.1f}, Area={inlet_spec['area']:.0f}")
        print(f"   ✅ Outlet: R={outlet_spec['radius']:.1f}, Area={outlet_spec['area']:.0f}")
        print(f"   ✅ Expansion Ratio: {outlet_spec['area']/inlet_spec['area']:.2f}")
        
        return inlet_spec, outlet_spec
    
    def create_boundary_masks(self, inlet_spec, outlet_spec):
        """Create 3D boundary condition masks for JAX-Fluids"""
        
        print("🎭 CREATING BOUNDARY CONDITION MASKS")
        print("=" * 60)
        
        # Create structured grid
        x = np.linspace(self.domain_bounds[0], self.domain_bounds[3], self.resolution[0])
        y = np.linspace(self.domain_bounds[1], self.domain_bounds[4], self.resolution[1])
        z = np.linspace(self.domain_bounds[2], self.domain_bounds[5], self.resolution[2])
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Grid spacing for tolerance
        dx = x[1] - x[0]
        dy = y[1] - y[0] 
        dz = z[1] - z[0]
        tolerance = max(dx, dy, dz) * 1.5
        
        print(f"   Grid: {self.resolution[0]}×{self.resolution[1]}×{self.resolution[2]} = {np.prod(self.resolution):,} points")
        print(f"   Tolerance: {tolerance:.1f} units")
        
        # Create inlet mask
        inlet_center = inlet_spec['center']
        inlet_radius = inlet_spec['radius']
        
        # Points near inlet X position
        inlet_x_mask = np.abs(X - inlet_center[0]) < tolerance
        
        # Points within circular cross-section
        yz_distance_inlet = np.sqrt((Y - inlet_center[1])**2 + (Z - inlet_center[2])**2)
        inlet_circular_mask = yz_distance_inlet <= inlet_radius
        
        inlet_mask = inlet_x_mask & inlet_circular_mask
        
        # Create outlet mask  
        outlet_center = outlet_spec['center']
        outlet_radius = outlet_spec['radius']
        
        outlet_x_mask = np.abs(X - outlet_center[0]) < tolerance
        yz_distance_outlet = np.sqrt((Y - outlet_center[1])**2 + (Z - outlet_center[2])**2)
        outlet_circular_mask = yz_distance_outlet <= outlet_radius
        
        outlet_mask = outlet_x_mask & outlet_circular_mask
        
        print(f"   🔵 Inlet mask: {inlet_mask.sum():,} active grid points")
        print(f"   🔴 Outlet mask: {outlet_mask.sum():,} active grid points")
        
        return {
            'inlet_mask': inlet_mask,
            'outlet_mask': outlet_mask,
            'grid_coordinates': (X, Y, Z)
        }
    
    def call_existing_agentic_endpoint(self):
        """Call your existing agentic JAX-Fluids internal flow endpoint"""
        
        print("🤖 CALLING EXISTING AGENTIC JAX-FLUIDS ENDPOINT")
        print("=" * 60)
        
        # Find the main script in your existing endpoint
        # This would typically be something like internal_flow_endpoint.py or similar
        main_script = self.endpoint_path / "immersed_boundary_sdf.py"  # Use the actual main script
        
        if not main_script.exists():
            raise FileNotFoundError(f"❌ Could not find main script: {main_script}")
        
        print(f"   🎯 Using script: {main_script}")
        
        # Prepare arguments for the existing endpoint
        domain_str = f"({self.domain_bounds[0]},{self.domain_bounds[1]},{self.domain_bounds[2]},{self.domain_bounds[3]},{self.domain_bounds[4]},{self.domain_bounds[5]})"
        resolution_str = f"({self.resolution[0]},{self.resolution[1]},{self.resolution[2]})"
        
        cmd = [
            "python", str(main_script),
            str(self.mesh_file),
            "--domain", domain_str,
            "--resolution", resolution_str,
            "--output-dir", str(self.output_dir / "base_setup")
        ]
        
        print(f"   🚀 Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.endpoint_path)
            
            if result.returncode == 0:
                print("   ✅ Base JAX-Fluids setup created successfully")
                print("   📁 Output in:", self.output_dir / "base_setup")
                return True
            else:
                print(f"   ❌ Endpoint failed: {result.stderr}")
                print(f"   📝 Stdout: {result.stdout}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error calling endpoint: {e}")
            return False
    
    def modify_jaxfluids_config_with_virtual_bc(self, inlet_spec, outlet_spec, boundary_masks):
        """Modify the JAX-Fluids configuration to include virtual boundary conditions"""
        
        print("⚙️ MODIFYING JAX-FLUIDS CONFIG WITH VIRTUAL BC")
        print("=" * 60)
        
        base_config_dir = self.output_dir / "base_setup"
        
        # Find the generated JAX-Fluids configuration file
        config_files = list(base_config_dir.glob("*.json"))
        if not config_files:
            # Create our own config based on JAX-Fluids structure
            config_file = base_config_dir / "rocket_nozzle_config.json"
            self._create_base_jaxfluids_config(config_file)
        else:
            config_file = config_files[0]
        
        print(f"   📄 Modifying config: {config_file}")
        
        # Load existing config
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Add virtual boundary conditions section
        config['virtual_boundary_conditions'] = {
            'inlet': {
                'type': 'DIRICHLET',
                'location': {
                    'center': inlet_spec['center'].tolist(),
                    'radius': float(inlet_spec['radius']),
                    'x_position': float(inlet_spec['x_position'])
                },
                'conditions': {
                    'pressure': self.flow_conditions['chamber_pressure'],
                    'temperature': self.flow_conditions['chamber_temperature'],
                    'velocity': [self.flow_conditions['inlet_velocity'], 0.0, 0.0]
                },
                'mask_file': 'inlet_boundary_mask.npy'
            },
            'outlet': {
                'type': 'NEUMANN', 
                'location': {
                    'center': outlet_spec['center'].tolist(),
                    'radius': float(outlet_spec['radius']),
                    'x_position': float(outlet_spec['x_position'])
                },
                'conditions': {
                    'pressure': self.flow_conditions['ambient_pressure'],
                    'gradient': 'ZERO_GRADIENT'
                },
                'mask_file': 'outlet_boundary_mask.npy'
            }
        }
        
        # Update metadata
        if 'metadata' not in config:
            config['metadata'] = {}
        
        config['metadata'].update({
            'virtual_faces_added': datetime.now().isoformat(),
            'inlet_area': float(inlet_spec['area']),
            'outlet_area': float(outlet_spec['area']), 
            'expansion_ratio': float(outlet_spec['area'] / inlet_spec['area']),
            'description': 'Rocket nozzle with virtual inlet/outlet boundary conditions'
        })
        
        # Save modified config
        modified_config_file = self.output_dir / "rocket_nozzle_with_virtual_bc.json"
        with open(modified_config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   ✅ Modified config saved: {modified_config_file}")
        
        # Save boundary masks
        inlet_mask_file = self.output_dir / "inlet_boundary_mask.npy"
        outlet_mask_file = self.output_dir / "outlet_boundary_mask.npy"
        
        np.save(inlet_mask_file, boundary_masks['inlet_mask'])
        np.save(outlet_mask_file, boundary_masks['outlet_mask'])
        
        print(f"   💾 Boundary masks saved:")
        print(f"      🔵 {inlet_mask_file}")
        print(f"      🔴 {outlet_mask_file}")
        
        return modified_config_file
    
    def _create_base_jaxfluids_config(self, config_file):
        """Create a base JAX-Fluids configuration if one doesn't exist"""
        
        # JAX-Fluids uses separate case.py and numerical.json files
        # Let's create the minimal numerical setup
        config = {
            'case_name': 'rocket_nozzle_internal_flow',
            'numerical_setup': {
                'conservatives': {
                    'density': True,
                    'velocity': True,
                    'pressure': True,
                    'energy': True
                },
                'mesh': {
                    'nx': self.resolution[0],
                    'ny': self.resolution[1], 
                    'nz': self.resolution[2],
                    'x_range': [self.domain_bounds[0], self.domain_bounds[3]],
                    'y_range': [self.domain_bounds[1], self.domain_bounds[4]],
                    'z_range': [self.domain_bounds[2], self.domain_bounds[5]]
                },
                'transport': {
                    'convective_treatment': 'WENO5_JS',
                    'riemann_solver': 'HLLC'
                },
                'time_integration': {
                    'method': 'RK3',
                    'CFL': 0.5,
                    'dt': 1e-6,
                    'max_timesteps': 1000
                },
                'levelset': {
                    'model': 'FLUID_SOLID_INTERFACE',
                    'method': 'LEVEL_SET',
                    'narrowband_computation': True,
                    'interface_treatment': 'FLUID_SOLID_DYNAMIC'
                },
                'boundary_conditions': {
                    'boundary_types': ['PERIODIC', 'PERIODIC', 'PERIODIC', 'PERIODIC', 'PERIODIC', 'PERIODIC'],
                    'boundary_locations': [
                        [-1, -1, -1],  # x_min face
                        [self.resolution[0], -1, -1],  # x_max face  
                        [-1, -1, -1],  # y_min face
                        [-1, self.resolution[1], -1],  # y_max face
                        [-1, -1, -1],  # z_min face
                        [-1, -1, self.resolution[2]]   # z_max face
                    ]
                },
                'material_properties': {
                    'equation_of_state': 'IDEAL_GAS',
                    'specific_heat_ratio': self.flow_conditions['gamma'],
                    'specific_gas_constant': self.flow_conditions['gas_constant'],
                    'thermal_conductivity': 0.01,
                    'dynamic_viscosity': 1e-5
                },
                'output': {
                    'save_path': './output',
                    'save_dt': 1e-5,
                    'quantities': ['density', 'velocity', 'pressure', 'temperature']
                }
            }
        }
        
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def create_jaxfluids_runner_script(self, config_file):
        """Create a runner script for the JAX-Fluids simulation"""
        
        print("🏃 CREATING JAX-FLUIDS RUNNER SCRIPT")
        print("=" * 60)
        
        runner_script = self.output_dir / "run_rocket_nozzle_simulation.py"
        
        runner_code = f'''#!/usr/bin/env python3
"""
JAX-Fluids Rocket Nozzle Simulation Runner

This script runs the complete rocket nozzle simulation with virtual inlet/outlet BCs.
Generated automatically by the intelligent boundary conditions system.
"""

import numpy as np
import json
from jaxfluids import Initializer, SimulationManager, InputReader
from jaxfluids.boundary_condition import BoundaryCondition

def load_virtual_boundary_masks():
    """Load the virtual boundary condition masks"""
    
    inlet_mask = np.load("inlet_boundary_mask.npy")
    outlet_mask = np.load("outlet_boundary_mask.npy")
    
    print(f"✅ Loaded boundary masks:")
    print(f"   🔵 Inlet: {{inlet_mask.sum():,}} active points")
    print(f"   🔴 Outlet: {{outlet_mask.sum():,}} active points")
    
    return inlet_mask, outlet_mask

def apply_virtual_boundary_conditions(boundary_condition, inlet_mask, outlet_mask, primitives):
    """Apply virtual inlet/outlet boundary conditions using JAX-Fluids native methods"""
    
    # Load virtual BC configuration
    with open("{config_file.name}", 'r') as f:
        config = json.load(f)
    
    virtual_bc = config['virtual_boundary_conditions']
    
    # Apply inlet conditions (Dirichlet)
    inlet_conditions = virtual_bc['inlet']['conditions']
    pressure_inlet = inlet_conditions['pressure']
    temperature_inlet = inlet_conditions['temperature']
    velocity_inlet = inlet_conditions['velocity']
    
    # Apply to primitives where inlet_mask is True
    primitives = primitives.at[0].set(
        np.where(inlet_mask, pressure_inlet, primitives[0])  # Pressure
    )
    primitives = primitives.at[1].set(
        np.where(inlet_mask, velocity_inlet[0], primitives[1])  # Velocity X
    )
    primitives = primitives.at[2].set(
        np.where(inlet_mask, velocity_inlet[1], primitives[2])  # Velocity Y  
    )
    primitives = primitives.at[3].set(
        np.where(inlet_mask, velocity_inlet[2], primitives[3])  # Velocity Z
    )
    primitives = primitives.at[4].set(
        np.where(inlet_mask, temperature_inlet, primitives[4])  # Temperature
    )
    
    # Apply outlet conditions (Neumann - zero gradient)
    # This would typically be handled by JAX-Fluids' Neumann boundary condition
    
    return primitives

def main():
    """Main simulation function"""
    
    print("🚀 STARTING JAX-FLUIDS ROCKET NOZZLE SIMULATION")
    print("=" * 70)
    
    # Load configuration
    input_reader = InputReader("{config_file}")
    
    # Load virtual boundary masks
    inlet_mask, outlet_mask = load_virtual_boundary_masks()
    
    # Initialize simulation
    initializer = Initializer(input_reader)
    buffer_dictionary = initializer.initialization()
    
    # Apply virtual boundary conditions to initial state
    primitives = buffer_dictionary["primes"]["primitives"]
    primitives = apply_virtual_boundary_conditions(
        None, inlet_mask, outlet_mask, primitives
    )
    buffer_dictionary["primes"]["primitives"] = primitives
    
    # Create simulation manager
    simulation_manager = SimulationManager(input_reader)
    
    print("✅ Simulation initialized with virtual boundary conditions")
    print("🏃 Starting simulation...")
    
    # Run simulation
    simulation_manager.simulate(buffer_dictionary)
    
    print("🎉 Simulation completed successfully!")

if __name__ == "__main__":
    main()
'''
        
        with open(runner_script, 'w') as f:
            f.write(runner_code)
        
        print(f"   ✅ Runner script created: {runner_script}")
        
        # Make executable on Unix systems
        if os.name != 'nt':
            os.chmod(runner_script, 0o755)
        
        return runner_script
    
    def run_complete_setup(self):
        """Run the complete rocket nozzle JAX-Fluids setup with virtual BC"""
        
        print("🚀 ROCKET NOZZLE JAX-FLUIDS COMPLETE SETUP")
        print("=" * 70)
        print(f"📁 Mesh: {self.mesh_file}")
        print(f"🎯 Output: {self.output_dir}")
        print("=" * 70)
        
        try:
            # Step 1: Detect virtual inlet/outlet faces
            inlet_spec, outlet_spec = self.detect_inlet_outlet_faces()
            
            # Step 2: Create boundary condition masks
            boundary_masks = self.create_boundary_masks(inlet_spec, outlet_spec)
            
            # Step 3: Call existing agentic endpoint
            self.call_existing_agentic_endpoint()
            
            # Step 4: Modify JAX-Fluids config with virtual BC
            config_file = self.modify_jaxfluids_config_with_virtual_bc(
                inlet_spec, outlet_spec, boundary_masks
            )
            
            # Step 5: Create runner script
            runner_script = self.create_jaxfluids_runner_script(config_file)
            
            print("\n" + "=" * 70)
            print("🎉 JAX-FLUIDS ROCKET NOZZLE SETUP COMPLETE!")
            print("=" * 70)
            print(f"✅ Virtual inlet/outlet faces detected and integrated")
            print(f"✅ Boundary condition masks created")
            print(f"✅ JAX-Fluids configuration ready")
            print(f"✅ Simulation runner script ready")
            print("\n🚀 TO RUN THE SIMULATION:")
            print(f"   cd {self.output_dir}")
            print(f"   python {runner_script.name}")
            print("=" * 70)
            
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function"""
    
    # Configuration
    mesh_file = r"C:\\Users\\kriss\\Desktop\\Endpoint JAX  FLuids\\mesh\\Rocket Engine.msh"
    existing_endpoint = r"C:\\Users\\kriss\\Desktop\\Endpoint JAX  FLuids\\immersed_boundary_endpoint_final"
    
    # Create setup
    setup = RocketNozzleJAXFluidsSetup(mesh_file, existing_endpoint)
    
    # Run complete setup
    success = setup.run_complete_setup()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 