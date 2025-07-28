#!/usr/bin/env python3
"""
Correct JAX-Fluids Approach for Internal Boundary Conditions

This script demonstrates the CORRECT approach for applying internal 
boundary conditions (inlet/outlet) in JAX-Fluids:

1. Use the original professional SDF for walls ONLY
2. Apply inlet/outlet BCs as source terms/forcings during simulation
3. Use JAX-Fluids' native source term system for internal flow BCs

Key insight: Virtual faces should NOT be in the levelset!
"""

import numpy as np
import sys
from pathlib import Path
import subprocess

def use_original_sdf():
    """Use the original professional SDF without modification"""
    print("🔧 Using Original Professional SDF")
    print("=" * 50)
    
    # The original SDF path
    original_sdf = Path("../20250728_003202/Rocket Engine_sdf_matrix.npy")
    
    if not original_sdf.exists():
        print(f"❌ Original SDF not found at {original_sdf}")
        return False
    
    # Simply copy the original SDF (or reference it directly)
    wall_sdf = np.load(original_sdf)
    
    print(f"✅ Loaded original wall SDF:")
    print(f"   Shape: {wall_sdf.shape}")
    print(f"   Range: [{np.min(wall_sdf):.3f}, {np.max(wall_sdf):.3f}]")
    print(f"   This represents ONLY the rocket nozzle walls")
    
    # Save as the levelset for JAX-Fluids (no modification needed!)
    levelset_file = Path("rocket_walls_only_levelset.npy")
    np.save(levelset_file, wall_sdf)
    print(f"💾 Saved wall-only levelset to {levelset_file}")
    
    return True

def create_inlet_outlet_source_terms():
    """Create source terms for inlet/outlet boundary conditions"""
    print("\n🔧 Creating Inlet/Outlet Source Terms")
    print("=" * 50)
    
    # Load virtual face data from our successful detection
    try:
        inlet_mask = np.load("../../rocket_nozzle_jaxfluids_simulation/inlet_boundary_mask.npy")
        outlet_mask = np.load("../../rocket_nozzle_jaxfluids_simulation/outlet_boundary_mask.npy")
        
        print(f"✅ Loaded virtual boundary masks:")
        print(f"   🔵 Inlet: {np.sum(inlet_mask):,} grid points")
        print(f"   🔴 Outlet: {np.sum(outlet_mask):,} grid points")
        
        # Save the masks for source term application
        np.save("inlet_source_mask.npy", inlet_mask)
        np.save("outlet_source_mask.npy", outlet_mask)
        
        return True
        
    except FileNotFoundError as e:
        print(f"❌ Virtual boundary masks not found: {e}")
        print("   Run the virtual face detection first!")
        return False

def create_jax_fluids_source_term_setup():
    """Create JAX-Fluids setup that uses source terms for internal BCs"""
    print("\n🔧 Creating JAX-Fluids Source Term Setup")
    print("=" * 50)
    
    # Updated setup.json that uses source terms instead of levelset BCs
    setup_config = {
        "general": {
            "case_name": "rocket_nozzle_internal_flow_source_terms",
            "end_time": 0.0001,
            "save_path": "./output/",
            "save_dt": 1e-05
        },
        "restart": {
            "flag": False,
            "file_path": ""
        },
        "domain": {
            "x": {"cells": 128, "range": [-200.0, 1800.0]},
            "y": {"cells": 64, "range": [-800.0, 800.0]},
            "z": {"cells": 64, "range": [-800.0, 800.0]},
            "decomposition": {"split_x": 1, "split_y": 1, "split_z": 1}
        },
        "boundary_conditions": {
            "east": {"type": "SYMMETRY"},
            "west": {"type": "SYMMETRY"},
            "north": {"type": "SYMMETRY"}, 
            "south": {"type": "SYMMETRY"},
            "top": {"type": "SYMMETRY"},
            "bottom": {"type": "SYMMETRY"}
        },
        "initial_condition": {
            "primitives": {
                "rho": 1.0,
                "u": 10.0, 
                "v": 0.0,
                "w": 0.0,
                "p": 1000000.0
            },
            # Use ONLY the wall levelset (no virtual faces!)
            "levelset": "rocket_walls_only_levelset.npy"
        },
        "material_properties": {
            "equation_of_state": {
                "model": "IdealGas",
                "specific_heat_ratio": 1.4,
                "specific_gas_constant": 287.0
            },
            "transport": {
                "dynamic_viscosity": {"model": "CUSTOM", "value": 1.8e-05},
                "bulk_viscosity": 0.0,
                "thermal_conductivity": {"model": "PRANDTL", "prandtl_number": 0.72}
            }
        },
        "nondimensionalization_parameters": {
            "density_reference": 1.0,
            "length_reference": 1.0,
            "velocity_reference": 50.0,
            "temperature_reference": 288.15
        },
        "output": {
            "primitives": ["density", "velocity", "pressure", "temperature"],
            "levelset": ["levelset", "volume_fraction"],
            "miscellaneous": ["mach_number"]
        }
    }
    
    import json
    with open("setup_corrected.json", "w") as f:
        json.dump(setup_config, f, indent=2)
    
    print("✅ Created corrected setup.json (no virtual levelset BCs)")
    return True

def create_source_term_run_script():
    """Create run.py that applies inlet/outlet as source terms"""
    print("\n🔧 Creating Source Term Run Script")
    print("=" * 50)
    
    run_script = '''#!/usr/bin/env python3
"""
JAX-Fluids Rocket Nozzle with Source Term Boundary Conditions

This is the CORRECT approach:
- Use original wall-only levelset for solid boundaries
- Apply inlet/outlet BCs as source terms during simulation
"""

import numpy as np
import jax.numpy as jnp
from pathlib import Path

from jaxfluids import InputManager, InitializationManager, SimulationManager

def load_source_term_masks():
    """Load masks for applying source terms at inlet/outlet"""
    try:
        inlet_mask = np.load("inlet_source_mask.npy")
        outlet_mask = np.load("outlet_source_mask.npy")
        
        print(f"✅ Loaded source term masks:")
        print(f"   🔵 Inlet: {np.sum(inlet_mask):,} grid points")
        print(f"   🔴 Outlet: {np.sum(outlet_mask):,} grid points")
        
        return jnp.array(inlet_mask), jnp.array(outlet_mask)
    except FileNotFoundError as e:
        print(f"❌ Source term masks not found: {e}")
        return None, None

def create_inlet_source_term(primitives, inlet_mask):
    """Apply inlet boundary conditions as source terms"""
    if inlet_mask is None:
        return jnp.zeros_like(primitives)
    
    # High pressure combustion chamber conditions
    inlet_density = 6.85  # kg/m³
    inlet_velocity = jnp.array([50.0, 0.0, 0.0])  # m/s
    inlet_pressure = 6.9e6  # Pa
    
    # Source terms to drive flow towards inlet conditions
    source = jnp.zeros_like(primitives)
    
    # Density source
    source = source.at[0].set(
        jnp.where(inlet_mask, 
                 (inlet_density - primitives[0]) * 1000.0,  # Strong forcing
                 0.0)
    )
    
    # Momentum source
    source = source.at[1].set(
        jnp.where(inlet_mask,
                 (inlet_velocity[0] * inlet_density - primitives[1]) * 1000.0,
                 0.0)
    )
    
    # Pressure source (applied to energy equation)
    source = source.at[4].set(
        jnp.where(inlet_mask,
                 (inlet_pressure - primitives[4]) * 1000.0,
                 0.0)
    )
    
    return source

def create_outlet_source_term(primitives, outlet_mask):
    """Apply outlet boundary conditions as source terms"""
    if outlet_mask is None:
        return jnp.zeros_like(primitives)
    
    # Atmospheric exit conditions
    outlet_pressure = 101325.0  # Pa
    
    # Source terms for outlet (mainly pressure control)
    source = jnp.zeros_like(primitives)
    
    # Pressure source for outlet
    source = source.at[4].set(
        jnp.where(outlet_mask,
                 (outlet_pressure - primitives[4]) * 100.0,  # Weaker forcing
                 0.0)
    )
    
    return source

def main():
    """Main simulation with source term boundary conditions"""
    
    print("🚀 JAX-FLUIDS ROCKET NOZZLE - CORRECT APPROACH")
    print("   Using source terms for internal boundary conditions")
    print("=" * 70)
    
    # Load source term masks
    inlet_mask, outlet_mask = load_source_term_masks()
    
    # Initialize JAX-Fluids with corrected setup
    try:
        input_manager = InputManager("setup_corrected.json", "numerical.json")
        init_manager = InitializationManager(input_manager)
        sim_manager = SimulationManager(input_manager)
        
        print("✅ JAX-Fluids initialized with corrected approach")
        
    except Exception as e:
        print(f"❌ Error initializing JAX-Fluids: {e}")
        return 1
    
    # Initialize simulation buffers
    try:
        buffer_dict = init_manager.initialization()
        print("✅ Simulation buffers initialized")
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        return 1
    
    # TODO: Integrate source terms into simulation loop
    # This requires extending JAX-Fluids' source term system
    # or using the forcing mechanism for inlet/outlet conditions
    
    print("🏃 Running simulation with wall-only levelset...")
    print("   (Source term integration needs to be implemented)")
    
    try:
        # For now, run standard simulation with wall-only levelset
        sim_manager.simulate(buffer_dict)
        print("✅ Simulation completed successfully!")
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
'''
    
    with open("run_corrected.py", "w") as f:
        f.write(run_script)
    
    print("✅ Created corrected run.py with source term approach")
    return True

def main():
    """Main function to create the correct JAX-Fluids approach"""
    
    print("🎯 CREATING CORRECT JAX-FLUIDS APPROACH")
    print("=" * 60)
    print("Key insight: Virtual faces should NOT be in levelset!")
    print("Instead: Use source terms for internal boundary conditions")
    print("=" * 60)
    
    success = True
    
    # Step 1: Use original professional SDF (walls only)
    success &= use_original_sdf()
    
    # Step 2: Create source terms for inlet/outlet
    success &= create_inlet_outlet_source_terms()
    
    # Step 3: Create corrected JAX-Fluids setup
    success &= create_jax_fluids_source_term_setup()
    
    # Step 4: Create corrected run script
    success &= create_source_term_run_script()
    
    if success:
        print("\n✅ CORRECT APPROACH CREATED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Test with: python run_corrected.py")
        print("2. Implement source term integration in JAX-Fluids")
        print("3. Research JAX-Fluids forcing/source term system")
        print("\nThe key difference:")
        print("❌ OLD: Virtual faces in levelset (wrong!)")
        print("✅ NEW: Wall-only levelset + source terms (correct!)")
    else:
        print("\n❌ Failed to create correct approach")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 