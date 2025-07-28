#!/usr/bin/env python3
"""
JAX-Fluids Rocket Nozzle Simulation with Virtual Boundary Conditions

This script runs a rocket nozzle internal flow simulation using JAX-Fluids
with virtual inlet/outlet boundary conditions detected from mesh hollow openings.

Usage:
    python run.py
"""

import numpy as np
import jax.numpy as jnp
import json
from pathlib import Path

# JAX-Fluids imports
from jaxfluids import SimulationManager, InitializationManager, InputManager

def load_boundary_masks():
    """Load virtual boundary condition masks"""
    try:
        inlet_mask = jnp.array(np.load("rocket_nozzle_jaxfluids_simulation/inlet_boundary_mask.npy"))
        outlet_mask = jnp.array(np.load("rocket_nozzle_jaxfluids_simulation/outlet_boundary_mask.npy"))
        
        print(f"✅ Loaded virtual boundary masks:")
        print(f"   🔵 Inlet: {inlet_mask.sum():,} active grid points")
        print(f"   🔴 Outlet: {outlet_mask.sum():,} active grid points")
        
        return inlet_mask, outlet_mask, True
        
    except FileNotFoundError:
        print("⚠️  Virtual boundary mask files not found")
        print("   Run rocket_nozzle_jaxfluids_ready.py first to generate masks")
        return None, None, False

def apply_virtual_boundary_conditions(primitives, inlet_mask, outlet_mask, virtual_bc_config):
    """Apply virtual inlet/outlet boundary conditions"""
    
    # Get virtual BC configuration
    virtual_bc = virtual_bc_config.get('virtual_boundary_conditions', {})
    
    if not virtual_bc:
        return primitives
        
    # Apply inlet conditions (Dirichlet)
    inlet_config = virtual_bc.get('inlet', {})
    if inlet_config and inlet_mask is not None:
        inlet_conditions = inlet_config.get('conditions', {})
        
        # Calculate inlet density from ideal gas law
        R = 287.0  # Specific gas constant
        inlet_pressure = inlet_conditions.get('pressure', 101325.0)
        inlet_temperature = inlet_conditions.get('temperature', 300.0)
        inlet_density = inlet_pressure / (R * inlet_temperature)
        
        inlet_velocity = inlet_conditions.get('velocity', [0.0, 0.0, 0.0])
        
        # Apply inlet conditions where mask is True
        primitives = primitives.at[0].set(  # Density
            jnp.where(inlet_mask, inlet_density, primitives[0])
        )
        primitives = primitives.at[1].set(  # Velocity U
            jnp.where(inlet_mask, inlet_velocity[0], primitives[1])
        )
        primitives = primitives.at[2].set(  # Velocity V
            jnp.where(inlet_mask, inlet_velocity[1], primitives[2])
        )
        primitives = primitives.at[3].set(  # Velocity W
            jnp.where(inlet_mask, inlet_velocity[2], primitives[3])
        )
        primitives = primitives.at[4].set(  # Pressure
            jnp.where(inlet_mask, inlet_pressure, primitives[4])
        )
    
    # Outlet conditions (Neumann) - typically handled by JAX-Fluids boundary system
    # For explicit outlet pressure, uncomment:
    # outlet_config = virtual_bc.get('outlet', {})
    # if outlet_config and outlet_mask is not None:
    #     outlet_pressure = outlet_config.get('conditions', {}).get('pressure', 101325.0)
    #     primitives = primitives.at[4].set(
    #         jnp.where(outlet_mask, outlet_pressure, primitives[4])
    #     )
    
    return primitives

def main():
    """Main simulation function"""
    
    print("=" * 70)
    print("🚀 JAX-FLUIDS ROCKET NOZZLE SIMULATION")
    print("   With Virtual Inlet/Outlet Boundary Conditions")
    print("=" * 70)
    
    # Load configuration files
    try:
        with open("setup.json", 'r') as f:
            setup_config = json.load(f)
        print("✅ Loaded setup.json")
    except FileNotFoundError:
        print("❌ setup.json not found")
        return 1
    
    try:
        with open("numerical.json", 'r') as f:
            numerical_config = json.load(f)
        print("✅ Loaded numerical.json")
    except FileNotFoundError:
        print("❌ numerical.json not found")
        return 1
    
    # Load boundary masks
    inlet_mask, outlet_mask, virtual_bc_available = load_boundary_masks()
    
    # Create input manager
    try:
        input_manager = InputManager(
            case_setup="setup.json",
            numerical_setup="numerical.json"
        )
        print("✅ Input manager created")
    except Exception as e:
        print(f"❌ Error creating input manager: {e}")
        return 1
    
    # Create initialization manager
    try:
        init_manager = InitializationManager(input_manager)
        print("✅ Initialization manager created")
    except Exception as e:
        print(f"❌ Error creating initialization manager: {e}")
        return 1
    
    # Initialize simulation
    try:
        buffer_dict = init_manager.initialization()
        print("✅ Simulation initialized")
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        return 1
    
    # Apply virtual boundary conditions to initial state
    if virtual_bc_available:
        try:
            primitives = buffer_dict["primitives"]
            primitives = apply_virtual_boundary_conditions(
                primitives, inlet_mask, outlet_mask, setup_config
            )
            buffer_dict["primitives"] = primitives
            print("✅ Virtual boundary conditions applied to initial state")
        except Exception as e:
            print(f"⚠️  Error applying virtual BCs: {e}")
    
    # Create simulation manager
    try:
        sim_manager = SimulationManager(input_manager)
        print("✅ Simulation manager created")
    except Exception as e:
        print(f"❌ Error creating simulation manager: {e}")
        return 1
    
    # Run simulation
    print("\n🏃 Starting simulation...")
    try:
        sim_manager.simulate(buffer_dict)
        print("🎉 Simulation completed successfully!")
        
        # Print results summary
        print("\n" + "=" * 70)
        print("📊 SIMULATION RESULTS")
        print("=" * 70)
        print(f"✅ Results saved to: {setup_config['case']['save_path']}")
        
        if virtual_bc_available:
            virtual_bc = setup_config['virtual_boundary_conditions']
            print(f"🔵 Inlet BC: {virtual_bc['inlet']['type']} at X={virtual_bc['inlet']['location']['x_position']}")
            print(f"   Pressure: {virtual_bc['inlet']['conditions']['pressure']:,.0f} Pa")
            print(f"   Temperature: {virtual_bc['inlet']['conditions']['temperature']:.0f} K")
            print(f"🔴 Outlet BC: {virtual_bc['outlet']['type']} at X={virtual_bc['outlet']['location']['x_position']}")
            print(f"   Pressure: {virtual_bc['outlet']['conditions']['pressure']:,.0f} Pa")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 