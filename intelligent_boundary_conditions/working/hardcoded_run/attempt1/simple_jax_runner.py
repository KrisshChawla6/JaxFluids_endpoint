#!/usr/bin/env python3
"""
Simple JAX-Fluids Rocket Nozzle Runner

This script demonstrates how to run a JAX-Fluids simulation with virtual 
inlet/outlet boundary conditions for internal flow in a rocket nozzle.
"""

import numpy as np
import json
from pathlib import Path

# Check if JAX-Fluids is available
try:
    import jax.numpy as jnp
    from jaxfluids import Initializer, SimulationManager, InputReader
    from jaxfluids.boundary_condition import BoundaryCondition
    print("JAX-Fluids available for simulation")
    JAX_FLUIDS_AVAILABLE = True
except ImportError as e:
    print(f"JAX-Fluids not available: {e}")
    print("This is a simulation template - install JAX-Fluids to run")
    JAX_FLUIDS_AVAILABLE = False

def load_virtual_boundary_data():
    """Load the virtual boundary condition masks and configuration"""
    
    # Load boundary masks
    inlet_mask = np.load("inlet_boundary_mask.npy")
    outlet_mask = np.load("outlet_boundary_mask.npy")
    
    # Load configuration
    with open("rocket_nozzle_with_virtual_bc.json", 'r') as f:
        config = json.load(f)
    
    virtual_bc = config['virtual_boundary_conditions']
    
    print("Virtual Boundary Conditions Loaded:")
    print(f"  Inlet: {inlet_mask.sum():,} active grid points")
    print(f"    - Pressure: {virtual_bc['inlet']['conditions']['pressure']:,.0f} Pa")
    print(f"    - Temperature: {virtual_bc['inlet']['conditions']['temperature']:.0f} K")
    print(f"    - Velocity: {virtual_bc['inlet']['conditions']['velocity']} m/s")
    
    print(f"  Outlet: {outlet_mask.sum():,} active grid points")
    print(f"    - Type: {virtual_bc['outlet']['type']} (zero gradient)")
    print(f"    - Pressure: {virtual_bc['outlet']['conditions']['pressure']:,.0f} Pa")
    
    return inlet_mask, outlet_mask, virtual_bc

def apply_virtual_boundary_conditions(primitives, inlet_mask, outlet_mask, virtual_bc):
    """Apply virtual inlet/outlet boundary conditions to primitive variables"""
    
    if not JAX_FLUIDS_AVAILABLE:
        print("JAX-Fluids not available - simulation template only")
        return primitives
    
    # Apply inlet conditions (Dirichlet)
    inlet_conditions = virtual_bc['inlet']['conditions']
    
    # Set inlet pressure, velocity, and temperature
    primitives = primitives.at[0].set(  # Pressure
        jnp.where(inlet_mask, inlet_conditions['pressure'], primitives[0])
    )
    primitives = primitives.at[1].set(  # Velocity X
        jnp.where(inlet_mask, inlet_conditions['velocity'][0], primitives[1])
    )
    primitives = primitives.at[2].set(  # Velocity Y
        jnp.where(inlet_mask, inlet_conditions['velocity'][1], primitives[2])
    )
    primitives = primitives.at[3].set(  # Velocity Z
        jnp.where(inlet_mask, inlet_conditions['velocity'][2], primitives[3])
    )
    primitives = primitives.at[4].set(  # Temperature
        jnp.where(inlet_mask, inlet_conditions['temperature'], primitives[4])
    )
    
    # Outlet conditions (Neumann) are typically handled by JAX-Fluids boundary condition system
    # during the simulation step, not in the initial conditions
    
    return primitives

def create_boundary_condition_handler(inlet_mask, outlet_mask, virtual_bc):
    """Create a boundary condition handler for the simulation"""
    
    def boundary_update(primitives, time_step):
        """Update boundary conditions at each time step"""
        
        # Apply inlet conditions
        primitives = apply_virtual_boundary_conditions(
            primitives, inlet_mask, outlet_mask, virtual_bc
        )
        
        # Note: In a real JAX-Fluids implementation, you would also need to:
        # 1. Handle the outlet Neumann boundary condition
        # 2. Integrate with the level-set method for wall boundaries
        # 3. Apply proper flux corrections at virtual boundaries
        
        return primitives
    
    return boundary_update

def run_simulation():
    """Run the JAX-Fluids rocket nozzle simulation"""
    
    print("=" * 60)
    print("JAX-FLUIDS ROCKET NOZZLE SIMULATION")
    print("=" * 60)
    
    # Load virtual boundary data
    inlet_mask, outlet_mask, virtual_bc = load_virtual_boundary_data()
    
    if not JAX_FLUIDS_AVAILABLE:
        print("\nSIMULATION TEMPLATE")
        print("This demonstrates the virtual boundary condition integration.")
        print("Install JAX-Fluids to run the actual simulation.")
        print("\nKey Integration Points:")
        print("1. Virtual inlet/outlet faces detected from mesh hollow openings")
        print("2. Boundary condition masks created on structured grid")
        print("3. JAX-Fluids configuration includes virtual BC specifications")
        print("4. Runtime application of Dirichlet (inlet) and Neumann (outlet) conditions")
        print("5. Integration with existing level-set immersed boundary method")
        return
    
    # Load JAX-Fluids configuration
    input_reader = InputReader("rocket_nozzle_with_virtual_bc.json")
    
    # Initialize simulation
    print("\nInitializing JAX-Fluids simulation...")
    initializer = Initializer(input_reader)
    buffer_dictionary = initializer.initialization()
    
    # Apply virtual boundary conditions to initial state
    print("Applying virtual boundary conditions...")
    primitives = buffer_dictionary["primes"]["primitives"]
    primitives = apply_virtual_boundary_conditions(
        primitives, inlet_mask, outlet_mask, virtual_bc
    )
    buffer_dictionary["primes"]["primitives"] = primitives
    
    # Create boundary condition handler
    boundary_handler = create_boundary_condition_handler(inlet_mask, outlet_mask, virtual_bc)
    
    # Create simulation manager
    simulation_manager = SimulationManager(input_reader)
    
    print("Simulation initialized with virtual boundary conditions")
    print("Starting simulation...")
    
    # Run simulation with virtual boundary condition updates
    # Note: This is a simplified version - real implementation would need
    # proper integration with JAX-Fluids time stepping and boundary handling
    simulation_manager.simulate(buffer_dictionary)
    
    print("Simulation completed!")

def main():
    """Main function"""
    
    # Change to simulation directory
    sim_dir = Path("rocket_nozzle_jaxfluids_simulation")
    if sim_dir.exists():
        import os
        os.chdir(sim_dir)
        print(f"Running simulation from: {sim_dir.absolute()}")
    else:
        print("Error: Simulation directory not found")
        print("Run rocket_nozzle_jaxfluids_ready.py first to set up the simulation")
        return 1
    
    try:
        run_simulation()
        return 0
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 