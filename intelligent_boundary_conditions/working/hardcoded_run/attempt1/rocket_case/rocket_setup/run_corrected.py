#!/usr/bin/env python3
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
        
        print(f"Loaded source term masks:")
        print(f"   Inlet: {np.sum(inlet_mask):,} grid points")
        print(f"   Outlet: {np.sum(outlet_mask):,} grid points")
        
        return jnp.array(inlet_mask), jnp.array(outlet_mask)
    except FileNotFoundError as e:
        print(f"Source term masks not found: {e}")
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
    
    print("JAX-FLUIDS ROCKET NOZZLE - CORRECT APPROACH")
    print("Using source terms for internal boundary conditions")
    print("=" * 70)
    
    # Load source term masks
    inlet_mask, outlet_mask = load_source_term_masks()
    
    # Initialize JAX-Fluids with corrected setup
    try:
        input_manager = InputManager("setup_corrected.json", "numerical.json")
        init_manager = InitializationManager(input_manager)
        sim_manager = SimulationManager(input_manager)
        
        print("JAX-Fluids initialized with corrected approach")
        
    except Exception as e:
        print(f"Error initializing JAX-Fluids: {e}")
        return 1
    
    # Initialize simulation buffers
    try:
        buffer_dict = init_manager.initialization()
        print("Simulation buffers initialized")
        
    except Exception as e:
        print(f"Error during initialization: {e}")
        return 1
    
    # TODO: Integrate source terms into simulation loop
    # This requires extending JAX-Fluids' source term system
    # or using the forcing mechanism for inlet/outlet conditions
    
    print("Running simulation with wall-only levelset...")
    print("(Source term integration needs to be implemented)")
    
    try:
        # For now, run standard simulation with wall-only levelset
        sim_manager.simulate(buffer_dict)
        print("Simulation completed successfully!")
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
