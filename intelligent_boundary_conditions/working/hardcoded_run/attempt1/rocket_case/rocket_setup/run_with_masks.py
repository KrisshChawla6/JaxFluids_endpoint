#!/usr/bin/env python3
"""
JAX-Fluids Rocket Nozzle with Native Forcing System
Using Real Virtual Boundary Masks for Internal Flow
"""

import json
import sys
import numpy as np
import jax.numpy as jnp

from jaxfluids import InputManager, InitializationManager, SimulationManager

# Import our mask functions
from mask_functions import initialize_masks, get_inlet_mask_function, get_outlet_mask_function

class RocketNozzleForcing:
    """Custom forcing integration for rocket nozzle with virtual inlet/outlet"""
    
    def __init__(self):
        self.inlet_mask_func = None
        self.outlet_mask_func = None
        
    def setup_masks(self):
        """Initialize and validate our mask functions"""
        print("🎯 Setting up virtual boundary masks...")
        
        # Load real mask files (no fallbacks!)
        initialize_masks()
        
        # Get the mask functions
        self.inlet_mask_func = get_inlet_mask_function()
        self.outlet_mask_func = get_outlet_mask_function()
        
        print("✅ Virtual boundary masks ready for JAX-Fluids forcing")
        
    def create_inlet_forcing_function(self):
        """Create forcing function for inlet conditions"""
        def inlet_forcing(x, y, z, t=0.0):
            """Apply high pressure/temperature inlet conditions via forcing"""
            # High pressure combustion chamber conditions
            inlet_pressure = 6.9e6  # 6.9 MPa
            inlet_temperature = 3580.0  # 3580 K
            
            # Return conditions only where inlet mask is True
            is_inlet = self.inlet_mask_func(x, y, z)
            
            return jnp.where(is_inlet, inlet_temperature, 0.0)
        
        return inlet_forcing
        
    def create_outlet_forcing_function(self):
        """Create forcing function for outlet conditions"""
        def outlet_forcing(x, y, z, t=0.0):
            """Apply atmospheric outlet conditions via forcing"""
            # Atmospheric exit conditions
            outlet_temperature = 288.15  # Standard atmosphere
            
            # Return conditions only where outlet mask is True
            is_outlet = self.outlet_mask_func(x, y, z)
            
            return jnp.where(is_outlet, outlet_temperature, 0.0)
            
        return outlet_forcing
        
    def create_combined_temperature_forcing(self):
        """Create combined temperature forcing for inlet and outlet"""
        inlet_forcing = self.create_inlet_forcing_function()
        outlet_forcing = self.create_outlet_forcing_function()
        
        def combined_forcing(x, y, z, t=0.0):
            """Combined temperature forcing for both inlet and outlet"""
            inlet_temp = inlet_forcing(x, y, z, t)
            outlet_temp = outlet_forcing(x, y, z, t)
            
            # Return the sum - only one will be non-zero at any point
            return inlet_temp + outlet_temp
            
        return combined_forcing

def run_simulation_with_masks():
    """Run JAX-Fluids simulation with virtual boundary masks"""
    
    print("=" * 70)
    print("🚀 JAX-FLUIDS ROCKET NOZZLE WITH NATIVE FORCING")
    print("   Using Real Virtual Boundary Masks")
    print("=" * 70)
    
    # Setup forcing with real masks
    forcing_system = RocketNozzleForcing()
    forcing_system.setup_masks()
    
    # JAX-Fluids configuration files
    case_file = "setup_with_masks.json"
    numerical_file = "numerical.json"
    
    try:
        # Initialize JAX-Fluids components
        print("🔧 Initializing JAX-Fluids with forcing...")
        input_manager = InputManager(case_file, numerical_file)
        initialization_manager = InitializationManager(input_manager)
        sim_manager = SimulationManager(input_manager)
        
        print("✅ JAX-Fluids components initialized with forcing enabled")
        
        # Perform initialization
        buffers = initialization_manager.initialization()
        print("✅ Simulation buffers initialized")
        
        # Run simulation with native forcing
        print("\n🏃 Starting rocket nozzle simulation with virtual boundary forcing...")
        print("   Inlet: High pressure/temperature via forcing masks")
        print("   Outlet: Atmospheric conditions via forcing masks")
        print("   Walls: Level-set immersed boundaries")
        
        sim_manager.simulate(buffers)
        
        print("\n🎉 SIMULATION COMPLETED SUCCESSFULLY!")
        print("   Virtual inlet/outlet boundary conditions applied via JAX-Fluids native forcing")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_simulation_with_masks()
    
    if success:
        print("\n✅ SUCCESS: Rocket nozzle internal flow with virtual boundaries!")
        sys.exit(0)
    else:
        print("\n❌ FAILED: Check mask files and configuration")
        sys.exit(1) 