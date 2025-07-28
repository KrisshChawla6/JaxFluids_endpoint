#!/usr/bin/env python3
"""
Simple JAX-Fluids Rocket Nozzle Internal Flow
Pure JAX-Fluids format - no custom modifications
"""

import json
import sys

from jaxfluids import InputManager, InitializationManager, SimulationManager

if __name__ == "__main__":
    case_file = "setup.json"
    numerical_file = "numerical.json"

    print("=== SIMPLE JAX-FLUIDS ROCKET NOZZLE SIMULATION ===")
    print("Pure JAX-Fluids - No custom modifications")
    
    try:
        # Initialize JAX-Fluids components
        input_manager = InputManager(case_file, numerical_file)
        initialization_manager = InitializationManager(input_manager)
        sim_manager = SimulationManager(input_manager)

        print("JAX-Fluids initialized successfully")

        # Perform initialization
        buffers = initialization_manager.initialization()
        print("Simulation buffers initialized")

        # Run the simulation
        print("Starting simulation...")
        sim_manager.simulate(buffers)
        
        print("Simulation completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        # Let's get more detailed error info
        import traceback
        traceback.print_exc()
        sys.exit(1) 