#!/usr/bin/env python3
"""
JAX-Fluids Rocket Nozzle with Multi-Region Approach
Using positive/negative levelset regions with different initial conditions
"""

import json
import sys

from jaxfluids import InputManager, InitializationManager, SimulationManager

if __name__ == "__main__":
    case_file = "setup_multi_region.json"
    numerical_file = "numerical.json"

    print("=== JAX-FLUIDS MULTI-REGION ROCKET NOZZLE ===")
    print("Using positive/negative levelset regions")
    
    try:
        # Initialize JAX-Fluids components
        input_manager = InputManager(case_file, numerical_file)
        initialization_manager = InitializationManager(input_manager)
        sim_manager = SimulationManager(input_manager)

        print("JAX-Fluids multi-region initialized successfully")

        # Perform initialization
        buffers = initialization_manager.initialization()
        print("Multi-region simulation buffers initialized")

        # Run the simulation
        print("Starting multi-region simulation...")
        sim_manager.simulate(buffers)
        
        print("Multi-region simulation completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        # Let's get more detailed error info
        import traceback
        traceback.print_exc()
        sys.exit(1) 