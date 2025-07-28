#!/usr/bin/env python3
"""
JAX-Fluids Rocket Nozzle Internal Supersonic Flow
Proper JAX-Fluids Format
"""

import json
import sys

from jaxfluids import InputManager, InitializationManager, SimulationManager

if __name__ == "__main__":
    case_file = "setup.json"
    numerical_file = "numerical.json"

    try:
        # Initialize JAX-Fluids components
        input_manager = InputManager(case_file, numerical_file)
        initialization_manager = InitializationManager(input_manager)
        sim_manager = SimulationManager(input_manager)

        # Perform initialization
        buffers = initialization_manager.initialization()

        # Run the simulation
        sim_manager.simulate(buffers)

        print("JAX-Fluids simulation completed successfully!")

    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during simulation: {e}")
        sys.exit(1) 