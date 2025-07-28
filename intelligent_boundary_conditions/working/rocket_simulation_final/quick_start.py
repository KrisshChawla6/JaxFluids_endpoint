#!/usr/bin/env python3
"""
Quick Start Script for Rocket Nozzle Simulation
Run this script to start the simulation with default parameters
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the rocket simulation with default settings"""
    print("STARTING ROCKET NOZZLE SIMULATION")
    print("=" * 50)
    
    # Run simulation with 100 iterations
    cmd = [sys.executable, "run_rocket_simulation.py", "--iterations", "100"]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\nSimulation completed successfully!")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\nSimulation failed with exit code {e.returncode}")
        return 1
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main())
