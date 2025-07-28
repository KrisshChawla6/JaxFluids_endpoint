#!/usr/bin/env python3
"""
Simple JAX-Fluids Test Script
Direct execution without optimization for debugging
"""

import os
import time

def main():
    """Simple JAX-Fluids simulation"""
    
    print("🚀 Simple JAX-Fluids Test")
    print("=" * 50)
    
    try:
        # Import JAX-Fluids
        from jaxfluids import InputManager, InitializationManager, SimulationManager
        print("✅ JAX-Fluids imported successfully")
        
        # Use the configuration files directly (already corrected)
        case_file = "jaxfluids_external_flow_1753655657.json"
        numerical_file = "numerical_setup.json"
        
        print(f"📄 Using case file: {case_file}")
        print(f"📄 Using numerical file: {numerical_file}")
        
        # Initialize JAX-Fluids
        print("🔧 Initializing InputManager...")
        input_manager = InputManager(case_file, numerical_file)
        print("✅ InputManager created")
        
        print("🔧 Initializing InitializationManager...")
        initialization_manager = InitializationManager(input_manager)
        print("✅ InitializationManager created")
        
        print("🔧 Initializing SimulationManager...")
        sim_manager = SimulationManager(input_manager)
        print("✅ SimulationManager created")
        
        # Initialize simulation
        print("🚀 Initializing simulation...")
        init_start = time.time()
        buffers = initialization_manager.initialization()
        init_time = time.time() - init_start
        print(f"✅ Initialization completed in {init_time:.2f} seconds")
        
        # Run simulation for a short time
        print("⏰ Running simulation...")
        sim_start = time.time()
        sim_manager.simulate(buffers)
        sim_time = time.time() - sim_start
        
        print("=" * 50)
        print(f"🎉 Simulation Completed Successfully!")
        print(f"   • Initialization: {init_time:.2f}s")
        print(f"   • Simulation: {sim_time:.2f}s")
        print(f"   • Total: {init_time + sim_time:.2f}s")
        print("=" * 50)
        
        return 0
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\n🏁 Exit code: {exit_code}") 