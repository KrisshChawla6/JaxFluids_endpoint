#!/usr/bin/env python3

"""
Test script to understand the JAX-Fluids API and execution methods
"""

import os
import sys
import traceback

def test_jaxfluids_api():
    """Test the JAX-Fluids API to understand execution methods"""
    
    print("🧪 Testing JAX-Fluids API")
    print("=" * 50)
    
    try:
        # Test basic imports
        print("📦 Testing imports...")
        from jaxfluids import InputManager, InitializationManager, SimulationManager
        print("✅ Core imports successful")
        
        # Test if we can find the working configuration
        config_dir = "propeller_fresh_setup/jaxfluids_external_flow_1753651803"
        case_file = os.path.join(config_dir, "jaxfluids_external_flow_1753651803.json")
        numerical_file = os.path.join(config_dir, "numerical_setup.json")
        
        if not os.path.exists(case_file):
            print(f"❌ Case file not found: {case_file}")
            return
        if not os.path.exists(numerical_file):
            print(f"❌ Numerical file not found: {numerical_file}")
            return
            
        print(f"✅ Configuration files found")
        
        # Test InputManager
        print("\n📋 Testing InputManager...")
        input_manager = InputManager(case_file, numerical_file)
        print("✅ InputManager created successfully")
        
        # Test InitializationManager  
        print("\n🚀 Testing InitializationManager...")
        initialization_manager = InitializationManager(input_manager)
        print("✅ InitializationManager created successfully")
        
        # Test creating initial buffers
        print("\n⚙️ Testing buffer initialization...")
        buffers = initialization_manager.initialization()
        print("✅ Initial buffers created successfully")
        
        # Test SimulationManager
        print("\n🎮 Testing SimulationManager...")
        sim_manager = SimulationManager(input_manager)
        print("✅ SimulationManager created successfully")
        
        # Check available methods
        print("\n🔍 Available SimulationManager methods:")
        methods = [attr for attr in dir(sim_manager) if not attr.startswith('_') and callable(getattr(sim_manager, attr))]
        for method in sorted(methods):
            print(f"   - {method}")
        
        # Try to understand the step method
        print("\n🔬 Analyzing simulation execution methods...")
        if hasattr(sim_manager, 'simulate'):
            print("   ✅ Has simulate() method")
        if hasattr(sim_manager, 'step'):
            print("   ✅ Has step() method")
        if hasattr(sim_manager, 'advance'):
            print("   ✅ Has advance() method")
        if hasattr(sim_manager, 'time_integrator'):
            print("   ✅ Has time_integrator attribute")
            time_integrator = sim_manager.time_integrator
            integrator_methods = [attr for attr in dir(time_integrator) if not attr.startswith('_') and callable(getattr(time_integrator, attr))]
            print(f"   Time integrator methods: {integrator_methods}")
        
        print("\n🎯 Summary:")
        print("   • VectraSim configuration: ✅ Compatible")
        print("   • JAX-Fluids loading: ✅ Successful")
        print("   • Buffer initialization: ✅ Working")
        print("   • SimulationManager: ✅ Available")
        print("   • Ready for execution: ✅ Yes")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in API test: {e}")
        print("🔍 Detailed traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_jaxfluids_api() 