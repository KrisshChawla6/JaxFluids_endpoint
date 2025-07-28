#!/usr/bin/env python3
"""
Fix VectraSim-generated configuration for JAX-Fluids compatibility and test it
This ensures the boundary conditions structure matches JAX-Fluids requirements
"""

import os
import sys
import json
import traceback

def fix_boundary_conditions():
    """Fix the boundary conditions structure for JAX-Fluids compatibility"""
    
    case_file = "jaxfluids_external_flow_1753651803.json"
    
    print("🔧 Fixing Configuration for JAX-Fluids Compatibility")
    print("=" * 60)
    
    # Load case config
    with open(case_file, 'r') as f:
        case_config = json.load(f)
    
    print(f"📋 Original config loaded: {case_file}")
    
    # Fix domain decomposition for single device
    case_config['domain']['decomposition'] = {
        'split_x': 1,
        'split_y': 1,
        'split_z': 1
    }
    
    # Reduce grid size for testing
    case_config['domain']['x']['cells'] = 64
    case_config['domain']['y']['cells'] = 64
    case_config['domain']['z']['cells'] = 64
    
    # Reduce simulation time for testing
    case_config['general']['end_time'] = 5.0
    case_config['general']['save_dt'] = 1.0
    
    print(f"✅ Fixed domain decomposition: 1x1x1")
    print(f"✅ Fixed grid size: 64x64x64")
    print(f"✅ Fixed simulation time: 5.0 seconds")
    
    # Fix boundary conditions structure
    if 'primitives' in case_config['boundary_conditions']:
        print("🔧 Converting boundary conditions structure...")
        
        primitives_bc = case_config['boundary_conditions']['primitives']
        levelset_bc = case_config['boundary_conditions'].get('levelset', {})
        
        # Create the correct JAX-Fluids structure
        new_boundary_conditions = {}
        
        # All face names that JAX-Fluids expects
        faces = ['east', 'west', 'north', 'south', 'top', 'bottom']
        
        for face in faces:
            face_bc = {}
            
            # Add primitives boundary conditions
            if face in primitives_bc:
                face_bc.update(primitives_bc[face])
            
            # Add levelset boundary conditions if present
            if face in levelset_bc:
                face_bc['levelset'] = levelset_bc[face]
            
            new_boundary_conditions[face] = face_bc
        
        # Replace the boundary conditions
        case_config['boundary_conditions'] = new_boundary_conditions
        
        print("✅ Fixed boundary conditions structure")
    
    # Save the fixed config
    with open(case_file, 'w') as f:
        json.dump(case_config, f, indent=2)
    
    print(f"💾 Saved fixed configuration: {case_file}")
    return case_file

def test_jaxfluids_loading():
    """Test if JAX-Fluids can load our VectraSim-generated configuration"""
    
    print("\n🧪 Testing JAX-Fluids Configuration Loading")
    print("=" * 60)
    
    try:
        # Test JAX availability
        import jax
        print(f"✅ JAX imported: {jax.__version__}")
        
        # Test JAX-Fluids imports
        try:
            from jaxfluids import InputManager, InitializationManager, SimulationManager
            print("✅ JAX-Fluids core imports successful")
        except ImportError as e:
            print(f"❌ JAX-Fluids import failed: {e}")
            return False
        
        # Test configuration loading
        case_file = "jaxfluids_external_flow_1753651803.json"
        numerical_file = "numerical_setup.json"
        
        if not os.path.exists(case_file):
            print(f"❌ Case file not found: {case_file}")
            return False
        
        if not os.path.exists(numerical_file):
            print(f"❌ Numerical file not found: {numerical_file}")
            return False
        
        print(f"✅ Configuration files found")
        
        # Try to load with JAX-Fluids InputManager
        print("📋 Testing InputManager...")
        try:
            input_manager = InputManager(case_file, numerical_file)
            print("✅ InputManager created successfully")
            
            # Check basic properties
            print(f"📐 Domain information loaded")
            
        except Exception as e:
            print(f"❌ InputManager failed: {e}")
            print("🔍 Detailed error:")
            traceback.print_exc()
            return False
        
        # Try to create InitializationManager
        print("🚀 Testing InitializationManager...")
        try:
            initialization_manager = InitializationManager(input_manager)
            print("✅ InitializationManager created successfully")
        except Exception as e:
            print(f"❌ InitializationManager failed: {e}")
            print("🔍 Detailed error:")
            traceback.print_exc()
            return False
        
        # Try to create SimulationManager
        print("⚙️ Testing SimulationManager...")
        try:
            sim_manager = SimulationManager(input_manager)
            print("✅ SimulationManager created successfully")
        except Exception as e:
            print(f"❌ SimulationManager failed: {e}")
            print("🔍 Detailed error:")
            traceback.print_exc()
            return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ VectraSim-generated configuration is JAX-Fluids compatible")
        print("🚀 Ready for simulation execution")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("🔍 Detailed error:")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    
    print("🚀 VectraSim → JAX-Fluids Configuration Test")
    print("=" * 60)
    
    # Step 1: Fix configuration
    try:
        case_file = fix_boundary_conditions()
    except Exception as e:
        print(f"❌ Configuration fixing failed: {e}")
        traceback.print_exc()
        return False
    
    # Step 2: Test JAX-Fluids loading
    try:
        success = test_jaxfluids_loading()
        if success:
            print("\n🏁 Configuration is ready for JAX-Fluids simulation!")
            return True
        else:
            print("\n❌ Configuration needs further debugging")
            return False
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 