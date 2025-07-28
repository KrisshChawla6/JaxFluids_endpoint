#!/usr/bin/env python3
"""
Test Production Endpoint
Simple test to verify the production endpoint works correctly
"""

import os
import sys

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from cfd_parameter_agent import CFDParameterAgent
        print("   ✅ CFDParameterAgent imported successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import CFDParameterAgent: {e}")
        return False
    
    try:
        from wind_tunnel_generator import (
            WindTunnelConfigGenerator, 
            WindTunnelConfig, 
            FlowType,
            create_preset_configs
        )
        print("   ✅ Wind tunnel generator imported successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import wind tunnel generator: {e}")
        return False
    
    return True

def test_agent_initialization():
    """Test agent initialization"""
    print("\n🚀 Testing agent initialization...")
    
    try:
        from cfd_parameter_agent import CFDParameterAgent
        agent = CFDParameterAgent()
        print("   ✅ Agent initialized successfully")
        return True
    except Exception as e:
        print(f"   ❌ Failed to initialize agent: {e}")
        return False

def test_config_generation():
    """Test configuration generation"""
    print("\n⚙️  Testing config generation...")
    
    try:
        from wind_tunnel_generator import WindTunnelConfigGenerator, WindTunnelConfig, FlowType
        
        # Create config
        config = WindTunnelConfig(
            flow_type=FlowType.EULER,
            mach_number=0.3,
            max_iterations=100
        )
        
        # Generate config file
        generator = WindTunnelConfigGenerator()
        output_file = "test_production_config.cfg"
        generator.generate_config(config, output_file)
        
        # Check file was created
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"   ✅ Config generated: {output_file} ({file_size} bytes)")
            
            # Clean up
            os.remove(output_file)
            return True
        else:
            print(f"   ❌ Config file not created")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to generate config: {e}")
        return False

def test_preset_configs():
    """Test preset configurations"""
    print("\n🎯 Testing preset configurations...")
    
    try:
        from wind_tunnel_generator import create_preset_configs
        
        presets = create_preset_configs()
        print(f"   ✅ Found {len(presets)} preset configurations:")
        
        for name in presets.keys():
            print(f"      • {name}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to load presets: {e}")
        return False

def test_natural_language_parsing():
    """Test natural language parsing"""
    print("\n🗣️  Testing natural language parsing...")
    
    try:
        from cfd_parameter_agent import CFDParameterAgent
        
        agent = CFDParameterAgent()
        
        # Test prompt parsing
        test_prompt = "Project 3 propeller analysis at 8 degrees"
        params = agent._parse_prompt_to_parameters(test_prompt)
        
        print(f"   ✅ Parsed prompt: '{test_prompt}'")
        print(f"      Flow type: {params.get('flow_type')}")
        print(f"      Angle of attack: {params.get('angle_of_attack')}°")
        print(f"      Mesh file: {params.get('mesh_filename')}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to parse prompt: {e}")
        return False

def test_simulation_creation():
    """Test simulation directory creation"""
    print("\n📁 Testing simulation creation...")
    
    try:
        from cfd_parameter_agent import CFDParameterAgent
        
        agent = CFDParameterAgent()
        
        # Create simulation (without running)
        sim_dir = agent.create_simulation_from_prompt(
            "Quick Euler analysis for propeller at Mach 0.3",
            output_dir="test_production_sim"
        )
        
        # Check if directory and config were created
        config_file = os.path.join(sim_dir, "config.cfg")
        
        if os.path.exists(sim_dir) and os.path.exists(config_file):
            print(f"   ✅ Simulation created: {sim_dir}")
            print(f"   ✅ Config file created: {config_file}")
            
            # Clean up
            import shutil
            shutil.rmtree(sim_dir)
            
            return True
        else:
            print(f"   ❌ Simulation directory or config not created")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to create simulation: {e}")
        return False

def test_compatibility_functions():
    """Test compatibility functions"""
    print("\n🔄 Testing compatibility functions...")
    
    try:
        from cfd_parameter_agent import create_config_with_extracted_markers, WindTunnelSimulation
        
        # Test compatibility function
        config = create_config_with_extracted_markers(
            mesh_file_path="test_mesh.su2",
            solver_type="EULER",
            mach_number=0.3,
            max_iterations=100
        )
        
        print("   ✅ Compatibility function works")
        
        # Test compatibility class
        sim = WindTunnelSimulation()
        print("   ✅ Compatibility class works")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Compatibility test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Production Endpoint Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_agent_initialization,
        test_config_generation,
        test_preset_configs,
        test_natural_language_parsing,
        test_simulation_creation,
        test_compatibility_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n" + "=" * 50)
    print(f"🎯 TEST RESULTS")
    print(f"=" * 50)
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ Production endpoint is ready for deployment!")
        print(f"🚀 You can now replace your existing simulation_configs_agent")
    else:
        print(f"\n⚠️  {total-passed} tests failed")
        print(f"🔍 Please check the error messages above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 