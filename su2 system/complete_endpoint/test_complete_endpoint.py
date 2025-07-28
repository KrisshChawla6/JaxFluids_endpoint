#!/usr/bin/env python3
"""
Test script for Complete Wind Tunnel Endpoint
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from complete_wind_tunnel_endpoint import (
    CompleteWindTunnelEndpoint,
    CompleteWindTunnelRequest,
    process_json_request
)

def test_basic_functionality():
    """Test basic endpoint functionality"""
    
    print("🧪 Testing Complete Wind Tunnel Endpoint")
    print("=" * 50)
    
    # Check if propeller file exists
    propeller_file = "../packaged_wind-tunnel_endpoint/propeller_only.vtk"
    if not os.path.exists(propeller_file):
        print(f"❌ Test file not found: {propeller_file}")
        print("   Please ensure the packaged wind tunnel endpoint has the propeller file")
        return False
    
    try:
        # Test 1: Direct API usage
        print("\n🔬 Test 1: Direct API Usage")
        print("-" * 30)
        
        request = CompleteWindTunnelRequest(
            object_mesh_file=propeller_file,
            tunnel_type="compact",
            flow_type="EULER",
            mach_number=0.2,
            angle_of_attack=3.0,
            max_iterations=50,
            output_directory="test_output_direct",
            simulation_name="test_simulation"
        )
        
        endpoint = CompleteWindTunnelEndpoint()
        result = endpoint.process_complete_request(request)
        
        if result.success:
            print(f"✅ Direct API test passed")
            print(f"   Wind tunnel file: {result.wind_tunnel_file}")
            print(f"   Config file: {result.config_file}")
            print(f"   Total time: {result.total_time:.2f}s")
        else:
            print(f"❌ Direct API test failed: {result.message}")
            return False
        
        # Test 2: JSON file processing
        print("\n🔬 Test 2: JSON File Processing")
        print("-" * 30)
        
        # Create temporary JSON file
        test_params = {
            "object_mesh_file": propeller_file,
            "tunnel_type": "standard",
            "flow_type": "EULER",
            "mach_number": 0.3,
            "angle_of_attack": 5.0,
            "max_iterations": 75,
            "output_directory": "test_output_json",
            "simulation_name": "json_test"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_params, f, indent=2)
            temp_json = f.name
        
        try:
            result = process_json_request(temp_json)
            
            if result.success:
                print(f"✅ JSON processing test passed")
                print(f"   Output files: {len(result.output_files)} files generated")
            else:
                print(f"❌ JSON processing test failed: {result.message}")
                return False
        finally:
            os.unlink(temp_json)
        
        # Test 3: Natural language prompt
        print("\n🔬 Test 3: Natural Language Prompt")
        print("-" * 30)
        
        request = CompleteWindTunnelRequest(
            object_mesh_file=propeller_file,
            prompt="Create a quick Euler simulation at Mach 0.25 with 2 degrees angle of attack and 60 iterations",
            output_directory="test_output_prompt",
            simulation_name="prompt_test"
        )
        
        result = endpoint.process_complete_request(request)
        
        if result.success:
            print(f"✅ Natural language prompt test passed")
            print(f"   Processing time: {result.total_time:.2f}s")
        else:
            print(f"❌ Natural language prompt test failed: {result.message}")
            return False
        
        print(f"\n🎉 All tests passed successfully!")
        print(f"📁 Test outputs created in:")
        print(f"   - test_output_direct/")
        print(f"   - test_output_json/")
        print(f"   - test_output_prompt/")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_validation():
    """Test parameter validation"""
    
    print("\n🧪 Testing Parameter Validation")
    print("=" * 50)
    
    try:
        endpoint = CompleteWindTunnelEndpoint()
        
        # Test invalid flow type
        print("\n🔬 Test: Invalid Flow Type")
        request = CompleteWindTunnelRequest(
            object_mesh_file="dummy.vtk",
            flow_type="INVALID_FLOW"
        )
        
        result = endpoint.process_complete_request(request)
        if not result.success:
            print("✅ Invalid flow type correctly rejected")
        else:
            print("❌ Invalid flow type should have been rejected")
            return False
        
        # Test missing file
        print("\n🔬 Test: Missing Input File")
        request = CompleteWindTunnelRequest(
            object_mesh_file="nonexistent_file.vtk"
        )
        
        result = endpoint.process_complete_request(request)
        if not result.success:
            print("✅ Missing input file correctly handled")
        else:
            print("❌ Missing input file should have been handled")
            return False
        
        print(f"\n✅ Parameter validation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Parameter validation test failed: {str(e)}")
        return False

def cleanup_test_outputs():
    """Clean up test output directories"""
    
    print("\n🧹 Cleaning up test outputs...")
    
    test_dirs = ["test_output_direct", "test_output_json", "test_output_prompt"]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            import shutil
            try:
                shutil.rmtree(test_dir)
                print(f"   ✅ Removed {test_dir}/")
            except Exception as e:
                print(f"   ⚠️ Could not remove {test_dir}/: {e}")

def main():
    """Main test function"""
    
    print("🚀 Complete Wind Tunnel Endpoint Test Suite")
    print("=" * 60)
    
    # Run tests
    basic_test_passed = test_basic_functionality()
    validation_test_passed = test_parameter_validation()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    if basic_test_passed:
        print("✅ Basic functionality tests: PASSED")
    else:
        print("❌ Basic functionality tests: FAILED")
    
    if validation_test_passed:
        print("✅ Parameter validation tests: PASSED")
    else:
        print("❌ Parameter validation tests: FAILED")
    
    overall_success = basic_test_passed and validation_test_passed
    
    if overall_success:
        print("\n🎉 ALL TESTS PASSED - Complete endpoint is working correctly!")
        
        # Ask about cleanup
        try:
            response = input("\n🧹 Clean up test output directories? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                cleanup_test_outputs()
        except KeyboardInterrupt:
            print("\n")
        
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Please check the error messages above")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 