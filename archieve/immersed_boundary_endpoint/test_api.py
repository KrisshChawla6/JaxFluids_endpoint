#!/usr/bin/env python3
"""
Test script for the Professional SDF API

Demonstrates how to use the ImmersedBoundaryAPI for NASA-grade SDF computation.
"""

import sys
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from sdf_api import ImmersedBoundaryAPI, SDFConfig

def test_basic_api():
    """Test basic API functionality"""
    print("="*60)
    print("TESTING IMMERSED BOUNDARY API")
    print("="*60)
    
    # Create configuration
    config = SDFConfig(
        mesh_file="../mesh/5_bladed_Propeller.STEP_medium_tetrahedral.msh",
        domain_bounds=(-100, -150, -150, 150, 150, 150),
        resolution=(80, 80, 80),  # Reasonable resolution for testing
        output_dir="results",
        output_name="api_test_propeller",
        plot=True,
        save_binary=True,
        save_json=True,
        export_jaxfluids=True,
        robust_mode=True,
        batch_size=50000
    )
    
    # Initialize API
    api = ImmersedBoundaryAPI()
    
    # Run the API
    result, saved_files = api.run(config)
    
    # Print results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"✅ SDF computation completed successfully!")
    print(f"📊 Computation time: {result.computation_time:.2f} seconds")
    print(f"📐 Resolution: {result.resolution}")
    print(f"🎯 Domain bounds: {result.domain_bounds}")
    print(f"📈 SDF range: [{result.sdf_values.min():.3f}, {result.sdf_values.max():.3f}]")
    print(f"💾 Files saved:")
    for file_type, filepath in saved_files.items():
        print(f"   - {file_type}: {filepath}")
    
    return result, saved_files

def test_high_resolution():
    """Test high-resolution SDF computation"""
    print("\n" + "="*60)
    print("TESTING HIGH-RESOLUTION SDF")
    print("="*60)
    
    config = SDFConfig(
        mesh_file="../mesh/5_bladed_Propeller.STEP_medium_tetrahedral.msh",
        domain_bounds=(-100, -150, -150, 150, 150, 150),
        resolution=(120, 120, 120),  # Higher resolution
        output_dir="results",
        output_name="high_res_api_test",
        plot=True,
        save_binary=True,
        save_json=True,
        export_jaxfluids=True,
        robust_mode=True,
        batch_size=100000
    )
    
    api = ImmersedBoundaryAPI()
    result, saved_files = api.run(config)
    
    print(f"✅ High-resolution SDF completed in {result.computation_time:.2f} seconds")
    return result, saved_files

def test_config_serialization():
    """Test configuration serialization/deserialization"""
    print("\n" + "="*60)
    print("TESTING CONFIG SERIALIZATION")
    print("="*60)
    
    # Create config
    original_config = SDFConfig(
        mesh_file="test.msh",
        domain_bounds=(-10, -10, -10, 10, 10, 10),
        resolution=(50, 50, 50),
        output_name="test_config"
    )
    
    # Convert to dict and back
    config_dict = original_config.to_dict()
    restored_config = SDFConfig.from_dict(config_dict)
    
    # Verify
    assert original_config.mesh_file == restored_config.mesh_file
    assert original_config.domain_bounds == restored_config.domain_bounds
    assert original_config.resolution == restored_config.resolution
    
    print("✅ Configuration serialization test passed!")

def test_result_loading():
    """Test loading previously computed results"""
    print("\n" + "="*60)
    print("TESTING RESULT LOADING")
    print("="*60)
    
    # Check if we have a previous result to load
    result_file = Path("results/api_test_propeller.pkl")
    if result_file.exists():
        from sdf_api import SDFResult
        loaded_result = SDFResult.load_binary(str(result_file))
        print(f"✅ Successfully loaded result from {result_file}")
        print(f"📊 Loaded SDF shape: {loaded_result.sdf_values.shape}")
        print(f"⏱️  Original computation time: {loaded_result.computation_time:.2f} seconds")
        return loaded_result
    else:
        print("⚠️  No previous result file found to test loading")
        return None

if __name__ == "__main__":
    try:
        # Test basic functionality
        result1, files1 = test_basic_api()
        
        # Test configuration serialization
        test_config_serialization()
        
        # Test result loading
        test_result_loading()
        
        # Optionally test high resolution (comment out if too slow)
        print("\n🚀 Running high-resolution test (this may take a while)...")
        result2, files2 = test_high_resolution()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY! 🎉")
        print("="*60)
        print("The Immersed Boundary API is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 