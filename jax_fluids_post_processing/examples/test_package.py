#!/usr/bin/env python3
"""
Test the new package structure with real data.
"""

def test_package_installation():
    """Test that the package is properly installed and importable."""
    try:
        import jax_fluids_postprocess as jfp
        print("✅ Package imported successfully!")
        
        # Test API functions
        functions = ['process_simulation', 'create_visualization', 'export_vtk', 'create_animation']
        for func in functions:
            if hasattr(jfp, func):
                print(f"  ✅ {func} available")
            else:
                print(f"  ❌ {func} missing")
        
        # Test classes
        classes = ['FluidProcessor', 'InteractiveVisualizer', 'DataReader']
        for cls in classes:
            if hasattr(jfp, cls):
                print(f"  ✅ {cls} available")
            else:
                print(f"  ❌ {cls} missing")
                
        return True
        
    except ImportError as e:
        print(f"❌ Package import failed: {e}")
        return False


def test_with_real_data():
    """Test with actual simulation data if available."""
    import jax_fluids_postprocess as jfp
    from pathlib import Path
    
    results_path = Path("subsonic_wind_tunnel_external_flow-1/domain")
    
    if not results_path.exists():
        print("⚠️  Test data not found, skipping real data test")
        return
    
    print("🔄 Testing with real simulation data...")
    
    try:
        # Test data reader
        reader = jfp.DataReader(str(results_path))
        variables = reader.get_variable_names()
        print(f"  ✅ Found variables: {variables}")
        
        # Test processor
        processor = jfp.FluidProcessor(
            results_path=str(results_path),
            output_path="test_output"
        )
        print("  ✅ FluidProcessor initialized")
        
        # Test data extraction (just metadata, not full processing)
        metadata = processor.get_simulation_metadata()
        print(f"  ✅ Metadata: {metadata['num_time_steps']} time steps")
        
        print("✅ Real data test completed!")
        
    except Exception as e:
        print(f"❌ Real data test failed: {e}")


def test_cli_availability():
    """Test that CLI commands are available."""
    import subprocess
    import sys
    
    try:
        result = subprocess.run([
            sys.executable, "-c", 
            "import jax_fluids_postprocess.cli; print('CLI module available')"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ CLI module available")
        else:
            print(f"❌ CLI module test failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ CLI test failed: {e}")


if __name__ == "__main__":
    print("JAX-Fluids Post-Processing Package Test")
    print("=" * 45)
    
    # Run tests
    test_package_installation()
    print()
    test_with_real_data()
    print()
    test_cli_availability()
    
    print("\n🎉 Package testing completed!")