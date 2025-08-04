#!/usr/bin/env python3
"""
Demo script for the new professional JAX-Fluids post-processing package.
"""

import jax_fluids_postprocess as jfp

def main():
    print("🚀 JAX-Fluids Post-Processing Package Demo")
    print("=" * 50)
    
    # Test package information
    print(f"📦 Package version: {jfp.__version__}")
    print(f"👥 Author: {jfp.__author__}")
    print()
    
    # Test basic API availability
    print("🔍 Available API Functions:")
    api_functions = [
        'process_simulation', 'create_visualization', 'export_vtk', 
        'create_animation', 'quick_visualization'
    ]
    
    for func in api_functions:
        status = "✅" if hasattr(jfp, func) else "❌"
        print(f"  {status} {func}")
    
    print()
    print("🏗️  Available Classes:")
    classes = [
        'FluidProcessor', 'DataReader', 'InteractiveVisualizer', 
        'AnimationCreator', 'H5Reader', 'VTKExporter'
    ]
    
    for cls in classes:
        status = "✅" if hasattr(jfp, cls) else "❌"
        print(f"  {status} {cls}")
    
    print()
    
    # Test with actual data if available
    results_path = "subsonic_wind_tunnel_external_flow-1/domain"
    
    try:
        print("🔄 Testing with simulation data...")
        
        # Initialize data reader
        reader = jfp.DataReader(results_path)
        print(f"  ✅ DataReader initialized")
        print(f"  📊 Grid dimensions: {reader.get_grid_dimensions()}")
        print(f"  ⏱️  Time steps: {len(reader.get_time_steps())}")
        print(f"  🔬 Variables: {reader.get_variable_names()}")
        
        # Test processor (just initialization)
        processor = jfp.FluidProcessor(
            results_path=results_path,
            output_path="demo_output"
        )
        print(f"  ✅ FluidProcessor initialized")
        
        print("  🎯 Real data test completed successfully!")
        
    except Exception as e:
        print(f"  ⚠️  Real data test failed: {e}")
        print("  (This is expected if simulation data is not available)")
    
    print()
    print("🎉 Package demo completed!")
    print("\n💡 Usage Examples:")
    print("   # Quick visualization")
    print("   jfp.quick_visualization('path/to/results', variable='pressure')")
    print()
    print("   # Full workflow")  
    print("   results = jfp.process_simulation('path/to/results', 'output')")
    print("   jfp.create_visualization(results['flow_data'])")
    print()
    print("   # Export to VTK")
    print("   jfp.export_vtk(results['flow_data'], 'vtk_output')")


if __name__ == "__main__":
    main()