#!/usr/bin/env python3
"""
Basic usage examples for JAX-Fluids post-processing package.
"""

from pathlib import Path
import jax_fluids_postprocess as jfp

def example_quick_visualization():
    """Example: Quick visualization of simulation results."""
    print("=== Quick Visualization Example ===")
    
    # Path to your simulation results
    results_path = "subsonic_wind_tunnel_external_flow-1/domain"
    mesh_path = "data/mesh/5_bladed_Propeller.STEP_medium_tetrahedral.msh"
    
    # Quick visualization (process + visualize in one call)
    jfp.quick_visualization(
        results_path=results_path,
        variable="velocity_magnitude",
        mesh_path=mesh_path
    )


def example_full_workflow():
    """Example: Full processing workflow with all features."""
    print("=== Full Workflow Example ===")
    
    # Paths
    results_path = "subsonic_wind_tunnel_external_flow-1/domain"
    output_path = "postprocess_output"
    mesh_path = "data/mesh/5_bladed_Propeller.STEP_medium_tetrahedral.msh"
    
    # Step 1: Process simulation data
    print("🔄 Processing simulation data...")
    results = jfp.process_simulation(
        results_path=results_path,
        output_path=output_path,
        time_index=-1  # Last time step
    )
    
    print(f"✓ Processed data with shape: {results['grid_info']['shape']}")
    print(f"✓ Available variables: {list(results['flow_data'].keys())}")
    
    # Step 2: Create interactive visualization
    print("🎨 Creating interactive visualization...")
    jfp.create_visualization(
        flow_data=results['flow_data'],
        variable="pressure",
        mesh_path=mesh_path,
        interactive=True
    )
    
    # Step 3: Export to VTK format
    print("📤 Exporting to VTK...")
    vtk_files = jfp.export_vtk(
        flow_data=results['flow_data'],
        output_path=output_path,
        mesh_path=mesh_path
    )
    
    print(f"✓ VTK files: {vtk_files}")
    
    # Step 4: Create animation
    print("🎬 Creating animation...")
    animation_path = jfp.create_animation(
        results_path=results_path,
        output_path=output_path,
        variable="velocity_magnitude",
        plane="xy",
        plane_value=0.5,
        fps=10
    )
    
    print(f"✓ Animation saved: {animation_path}")


def example_using_classes():
    """Example: Using the classes directly for advanced control."""
    print("=== Advanced Class Usage Example ===")
    
    # Initialize processor
    processor = jfp.FluidProcessor(
        results_path="subsonic_wind_tunnel_external_flow-1/domain",
        output_path="advanced_output"
    )
    
    # Extract flow data for specific time step
    flow_data = processor.extract_flow_variables(time_index=5)
    
    # Get metadata
    metadata = processor.get_simulation_metadata()
    print(f"Simulation info: {metadata['num_time_steps']} time steps")
    
    # Create custom visualization
    visualizer = jfp.InteractiveVisualizer()
    visualizer.show_interactive(
        flow_data=flow_data,
        variable="density",
        mesh_path="data/mesh/5_bladed_Propeller.STEP_medium_tetrahedral.msh"
    )
    
    # Save summary
    summary_path = processor.save_summary(flow_data)
    print(f"✓ Summary saved: {summary_path}")


def example_data_reader():
    """Example: Using the data reader for custom analysis."""
    print("=== Data Reader Example ===")
    
    # Initialize data reader
    reader = jfp.DataReader("subsonic_wind_tunnel_external_flow-1/domain")
    
    # Get available variables
    variables = reader.get_variable_names()
    print(f"Available variables: {variables}")
    
    # Read specific variable
    pressure = reader.read_variable("pressure", time_index=-1)
    print(f"Pressure field shape: {pressure.shape}")
    print(f"Pressure range: {pressure.min():.3f} to {pressure.max():.3f}")
    
    # Read multiple variables
    multi_data = reader.read_variables(["velocity", "pressure"], time_index=0)
    print(f"Multi-variable data: {list(multi_data.keys())}")
    
    # Get metadata
    metadata = reader.get_metadata()
    print(f"Grid dimensions: {metadata['grid_shape']}")
    print(f"Time range: {metadata['time_range']}")


def example_batch_processing():
    """Example: Batch processing multiple time steps."""
    print("=== Batch Processing Example ===")
    
    results = jfp.batch_export(
        results_path="subsonic_wind_tunnel_external_flow-1/domain",
        output_path="batch_output",
        variables=["velocity_magnitude", "pressure", "density"],
        time_indices=[0, 5, 10]  # Process specific time steps
    )
    
    print(f"✓ Processed {len(results)} time steps")
    for time_idx, result in results.items():
        print(f"  Time {time_idx}: {len(result['vtk_files'])} VTK files")


if __name__ == "__main__":
    # Run examples (comment out the ones you don't want to run)
    
    print("JAX-Fluids Post-Processing Examples")
    print("=" * 40)
    
    # Basic examples
    # example_quick_visualization()
    # example_full_workflow()
    
    # Advanced examples
    # example_using_classes()
    # example_data_reader()
    # example_batch_processing()
    
    print("\n✅ Examples completed!")
    print("\nTo run specific examples, uncomment the desired function calls above.")