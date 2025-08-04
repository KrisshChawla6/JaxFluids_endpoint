#!/usr/bin/env python3
"""
Example Usage of Intelligent Boundary Conditions Processor
Demonstrates how to use the endpoint to convert mesh files to JAX-Fluids simulations
"""

import logging
from pathlib import Path
from intelligent_boundary_processor import IntelligentBoundaryProcessor

def setup_logging():
    """Setup logging for the example"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('example_processing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def example_basic_usage():
    """Basic usage example"""
    logger = setup_logging()
    logger.info("=== BASIC USAGE EXAMPLE ===")
    
    # Path to your mesh file
    mesh_file = "../mesh/Rocket Engine.msh"  # Adjust path as needed
    
    if not Path(mesh_file).exists():
        logger.error(f"Mesh file not found: {mesh_file}")
        logger.info("Please provide a valid mesh file path")
        return
    
    try:
        # Create processor with default settings
        processor = IntelligentBoundaryProcessor(
            mesh_file=mesh_file,
            output_dir="basic_rocket_simulation"
        )
        
        # Process the mesh (complete pipeline)
        results = processor.process_mesh()
        
        # Print summary
        print(processor.get_processing_summary())
        
        # Create visualizations
        processor.visualize_results()
        
        logger.info("Basic usage example completed successfully!")
        
    except Exception as e:
        logger.error(f"Basic usage failed: {e}")

def example_custom_parameters():
    """Example with custom parameters"""
    logger = setup_logging()
    logger.info("=== CUSTOM PARAMETERS EXAMPLE ===")
    
    mesh_file = "../mesh/Rocket Engine.msh"
    
    if not Path(mesh_file).exists():
        logger.warning(f"Mesh file not found: {mesh_file} - using default path")
        return
    
    try:
        # Custom domain and grid
        custom_domain = [-300.0, -1000.0, -1000.0, 2000.0, 1000.0, 1000.0]
        custom_grid = (160, 80, 80)  # Higher resolution
        
        processor = IntelligentBoundaryProcessor(
            mesh_file=mesh_file,
            output_dir="high_res_rocket_simulation",
            domain_bounds=custom_domain,
            grid_shape=custom_grid
        )
        
        # Process with custom settings
        results = processor.process_mesh()
        
        # Customize forcing parameters after processing
        if processor.config_generator:
            processor.config_generator.customize_forcing_parameters(
                inlet_mass_flow=20.0,      # Higher mass flow
                inlet_temperature=1800.0,   # Higher temperature  
                outlet_pressure=500000.0    # Specified outlet pressure
            )
            
            # Save updated configurations
            processor.config_generator.save_configurations(
                processor.output_dir / "config"
            )
        
        print(processor.get_processing_summary())
        logger.info("Custom parameters example completed!")
        
    except Exception as e:
        logger.error(f"Custom parameters example failed: {e}")

def example_step_by_step():
    """Step-by-step processing example"""
    logger = setup_logging()
    logger.info("=== STEP-BY-STEP PROCESSING EXAMPLE ===")
    
    mesh_file = "../mesh/Rocket Engine.msh"
    
    if not Path(mesh_file).exists():
        logger.warning(f"Mesh file not found: {mesh_file}")
        return
    
    try:
        processor = IntelligentBoundaryProcessor(
            mesh_file=mesh_file,
            output_dir="step_by_step_simulation"
        )
        
        # Manual step-by-step processing
        logger.info("Step 1: Detecting virtual faces...")
        processor._detect_virtual_faces()
        
        logger.info("Step 2: Generating SDF...")
        processor._generate_sdf()
        
        logger.info("Step 3: Creating masks...")
        processor._create_boundary_masks()
        
        logger.info("Step 4: Generating configurations...")
        processor._generate_configurations()
        
        logger.info("Step 5: Setting up directory...")
        processor._setup_simulation_directory()
        
        # Get detailed results
        if processor.virtual_faces:
            inlet_face = processor.virtual_faces["inlet"]
            outlet_face = processor.virtual_faces["outlet"]
            
            logger.info(f"Virtual faces detected:")
            logger.info(f"  Inlet: center={inlet_face['center']}, radius={inlet_face['radius']:.2f}")
            logger.info(f"  Outlet: center={outlet_face['center']}, radius={outlet_face['radius']:.2f}")
        
        if processor.sdf_matrix is not None:
            stats = processor.sdf_generator.get_sdf_stats()
            logger.info(f"SDF statistics: {stats}")
            
        if processor.boundary_masks:
            inlet_points = processor.boundary_masks["inlet"].sum()
            outlet_points = processor.boundary_masks["outlet"].sum()
            logger.info(f"Boundary masks: inlet={inlet_points}, outlet={outlet_points}")
        
        logger.info("Step-by-step processing completed!")
        
    except Exception as e:
        logger.error(f"Step-by-step processing failed: {e}")

def example_production_workflow():
    """Production-ready workflow example"""
    logger = setup_logging()
    logger.info("=== PRODUCTION WORKFLOW EXAMPLE ===")
    
    # Production parameters
    mesh_file = "../mesh/Rocket Engine.msh"
    output_base = "production_simulations"
    
    if not Path(mesh_file).exists():
        logger.warning(f"Mesh file not found: {mesh_file}")
        return
    
    # Multiple simulation cases
    cases = [
        {
            "name": "baseline",
            "mass_flow": 15.0,
            "temperature": 1500.0,
            "grid": (128, 64, 64)
        },
        {
            "name": "high_flow",
            "mass_flow": 25.0,
            "temperature": 1800.0,
            "grid": (160, 80, 80)
        },
        {
            "name": "low_temp",
            "mass_flow": 15.0,
            "temperature": 1200.0,
            "grid": (128, 64, 64)
        }
    ]
    
    for case in cases:
        try:
            logger.info(f"Processing case: {case['name']}")
            
            output_dir = f"{output_base}/{case['name']}_simulation"
            
            processor = IntelligentBoundaryProcessor(
                mesh_file=mesh_file,
                output_dir=output_dir,
                grid_shape=case["grid"]
            )
            
            # Process mesh
            results = processor.process_mesh()
            
            # Customize for this case
            if processor.config_generator:
                processor.config_generator.customize_forcing_parameters(
                    inlet_mass_flow=case["mass_flow"],
                    inlet_temperature=case["temperature"]
                )
                
                # Update case name
                processor.config_generator.case_config["general"]["case_name"] = f"rocket_nozzle_{case['name']}"
                
                # Save updated configurations
                processor.config_generator.save_configurations(
                    processor.output_dir / "config"
                )
            
            logger.info(f"Case {case['name']} completed: {output_dir}")
            
        except Exception as e:
            logger.error(f"Case {case['name']} failed: {e}")
    
    logger.info("Production workflow completed!")

def run_all_examples():
    """Run all examples"""
    print("=" * 70)
    print("INTELLIGENT BOUNDARY CONDITIONS - EXAMPLE USAGE")
    print("=" * 70)
    
    # Check for mesh file
    possible_mesh_paths = [
        "../mesh/Rocket Engine.msh",
        "../../mesh/Rocket Engine.msh", 
        "../../../mesh/Rocket Engine.msh",
        "mesh/Rocket Engine.msh"
    ]
    
    mesh_found = None
    for path in possible_mesh_paths:
        if Path(path).exists():
            mesh_found = path
            break
    
    if not mesh_found:
        print("❌ No mesh file found!")
        print("Please ensure 'Rocket Engine.msh' is available in one of these locations:")
        for path in possible_mesh_paths:
            print(f"  - {path}")
        print("\nSkipping examples...")
        return
    
    print(f"✅ Found mesh file: {mesh_found}")
    print()
    
    # Update examples to use found mesh
    global mesh_file
    mesh_file = mesh_found
    
    # Run examples
    try:
        example_basic_usage()
        print("\n" + "="*50 + "\n")
        
        example_custom_parameters()
        print("\n" + "="*50 + "\n")
        
        example_step_by_step()
        print("\n" + "="*50 + "\n")
        
        example_production_workflow()
        
    except KeyboardInterrupt:
        print("\n❌ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Examples failed: {e}")
    
    print("\n" + "=" * 70)
    print("EXAMPLES COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    run_all_examples() 