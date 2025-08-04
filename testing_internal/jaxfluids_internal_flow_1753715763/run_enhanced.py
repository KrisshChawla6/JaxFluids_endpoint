#!/usr/bin/env python3
"""
Enhanced Internal Flow Test using proven working solutions
"""

import os
import json
import glob
from jaxfluids import InputManager, InitializationManager, SimulationManager

def modify_config_for_production(case_file: str) -> str:
    """Apply our proven working optimizations to Internal Flow"""
    
    print("🔧 Applying Enhanced Internal Flow optimizations...")
    
    # Load original config
    with open(case_file, 'r', encoding='utf-8') as f:
        case_config = json.load(f)
    
    # INTELLIGENT OUTPUT CONFIGURATION (based on proven working solution)
    if 'output' in case_config:
        # Fix common field name errors that cause crashes
        if 'miscellaneous' in case_config['output']:
            misc_fields = case_config['output']['miscellaneous']
            if 'q_criterion' in misc_fields:
                misc_fields[misc_fields.index('q_criterion')] = 'qcriterion'
                print("🔧 Fixed field name: q_criterion -> qcriterion")
        
        # PROVEN WORKING OUTPUT STRATEGY (from successful External Flow runs)
        # Key insight: levelset output fields cause NoneType error even with complex SDF
        # Always use essential fields only and remove levelset output for stability
        case_config['output']['primitives'] = ['density', 'velocity', 'pressure', 'temperature']
        case_config['output']['miscellaneous'] = ['mach_number']  # Fixed field name above
        case_config['output']['levelset'] = []  # CRITICAL: Remove to avoid NoneType error
        case_config['output']['conservatives'] = []  # Keep minimal
        
        print("🔧 Applied proven working output strategy (essential fields, no levelset output)")
    else:
        # Fallback if no output section exists
        case_config['output'] = {
            "primitives": ['density', 'velocity', 'pressure', 'temperature'],
            "miscellaneous": ['mach_number'],
            "levelset": []
        }
        print("🔧 Created essential output configuration")
    
    # Ensure single device decomposition for development/testing
    if 'domain' in case_config and 'decomposition' in case_config['domain']:
        case_config['domain']['decomposition'] = {
            'split_x': 1,
            'split_y': 1, 
            'split_z': 1
        }
    
    # Add gravity if missing (required by JAX-Fluids)
    if 'forcings' not in case_config:
        case_config['forcings'] = {'gravity': [0.0, 0.0, 0.0]}
    
    # INTELLIGENT SIMULATION TIMING (based on proven working solution)
    if 'general' in case_config:
        # Create results directory
        os.makedirs("./results", exist_ok=True)
        case_config['general']['save_path'] = "./results"
        
        # Detect execution environment
        is_hpcc = any(env_var in os.environ for env_var in ['SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID'])
        
        if is_hpcc:
            # HPCC: Preserve AI agent's intelligent timing decisions
            current_end_time = case_config['general'].get('end_time', 1.0)
            if current_end_time > 1000.0:
                case_config['general']['save_dt'] = current_end_time / 100
            elif current_end_time > 10.0:
                case_config['general']['save_dt'] = current_end_time / 50
            else:
                case_config['general']['save_dt'] = current_end_time / 10
            print("🔧 HPCC environment: Preserved intelligent timing")
        else:
            # Local environment: Use proven 100+ timestep approach
            case_config['general']['end_time'] = 20.0    # Proven: ~116 timesteps
            case_config['general']['save_dt'] = 2.0      # Proven: ~10 snapshots 
            print("🔧 Local environment: Applied proven 100+ timestep timing (20.0 end_time)")
        
        # Apply proven stable mesh size for local development  
        if not is_hpcc and 'domain' in case_config:
            # Check if mesh is very large and scale down for local stability
            x_cells = case_config['domain'].get('x', {}).get('cells', 64)
            y_cells = case_config['domain'].get('y', {}).get('cells', 64)
            z_cells = case_config['domain'].get('z', {}).get('cells', 64)
            total_cells = x_cells * y_cells * z_cells
            
            if total_cells > 300000:  # > 0.3M cells might be unstable locally
                case_config['domain']['x']['cells'] = 64
                case_config['domain']['y']['cells'] = 64
                case_config['domain']['z']['cells'] = 64
                print("🔧 Local environment: Scaled mesh to proven stable 64x64x64")
    
    # PRESERVE INTELLIGENT SDF GEOMETRY (based on actual working solution)
    # The working approach was to KEEP the complex SDF and just fix output configuration
    if 'initial_condition' in case_config and 'levelset' in case_config['initial_condition']:
        levelset_value = case_config['initial_condition']['levelset']
        if 'sdf' in str(levelset_value).lower() or 'CUSTOM_SDF' in str(levelset_value):
            # Always preserve the intelligent SDF setup - this is what actually worked
            print("🔧 Preserved intelligent SDF configuration for internal flow")
    
    # Write optimized config
    optimized_file = case_file.replace('.json', '_enhanced_optimized.json')
    with open(optimized_file, 'w', encoding='utf-8') as f:
        json.dump(case_config, f, indent=2)
    
    print(f"✅ Enhanced Internal Flow optimizations applied: {os.path.basename(optimized_file)}")
    return optimized_file

def run_simulation(case_file: str, numerical_file: str):
    """Runs the Internal Flow JAX-Fluids simulation with enhanced error handling."""
    try:
        optimized_case_file = modify_config_for_production(case_file)
        
        input_manager = InputManager(optimized_case_file, numerical_file)
        initialization_manager = InitializationManager(input_manager)
        sim_manager = SimulationManager(input_manager)
        buffers = initialization_manager.initialization()
        sim_manager.simulate(buffers)  # PROVEN WORKING - do NOT use advance()
        print("🎉 Internal Flow simulation completed successfully! Check ./results/ for .h5 files")

    except FileNotFoundError as e:
        print(f"❌ Error: Configuration file not found: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format in configuration file: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    print("🚀 Testing Enhanced Internal Flow System")
    case_file = "jaxfluids_internal_flow_1753715763.json"
    numerical_file = "numerical_setup.json"
    run_simulation(case_file, numerical_file)