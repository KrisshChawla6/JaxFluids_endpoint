#!/usr/bin/env python3
"""
Debug script to check why adaptive agent detects 1D instead of 3D
"""

import os
import sys
import json
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# Add External_flow_endpoint to path
sys.path.insert(0, str(Path(__file__).parent.parent / "External_flow_endpoint"))

def debug_detection():
    """Debug the simulation detection"""
    
    print("🔍 DEBUGGING ADAPTIVE AGENT DETECTION")
    print("=" * 50)
    
    # Find the latest config directory
    config_dirs = [d for d in Path("production_test_config").iterdir() 
                   if d.is_dir() and d.name.startswith("jaxfluids_external_flow")]
    
    if not config_dirs:
        print("❌ No config directories found")
        return
    
    # Get the latest one
    latest_config = max(config_dirs, key=lambda x: x.name)
    print(f"📁 Using config: {latest_config.name}")
    
    # Find the files
    case_file = latest_config / f"{latest_config.name}.json"
    numerical_file = latest_config / "numerical_setup.json"
    
    print(f"📄 Case file: {case_file}")
    print(f"📄 Numerical file: {numerical_file}")
    
    if not case_file.exists():
        print(f"❌ Case file doesn't exist: {case_file}")
        return
    
    if not numerical_file.exists():
        print(f"❌ Numerical file doesn't exist: {numerical_file}")
        return
    
    # Load the configurations
    with open(case_file, 'r', encoding='utf-8') as f:
        case_config = json.load(f)
    
    with open(numerical_file, 'r', encoding='utf-8') as f:
        numerical_config = json.load(f)
    
    print("\n📊 CASE CONFIG DOMAIN:")
    domain = case_config.get('domain', {})
    for dim in ['x', 'y', 'z']:
        if dim in domain:
            cells = domain[dim].get('cells', 0)
            range_val = domain[dim].get('range', [])
            print(f"  {dim}: {cells} cells, range {range_val}")
        else:
            print(f"  {dim}: NOT FOUND")
    
    # Test the adaptive agent detection
    try:
        from adaptive_jaxfluids_agent import AdaptiveJAXFluidsAgent, AgenticConfig, PlottingMode
        
        agent = AdaptiveJAXFluidsAgent()
        
        config = AgenticConfig(
            case_file=case_file.name,
            numerical_file=numerical_file.name,
            simulation_intent="external flow around propeller for comprehensive aerodynamic analysis",
            plotting_mode=PlottingMode.ADVANCED
        )
        
        print("\n🤖 ADAPTIVE AGENT ANALYSIS:")
        analysis = agent.analyze_simulation(case_config, numerical_config, config)
        
        print(f"🔍 Detected dimension: {analysis['dimension']}")
        print(f"🔬 Simulation type: {analysis['simulation_type']}")
        print(f"🧮 Physics: {analysis['physics']}")
        print(f"📈 Quantities: {analysis['quantities']}")
        
        # Test dimension detection specifically
        print("\n🔎 DIMENSION DETECTION DEBUG:")
        dimension = agent._determine_dimension(case_config, None)
        print(f"Detected dimension: {dimension}")
        
        # Manual calculation
        active_dims = 0
        for dim in ['x', 'y', 'z']:
            if dim in domain:
                cells = domain[dim].get('cells', 1)
                print(f"  {dim}: {cells} cells ({'active' if cells > 1 else 'inactive'})")
                if cells > 1:
                    active_dims += 1
        
        print(f"Active dimensions: {active_dims}")
        print(f"Should be: {'1D' if active_dims <= 1 else '2D' if active_dims <= 2 else '3D'}")
        
    except Exception as e:
        print(f"❌ Error testing adaptive agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_detection() 