#!/usr/bin/env python3
"""
Test External Flow Endpoint Configuration Generation Only
Focus on testing the AI agents generating numerical_setup.json and case configuration
"""

import os
import sys
import json
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    print("Warning: python-dotenv not installed, trying without .env file loading")

# Add External_flow_endpoint to path
sys.path.insert(0, str(Path(__file__).parent.parent / "External_flow_endpoint"))

def test_endpoint_configuration():
    """Test only the configuration generation from external_flow_endpoint"""
    
    print("🔧 TESTING EXTERNAL FLOW ENDPOINT CONFIGURATION")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not set!")
        return False
    
    print(f"✅ API key found: {api_key[:15]}...")
    
    try:
        # Import the main API
        from main_external_flow_api import create_external_flow_simulation
        
        print("✅ External flow API imported successfully")
        
        # Test configuration generation
        user_prompt = "Create a subsonic wind tunnel simulation for propeller aerodynamics at Mach 0.3, 5 degrees angle of attack"
        output_dir = "test_config_generation"
        
        print(f"📝 Test prompt: {user_prompt}")
        print(f"📁 Output directory: {output_dir}")
        print("\n🚀 Generating configuration...")
        
        response = create_external_flow_simulation(
            user_prompt=user_prompt,
            output_directory=output_dir
        )
        
        if response.success:
            print("✅ CONFIGURATION GENERATION SUCCESSFUL!")
            print(f"📁 Simulation directory: {response.simulation_directory}")
            print(f"⏱️ Processing time: {response.processing_time:.2f}s")
            
            # Check generated files
            sim_dir = Path(response.simulation_directory)
            
            # Check for numerical_setup.json
            numerical_file = sim_dir / "numerical_setup.json"
            if numerical_file.exists():
                print("✅ numerical_setup.json generated")
                with open(numerical_file, 'r') as f:
                    numerical_config = json.load(f)
                print(f"   📊 Numerical parameters: {len(str(numerical_config).split(','))}")
                print(f"   🔧 Solver: {numerical_config.get('conservatives', {}).get('convective_fluxes', {}).get('godunov', {}).get('riemann_solver', 'Unknown')}")
                print(f"   📐 Reconstruction: {numerical_config.get('conservatives', {}).get('convective_fluxes', {}).get('godunov', {}).get('reconstruction_stencil', 'Unknown')}")
            else:
                print("❌ numerical_setup.json not found")
                return False
            
            # Check for case configuration file
            case_files = list(sim_dir.glob("*.json"))
            case_files = [f for f in case_files if f.name != "numerical_setup.json" and f.name != "simulation_summary.json"]
            
            if case_files:
                case_file = case_files[0]
                print(f"✅ Case configuration generated: {case_file.name}")
                with open(case_file, 'r') as f:
                    case_config = json.load(f)
                
                # Display key parameters
                domain = case_config.get('domain', {})
                print(f"   🏗️ Domain size: {domain.get('x', {}).get('cells', '?')}×{domain.get('y', {}).get('cells', '?')}×{domain.get('z', {}).get('cells', '?')}")
                print(f"   📏 X range: {domain.get('x', {}).get('range', '?')}")
                
                # Check boundary conditions
                bc = case_config.get('boundary_conditions', {})
                print(f"   🌬️ Inlet BC: {bc.get('primitives', {}).get('west', {}).get('type', 'Unknown')}")
                print(f"   🌬️ Outlet BC: {bc.get('primitives', {}).get('east', {}).get('type', 'Unknown')}")
                
            else:
                print("❌ Case configuration file not found")
                return False
            
            # Check for run.py (if generated)
            run_file = sim_dir / "run.py"
            if run_file.exists():
                print("✅ run.py script generated")
                print(f"   📝 Script size: {run_file.stat().st_size} bytes")
            else:
                print("⚠️ run.py not generated (expected for configuration-only test)")
            
            print("\n🎉 EXTERNAL FLOW ENDPOINT WORKING CORRECTLY!")
            print(f"   📂 Generated files in: {sim_dir}")
            print("   ✅ Numerical setup configured")
            print("   ✅ Case setup configured") 
            print("   ✅ Ready for JAX-Fluids simulation")
            
            return True
            
        else:
            print("❌ CONFIGURATION GENERATION FAILED!")
            print(f"Error: {response.message}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_endpoint_configuration()
    print("\n" + "=" * 60)
    if success:
        print("🎉 EXTERNAL FLOW ENDPOINT CONFIGURATION TEST PASSED!")
        print("✅ AI agents successfully generated JAX-Fluids configuration")
        print("✅ Ready to proceed with run.py generation agent")
    else:
        print("❌ EXTERNAL FLOW ENDPOINT NEEDS ATTENTION")
        print("🔧 Check API key and agent configuration")
    print("=" * 60) 