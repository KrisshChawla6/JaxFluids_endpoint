#!/usr/bin/env python3
"""
Test Clean Agentic JAX-Fluids System
Verify the system works with the proper Gemini 2.5 Pro integration
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    print("Warning: python-dotenv not installed, trying without .env file loading")

# Add External_flow_endpoint to path
sys.path.insert(0, str(Path(__file__).parent.parent / "External_flow_endpoint"))

def test_system():
    """Test the complete agentic system"""
    print("JAX-FLUIDS CLEAN AGENTIC SYSTEM TEST")
    print("=" * 45)
    
    # Test API key
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print(f"✓ API key found: {api_key[:15]}...")
    else:
        print("✗ No API key found")
        return False
    
    # Test SDF integration
    try:
        from sdf_integration import find_latest_sdf_file
        sdf_file, metadata = find_latest_sdf_file("../testing_externalflow")
        print(f"✓ SDF found: {Path(sdf_file).name}")
    except Exception as e:
        print(f"✗ SDF error: {e}")
        return False
    
    # Test API initialization
    try:
        from main_external_flow_api import ExternalFlowAPI, ExternalFlowRequest
        
        print("Initializing External Flow API...")
        api = ExternalFlowAPI()
        print("✓ API initialized successfully")
        
        # Test simulation generation
        request = ExternalFlowRequest(
            user_prompt="Create a subsonic wind tunnel for aerodynamic analysis of a propeller. Flow direction +X, Mach 0.3, high accuracy CFD.",
            output_directory="../testing_externalflow",
            simulation_name="clean_agentic_test",
            custom_sdf_directory="../testing_externalflow"
        )
        
        print("Processing simulation request...")
        response = api.process_external_flow_request(request)
        
        if response.success:
            print("✓ Simulation completed successfully!")
            print(f"✓ Directory: {response.simulation_directory}")
            print(f"✓ Time: {response.processing_time:.2f}s")
            
            # List generated files
            sim_dir = Path(response.simulation_directory)
            files = list(sim_dir.glob("*.json")) + list(sim_dir.glob("*.py"))
            print(f"✓ Generated {len(files)} files:")
            for file in files:
                print(f"  - {file.name}")
            
            return True
        else:
            print(f"✗ Simulation failed: {response.error_message}")
            return False
            
    except Exception as e:
        print(f"✗ System test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_system()
    print("\n" + "=" * 45)
    if success:
        print("🎉 CLEAN AGENTIC SYSTEM WORKING PERFECTLY!")
    else:
        print("❌ SYSTEM NEEDS ATTENTION")
    print("=" * 45) 