#!/usr/bin/env python3
"""
Test Simple JAX-Fluids Agent
Clean test based on SU2 agent pattern
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

from jaxfluids_agent import generate_simulation
from sdf_integration import find_latest_sdf_file

def test_simple_agent():
    """Test the simple JAX-Fluids agent"""
    
    print("TESTING SIMPLE JAX-FLUIDS AGENT")
    print("=" * 40)
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        return False
    
    print(f"API Key: {api_key[:15]}...")
    
    # Find SDF file
    try:
        sdf_file, metadata = find_latest_sdf_file("../testing_externalflow")
        print(f"SDF Found: {Path(sdf_file).name}")
    except Exception as e:
        print(f"SDF Error: {e}")
        sdf_file = None
    
    # Generate simulation
    user_prompt = "Create a subsonic wind tunnel for aerodynamic analysis of a propeller. Flow direction +X, Mach 0.3, high accuracy CFD."
    
    print(f"Prompt: {user_prompt[:50]}...")
    
    try:
        result = generate_simulation(
            user_prompt=user_prompt,
            sdf_file=sdf_file,
            output_dir=None  # Let it create directory
        )
        
        if result['success']:
            print("SUCCESS!")
            print(f"Directory: {result['simulation_directory']}")
            print(f"Files: {len(result['files_generated'])}")
            
            # List files
            for file in result['files_generated']:
                print(f"  - {Path(file).name}")
            
            return True
        else:
            print(f"FAILED: {result['error']}")
            return False
            
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_agent()
    print("\n" + "=" * 40)
    if success:
        print("SIMPLE AGENT WORKING!")
    else:
        print("AGENT NEEDS FIXING")
    print("=" * 40) 