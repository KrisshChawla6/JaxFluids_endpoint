#!/usr/bin/env python3
"""
Quick Propeller Wind Tunnel Test
Simple test script to verify the external flow system works
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

def main():
    """Quick test of propeller wind tunnel setup"""
    
    print("🌪️ QUICK PROPELLER WIND TUNNEL TEST")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not set!")
        print("Set it with: export GEMINI_API_KEY='your_key'")
        return
    
    try:
        from main_external_flow_api import create_external_flow_simulation
        
        print("🚀 Running external flow simulation...")
        
        response = create_external_flow_simulation(
            user_prompt="create a windtunnel in the x+ direction and subsonic speed for aerodynamics of propeller",
            output_directory=r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\testing_externalflow",
            custom_sdf_directory=r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\testing_externalflow"
        )
        
        if response.success:
            print("✅ SUCCESS!")
            print(f"📁 Files saved to: {response.simulation_directory}")
            print(f"🔢 Numerical setup: {response.numerical_setup_file}")
            print(f"🌪️ Case setup: {response.case_setup_file}")
            print(f"🚀 Run script: {response.run_script_file}")
            print(f"⏱️ Time: {response.processing_time:.2f}s")
        else:
            print("❌ FAILED!")
            print(f"Error: {response.message}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 