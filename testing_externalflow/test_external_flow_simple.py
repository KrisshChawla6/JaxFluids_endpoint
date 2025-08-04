#!/usr/bin/env python3
"""
Simple External Flow Test
Tests the external flow endpoint with a basic subsonic wind tunnel setup
"""

import os
import sys
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    print("⚠️ python-dotenv not found, using system environment variables")

# Add External_flow_endpoint to path
sys.path.insert(0, str(Path(__file__).parent.parent / "External_flow_endpoint"))

def main():
    """Run a simple external flow test"""
    
    print("🚀 SIMPLE EXTERNAL FLOW ENDPOINT TEST")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable is required")
        return False
    
    print(f"✅ API Key found: {api_key[:20]}...")
    
    # Set output directory
    output_dir = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\testing_externalflow"
    
    try:
        from main_external_flow_api import create_external_flow_simulation
        print("✅ External flow API imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import external flow API: {e}")
        return False
    
    # Simple test prompt
    test_prompt = "Create a subsonic external flow wind tunnel for the propeller in the x+ direction"
    
    print(f"\n🔧 SIMULATION CONFIGURATION")
    print(f"📝 Prompt: {test_prompt}")
    print(f"📁 Output Directory: {output_dir}")
    
    print(f"\n🚀 STARTING SIMULATION GENERATION...")
    print("-" * 50)
    
    try:
        # Run the external flow simulation
        response = create_external_flow_simulation(
            user_prompt=test_prompt,
            output_directory=output_dir,
            gemini_api_key=api_key
        )
        
        if response.success:
            print(f"\n✅ SIMULATION SETUP COMPLETED SUCCESSFULLY!")
            print(f"📁 Simulation Directory: {response.simulation_directory}")
            print(f"📊 Processing Time: {response.processing_time:.2f} seconds")
            
            print(f"\n📋 GENERATED FILES:")
            if response.numerical_setup_file:
                print(f"   • Numerical Setup: {os.path.basename(response.numerical_setup_file)}")
            if response.case_setup_file:
                print(f"   • Case Setup: {os.path.basename(response.case_setup_file)}")
            if response.run_script_file:
                print(f"   • Run Script: {os.path.basename(response.run_script_file)}")
            
            if response.extracted_parameters:
                print(f"\n🔧 EXTRACTED PARAMETERS:")
                for category, params in response.extracted_parameters.items():
                    if params:
                        print(f"   • {category}: {len(params)} parameters")
            
            print(f"\n🎯 NEXT STEPS:")
            print(f"   1. Review generated configuration files")
            print(f"   2. Execute the run script to start simulation")
            print(f"   3. Monitor simulation progress")
            
            if response.run_script_file:
                print(f"\n💡 To run the simulation:")
                print(f"   cd \"{response.simulation_directory}\"")
                print(f"   python {os.path.basename(response.run_script_file)}")
            
            return True
            
        else:
            print(f"\n❌ SIMULATION SETUP FAILED!")
            print(f"Error: {response.message}")
            if response.error_details:
                print(f"Details: {response.error_details}")
            return False
            
    except Exception as e:
        print(f"\n❌ SIMULATION GENERATION FAILED!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎉 EXTERNAL FLOW TEST COMPLETE!")
        print("The simulation setup is ready for execution.")
    else:
        print(f"\n💥 EXTERNAL FLOW TEST FAILED!")
        sys.exit(1)