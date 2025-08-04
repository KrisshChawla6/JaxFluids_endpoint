#!/usr/bin/env python3
"""
Test script for enhanced Internal Flow Endpoint with intelligent boundary conditions
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from main_internal_flow_api import create_internal_flow_simulation

def test_internal_flow_stack():
    """Test the enhanced internal flow endpoint"""
    
    print("🚀 TESTING ENHANCED INTERNAL FLOW ENDPOINT")
    print("=" * 60)
    
    # Set up test parameters
    mesh_file = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\mesh\Rocket Engine.msh"
    output_directory = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\testing_internal"
    
    print(f"📐 Mesh file: {mesh_file}")
    print(f"📁 Output directory: {output_directory}")
    print(f"🌪️ Testing hypersonic rocket optimization simulation...")
    
    try:
        # Test the enhanced internal flow endpoint
        response = create_internal_flow_simulation(
            user_prompt="create an hypersonic simulation for internal flow for optimization of rocket",
            mesh_file=mesh_file,
            output_directory=output_directory,
            flow_type="rocket_engine",
            mach_number=4.0,
            pressure_ratio=70.0,
            temperature_inlet=3580.0
        )
        
        # Print results
        print("\n" + "="*80)
        print("🚀 INTERNAL FLOW ENDPOINT TEST RESULTS")
        print("="*80)
        print(f"✅ Success: {response.success}")
        
        if response.success:
            print(f"📁 Simulation Directory: {response.simulation_directory}")
            print(f"📄 Case File: {response.case_file}")
            print(f"🔢 Numerical File: {response.numerical_file}")  
            print(f"🚀 Run Script: {response.run_script}")
            
            if response.boundary_conditions:
                print(f"🔴 Inlet Points: {response.boundary_conditions['inlet_points']:,}")
                print(f"🟢 Outlet Points: {response.boundary_conditions['outlet_points']:,}")
                print(f"🧠 BC Storage: {response.boundary_conditions.get('bc_storage_dir', 'N/A')}")
                
                # List generated files
                print("\n📋 GENERATED FILES:")
                if os.path.exists(response.simulation_directory):
                    for file in os.listdir(response.simulation_directory):
                        print(f"   📄 {file}")
                        
                # List BC files
                bc_dir = response.boundary_conditions.get('bc_storage_dir', '')
                if bc_dir and os.path.exists(bc_dir):
                    print(f"\n🧠 BOUNDARY CONDITION FILES ({bc_dir}):")
                    for file in os.listdir(bc_dir):
                        print(f"   🔴 {file}")
        else:
            print(f"❌ Error: {response.error_message}")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_internal_flow_stack() 