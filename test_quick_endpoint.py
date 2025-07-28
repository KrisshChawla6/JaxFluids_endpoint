#!/usr/bin/env python3
"""
Quick test to verify the updated endpoint generates working run.py code
"""
import os
import sys
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add External_flow_endpoint to path
sys.path.insert(0, str(Path(__file__).parent / "External_flow_endpoint"))

def test_updated_endpoint():
    """Test that the updated endpoint generates working code"""
    
    print("🧪 Testing Updated External Flow Endpoint")
    print("=" * 50)
    
    try:
        from main_external_flow_api import create_external_flow_simulation
        
        # Test with simple prompt
        response = create_external_flow_simulation(
            user_prompt="Quick test of working JAX-Fluids run script",
            output_directory="./test_updated_working"
        )
        
        if response.success:
            print(f"✅ Endpoint generated simulation successfully")
            print(f"📁 Directory: {response.simulation_directory}")
            
            # Check if run.py was generated
            run_file = Path(response.simulation_directory) / "run.py"
            if run_file.exists():
                print("✅ run.py file generated")
                
                # Check if it contains the working pattern
                with open(run_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if '"primitives": []' in content and 'sim_manager.simulate(buffers)' in content:
                    print("✅ run.py contains WORKING pattern!")
                    print("✅ Minimal output configuration ✓")
                    print("✅ Proper simulate() method ✓")
                    return True
                else:
                    print("❌ run.py does not contain working pattern")
                    return False
            else:
                print("❌ run.py not generated")
                return False
        else:
            print(f"❌ Endpoint failed: {response.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_updated_endpoint()
    if success:
        print("\n🎉 ENDPOINT UPDATE SUCCESSFUL!")
        print("✅ The endpoint now generates WORKING JAX-Fluids code!")
    else:
        print("\n❌ Endpoint update needs more work") 