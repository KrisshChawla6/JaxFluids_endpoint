#!/usr/bin/env python3
"""
Debug JAX-Fluids Generator
Find the exact issue causing empty output
"""

import os
import sys
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# Add External_flow_endpoint to path
sys.path.insert(0, str(Path(__file__).parent.parent / "External_flow_endpoint"))

def debug_generator():
    """Debug the generator step by step"""
    
    print("🔍 DEBUGGING JAX-FLUIDS GENERATOR")
    print("=" * 50)
    
    try:
        # Test basic import
        print("1. Testing imports...")
        from jaxfluids_run_generator import JAXFluidsRunGenerator, RunScriptConfig
        print("✅ Imports successful")
        
        # Test basic initialization
        print("2. Testing initialization...")
        generator = JAXFluidsRunGenerator()
        print("✅ Generator initialized")
        
        # Test config creation
        print("3. Testing config creation...")
        config = RunScriptConfig(
            case_setup_file="test_case.json",
            numerical_setup_file="test_numerical.json"
        )
        print("✅ Config created")
        
        # Test analysis with minimal data
        print("4. Testing analysis...")
        case_config = {
            "domain": {
                "x": {"cells": 100, "range": [-1, 1]},
                "y": {"cells": 50, "range": [-1, 1]},
                "z": {"cells": 50, "range": [-1, 1]}
            }
        }
        numerical_config = {
            "active_physics": {
                "is_convective_flux": True,
                "is_viscous_flux": True,
                "is_levelset": True
            }
        }
        
        analysis = generator.analyze_simulation_config(case_config, numerical_config)
        print(f"✅ Analysis successful: {analysis}")
        
        # Test template generation
        print("5. Testing template...")
        template = generator._get_base_template()
        print(f"✅ Template length: {len(template)} characters")
        
        # Test fallback script
        print("6. Testing fallback script...")
        fallback = generator._get_fallback_script(config)
        print(f"✅ Fallback script length: {len(fallback)} characters")
        
        # Test script content generation
        print("7. Testing script content generation...")
        script_content = generator._generate_script_content(config, analysis)
        print(f"✅ Script content length: {len(script_content)} characters")
        
        if len(script_content) > 0:
            print("✅ Script generation successful!")
            
            # Write test file
            test_file = Path("debug_test_run.py")
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(script_content)
            print(f"✅ Test script written to: {test_file}")
            
            # Show first few lines
            lines = script_content.split('\n')
            print("\n📄 Generated script preview:")
            for i, line in enumerate(lines[:10]):
                print(f"   {i+1:2d}: {line}")
                
            return True
        else:
            print("❌ Script content is empty!")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_generator()
    print("\n" + "=" * 50)
    if success:
        print("✅ DEBUG SUCCESSFUL - Generator working!")
    else:
        print("❌ DEBUG FAILED - Found the issue") 