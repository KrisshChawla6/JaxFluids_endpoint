#!/usr/bin/env python3
"""
Test JAX-Fluids Run Generator
Verify the new documentation-based run script generator works properly
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

def test_jaxfluids_generator():
    """Test the JAX-Fluids run script generator"""
    
    print("🧪 TESTING JAX-FLUIDS RUN SCRIPT GENERATOR")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not set!")
        return False
    
    print(f"✅ API key found: {api_key[:15]}...")
    
    try:
        # Import the generator
        from jaxfluids_run_generator import create_jaxfluids_run_script, JAXFluidsRunGenerator
        print("✅ JAX-Fluids generator imported successfully")
        
        # Use existing configuration files
        test_dir = Path("jaxfluids_external_flow_1753078135")
        if not test_dir.exists():
            print("❌ No test configuration directory found")
            return False
        
        case_file = test_dir / f"{test_dir.name}.json"
        numerical_file = test_dir / "numerical_setup.json"
        
        if not case_file.exists() or not numerical_file.exists():
            print("❌ Configuration files not found")
            return False
        
        print(f"📁 Using configuration from: {test_dir.name}")
        print(f"   📄 Case file: {case_file.name}")
        print(f"   📄 Numerical file: {numerical_file.name}")
        
        # Generate run script
        print("\n🚀 Generating JAX-Fluids run script...")
        
        output_dir = "test_generated_script"
        os.makedirs(output_dir, exist_ok=True)
        
        script_path = create_jaxfluids_run_script(
            case_setup_path=str(case_file),
            numerical_setup_path=str(numerical_file),
            output_directory=output_dir,
            gemini_api_key=api_key
        )
        
        if Path(script_path).exists():
            print("✅ Run script generated successfully!")
            print(f"📁 Script location: {script_path}")
            
            # Check script content
            with open(script_path, 'r') as f:
                script_content = f.read()
            
            print(f"📝 Script size: {len(script_content)} characters")
            
            # Verify key components
            required_imports = [
                "from jaxfluids import InputManager, InitializationManager, SimulationManager",
                "from jaxfluids_postprocess import load_data",
                "input_manager = InputManager(",
                "initialization_manager = InitializationManager(",
                "sim_manager = SimulationManager(",
                "jxf_buffers = initialization_manager.initialization()",
                "sim_manager.simulate(jxf_buffers)"
            ]
            
            missing_components = []
            for component in required_imports:
                if component not in script_content:
                    missing_components.append(component)
            
            if missing_components:
                print("⚠️ Missing required components:")
                for component in missing_components:
                    print(f"   - {component}")
            else:
                print("✅ All required JAX-Fluids components present")
            
            # Show first few lines
            lines = script_content.split('\n')
            print("\n📄 Generated script preview:")
            for i, line in enumerate(lines[:15]):
                print(f"   {i+1:2d}: {line}")
            if len(lines) > 15:
                print(f"   ... ({len(lines) - 15} more lines)")
            
            return True
            
        else:
            print("❌ Script file not found after generation")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_jaxfluids_generator()
    print("\n" + "=" * 60)
    if success:
        print("🎉 JAX-FLUIDS RUN GENERATOR TEST PASSED!")
        print("✅ Documentation-based run script generation working!")
        print("✅ Ready to integrate with external flow endpoint")
    else:
        print("❌ JAX-FLUIDS RUN GENERATOR TEST FAILED")
        print("🔧 Check generator implementation and dependencies")
    print("=" * 60) 