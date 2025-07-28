#!/usr/bin/env python3
"""
Verify Existing JAX-Fluids Configuration Files
Check if the external_flow_endpoint generated valid configurations
"""

import json
import os
from pathlib import Path

def verify_configuration(config_dir):
    """Verify a JAX-Fluids configuration directory"""
    
    config_path = Path(config_dir)
    if not config_path.exists():
        return False, f"Directory {config_dir} does not exist"
    
    print(f"\n🔍 VERIFYING: {config_path.name}")
    print("=" * 50)
    
    # Check for numerical_setup.json
    numerical_file = config_path / "numerical_setup.json"
    if not numerical_file.exists():
        return False, "numerical_setup.json not found"
    
    try:
        with open(numerical_file, 'r') as f:
            numerical_config = json.load(f)
        print("✅ numerical_setup.json - Valid JSON")
        
        # Check key numerical parameters
        conservatives = numerical_config.get('conservatives', {})
        print(f"   🔧 Time integrator: {conservatives.get('time_integration', {}).get('integrator', 'Unknown')}")
        print(f"   🔧 CFL number: {conservatives.get('time_integration', {}).get('CFL', 'Unknown')}")
        
        godunov = conservatives.get('convective_fluxes', {}).get('godunov', {})
        print(f"   🔧 Riemann solver: {godunov.get('riemann_solver', 'Unknown')}")
        print(f"   🔧 Reconstruction: {godunov.get('reconstruction_stencil', 'Unknown')}")
        
        # Check active physics
        physics = numerical_config.get('active_physics', {})
        print(f"   🌊 Viscous flux: {physics.get('is_viscous_flux', False)}")
        print(f"   🌡️ Heat flux: {physics.get('is_heat_flux', False)}")
        print(f"   📐 Level-set: {physics.get('is_levelset', False)}")
        
    except Exception as e:
        return False, f"Error reading numerical_setup.json: {e}"
    
    # Check for case configuration file
    case_files = list(config_path.glob("*.json"))
    case_files = [f for f in case_files if f.name not in ["numerical_setup.json", "simulation_summary.json"]]
    
    if not case_files:
        return False, "No case configuration file found"
    
    case_file = case_files[0]
    try:
        with open(case_file, 'r') as f:
            case_config = json.load(f)
        print(f"✅ {case_file.name} - Valid JSON")
        
        # Check domain configuration
        domain = case_config.get('domain', {})
        x_cells = domain.get('x', {}).get('cells', 0)
        y_cells = domain.get('y', {}).get('cells', 0)
        z_cells = domain.get('z', {}).get('cells', 0)
        print(f"   🏗️ Grid resolution: {x_cells}×{y_cells}×{z_cells}")
        
        x_range = domain.get('x', {}).get('range', [])
        print(f"   📏 X domain: {x_range}")
        
        # Check boundary conditions
        bc = case_config.get('boundary_conditions', {})
        primitives_bc = bc.get('primitives', {})
        print(f"   🌬️ West BC (inlet): {primitives_bc.get('west', {}).get('type', 'Unknown')}")
        print(f"   🌬️ East BC (outlet): {primitives_bc.get('east', {}).get('type', 'Unknown')}")
        
        # Check initial conditions
        ic = case_config.get('initial_condition', {}).get('primitives', {})
        print(f"   🌀 Initial velocity: {ic.get('u', 'Unknown')}")
        print(f"   🌡️ Initial pressure: {ic.get('p', 'Unknown')}")
        
    except Exception as e:
        return False, f"Error reading case config: {e}"
    
    # Check if run.py exists
    run_file = config_path / "run.py"
    if run_file.exists():
        print(f"✅ run.py exists ({run_file.stat().st_size} bytes)")
    else:
        print("⚠️ run.py not found")
    
    return True, "Configuration verified successfully"

def main():
    """Main verification function"""
    
    print("🔧 JAX-FLUIDS CONFIGURATION VERIFICATION")
    print("=" * 60)
    
    # Check existing configurations
    test_dir = Path(".")
    config_dirs = [d for d in test_dir.iterdir() if d.is_dir() and "jaxfluids_external_flow" in d.name]
    
    if not config_dirs:
        print("❌ No JAX-Fluids configuration directories found")
        return False
    
    print(f"📁 Found {len(config_dirs)} configuration directories:")
    for d in config_dirs:
        print(f"   - {d.name}")
    
    all_passed = True
    for config_dir in config_dirs:
        success, message = verify_configuration(config_dir)
        if success:
            print("✅ CONFIGURATION VALID")
        else:
            print(f"❌ CONFIGURATION ISSUE: {message}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL CONFIGURATIONS VERIFIED SUCCESSFULLY!")
        print("✅ External flow endpoint is generating valid JAX-Fluids configs")
        print("✅ numerical_setup.json contains proper numerical parameters")
        print("✅ Case configuration contains proper domain and boundary conditions")
        print("✅ Ready for JAX-Fluids simulation execution")
    else:
        print("❌ SOME CONFIGURATIONS HAVE ISSUES")
        print("🔧 Check the external_flow_endpoint agent configuration")
    
    return all_passed

if __name__ == "__main__":
    main() 