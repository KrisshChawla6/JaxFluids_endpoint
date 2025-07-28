#!/usr/bin/env python3
"""
Universal Agent Generation Endpoint
Universal interface for generating simulation configs with any mesh file and prompt
"""

import os
import sys
import time
import json
from pathlib import Path
from cfd_parameter_agent import CFDParameterAgent
from wind_tunnel_generator import create_config_with_extracted_markers, WindTunnelOrientation, SU2ConfigGenerator

def generate_universal_simulation_config(user_prompt, mesh_file_path, output_dir=None, api_key=None):
    """
    Universal function to generate simulation configuration for any mesh file and prompt
    
    Args:
        user_prompt (str): Natural language description of the simulation
        mesh_file_path (str): Full path to the mesh file to use
        output_dir (str, optional): Directory to save the configuration. If None, creates in 'simulations/'
        api_key (str, optional): Gemini API key. If None, uses environment variable
    
    Returns:
        dict: {
            'success': bool,
            'simulation_directory': str,
            'config_file': str,
            'mesh_file': str,
            'metadata_file': str,
            'ai_metadata_file': str,
            'extracted_parameters': dict,
            'normalized_parameters': dict,
            'generation_time': float,
            'error': str (if failed)
        }
    """
    
    try:
        # Initialize API key
        if not api_key:
            api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyB2mzctXXTAK8RRc5IHaKJ87b9inm4x9A4')
        
        # Validate mesh file exists
        if not os.path.exists(mesh_file_path):
            return {
                'success': False,
                'error': f'Mesh file not found: {mesh_file_path}'
            }
        
        print(f"🤖 Universal Agent: Generating config for '{user_prompt}'")
        print(f"📁 Using mesh file: {mesh_file_path}")
        
        start_time = time.time()
        
        # Step 1: Initialize AI Agent
        agent = CFDParameterAgent(api_key)
        
        # Step 2: Extract Parameters from Natural Language
        raw_params = agent.extract_parameters(user_prompt)
        print(f"✅ Extracted parameters: {raw_params}")
        
        # Step 3: Validate and Normalize Parameters
        validated_params = agent.validate_and_normalize_parameters(raw_params)
        print(f"✅ Validated parameters: {validated_params}")
        
        # Step 4: Generate unique simulation directory name or use provided output_dir
        if not output_dir:
            # Create a safe filename from the prompt
            safe_name = "".join(c for c in user_prompt.lower().replace(" ", "_") if c.isalnum() or c in "_-")[:50]
            timestamp = int(time.time())
            sim_dir_name = f"sim_{timestamp}_{safe_name}"
            output_dir = os.path.join("simulations", sim_dir_name)
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            temp_dir_created = True
        else:
            # Use existing directory (like project configs directory)
            temp_dir_created = False
        
        # Step 5: Copy mesh file to simulation directory with correct name
        mesh_filename = os.path.basename(mesh_file_path)
        
        # Replace spaces with underscores in mesh filename for SU2 compatibility
        if ' ' in mesh_filename:
            safe_mesh_filename = mesh_filename.replace(' ', '_')
            print(f"⚠️  Mesh filename contains spaces, renaming: '{mesh_filename}' -> '{safe_mesh_filename}'")
        else:
            safe_mesh_filename = mesh_filename
        
        # Only copy mesh if this is a temporary directory
        # If using project directory, assume mesh is already in proper location
        if temp_dir_created:
            sim_mesh_path = os.path.join(output_dir, safe_mesh_filename)
            import shutil
            shutil.copy2(mesh_file_path, sim_mesh_path)
            print(f"✅ Copied mesh file to: {sim_mesh_path}")
        else:
            # For project directories, use the existing mesh file path
            sim_mesh_path = mesh_file_path
            print(f"✅ Using existing mesh file: {sim_mesh_path}")
        
        # Step 6: Generate configuration using the actual mesh file
        config_filename = f"config_{int(time.time())}.cfg"
        config_file_path = os.path.join(output_dir, config_filename)
        
        # Convert orientation string to enum
        orientation_map = {
            "+X": WindTunnelOrientation.POSITIVE_X,
            "-X": WindTunnelOrientation.NEGATIVE_X,
            "+Y": WindTunnelOrientation.POSITIVE_Y,
            "-Y": WindTunnelOrientation.NEGATIVE_Y,
            "+Z": WindTunnelOrientation.POSITIVE_Z,
            "-Z": WindTunnelOrientation.NEGATIVE_Z
        }
        orientation = orientation_map.get(validated_params['wind_tunnel_orientation'], WindTunnelOrientation.POSITIVE_X)
        
        # Generate configuration with the actual mesh file
        config = create_config_with_extracted_markers(
            mesh_file_path=sim_mesh_path,  # Use the copied mesh file
            solver_type=validated_params['solver_type'],
            mach_number=validated_params['mach_number'],
            angle_of_attack=validated_params['angle_of_attack'],
            reynolds_number=validated_params['reynolds_number'],
            max_iterations=validated_params['max_iterations'],
            wind_tunnel_orientation=orientation,
            turbulence_model=validated_params['turbulence_model']
        )
        
        # Step 7: Write configuration to file using SU2ConfigGenerator
        generator = SU2ConfigGenerator()
        config_content = generator.generate_config(config)
        
        # Update the mesh filename in the config to use the relative path
        # and add quotes if the filename contains spaces
        if ' ' in safe_mesh_filename:
            quoted_mesh_filename = f'"{safe_mesh_filename}"'
        else:
            quoted_mesh_filename = safe_mesh_filename
        
        config_content = config_content.replace(sim_mesh_path, quoted_mesh_filename)
        
        with open(config_file_path, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Generated config file: {config_file_path}")
        
        # Step 8: Save metadata
        metadata = {
            'user_prompt': user_prompt,
            'original_mesh_file': mesh_file_path,
            'mesh_file': safe_mesh_filename,
            'config_file': config_filename,
            'generated_at': time.time(),
            'generation_time': time.time() - start_time,
            'solver_type': validated_params['solver_type'],
            'mach_number': validated_params['mach_number'],
            'angle_of_attack': validated_params['angle_of_attack'],
            'reynolds_number': validated_params['reynolds_number'],
            'wind_tunnel_orientation': validated_params['wind_tunnel_orientation']
        }
        
        metadata_file = os.path.join(output_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Step 9: Save AI metadata
        ai_metadata = {
            'user_prompt': user_prompt,
            'extracted_parameters': raw_params,
            'normalized_parameters': validated_params,
            'generated_by': 'Universal CFD Agent',
            'model': 'gemini-2.0-flash-exp',
            'mesh_file': safe_mesh_filename,
            'original_mesh_path': mesh_file_path,
            'generation_timestamp': time.time()
        }
        
        ai_metadata_file = os.path.join(output_dir, 'ai_metadata.json')
        with open(ai_metadata_file, 'w') as f:
            json.dump(ai_metadata, f, indent=2)
        
        generation_time = time.time() - start_time
        
        print(f"✅ Universal config generation completed in {generation_time:.2f}s")
        print(f"📁 Simulation directory: {output_dir}")
        
        return {
            'success': True,
            'simulation_directory': output_dir,
            'config_file': config_file_path,
            'mesh_file': sim_mesh_path,
            'metadata_file': metadata_file,
            'ai_metadata_file': ai_metadata_file,
            'extracted_parameters': raw_params,
            'normalized_parameters': validated_params,
            'generation_time': generation_time,
            'config_filename': config_filename,
            'mesh_filename': safe_mesh_filename
        }
        
    except Exception as e:
        print(f"❌ Universal config generation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def test_universal_endpoint():
    """Test function for the universal endpoint"""
    
    print("🧪 Testing Universal Agent Generation Endpoint")
    print("=" * 60)
    
    # Example test
    test_prompt = "Analyze airfoil at 8 degrees angle of attack with Mach 0.15"
    test_mesh = "test_mesh.su2"  # Would be replaced with actual mesh file
    
    result = generate_universal_simulation_config(
        user_prompt=test_prompt,
        mesh_file_path=test_mesh
    )
    
    if result['success']:
        print("✅ Universal endpoint test passed!")
        print(f"📁 Generated in: {result['simulation_directory']}")
        print(f"⏱️ Generation time: {result['generation_time']:.2f}s")
    else:
        print(f"❌ Universal endpoint test failed: {result['error']}")
    
    return result

if __name__ == "__main__":
    test_universal_endpoint() 