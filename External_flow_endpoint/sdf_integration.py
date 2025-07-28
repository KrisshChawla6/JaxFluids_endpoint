#!/usr/bin/env python3
"""
SDF Integration Module
Provides programmatic interface to the immersed boundary SDF generation
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

# Add the immersed_boundary_endpoint_final to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'immersed_boundary_endpoint_final'))

try:
    from immersed_boundary_sdf import parse_gmsh_mesh, compute_sdf, store_sdf_data
    SDF_MODULE_AVAILABLE = True
except ImportError:
    SDF_MODULE_AVAILABLE = False

def find_latest_sdf_file(sdf_directory: str) -> Tuple[str, Dict[str, Any]]:
    """
    Find the latest timestamped SDF file in the given directory
    
    Args:
        sdf_directory: Directory to search for SDF files
        
    Returns:
        Tuple of (sdf_file_path, metadata_dict)
        
    Raises:
        FileNotFoundError: If no timestamped SDF directories are found
    """
    sdf_path = Path(sdf_directory)
    
    if not sdf_path.exists():
        raise FileNotFoundError(f"SDF directory not found: {sdf_directory}")
    
    # Look for timestamped directories (format: YYYYMMDD_HHMMSS)
    timestamped_dirs = []
    for item in sdf_path.iterdir():
        if item.is_dir() and len(item.name) == 15 and item.name[8] == '_':
            try:
                # Validate timestamp format
                time.strptime(item.name, "%Y%m%d_%H%M%S")
                timestamped_dirs.append(item)
            except ValueError:
                continue
    
    if not timestamped_dirs:
        raise FileNotFoundError(f"No timestamped SDF run directories found in {sdf_directory}")
    
    # Get the latest directory
    latest_run = max(timestamped_dirs, key=lambda x: x.name)
    
    # Look for SDF matrix file (best for JAX-Fluids)
    sdf_matrix_files = list(latest_run.glob("*_sdf_matrix.npy"))
    
    if not sdf_matrix_files:
        raise FileNotFoundError(f"No SDF matrix files found in {latest_run}")
    
    sdf_file = str(sdf_matrix_files[0])
    
    # Load metadata if available
    metadata_files = list(latest_run.glob("*_metadata.json"))
    metadata = {}
    
    if metadata_files:
        with open(metadata_files[0], 'r') as f:
            metadata = json.load(f)
    
    return sdf_file, metadata

def generate_sdf_for_domain(
    mesh_file: str,
    domain_bounds: Tuple[float, float, float, float, float, float],
    resolution: Tuple[int, int, int],
    output_directory: str,
    mesh_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate SDF for the given mesh and domain parameters
    
    Args:
        mesh_file: Path to the Gmsh mesh file
        domain_bounds: (xmin, ymin, zmin, xmax, ymax, zmax)
        resolution: (nx, ny, nz) grid resolution
        output_directory: Directory to store SDF files
        mesh_name: Optional name for the mesh (defaults to filename stem)
        
    Returns:
        Dictionary with SDF generation results
    """
    
    if not SDF_MODULE_AVAILABLE:
        # Fallback to subprocess call
        return _generate_sdf_subprocess(mesh_file, domain_bounds, resolution, output_directory, mesh_name)
    
    if mesh_name is None:
        mesh_name = Path(mesh_file).stem
    
    print(f"🔧 Generating SDF for {mesh_name}")
    print(f"📐 Domain: {domain_bounds}")
    print(f"🔢 Resolution: {resolution}")
    
    try:
        start_time = time.time()
        
        # Parse mesh
        print("📁 Parsing mesh...")
        boundary_triangles = parse_gmsh_mesh(mesh_file)
        
        # Compute SDF
        print("🧮 Computing SDF...")
        X, Y, Z, sdf_grid = compute_sdf(boundary_triangles, domain_bounds, resolution)
        
        # Store SDF data
        print("💾 Storing SDF data...")
        stored_files = store_sdf_data(X, Y, Z, sdf_grid, output_directory, mesh_name)
        
        generation_time = time.time() - start_time
        
        print(f"✅ SDF generation completed in {generation_time:.2f}s")
        print(f"📊 SDF range: [{sdf_grid.min():.3f}, {sdf_grid.max():.3f}]")
        
        return {
            'success': True,
            'sdf_file_path': stored_files['sdf_file'],
            'metadata_file': stored_files['metadata_file'],
            'run_directory': stored_files['run_path'],
            'generation_time': generation_time,
            'sdf_range': [float(sdf_grid.min()), float(sdf_grid.max())],
            'resolution': resolution,
            'domain_bounds': domain_bounds,
            'mesh_name': mesh_name
        }
        
    except Exception as e:
        print(f"❌ SDF generation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'mesh_file': mesh_file,
            'domain_bounds': domain_bounds,
            'resolution': resolution
        }

def _generate_sdf_subprocess(
    mesh_file: str,
    domain_bounds: Tuple[float, float, float, float, float, float],
    resolution: Tuple[int, int, int],
    output_directory: str,
    mesh_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fallback method using subprocess to call the SDF script
    """
    
    print("🔄 Using subprocess fallback for SDF generation...")
    
    # Construct the command
    sdf_script = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        'immersed_boundary_endpoint_final', 
        'immersed_boundary_sdf.py'
    )
    
    domain_str = f"({domain_bounds[0]},{domain_bounds[1]},{domain_bounds[2]},{domain_bounds[3]},{domain_bounds[4]},{domain_bounds[5]})"
    resolution_str = f"({resolution[0]},{resolution[1]},{resolution[2]})"
    
    cmd = [
        sys.executable, 
        sdf_script,
        mesh_file,
        '--domain', domain_str,
        '--resolution', resolution_str,
        '--output-dir', output_directory
    ]
    
    try:
        start_time = time.time()
        
        print(f"🚀 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        generation_time = time.time() - start_time
        
        if result.returncode == 0:
            print("✅ SDF generation completed via subprocess")
            
            # Try to find the generated files
            if mesh_name is None:
                mesh_name = Path(mesh_file).stem
                
            # Look for the timestamped directory
            sdf_base_dir = Path(output_directory)
            run_dirs = [d for d in sdf_base_dir.iterdir() if d.is_dir() and d.name.startswith('20')]
            
            if run_dirs:
                latest_run = max(run_dirs, key=lambda d: d.stat().st_mtime)
                sdf_file = latest_run / f"{mesh_name}_sdf.npy"
                metadata_file = latest_run / f"{mesh_name}_metadata.json"
                
                return {
                    'success': True,
                    'sdf_file_path': str(sdf_file) if sdf_file.exists() else None,
                    'metadata_file': str(metadata_file) if metadata_file.exists() else None,
                    'run_directory': str(latest_run),
                    'generation_time': generation_time,
                    'subprocess_output': result.stdout,
                    'mesh_name': mesh_name
                }
            else:
                return {
                    'success': True,
                    'message': 'SDF generated but files not found in expected location',
                    'subprocess_output': result.stdout,
                    'generation_time': generation_time
                }
        else:
            print(f"❌ SDF generation failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            
            return {
                'success': False,
                'error': f"Subprocess failed with return code {result.returncode}",
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'SDF generation timed out (5 minutes)',
            'timeout': True
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'exception_type': type(e).__name__
        }

def extract_domain_from_case_setup(case_setup: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
    """
    Extract domain bounds from JAX-Fluids case setup
    
    Returns:
        (xmin, ymin, zmin, xmax, ymax, zmax)
    """
    
    domain = case_setup.get('domain', {})
    
    # Extract domain bounds
    x_range = domain.get('x', {}).get('range', [-5.0, 5.0])
    y_range = domain.get('y', {}).get('range', [-5.0, 5.0])
    z_range = domain.get('z', {}).get('range', [-5.0, 5.0])
    
    return (x_range[0], y_range[0], z_range[0], x_range[1], y_range[1], z_range[1])

def extract_resolution_from_case_setup(case_setup: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Extract grid resolution from JAX-Fluids case setup
    
    Returns:
        (nx, ny, nz)
    """
    
    domain = case_setup.get('domain', {})
    
    # Extract resolution
    nx = domain.get('x', {}).get('cells', 100)
    ny = domain.get('y', {}).get('cells', 100)
    nz = domain.get('z', {}).get('cells', 100)
    
    return (nx, ny, nz)

def integrate_sdf_with_case_setup(
    mesh_file: str,
    case_setup: Dict[str, Any],
    output_directory: str,
    mesh_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate SDF using parameters from JAX-Fluids case setup
    
    Args:
        mesh_file: Path to the mesh file
        case_setup: JAX-Fluids case setup dictionary
        output_directory: Directory to store SDF files
        mesh_name: Optional mesh name
        
    Returns:
        SDF generation results with integration info
    """
    
    print("🔗 Integrating SDF generation with JAX-Fluids case setup...")
    
    # Extract domain parameters from case setup
    domain_bounds = extract_domain_from_case_setup(case_setup)
    resolution = extract_resolution_from_case_setup(case_setup)
    
    print(f"📊 Extracted domain bounds: {domain_bounds}")
    print(f"🔢 Extracted resolution: {resolution}")
    
    # Generate SDF
    sdf_result = generate_sdf_for_domain(
        mesh_file=mesh_file,
        domain_bounds=domain_bounds,
        resolution=resolution,
        output_directory=output_directory,
        mesh_name=mesh_name
    )
    
    # Add integration metadata
    sdf_result['integrated_from_case_setup'] = True
    sdf_result['case_domain'] = case_setup.get('domain', {})
    
    return sdf_result 