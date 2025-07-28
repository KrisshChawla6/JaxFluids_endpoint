#!/usr/bin/env python3
"""
JAX-Fluids Boundary Integrator

Uses existing immersed_boundary_endpoint_final for SDF creation and focuses on
adding virtual inlet/outlet boundary condition masks for JAX-Fluids.

This integrates our circular virtual faces with JAX-Fluids boundary condition system.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any
import sys
import os

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "immersed_boundary_endpoint_final"))

try:
    import pyvista as pv
    import meshio
    print("✅ Required libraries available")
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    exit(1)

# Import the existing SDF functions - if this fails, we'll use subprocess approach
SDF_AVAILABLE = False
try:
    sys.path.append(r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\immersed_boundary_endpoint_final")
    from immersed_boundary_sdf import main as create_sdf, parse_gmsh_mesh
    SDF_AVAILABLE = True
    print("✅ Imported existing SDF functionality")
except ImportError as e:
    print(f"⚠️  Could not import SDF functions directly: {e}")
    print("   Will use subprocess to call SDF endpoint")
    SDF_AVAILABLE = False

@dataclass
class VirtualFaceSpec:
    """Specification for a virtual face (inlet or outlet)"""
    face_type: str                    # "inlet" or "outlet"
    center: np.ndarray               # Shape: (3,) [X, Y, Z]
    radius: float                    # Circle radius
    x_position: float                # Axial position
    normal_vector: np.ndarray        # Shape: (3,) unit normal
    area: float                      # Total face area

@dataclass 
class JAXFluidsBoundaryConfig:
    """JAX-Fluids boundary configuration with virtual faces"""
    inlet_spec: VirtualFaceSpec
    outlet_spec: VirtualFaceSpec
    domain_bounds: np.ndarray        # [xmin, ymin, zmin, xmax, ymax, zmax]
    sdf_files: Dict[str, str]        # SDF file paths from existing endpoint
    boundary_masks: Dict[str, np.ndarray]  # Inlet/outlet masks
    jax_fluids_config: Dict[str, Any]      # Complete JAX-Fluids configuration

class VirtualFaceDetector:
    """Detect virtual faces using our proven circular edge detection"""
    
    def detect_virtual_faces(self, mesh_file) -> Tuple[Optional[VirtualFaceSpec], Optional[VirtualFaceSpec]]:
        """Detect inlet and outlet virtual faces from mesh"""
        
        print("🔍 DETECTING VIRTUAL FACES")
        print("=" * 50)
        
        # Load mesh and extract surface
        mesh = meshio.read(mesh_file)
        pv_mesh = pv.from_meshio(mesh)
        surface = pv_mesh.extract_surface()
        
        # Extract boundary edges (using proven method)
        boundary_edges = surface.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=True,
            feature_edges=True,
            manifold_edges=False
        )
        
        if boundary_edges.n_points == 0:
            print("❌ No boundary edges found")
            return None, None
        
        edge_points = boundary_edges.points
        print(f"   Found {len(edge_points)} boundary edge points")
        
        # Cluster by X position to find inlet/outlet
        x_coords = edge_points[:, 0]
        x_min, x_max = x_coords.min(), x_coords.max()
        x_range = x_max - x_min
        
        # Inlet (low X) and outlet (high X)
        inlet_mask = x_coords < (x_min + 0.1 * x_range)
        outlet_mask = x_coords > (x_max - 0.1 * x_range)
        
        inlet_points = edge_points[inlet_mask]
        outlet_points = edge_points[outlet_mask]
        
        print(f"   Inlet: {len(inlet_points)} points at X≈{inlet_points[:, 0].mean():.1f}")
        print(f"   Outlet: {len(outlet_points)} points at X≈{outlet_points[:, 0].mean():.1f}")
        
        # Fit circles to each region
        inlet_spec = self._fit_circular_face(inlet_points, "inlet")
        outlet_spec = self._fit_circular_face(outlet_points, "outlet")
        
        return inlet_spec, outlet_spec
    
    def _fit_circular_face(self, boundary_points, face_type):
        """Fit circle to boundary points and create face specification"""
        
        if len(boundary_points) < 10:
            print(f"❌ Not enough points for {face_type}")
            return None
        
        # Average X position
        x_pos = boundary_points[:, 0].mean()
        
        # Project to Y-Z plane and fit circle
        yz_points = boundary_points[:, 1:]
        center_y = yz_points[:, 0].mean()
        center_z = yz_points[:, 1].mean()
        
        # Calculate radius as average distance from center
        distances = np.sqrt((yz_points[:, 0] - center_y)**2 + (yz_points[:, 1] - center_z)**2)
        radius = distances.mean()
        
        # Create specification
        center = np.array([x_pos, center_y, center_z])
        normal = np.array([1.0, 0.0, 0.0])  # Flow direction
        area = np.pi * radius**2
        
        spec = VirtualFaceSpec(
            face_type=face_type,
            center=center,
            radius=radius,
            x_position=x_pos,
            normal_vector=normal,
            area=area
        )
        
        print(f"   {face_type.title()}: center=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}), radius={radius:.1f}")
        
        return spec

class JAXFluidsBoundaryCreator:
    """Create JAX-Fluids boundary conditions with virtual faces"""
    
    def __init__(self, sdf_endpoint_path: str):
        self.sdf_endpoint_path = Path(sdf_endpoint_path)
    
    def create_boundary_masks(self, inlet_spec: VirtualFaceSpec, outlet_spec: VirtualFaceSpec, 
                            grid_coords: np.ndarray, grid_shape: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
        """Create 3D boundary condition masks for inlet/outlet faces"""
        
        print("🎯 Creating boundary condition masks...")
        
        # Initialize masks
        inlet_mask = np.zeros(grid_shape, dtype=bool)
        outlet_mask = np.zeros(grid_shape, dtype=bool)
        
        # Reshape grid coordinates
        coords = grid_coords.reshape(-1, 3)
        
        # Create inlet mask (circular region at inlet X position)
        inlet_center = inlet_spec.center
        inlet_radius = inlet_spec.radius
        
        # Find points near inlet X position and within circular face
        x_tolerance = (grid_coords[1,0,0,0] - grid_coords[0,0,0,0]) * 2  # 2 grid spacings
        inlet_x_mask = np.abs(coords[:, 0] - inlet_center[0]) < x_tolerance
        
        for i, point in enumerate(coords):
            if inlet_x_mask[i]:
                yz_distance = np.sqrt((point[1] - inlet_center[1])**2 + (point[2] - inlet_center[2])**2)
                if yz_distance <= inlet_radius:
                    # Convert flat index to 3D index
                    idx_3d = np.unravel_index(i, grid_shape)
                    inlet_mask[idx_3d] = True
        
        # Create outlet mask (similar process)
        outlet_center = outlet_spec.center
        outlet_radius = outlet_spec.radius
        
        outlet_x_mask = np.abs(coords[:, 0] - outlet_center[0]) < x_tolerance
        
        for i, point in enumerate(coords):
            if outlet_x_mask[i]:
                yz_distance = np.sqrt((point[1] - outlet_center[1])**2 + (point[2] - outlet_center[2])**2)
                if yz_distance <= outlet_radius:
                    idx_3d = np.unravel_index(i, grid_shape)
                    outlet_mask[idx_3d] = True
        
        print(f"   Inlet mask: {inlet_mask.sum()} grid points")
        print(f"   Outlet mask: {outlet_mask.sum()} grid points")
        
        return {
            'inlet_mask': inlet_mask,
            'outlet_mask': outlet_mask
        }
    
    def create_jax_fluids_config(self, inlet_spec: VirtualFaceSpec, outlet_spec: VirtualFaceSpec,
                               sdf_files: Dict, boundary_masks: Dict, 
                               flow_conditions: Optional[Dict] = None) -> Dict:
        """Create complete JAX-Fluids configuration with virtual boundary conditions"""
        
        print("🚀 Creating JAX-Fluids configuration...")
        
        # Default flow conditions for rocket nozzle
        if flow_conditions is None:
            flow_conditions = {
                'inlet_pressure': 6.9e6,      # 6.9 MPa chamber pressure
                'inlet_temperature': 3580.0,   # 3580 K chamber temperature
                'inlet_velocity': 100.0,       # m/s initial estimate
                'outlet_pressure': 101325.0,   # Atmospheric pressure
                'gamma': 1.3,                  # Heat capacity ratio
                'gas_constant': 287.0          # J/(kg·K)
            }
        
        # Extract domain info from SDF metadata
        with open(sdf_files['metadata'], 'r') as f:
            sdf_metadata = json.load(f)
        
        domain_bounds = sdf_metadata['domain_bounds']  # [xmin, ymin, zmin, xmax, ymax, zmax]
        resolution = sdf_metadata['resolution']        # [nx, ny, nz]
        
        # Create JAX-Fluids configuration
        config = {
            'case_name': 'rocket_nozzle_internal_flow_with_virtual_faces',
            
            # Domain configuration
            'domain': {
                'x': {'cells': resolution[0], 'range': [domain_bounds[0], domain_bounds[3]]},
                'y': {'cells': resolution[1], 'range': [domain_bounds[1], domain_bounds[4]]}, 
                'z': {'cells': resolution[2], 'range': [domain_bounds[2], domain_bounds[5]]}
            },
            
            # Standard boundary conditions for domain faces (walls/symmetry)
            'boundary_conditions': {
                'domain_faces': {
                    'x_min': 'WALL',     # Will be overridden by virtual inlet
                    'x_max': 'WALL',     # Will be overridden by virtual outlet
                    'y_min': 'SYMMETRY',
                    'y_max': 'SYMMETRY',
                    'z_min': 'SYMMETRY',
                    'z_max': 'SYMMETRY'
                }
            },
            
            # Virtual face boundary conditions (our innovation!)
            'virtual_boundary_conditions': {
                'inlet': {
                    'type': 'DIRICHLET',
                    'location': {
                        'center': inlet_spec.center.tolist(),
                        'radius': float(inlet_spec.radius),
                        'normal': inlet_spec.normal_vector.tolist(),
                        'x_position': float(inlet_spec.x_position)
                    },
                    'conditions': {
                        'pressure': flow_conditions['inlet_pressure'],
                        'temperature': flow_conditions['inlet_temperature'],
                        'velocity': [flow_conditions['inlet_velocity'], 0.0, 0.0]
                    },
                    'mask_file': 'inlet_mask.npy'  # Will be saved separately
                },
                'outlet': {
                    'type': 'NEUMANN',
                    'location': {
                        'center': outlet_spec.center.tolist(),
                        'radius': float(outlet_spec.radius),
                        'normal': outlet_spec.normal_vector.tolist(),
                        'x_position': float(outlet_spec.x_position)
                    },
                    'conditions': {
                        'pressure': flow_conditions['outlet_pressure'],
                        'gradient': 'ZERO_GRADIENT'
                    },
                    'mask_file': 'outlet_mask.npy'  # Will be saved separately
                }
            },
            
            # Immersed boundary (walls) using existing SDF
            'levelset': {
                'model': 'FLUID_SOLID_INTERFACE',
                'sdf_file': sdf_files['sdf_matrix'],
                'method': 'LEVEL_SET'
            },
            
            # Physical models
            'physics': {
                'viscous': True,
                'thermal': True,
                'compressible': True
            },
            
            # Numerical methods
            'numerical_setup': {
                'spatial_reconstruction': 'WENO5',
                'riemann_solver': 'HLLC',
                'time_integration': 'RK3'
            },
            
            # Initial conditions
            'initial_conditions': {
                'pressure': flow_conditions['outlet_pressure'],
                'temperature': 300.0,  # Ambient temperature initially
                'velocity': [0.0, 0.0, 0.0]
            },
            
            # Material properties
            'material': {
                'gamma': flow_conditions['gamma'],
                'gas_constant': flow_conditions['gas_constant']
            },
            
            # File references
            'files': {
                'sdf_data': sdf_files,
                'inlet_mask': 'inlet_mask.npy',
                'outlet_mask': 'outlet_mask.npy'
            },
            
            # Metadata
            'metadata': {
                'created': datetime.now().isoformat(),
                'description': 'Rocket nozzle internal flow with virtual inlet/outlet faces',
                'inlet_area': float(inlet_spec.area),
                'outlet_area': float(outlet_spec.area),
                'expansion_ratio': float(outlet_spec.area / inlet_spec.area)
            }
        }
        
        return config
    
    def process_mesh_with_virtual_faces(self, mesh_file: str, 
                                      domain_bounds: Tuple[float, float, float, float, float, float],
                                      resolution: Tuple[int, int, int],
                                      output_dir: str = "jax_fluids_virtual_bc") -> JAXFluidsBoundaryConfig:
        """Complete processing pipeline: detect faces, create SDF, generate boundary conditions"""
        
        print("🚀 JAX-FLUIDS VIRTUAL BOUNDARY PROCESSOR")
        print("=" * 70)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Step 1: Detect virtual faces
        detector = VirtualFaceDetector()
        inlet_spec, outlet_spec = detector.detect_virtual_faces(mesh_file)
        
        if inlet_spec is None or outlet_spec is None:
            raise ValueError("Could not detect inlet/outlet virtual faces")
        
        # Step 2: Create SDF using existing endpoint
        print("\n📐 Creating SDF using existing immersed boundary endpoint...")
        
        # Call existing SDF creation (this creates the wall geometry SDF)
        domain_str = f"({domain_bounds[0]},{domain_bounds[1]},{domain_bounds[2]},{domain_bounds[3]},{domain_bounds[4]},{domain_bounds[5]})"
        resolution_str = f"({resolution[0]},{resolution[1]},{resolution[2]})"
        
        # Use subprocess to call existing SDF endpoint
        import subprocess
        sdf_script = self.sdf_endpoint_path / "immersed_boundary_sdf.py"
        sdf_output = output_path / "sdf_files"
        
        cmd = [
            "python", str(sdf_script),
            mesh_file,
            "--domain", domain_str,
            "--resolution", resolution_str,
            "--output-dir", str(sdf_output)
        ]
        
        print(f"   Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.sdf_endpoint_path))
        
        if result.returncode != 0:
            print(f"❌ SDF creation failed:")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            raise RuntimeError("SDF creation failed")
        
        print(f"   ✅ SDF creation completed")
        print(f"   Output: {result.stdout}")
        
        # Find the latest SDF files
        sdf_runs = list(sdf_output.glob("*"))
        if not sdf_runs:
            raise RuntimeError("No SDF files generated")
        
        latest_run = max(sdf_runs, key=lambda p: p.stat().st_ctime)
        
        # Look for SDF files with correct naming
        mesh_base_name = Path(mesh_file).stem.replace(" ", "%20")  # Handle spaces in filename
        
        # Try different possible file names
        possible_names = [
            f"{mesh_base_name}_sdf_matrix.npy",
            f"Rocket Engine_sdf_matrix.npy", 
            f"Rocket_Engine_sdf_matrix.npy",
            f"rocket_engine_sdf_matrix.npy"
        ]
        
        sdf_matrix_file = None
        metadata_file = None
        
        for name in possible_names:
            sdf_path = latest_run / name
            if sdf_path.exists():
                sdf_matrix_file = str(sdf_path)
                # Find corresponding metadata file
                meta_name = name.replace("_sdf_matrix.npy", "_metadata.json")
                meta_path = latest_run / meta_name
                if meta_path.exists():
                    metadata_file = str(meta_path)
                break
        
        if sdf_matrix_file is None:
            # List all files to help debug
            files_in_run = list(latest_run.glob("*"))
            print(f"   Available files in {latest_run}:")
            for f in files_in_run:
                print(f"     {f.name}")
            raise RuntimeError("Could not find SDF matrix file")
        
        sdf_files = {
            'sdf_matrix': sdf_matrix_file,
            'metadata': metadata_file
        }
        
        print(f"   ✅ SDF files found:")
        print(f"     Matrix: {sdf_matrix_file}")
        print(f"     Metadata: {metadata_file}")
        
        # Step 3: Create grid coordinates for boundary masks
        print("\n🎯 Creating boundary condition masks...")
        
        x = np.linspace(domain_bounds[0], domain_bounds[3], resolution[0])
        y = np.linspace(domain_bounds[1], domain_bounds[4], resolution[1])
        z = np.linspace(domain_bounds[2], domain_bounds[5], resolution[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_coords = np.stack([X, Y, Z], axis=-1)
        
        # Step 4: Create boundary masks
        boundary_masks = self.create_boundary_masks(inlet_spec, outlet_spec, grid_coords, resolution)
        
        # Step 5: Save boundary masks
        inlet_mask_file = output_path / "inlet_mask.npy"
        outlet_mask_file = output_path / "outlet_mask.npy"
        
        np.save(inlet_mask_file, boundary_masks['inlet_mask'])
        np.save(outlet_mask_file, boundary_masks['outlet_mask'])
        
        print(f"   💾 Masks saved: {inlet_mask_file}, {outlet_mask_file}")
        
        # Step 6: Create complete JAX-Fluids configuration
        jax_config = self.create_jax_fluids_config(inlet_spec, outlet_spec, sdf_files, boundary_masks)
        
        # Step 7: Save JAX-Fluids configuration
        config_file = output_path / "jax_fluids_config.json"
        with open(config_file, 'w') as f:
            json.dump(jax_config, f, indent=2)
        
        print(f"   🚀 JAX-Fluids config saved: {config_file}")
        
        # Return complete configuration
        result = JAXFluidsBoundaryConfig(
            inlet_spec=inlet_spec,
            outlet_spec=outlet_spec,
            domain_bounds=np.array(domain_bounds),
            sdf_files=sdf_files,
            boundary_masks=boundary_masks,
            jax_fluids_config=jax_config
        )
        
        print("\n" + "=" * 70)
        print("🎉 JAX-FLUIDS VIRTUAL BOUNDARY CONDITIONS CREATED!")
        print("=" * 70)
        print(f"✅ Inlet: R={inlet_spec.radius:.1f} at X={inlet_spec.x_position:.1f}")
        print(f"✅ Outlet: R={outlet_spec.radius:.1f} at X={outlet_spec.x_position:.1f}")
        print(f"✅ Expansion ratio: {outlet_spec.area/inlet_spec.area:.2f}")
        print(f"✅ Ready for JAX-Fluids simulation!")
        print("=" * 70)
        
        return result

def main():
    """Main function for testing"""
    
    # Configuration
    mesh_file = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\mesh\Rocket Engine.msh"
    sdf_endpoint_path = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\immersed_boundary_endpoint_final"
    
    # Domain bounds (can be adjusted based on mesh)
    domain_bounds = (-200, -800, -800, 1800, 800, 800)
    resolution = (128, 64, 64)  # Reasonable resolution for testing
    
    try:
        # Create boundary processor
        processor = JAXFluidsBoundaryCreator(sdf_endpoint_path)
        
        # Process mesh and create virtual boundary conditions
        result = processor.process_mesh_with_virtual_faces(
            mesh_file, 
            domain_bounds, 
            resolution,
            output_dir="jax_fluids_virtual_bc"
        )
        
        print("\n🚀 Success! JAX-Fluids configuration with virtual faces ready.")
        print(f"   Config file: jax_fluids_virtual_bc/jax_fluids_config.json")
        print(f"   Use with JAX-Fluids BoundaryCondition.fill_boundary_levelset()")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 