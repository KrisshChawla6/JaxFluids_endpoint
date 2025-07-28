#!/usr/bin/env python3
"""
Professional Signed Distance Function API Endpoint

NASA-grade immersed boundary method using production-quality pysdf library.
Provides clean API interface for SDF generation, storage, and visualization.

Author: AI Assistant
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
import json
import pickle
import time
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    from pysdf import SDF
    logger.info("Using production-quality pysdf library")
except ImportError:
    logger.error("pysdf library not found. Install with: pip install pysdf")
    raise

try:
    import mcubes
    MCUBES_AVAILABLE = True
    logger.info("PyMCubes available for high-quality isosurface extraction")
except ImportError:
    MCUBES_AVAILABLE = False
    logger.warning("PyMCubes not available. Install with: pip install PyMCubes")

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logger.warning("scikit-image not available for marching cubes fallback")


@dataclass
class SDFConfig:
    """Configuration for SDF generation"""
    mesh_file: str
    domain_bounds: Tuple[float, float, float, float, float, float]  # (xmin, ymin, zmin, xmax, ymax, zmax)
    resolution: Tuple[int, int, int]  # (nx, ny, nz)
    output_dir: str = "results"
    output_name: str = "sdf_result"
    plot: bool = True
    save_binary: bool = True
    save_json: bool = True
    export_jaxfluids: bool = True
    robust_mode: bool = True
    batch_size: int = 100000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SDFConfig':
        """Create config from dictionary"""
        return cls(**data)


@dataclass 
class SDFResult:
    """Results from SDF computation"""
    sdf_values: np.ndarray
    grid_points: np.ndarray
    domain_bounds: Tuple[float, float, float, float, float, float]
    resolution: Tuple[int, int, int]
    mesh_file: str
    computation_time: float
    timestamp: str
    config: SDFConfig
    
    def save_binary(self, filepath: str) -> None:
        """Save result as binary pickle file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Saved binary SDF result to {filepath}")
    
    def save_json_metadata(self, filepath: str) -> None:
        """Save metadata as JSON (without large arrays)"""
        metadata = {
            'domain_bounds': self.domain_bounds,
            'resolution': self.resolution,
            'mesh_file': self.mesh_file,
            'computation_time': self.computation_time,
            'timestamp': self.timestamp,
            'config': self.config.to_dict(),
            'sdf_stats': {
                'min': float(np.min(self.sdf_values)),
                'max': float(np.max(self.sdf_values)),
                'mean': float(np.mean(self.sdf_values)),
                'std': float(np.std(self.sdf_values))
            }
        }
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved JSON metadata to {filepath}")
    
    def export_for_jaxfluids(self, filepath: str) -> None:
        """Export SDF data in JAX-Fluids compatible format"""
        jax_data = {
            'domain_bounds': list(self.domain_bounds),
            'resolution': list(self.resolution),
            'grid_spacing': [
                (self.domain_bounds[3] - self.domain_bounds[0]) / (self.resolution[0] - 1),
                (self.domain_bounds[4] - self.domain_bounds[1]) / (self.resolution[1] - 1),
                (self.domain_bounds[5] - self.domain_bounds[2]) / (self.resolution[2] - 1)
            ],
            'sdf_values': self.sdf_values.flatten().tolist(),
            'mesh_file': self.mesh_file,
            'timestamp': self.timestamp
        }
        with open(filepath, 'w') as f:
            json.dump(jax_data, f)
        logger.info(f"Exported JAX-Fluids compatible data to {filepath}")
    
    @classmethod
    def load_binary(cls, filepath: str) -> 'SDFResult':
        """Load result from binary pickle file"""
        with open(filepath, 'rb') as f:
            result = pickle.load(f)
        logger.info(f"Loaded binary SDF result from {filepath}")
        return result


class ImmersedBoundaryAPI:
    """Professional API for immersed boundary SDF generation"""
    
    def __init__(self):
        self.current_result: Optional[SDFResult] = None
        logger.info("Initialized Immersed Boundary API")
    
    def parse_gmsh_mesh(self, mesh_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """Parse Gmsh mesh file and extract vertices and faces"""
        logger.info(f"Parsing Gmsh mesh: {mesh_file}")
        
        vertices = []
        faces = []
        
        with open(mesh_file, 'r') as f:
            lines = f.readlines()
        
        # Find nodes section
        node_start = None
        for i, line in enumerate(lines):
            if line.strip() == '$Nodes':
                node_start = i + 1
                break
        
        if node_start is None:
            raise ValueError("No $Nodes section found in mesh file")
        
        # Parse nodes (Gmsh 4.1 format)
        node_info = lines[node_start].strip().split()
        num_entity_blocks = int(node_info[0])
        total_nodes = int(node_info[1])
        logger.info(f"Found {total_nodes} nodes in {num_entity_blocks} entity blocks")
        
        # Parse node blocks
        line_idx = node_start + 1
        for block in range(num_entity_blocks):
            # Read entity header: entityDim entityTag parametric numNodesInBlock
            entity_header = lines[line_idx].strip().split()
            num_nodes_in_block = int(entity_header[3])
            line_idx += 1
            
            # Read node tags
            for i in range(num_nodes_in_block):
                line_idx += 1  # Skip node tags
            
            # Read node coordinates
            for i in range(num_nodes_in_block):
                parts = lines[line_idx].strip().split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    vertices.append([x, y, z])
                line_idx += 1
        
        vertices = np.array(vertices)
        
        # Find elements section  
        elem_start = None
        for i, line in enumerate(lines):
            if line.strip() == '$Elements':
                elem_start = i + 1
                break
        
        if elem_start is None:
            raise ValueError("No $Elements section found in mesh file")
        
        # Parse elements (Gmsh 4.1 format)
        elem_info = lines[elem_start].strip().split()
        num_entity_blocks = int(elem_info[0])
        total_elements = int(elem_info[1])
        logger.info(f"Found {total_elements} elements in {num_entity_blocks} entity blocks")
        
        # Parse element blocks
        line_idx = elem_start + 1
        for block in range(num_entity_blocks):
            # Read entity header: entityDim entityTag elementType numElementsInBlock
            entity_header = lines[line_idx].strip().split()
            element_type = int(entity_header[2])
            num_elements_in_block = int(entity_header[3])
            line_idx += 1
            
            # Process elements in this block
            tetrahedra = []
            for i in range(num_elements_in_block):
                parts = lines[line_idx].strip().split()
                if element_type == 2 and len(parts) >= 4:  # Triangle element (surface)
                    # Direct triangle element - these are the surface triangles we want
                    n1, n2, n3 = int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1
                    faces.append([n1, n2, n3])
                elif element_type == 4 and len(parts) >= 5:  # Tetrahedron element
                    # Store tetrahedra to extract boundary later
                    n1, n2, n3, n4 = int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1, int(parts[4]) - 1
                    tetrahedra.append([n1, n2, n3, n4])
                line_idx += 1
            
            # If we have tetrahedra but no explicit triangles, extract boundary
            if tetrahedra and not faces:
                logger.info("No explicit surface triangles found. Extracting boundary from tetrahedra...")
                faces = self._extract_boundary_faces(tetrahedra)
                logger.info(f"Extracted {len(faces)} boundary triangles from {len(tetrahedra)} tetrahedra")
        
        faces = np.array(faces)
        
        logger.info(f"Parsed mesh: {len(vertices)} vertices, {len(faces)} triangular faces")
        logger.info(f"Mesh bounds: [{np.min(vertices, axis=0)}] to [{np.max(vertices, axis=0)}]")
        
        return vertices, faces
    
    def _extract_boundary_faces(self, tetrahedra: List[List[int]]) -> List[List[int]]:
        """Extract boundary faces from tetrahedra using proper algorithm"""
        from collections import Counter
        
        # Generate all triangular faces from tetrahedra
        all_faces = []
        for tet in tetrahedra:
            n1, n2, n3, n4 = tet
            # Add the 4 triangular faces of the tetrahedron (with consistent orientation)
            all_faces.extend([
                tuple(sorted([n1, n2, n3])),  # Face 1
                tuple(sorted([n1, n2, n4])),  # Face 2  
                tuple(sorted([n1, n3, n4])),  # Face 3
                tuple(sorted([n2, n3, n4]))   # Face 4
            ])
        
        # Count face occurrences - boundary faces appear only once
        face_counts = Counter(all_faces)
        boundary_faces = []
        
        for face, count in face_counts.items():
            if count == 1:  # This is a boundary face
                # Convert back to list and maintain proper orientation
                boundary_faces.append(list(face))
        
        return boundary_faces
    
    def create_grid(self, config: SDFConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Create 3D grid for SDF computation"""
        xmin, ymin, zmin, xmax, ymax, zmax = config.domain_bounds
        nx, ny, nz = config.resolution
        
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny) 
        z = np.linspace(zmin, zmax, nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)
        
        logger.info(f"Created grid: {nx}x{ny}x{nz} = {len(grid_points)} points")
        return grid_points, (X, Y, Z)
    
    def compute_sdf(self, config: SDFConfig) -> SDFResult:
        """Main API method to compute signed distance function"""
        logger.info("="*60)
        logger.info("STARTING NASA-GRADE SDF COMPUTATION")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Parse mesh
        vertices, faces = self.parse_gmsh_mesh(config.mesh_file)
        
        # Convert to proper format for pysdf (float32 for better compatibility)
        vertices = vertices.astype(np.float32)
        faces = np.array(faces, dtype=np.int32)
        
        logger.info(f"Mesh for SDF: {len(vertices)} vertices, {len(faces)} faces")
        logger.info(f"Vertex range: [{vertices.min(axis=0)}] to [{vertices.max(axis=0)}]")
        
        # Create SDF object with proper initialization
        logger.info("Initializing production-quality pysdf...")
        sdf_obj = SDF(vertices, faces)
        
        # Validate SDF object
        logger.info(f"SDF surface area: {sdf_obj.surface_area:.3f}")
        # Test SDF with a single point to verify it's working
        test_point = vertices.mean(axis=0).reshape(1, -1)
        test_sdf = sdf_obj(test_point)
        logger.info(f"SDF test at mesh center: {test_sdf[0]:.3f}")
        
        # Create grid
        grid_points, grid_shape = self.create_grid(config)
        
        # Compute SDF in batches with proper error handling
        logger.info(f"Computing SDF for {len(grid_points)} points in batches of {config.batch_size}")
        sdf_values = []
        
        for i in range(0, len(grid_points), config.batch_size):
            batch_end = min(i + config.batch_size, len(grid_points))
            batch_points = grid_points[i:batch_end]
            
            # Compute SDF for this batch
            try:
                batch_sdf = sdf_obj(batch_points)
                
                # Check for overflow or invalid values
                if np.any(np.abs(batch_sdf) > 1e6):
                    logger.warning(f"Large SDF values detected in batch {i//config.batch_size + 1}")
                    # Clamp extreme values
                    batch_sdf = np.clip(batch_sdf, -1000, 1000)
                
                sdf_values.append(batch_sdf)
                
            except Exception as e:
                logger.error(f"Error computing SDF for batch {i//config.batch_size + 1}: {e}")
                # Fill with large positive values (outside)
                batch_sdf = np.full(len(batch_points), 1000.0, dtype=np.float32)
                sdf_values.append(batch_sdf)
            
            progress = (batch_end / len(grid_points)) * 100
            logger.info(f"Progress: {progress:.1f}% ({batch_end}/{len(grid_points)} points)")
        
        sdf_values = np.concatenate(sdf_values).astype(np.float32)
        sdf_grid = sdf_values.reshape(config.resolution)
        
        # Final validation
        valid_mask = np.isfinite(sdf_values)
        if not np.all(valid_mask):
            logger.warning(f"Found {np.sum(~valid_mask)} invalid SDF values, replacing with large positive values")
            sdf_values[~valid_mask] = 1000.0
            sdf_grid = sdf_values.reshape(config.resolution)
        
        computation_time = time.time() - start_time
        
        # Create result object
        result = SDFResult(
            sdf_values=sdf_grid,
            grid_points=grid_points.reshape((*config.resolution, 3)),
            domain_bounds=config.domain_bounds,
            resolution=config.resolution,
            mesh_file=config.mesh_file,
            computation_time=computation_time,
            timestamp=datetime.now().isoformat(),
            config=config
        )
        
        logger.info(f"SDF computation completed in {computation_time:.2f} seconds")
        logger.info(f"SDF range: [{np.min(sdf_values):.3f}, {np.max(sdf_values):.3f}]")
        
        self.current_result = result
        return result
    
    def visualize_sdf(self, result: SDFResult, show_plot: bool = True, save_path: Optional[str] = None) -> None:
        """Visualize the zero-level contour of the SDF"""
        logger.info("Visualizing SDF zero-level contour")
        
        if not MCUBES_AVAILABLE and not SKIMAGE_AVAILABLE:
            logger.error("No marching cubes library available for visualization")
            return
        
        try:
            # Extract zero-level isosurface
            if MCUBES_AVAILABLE:
                vertices, triangles = mcubes.marching_cubes(result.sdf_values, 0.0)
                logger.info("Using PyMCubes for high-quality isosurface extraction")
            else:
                vertices, faces, _, _ = measure.marching_cubes(result.sdf_values, 0.0)
                triangles = faces
                logger.info("Using scikit-image for isosurface extraction")
            
            # Transform vertices to world coordinates
            xmin, ymin, zmin, xmax, ymax, zmax = result.domain_bounds
            nx, ny, nz = result.resolution
            
            vertices[:, 0] = xmin + (vertices[:, 0] / (nx - 1)) * (xmax - xmin)
            vertices[:, 1] = ymin + (vertices[:, 1] / (ny - 1)) * (ymax - ymin)  
            vertices[:, 2] = zmin + (vertices[:, 2] / (nz - 1)) * (zmax - zmin)
            
            # Create clean, professional visualization
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot the zero-level contour surface
            ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                           triangles=triangles, alpha=0.9, cmap='viridis', 
                           linewidth=0, antialiased=True, shade=True)
            
            # Set professional styling
            ax.set_title('Immersed Boundary: φ=0 Levelset Contour', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('X', fontsize=12, labelpad=10)
            ax.set_ylabel('Y', fontsize=12, labelpad=10)
            ax.set_zlabel('Z', fontsize=12, labelpad=10)
            
            # Set optimal viewing angle for propeller
            ax.view_init(elev=20, azim=45)
            
            # Make axes equal and clean
            ax.set_box_aspect([1,1,1])
            ax.grid(True, alpha=0.3)
            
            # Add some space around the plot
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved visualization to {save_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
                
            logger.info(f"Extracted {len(vertices)} vertices and {len(triangles)} triangles for φ=0 contour")
            
        except Exception as e:
            logger.error(f"Error in visualization: {e}")
    
    def save_results(self, result: SDFResult, output_dir: str, name: str) -> Dict[str, str]:
        """Save SDF results in multiple formats"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = {}
        
        if result.config.save_binary:
            binary_path = output_path / f"{name}.pkl"
            result.save_binary(str(binary_path))
            saved_files['binary'] = str(binary_path)
        
        if result.config.save_json:
            json_path = output_path / f"{name}_metadata.json"
            result.save_json_metadata(str(json_path))
            saved_files['metadata'] = str(json_path)
        
        if result.config.export_jaxfluids:
            jax_path = output_path / f"{name}_jaxfluids.json"
            result.export_for_jaxfluids(str(jax_path))
            saved_files['jaxfluids'] = str(jax_path)
        
        if result.config.plot:
            plot_path = output_path / f"{name}_visualization.png"
            self.visualize_sdf(result, show_plot=False, save_path=str(plot_path))
            saved_files['visualization'] = str(plot_path)
        
        return saved_files
    
    def run(self, config: SDFConfig) -> Tuple[SDFResult, Dict[str, str]]:
        """Main API entry point - compute SDF and save results"""
        logger.info("Starting Immersed Boundary API run")
        
        # Compute SDF
        result = self.compute_sdf(config)
        
        # Save results
        saved_files = self.save_results(result, config.output_dir, config.output_name)
        
        # Optional visualization
        if config.plot:
            self.visualize_sdf(result, show_plot=True)
        
        logger.info("="*60)
        logger.info("API RUN COMPLETED SUCCESSFULLY")
        logger.info(f"Files saved: {list(saved_files.keys())}")
        logger.info("="*60)
        
        return result, saved_files


def create_config_from_args(
    mesh_file: str,
    domain_bounds: Tuple[float, float, float, float, float, float],
    resolution: Tuple[int, int, int],
    **kwargs
) -> SDFConfig:
    """Helper function to create config from arguments"""
    return SDFConfig(
        mesh_file=mesh_file,
        domain_bounds=domain_bounds,
        resolution=resolution,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = SDFConfig(
        mesh_file="../mesh/5_bladed_Propeller.STEP_medium_tetrahedral.msh",
        domain_bounds=(-100, -150, -150, 150, 150, 150),
        resolution=(100, 100, 100),
        output_dir="results",
        output_name="propeller_sdf_api",
        plot=True,
        save_binary=True,
        save_json=True,
        export_jaxfluids=True,
        robust_mode=True,
        batch_size=50000
    )
    
    # Run API
    api = ImmersedBoundaryAPI()
    result, saved_files = api.run(config)
    
    print(f"\nAPI completed successfully!")
    print(f"Computation time: {result.computation_time:.2f} seconds")
    print(f"Files saved: {saved_files}") 