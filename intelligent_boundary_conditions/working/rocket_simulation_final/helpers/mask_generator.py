#!/usr/bin/env python3
"""
Virtual Boundary Mask Generator
Production-ready mask generation for JAX-Fluids rocket nozzle simulation
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import sys
import logging
from typing import Tuple, Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from circular_face_creator import find_circular_boundary_edges, fit_circle_and_create_face

class VirtualBoundaryMaskGenerator:
    """Professional mask generator for rocket nozzle virtual boundaries"""
    
    def __init__(self, 
                 domain_bounds: list = [-200.0, -800.0, -800.0, 1800.0, 800.0, 800.0],
                 grid_shape: tuple = (128, 64, 64),
                 output_dir: str = "masks"):
        """
        Initialize the mask generator
        
        Args:
            domain_bounds: [x_min, y_min, z_min, x_max, y_max, z_max]
            grid_shape: (nx, ny, nz)
            output_dir: Directory to save masks
        """
        self.domain_bounds = np.array(domain_bounds)
        self.grid_shape = grid_shape
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def find_mesh_file(self) -> Path:
        """Find the rocket mesh file"""
        possible_paths = [
            Path("../../../mesh/Rocket Engine.msh"),
            Path("../../mesh/Rocket Engine.msh"),
            Path("../mesh/Rocket Engine.msh"),
            Path("mesh/Rocket Engine.msh")
        ]
        
        for mesh_file in possible_paths:
            if mesh_file.exists():
                self.logger.info(f"Found mesh file: {mesh_file}")
                return mesh_file
        
        raise FileNotFoundError("❌ Rocket mesh file not found in any expected location")
    
    def detect_virtual_faces(self, mesh_file: Path) -> Tuple[Dict, Dict]:
        """Detect inlet and outlet virtual faces"""
        self.logger.info("🔍 Detecting virtual inlet/outlet faces...")
        
        # Find boundary edges
        inlet_points, outlet_points = find_circular_boundary_edges(str(mesh_file))
        
        if inlet_points is None or outlet_points is None:
            raise RuntimeError("❌ Failed to detect inlet/outlet boundary edges")
        
        self.logger.info(f"   ✅ Inlet: {len(inlet_points)} boundary points")
        self.logger.info(f"   ✅ Outlet: {len(outlet_points)} boundary points")
        
        # Create virtual faces
        self.logger.info("🔧 Creating virtual circular faces...")
        inlet_face = fit_circle_and_create_face(inlet_points)
        outlet_face = fit_circle_and_create_face(outlet_points)
        
        self.logger.info(f"   🔵 Inlet: center=({inlet_face['center'][0]:.1f}, {inlet_face['center'][1]:.1f}, {inlet_face['center'][2]:.1f}), radius={inlet_face['radius']:.1f}")
        self.logger.info(f"   🔴 Outlet: center=({outlet_face['center'][0]:.1f}, {outlet_face['center'][1]:.1f}, {outlet_face['center'][2]:.1f}), radius={outlet_face['radius']:.1f}")
        
        return inlet_face, outlet_face
    
    def create_structured_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create JAX-Fluids structured grid"""
        nx, ny, nz = self.grid_shape
        x_min, y_min, z_min, x_max, y_max, z_max = self.domain_bounds
        
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        z = np.linspace(z_min, z_max, nz)
        
        return np.meshgrid(x, y, z, indexing='ij')
    
    def generate_masks(self, inlet_face: Dict, outlet_face: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 3D boolean masks for inlet/outlet regions"""
        self.logger.info("🎯 Generating JAX-Fluids compatible masks...")
        
        # Create structured grid
        X, Y, Z = self.create_structured_grid()
        nx, ny, nz = self.grid_shape
        
        # Initialize masks
        inlet_mask = np.zeros(self.grid_shape, dtype=bool)
        outlet_mask = np.zeros(self.grid_shape, dtype=bool)
        
        # Get face parameters
        inlet_center = inlet_face['center']
        inlet_radius = inlet_face['radius']
        inlet_x = inlet_center[0]
        
        outlet_center = outlet_face['center']
        outlet_radius = outlet_face['radius']
        outlet_x = outlet_center[0]
        
        # Grid spacing for tolerance
        x_tolerance = (self.domain_bounds[3] - self.domain_bounds[0]) / nx * 2
        
        # Generate masks using vectorized operations for efficiency
        self.logger.info("   Generating inlet mask...")
        inlet_x_mask = np.abs(X - inlet_x) < x_tolerance
        inlet_distance = np.sqrt((Y - inlet_center[1])**2 + (Z - inlet_center[2])**2)
        inlet_mask = inlet_x_mask & (inlet_distance <= inlet_radius)
        
        self.logger.info("   Generating outlet mask...")
        outlet_x_mask = np.abs(X - outlet_x) < x_tolerance
        outlet_distance = np.sqrt((Y - outlet_center[1])**2 + (Z - outlet_center[2])**2)
        outlet_mask = outlet_x_mask & (outlet_distance <= outlet_radius)
        
        # Validate masks
        inlet_count = np.sum(inlet_mask)
        outlet_count = np.sum(outlet_mask)
        
        if inlet_count == 0:
            raise ValueError("❌ Generated inlet mask is empty")
        if outlet_count == 0:
            raise ValueError("❌ Generated outlet mask is empty")
        
        self.logger.info(f"   ✅ Inlet mask: {inlet_count:,} active grid points")
        self.logger.info(f"   ✅ Outlet mask: {outlet_count:,} active grid points")
        
        return inlet_mask, outlet_mask
    
    def save_masks(self, inlet_mask: np.ndarray, outlet_mask: np.ndarray) -> Dict[str, Path]:
        """Save masks to files"""
        inlet_file = self.output_dir / "inlet_boundary_mask.npy"
        outlet_file = self.output_dir / "outlet_boundary_mask.npy"
        
        np.save(inlet_file, inlet_mask)
        np.save(outlet_file, outlet_mask)
        
        self.logger.info(f"💾 Saved masks:")
        self.logger.info(f"   ✅ Inlet: {inlet_file}")
        self.logger.info(f"   ✅ Outlet: {outlet_file}")
        
        return {"inlet": inlet_file, "outlet": outlet_file}
    
    def validate_masks(self, inlet_mask: np.ndarray, outlet_mask: np.ndarray) -> bool:
        """Validate generated masks"""
        self.logger.info("🔬 Validating masks...")
        
        # Check for overlap
        overlap = np.sum(inlet_mask & outlet_mask)
        if overlap > 0:
            self.logger.warning(f"⚠️  Mask overlap detected: {overlap} points")
        
        # Check mask size ratios (outlet should be larger)
        inlet_size = np.sum(inlet_mask)
        outlet_size = np.sum(outlet_mask)
        
        if outlet_size <= inlet_size:
            self.logger.warning("⚠️  Outlet mask should be larger than inlet mask")
        
        # Check positioning (outlet should be at higher X)
        inlet_indices = np.where(inlet_mask)
        outlet_indices = np.where(outlet_mask)
        
        inlet_x_mean = np.mean(inlet_indices[0])
        outlet_x_mean = np.mean(outlet_indices[0])
        
        if outlet_x_mean <= inlet_x_mean:
            self.logger.warning("⚠️  Outlet should be downstream of inlet")
        
        self.logger.info("✅ Mask validation complete")
        return True
    
    def generate_all(self) -> Dict[str, Any]:
        """Complete mask generation pipeline"""
        self.logger.info("🚀 STARTING VIRTUAL BOUNDARY MASK GENERATION")
        self.logger.info("=" * 60)
        
        try:
            # Find mesh file
            mesh_file = self.find_mesh_file()
            
            # Detect virtual faces
            inlet_face, outlet_face = self.detect_virtual_faces(mesh_file)
            
            # Generate masks
            inlet_mask, outlet_mask = self.generate_masks(inlet_face, outlet_face)
            
            # Validate masks
            self.validate_masks(inlet_mask, outlet_mask)
            
            # Save masks
            mask_files = self.save_masks(inlet_mask, outlet_mask)
            
            result = {
                "inlet_face": inlet_face,
                "outlet_face": outlet_face,
                "inlet_mask": inlet_mask,
                "outlet_mask": outlet_mask,
                "files": mask_files,
                "domain_bounds": self.domain_bounds,
                "grid_shape": self.grid_shape
            }
            
            self.logger.info("🎉 MASK GENERATION COMPLETED SUCCESSFULLY!")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Mask generation failed: {e}")
            raise

def main():
    """Main function for standalone execution"""
    generator = VirtualBoundaryMaskGenerator()
    result = generator.generate_all()
    
    print("\n" + "="*60)
    print("MASK GENERATION SUMMARY")
    print("="*60)
    print(f"Inlet points: {np.sum(result['inlet_mask']):,}")
    print(f"Outlet points: {np.sum(result['outlet_mask']):,}")
    print(f"Domain: {result['domain_bounds']}")
    print(f"Grid shape: {result['grid_shape']}")
    print("="*60)

if __name__ == "__main__":
    main() 