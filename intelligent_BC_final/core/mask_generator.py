#!/usr/bin/env python3
"""
Boundary Mask Generator
Maps virtual inlet/outlet faces to JAX-Fluids Cartesian grid masks
Based on the working generate_jax_masks.py implementation
"""

import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

class BoundaryMaskGenerator:
    """
    Generates 3D boolean masks for JAX-Fluids forcing system
    Maps virtual circular faces to Cartesian grid points
    """
    
    def __init__(self, 
                 domain_bounds: list,
                 grid_shape: tuple,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize mask generator
        
        Args:
            domain_bounds: [x_min, y_min, z_min, x_max, y_max, z_max]
            grid_shape: (nx, ny, nz) 
            logger: Optional logger instance
        """
        self.domain_bounds = np.array(domain_bounds)
        self.grid_shape = grid_shape
        self.logger = logger or logging.getLogger(__name__)
        
        # Create grid coordinate arrays
        self._setup_grid_coordinates()
        
        self.inlet_mask = None
        self.outlet_mask = None
        
    def _setup_grid_coordinates(self):
        """Setup Cartesian grid coordinates"""
        x_min, y_min, z_min, x_max, y_max, z_max = self.domain_bounds
        nx, ny, nz = self.grid_shape
        
        # Create coordinate arrays (cell centers)
        self.x_coords = np.linspace(x_min, x_max, nx)
        self.y_coords = np.linspace(y_min, y_max, ny) 
        self.z_coords = np.linspace(z_min, z_max, nz)
        
        # Create 3D coordinate grids
        self.X, self.Y, self.Z = np.meshgrid(
            self.x_coords, self.y_coords, self.z_coords, indexing='ij'
        )
        
        self.logger.debug(f"Grid setup: {nx}x{ny}x{nz}")
        self.logger.debug(f"Domain: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}], Z=[{z_min}, {z_max}]")
        
    def generate_masks_from_faces(self, 
                                  inlet_face_data: Dict[str, Any],
                                  outlet_face_data: Dict[str, Any],
                                  thickness: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 3D masks from virtual face data
        
        Args:
            inlet_face_data: Inlet face geometry data
            outlet_face_data: Outlet face geometry data
            thickness: Mask thickness in each direction
            
        Returns:
            Tuple of (inlet_mask, outlet_mask) as boolean arrays
        """
        self.logger.info("Generating boundary masks from virtual faces...")
        
        # Generate inlet mask
        self.inlet_mask = self._create_circular_mask(
            inlet_face_data, thickness, "inlet"
        )
        
        # Generate outlet mask  
        self.outlet_mask = self._create_circular_mask(
            outlet_face_data, thickness, "outlet"
        )
        
        # Validate masks
        self._validate_masks()
        
        self.logger.info(f"Inlet mask: {np.sum(self.inlet_mask)} active points")
        self.logger.info(f"Outlet mask: {np.sum(self.outlet_mask)} active points")
        
        return self.inlet_mask, self.outlet_mask
        
    def _create_circular_mask(self, 
                             face_data: Dict[str, Any], 
                             thickness: float,
                             face_type: str) -> np.ndarray:
        """Create 3D mask for circular virtual face"""
        self.logger.debug(f"Creating {face_type} mask...")
        
        center = face_data["center"]
        radius = face_data["radius"]
        normal = face_data["normal"]
        
        self.logger.debug(f"{face_type} - center: {center}, radius: {radius:.2f}")
        
        # Initialize mask
        mask = np.zeros(self.grid_shape, dtype=bool)
        
        # Get all grid points as a flat array
        grid_points = np.column_stack([
            self.X.flatten(),
            self.Y.flatten(), 
            self.Z.flatten()
        ])
        
        # Vectorized distance calculations
        # Distance from center
        distances_from_center = np.linalg.norm(grid_points - center, axis=1)
        
        # Distance along normal direction (perpendicular distance from face plane)
        face_plane_distances = np.abs(np.dot(grid_points - center, normal))
        
        # Create cylindrical region: within radius and within thickness
        within_radius = distances_from_center <= (radius * 1.2)  # 20% buffer for discretization
        within_thickness = face_plane_distances <= thickness
        
        # Combine conditions
        in_mask = within_radius & within_thickness
        
        # Additional geometric check for circular cross-section
        # Project points to face plane and check if within circle
        for i, point in enumerate(grid_points):
            if not in_mask[i]:
                continue
                
            # Vector from center to point
            to_point = point - center
            
            # Remove component along normal (project to face plane)
            projected_vector = to_point - np.dot(to_point, normal) * normal
            projected_distance = np.linalg.norm(projected_vector)
            
            # Check if within circular cross-section
            if projected_distance > radius:
                in_mask[i] = False
                
        # Reshape back to 3D grid
        mask = in_mask.reshape(self.grid_shape)
        
        self.logger.debug(f"{face_type} mask: {np.sum(mask)} points")
        
        return mask
        
    def generate_masks_from_coordinates(self,
                                       inlet_center: np.ndarray,
                                       inlet_radius: float,
                                       outlet_center: np.ndarray, 
                                       outlet_radius: float,
                                       thickness: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate masks from simple coordinate specifications
        
        Args:
            inlet_center: [x, y, z] coordinates
            inlet_radius: Inlet radius
            outlet_center: [x, y, z] coordinates
            outlet_radius: Outlet radius  
            thickness: Mask thickness
            
        Returns:
            Tuple of (inlet_mask, outlet_mask)
        """
        self.logger.info("Generating masks from coordinates...")
        
        # Create simple face data dictionaries
        inlet_face_data = {
            "center": np.array(inlet_center),
            "radius": inlet_radius,
            "normal": np.array([1, 0, 0])  # Assume X-normal for simplicity
        }
        
        outlet_face_data = {
            "center": np.array(outlet_center), 
            "radius": outlet_radius,
            "normal": np.array([1, 0, 0])  # Assume X-normal for simplicity
        }
        
        return self.generate_masks_from_faces(inlet_face_data, outlet_face_data, thickness)
        
    def _validate_masks(self):
        """Validate generated masks"""
        if self.inlet_mask is None or self.outlet_mask is None:
            raise RuntimeError("Masks not generated")
            
        # Check mask shapes
        if self.inlet_mask.shape != self.grid_shape:
            raise ValueError(f"Inlet mask shape {self.inlet_mask.shape} != grid shape {self.grid_shape}")
        if self.outlet_mask.shape != self.grid_shape:
            raise ValueError(f"Outlet mask shape {self.outlet_mask.shape} != grid shape {self.grid_shape}")
            
        # Check for overlapping masks
        overlap = np.logical_and(self.inlet_mask, self.outlet_mask)
        if np.any(overlap):
            overlap_count = np.sum(overlap)
            self.logger.warning(f"Mask overlap detected: {overlap_count} points")
            
        # Check for empty masks
        if np.sum(self.inlet_mask) == 0:
            raise ValueError("Inlet mask is empty")
        if np.sum(self.outlet_mask) == 0:
            raise ValueError("Outlet mask is empty")
            
        self.logger.debug("Mask validation passed")
        
    def save_masks(self, output_dir: str):
        """Save masks to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.inlet_mask is None or self.outlet_mask is None:
            raise RuntimeError("Masks not generated")
            
        # Save masks
        inlet_file = output_path / "inlet_boundary_mask.npy"
        outlet_file = output_path / "outlet_boundary_mask.npy"
        
        np.save(inlet_file, self.inlet_mask)
        np.save(outlet_file, self.outlet_mask)
        
        self.logger.info(f"Masks saved:")
        self.logger.info(f"  Inlet: {inlet_file}")
        self.logger.info(f"  Outlet: {outlet_file}")
        
        # Save metadata
        metadata = {
            "domain_bounds": self.domain_bounds.tolist(),
            "grid_shape": self.grid_shape,
            "inlet_points": int(np.sum(self.inlet_mask)),
            "outlet_points": int(np.sum(self.outlet_mask))
        }
        
        import json
        metadata_file = output_path / "mask_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"  Metadata: {metadata_file}")
        
    def visualize_masks(self, output_path: str = "mask_visualization.png"):
        """Create visualization of generated masks"""
        if self.inlet_mask is None or self.outlet_mask is None:
            raise RuntimeError("Masks not generated")
            
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with multiple views
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Get center slices
            nx, ny, nz = self.grid_shape
            x_center, y_center, z_center = nx//2, ny//2, nz//2
            
            # XY slices (Z = center)
            ax = axes[0,0]
            inlet_xy = self.inlet_mask[:, :, z_center].T
            outlet_xy = self.outlet_mask[:, :, z_center].T
            combined = inlet_xy.astype(int) + 2 * outlet_xy.astype(int)
            im = ax.imshow(combined, origin='lower', cmap='Set1', vmin=0, vmax=3)
            ax.set_title('XY slice - Inlet(blue) Outlet(red)')
            
            # XZ slices (Y = center)
            ax = axes[0,1] 
            inlet_xz = self.inlet_mask[:, y_center, :].T
            outlet_xz = self.outlet_mask[:, y_center, :].T
            combined = inlet_xz.astype(int) + 2 * outlet_xz.astype(int)
            ax.imshow(combined, origin='lower', cmap='Set1', vmin=0, vmax=3)
            ax.set_title('XZ slice - Side view')
            
            # YZ slices (X = center)
            ax = axes[0,2]
            inlet_yz = self.inlet_mask[x_center, :, :].T  
            outlet_yz = self.outlet_mask[x_center, :, :].T
            combined = inlet_yz.astype(int) + 2 * outlet_yz.astype(int)
            ax.imshow(combined, origin='lower', cmap='Set1', vmin=0, vmax=3)
            ax.set_title('YZ slice - Cross section')
            
            # Statistics plots
            ax = axes[1,0]
            inlet_points = np.sum(self.inlet_mask, axis=(1,2))
            outlet_points = np.sum(self.outlet_mask, axis=(1,2))
            ax.plot(self.x_coords, inlet_points, 'b-', label='Inlet', linewidth=2)
            ax.plot(self.x_coords, outlet_points, 'r-', label='Outlet', linewidth=2) 
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Points per slice')
            ax.set_title('Mask distribution along X')
            ax.legend()
            ax.grid(True)
            
            # Mask overlap check
            ax = axes[1,1]
            overlap = np.logical_and(self.inlet_mask, self.outlet_mask)
            ax.bar(['Inlet only', 'Outlet only', 'Overlap'], 
                  [np.sum(self.inlet_mask & ~self.outlet_mask),
                   np.sum(self.outlet_mask & ~self.inlet_mask),
                   np.sum(overlap)],
                  color=['blue', 'red', 'purple'])
            ax.set_ylabel('Number of points')
            ax.set_title('Mask Statistics')
            
            # Summary text
            ax = axes[1,2]
            ax.axis('off')
            summary_text = f"""
            Grid Shape: {self.grid_shape}
            Domain: X=[{self.domain_bounds[0]:.0f}, {self.domain_bounds[3]:.0f}]
                   Y=[{self.domain_bounds[1]:.0f}, {self.domain_bounds[4]:.0f}]
                   Z=[{self.domain_bounds[2]:.0f}, {self.domain_bounds[5]:.0f}]
            
            Inlet Points: {np.sum(self.inlet_mask)}
            Outlet Points: {np.sum(self.outlet_mask)}
            Overlap Points: {np.sum(overlap)}
            
            Coverage:
            Inlet: {100*np.sum(self.inlet_mask)/np.prod(self.grid_shape):.3f}%
            Outlet: {100*np.sum(self.outlet_mask)/np.prod(self.grid_shape):.3f}%
            """
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Mask visualization saved: {output_path}")
            
        except ImportError:
            self.logger.warning("Matplotlib not available, skipping visualization")
            
    def get_mask_centroids(self) -> Dict[str, np.ndarray]:
        """Get centroids of active mask regions"""
        if self.inlet_mask is None or self.outlet_mask is None:
            raise RuntimeError("Masks not generated")
            
        # Find centroids
        inlet_indices = np.where(self.inlet_mask)
        outlet_indices = np.where(self.outlet_mask)
        
        inlet_centroid = np.array([
            self.x_coords[inlet_indices[0]].mean(),
            self.y_coords[inlet_indices[1]].mean(), 
            self.z_coords[inlet_indices[2]].mean()
        ])
        
        outlet_centroid = np.array([
            self.x_coords[outlet_indices[0]].mean(),
            self.y_coords[outlet_indices[1]].mean(),
            self.z_coords[outlet_indices[2]].mean()
        ])
        
        return {
            "inlet": inlet_centroid,
            "outlet": outlet_centroid
        } 