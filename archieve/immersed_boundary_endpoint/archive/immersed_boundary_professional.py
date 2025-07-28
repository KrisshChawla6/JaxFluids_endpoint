#!/usr/bin/env python3
"""
Professional Immersed Boundary Implementation for CFD

This module creates clean, precise immersed boundaries suitable for 
professional CFD simulations with JAX-Fluids.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import time

logger = logging.getLogger(__name__)

from mesh_processor import GmshProcessor
from proper_sdf_3d import ProperSDF3D
from proper_contour_viz import ProperContourVisualizer

class ImmersedBoundaryProfessional:
    """
    Professional immersed boundary implementation for CFD applications.
    
    Creates clean, precise object boundaries with proper resolution
    control for high-quality CFD simulations.
    """
    
    def __init__(self, mesh_processor: GmshProcessor):
        """Initialize with mesh processor."""
        self.mesh_processor = mesh_processor
        self.sdf_computer = ProperSDF3D(mesh_processor.surface_triangles)
        self.visualizer = ProperContourVisualizer(mesh_processor)
        
        # Get object bounds for focused domain
        all_vertices = np.array(mesh_processor.surface_triangles).reshape(-1, 3)
        self.object_min = all_vertices.min(axis=0)
        self.object_max = all_vertices.max(axis=0)
        self.object_center = (self.object_min + self.object_max) / 2
        self.object_size = self.object_max - self.object_min
        
        logger.info(f"Initialized ImmersedBoundaryProfessional")
        logger.info(f"  Object center: {self.object_center}")
        logger.info(f"  Object size: {self.object_size}")
        
    def create_focused_domain(self, 
                            boundary_layers: int = 10,
                            layer_thickness: float = None,
                            resolution_factor: float = 2.0) -> Dict[str, Any]:
        """
        Create a focused computational domain around the object.
        
        Args:
            boundary_layers: Number of boundary layers around object
            layer_thickness: Thickness of each boundary layer (auto if None)
            resolution_factor: Resolution multiplier around object
            
        Returns:
            Dictionary with domain information
        """
        if layer_thickness is None:
            # Auto-determine layer thickness based on object size
            layer_thickness = np.min(self.object_size) * 0.05
        
        # Create focused domain bounds
        total_padding = boundary_layers * layer_thickness
        domain_min = self.object_min - total_padding
        domain_max = self.object_max + total_padding
        domain_size = domain_max - domain_min
        
        logger.info(f"Created focused domain:")
        logger.info(f"  Boundary layers: {boundary_layers}")
        logger.info(f"  Layer thickness: {layer_thickness:.3f}")
        logger.info(f"  Total padding: {total_padding:.3f}")
        logger.info(f"  Domain size: {domain_size}")
        
        return {
            'domain_min': domain_min,
            'domain_max': domain_max,
            'domain_size': domain_size,
            'boundary_layers': boundary_layers,
            'layer_thickness': layer_thickness,
            'total_padding': total_padding
        }
    
    def create_high_resolution_grid(self, 
                                  domain_info: Dict[str, Any],
                                  base_resolution: Tuple[int, int, int] = (80, 80, 80),
                                  near_boundary_factor: float = 1.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create high-resolution grid with refinement near boundaries.
        
        Args:
            domain_info: Domain information from create_focused_domain
            base_resolution: Base grid resolution
            near_boundary_factor: Resolution multiplier near boundaries
            
        Returns:
            Grid coordinate arrays (X, Y, Z)
        """
        domain_min = domain_info['domain_min']
        domain_max = domain_info['domain_max']
        
        # Create base uniform grid
        nx, ny, nz = base_resolution
        
        # For now, use uniform grid (could implement adaptive refinement later)
        x = np.linspace(domain_min[0], domain_max[0], nx)
        y = np.linspace(domain_min[1], domain_max[1], ny) 
        z = np.linspace(domain_min[2], domain_max[2], nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        total_points = np.prod(base_resolution)
        logger.info(f"Created high-resolution grid: {base_resolution} = {total_points:,} points")
        
        return X, Y, Z
    
    def compute_immersed_boundary_sdf(self, 
                                    grid_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                    show_progress: bool = True) -> np.ndarray:
        """
        Compute SDF for immersed boundary with high accuracy.
        
        Args:
            grid_coords: Grid coordinate arrays
            show_progress: Whether to show computation progress
            
        Returns:
            SDF values on the grid
        """
        X, Y, Z = grid_coords
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        logger.info(f"Computing immersed boundary SDF for {len(grid_points):,} points...")
        
        start_time = time.time()
        sdf_values = self.sdf_computer.compute_sdf_batch_parallel(grid_points)
        elapsed = time.time() - start_time
        
        # Quality metrics
        inside_fraction = np.sum(sdf_values < 0) / len(sdf_values)
        boundary_points = np.sum(np.abs(sdf_values) < 0.1)
        
        logger.info(f"✓ SDF computation completed in {elapsed:.1f}s")
        logger.info(f"  - Processing rate: {len(grid_points)/elapsed:.0f} pts/sec")
        logger.info(f"  - SDF range: [{sdf_values.min():.3f}, {sdf_values.max():.3f}]")
        logger.info(f"  - Inside fraction: {inside_fraction*100:.2f}%")
        logger.info(f"  - Near-boundary points: {boundary_points:,}")
        
        return sdf_values
    
    def visualize_immersed_boundary(self, 
                                  grid_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                  sdf_values: np.ndarray,
                                  output_dir: Path,
                                  show_original_mesh: bool = True) -> Dict[str, Path]:
        """
        Create professional visualization of the immersed boundary.
        
        Args:
            grid_coords: Grid coordinates
            sdf_values: SDF values
            output_dir: Output directory for images
            show_original_mesh: Whether to show original mesh points
            
        Returns:
            Dictionary of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # 1. Clean boundary surface (φ=0)
        logger.info("Creating clean boundary surface visualization...")
        try:
            fig1 = self.visualizer.plot_professional_3d_contour(
                grid_coords, sdf_values,
                levels=[0.0],
                figsize=(16, 12),
                show_mesh=show_original_mesh,
                alpha=0.9
            )
            fig1.suptitle("Professional Immersed Boundary (φ=0)", fontsize=16, y=0.95)
            
            # Enhanced styling for CFD presentation
            ax = fig1.get_axes()[0]
            ax.set_xlabel('X [m]', fontsize=12, labelpad=10)
            ax.set_ylabel('Y [m]', fontsize=12, labelpad=10) 
            ax.set_zlabel('Z [m]', fontsize=12, labelpad=10)
            
            # Better viewing angle for propeller
            ax.view_init(elev=15, azim=30)
            
            boundary_path = output_dir / "immersed_boundary_clean.png"
            fig1.savefig(boundary_path, dpi=300, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
            plt.show()
            plt.close(fig1)
            
            saved_files['boundary'] = boundary_path
            logger.info(f"✓ Saved clean boundary to {boundary_path}")
            
        except Exception as e:
            logger.error(f"✗ Boundary visualization failed: {e}")
        
        # 2. Near-boundary field structure
        logger.info("Creating near-boundary field structure...")
        try:
            # Focus on levels near the boundary
            max_dist = max(abs(sdf_values.min()), abs(sdf_values.max()))
            boundary_range = min(max_dist * 0.2, np.min(self.object_size) * 0.5)
            
            levels = [-boundary_range/2, -boundary_range/4, 0.0, boundary_range/4, boundary_range/2]
            
            fig2 = self.visualizer.plot_professional_3d_contour(
                grid_coords, sdf_values,
                levels=levels,
                figsize=(18, 12),
                show_mesh=False,  # Too cluttered with multiple levels
                alpha=0.7,
                colors=['darkred', 'red', 'blue', 'cyan', 'lightblue']
            )
            fig2.suptitle("Immersed Boundary: Near-Field Structure", fontsize=16, y=0.95)
            
            ax = fig2.get_axes()[0]
            ax.view_init(elev=15, azim=30)
            
            nearfield_path = output_dir / "immersed_boundary_nearfield.png"
            fig2.savefig(nearfield_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.show()
            plt.close(fig2)
            
            saved_files['nearfield'] = nearfield_path
            logger.info(f"✓ Saved near-field structure to {nearfield_path}")
            
        except Exception as e:
            logger.error(f"✗ Near-field visualization failed: {e}")
        
        # 3. Cross-section view
        logger.info("Creating cross-section view...")
        try:
            fig3, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig3.suptitle("Immersed Boundary: Cross-Section Analysis", fontsize=16)
            
            X, Y, Z = grid_coords
            sdf_3d = sdf_values.reshape(X.shape)
            
            # Find center indices
            center_i = X.shape[0] // 2
            center_j = X.shape[1] // 2  
            center_k = X.shape[2] // 2
            
            # XY plane (constant Z)
            ax1 = axes[0, 0]
            im1 = ax1.contourf(X[:, :, center_k], Y[:, :, center_k], sdf_3d[:, :, center_k], 
                              levels=20, cmap='RdBu_r')
            ax1.contour(X[:, :, center_k], Y[:, :, center_k], sdf_3d[:, :, center_k], 
                       levels=[0], colors='black', linewidths=2)
            ax1.set_title('XY Cross-Section (φ=0 in black)')
            ax1.set_xlabel('X [m]')
            ax1.set_ylabel('Y [m]')
            ax1.set_aspect('equal')
            plt.colorbar(im1, ax=ax1, label='SDF Value')
            
            # XZ plane (constant Y)
            ax2 = axes[0, 1]
            im2 = ax2.contourf(X[:, center_j, :], Z[:, center_j, :], sdf_3d[:, center_j, :], 
                              levels=20, cmap='RdBu_r')
            ax2.contour(X[:, center_j, :], Z[:, center_j, :], sdf_3d[:, center_j, :], 
                       levels=[0], colors='black', linewidths=2)
            ax2.set_title('XZ Cross-Section (φ=0 in black)')
            ax2.set_xlabel('X [m]')
            ax2.set_ylabel('Z [m]')
            ax2.set_aspect('equal')
            plt.colorbar(im2, ax=ax2, label='SDF Value')
            
            # YZ plane (constant X)
            ax3 = axes[1, 0]
            im3 = ax3.contourf(Y[center_i, :, :], Z[center_i, :, :], sdf_3d[center_i, :, :], 
                              levels=20, cmap='RdBu_r')
            ax3.contour(Y[center_i, :, :], Z[center_i, :, :], sdf_3d[center_i, :, :], 
                       levels=[0], colors='black', linewidths=2)
            ax3.set_title('YZ Cross-Section (φ=0 in black)')
            ax3.set_xlabel('Y [m]')
            ax3.set_ylabel('Z [m]')
            ax3.set_aspect('equal')
            plt.colorbar(im3, ax=ax3, label='SDF Value')
            
            # SDF histogram
            ax4 = axes[1, 1]
            ax4.hist(sdf_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='φ=0 (Boundary)')
            ax4.set_xlabel('SDF Value')
            ax4.set_ylabel('Frequency')
            ax4.set_title('SDF Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            crosssection_path = output_dir / "immersed_boundary_crosssection.png"
            fig3.savefig(crosssection_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.show()
            plt.close(fig3)
            
            saved_files['crosssection'] = crosssection_path
            logger.info(f"✓ Saved cross-section analysis to {crosssection_path}")
            
        except Exception as e:
            logger.error(f"✗ Cross-section visualization failed: {e}")
        
        return saved_files
    
    def export_for_jax_fluids(self, 
                            grid_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
                            sdf_values: np.ndarray,
                            output_dir: Path) -> Dict[str, Path]:
        """
        Export immersed boundary data in JAX-Fluids compatible format.
        
        Args:
            grid_coords: Grid coordinates
            sdf_values: SDF values
            output_dir: Output directory
            
        Returns:
            Dictionary of exported file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # Export SDF grid data
        logger.info("Exporting SDF data for JAX-Fluids...")
        
        X, Y, Z = grid_coords
        sdf_3d = sdf_values.reshape(X.shape)
        
        # Save as numpy arrays
        sdf_data_path = output_dir / "immersed_boundary_sdf.npz"
        np.savez_compressed(
            sdf_data_path,
            sdf_values=sdf_3d,
            x_coords=X[:, 0, 0],
            y_coords=Y[0, :, 0], 
            z_coords=Z[0, 0, :],
            grid_shape=X.shape,
            object_center=self.object_center,
            object_size=self.object_size
        )
        exported_files['sdf_data'] = sdf_data_path
        
        # Export boundary mesh
        boundary_mesh_path = output_dir / "immersed_boundary.obj"
        success = self.visualizer.export_isosurface_mesh(
            grid_coords, sdf_values, level=0.0, filename=str(boundary_mesh_path)
        )
        if success:
            exported_files['boundary_mesh'] = boundary_mesh_path
        
        # Export configuration file for JAX-Fluids
        config_path = output_dir / "immersed_boundary_config.json"
        import json
        
        config = {
            "immersed_boundary": {
                "method": "levelset",
                "sdf_file": str(sdf_data_path.name),
                "mesh_file": str(boundary_mesh_path.name),
                "grid_shape": X.shape,
                "domain_bounds": {
                    "min": [float(X.min()), float(Y.min()), float(Z.min())],
                    "max": [float(X.max()), float(Y.max()), float(Z.max())]
                },
                "object_info": {
                    "center": [float(x) for x in self.object_center],
                    "size": [float(x) for x in self.object_size],
                    "num_triangles": len(self.mesh_processor.surface_triangles)
                }
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        exported_files['config'] = config_path
        
        logger.info(f"✓ Exported JAX-Fluids data:")
        for key, path in exported_files.items():
            logger.info(f"  - {key}: {path}")
        
        return exported_files


def create_professional_immersed_boundary():
    """Main function to create professional immersed boundary setup."""
    
    logger.info("="*80)
    logger.info("PROFESSIONAL IMMERSED BOUNDARY FOR CFD")
    logger.info("="*80)
    
    # Load mesh
    mesh_file = Path("../mesh/5_bladed_Propeller.STEP_medium_tetrahedral.msh")
    processor = GmshProcessor(str(mesh_file))
    processor.read_mesh()
    
    # Initialize professional immersed boundary
    ib_professional = ImmersedBoundaryProfessional(processor)
    
    # Create focused domain around object
    domain_info = ib_professional.create_focused_domain(
        boundary_layers=8,
        layer_thickness=None,  # Auto-determine
        resolution_factor=2.0
    )
    
    # Create high-resolution grid
    grid_coords = ib_professional.create_high_resolution_grid(
        domain_info,
        base_resolution=(100, 80, 80),  # High resolution for professional CFD
        near_boundary_factor=1.5
    )
    
    # Compute immersed boundary SDF
    sdf_values = ib_professional.compute_immersed_boundary_sdf(grid_coords)
    
    # Create professional visualizations
    output_dir = Path("output/immersed_boundary_professional")
    visualization_files = ib_professional.visualize_immersed_boundary(
        grid_coords, sdf_values, output_dir, show_original_mesh=True
    )
    
    # Export for JAX-Fluids
    export_files = ib_professional.export_for_jax_fluids(
        grid_coords, sdf_values, output_dir
    )
    
    logger.info("\n" + "="*80)
    logger.info("✓ PROFESSIONAL IMMERSED BOUNDARY COMPLETED")
    logger.info("="*80)
    logger.info("\nFiles created:")
    all_files = {**visualization_files, **export_files}
    for key, path in all_files.items():
        logger.info(f"  - {key}: {path}")
    
    return {
        'grid_coords': grid_coords,
        'sdf_values': sdf_values,
        'domain_info': domain_info,
        'files': all_files
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    results = create_professional_immersed_boundary()
    
    logger.info("\n🎯 Professional immersed boundary ready for CFD!")
    logger.info("   - Clean φ=0 boundary surface")  
    logger.info("   - High-resolution near-field")
    logger.info("   - JAX-Fluids compatible export")
    logger.info("   - Cross-section analysis") 