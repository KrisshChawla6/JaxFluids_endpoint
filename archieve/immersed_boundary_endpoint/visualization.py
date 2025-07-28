"""
Visualization Tools

This module provides visualization capabilities for signed distance functions,
contour plots, and mesh geometry validation.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from typing import Tuple, Optional, List, Dict, Any
import logging
try:
    from .sdf_generator import SignedDistanceFunction
    from .mesh_processor import GmshProcessor
except ImportError:
    from sdf_generator import SignedDistanceFunction
    from mesh_processor import GmshProcessor

logger = logging.getLogger(__name__)


class SDFVisualizer:
    """
    Visualization tools for signed distance functions and mesh geometry.
    
    Provides:
    - 2D/3D SDF contour plots
    - Mesh geometry visualization  
    - Cross-sectional views
    - Validation plots
    """
    
    def __init__(self, 
                 mesh_processor: Optional[GmshProcessor] = None,
                 sdf_generator: Optional[SignedDistanceFunction] = None):
        """
        Initialize visualizer.
        
        Args:
            mesh_processor: Mesh processor for geometry visualization
            sdf_generator: SDF generator for distance field visualization
        """
        self.mesh_processor = mesh_processor
        self.sdf_generator = sdf_generator
        
        # Setup custom colormap for SDF visualization
        self._setup_colormaps()
        
        logger.info("Initialized SDFVisualizer")
    
    def _setup_colormaps(self) -> None:
        """Setup custom colormaps for SDF visualization."""
        # Create a diverging colormap for SDF (negative=blue, zero=white, positive=red)
        colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#ffffff',
                 '#fdbf6f', '#ff7f00', '#e31a1c', '#b10026']
        n_bins = 256
        self.sdf_cmap = LinearSegmentedColormap.from_list('sdf', colors, N=n_bins)
    
    def plot_mesh_geometry(self, 
                          figsize: Tuple[int, int] = (12, 8),
                          show_triangles: bool = True,
                          show_normals: bool = False,
                          alpha: float = 0.7) -> plt.Figure:
        """
        Plot the mesh geometry in 3D.
        
        Args:
            figsize: Figure size
            show_triangles: Whether to show triangle wireframe
            show_normals: Whether to show triangle normals
            alpha: Transparency level
            
        Returns:
            Matplotlib figure
        """
        if self.mesh_processor is None:
            raise ValueError("Mesh processor not provided")
        
        if not self.mesh_processor.surface_triangles:
            raise ValueError("No surface triangles found. Call read_mesh() first.")
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Collect all vertices for plotting
        all_vertices = []
        all_triangles = []
        
        # Plot surface triangles
        for tri_idx, triangle in enumerate(self.mesh_processor.surface_triangles):
            if show_triangles:
                # Plot triangle edges
                for i in range(3):
                    start = triangle[i]
                    end = triangle[(i + 1) % 3]
                    ax.plot([start[0], end[0]], 
                           [start[1], end[1]], 
                           [start[2], end[2]], 'b-', alpha=alpha, linewidth=0.5)
            
            # Collect vertices for surface plotting
            all_vertices.extend(triangle)
        
        # Plot the entire surface as one mesh
        if all_vertices and not show_triangles:
            vertices = np.array(all_vertices)
            # Sample vertices for cleaner visualization (avoid too many points)
            if len(vertices) > 3000:
                step = len(vertices) // 3000
                vertices = vertices[::step]
            
            # Create scatter plot for the surface
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                      c='lightblue', alpha=alpha, s=1)
        
        # Plot normals if requested
        if show_normals and self.sdf_generator is not None:
            for i, (center, normal) in enumerate(zip(
                self.sdf_generator.triangle_centers,
                self.sdf_generator.triangle_normals
            )):
                if i % 10 == 0:  # Show every 10th normal to avoid clutter
                    end_point = center + normal * 0.1  # Scale normal for visibility
                    ax.quiver(center[0], center[1], center[2],
                             normal[0], normal[1], normal[2],
                             color='red', alpha=0.8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Mesh Geometry')
        
        # Set equal aspect ratio
        max_range = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]).max()
        ax.set_xlim(-max_range/2, max_range/2)
        ax.set_ylim(-max_range/2, max_range/2)
        ax.set_zlim(-max_range/2, max_range/2)
        
        plt.tight_layout()
        return fig
    
    def plot_sdf_2d_slice(self,
                         grid_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
                         sdf_values: np.ndarray,
                         slice_axis: str = 'z',
                         slice_position: float = 0.0,
                         figsize: Tuple[int, int] = (10, 8),
                         contour_levels: int = 20,
                         show_zero_contour: bool = True) -> plt.Figure:
        """
        Plot 2D slice of SDF with contour lines.
        
        Args:
            grid_coords: Grid coordinates (X, Y, Z)
            sdf_values: SDF values on grid
            slice_axis: Axis to slice along ('x', 'y', or 'z')
            slice_position: Position of slice
            figsize: Figure size
            contour_levels: Number of contour levels
            show_zero_contour: Whether to highlight zero contour
            
        Returns:
            Matplotlib figure
        """
        if self.sdf_generator is None:
            raise ValueError("SDF generator not provided")
        
        X, Y, Z = grid_coords
        
        # Find slice index
        if slice_axis.lower() == 'z':
            axis_coords = Z[0, 0, :]
            slice_idx = np.argmin(np.abs(axis_coords - slice_position))
            slice_x, slice_y = X[:, :, slice_idx], Y[:, :, slice_idx]
            slice_sdf = sdf_values[:, :, slice_idx]
            xlabel, ylabel = 'X', 'Y'
        elif slice_axis.lower() == 'y':
            axis_coords = Y[0, :, 0]
            slice_idx = np.argmin(np.abs(axis_coords - slice_position))
            slice_x, slice_y = X[:, slice_idx, :], Z[:, slice_idx, :]
            slice_sdf = sdf_values[:, slice_idx, :]
            xlabel, ylabel = 'X', 'Z'
        elif slice_axis.lower() == 'x':
            axis_coords = X[:, 0, 0]
            slice_idx = np.argmin(np.abs(axis_coords - slice_position))
            slice_x, slice_y = Y[slice_idx, :, :], Z[slice_idx, :, :]
            slice_sdf = sdf_values[slice_idx, :, :]
            xlabel, ylabel = 'Y', 'Z'
        else:
            raise ValueError("slice_axis must be 'x', 'y', or 'z'")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Contour plot
        contours = ax1.contour(slice_x, slice_y, slice_sdf, 
                              levels=contour_levels, colors='black', alpha=0.6)
        ax1.clabel(contours, inline=True, fontsize=8)
        
        # Filled contour plot
        contourf = ax1.contourf(slice_x, slice_y, slice_sdf, 
                               levels=contour_levels, cmap=self.sdf_cmap)
        
        if show_zero_contour:
            ax1.contour(slice_x, slice_y, slice_sdf, levels=[0], 
                       colors='white', linewidths=3)
        
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title(f'SDF Contours - {slice_axis.upper()}={slice_position:.3f}')
        ax1.set_aspect('equal')
        
        # Colorbar
        cbar = plt.colorbar(contourf, ax=ax1)
        cbar.set_label('Signed Distance')
        
        # 3D surface plot
        ax2 = fig.add_subplot(122, projection='3d')
        surf = ax2.plot_surface(slice_x, slice_y, slice_sdf, 
                               cmap=self.sdf_cmap, alpha=0.8)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.set_zlabel('SDF Value')
        ax2.set_title('SDF Surface')
        
        plt.tight_layout()
        return fig
    
    def plot_sdf_cross_sections(self,
                               grid_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
                               sdf_values: np.ndarray,
                               num_slices: int = 3,
                               figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot multiple cross-sections of the SDF.
        
        Args:
            grid_coords: Grid coordinates
            sdf_values: SDF values
            num_slices: Number of slices per axis
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        X, Y, Z = grid_coords
        
        fig, axes = plt.subplots(3, num_slices, figsize=figsize)
        
        # Z slices
        z_positions = np.linspace(Z.min(), Z.max(), num_slices)
        for i, z_pos in enumerate(z_positions):
            z_idx = np.argmin(np.abs(Z[0, 0, :] - z_pos))
            slice_sdf = sdf_values[:, :, z_idx]
            
            im = axes[0, i].contourf(X[:, :, z_idx], Y[:, :, z_idx], slice_sdf, 
                                    levels=20, cmap=self.sdf_cmap)
            axes[0, i].contour(X[:, :, z_idx], Y[:, :, z_idx], slice_sdf, 
                              levels=[0], colors='white', linewidths=2)
            axes[0, i].set_title(f'Z = {z_pos:.3f}')
            axes[0, i].set_aspect('equal')
        
        # Y slices
        y_positions = np.linspace(Y.min(), Y.max(), num_slices)
        for i, y_pos in enumerate(y_positions):
            y_idx = np.argmin(np.abs(Y[0, :, 0] - y_pos))
            slice_sdf = sdf_values[:, y_idx, :]
            
            im = axes[1, i].contourf(X[:, y_idx, :], Z[:, y_idx, :], slice_sdf,
                                    levels=20, cmap=self.sdf_cmap)
            axes[1, i].contour(X[:, y_idx, :], Z[:, y_idx, :], slice_sdf,
                              levels=[0], colors='white', linewidths=2)
            axes[1, i].set_title(f'Y = {y_pos:.3f}')
            axes[1, i].set_aspect('equal')
        
        # X slices
        x_positions = np.linspace(X.min(), X.max(), num_slices)
        for i, x_pos in enumerate(x_positions):
            x_idx = np.argmin(np.abs(X[:, 0, 0] - x_pos))
            slice_sdf = sdf_values[x_idx, :, :]
            
            im = axes[2, i].contourf(Y[x_idx, :, :], Z[x_idx, :, :], slice_sdf,
                                    levels=20, cmap=self.sdf_cmap)
            axes[2, i].contour(Y[x_idx, :, :], Z[x_idx, :, :], slice_sdf,
                              levels=[0], colors='white', linewidths=2)
            axes[2, i].set_title(f'X = {x_pos:.3f}')
            axes[2, i].set_aspect('equal')
        
        # Add colorbar
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='Signed Distance')
        
        plt.tight_layout()
        return fig
    
    def plot_validation_comparison(self,
                                  test_points: np.ndarray,
                                  analytical_sdf: np.ndarray,
                                  computed_sdf: np.ndarray,
                                  figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot validation comparison between analytical and computed SDF.
        
        Args:
            test_points: Test point coordinates
            analytical_sdf: Analytical SDF values
            computed_sdf: Computed SDF values
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Scatter plot comparison
        axes[0, 0].scatter(analytical_sdf, computed_sdf, alpha=0.6)
        axes[0, 0].plot([analytical_sdf.min(), analytical_sdf.max()],
                       [analytical_sdf.min(), analytical_sdf.max()], 'r--', label='Perfect match')
        axes[0, 0].set_xlabel('Analytical SDF')
        axes[0, 0].set_ylabel('Computed SDF')
        axes[0, 0].set_title('SDF Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error distribution
        error = computed_sdf - analytical_sdf
        axes[0, 1].hist(error, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Error Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error vs distance
        axes[1, 0].scatter(np.abs(analytical_sdf), np.abs(error), alpha=0.6)
        axes[1, 0].set_xlabel('|Analytical SDF|')
        axes[1, 0].set_ylabel('|Error|')
        axes[1, 0].set_title('Error vs Distance from Surface')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Statistics
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))
        max_error = np.max(np.abs(error))
        
        stats_text = f"""
        RMSE: {rmse:.6f}
        MAE: {mae:.6f}
        Max Error: {max_error:.6f}
        Num Points: {len(test_points)}
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 1].set_title('Validation Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_animation_frames(self,
                               grid_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
                               sdf_values: np.ndarray,
                               slice_axis: str = 'z',
                               output_dir: str = 'animation_frames') -> List[str]:
        """
        Create animation frames for SDF visualization.
        
        Args:
            grid_coords: Grid coordinates
            sdf_values: SDF values
            slice_axis: Axis to slice along
            output_dir: Output directory for frames
            
        Returns:
            List of frame filenames
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        X, Y, Z = grid_coords
        
        if slice_axis.lower() == 'z':
            axis_coords = Z[0, 0, :]
        elif slice_axis.lower() == 'y':
            axis_coords = Y[0, :, 0]
        elif slice_axis.lower() == 'x':
            axis_coords = X[:, 0, 0]
        
        frame_files = []
        
        for i, pos in enumerate(axis_coords[::2]):  # Every other slice
            fig = self.plot_sdf_2d_slice(grid_coords, sdf_values, 
                                        slice_axis, pos, figsize=(8, 6))
            
            filename = os.path.join(output_dir, f'frame_{i:04d}.png')
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            frame_files.append(filename)
        
        logger.info(f"Created {len(frame_files)} animation frames in {output_dir}")
        return frame_files
    
    def save_all_plots(self,
                      grid_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
                      sdf_values: np.ndarray,
                      output_prefix: str = 'sdf_visualization') -> Dict[str, str]:
        """
        Save all visualization plots to files.
        
        Args:
            grid_coords: Grid coordinates
            sdf_values: SDF values
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary of plot type to filename
        """
        saved_files = {}
        
        # Mesh geometry
        if self.mesh_processor is not None:
            try:
                fig = self.plot_mesh_geometry()
                filename = f"{output_prefix}_mesh_geometry.png"
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                saved_files['mesh_geometry'] = filename
            except Exception as e:
                logger.warning(f"Could not save mesh geometry plot: {e}")
        
        # Cross sections
        try:
            fig = self.plot_sdf_cross_sections(grid_coords, sdf_values)
            filename = f"{output_prefix}_cross_sections.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_files['cross_sections'] = filename
        except Exception as e:
            logger.warning(f"Could not save cross sections plot: {e}")
        
        # Individual slices
        for axis in ['x', 'y', 'z']:
            try:
                fig = self.plot_sdf_2d_slice(grid_coords, sdf_values, 
                                           slice_axis=axis, slice_position=0.0)
                filename = f"{output_prefix}_{axis}_slice.png"
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                saved_files[f'{axis}_slice'] = filename
            except Exception as e:
                logger.warning(f"Could not save {axis} slice plot: {e}")
        
        logger.info(f"Saved {len(saved_files)} visualization plots")
        return saved_files 

    def plot_sdf_3d_contour(self,
                           grid_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
                           sdf_values: np.ndarray,
                           contour_level: float = 0.0,
                           figsize: Tuple[int, int] = (16, 12),
                           alpha: float = 0.8,
                           show_mesh: bool = True,
                           smooth_surface: bool = True,
                           high_quality: bool = True) -> plt.Figure:
        """
        Plot high-quality 3D SDF contour surface with multiple visualization techniques.
        
        Args:
            grid_coords: Grid coordinate arrays (X, Y, Z)
            sdf_values: SDF values on the grid
            contour_level: SDF level to visualize (0.0 = boundary)
            figsize: Figure size
            alpha: Transparency
            show_mesh: Whether to show original mesh
            smooth_surface: Whether to apply surface smoothing
            high_quality: Whether to use high-quality rendering
            
        Returns:
            Matplotlib figure
        """
        from skimage import measure
        from scipy.ndimage import gaussian_filter
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Show original mesh if requested (with better sampling)
        if show_mesh and self.mesh_processor is not None:
            try:
                # Convert surface triangles to numpy array if needed
                if isinstance(self.mesh_processor.surface_triangles, list):
                    triangles_array = np.array(self.mesh_processor.surface_triangles)
                else:
                    triangles_array = self.mesh_processor.surface_triangles
                
                # Sample vertices more intelligently
                vertices = triangles_array.reshape(-1, 3)
                if len(vertices) > 5000:
                    # Use random sampling instead of regular stepping for better coverage
                    indices = np.random.choice(len(vertices), 5000, replace=False)
                    vertices = vertices[indices]
                
                ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                          c='red', alpha=0.2, s=1, label='Original Mesh', marker='.')
            except Exception as e:
                logger.warning(f"Could not plot original mesh: {e}")
        
        # Extract isosurface using improved marching cubes
        try:
            # Get grid coordinates and determine dimensions
            X, Y, Z = grid_coords
            
            # Handle different grid formats
            if len(X.shape) == 3:
                # Already in 3D grid format
                nx, ny, nz = X.shape
                x_coords = X[:, 0, 0]
                y_coords = Y[0, :, 0]
                z_coords = Z[0, 0, :]
            else:
                # Flatten and get unique coordinates
                x_coords = np.unique(X.ravel())
                y_coords = np.unique(Y.ravel())
                z_coords = np.unique(Z.ravel())
                nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
            
            logger.info(f"Grid dimensions: {nx} x {ny} x {nz} = {nx*ny*nz:,} points")
            
            # Reshape SDF values to 3D grid with proper ordering
            if len(sdf_values.shape) == 1:
                # Reshape with proper ordering (important!)
                sdf_3d = sdf_values.reshape(nx, ny, nz, order='C')
            else:
                sdf_3d = sdf_values
            
            # Apply smoothing if requested (helps with visualization quality)
            if smooth_surface and high_quality:
                logger.info("Applying Gaussian smoothing to SDF field...")
                sigma = min(2.0, max(0.5, min(nx, ny, nz) / 50))  # Adaptive smoothing
                sdf_3d = gaussian_filter(sdf_3d, sigma=sigma)
            
            # Calculate proper spacing
            dx = (x_coords.max() - x_coords.min()) / (nx - 1) if nx > 1 else 1.0
            dy = (y_coords.max() - y_coords.min()) / (ny - 1) if ny > 1 else 1.0
            dz = (z_coords.max() - z_coords.min()) / (nz - 1) if nz > 1 else 1.0
            spacing = (dx, dy, dz)
            
            logger.info(f"Grid spacing: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}")
            
            # Check SDF value range around contour level
            near_contour = np.abs(sdf_3d - contour_level) < np.std(sdf_3d) * 2
            if np.sum(near_contour) < 100:
                logger.warning(f"Very few points near contour level {contour_level}")
                logger.info(f"SDF range: [{sdf_3d.min():.3f}, {sdf_3d.max():.3f}]")
                logger.info(f"SDF std: {sdf_3d.std():.3f}")
            
            # Extract isosurface with marching cubes
            logger.info(f"Extracting isosurface at level {contour_level}...")
            verts, faces, normals, values = measure.marching_cubes(
                sdf_3d, 
                level=contour_level, 
                spacing=spacing,
                step_size=1 if high_quality else 2,  # Higher resolution for quality
                allow_degenerate=False
            )
            
            # Transform vertices to actual coordinates
            verts[:, 0] += x_coords.min()
            verts[:, 1] += y_coords.min()
            verts[:, 2] += z_coords.min()
            
            logger.info(f"Generated surface with {len(verts):,} vertices and {len(faces):,} faces")
            
            # Plot the isosurface with better lighting and colors
            if len(faces) > 0:
                # Use plot_trisurf for better rendering
                surf = ax.plot_trisurf(
                    verts[:, 0], verts[:, 1], verts[:, 2], 
                    triangles=faces, 
                    alpha=alpha, 
                    color='blue',
                    shade=True,
                    lightsource=plt.matplotlib.colors.LightSource(azdeg=45, altdeg=65),
                    edgecolor='none' if high_quality else 'black',
                    linewidth=0.1,
                    label=f'SDF Contour (φ={contour_level})'
                )
                
                # Add some visual enhancements
                if high_quality:
                    # Add subtle wireframe on a subset of triangles
                    if len(faces) > 1000:
                        step = len(faces) // 500  # Show ~500 wireframe triangles
                        wire_faces = faces[::step]
                        ax.plot_trisurf(
                            verts[:, 0], verts[:, 1], verts[:, 2],
                            triangles=wire_faces,
                            alpha=0.1, color='darkblue', shade=False
                        )
            else:
                logger.warning("No triangular faces found in isosurface!")
                
        except Exception as e:
            logger.error(f"Marching cubes failed: {e}")
            logger.info("Falling back to scatter plot visualization...")
            
            # Enhanced fallback: show SDF values as scatter plot with better filtering
            points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            
            # Show points near the contour level with adaptive threshold
            threshold = max(0.1, np.std(sdf_values) * 0.3)
            mask = np.abs(sdf_values - contour_level) < threshold
            near_contour_points = points[mask]
            near_contour_values = sdf_values[mask]
            
            if len(near_contour_points) > 0:
                # Subsample if too many points
                if len(near_contour_points) > 10000:
                    indices = np.random.choice(len(near_contour_points), 10000, replace=False)
                    near_contour_points = near_contour_points[indices]
                    near_contour_values = near_contour_values[indices]
                
                scatter = ax.scatter(
                    near_contour_points[:, 0], 
                    near_contour_points[:, 1], 
                    near_contour_points[:, 2],
                    c=near_contour_values, 
                    cmap='RdBu_r', 
                    alpha=alpha, 
                    s=3,
                    label=f'Near φ={contour_level}'
                )
                plt.colorbar(scatter, ax=ax, label='SDF Value', shrink=0.6)
                logger.info(f"Showing {len(near_contour_points):,} points near contour level")
            else:
                logger.warning("No points found near contour level!")
        
        # Enhanced formatting and styling
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(f'High-Quality 3D SDF Contour Surface (φ = {contour_level})', fontsize=14, pad=20)
        
        # Better lighting and viewing angle
        ax.view_init(elev=20, azim=45)
        
        # Equal aspect ratio with better bounds
        self._set_equal_aspect_3d_enhanced(ax, grid_coords)
        
        # Add grid and better styling
        ax.grid(True, alpha=0.3)
        
        if show_mesh or 'scatter' in locals():
            ax.legend(loc='upper right', fontsize=10)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def _set_equal_aspect_3d_enhanced(self, ax: Axes3D, grid_coords: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """
        Set enhanced equal aspect ratio for a 3D plot with better bounds.
        
        Args:
            ax: Matplotlib Axes3D object
            grid_coords: Grid coordinate arrays (X, Y, Z)
        """
        X, Y, Z = grid_coords
        
        # Get actual data bounds
        x_min, x_max = X.min(), X.max()
        y_min, y_max = Y.min(), Y.max()
        z_min, z_max = Z.min(), Z.max()
        
        # Calculate ranges
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        # Find the maximum range
        max_range = max(x_range, y_range, z_range)
        
        # Calculate centers
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        
        # Set equal aspect with some padding
        padding = max_range * 0.1
        half_range = max_range / 2 + padding
        
        ax.set_xlim(x_center - half_range, x_center + half_range)
        ax.set_ylim(y_center - half_range, y_center + half_range)
        ax.set_zlim(z_center - half_range, z_center + half_range)

    def plot_sdf_advanced_contour(self,
                                 grid_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                 sdf_values: np.ndarray,
                                 main_level: float = 0.0,
                                 additional_levels: List[float] = None,
                                 figsize: Tuple[int, int] = (18, 12),
                                 show_mesh: bool = True) -> plt.Figure:
        """
        Advanced SDF visualization with multiple contour levels and enhanced surface quality.
        
        Args:
            grid_coords: Grid coordinate arrays (X, Y, Z)
            sdf_values: SDF values on the grid
            main_level: Primary contour level (usually 0.0 for boundary)
            additional_levels: Additional contour levels to show
            figsize: Figure size
            show_mesh: Whether to show original mesh
            
        Returns:
            Matplotlib figure with advanced visualization
        """
        from skimage import measure
        from scipy.ndimage import gaussian_filter
        import matplotlib.colors as mcolors
        
        if additional_levels is None:
            # Automatically choose meaningful levels
            sdf_std = np.std(sdf_values)
            additional_levels = [main_level - sdf_std*0.5, main_level + sdf_std*0.5]
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Show original mesh with better transparency
        if show_mesh and self.mesh_processor is not None:
            try:
                triangles_array = np.array(self.mesh_processor.surface_triangles)
                vertices = triangles_array.reshape(-1, 3)
                
                # Intelligent subsampling
                if len(vertices) > 3000:
                    indices = np.random.choice(len(vertices), 3000, replace=False)
                    vertices = vertices[indices]
                
                ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                          c='red', alpha=0.15, s=0.8, label='Original Mesh', marker='.')
            except Exception as e:
                logger.warning(f"Could not plot original mesh: {e}")
        
        # Prepare grid data
        X, Y, Z = grid_coords
        if len(X.shape) == 3:
            nx, ny, nz = X.shape
            x_coords = X[:, 0, 0]
            y_coords = Y[0, :, 0] 
            z_coords = Z[0, 0, :]
        else:
            x_coords = np.unique(X.ravel())
            y_coords = np.unique(Y.ravel())
            z_coords = np.unique(Z.ravel())
            nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
        
        # Reshape and smooth SDF
        if len(sdf_values.shape) == 1:
            sdf_3d = sdf_values.reshape(nx, ny, nz, order='C')
        else:
            sdf_3d = sdf_values
        
        # Apply moderate smoothing for better surface quality
        sigma = min(1.5, max(0.8, min(nx, ny, nz) / 60))
        sdf_3d_smooth = gaussian_filter(sdf_3d, sigma=sigma)
        
        # Calculate spacing
        dx = (x_coords.max() - x_coords.min()) / (nx - 1) if nx > 1 else 1.0
        dy = (y_coords.max() - y_coords.min()) / (ny - 1) if ny > 1 else 1.0
        dz = (z_coords.max() - z_coords.min()) / (nz - 1) if nz > 1 else 1.0
        spacing = (dx, dy, dz)
        
        # Colors for different contour levels
        colors = ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
        alphas = [0.9, 0.6, 0.4, 0.4, 0.6, 0.9]
        
        # Plot multiple contour levels
        all_levels = [main_level] + additional_levels
        surfaces_plotted = 0
        
        for i, level in enumerate(all_levels):
            try:
                logger.info(f"Extracting contour at level {level:.3f}...")
                
                verts, faces, normals, values = measure.marching_cubes(
                    sdf_3d_smooth,
                    level=level,
                    spacing=spacing,
                    step_size=1,
                    allow_degenerate=False
                )
                
                # Transform to actual coordinates
                verts[:, 0] += x_coords.min()
                verts[:, 1] += y_coords.min()
                verts[:, 2] += z_coords.min()
                
                if len(faces) > 0:
                    color = colors[i % len(colors)]
                    alpha = alphas[i % len(alphas)]
                    
                    # Main surface gets special treatment
                    if i == 0:  # Main level
                        ax.plot_trisurf(
                            verts[:, 0], verts[:, 1], verts[:, 2],
                            triangles=faces,
                            alpha=alpha,
                            color=color,
                            shade=True,
                            lightsource=plt.matplotlib.colors.LightSource(azdeg=45, altdeg=65),
                            edgecolor='darkblue',
                            linewidth=0.05,
                            label=f'Main Boundary (φ={level:.2f})'
                        )
                    else:
                        ax.plot_trisurf(
                            verts[:, 0], verts[:, 1], verts[:, 2],
                            triangles=faces,
                            alpha=alpha,
                            color=color,
                            shade=True,
                            label=f'Contour φ={level:.2f}'
                        )
                    
                    surfaces_plotted += 1
                    logger.info(f"Plotted contour {level:.3f}: {len(verts):,} vertices, {len(faces):,} faces")
                
            except Exception as e:
                logger.warning(f"Could not extract contour at level {level}: {e}")
        
        if surfaces_plotted == 0:
            logger.warning("No surfaces could be extracted! Falling back to scatter plot...")
            # Fallback visualization
            points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            threshold = np.std(sdf_values) * 0.4
            
            for level in all_levels[:2]:  # Show first two levels only
                mask = np.abs(sdf_values - level) < threshold
                if np.sum(mask) > 0:
                    level_points = points[mask]
                    level_values = sdf_values[mask]
                    
                    if len(level_points) > 5000:
                        indices = np.random.choice(len(level_points), 5000, replace=False)
                        level_points = level_points[indices]
                        level_values = level_values[indices]
                    
                    scatter = ax.scatter(
                        level_points[:, 0], level_points[:, 1], level_points[:, 2],
                        c=level_values,
                        cmap='RdBu_r',
                        alpha=0.6,
                        s=2,
                        label=f'Near φ={level:.2f}'
                    )
        
        # Enhanced styling
        ax.set_xlabel('X [units]', fontsize=12)
        ax.set_ylabel('Y [units]', fontsize=12)
        ax.set_zlabel('Z [units]', fontsize=12)
        ax.set_title('Advanced Multi-Level SDF Visualization', fontsize=16, pad=25)
        
        # Optimal viewing angle for propeller-like objects
        ax.view_init(elev=25, azim=30)
        
        # Enhanced equal aspect ratio
        self._set_equal_aspect_3d_enhanced(ax, grid_coords)
        
        # Better grid and styling
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Make pane edges more subtle
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        # Legend
        ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=10)
        
        plt.tight_layout()
        return fig 