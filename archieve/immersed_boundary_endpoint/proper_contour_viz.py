#!/usr/bin/env python3
"""
Proper 3D Levelset Contour Visualization

Using professional marching cubes implementations and proper isosurface techniques.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
from typing import Tuple, Optional, List
import time

logger = logging.getLogger(__name__)

try:
    import mcubes
    MCUBES_AVAILABLE = True
    logger.info("PyMCubes available - using professional marching cubes")
except ImportError:
    MCUBES_AVAILABLE = False
    logger.warning("PyMCubes not available - falling back to scikit-image")
    try:
        from skimage import measure
        SKIMAGE_AVAILABLE = True
    except ImportError:
        SKIMAGE_AVAILABLE = False
        logger.error("Neither PyMCubes nor scikit-image available!")

class ProperContourVisualizer:
    """
    Professional 3D levelset contour visualization using proper marching cubes.
    """
    
    def __init__(self, mesh_processor=None):
        """Initialize visualizer."""
        self.mesh_processor = mesh_processor
        logger.info("Initialized ProperContourVisualizer")
    
    def extract_isosurface_professional(self, 
                                      grid_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                      sdf_values: np.ndarray,
                                      level: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract isosurface using professional marching cubes implementation.
        
        Args:
            grid_coords: Grid coordinate arrays (X, Y, Z)
            sdf_values: SDF values on the grid
            level: Isosurface level to extract
            
        Returns:
            (vertices, triangles) of the isosurface
        """
        X, Y, Z = grid_coords
        
        # Determine grid dimensions and spacing
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
        
        # Reshape SDF to 3D grid
        if len(sdf_values.shape) == 1:
            sdf_3d = sdf_values.reshape(nx, ny, nz, order='C')
        else:
            sdf_3d = sdf_values
        
        logger.info(f"Extracting isosurface at level {level:.3f} from {nx}x{ny}x{nz} grid")
        
        if MCUBES_AVAILABLE:
            return self._extract_with_pymcubes(sdf_3d, x_coords, y_coords, z_coords, level)
        elif SKIMAGE_AVAILABLE:
            return self._extract_with_skimage(sdf_3d, x_coords, y_coords, z_coords, level)
        else:
            raise RuntimeError("No marching cubes implementation available!")
    
    def _extract_with_pymcubes(self, sdf_3d, x_coords, y_coords, z_coords, level):
        """Extract isosurface using PyMCubes (professional implementation)."""
        start_time = time.time()
        
        try:
            # PyMCubes expects the grid in a specific format
            vertices, triangles = mcubes.marching_cubes(sdf_3d, level)
            
            # Transform vertices to actual coordinates
            vertices[:, 0] = vertices[:, 0] / (len(x_coords) - 1) * (x_coords.max() - x_coords.min()) + x_coords.min()
            vertices[:, 1] = vertices[:, 1] / (len(y_coords) - 1) * (y_coords.max() - y_coords.min()) + y_coords.min()
            vertices[:, 2] = vertices[:, 2] / (len(z_coords) - 1) * (z_coords.max() - z_coords.min()) + z_coords.min()
            
            elapsed = time.time() - start_time
            logger.info(f"PyMCubes extracted {len(vertices):,} vertices, {len(triangles):,} triangles in {elapsed:.2f}s")
            
            return vertices, triangles
            
        except Exception as e:
            logger.error(f"PyMCubes failed: {e}")
            raise
    
    def _extract_with_skimage(self, sdf_3d, x_coords, y_coords, z_coords, level):
        """Extract isosurface using scikit-image (fallback)."""
        start_time = time.time()
        
        try:
            # Calculate proper spacing
            dx = (x_coords.max() - x_coords.min()) / (len(x_coords) - 1) if len(x_coords) > 1 else 1.0
            dy = (y_coords.max() - y_coords.min()) / (len(y_coords) - 1) if len(y_coords) > 1 else 1.0
            dz = (z_coords.max() - z_coords.min()) / (len(z_coords) - 1) if len(z_coords) > 1 else 1.0
            spacing = (dx, dy, dz)
            
            vertices, faces, normals, values = measure.marching_cubes(
                sdf_3d, 
                level=level, 
                spacing=spacing,
                step_size=1,
                allow_degenerate=False
            )
            
            # Transform to actual coordinates
            vertices[:, 0] += x_coords.min()
            vertices[:, 1] += y_coords.min()
            vertices[:, 2] += z_coords.min()
            
            elapsed = time.time() - start_time
            logger.info(f"scikit-image extracted {len(vertices):,} vertices, {len(faces):,} triangles in {elapsed:.2f}s")
            
            return vertices, faces
            
        except Exception as e:
            logger.error(f"scikit-image marching cubes failed: {e}")
            raise
    
    def plot_professional_3d_contour(self,
                                   grid_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                   sdf_values: np.ndarray,
                                   levels: List[float] = [0.0],
                                   figsize: Tuple[int, int] = (16, 12),
                                   show_mesh: bool = True,
                                   alpha: float = 0.8,
                                   colors: Optional[List[str]] = None) -> plt.Figure:
        """
        Create professional 3D contour visualization.
        
        Args:
            grid_coords: Grid coordinate arrays (X, Y, Z)
            sdf_values: SDF values on the grid
            levels: Contour levels to extract
            figsize: Figure size
            show_mesh: Whether to show original mesh
            alpha: Surface transparency
            colors: Colors for each level
            
        Returns:
            Matplotlib figure
        """
        if colors is None:
            colors = ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Show original mesh if available
        if show_mesh and self.mesh_processor is not None:
            try:
                triangles_array = np.array(self.mesh_processor.surface_triangles)
                vertices = triangles_array.reshape(-1, 3)
                
                # Intelligent subsampling for performance
                if len(vertices) > 5000:
                    indices = np.random.choice(len(vertices), 5000, replace=False)
                    vertices = vertices[indices]
                
                ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                          c='red', alpha=0.3, s=1, label='Original Mesh', marker='.')
                
            except Exception as e:
                logger.warning(f"Could not plot original mesh: {e}")
        
        # Extract and plot each contour level
        surfaces_plotted = 0
        
        for i, level in enumerate(levels):
            try:
                logger.info(f"Processing contour level {level:.3f}...")
                
                # Extract isosurface
                vertices, triangles = self.extract_isosurface_professional(
                    grid_coords, sdf_values, level
                )
                
                if len(triangles) == 0:
                    logger.warning(f"No surface found at level {level:.3f}")
                    continue
                
                # Choose color
                color = colors[i % len(colors)]
                
                # Plot the surface
                if i == 0:  # Main level gets special treatment
                    ax.plot_trisurf(
                        vertices[:, 0], vertices[:, 1], vertices[:, 2],
                        triangles=triangles,
                        alpha=alpha,
                        color=color,
                        shade=True,
                        lightsource=plt.matplotlib.colors.LightSource(azdeg=45, altdeg=65),
                        edgecolor='darkblue' if level == 0.0 else 'none',
                        linewidth=0.1,
                        label=f'Levelset φ={level:.2f}'
                    )
                else:
                    ax.plot_trisurf(
                        vertices[:, 0], vertices[:, 1], vertices[:, 2],
                        triangles=triangles,
                        alpha=alpha * 0.7,  # Make secondary levels more transparent
                        color=color,
                        shade=True,
                        label=f'Levelset φ={level:.2f}'
                    )
                
                surfaces_plotted += 1
                logger.info(f"Plotted level {level:.3f}: {len(vertices):,} vertices, {len(triangles):,} triangles")
                
            except Exception as e:
                logger.error(f"Failed to extract/plot level {level:.3f}: {e}")
        
        if surfaces_plotted == 0:
            logger.error("No surfaces could be extracted!")
            # Add some diagnostic information
            logger.info(f"SDF range: [{sdf_values.min():.3f}, {sdf_values.max():.3f}]")
            logger.info(f"Requested levels: {levels}")
            
            # Show SDF statistics as text
            ax.text2D(0.05, 0.95, f"No surfaces found!\nSDF range: [{sdf_values.min():.3f}, {sdf_values.max():.3f}]",
                     transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        # Enhanced styling
        ax.set_xlabel('X [units]', fontsize=12)
        ax.set_ylabel('Y [units]', fontsize=12)
        ax.set_zlabel('Z [units]', fontsize=12)
        
        title = f'Professional 3D Levelset Visualization'
        if len(levels) == 1:
            title += f' (φ = {levels[0]:.2f})'
        ax.set_title(title, fontsize=14, pad=20)
        
        # Optimal viewing angle for complex shapes
        ax.view_init(elev=20, azim=45)
        
        # Equal aspect ratio
        self._set_equal_aspect_3d(ax, grid_coords)
        
        # Better styling
        ax.grid(True, alpha=0.3)
        
        # Remove pane backgrounds for cleaner look
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
        if surfaces_plotted > 0 or show_mesh:
            ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_multiple_levelsets(self,
                              grid_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
                              sdf_values: np.ndarray,
                              num_levels: int = 5,
                              level_range: Optional[Tuple[float, float]] = None,
                              figsize: Tuple[int, int] = (18, 12)) -> plt.Figure:
        """
        Plot multiple levelset contours to show the 3D field structure.
        
        Args:
            grid_coords: Grid coordinate arrays
            sdf_values: SDF values
            num_levels: Number of contour levels
            level_range: (min_level, max_level) or None for auto
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if level_range is None:
            sdf_min, sdf_max = sdf_values.min(), sdf_values.max()
            # Focus on range around zero
            abs_max = max(abs(sdf_min), abs(sdf_max))
            level_range = (-abs_max * 0.3, abs_max * 0.3)
        
        levels = np.linspace(level_range[0], level_range[1], num_levels)
        
        # Ensure zero is included
        if not np.any(np.abs(levels) < 1e-6):
            levels = np.append(levels, 0.0)
            levels = np.sort(levels)
        
        logger.info(f"Plotting {len(levels)} levelsets: {levels}")
        
        return self.plot_professional_3d_contour(
            grid_coords, sdf_values, 
            levels=levels.tolist(),
            figsize=figsize,
            show_mesh=True,
            alpha=0.6
        )
    
    def _set_equal_aspect_3d(self, ax: Axes3D, grid_coords: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Set equal aspect ratio for 3D plot."""
        X, Y, Z = grid_coords
        
        # Get actual data bounds
        x_min, x_max = X.min(), X.max()
        y_min, y_max = Y.min(), Y.max()
        z_min, z_max = Z.min(), Z.max()
        
        # Calculate ranges
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        # Find maximum range
        max_range = max(x_range, y_range, z_range)
        
        # Calculate centers
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        
        # Set equal aspect with padding
        padding = max_range * 0.1
        half_range = max_range / 2 + padding
        
        ax.set_xlim(x_center - half_range, x_center + half_range)
        ax.set_ylim(y_center - half_range, y_center + half_range)
        ax.set_zlim(z_center - half_range, z_center + half_range)
    
    def export_isosurface_mesh(self,
                             grid_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
                             sdf_values: np.ndarray,
                             level: float = 0.0,
                             filename: str = "isosurface.obj") -> bool:
        """
        Export isosurface as mesh file.
        
        Args:
            grid_coords: Grid coordinates
            sdf_values: SDF values
            level: Isosurface level
            filename: Output filename (.obj, .ply, .stl supported)
            
        Returns:
            True if successful
        """
        try:
            vertices, triangles = self.extract_isosurface_professional(
                grid_coords, sdf_values, level
            )
            
            if len(triangles) == 0:
                logger.error("No surface to export!")
                return False
            
            # Export based on file extension
            ext = filename.lower().split('.')[-1]
            
            if ext == 'obj':
                self._export_obj(vertices, triangles, filename)
            elif MCUBES_AVAILABLE and ext in ['ply', 'dae']:
                mcubes.export_mesh(vertices, triangles, filename)
            else:
                logger.error(f"Unsupported format: {ext}")
                return False
            
            logger.info(f"Exported isosurface to {filename}: {len(vertices):,} vertices, {len(triangles):,} triangles")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export isosurface: {e}")
            return False
    
    def _export_obj(self, vertices, triangles, filename):
        """Export mesh as OBJ file."""
        with open(filename, 'w') as f:
            f.write("# OBJ file generated by ProperContourVisualizer\n")
            
            # Write vertices
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            # Write faces (OBJ uses 1-based indexing)
            for triangle in triangles:
                f.write(f"f {triangle[0]+1} {triangle[1]+1} {triangle[2]+1}\n") 