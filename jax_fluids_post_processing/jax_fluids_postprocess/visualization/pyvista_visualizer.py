#!/usr/bin/env python3
"""
PyVista-based 3D Visualization for JAX-Fluids Data

This module provides high-level functions to create 3D visualizations using PyVista.
Supports volume rendering, isosurfaces, streamlines, and more.

References:
- https://docs.pyvista.org/
"""

import numpy as np
import pyvista as pv
from typing import Dict, List, Tuple, Optional, Union
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..io.h5_reader import JAXFluidsReader
from ..core.data_extractor import DataExtractor


class PyVistaVisualizer:
    """PyVista-based visualizer for JAX-Fluids data."""
    
    def __init__(self, reader: JAXFluidsReader, extractor: DataExtractor):
        """
        Initialize the visualizer.
        
        Args:
            reader: JAXFluidsReader instance
            extractor: DataExtractor instance
        """
        self.reader = reader
        self.extractor = extractor
        
        # Configure PyVista
        pv.set_plot_theme('document')  # Clean white background
        
        # Check if we have grid information
        if self.reader.grid_info is None:
            raise ValueError("Grid information not available from reader")
        
        self.grid_shape = self.reader.grid_info.get('shape')
        if self.grid_shape is None:
            raise ValueError("Grid shape not available")
    
    def create_structured_grid(self, file_index: int = 0, 
                             variables: Optional[List[str]] = None) -> pv.StructuredGrid:
        """
        Create a PyVista StructuredGrid from JAX-Fluids data.
        
        Args:
            file_index: Index of the file to read
            variables: List of variables to include in the grid
            
        Returns:
            PyVista StructuredGrid with the flow data
        """
        # Get coordinate arrays
        if self.extractor.X is None or self.extractor.Y is None or self.extractor.Z is None:
            raise ValueError("Coordinate arrays not available")
        
        X, Y, Z = self.extractor.X, self.extractor.Y, self.extractor.Z
        
        # Create structured grid
        grid = pv.StructuredGrid(X, Y, Z)
        
        # Add primitive variables
        primitives = self.reader.read_primitives(file_index, variables)
        
        for var_name, var_data in primitives.items():
            if var_data.ndim == 3 and var_data.shape == self.grid_shape:
                # Flatten for PyVista (uses VTK ordering)
                grid[var_name] = var_data.flatten(order='F')
        
        # Add derived quantities
        self._add_derived_quantities(grid, file_index)
        
        return grid
    
    def _add_derived_quantities(self, grid: pv.StructuredGrid, file_index: int):
        """Add derived quantities to the grid."""
        
        # Velocity magnitude
        vel_mag = self.extractor.compute_velocity_magnitude(file_index)
        if vel_mag is not None:
            grid['velocity_magnitude'] = vel_mag.flatten(order='F')
        
        # Vorticity magnitude
        vort_mag = self.extractor.compute_vorticity_magnitude(file_index)
        if vort_mag is not None:
            grid['vorticity_magnitude'] = vort_mag.flatten(order='F')
        
        # Q-criterion
        q_criterion = self.extractor.compute_q_criterion(file_index)
        if q_criterion is not None:
            grid['q_criterion'] = q_criterion.flatten(order='F')
        
        # Velocity vectors (if components available)
        velocity = self.extractor.extract_velocity_components(file_index)
        if all(comp in velocity for comp in ['u', 'v', 'w']):
            u, v, w = velocity['u'], velocity['v'], velocity['w']
            
            # Ensure arrays are the right shape and type
            u_flat = np.asarray(u).flatten(order='F')
            v_flat = np.asarray(v).flatten(order='F')
            w_flat = np.asarray(w).flatten(order='F')
            
            # Create vector array
            vectors = np.column_stack([u_flat, v_flat, w_flat])
            grid['velocity_vectors'] = vectors
    
    def plot_volume_rendering(self, file_index: int = 0, 
                            variable: str = 'velocity_magnitude',
                            opacity_transfer: Optional[List[Tuple[float, float]]] = None,
                            color_transfer: Optional[List[Tuple[float, str]]] = None,
                            screenshot_path: Optional[str] = None) -> pv.Plotter:
        """
        Create a volume rendering visualization.
        
        Args:
            file_index: Index of the file to read
            variable: Variable to visualize
            opacity_transfer: List of (value, opacity) tuples for transfer function
            color_transfer: List of (value, color) tuples for color mapping
            screenshot_path: Path to save screenshot (optional)
            
        Returns:
            PyVista plotter object
        """
        grid = self.create_structured_grid(file_index)
        
        if variable not in grid.array_names:
            raise ValueError(f"Variable '{variable}' not found in grid. Available: {grid.array_names}")
        
        # Create plotter
        plotter = pv.Plotter(off_screen=True)
        
        # Set up volume rendering
        if opacity_transfer is None:
            # Default opacity transfer function
            data_range = grid[variable].min(), grid[variable].max()
            
            # Handle uniform data (no variation)
            if data_range[1] - data_range[0] < 1e-10:
                # Uniform data - create simple opacity
                mid_val = data_range[0]
                opacity_transfer = [
                    (mid_val - 0.1, 0.0),
                    (mid_val, 0.3),
                    (mid_val + 0.1, 0.3)
                ]
            else:
                # Varying data - normal transfer function
                opacity_transfer = [
                    (data_range[0], 0.0),
                    (data_range[0] + 0.3 * (data_range[1] - data_range[0]), 0.02),
                    (data_range[0] + 0.7 * (data_range[1] - data_range[0]), 0.1),
                    (data_range[1], 0.3)
                ]
        
        # Add volume
        volume = plotter.add_volume(
            grid,
            scalars=variable,
            opacity=opacity_transfer,
            cmap='viridis' if color_transfer is None else color_transfer,
            show_scalar_bar=True
        )
        
        # Set up camera and lighting
        plotter.camera_position = 'iso'
        plotter.add_axes()
        plotter.add_text(f'{variable} - Time: {self.reader.get_time_from_file(file_index):.4f}', 
                        position='upper_left')
        
        if screenshot_path:
            plotter.screenshot(screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")
        
        return plotter
    
    def plot_isosurfaces(self, file_index: int = 0,
                        variable: str = 'velocity_magnitude',
                        isovalues: Optional[List[float]] = None,
                        color_variable: Optional[str] = None,
                        screenshot_path: Optional[str] = None) -> pv.Plotter:
        """
        Create isosurface visualization.
        
        Args:
            file_index: Index of the file to read
            variable: Variable to create isosurfaces for
            isovalues: List of isovalues. If None, automatically determined
            color_variable: Variable to color the isosurfaces
            screenshot_path: Path to save screenshot (optional)
            
        Returns:
            PyVista plotter object
        """
        grid = self.create_structured_grid(file_index)
        
        if variable not in grid.array_names:
            raise ValueError(f"Variable '{variable}' not found in grid. Available: {grid.array_names}")
        
        # Auto-generate isovalues if not provided
        if isovalues is None:
            data_min, data_max = grid[variable].min(), grid[variable].max()
            
            # Handle uniform data
            if data_max - data_min < 1e-10:
                # For uniform data, create a single isosurface at the value
                isovalues = [data_min]
            else:
                # For varying data, create multiple isosurfaces
                isovalues = np.linspace(data_min + 0.2 * (data_max - data_min),
                                      data_max - 0.1 * (data_max - data_min), 5)
        
        # Create plotter
        plotter = pv.Plotter(off_screen=True)
        
        # Add isosurfaces
        for i, isovalue in enumerate(isovalues):
            isosurface = grid.contour(isosurfaces=[isovalue], scalars=variable)
            
            if isosurface.n_points > 0:  # Check if isosurface exists
                color_var = color_variable if color_variable and color_variable in isosurface.array_names else variable
                plotter.add_mesh(
                    isosurface,
                    scalars=color_var,
                    opacity=0.7,
                    cmap='viridis',
                    show_scalar_bar=(i == 0)  # Only show colorbar for first isosurface
                )
        
        # Set up camera and labels
        plotter.camera_position = 'iso'
        plotter.add_axes()
        plotter.add_text(f'{variable} Isosurfaces - Time: {self.reader.get_time_from_file(file_index):.4f}', 
                        position='upper_left')
        
        if screenshot_path:
            plotter.screenshot(screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")
        
        return plotter
    
    def plot_streamlines(self, file_index: int = 0,
                        seed_points: Optional[np.ndarray] = None,
                        integration_direction: str = 'both',
                        max_time: float = 100.0,
                        screenshot_path: Optional[str] = None) -> pv.Plotter:
        """
        Create streamline visualization.
        
        Args:
            file_index: Index of the file to read
            seed_points: Array of seed points for streamlines. If None, auto-generated
            integration_direction: 'forward', 'backward', or 'both'
            max_time: Maximum integration time
            screenshot_path: Path to save screenshot (optional)
            
        Returns:
            PyVista plotter object
        """
        grid = self.create_structured_grid(file_index)
        
        if 'velocity_vectors' not in grid.array_names:
            raise ValueError("Velocity vectors not available for streamline computation")
        
        # Auto-generate seed points if not provided
        if seed_points is None:
            # Create a plane of seed points
            bounds = grid.bounds
            x_seed = np.linspace(bounds[0], bounds[1], 10)
            y_seed = np.linspace(bounds[2], bounds[3], 10)
            z_seed = bounds[4] + 0.1 * (bounds[5] - bounds[4])  # Near the bottom
            
            X_seed, Y_seed = np.meshgrid(x_seed, y_seed)
            seed_points = np.column_stack([
                X_seed.flatten(),
                Y_seed.flatten(),
                np.full(X_seed.size, z_seed)
            ])
        
        # Create seed point mesh
        seed_mesh = pv.PolyData(seed_points)
        
        # Generate streamlines
        streamlines = grid.streamlines_from_source(
            seed_mesh,
            vectors='velocity_vectors',
            integration_direction=integration_direction,
            max_time=max_time,
            initial_step_length=0.1,
            max_step_length=1.0
        )
        
        # Create plotter
        plotter = pv.Plotter(off_screen=True)
        
        # Add streamlines
        if streamlines.n_points > 0:
            plotter.add_mesh(
                streamlines,
                scalars='velocity_magnitude' if 'velocity_magnitude' in streamlines.array_names else None,
                line_width=3,
                cmap='viridis',
                show_scalar_bar=True
            )
        
        # Add seed points
        plotter.add_mesh(seed_mesh, color='red', point_size=5, render_points_as_spheres=True)
        
        # Add a semi-transparent volume for context
        if 'velocity_magnitude' in grid.array_names:
            plotter.add_volume(
                grid,
                scalars='velocity_magnitude',
                opacity=[0.0, 0.0, 0.05, 0.1],
                cmap='viridis',
                show_scalar_bar=False
            )
        
        # Set up camera and labels
        plotter.camera_position = 'iso'
        plotter.add_axes()
        plotter.add_text(f'Streamlines - Time: {self.reader.get_time_from_file(file_index):.4f}', 
                        position='upper_left')
        
        if screenshot_path:
            plotter.screenshot(screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")
        
        return plotter
    
    def plot_slice(self, file_index: int = 0,
                  variable: str = 'velocity_magnitude',
                  normal: Tuple[float, float, float] = (0, 0, 1),
                  origin: Optional[Tuple[float, float, float]] = None,
                  screenshot_path: Optional[str] = None) -> pv.Plotter:
        """
        Create a slice plot through the 3D data.
        
        Args:
            file_index: Index of the file to read
            variable: Variable to visualize on the slice
            normal: Normal vector of the slice plane
            origin: Origin point of the slice plane. If None, uses domain center
            screenshot_path: Path to save screenshot (optional)
            
        Returns:
            PyVista plotter object
        """
        grid = self.create_structured_grid(file_index)
        
        if variable not in grid.array_names:
            raise ValueError(f"Variable '{variable}' not found in grid. Available: {grid.array_names}")
        
        # Set origin to center if not provided
        if origin is None:
            bounds = grid.bounds
            origin = (
                (bounds[0] + bounds[1]) / 2,
                (bounds[2] + bounds[3]) / 2,
                (bounds[4] + bounds[5]) / 2
            )
        
        # Create slice
        slice_mesh = grid.slice(normal=normal, origin=origin)
        
        # Create plotter
        plotter = pv.Plotter(off_screen=True)
        
        # Add slice
        plotter.add_mesh(
            slice_mesh,
            scalars=variable,
            cmap='viridis',
            show_scalar_bar=True
        )
        
        # Add wireframe outline of full domain
        plotter.add_mesh(grid.outline(), color='black', line_width=2)
        
        # Set up camera and labels
        plotter.camera_position = 'iso'
        plotter.add_axes()
        plotter.add_text(f'{variable} Slice - Time: {self.reader.get_time_from_file(file_index):.4f}', 
                        position='upper_left')
        
        if screenshot_path:
            plotter.screenshot(screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")
        
        return plotter
    
    def create_multi_plot(self, file_index: int = 0,
                         variables: List[str] = None,
                         screenshot_path: Optional[str] = None) -> pv.Plotter:
        """
        Create a multi-panel plot with different visualizations.
        
        Args:
            file_index: Index of the file to read
            variables: List of variables to visualize
            screenshot_path: Path to save screenshot (optional)
            
        Returns:
            PyVista plotter object
        """
        grid = self.create_structured_grid(file_index)
        
        if variables is None:
            # Default variables to show
            available_vars = grid.array_names
            variables = []
            for var in ['velocity_magnitude', 'pressure', 'density', 'vorticity_magnitude']:
                if var in available_vars:
                    variables.append(var)
        
        # Create subplot plotter
        n_vars = len(variables)
        if n_vars == 0:
            raise ValueError("No valid variables found for plotting")
        
        # Determine subplot layout
        if n_vars == 1:
            shape = (1, 1)
        elif n_vars == 2:
            shape = (1, 2)
        elif n_vars <= 4:
            shape = (2, 2)
        else:
            shape = (2, 3)  # Max 6 subplots
            variables = variables[:6]
        
        plotter = pv.Plotter(shape=shape)
        
        # Add each variable as a slice or volume
        for i, variable in enumerate(variables):
            row = i // shape[1]
            col = i % shape[1]
            
            plotter.subplot(row, col)
            
            # Create slice for this variable
            slice_mesh = grid.slice(normal=(0, 0, 1), origin=None)
            
            plotter.add_mesh(
                slice_mesh,
                scalars=variable,
                cmap='viridis',
                show_scalar_bar=True
            )
            
            plotter.add_text(f'{variable}', position='upper_left', font_size=10)
            plotter.camera_position = 'xy'
        
        if screenshot_path:
            plotter.screenshot(screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")
        
        return plotter


def create_example_visualizations(domain_path: str, output_dir: str = "visualization_output"):
    """
    Create example visualizations for demonstration.
    
    Args:
        domain_path: Path to the domain folder
        output_dir: Directory to save outputs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize reader and visualizer
    reader = JAXFluidsReader(domain_path)
    extractor = DataExtractor(reader)
    viz = PyVistaVisualizer(reader, extractor)
    
    print("Creating example visualizations...")
    
    # Try different visualization types
    try:
        # Volume rendering
        print("1. Creating volume rendering...")
        plotter1 = viz.plot_volume_rendering(
            file_index=0,
            variable='velocity_magnitude',
            screenshot_path=os.path.join(output_dir, "volume_rendering.png")
        )
        plotter1.close()
        
        # Isosurfaces
        print("2. Creating isosurfaces...")
        plotter2 = viz.plot_isosurfaces(
            file_index=0,
            variable='velocity_magnitude',
            screenshot_path=os.path.join(output_dir, "isosurfaces.png")
        )
        plotter2.close()
        
        # Streamlines
        print("3. Creating streamlines...")
        plotter3 = viz.plot_streamlines(
            file_index=0,
            screenshot_path=os.path.join(output_dir, "streamlines.png")
        )
        plotter3.close()
        
        # Slice plot
        print("4. Creating slice plot...")
        plotter4 = viz.plot_slice(
            file_index=0,
            variable='velocity_magnitude',
            screenshot_path=os.path.join(output_dir, "slice_plot.png")
        )
        plotter4.close()
        
        print(f"Visualizations saved to {output_dir}/")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        print("This might be due to missing data or incompatible grid structure")


if __name__ == "__main__":
    # Example usage
    domain_path = "../propeller_subsonic_wind_tunnel-2/domain"
    if os.path.exists(domain_path):
        create_example_visualizations(domain_path)
    else:
        print(f"Domain path {domain_path} not found")
        print("Please provide a valid path to the JAX-Fluids domain folder")