#!/usr/bin/env python3
"""
VTK Export Utilities for JAX-Fluids Data

This module provides functions to export PyVista visualizations and data to VTK format
for use with external visualization tools like ParaView.

References:
- https://docs.pyvista.org/examples/00-load/create-structured-surface.html
- https://github.com/metialex/h5ToVTK
"""

import numpy as np
import pyvista as pv
import os
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .h5_reader import JAXFluidsReader
from ..core.data_extractor import DataExtractor


class VTKExporter:
    """Export JAX-Fluids data to VTK format."""
    
    def __init__(self, reader: JAXFluidsReader, extractor: DataExtractor):
        """
        Initialize the VTK exporter.
        
        Args:
            reader: JAXFluidsReader instance
            extractor: DataExtractor instance
        """
        self.reader = reader
        self.extractor = extractor
        
        if self.reader.grid_info is None:
            raise ValueError("Grid information not available from reader")
        
        self.grid_shape = self.reader.grid_info.get('shape')
        if self.grid_shape is None:
            raise ValueError("Grid shape not available")
    
    def export_structured_grid(self, file_index: int = 0,
                             output_path: str = "output.vts",
                             variables: Optional[List[str]] = None,
                             include_derived: bool = True) -> str:
        """
        Export structured grid data to VTK format.
        
        Args:
            file_index: Index of the file to export
            output_path: Path for the output VTK file
            variables: List of variables to include. If None, includes all available
            include_derived: Whether to include derived quantities (vorticity, Q-criterion, etc.)
            
        Returns:
            Path to the exported file
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
        
        # Add derived quantities if requested
        if include_derived:
            self._add_derived_quantities_to_grid(grid, file_index)
        
        # Add time information as metadata
        time_value = self.reader.get_time_from_file(file_index)
        grid.field_data['Time'] = np.array([time_value])
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to VTK format
        grid.save(str(output_path))
        
        print(f"Exported structured grid to {output_path}")
        return str(output_path)
    
    def _add_derived_quantities_to_grid(self, grid: pv.StructuredGrid, file_index: int):
        """Add derived quantities to the grid."""
        
        # Velocity magnitude
        vel_mag = self.extractor.compute_velocity_magnitude(file_index)
        if vel_mag is not None:
            grid['velocity_magnitude'] = vel_mag.flatten(order='F')
        
        # Vorticity components and magnitude
        vorticity = self.extractor.compute_vorticity(file_index)
        if vorticity is not None:
            for comp_name, comp_data in vorticity.items():
                grid[comp_name] = comp_data.flatten(order='F')
            
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
    
    def export_time_series(self, output_dir: str = "vtk_output",
                          file_indices: Optional[List[int]] = None,
                          variables: Optional[List[str]] = None,
                          include_derived: bool = True,
                          base_name: str = "flow_data") -> List[str]:
        """
        Export multiple time steps as a time series.
        
        Args:
            output_dir: Directory to save VTK files
            file_indices: List of file indices to export. If None, exports all available
            variables: List of variables to include
            include_derived: Whether to include derived quantities
            base_name: Base name for output files
            
        Returns:
            List of exported file paths
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine which files to export
        if file_indices is None:
            file_indices = list(range(len(self.reader.available_files)))
        
        exported_files = []
        
        for i, file_idx in enumerate(file_indices):
            # Get time for filename
            time_value = self.reader.get_time_from_file(file_idx)
            
            # Create filename with time step
            filename = f"{base_name}_{i:04d}_t{time_value:.6f}.vts"
            file_path = output_path / filename
            
            try:
                exported_path = self.export_structured_grid(
                    file_index=file_idx,
                    output_path=str(file_path),
                    variables=variables,
                    include_derived=include_derived
                )
                exported_files.append(exported_path)
                print(f"Exported time step {i+1}/{len(file_indices)}: {filename}")
                
            except Exception as e:
                print(f"Error exporting file index {file_idx}: {e}")
        
        # Create ParaView state file for time series
        self._create_paraview_state_file(exported_files, output_path, base_name)
        
        return exported_files
    
    def _create_paraview_state_file(self, vtk_files: List[str], 
                                   output_dir: Path, base_name: str):
        """Create a simple ParaView state file for the time series."""
        
        if not vtk_files:
            return
        
        state_file = output_dir / f"{base_name}_paraview.pvsm"
        
        # Get relative paths for portability
        rel_files = [os.path.relpath(f, output_dir) for f in vtk_files]
        
        # Simple ParaView state template
        state_content = f'''<?xml version="1.0"?>
<ParaView version="5.9.0">
  <ServerManagerState version="5.9.0">
    <Source id="1" type="XMLStructuredGridReader">
      <Property name="FileName" value="{rel_files[0]}"/>
      <Property name="TimestepValues">
'''
        
        # Add timestep values
        for vtk_file in vtk_files:
            # Extract time from filename
            basename = os.path.basename(vtk_file)
            time_str = basename.split('_t')[1].split('.vts')[0]
            state_content += f'        <Element index="0" value="{time_str}"/>\n'
        
        state_content += '''      </Property>
    </Source>
  </ServerManagerState>
</ParaView>'''
        
        with open(state_file, 'w') as f:
            f.write(state_content)
        
        print(f"Created ParaView state file: {state_file}")
    
    def export_isosurfaces(self, file_index: int = 0,
                          variable: str = 'velocity_magnitude',
                          isovalues: Optional[List[float]] = None,
                          output_dir: str = "vtk_isosurfaces") -> List[str]:
        """
        Export isosurfaces to VTK format.
        
        Args:
            file_index: Index of the file to process
            variable: Variable to create isosurfaces for
            isovalues: List of isovalues. If None, automatically determined
            output_dir: Directory to save isosurface files
            
        Returns:
            List of exported isosurface file paths
        """
        # Create structured grid
        if self.extractor.X is None or self.extractor.Y is None or self.extractor.Z is None:
            raise ValueError("Coordinate arrays not available")
        
        X, Y, Z = self.extractor.X, self.extractor.Y, self.extractor.Z
        grid = pv.StructuredGrid(X, Y, Z)
        
        # Add variables
        primitives = self.reader.read_primitives(file_index)
        for var_name, var_data in primitives.items():
            if var_data.ndim == 3 and var_data.shape == self.grid_shape:
                grid[var_name] = var_data.flatten(order='F')
        
        self._add_derived_quantities_to_grid(grid, file_index)
        
        if variable not in grid.array_names:
            raise ValueError(f"Variable '{variable}' not found in grid. Available: {grid.array_names}")
        
        # Auto-generate isovalues if not provided
        if isovalues is None:
            data_min, data_max = grid[variable].min(), grid[variable].max()
            isovalues = np.linspace(data_min + 0.2 * (data_max - data_min),
                                  data_max - 0.1 * (data_max - data_min), 5)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        time_value = self.reader.get_time_from_file(file_index)
        
        for i, isovalue in enumerate(isovalues):
            isosurface = grid.contour(isosurfaces=[isovalue], scalars=variable)
            
            if isosurface.n_points > 0:
                filename = f"isosurface_{variable}_{isovalue:.6f}_t{time_value:.6f}.vtp"
                file_path = output_path / filename
                
                isosurface.save(str(file_path))
                exported_files.append(str(file_path))
                print(f"Exported isosurface {i+1}: {filename}")
        
        return exported_files
    
    def export_streamlines(self, file_index: int = 0,
                          seed_points: Optional[np.ndarray] = None,
                          output_path: str = "streamlines.vtp",
                          integration_direction: str = 'both',
                          max_time: float = 100.0) -> str:
        """
        Export streamlines to VTK format.
        
        Args:
            file_index: Index of the file to process
            seed_points: Array of seed points for streamlines
            output_path: Path for the output VTK file
            integration_direction: 'forward', 'backward', or 'both'
            max_time: Maximum integration time
            
        Returns:
            Path to the exported streamlines file
        """
        # Create structured grid with velocity vectors
        if self.extractor.X is None or self.extractor.Y is None or self.extractor.Z is None:
            raise ValueError("Coordinate arrays not available")
        
        X, Y, Z = self.extractor.X, self.extractor.Y, self.extractor.Z
        grid = pv.StructuredGrid(X, Y, Z)
        
        # Add variables including velocity vectors
        primitives = self.reader.read_primitives(file_index)
        for var_name, var_data in primitives.items():
            if var_data.ndim == 3 and var_data.shape == self.grid_shape:
                grid[var_name] = var_data.flatten(order='F')
        
        self._add_derived_quantities_to_grid(grid, file_index)
        
        if 'velocity_vectors' not in grid.array_names:
            raise ValueError("Velocity vectors not available for streamline computation")
        
        # Auto-generate seed points if not provided
        if seed_points is None:
            bounds = grid.bounds
            x_seed = np.linspace(bounds[0], bounds[1], 10)
            y_seed = np.linspace(bounds[2], bounds[3], 10)
            z_seed = bounds[4] + 0.1 * (bounds[5] - bounds[4])
            
            X_seed, Y_seed = np.meshgrid(x_seed, y_seed)
            seed_points = np.column_stack([
                X_seed.flatten(),
                Y_seed.flatten(),
                np.full(X_seed.size, z_seed)
            ])
        
        # Create seed point mesh
        seed_mesh = pv.PolyData(seed_points)
        
        # Generate streamlines
        streamlines = grid.streamlines(
            seed_mesh,
            vectors='velocity_vectors',
            integration_direction=integration_direction,
            max_time=max_time,
            initial_step_length=0.1,
            max_step_length=1.0
        )
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add time information
        time_value = self.reader.get_time_from_file(file_index)
        streamlines.field_data['Time'] = np.array([time_value])
        
        # Save streamlines
        streamlines.save(str(output_path))
        
        print(f"Exported streamlines to {output_path}")
        return str(output_path)
    
    def export_slices(self, file_index: int = 0,
                     variables: Optional[List[str]] = None,
                     slice_normals: List[Tuple[float, float, float]] = None,
                     output_dir: str = "vtk_slices") -> List[str]:
        """
        Export slice data to VTK format.
        
        Args:
            file_index: Index of the file to process
            variables: List of variables to include on slices
            slice_normals: List of normal vectors for slice planes
            output_dir: Directory to save slice files
            
        Returns:
            List of exported slice file paths
        """
        # Default slice normals (XY, XZ, YZ planes)
        if slice_normals is None:
            slice_normals = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
        
        # Create structured grid
        if self.extractor.X is None or self.extractor.Y is None or self.extractor.Z is None:
            raise ValueError("Coordinate arrays not available")
        
        X, Y, Z = self.extractor.X, self.extractor.Y, self.extractor.Z
        grid = pv.StructuredGrid(X, Y, Z)
        
        # Add variables
        primitives = self.reader.read_primitives(file_index, variables)
        for var_name, var_data in primitives.items():
            if var_data.ndim == 3 and var_data.shape == self.grid_shape:
                grid[var_name] = var_data.flatten(order='F')
        
        self._add_derived_quantities_to_grid(grid, file_index)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        time_value = self.reader.get_time_from_file(file_index)
        
        # Create slices
        for i, normal in enumerate(slice_normals):
            # Use domain center as origin
            bounds = grid.bounds
            origin = (
                (bounds[0] + bounds[1]) / 2,
                (bounds[2] + bounds[3]) / 2,
                (bounds[4] + bounds[5]) / 2
            )
            
            slice_mesh = grid.slice(normal=normal, origin=origin)
            
            if slice_mesh.n_points > 0:
                # Determine slice name based on normal
                if normal == (0, 0, 1):
                    slice_name = "xy_plane"
                elif normal == (0, 1, 0):
                    slice_name = "xz_plane"
                elif normal == (1, 0, 0):
                    slice_name = "yz_plane"
                else:
                    slice_name = f"slice_{i}"
                
                filename = f"{slice_name}_t{time_value:.6f}.vtp"
                file_path = output_path / filename
                
                # Add time information
                slice_mesh.field_data['Time'] = np.array([time_value])
                
                slice_mesh.save(str(file_path))
                exported_files.append(str(file_path))
                print(f"Exported slice {i+1}: {filename}")
        
        return exported_files


def batch_export_vtk(domain_path: str, output_base_dir: str = "vtk_exports"):
    """
    Batch export all available data to VTK format.
    
    Args:
        domain_path: Path to the JAX-Fluids domain folder
        output_base_dir: Base directory for all VTK exports
    """
    print("Starting batch VTK export...")
    
    # Initialize reader and exporter
    reader = JAXFluidsReader(domain_path)
    extractor = DataExtractor(reader)
    exporter = VTKExporter(reader, extractor)
    
    # Create base output directory
    output_path = Path(output_base_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting {len(reader.available_files)} time steps...")
    
    try:
        # Export time series of structured grids
        print("\n1. Exporting structured grids...")
        grid_files = exporter.export_time_series(
            output_dir=str(output_path / "structured_grids"),
            include_derived=True
        )
        
        # Export isosurfaces for the first few time steps
        print("\n2. Exporting isosurfaces...")
        max_iso_files = min(3, len(reader.available_files))
        for i in range(max_iso_files):
            iso_files = exporter.export_isosurfaces(
                file_index=i,
                variable='velocity_magnitude',
                output_dir=str(output_path / f"isosurfaces_t{i}")
            )
        
        # Export streamlines for first time step
        print("\n3. Exporting streamlines...")
        if len(reader.available_files) > 0:
            streamline_file = exporter.export_streamlines(
                file_index=0,
                output_path=str(output_path / "streamlines" / "streamlines_t0.vtp")
            )
        
        # Export slices for first time step
        print("\n4. Exporting slices...")
        if len(reader.available_files) > 0:
            slice_files = exporter.export_slices(
                file_index=0,
                output_dir=str(output_path / "slices")
            )
        
        print(f"\nBatch export completed. All files saved to: {output_path}")
        print("\nTo view the results:")
        print(f"1. Open ParaView")
        print(f"2. Load the structured grids from: {output_path / 'structured_grids'}")
        print(f"3. For time series, use the ParaView state file if available")
        
    except Exception as e:
        print(f"Error during batch export: {e}")
        print("Some exports may have failed due to missing data or incompatible format")


if __name__ == "__main__":
    # Example usage
    domain_path = "../propeller_subsonic_wind_tunnel-2/domain"
    if os.path.exists(domain_path):
        batch_export_vtk(domain_path)
    else:
        print(f"Domain path {domain_path} not found")
        print("Please provide a valid path to the JAX-Fluids domain folder")