#!/usr/bin/env python3
"""
JAX-Fluids Data Extraction Utilities

This module provides functions to extract and compute derived quantities from JAX-Fluids data.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from ..io.h5_reader import JAXFluidsReader


class DataExtractor:
    """Class to extract and compute flow quantities from JAX-Fluids data."""
    
    def __init__(self, reader: JAXFluidsReader):
        """
        Initialize with a JAXFluidsReader instance.
        
        Args:
            reader: JAXFluidsReader instance
        """
        self.reader = reader
        self.X, self.Y, self.Z = None, None, None
        
        # Create coordinate arrays if grid info is available
        if self.reader.grid_info is not None:
            try:
                self.X, self.Y, self.Z = self.reader.create_coordinate_arrays()
            except ValueError as e:
                print(f"Warning: Could not create coordinate arrays: {e}")
    
    def extract_velocity_components(self, file_index: int = 0) -> Dict[str, np.ndarray]:
        """
        Extract velocity components (u, v, w) from the data.
        
        Args:
            file_index: Index of the file to read
            
        Returns:
            Dictionary with 'u', 'v', 'w' velocity components
        """
        # First try individual components
        primitives = self.reader.read_primitives(file_index, ['u', 'v', 'w'])
        
        velocity = {}
        if all(comp in primitives for comp in ['u', 'v', 'w']):
            # Individual components available
            velocity['u'] = primitives['u']
            velocity['v'] = primitives['v']
            velocity['w'] = primitives['w']
        else:
            # Try JAX-Fluids vector format
            primitives = self.reader.read_primitives(file_index, ['velocity'])
            if 'velocity' in primitives:
                vel_data = primitives['velocity']
                
                # JAX-Fluids stores velocity as (nx, ny, nz, 3)
                if len(vel_data.shape) == 4 and vel_data.shape[-1] == 3:
                    velocity['u'] = vel_data[:, :, :, 0]
                    velocity['v'] = vel_data[:, :, :, 1] 
                    velocity['w'] = vel_data[:, :, :, 2]
                    print(f"Extracted velocity components from JAX-Fluids vector field: shape {vel_data.shape}")
                else:
                    print(f"Warning: Unexpected velocity field shape: {vel_data.shape}")
            else:
                print("Warning: No velocity data found (neither components nor vector field)")
        
        return velocity
    
    def extract_thermodynamic_quantities(self, file_index: int = 0) -> Dict[str, np.ndarray]:
        """
        Extract thermodynamic quantities (density, pressure, temperature).
        
        Args:
            file_index: Index of the file to read
            
        Returns:
            Dictionary with thermodynamic quantities
        """
        # Common primitive variable names in CFD
        thermo_vars = ['rho', 'p', 'T', 'density', 'pressure', 'temperature']
        primitives = self.reader.read_primitives(file_index, thermo_vars)
        
        # Standardize names
        result = {}
        
        # Density
        if 'rho' in primitives:
            result['density'] = primitives['rho']
        elif 'density' in primitives:
            result['density'] = primitives['density']
        
        # Pressure
        if 'p' in primitives:
            result['pressure'] = primitives['p']
        elif 'pressure' in primitives:
            result['pressure'] = primitives['pressure']
        
        # Temperature
        if 'T' in primitives:
            result['temperature'] = primitives['T']
        elif 'temperature' in primitives:
            result['temperature'] = primitives['temperature']
        
        return result
    
    def compute_velocity_magnitude(self, file_index: int = 0) -> Optional[np.ndarray]:
        """
        Compute velocity magnitude from velocity components.
        
        Args:
            file_index: Index of the file to read
            
        Returns:
            Velocity magnitude array or None if velocity components not available
        """
        velocity = self.extract_velocity_components(file_index)
        
        if all(comp in velocity for comp in ['u', 'v', 'w']):
            u, v, w = velocity['u'], velocity['v'], velocity['w']
            return np.sqrt(u**2 + v**2 + w**2)
        else:
            print("Warning: Cannot compute velocity magnitude - missing components")
            return None
    
    def compute_vorticity(self, file_index: int = 0) -> Optional[Dict[str, np.ndarray]]:
        """
        Compute vorticity components using finite differences.
        
        Args:
            file_index: Index of the file to read
            
        Returns:
            Dictionary with vorticity components ('omega_x', 'omega_y', 'omega_z')
        """
        velocity = self.extract_velocity_components(file_index)
        
        if not all(comp in velocity for comp in ['u', 'v', 'w']):
            print("Warning: Cannot compute vorticity - missing velocity components")
            return None
        
        if self.reader.grid_info is None:
            print("Warning: Cannot compute vorticity - grid information not available")
            return None
        
        u, v, w = velocity['u'], velocity['v'], velocity['w']
        
        # Get grid spacing
        dx = self.reader.grid_info['cellsizeX']
        dy = self.reader.grid_info['cellsizeY']
        dz = self.reader.grid_info['cellsizeZ']
        
        if any(ds is None for ds in [dx, dy, dz]):
            print("Warning: Cannot compute vorticity - grid spacing not available")
            return None
        
        # Compute vorticity using central differences
        # ω_x = ∂w/∂y - ∂v/∂z
        # ω_y = ∂u/∂z - ∂w/∂x
        # ω_z = ∂v/∂x - ∂u/∂y
        
        vorticity = {}
        
        # Central difference gradients (interior points)
        dwdy = np.zeros_like(w)
        dvdz = np.zeros_like(v)
        dudz = np.zeros_like(u)
        dwdx = np.zeros_like(w)
        dvdx = np.zeros_like(v)
        dudy = np.zeros_like(u)
        
        # ∂w/∂y
        dwdy[:, 1:-1, :] = (w[:, 2:, :] - w[:, :-2, :]) / (2 * dy)
        # ∂v/∂z
        dvdz[:, :, 1:-1] = (v[:, :, 2:] - v[:, :, :-2]) / (2 * dz)
        # ∂u/∂z
        dudz[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dz)
        # ∂w/∂x
        dwdx[1:-1, :, :] = (w[2:, :, :] - w[:-2, :, :]) / (2 * dx)
        # ∂v/∂x
        dvdx[1:-1, :, :] = (v[2:, :, :] - v[:-2, :, :]) / (2 * dx)
        # ∂u/∂y
        dudy[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dy)
        
        vorticity['omega_x'] = dwdy - dvdz
        vorticity['omega_y'] = dudz - dwdx
        vorticity['omega_z'] = dvdx - dudy
        
        return vorticity
    
    def compute_vorticity_magnitude(self, file_index: int = 0) -> Optional[np.ndarray]:
        """
        Compute vorticity magnitude.
        
        Args:
            file_index: Index of the file to read
            
        Returns:
            Vorticity magnitude array
        """
        vorticity = self.compute_vorticity(file_index)
        
        if vorticity is None:
            return None
        
        omega_x = vorticity['omega_x']
        omega_y = vorticity['omega_y']
        omega_z = vorticity['omega_z']
        
        return np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
    
    def compute_q_criterion(self, file_index: int = 0) -> Optional[np.ndarray]:
        """
        Compute Q-criterion for vortex identification.
        Q = 0.5 * (||Ω||² - ||S||²) where Ω is vorticity tensor and S is strain rate tensor
        
        Args:
            file_index: Index of the file to read
            
        Returns:
            Q-criterion array
        """
        velocity = self.extract_velocity_components(file_index)
        
        if not all(comp in velocity for comp in ['u', 'v', 'w']):
            print("Warning: Cannot compute Q-criterion - missing velocity components")
            return None
        
        if self.reader.grid_info is None:
            print("Warning: Cannot compute Q-criterion - grid information not available")
            return None
        
        u, v, w = velocity['u'], velocity['v'], velocity['w']
        
        # Get grid spacing
        dx = self.reader.grid_info['cellsizeX']
        dy = self.reader.grid_info['cellsizeY']
        dz = self.reader.grid_info['cellsizeZ']
        
        if any(ds is None for ds in [dx, dy, dz]):
            print("Warning: Cannot compute Q-criterion - grid spacing not available")
            return None
        
        # Compute velocity gradients
        dudx = np.zeros_like(u)
        dudy = np.zeros_like(u)
        dudz = np.zeros_like(u)
        dvdx = np.zeros_like(v)
        dvdy = np.zeros_like(v)
        dvdz = np.zeros_like(v)
        dwdx = np.zeros_like(w)
        dwdy = np.zeros_like(w)
        dwdz = np.zeros_like(w)
        
        # Central differences for interior points
        dudx[1:-1, :, :] = (u[2:, :, :] - u[:-2, :, :]) / (2 * dx)
        dudy[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dy)
        dudz[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dz)
        
        dvdx[1:-1, :, :] = (v[2:, :, :] - v[:-2, :, :]) / (2 * dx)
        dvdy[:, 1:-1, :] = (v[:, 2:, :] - v[:, :-2, :]) / (2 * dy)
        dvdz[:, :, 1:-1] = (v[:, :, 2:] - v[:, :, :-2]) / (2 * dz)
        
        dwdx[1:-1, :, :] = (w[2:, :, :] - w[:-2, :, :]) / (2 * dx)
        dwdy[:, 1:-1, :] = (w[:, 2:, :] - w[:, :-2, :]) / (2 * dy)
        dwdz[:, :, 1:-1] = (w[:, :, 2:] - w[:, :, :-2]) / (2 * dz)
        
        # Strain rate tensor components
        S11 = dudx
        S22 = dvdy
        S33 = dwdz
        S12 = 0.5 * (dudy + dvdx)
        S13 = 0.5 * (dudz + dwdx)
        S23 = 0.5 * (dvdz + dwdy)
        
        # Vorticity tensor components
        O12 = 0.5 * (dudy - dvdx)
        O13 = 0.5 * (dudz - dwdx)
        O23 = 0.5 * (dvdz - dwdy)
        
        # ||S||² = 2 * (S11² + S22² + S33² + 2*(S12² + S13² + S23²))
        S_squared = 2 * (S11**2 + S22**2 + S33**2 + 2*(S12**2 + S13**2 + S23**2))
        
        # ||Ω||² = 2 * (O12² + O13² + O23²)
        Omega_squared = 2 * (O12**2 + O13**2 + O23**2)
        
        # Q-criterion
        Q = 0.5 * (Omega_squared - S_squared)
        
        return Q
    
    def extract_surface_data(self, file_index: int = 0, 
                           surface_coord: str = 'z', 
                           surface_value: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Extract data on a surface (plane) for 2D visualization.
        
        Args:
            file_index: Index of the file to read
            surface_coord: Coordinate normal to the surface ('x', 'y', or 'z')
            surface_value: Value of the coordinate defining the surface
            
        Returns:
            Dictionary with 2D arrays of variables on the surface
        """
        # Get all primitive variables
        primitives = self.reader.read_primitives(file_index)
        
        if not primitives:
            print("Warning: No primitive variables found")
            return {}
        
        if self.X is None or self.Y is None or self.Z is None:
            print("Warning: Coordinate arrays not available")
            return {}
        
        # Find the closest index to the surface value
        if surface_coord == 'x':
            coord_array = self.reader.grid_info['gridX']
            surface_idx = np.argmin(np.abs(coord_array - surface_value))
            slice_obj = (surface_idx, slice(None), slice(None))
            x_2d, y_2d = self.Y[surface_idx, :, :], self.Z[surface_idx, :, :]
        elif surface_coord == 'y':
            coord_array = self.reader.grid_info['gridY']
            surface_idx = np.argmin(np.abs(coord_array - surface_value))
            slice_obj = (slice(None), surface_idx, slice(None))
            x_2d, y_2d = self.X[:, surface_idx, :], self.Z[:, surface_idx, :]
        elif surface_coord == 'z':
            coord_array = self.reader.grid_info['gridZ']
            surface_idx = np.argmin(np.abs(coord_array - surface_value))
            slice_obj = (slice(None), slice(None), surface_idx)
            x_2d, y_2d = self.X[:, :, surface_idx], self.Y[:, :, surface_idx]
        else:
            raise ValueError("surface_coord must be 'x', 'y', or 'z'")
        
        result = {
            'x_2d': x_2d,
            'y_2d': y_2d,
            f'{surface_coord}_value': coord_array[surface_idx]
        }
        
        # Extract all variables on the surface
        for var_name, var_data in primitives.items():
            if var_data.ndim == 3:  # Scalar field
                result[var_name] = var_data[slice_obj]
        
        return result
    
    def get_flow_statistics(self, file_index: int = 0) -> Dict[str, Dict[str, float]]:
        """
        Compute basic flow statistics.
        
        Args:
            file_index: Index of the file to read
            
        Returns:
            Dictionary with statistics for each variable
        """
        primitives = self.reader.read_primitives(file_index)
        
        if not primitives:
            print("Warning: No primitive variables found")
            return {}
        
        stats = {}
        
        for var_name, var_data in primitives.items():
            if var_data.size > 0:
                stats[var_name] = {
                    'min': float(np.min(var_data)),
                    'max': float(np.max(var_data)),
                    'mean': float(np.mean(var_data)),
                    'std': float(np.std(var_data)),
                    'shape': var_data.shape
                }
        
        # Add derived quantities if possible
        vel_mag = self.compute_velocity_magnitude(file_index)
        if vel_mag is not None:
            stats['velocity_magnitude'] = {
                'min': float(np.min(vel_mag)),
                'max': float(np.max(vel_mag)),
                'mean': float(np.mean(vel_mag)),
                'std': float(np.std(vel_mag))
            }
        
        return stats


if __name__ == "__main__":
    # Example usage
    import os
    from .h5_reader import JAXFluidsReader
    
    domain_path = "../propeller_subsonic_wind_tunnel-2/domain"
    if os.path.exists(domain_path):
        reader = JAXFluidsReader(domain_path)
        extractor = FlowDataExtractor(reader)
        
        print("=== Flow Data Extraction Example ===")
        
        # Get statistics
        stats = extractor.get_flow_statistics(0)
        if stats:
            print("\nFlow Statistics:")
            for var, stat in stats.items():
                print(f"{var}: min={stat['min']:.3e}, max={stat['max']:.3e}, mean={stat['mean']:.3e}")
        
        # Try to compute derived quantities
        vel_mag = extractor.compute_velocity_magnitude(0)
        if vel_mag is not None:
            print(f"\nVelocity magnitude computed: shape={vel_mag.shape}")
        
        vorticity = extractor.compute_vorticity(0)
        if vorticity is not None:
            print(f"Vorticity components computed")
    else:
        print(f"Domain path {domain_path} not found")