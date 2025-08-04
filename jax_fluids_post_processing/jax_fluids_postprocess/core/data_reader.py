"""
Data reader interface for JAX-Fluids simulation results.
Provides a clean abstraction over the H5 file reader.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np

from ..io.h5_reader import JAXFluidsReader


class DataReader:
    """
    High-level interface for reading JAX-Fluids simulation data.
    
    Wraps the H5Reader with additional convenience methods and validation.
    """
    
    def __init__(self, results_path: Union[str, Path]):
        """
        Initialize data reader.
        
        Args:
            results_path: Path to simulation results directory
        """
        self.results_path = Path(results_path)
        self.h5_reader = JAXFluidsReader(str(results_path))
        
        # Validate data
        if self.h5_reader.get_num_time_steps() == 0:
            raise ValueError(f"No valid time steps found in {results_path}")
    
    def get_variable_names(self) -> List[str]:
        """Get list of available variable names."""
        return self.h5_reader.get_available_variables()
    
    def get_time_steps(self) -> np.ndarray:
        """Get array of simulation times."""
        return self.h5_reader.get_times()
    
    def get_grid_dimensions(self) -> Tuple[int, int, int]:
        """Get grid dimensions (nx, ny, nz)."""
        return self.h5_reader.get_grid_shape()
    
    def get_domain_extent(self) -> Tuple[float, float, float, float, float, float]:
        """Get domain bounds (xmin, xmax, ymin, ymax, zmin, zmax)."""
        return self.h5_reader.get_domain_bounds()
    
    def read_variable(
        self, 
        variable_name: str, 
        time_index: int = -1
    ) -> np.ndarray:
        """
        Read a single variable at specified time step.
        
        Args:
            variable_name: Name of variable to read
            time_index: Time step index (-1 for last)
            
        Returns:
            Variable data as numpy array
        """
        primitives = self.h5_reader.read_primitives(time_index, [variable_name])
        
        if variable_name not in primitives:
            available = self.get_variable_names()
            raise ValueError(f"Variable '{variable_name}' not found. Available: {available}")
        
        return primitives[variable_name]
    
    def read_variables(
        self, 
        variable_names: List[str], 
        time_index: int = -1
    ) -> Dict[str, np.ndarray]:
        """
        Read multiple variables at specified time step.
        
        Args:
            variable_names: List of variable names to read
            time_index: Time step index (-1 for last)
            
        Returns:
            Dictionary mapping variable names to numpy arrays
        """
        return self.h5_reader.read_primitives(time_index, variable_names)
    
    def read_time_series(
        self, 
        variable_name: str, 
        time_indices: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """
        Read variable data across multiple time steps.
        
        Args:
            variable_name: Name of variable to read
            time_indices: List of time indices (None for all)
            
        Returns:
            List of numpy arrays, one per time step
        """
        if time_indices is None:
            time_indices = list(range(self.h5_reader.get_num_time_steps()))
        
        time_series = []
        for t_idx in time_indices:
            data = self.read_variable(variable_name, t_idx)
            time_series.append(data)
        
        return time_series
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get comprehensive metadata about the simulation."""
        return {
            'path': str(self.results_path),
            'num_time_steps': self.h5_reader.get_num_time_steps(),
            'grid_shape': self.get_grid_dimensions(),
            'domain_bounds': self.get_domain_extent(),
            'time_range': self.h5_reader.get_time_range(),
            'available_variables': self.get_variable_names(),
            'file_info': {
                'num_files': len(self.h5_reader.available_files),
                'files': [f.name for f in self.h5_reader.available_files]
            }
        }
    
    def validate_time_index(self, time_index: int) -> int:
        """
        Validate and normalize time index.
        
        Args:
            time_index: Time index to validate
            
        Returns:
            Normalized time index
        """
        num_steps = self.h5_reader.get_num_time_steps()
        
        if time_index < 0:
            time_index = num_steps + time_index
        
        if time_index < 0 or time_index >= num_steps:
            raise ValueError(f"Time index {time_index} out of range [0, {num_steps-1}]")
        
        return time_index
    
    def sample_data(
        self, 
        variable_name: str, 
        time_index: int = -1,
        subsample_factor: int = 1
    ) -> np.ndarray:
        """
        Read variable data with optional subsampling for large datasets.
        
        Args:
            variable_name: Name of variable to read
            time_index: Time step index
            subsample_factor: Factor to subsample by (1 = no subsampling)
            
        Returns:
            Subsampled variable data
        """
        data = self.read_variable(variable_name, time_index)
        
        if subsample_factor > 1:
            # Subsample along all spatial dimensions
            slices = tuple(slice(None, None, subsample_factor) for _ in range(data.ndim))
            data = data[slices]
        
        return data