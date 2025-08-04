#!/usr/bin/env python3
"""
JAX-Fluids H5 Data Reader

This module provides utilities to read and extract data from JAX-Fluids simulation output files.
Based on the JAXFLUIDS postprocess module structure.

References:
- https://github.com/tumaer/JAXFLUIDS/tree/main/src/jaxfluids_postprocess
- https://cerfacs.fr/coop/typical-postprocessing
"""

import h5py
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Union
import json
from pathlib import Path


class JAXFluidsReader:
    """Reader class for JAX-Fluids H5 output files."""
    
    def __init__(self, domain_path: str, config_path: Optional[str] = None):
        """
        Initialize the reader with the domain path containing H5 files.
        
        Args:
            domain_path: Path to the domain folder containing data_*.h5 files
            config_path: Optional path to the configuration JSON file
        """
        self.domain_path = Path(domain_path)
        self.config_path = Path(config_path) if config_path else None
        self.config = None
        self.available_files = []
        self.grid_info = None
        
        # Load configuration if provided
        if self.config_path and self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        
        # Scan for available data files
        self._scan_files()
        
        # Load grid information from the first file
        if self.available_files:
            self._load_grid_info()
    
    def _scan_files(self):
        """Scan the domain directory for available H5 files."""
        if not self.domain_path.exists():
            raise FileNotFoundError(f"Domain path {self.domain_path} does not exist")
        
        # Find all data_*.h5 files
        pattern = "data_*.h5"
        h5_files = list(self.domain_path.glob(pattern))
        
        # Sort by time value extracted from filename
        def extract_time(filepath):
            name = filepath.stem  # Remove .h5 extension
            time_str = name.split('_')[1]  # Extract time part after 'data_'
            return float(time_str)
        
        self.available_files = sorted(h5_files, key=extract_time)
        
        if not self.available_files:
            print(f"Warning: No H5 files found in {self.domain_path}")
        else:
            print(f"Found {len(self.available_files)} H5 files")
            times = [extract_time(f) for f in self.available_files]
            print(f"Time range: {min(times):.6f} to {max(times):.6f}")
    
    def _load_grid_info(self):
        """Load grid information from the first available file."""
        if not self.available_files:
            return
        
        with h5py.File(self.available_files[0], 'r') as f:
            if 'domain' in f:
                domain = f['domain']
                self.grid_info = {
                    'dim': int(domain['dim'][()]) if 'dim' in domain else 3,
                    'gridX': domain['gridX'][:] if 'gridX' in domain else None,
                    'gridY': domain['gridY'][:] if 'gridY' in domain else None,
                    'gridZ': domain['gridZ'][:] if 'gridZ' in domain else None,
                    'gridFX': domain['gridFX'][:] if 'gridFX' in domain else None,
                    'gridFY': domain['gridFY'][:] if 'gridFY' in domain else None,
                    'gridFZ': domain['gridFZ'][:] if 'gridFZ' in domain else None,
                    'cellsizeX': float(domain['cellsizeX'][()]) if 'cellsizeX' in domain else None,
                    'cellsizeY': float(domain['cellsizeY'][()]) if 'cellsizeY' in domain else None,
                    'cellsizeZ': float(domain['cellsizeZ'][()]) if 'cellsizeZ' in domain else None,
                }
                
                if self.grid_info['gridX'] is not None:
                    nx, ny, nz = len(self.grid_info['gridX']), len(self.grid_info['gridY']), len(self.grid_info['gridZ'])
                    self.grid_info['shape'] = (nx, ny, nz)
                    print(f"Grid shape: {nx} x {ny} x {nz}")
    
    def get_available_times(self) -> List[float]:
        """Get list of available simulation times."""
        def extract_time(filepath):
            name = filepath.stem
            time_str = name.split('_')[1]
            return float(time_str)
        
        return [extract_time(f) for f in self.available_files]
    
    def get_file_info(self, file_index: int = 0) -> Dict:
        """Get information about a specific file's contents."""
        if not (0 <= file_index < len(self.available_files)):
            raise IndexError(f"File index {file_index} out of range [0, {len(self.available_files)-1}]")
        
        filepath = self.available_files[file_index]
        info = {'filepath': str(filepath), 'groups': {}}
        
        with h5py.File(filepath, 'r') as f:
            def collect_info(name, obj):
                if isinstance(obj, h5py.Group):
                    info['groups'][name] = {'type': 'group', 'keys': list(obj.keys())}
                elif isinstance(obj, h5py.Dataset):
                    info['groups'][name] = {
                        'type': 'dataset',
                        'shape': obj.shape,
                        'dtype': str(obj.dtype)
                    }
            
            info['root_keys'] = list(f.keys())
            f.visititems(collect_info)
        
        return info
    
    def read_primitives(self, file_index: int = 0, variables: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Read primitive variables from a specific file.
        
        Args:
            file_index: Index of the file to read
            variables: List of variable names to read. If None, read all available.
            
        Returns:
            Dictionary with variable names as keys and numpy arrays as values
        """
        if not (0 <= file_index < len(self.available_files)):
            raise IndexError(f"File index {file_index} out of range")
        
        filepath = self.available_files[file_index]
        result = {}
        
        with h5py.File(filepath, 'r') as f:
            if 'primitives' not in f:
                print("Warning: No 'primitives' group found in file")
                return result
            
            primitives = f['primitives']
            available_vars = list(primitives.keys())
            
            if not available_vars:
                print("Warning: Primitives group is empty")
                return result
            
            # If no specific variables requested, read all
            if variables is None:
                variables = available_vars
            
            for var in variables:
                if var in primitives:
                    result[var] = primitives[var][:]
                else:
                    print(f"Warning: Variable '{var}' not found. Available: {available_vars}")
        
        return result
    
    def read_levelset(self, file_index: int = 0) -> Dict[str, np.ndarray]:
        """Read levelset data from a specific file."""
        if not (0 <= file_index < len(self.available_files)):
            raise IndexError(f"File index {file_index} out of range")
        
        filepath = self.available_files[file_index]
        result = {}
        
        with h5py.File(filepath, 'r') as f:
            if 'levelset' in f and len(f['levelset'].keys()) > 0:
                levelset = f['levelset']
                for key in levelset.keys():
                    result[key] = levelset[key][:]
        
        return result
    
    def read_metadata(self, file_index: int = 0) -> Dict:
        """Read metadata from a specific file."""
        if not (0 <= file_index < len(self.available_files)):
            raise IndexError(f"File index {file_index} out of range")
        
        filepath = self.available_files[file_index]
        result = {}
        
        with h5py.File(filepath, 'r') as f:
            if 'metadata' in f:
                metadata = f['metadata']
                
                def read_metadata_recursive(group, result_dict):
                    for key in group.keys():
                        if isinstance(group[key], h5py.Group):
                            result_dict[key] = {}
                            read_metadata_recursive(group[key], result_dict[key])
                        else:
                            data = group[key][()]
                            # Handle different data types
                            if isinstance(data, bytes):
                                result_dict[key] = data.decode('utf-8')
                            elif hasattr(data, 'item') and data.size == 1:
                                result_dict[key] = data.item()
                            elif hasattr(data, 'tolist'):
                                result_dict[key] = data.tolist()
                            else:
                                result_dict[key] = data
                
                read_metadata_recursive(metadata, result)
        
        return result
    
    def get_time_from_file(self, file_index: int = 0) -> float:
        """Get the simulation time from a specific file."""
        if not (0 <= file_index < len(self.available_files)):
            raise IndexError(f"File index {file_index} out of range")
        
        filepath = self.available_files[file_index]
        
        with h5py.File(filepath, 'r') as f:
            if 'time' in f:
                return float(f['time'][()])
            else:
                # Extract from filename as fallback
                name = filepath.stem
                time_str = name.split('_')[1]
                return float(time_str)
    
    def create_coordinate_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create 3D coordinate arrays for the grid.
        
        Returns:
            X, Y, Z coordinate arrays with shape matching the grid
        """
        if self.grid_info is None:
            raise ValueError("Grid information not available")
        
        if any(coord is None for coord in [self.grid_info['gridX'], self.grid_info['gridY'], self.grid_info['gridZ']]):
            raise ValueError("Grid coordinates not available")
        
        # Create 3D coordinate arrays
        X, Y, Z = np.meshgrid(
            self.grid_info['gridX'],
            self.grid_info['gridY'],
            self.grid_info['gridZ'],
            indexing='ij'
        )
        
        return X, Y, Z


def create_sample_reader(domain_path: str) -> JAXFluidsReader:
    """Convenience function to create a reader and print basic info."""
    reader = JAXFluidsReader(domain_path)
    
    if reader.available_files:
        print("\n=== Basic Information ===")
        print(f"Number of files: {len(reader.available_files)}")
        print(f"Time range: {min(reader.get_available_times()):.6f} to {max(reader.get_available_times()):.6f}")
        
        if reader.grid_info:
            print(f"Grid shape: {reader.grid_info.get('shape', 'Unknown')}")
            print(f"Grid dimension: {reader.grid_info.get('dim', 'Unknown')}")
        
        print("\n=== First File Contents ===")
        info = reader.get_file_info(0)
        print(f"Root groups: {info['root_keys']}")
        
        # Check for primitives
        primitives = reader.read_primitives(0)
        if primitives:
            print(f"Available primitive variables: {list(primitives.keys())}")
        else:
            print("No primitive variables found in first file")
        
        # Check metadata
        metadata = reader.read_metadata(0)
        if metadata:
            print("Metadata available")
    
    return reader


if __name__ == "__main__":
    # Example usage
    domain_path = "../propeller_subsonic_wind_tunnel-2/domain"
    if os.path.exists(domain_path):
        reader = create_sample_reader(domain_path)
    else:
        print(f"Domain path {domain_path} not found")