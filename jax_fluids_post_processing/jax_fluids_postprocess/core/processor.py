"""
Main FluidProcessor class for JAX-Fluids post-processing.
Provides a clean, functional interface for processing simulation results.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
import json

from ..io.h5_reader import JAXFluidsReader
from .data_extractor import DataExtractor


class FluidProcessor:
    """
    Main processor class for JAX-Fluids simulation results.
    
    Provides a high-level interface for:
    - Loading simulation data
    - Extracting flow variables  
    - Computing derived quantities
    - Saving results and metadata
    
    Example:
        >>> processor = FluidProcessor("simulation/domain")
        >>> flow_data = processor.extract_flow_variables()
        >>> processor.save_summary(flow_data)
    """
    
    def __init__(
        self, 
        results_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        mesh_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the fluid processor.
        
        Args:
            results_path: Path to simulation results directory containing .h5 files
            output_path: Optional path for saving outputs (default: results_path/postprocess)
            mesh_path: Optional path to mesh file for visualization
        """
        self.results_path = Path(results_path)
        self.output_path = Path(output_path) if output_path else self.results_path / "postprocess"
        self.mesh_path = Path(mesh_path) if mesh_path else None
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize reader and extractor
        self.reader = JAXFluidsReader(str(self.results_path))
        self.extractor = DataExtractor(self.reader)
        
        # Try to load official JAX-Fluids data if available
        self.official_data = None
        self._try_load_official_data()
        
        print(f"📂 Loaded JAX-Fluids data from: {self.results_path}")
        print(f"✓ Found {self.reader.get_num_time_steps()} time steps")
        print(f"✓ Grid shape: {self.reader.get_grid_shape()}")
        
        time_range = self.reader.get_time_range()
        print(f"✓ Time range: {time_range[0]:.3f} to {time_range[1]:.3f}")
    
    def _try_load_official_data(self):
        """Try to load data using official JAX-Fluids API if available."""
        try:
            from jaxfluids_postprocess import load_data
            print("🔄 Loading data with official JAX-Fluids API...")
            self.official_data = load_data(
                str(self.results_path), 
                ["velocity", "pressure", "density"]
            )
            print(f"✓ Official data loaded: {list(self.official_data.data.keys())}")
        except ImportError:
            print("ℹ️  Official JAX-Fluids API not available, using custom reader")
        except Exception as e:
            print(f"⚠️  Official load_data failed: {e}. Using custom reader.")
    
    def extract_flow_variables(self, time_index: int = -1) -> Dict[str, np.ndarray]:
        """
        Extract flow variables from simulation data.
        
        Args:
            time_index: Time step index (-1 for last step)
            
        Returns:
            Dictionary containing extracted flow variables:
            - 'velocity_u', 'velocity_v', 'velocity_w': Velocity components
            - 'velocity_magnitude': Velocity magnitude
            - 'pressure': Pressure field
            - 'density': Density field  
            - 'temperature': Temperature field (if available)
            - 'vorticity': Vorticity magnitude
            - 'q_criterion': Q-criterion for vortex detection
        """
        print(f"📊 Extracting flow variables from time step {time_index}...")
        
        # Get simulation time for this step
        times = self.reader.get_times()
        sim_time = times[time_index]
        print(f"Simulation time: {sim_time:.6f}")
        
        # Extract primitive variables
        flow_data = {}
        
        # Get basic variables using extractor
        velocity = self.extractor.extract_velocity_components(time_index)
        flow_data.update(velocity)
        
        # Compute velocity magnitude
        if all(k in flow_data for k in ['velocity_u', 'velocity_v', 'velocity_w']):
            u, v, w = flow_data['velocity_u'], flow_data['velocity_v'], flow_data['velocity_w']
            flow_data['velocity_magnitude'] = np.sqrt(u**2 + v**2 + w**2)
            vel_range = (flow_data['velocity_magnitude'].min(), flow_data['velocity_magnitude'].max())
            print(f"✓ Velocity magnitude: {vel_range[0]:.3f} to {vel_range[1]:.3f} m/s")
        
        # Extract other primitive variables
        primitives = self.reader.read_primitives(time_index, ['pressure', 'density', 'temperature'])
        flow_data.update(primitives)
        
        # Compute derived quantities
        try:
            vorticity = self.extractor.compute_vorticity_magnitude(time_index)
            flow_data['vorticity'] = vorticity
            vort_range = (vorticity.min(), vorticity.max())
            print(f"✓ Vorticity magnitude: {vort_range[0]:.3e} to {vort_range[1]:.3e} /s")
        except Exception as e:
            print(f"⚠️  Could not compute vorticity: {e}")
        
        try:
            q_criterion = self.extractor.compute_q_criterion(time_index)
            flow_data['q_criterion'] = q_criterion
            q_range = (q_criterion.min(), q_criterion.max())
            print(f"✓ Q-criterion: {q_range[0]:.3e} to {q_range[1]:.3e}")
        except Exception as e:
            print(f"⚠️  Could not compute Q-criterion: {e}")
        
        # Add metadata
        flow_data['_metadata'] = {
            'time_index': time_index,
            'simulation_time': sim_time,
            'grid_shape': self.reader.get_grid_shape(),
            'domain_bounds': self.reader.get_domain_bounds(),
            'extraction_timestamp': self._get_timestamp()
        }
        
        return flow_data
    
    def get_simulation_metadata(self) -> Dict[str, Any]:
        """Get simulation metadata and configuration."""
        metadata = {
            'num_time_steps': self.reader.get_num_time_steps(),
            'grid_shape': self.reader.get_grid_shape(),
            'time_range': self.reader.get_time_range(),
            'domain_bounds': self.reader.get_domain_bounds(),
            'available_variables': self.reader.get_available_variables(),
            'results_path': str(self.results_path),
            'output_path': str(self.output_path)
        }
        
        if self.mesh_path:
            metadata['mesh_path'] = str(self.mesh_path)
            
        return metadata
    
    def get_grid_info(self) -> Dict[str, Any]:
        """Get grid information and spacing."""
        grid_shape = self.reader.get_grid_shape()
        domain_bounds = self.reader.get_domain_bounds()
        
        # Compute grid spacing
        dx = (domain_bounds[1] - domain_bounds[0]) / (grid_shape[0] - 1)
        dy = (domain_bounds[3] - domain_bounds[2]) / (grid_shape[1] - 1) 
        dz = (domain_bounds[5] - domain_bounds[4]) / (grid_shape[2] - 1)
        
        return {
            'shape': grid_shape,
            'bounds': domain_bounds,
            'spacing': (dx, dy, dz),
            'dimensions': len(grid_shape)
        }
    
    def get_time_info(self) -> Dict[str, Any]:
        """Get time step information."""
        times = self.reader.get_times()
        time_range = self.reader.get_time_range()
        
        return {
            'num_steps': len(times),
            'times': times.tolist(),
            'range': time_range,
            'dt_avg': np.mean(np.diff(times)) if len(times) > 1 else 0.0
        }
    
    def save_summary(self, flow_data: Dict[str, np.ndarray]) -> Path:
        """
        Save processing summary and metadata to JSON file.
        
        Args:
            flow_data: Flow data dictionary from extract_flow_variables()
            
        Returns:
            Path to saved summary file
        """
        summary = {
            'processing_info': {
                'package_version': "1.0.0",
                'timestamp': self._get_timestamp(),
                'results_path': str(self.results_path),
                'output_path': str(self.output_path)
            },
            'simulation_metadata': self.get_simulation_metadata(),
            'grid_info': self.get_grid_info(),
            'time_info': self.get_time_info(),
            'extracted_variables': {}
        }
        
        # Add info about extracted variables
        for var_name, var_data in flow_data.items():
            if isinstance(var_data, np.ndarray):
                summary['extracted_variables'][var_name] = {
                    'shape': var_data.shape,
                    'dtype': str(var_data.dtype),
                    'min': float(var_data.min()),
                    'max': float(var_data.max()),
                    'mean': float(var_data.mean()),
                    'std': float(var_data.std())
                }
        
        # Save to file
        summary_path = self.output_path / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Summary saved: {summary_path}")
        return summary_path
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()