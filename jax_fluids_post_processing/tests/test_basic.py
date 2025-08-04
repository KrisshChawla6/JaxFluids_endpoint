"""Basic tests for package functionality."""

import pytest
import numpy as np
from pathlib import Path

def test_package_import():
    """Test that the package imports correctly."""
    import jax_fluids_postprocess as jfp
    
    # Test main API functions exist
    assert hasattr(jfp, 'process_simulation')
    assert hasattr(jfp, 'create_visualization') 
    assert hasattr(jfp, 'export_vtk')
    assert hasattr(jfp, 'create_animation')
    
    # Test classes exist
    assert hasattr(jfp, 'FluidProcessor')
    assert hasattr(jfp, 'InteractiveVisualizer')
    assert hasattr(jfp, 'DataReader')


def test_grid_utils():
    """Test grid utility functions."""
    from jax_fluids_postprocess.utils.grid_utils import compute_grid_spacing
    
    grid_shape = (10, 20, 30)
    domain_bounds = [0, 1, 0, 2, 0, 3]
    
    dx, dy, dz = compute_grid_spacing(grid_shape, domain_bounds)
    
    assert abs(dx - 1.0/9) < 1e-10
    assert abs(dy - 2.0/19) < 1e-10  
    assert abs(dz - 3.0/29) < 1e-10


def test_math_utils():
    """Test mathematical utility functions."""
    from jax_fluids_postprocess.utils.math_utils import compute_vorticity_magnitude
    
    # Create simple test velocity field
    nx, ny, nz = 10, 10, 10
    u = np.ones((nx, ny, nz))
    v = np.zeros((nx, ny, nz))
    w = np.zeros((nx, ny, nz))
    
    spacing = (1.0, 1.0, 1.0)
    
    vorticity = compute_vorticity_magnitude(u, v, w, spacing)
    
    assert vorticity.shape == (nx, ny, nz)
    assert isinstance(vorticity, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__])