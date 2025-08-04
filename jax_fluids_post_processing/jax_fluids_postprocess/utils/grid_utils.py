"""
Grid utilities for creating structured grids and handling spatial data.
"""

from typing import Dict, Tuple, List, Union
import numpy as np
import pyvista as pv


def create_structured_grid(
    data_dict: Dict[str, np.ndarray],
    grid_shape: Tuple[int, int, int],
    domain_bounds: List[float],
    exclude_keys: List[str] = None
) -> pv.StructuredGrid:
    """
    Create PyVista structured grid from flow data.
    
    Args:
        data_dict: Dictionary of flow variables
        grid_shape: Grid dimensions (nx, ny, nz)
        domain_bounds: Domain bounds [xmin, xmax, ymin, ymax, zmin, zmax]
        exclude_keys: Keys to exclude from grid data
        
    Returns:
        PyVista structured grid with flow data
    """
    if exclude_keys is None:
        exclude_keys = ['_metadata']
    
    nx, ny, nz = grid_shape
    xmin, xmax, ymin, ymax, zmin, zmax = domain_bounds
    
    # Create coordinate arrays
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create structured grid
    grid = pv.StructuredGrid(X, Y, Z)
    
    # Add data arrays
    for key, data in data_dict.items():
        if key in exclude_keys or not isinstance(data, np.ndarray):
            continue
            
        if data.shape == grid_shape:
            # Scalar data
            grid[key] = data.ravel(order='F')
        elif data.shape == (*grid_shape, 3):
            # Vector data (e.g., velocity components)
            # Reshape to (n_points, 3)
            vector_data = data.reshape(-1, 3, order='F')
            grid[key] = vector_data
    
    # Create velocity vector if components exist
    if all(f'velocity_{c}' in data_dict for c in ['u', 'v', 'w']):
        u = data_dict['velocity_u'].ravel(order='F')
        v = data_dict['velocity_v'].ravel(order='F')
        w = data_dict['velocity_w'].ravel(order='F')
        grid['velocity'] = np.column_stack([u, v, w])
    
    return grid


def compute_grid_spacing(
    grid_shape: Tuple[int, int, int],
    domain_bounds: List[float]
) -> Tuple[float, float, float]:
    """
    Compute grid spacing in each direction.
    
    Args:
        grid_shape: Grid dimensions (nx, ny, nz)
        domain_bounds: Domain bounds [xmin, xmax, ymin, ymax, zmin, zmax]
        
    Returns:
        Grid spacing (dx, dy, dz)
    """
    nx, ny, nz = grid_shape
    xmin, xmax, ymin, ymax, zmin, zmax = domain_bounds
    
    dx = (xmax - xmin) / (nx - 1) if nx > 1 else 0
    dy = (ymax - ymin) / (ny - 1) if ny > 1 else 0
    dz = (zmax - zmin) / (nz - 1) if nz > 1 else 0
    
    return dx, dy, dz


def create_uniform_grid(
    grid_shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> pv.StructuredGrid:
    """
    Create uniform structured grid.
    
    Args:
        grid_shape: Grid dimensions (nx, ny, nz)
        spacing: Grid spacing (dx, dy, dz)
        origin: Grid origin (x0, y0, z0)
        
    Returns:
        Empty PyVista structured grid
    """
    nx, ny, nz = grid_shape
    dx, dy, dz = spacing
    x0, y0, z0 = origin
    
    # Create coordinate arrays
    x = np.arange(nx) * dx + x0
    y = np.arange(ny) * dy + y0
    z = np.arange(nz) * dz + z0
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    return pv.StructuredGrid(X, Y, Z)


def extract_slice_from_grid(
    grid: pv.StructuredGrid,
    plane: str,
    position: float,
    variable: str
) -> np.ndarray:
    """
    Extract 2D slice from structured grid.
    
    Args:
        grid: PyVista structured grid
        plane: Plane orientation ('xy', 'xz', 'yz')
        position: Position along plane normal (world coordinates)
        variable: Variable name to extract
        
    Returns:
        2D array of variable data
    """
    if variable not in grid.array_names:
        raise ValueError(f"Variable '{variable}' not found in grid")
    
    # Create slice
    if plane == 'xy':
        slice_obj = grid.slice(normal='z', origin=[0, 0, position])
    elif plane == 'xz':
        slice_obj = grid.slice(normal='y', origin=[0, position, 0])
    elif plane == 'yz':
        slice_obj = grid.slice(normal='x', origin=[position, 0, 0])
    else:
        raise ValueError(f"Invalid plane '{plane}'. Use 'xy', 'xz', or 'yz'")
    
    return slice_obj[variable]


def interpolate_to_points(
    grid: pv.StructuredGrid,
    points: np.ndarray,
    variable: str
) -> np.ndarray:
    """
    Interpolate grid data to arbitrary points.
    
    Args:
        grid: PyVista structured grid
        points: Array of points (n_points, 3)
        variable: Variable to interpolate
        
    Returns:
        Interpolated values at points
    """
    if variable not in grid.array_names:
        raise ValueError(f"Variable '{variable}' not found in grid")
    
    # Create point cloud
    point_cloud = pv.PolyData(points)
    
    # Interpolate
    interpolated = point_cloud.sample(grid)
    
    return interpolated[variable]


def compute_gradient(
    data: np.ndarray,
    spacing: Tuple[float, float, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute gradient of 3D scalar field.
    
    Args:
        data: 3D scalar field (nx, ny, nz)
        spacing: Grid spacing (dx, dy, dz)
        
    Returns:
        Gradient components (grad_x, grad_y, grad_z)
    """
    dx, dy, dz = spacing
    
    grad_x = np.gradient(data, dx, axis=0)
    grad_y = np.gradient(data, dy, axis=1)
    grad_z = np.gradient(data, dz, axis=2)
    
    return grad_x, grad_y, grad_z


def compute_divergence(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    spacing: Tuple[float, float, float]
) -> np.ndarray:
    """
    Compute divergence of 3D vector field.
    
    Args:
        u, v, w: Velocity components (nx, ny, nz)
        spacing: Grid spacing (dx, dy, dz)
        
    Returns:
        Divergence field
    """
    dx, dy, dz = spacing
    
    du_dx = np.gradient(u, dx, axis=0)
    dv_dy = np.gradient(v, dy, axis=1)
    dw_dz = np.gradient(w, dz, axis=2)
    
    return du_dx + dv_dy + dw_dz