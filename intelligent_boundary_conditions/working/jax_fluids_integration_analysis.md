# JAX-Fluids Integration Analysis
# Virtual Inlet/Outlet Faces for Rocket Nozzle Simulations

## Overview

This document analyzes the integration of our successful circular virtual face creator with JAX-Fluids for 3D internal supersonic flow simulations in rocket nozzles.

## 1. Virtual Face Reconstruction Analysis

### 1.1 How Virtual Faces are Reconstructed

Our circular face creator implements the following pipeline:

```
Mesh Input (.msh) → Boundary Edge Detection → Circle Fitting → Virtual Face Creation
```

#### Key Steps:
1. **Boundary Edge Extraction**: Uses PyVista's `extract_feature_edges()` with non-manifold edge detection
2. **Spatial Clustering**: Groups edge points by X-coordinate to separate inlet (X≈0) and outlet (X≈1717)
3. **Circle Fitting**: Least-squares optimization for center and radius in Y-Z plane
4. **Triangulation**: Fan-based triangulation from center point (64 triangular segments per face)

#### Node Data Storage:
```python
face_data = {
    'triangles': np.array(triangles),        # Shape: (n_triangles, 3, 3) - vertex coordinates
    'center': center_point,                  # Shape: (3,) - circle center [X, Y, Z]
    'radius': radius,                        # Scalar - fitted circle radius
    'x_position': x_pos,                     # Scalar - axial position
    'boundary_points': boundary_points,      # Shape: (n_points, 3) - original edge points
    'fit_error': fit_error,                  # Scalar - fitting quality metric
    'n_triangles': len(triangles)            # Scalar - triangle count
}
```

### 1.2 Mesh Data Structure
- **Original mesh**: 415,164 wall surface triangles
- **Virtual inlet**: 64 triangular faces at X=0.0, R=313.6
- **Virtual outlet**: 64 triangular faces at X=1717.2, R=602.7
- **Combined mesh**: Face type scalars (0=inlet, 1=wall, 2=outlet)

## 2. JAX-Fluids Boundary Condition Implementation

### 2.1 JAX-Fluids Boundary Condition Types

Based on documentation research, JAX-Fluids supports:

- **Primitive BCs**: Symmetry, Periodic, Wall, Dirichlet, Neumann
- **Two-phase flows**: Level-set method and diffuse-interface method
- **Immersed boundaries**: Level-set based solid boundaries
- **Cartesian grid**: Finite-volume method on structured grids

### 2.2 Typical JAX-Fluids Workflow

```python
# Typical JAX-Fluids setup
domain_info = {
    "x": {"cells": 256, "range": [0.0, 1.0]},
    "y": {"cells": 128, "range": [-0.5, 0.5]}, 
    "z": {"cells": 128, "range": [-0.5, 0.5]}
}

boundary_conditions = {
    "x_min": {"type": "DIRICHLET", "value": inlet_conditions},
    "x_max": {"type": "NEUMANN", "value": outlet_conditions},
    "y_min": "SYMMETRY", "y_max": "SYMMETRY",
    "z_min": "SYMMETRY", "z_max": "SYMMETRY"
}
```

### 2.3 Challenge: Mesh vs Grid

**Problem**: JAX-Fluids uses structured Cartesian grids, but our mesh is unstructured triangular surfaces.

**Solutions**:
1. **Level-Set Approach**: Convert virtual faces to level-set function φ(x,y,z)
2. **Grid Mapping**: Project virtual face coordinates onto structured grid points
3. **Immersed Boundary**: Use JAX-Fluids' built-in level-set immersed boundary method

## 3. Integration Strategies

### 3.1 Strategy 1: Level-Set Conversion

Convert virtual faces to signed distance function:

```python
def virtual_faces_to_levelset(inlet_data, outlet_data, grid_coords):
    """Convert virtual circular faces to level-set function"""
    
    # Create level-set for solid geometry (walls)
    phi_solid = compute_wall_levelset(grid_coords)
    
    # Define inlet region (circular opening)
    inlet_center = inlet_data['center']
    inlet_radius = inlet_data['radius']
    inlet_mask = compute_circular_mask(grid_coords, inlet_center, inlet_radius)
    
    # Define outlet region  
    outlet_center = outlet_data['center']
    outlet_radius = outlet_data['radius']
    outlet_mask = compute_circular_mask(grid_coords, outlet_center, outlet_radius)
    
    # Combine: negative inside fluid, positive inside solid
    phi_combined = np.where(inlet_mask | outlet_mask, -1.0, phi_solid)
    
    return phi_combined, inlet_mask, outlet_mask
```

### 3.2 Strategy 2: Boundary Condition Masks

Create 3D boolean arrays for boundary condition application:

```python
def create_boundary_masks(inlet_data, outlet_data, grid_shape, domain_bounds):
    """Create 3D masks for inlet/outlet boundary conditions"""
    
    inlet_mask = np.zeros(grid_shape, dtype=bool)
    outlet_mask = np.zeros(grid_shape, dtype=bool)
    
    # Map virtual face coordinates to grid indices
    inlet_indices = map_face_to_grid(inlet_data, grid_shape, domain_bounds)
    outlet_indices = map_face_to_grid(outlet_data, grid_shape, domain_bounds)
    
    inlet_mask[inlet_indices] = True
    outlet_mask[outlet_indices] = True
    
    return inlet_mask, outlet_mask

def apply_boundary_conditions(state, inlet_mask, outlet_mask, inlet_bc, outlet_bc):
    """Apply boundary conditions using masks"""
    
    # Apply inlet BC (e.g., fixed pressure/temperature)
    state = state.at[inlet_mask].set(inlet_bc)
    
    # Apply outlet BC (e.g., zero gradient)
    state = apply_outlet_bc(state, outlet_mask, outlet_bc)
    
    return state
```

### 3.3 Strategy 3: JAX-Fluids Integration

Integrate with existing JAX-Fluids boundary condition system:

```python
def create_jaxfluids_config(inlet_data, outlet_data, flow_conditions):
    """Generate JAX-Fluids compatible configuration"""
    
    config = {
        "case": {
            "case_name": "rocket_nozzle_internal_flow"
        },
        "domain": {
            "x": {"cells": 512, "range": [0.0, inlet_data['x_position'] + outlet_data['x_position']]},
            "y": {"cells": 256, "range": [-outlet_data['radius']*1.5, outlet_data['radius']*1.5]},
            "z": {"cells": 256, "range": [-outlet_data['radius']*1.5, outlet_data['radius']*1.5]}
        },
        "boundary_conditions": {
            "x_min": {
                "type": "DIRICHLET", 
                "primitive_variables": {
                    "pressure": flow_conditions['inlet_pressure'],
                    "temperature": flow_conditions['inlet_temperature'],
                    "velocity": [flow_conditions['inlet_velocity'], 0.0, 0.0]
                }
            },
            "x_max": {
                "type": "NEUMANN",
                "primitive_variables": "ZERO_GRADIENT"
            }
        },
        "initial_condition": flow_conditions['initial_state'],
        "levelset": {
            "model": "FLUID_SOLID_INTERFACE",
            "geometry": virtual_faces_to_levelset(inlet_data, outlet_data)
        }
    }
    
    return config
```

## 4. Data Storage and Transfer

### 4.1 Virtual Face Data Format

```python
@dataclass
class VirtualFaceData:
    """Standard format for virtual face data"""
    face_type: str                    # "inlet" or "outlet"
    center: np.ndarray               # Shape: (3,) [X, Y, Z]
    radius: float                    # Circle radius
    x_position: float                # Axial position
    triangles: np.ndarray            # Shape: (n_triangles, 3, 3)
    boundary_points: np.ndarray      # Shape: (n_points, 3)
    normal_vector: np.ndarray        # Shape: (3,) unit normal
    area: float                      # Total face area
    
class RocketNozzleGeometry:
    """Complete nozzle geometry with virtual faces"""
    inlet_face: VirtualFaceData
    outlet_face: VirtualFaceData
    wall_mesh: Any                   # PyVista mesh for walls
    bounding_box: Tuple[np.ndarray, np.ndarray]  # Min/max coordinates
```

### 4.2 JAX-Fluids Interface

```python
def convert_to_jaxfluids_format(rocket_geometry: RocketNozzleGeometry, 
                               grid_resolution: Tuple[int, int, int],
                               flow_conditions: dict) -> dict:
    """Convert virtual face data to JAX-Fluids input format"""
    
    # Create structured grid
    grid = create_structured_grid(rocket_geometry.bounding_box, grid_resolution)
    
    # Convert to level-set
    levelset = geometry_to_levelset(rocket_geometry, grid)
    
    # Create boundary condition masks
    bc_masks = create_boundary_masks(rocket_geometry, grid)
    
    # Generate JAX-Fluids config
    jaxfluids_config = {
        "domain": grid_config,
        "levelset": levelset,
        "boundary_conditions": bc_masks,
        "initial_conditions": flow_conditions,
        "numerical_setup": {
            "spatial_reconstruction": "WENO5",
            "riemann_solver": "HLLC", 
            "time_integration": "RK3"
        }
    }
    
    return jaxfluids_config
```

## 5. Implementation Recommendations

### 5.1 Immediate Next Steps

1. **Create Level-Set Converter**: Implement `virtual_faces_to_levelset()` function
2. **Test with Simple Geometry**: Start with cylindrical test case before complex nozzle
3. **Validate Against Known Solutions**: Compare with analytical rocket nozzle flow solutions
4. **Benchmark Performance**: Measure computational cost vs accuracy trade-offs

### 5.2 Integration Architecture

```
Virtual Face Creator → Level-Set Converter → JAX-Fluids Interface → Simulation
       ↓                      ↓                     ↓                ↓
  - Circle fitting      - SDF generation     - Config creation   - CFD solve
  - Triangle mesh       - Grid mapping       - BC application    - Result output
  - Boundary points     - Mask creation      - Initial conditions
```

### 5.3 Validation Strategy

1. **Geometric Validation**: Verify level-set accurately represents virtual faces
2. **Flow Validation**: Compare inlet/outlet flow rates with analytical predictions
3. **Convergence Study**: Grid resolution effects on solution accuracy
4. **Physical Validation**: Check for expected nozzle flow phenomena (sonic throat, expansion)

## 6. Technical Challenges and Solutions

### 6.1 Resolution Requirements
- **Challenge**: Virtual faces need sufficient grid resolution to capture circular geometry
- **Solution**: Adaptive mesh refinement or high-resolution regions near inlet/outlet

### 6.2 Boundary Condition Consistency
- **Challenge**: Ensuring mass conservation across inlet/outlet
- **Solution**: Implement conserved variable boundary conditions with feedback control

### 6.3 Level-Set Accuracy
- **Challenge**: Circular faces may become pixelated on Cartesian grid
- **Solution**: Sub-grid level-set interpolation or higher-order boundary treatment

## 7. Expected Outputs

### 7.1 Simulation Results
- Pressure, temperature, velocity fields throughout nozzle
- Mass flow rate conservation check
- Boundary layer development along walls
- Shock structure (if supersonic)

### 7.2 Boundary Condition Validation
- Inlet: Specified total pressure/temperature maintained
- Outlet: Zero-gradient conditions properly applied
- Walls: No-slip velocity, adiabatic or isothermal temperature

This analysis provides a roadmap for integrating our successful virtual face creator with JAX-Fluids for rocket nozzle internal flow simulations. 