# Intelligent Boundary Conditions Endpoint

## Overview

The Intelligent Boundary Conditions endpoint automatically tags inlet and outlet faces of rocket nozzles and complex 3D internal flow geometries for JAX-Fluids simulations. This system eliminates the manual process of identifying boundary condition faces in complex geometries like rocket engines.

## Features

### 🚀 **Automated Face Detection**
- **Heuristic-based tagging**: Automatically identify inlet/outlet faces using geometric heuristics
- **Multiple algorithms**: Z-axis, X-axis, flow direction, and clustering-based methods
- **Rocket nozzle specific**: Optimized for converging-diverging nozzles, bell nozzles, and other rocket geometries

### 🎯 **Manual Interactive Tagging**
- **PyVista integration**: 3D interactive face selection with visualization
- **Matplotlib fallback**: 3D scatter plot interface for face selection
- **Console interface**: Text-based face selection for headless environments

### 📐 **Geometry Support**
- **Multiple formats**: STL, MSH (GMSH), OBJ, PLY files
- **CAD integration**: Experimental support for STEP/IGES files
- **Robust parsing**: Handles complex triangular and quad meshes

### ⚡ **JAX-Fluids Integration**
- **Native configuration**: Generates complete JAX-Fluids case configurations
- **Boundary masks**: Creates 3D masks for direct solver integration
- **SDF support**: Integrates with existing SDF/level-set workflows
- **Rocket conditions**: Built-in rocket engine operating conditions

## Installation

```bash
# Clone the repository (if part of larger project)
cd intelligent_boundary_conditions

# Install dependencies
pip install -r requirements.txt

# Optional: Install advanced visualization
pip install pyvista[all]

# Optional: Install JAX for GPU acceleration
pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Quick Start

### Basic Usage

```python
from intelligent_boundary_conditions import generate_intelligent_boundary_conditions

# Simple rocket nozzle boundary conditions
response = generate_intelligent_boundary_conditions(
    geometry_file="rocket_nozzle.stl",
    output_directory="./bc_output",
    tagging_method="z_axis_heuristic",
    nozzle_type="converging_diverging",
    flow_axis='x'
)

if response.success:
    print(f"Generated config: {response.config_file}")
    print(f"Boundary masks: {response.mask_directory}")
else:
    print(f"Error: {response.message}")
```

### Advanced Usage

```python
from intelligent_boundary_conditions import IntelligentBoundaryConditionsAPI
from intelligent_boundary_conditions.main_api import IntelligentBCRequest
from intelligent_boundary_conditions.face_tagger import TaggingMethod, RocketNozzleType

# Create detailed request
request = IntelligentBCRequest(
    geometry_file="complex_nozzle.msh",
    output_directory="./advanced_bc",
    tagging_method=TaggingMethod.AUTOMATIC_FLOW_DIRECTION,
    nozzle_type=RocketNozzleType.BELL_NOZZLE,
    flow_axis='x',
    fuel_type="hydrogen",
    chamber_pressure=7.0e6,  # Pa
    chamber_temperature=3600,  # K
    ambient_pressure=101325,  # Pa
    domain_resolution=(400, 200, 100),
    generate_masks=True,
    generate_config=True,
    sdf_integration=True
)

# Process request
api = IntelligentBoundaryConditionsAPI()
response = api.process_request(request)
```

### Manual Interactive Tagging

```python
# For complex geometries requiring manual verification
response = generate_intelligent_boundary_conditions(
    geometry_file="custom_nozzle.stl",
    output_directory="./manual_bc",
    manual_tagging=True,
    visualization=True  # Opens 3D interface
)
```

## API Reference

### Main Functions

#### `generate_intelligent_boundary_conditions()`
High-level function for most use cases.

**Parameters:**
- `geometry_file` (str): Path to geometry file (STL, MSH, etc.)
- `output_directory` (str): Directory to save outputs
- `tagging_method` (str): Method for face tagging
  - `"z_axis_heuristic"`: Tag based on Z-axis extremes
  - `"x_axis_heuristic"`: Tag based on X-axis extremes  
  - `"flow_direction_heuristic"`: Use normal vectors and flow direction
  - `"clustering_based"`: Machine learning clustering approach
- `nozzle_type` (str): Type of rocket nozzle
  - `"converging_diverging"`: CD nozzle (default)
  - `"bell_nozzle"`: Bell-shaped nozzle
  - `"conical_nozzle"`: Conical nozzle
  - `"aerospike"`: Aerospike nozzle
- `flow_axis` (str): Primary flow direction ('x', 'y', 'z')
- `fuel_type` (str): Fuel type ("hydrogen", "kerosene", "methane")
- `chamber_pressure` (float): Chamber pressure in Pa
- `chamber_temperature` (float): Chamber temperature in K
- `domain_resolution` (tuple): Grid resolution (nx, ny, nz)
- `manual_tagging` (bool): Use manual face selection
- `visualization` (bool): Show 3D visualization

#### `quick_rocket_nozzle_bc()`
Quick setup with sensible defaults for rocket nozzles.

### Core Classes

#### `GeometryParser`
Handles parsing of various geometry formats.

```python
from intelligent_boundary_conditions.geometry_parser import GeometryParser

parser = GeometryParser("nozzle.stl")
geometry_data = parser.parse_geometry()
summary = parser.get_geometry_summary()
```

#### `FaceTagger`
Implements face tagging algorithms.

```python
from intelligent_boundary_conditions.face_tagger import FaceTagger, TaggingMethod

tagger = FaceTagger(geometry_parser)
tagged_faces = tagger.auto_tag_faces(
    method=TaggingMethod.AUTOMATIC_FLOW_DIRECTION,
    flow_axis='x'
)
```

#### `BoundaryConditionGenerator`
Generates JAX-Fluids configurations and masks.

```python
from intelligent_boundary_conditions.boundary_condition_generator import (
    BoundaryConditionGenerator, RocketEngineConditions
)

# Generate flow conditions
flow_conditions = RocketEngineConditions.get_rocket_conditions(
    fuel_type="hydrogen",
    chamber_pressure=7e6
)

# Generate boundary conditions
bc_gen = BoundaryConditionGenerator(geometry_parser, face_tagger)
config = bc_gen.generate_jaxfluids_config(flow_conditions, domain_config)
```

## Rocket Engine Configuration

### Supported Fuel Types
- **Hydrogen**: H2/O2 combustion products (R = 4124 J/kg·K)
- **Kerosene**: RP-1/O2 combustion products (R = 287 J/kg·K)  
- **Methane**: CH4/O2 combustion products (R = 518 J/kg·K)

### Standard Operating Conditions
- **Chamber Pressure**: 6.9 MPa (1000 psi) - typical rocket engine
- **Chamber Temperature**: 3580 K - hydrogen combustion
- **Ambient Pressure**: 101.325 kPa (sea level)
- **Specific Heat Ratio**: 1.3 - typical for combustion products

### Example Configurations

#### High-Performance Rocket Engine
```python
flow_conditions = RocketEngineConditions.get_rocket_conditions(
    fuel_type="hydrogen",
    chamber_pressure=10e6,  # 10 MPa
    chamber_temperature=3800,  # K
    ambient_pressure=1000,  # High altitude
    gamma=1.25
)
```

#### Small-Scale Test Engine
```python
flow_conditions = RocketEngineConditions.get_rocket_conditions(
    fuel_type="kerosene",
    chamber_pressure=2e6,  # 2 MPa
    chamber_temperature=3000,  # K
    ambient_pressure=101325,  # Sea level
    gamma=1.35
)
```

## Geometry Requirements

### File Formats
- **STL**: ASCII or binary, recommended for simple geometries
- **MSH**: GMSH format, good for complex meshes with boundary tags
- **OBJ**: Wavefront OBJ, good for visualization meshes
- **PLY**: Stanford PLY, good for point cloud derived meshes

### Geometry Guidelines
1. **Closed surfaces**: Ensure watertight geometry for best results
2. **Proper orientation**: Normals should point outward from fluid domain
3. **Reasonable resolution**: Balance detail with computational efficiency
4. **Clean mesh**: Remove duplicate vertices and degenerate faces

### Coordinate System
- **Flow direction**: Typically X-axis (configurable)
- **Inlet**: Usually at minimum coordinate (upstream)
- **Outlet**: Usually at maximum coordinate (downstream)
- **Units**: Meters (SI units) recommended

## Integration with JAX-Fluids

### Generated Outputs

#### Configuration File (`intelligent_boundary_config.json`)
Complete JAX-Fluids case configuration including:
- Boundary conditions (inlet, outlet, wall)
- Initial conditions
- Material properties
- Domain discretization

#### Boundary Masks (`boundary_masks/`)
- `inlet_mask.npy`: Boolean mask for inlet faces
- `outlet_mask.npy`: Boolean mask for outlet faces  
- `wall_mask.npy`: Boolean mask for wall faces
- `mask_metadata.json`: Metadata and grid information

### JAX-Fluids Integration Example

```python
import jax.numpy as jnp
import numpy as np

# Load generated masks
inlet_mask = np.load("boundary_masks/inlet_mask.npy")
outlet_mask = np.load("boundary_masks/outlet_mask.npy")
wall_mask = np.load("boundary_masks/wall_mask.npy")

# Convert to JAX arrays
inlet_mask_jax = jnp.array(inlet_mask)
outlet_mask_jax = jnp.array(outlet_mask)
wall_mask_jax = jnp.array(wall_mask)

# Use in JAX-Fluids boundary condition application
def apply_boundary_conditions(state, masks):
    # Apply inlet conditions where inlet_mask is True
    # Apply outlet conditions where outlet_mask is True  
    # Apply wall conditions where wall_mask is True
    pass
```

## SDF Integration

For complex geometries that require SDF (Signed Distance Function) integration:

```python
# Generate with SDF support
response = generate_intelligent_boundary_conditions(
    geometry_file="complex_nozzle.msh",
    output_directory="./sdf_bc",
    sdf_integration=True,
    sdf_file="path/to/existing/sdf.npy"  # Optional: use existing SDF
)
```

The system will:
1. Tag boundary faces on the SDF geometry
2. Generate level-set compatible boundary conditions
3. Create masks that work with immersed boundary methods

## Troubleshooting

### Common Issues

#### Geometry Parsing Errors
```python
# Check file format support
from intelligent_boundary_conditions.geometry_parser import GeometryParser

try:
    parser = GeometryParser("geometry.xyz")  # Unsupported format
except ValueError as e:
    print(f"Format error: {e}")
```

#### Face Tagging Issues
```python
# Validate tagging results
validation = face_tagger.validate_tagging()
if not validation['valid']:
    print("Tagging errors:", validation['errors'])
    print("Tagging warnings:", validation['warnings'])
```

#### Visualization Problems
```python
# Check visualization dependencies
try:
    import pyvista as pv
    print("PyVista available for 3D visualization")
except ImportError:
    print("PyVista not available - install with: pip install pyvista")
```

### Performance Optimization

#### For Large Meshes
```python
# Use clustering-based method for complex geometries
response = generate_intelligent_boundary_conditions(
    geometry_file="large_nozzle.stl",
    tagging_method="clustering_based",
    domain_resolution=(100, 50, 1)  # Reduced resolution
)
```

#### Memory Management
```python
# Process in chunks for very large geometries
# Consider converting large STL files to lower resolution
```

## Examples

### Example 1: Converging-Diverging Nozzle
```python
# Standard CD nozzle with hydrogen propellant
response = generate_intelligent_boundary_conditions(
    geometry_file="cd_nozzle.stl",
    output_directory="./cd_nozzle_bc",
    tagging_method="z_axis_heuristic",
    nozzle_type="converging_diverging",
    fuel_type="hydrogen",
    chamber_pressure=7e6,
    domain_resolution=(300, 150, 1)
)
```

### Example 2: Bell Nozzle with Manual Verification
```python
# Bell nozzle with manual verification step
response = generate_intelligent_boundary_conditions(
    geometry_file="bell_nozzle.msh",
    output_directory="./bell_nozzle_bc",
    tagging_method="flow_direction_heuristic",
    nozzle_type="bell_nozzle",
    manual_tagging=True,  # Manual verification
    visualization=True,
    fuel_type="kerosene"
)
```

### Example 3: Complex 3D Nozzle
```python
# Complex geometry with clustering
response = generate_intelligent_boundary_conditions(
    geometry_file="complex_3d_nozzle.stl",
    output_directory="./complex_bc",
    tagging_method="clustering_based",
    nozzle_type="bell_nozzle",
    domain_resolution=(400, 200, 200),  # 3D domain
    generate_masks=True,
    sdf_integration=True
)
```

## Contributing

### Development Setup
```bash
git clone <repository>
cd intelligent_boundary_conditions
pip install -e .
pip install -r requirements-dev.txt  # If available
```

### Adding New Tagging Methods
Extend the `FaceTagger` class with new algorithms:

```python
def _tag_by_custom_method(self, **kwargs):
    # Implement custom tagging logic
    tagged_faces = {'inlet': [], 'outlet': [], 'wall': []}
    # ... custom algorithm ...
    return tagged_faces
```

### Adding New Geometry Formats
Extend the `GeometryParser` class:

```python
def _parse_custom_format(self):
    # Implement custom format parsing
    # Return geometry data dictionary
    pass
```

## License

This endpoint is part of the JAX-Fluids Agentic System. See the main repository for license information.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the examples for similar use cases
3. Open an issue in the main repository

## Related Documentation

- [JAX-Fluids Documentation](https://github.com/tumaer/JAXFLUIDS)
- [Internal Flow Endpoint](../Internal_flow_endpoint/README.md)
- [Immersed Boundary Endpoint](../immersed_boundary_endpoint_final/README.md) 