# Professional Immersed Boundary SDF API

NASA-grade signed distance function computation for immersed boundary methods in computational fluid dynamics.

## Overview

This package provides a professional API for computing signed distance functions (SDFs) from 3D triangulated meshes, specifically designed for immersed boundary methods in CFD applications. It uses the production-quality [pysdf](https://github.com/sxyu/sdf) library for fast, parallel SDF computation.

## Features

- 🚀 **NASA-grade quality**: Production-ready SDF computation using proven algorithms
- ⚡ **High performance**: Parallel computation with configurable batch processing
- 🎯 **Precision**: Robust mode for accurate SDF computation near boundaries
- 📊 **Professional visualization**: High-quality isosurface extraction and plotting
- 💾 **Multiple export formats**: Binary, JSON, and JAX-Fluids compatible formats
- 🔧 **Easy API**: Clean, object-oriented interface with configuration management
- 📈 **Scalable**: Handles high-resolution grids (tested up to 200³)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually:
pip install numpy matplotlib scipy pysdf PyMCubes scikit-image
```

## Quick Start

```python
from sdf_api import ImmersedBoundaryAPI, SDFConfig

# Create configuration
config = SDFConfig(
    mesh_file="propeller.msh",
    domain_bounds=(-100, -150, -150, 150, 150, 150),
    resolution=(100, 100, 100),
    output_dir="results",
    output_name="propeller_sdf",
    plot=True,
    robust_mode=True
)

# Run API
api = ImmersedBoundaryAPI()
result, saved_files = api.run(config)

print(f"SDF computed in {result.computation_time:.2f} seconds")
print(f"Files saved: {list(saved_files.keys())}")
```

## API Reference

### SDFConfig

Configuration class for SDF computation:

```python
@dataclass
class SDFConfig:
    mesh_file: str                    # Path to Gmsh mesh file
    domain_bounds: Tuple[float, ...]  # (xmin, ymin, zmin, xmax, ymax, zmax)
    resolution: Tuple[int, int, int]  # (nx, ny, nz) grid resolution
    output_dir: str = "results"       # Output directory
    output_name: str = "sdf_result"   # Base name for output files
    plot: bool = True                 # Generate visualization
    save_binary: bool = True          # Save binary pickle file
    save_json: bool = True            # Save JSON metadata
    export_jaxfluids: bool = True     # Export JAX-Fluids format
    robust_mode: bool = True          # Use robust SDF computation
    batch_size: int = 100000          # Batch size for computation
```

### ImmersedBoundaryAPI

Main API class:

```python
class ImmersedBoundaryAPI:
    def __init__(self):
        """Initialize the API"""
    
    def compute_sdf(self, config: SDFConfig) -> SDFResult:
        """Compute signed distance function"""
    
    def visualize_sdf(self, result: SDFResult, show_plot: bool = True, 
                     save_path: Optional[str] = None) -> None:
        """Visualize zero-level contour"""
    
    def run(self, config: SDFConfig) -> Tuple[SDFResult, Dict[str, str]]:
        """Main entry point - compute SDF and save results"""
```

### SDFResult

Result container:

```python
@dataclass
class SDFResult:
    sdf_values: np.ndarray           # 3D SDF grid
    grid_points: np.ndarray          # Grid coordinates
    domain_bounds: Tuple[float, ...] # Domain boundaries
    resolution: Tuple[int, int, int] # Grid resolution
    mesh_file: str                   # Source mesh file
    computation_time: float          # Computation time
    timestamp: str                   # Computation timestamp
    config: SDFConfig               # Configuration used
    
    def save_binary(self, filepath: str) -> None:
        """Save as pickle file"""
    
    def export_for_jaxfluids(self, filepath: str) -> None:
        """Export for JAX-Fluids"""
    
    @classmethod
    def load_binary(cls, filepath: str) -> 'SDFResult':
        """Load from pickle file"""
```

## Usage Examples

### Basic Usage

```python
# Simple propeller SDF
config = SDFConfig(
    mesh_file="propeller.msh",
    domain_bounds=(-50, -50, -50, 50, 50, 50),
    resolution=(80, 80, 80)
)

api = ImmersedBoundaryAPI()
result, files = api.run(config)
```

### High-Resolution Production Run

```python
# High-resolution for production CFD
config = SDFConfig(
    mesh_file="complex_geometry.msh",
    domain_bounds=(-100, -150, -150, 150, 150, 150),
    resolution=(200, 200, 200),
    output_name="production_sdf",
    robust_mode=True,
    batch_size=100000
)

api = ImmersedBoundaryAPI()
result, files = api.run(config)
```

### Configuration from Dictionary

```python
# Load config from JSON/dict
config_dict = {
    "mesh_file": "geometry.msh",
    "domain_bounds": (-10, -10, -10, 10, 10, 10),
    "resolution": (50, 50, 50),
    "plot": False
}

config = SDFConfig.from_dict(config_dict)
result, files = api.run(config)
```

### Loading Previous Results

```python
# Load previously computed SDF
result = SDFResult.load_binary("results/propeller_sdf.pkl")
api = ImmersedBoundaryAPI()
api.visualize_sdf(result, show_plot=True)
```

## Output Formats

The API generates several output formats:

1. **Binary (.pkl)**: Complete result object for Python loading
2. **Metadata (.json)**: Configuration and statistics in JSON format
3. **JAX-Fluids (.json)**: Compatible format for JAX-Fluids CFD solver
4. **Visualization (.png)**: High-quality 3D plots of the φ=0 contour

### JAX-Fluids Format

```json
{
  "domain_bounds": [-100, -150, -150, 150, 150, 150],
  "resolution": [100, 100, 100],
  "grid_spacing": [2.0202, 3.0303, 3.0303],
  "sdf_values": [...],
  "mesh_file": "propeller.msh",
  "timestamp": "2024-01-15T10:30:00"
}
```

## Supported Mesh Formats

- **Gmsh (.msh)**: Version 4.1 ASCII format
- **Triangulated surfaces**: 3D triangle elements
- **Any size**: Tested with meshes up to 100K+ triangles

## Performance

Typical performance on modern hardware:

| Resolution | Points | Time (approx) | Memory |
|------------|--------|---------------|--------|
| 50³        | 125K   | 5-10 seconds  | ~100MB |
| 100³       | 1M     | 30-60 seconds | ~1GB   |
| 200³       | 8M     | 5-10 minutes  | ~8GB   |

## Testing

Run the test suite:

```bash
python test_api.py
```

This will:
- Test basic API functionality
- Test configuration serialization
- Test result loading/saving
- Run high-resolution computation
- Verify all output formats

## Error Handling

The API includes comprehensive error handling:

- Mesh file validation
- Domain bounds checking  
- Memory usage monitoring
- Progress reporting
- Graceful failure recovery

## Integration with JAX-Fluids

The generated SDF can be directly used with JAX-Fluids:

```python
# Generate SDF
config = SDFConfig(
    mesh_file="geometry.msh",
    domain_bounds=(-50, -50, -50, 50, 50, 50),
    resolution=(100, 100, 100),
    export_jaxfluids=True
)

result, files = api.run(config)

# Use in JAX-Fluids
jax_config_file = files['jaxfluids']
# Load this in your JAX-Fluids simulation setup
```

## Advanced Features

### Custom Visualization

```python
# Custom visualization settings
api = ImmersedBoundaryAPI()
result = api.compute_sdf(config)

# Save without showing
api.visualize_sdf(result, show_plot=False, save_path="custom_viz.png")
```

### Batch Processing

```python
# Process multiple geometries
geometries = ["prop1.msh", "prop2.msh", "prop3.msh"]

for mesh_file in geometries:
    config = SDFConfig(
        mesh_file=mesh_file,
        domain_bounds=(-50, -50, -50, 50, 50, 50),
        resolution=(100, 100, 100),
        output_name=f"sdf_{Path(mesh_file).stem}"
    )
    result, files = api.run(config)
```

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues and questions:
- Check the test script for usage examples
- Review the API documentation above
- Ensure all dependencies are correctly installed

## Citation

If you use this in academic work, please cite the underlying pysdf library:
- [pysdf GitHub repository](https://github.com/sxyu/sdf) 