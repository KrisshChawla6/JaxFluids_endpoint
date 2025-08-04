# Intelligent Boundary Conditions - Final Production Endpoint

A comprehensive, production-ready tool for automatically generating JAX-Fluids compatible internal flow simulations from mesh files. This endpoint intelligently detects virtual inlet/outlet faces and creates the necessary boundary condition masks and configurations for supersonic internal flow simulations.

## Overview

This tool bridges the gap between complex 3D mesh geometries and JAX-Fluids' Cartesian grid-based CFD solver by:

1. **Automatically detecting** circular inlet/outlet openings in rocket nozzle geometries
2. **Generating virtual boundary faces** with precise geometric fitting
3. **Creating 3D boundary masks** mapped to JAX-Fluids' Cartesian grid
4. **Producing complete JAX-Fluids configurations** with forcing terms for internal boundary conditions
5. **Setting up production-ready simulation directories** with all necessary files

## Key Technical Innovation

### Virtual Boundary Conditions via Forcing System
Unlike traditional CFD boundary conditions applied at domain faces, this approach implements **virtual internal boundary conditions** using JAX-Fluids' native forcing system:

- **Inlet forcing**: Mass flow and temperature conditions applied at detected inlet regions
- **Outlet forcing**: Pressure or outflow conditions at detected outlet regions  
- **Immersed boundaries**: SDF-based representation of solid rocket walls
- **Domain boundaries**: Symmetry conditions on all external faces

This approach enables **true internal flow simulations** within complex hollow geometries.

## Architecture

```
intelligent_BC_final/
├── __init__.py                          # Package initialization
├── intelligent_boundary_processor.py    # Main orchestration class
├── core/                               # Core processing modules
│   ├── __init__.py
│   ├── virtual_face_detector.py        # Detects circular inlet/outlet faces
│   ├── sdf_generator.py               # Generates signed distance functions
│   ├── mask_generator.py              # Creates 3D boundary masks
│   └── jax_config_generator.py        # Generates JAX-Fluids configurations
├── example_usage.py                   # Comprehensive usage examples
└── README.md                          # This file
```

## Installation & Dependencies

### Required Dependencies
```bash
# Core scientific computing
pip install numpy scipy matplotlib

# Mesh processing and visualization  
pip install pyvista meshio

# Machine learning for clustering
pip install scikit-learn

# JAX-Fluids (for running simulations)
pip install jaxfluids

# Optional: Gmsh for mesh manipulation
pip install gmsh
```

### System Requirements
- Python 3.8+
- 8GB+ RAM (for large grids)
- JAX-Fluids compatible system

## Quick Start

### Basic Usage
```python
from intelligent_BC_final import IntelligentBoundaryProcessor

# Create processor
processor = IntelligentBoundaryProcessor(
    mesh_file="path/to/rocket_engine.msh",
    output_dir="my_rocket_simulation"
)

# Process mesh and generate complete simulation setup
results = processor.process_mesh()

# Get summary
print(processor.get_processing_summary())

# Run simulation (requires JAX-Fluids)
# cd my_rocket_simulation
# python quick_start.py
```

### Advanced Usage
```python
# Custom domain and grid resolution
processor = IntelligentBoundaryProcessor(
    mesh_file="rocket_engine.msh",
    output_dir="high_res_simulation",
    domain_bounds=[-300, -1000, -1000, 2000, 1000, 1000],
    grid_shape=(160, 80, 80)  # Higher resolution
)

results = processor.process_mesh()

# Customize flow conditions
processor.config_generator.customize_forcing_parameters(
    inlet_mass_flow=25.0,
    inlet_temperature=1800.0,
    outlet_pressure=500000.0
)

# Save updated configuration
processor.config_generator.save_configurations("config/")
```

## Technical Approach

### 1. Virtual Face Detection
- **Boundary edge extraction** from mesh geometry
- **Clustering algorithms** (DBSCAN) to separate inlet/outlet regions
- **Circle fitting** with least-squares optimization
- **Geometric validation** and quality checks

### 2. SDF Generation  
- **PyVista-based** distance computation on Cartesian grids
- **Inside/outside determination** using ray casting
- **Proper sign handling** (negative inside, positive outside)
- **Grid-aligned output** for JAX-Fluids compatibility

### 3. Mask Generation
- **3D boolean arrays** identifying forcing regions
- **Vectorized distance calculations** for efficiency
- **Cylindrical region mapping** from virtual faces to grid
- **Overlap detection** and validation

### 4. Configuration Generation
- **Complete JAX-Fluids setup** with all required sections
- **Forcing system integration** for virtual boundary conditions
- **Immersed boundary levelset** configuration
- **Production-ready numerical parameters**

## Output Structure

Each processed mesh generates a complete simulation directory:

```
simulation_output/
├── config/                      # JAX-Fluids configuration files
│   ├── rocket_setup.json        # Case setup with forcing terms
│   ├── numerical_setup.json     # Numerical methods configuration
│   └── simulation_parameters.json # Runtime parameters
├── masks/                       # Virtual boundary condition masks  
│   ├── inlet_boundary_mask.npy  # 3D inlet forcing mask
│   ├── outlet_boundary_mask.npy # 3D outlet forcing mask
│   └── mask_metadata.json       # Mask generation metadata
├── sdf/                         # Signed distance function
│   └── rocket_sdf.npy          # SDF for immersed boundaries
├── virtual_faces/               # Virtual face detection results
│   ├── inlet_points.npy         # Detected inlet boundary points
│   └── outlet_points.npy        # Detected outlet boundary points
├── output/                      # Simulation output (created during run)
├── logs/                        # Simulation and processing logs
├── run_simulation.py            # Main JAX-Fluids runner
├── quick_start.py              # One-click simulation starter
└── README.md                   # Simulation-specific documentation
```

## Validation & Quality Assurance

### Geometric Validation
- **Virtual face detection accuracy** using geometric consistency checks
- **SDF quality verification** with inside/outside point counting
- **Mask coverage analysis** ensuring proper boundary representation
- **Configuration validation** against JAX-Fluids requirements

### Physical Validation
- **Mass conservation** through inlet/outlet flow monitoring
- **Pressure gradient consistency** along nozzle axis
- **Mach number progression** from subsonic to supersonic
- **Temperature distribution** following expansion physics

## Examples

See `example_usage.py` for comprehensive examples including:

- **Basic usage** with default parameters
- **Custom domain and grid** specifications  
- **Step-by-step processing** for debugging
- **Production workflow** for multiple simulation cases
- **Parameter sweeps** for design optimization

## Performance Characteristics

### Processing Time
- **Small meshes** (<50k points): ~30 seconds
- **Medium meshes** (50k-200k points): ~2-5 minutes  
- **Large meshes** (>200k points): ~5-15 minutes

### Memory Usage
- **SDF generation**: ~8 bytes per grid point
- **Mask storage**: ~1 byte per grid point
- **Peak memory**: ~2x grid size during processing

### Grid Scaling
- **128×64×64**: Recommended for development/testing
- **160×80×80**: Good balance of accuracy and speed
- **256×128×128**: High-resolution production runs

## Troubleshooting

### Common Issues

1. **Virtual face detection fails**
   - Check mesh quality and completeness
   - Verify inlet/outlet regions are clearly defined
   - Adjust DBSCAN clustering parameters

2. **SDF generation errors**
   - Ensure mesh is watertight (closed surface)
   - Check domain bounds encompass entire geometry
   - Verify mesh file format compatibility

3. **JAX-Fluids configuration errors**
   - Validate all required sections are present
   - Check forcing parameters are physically reasonable
   - Ensure levelset path is correct

4. **Simulation convergence issues**
   - Reduce CFL number in numerical setup
   - Adjust forcing target values
   - Check initial conditions compatibility

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

processor = IntelligentBoundaryProcessor(mesh_file, logger=logging.getLogger())
```

## Contributing

### Code Organization
- **Core modules**: Self-contained processing components
- **Error handling**: Comprehensive exception management
- **Logging**: Detailed progress and debug information
- **Validation**: Built-in quality checks throughout pipeline

### Testing
- **Unit tests**: Individual component validation
- **Integration tests**: End-to-end pipeline verification  
- **Regression tests**: Consistency across updates
- **Performance benchmarks**: Scaling and optimization

## Performance Tips

1. **Use appropriate grid resolution** for your accuracy needs
2. **Leverage vectorized operations** in custom modifications
3. **Monitor memory usage** for large simulations
4. **Cache intermediate results** during development
5. **Parallelize multiple cases** for parameter studies

## License & Citation

Based on the successful JAX-Fluids internal flow implementation developed through the endpoint boundary conditions project. This production endpoint encapsulates the proven methodology for virtual boundary condition implementation.

---

**Generated by the Intelligent Boundary Conditions Team**  
**Version 1.0.0 - Production Ready** 