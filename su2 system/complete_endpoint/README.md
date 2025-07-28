# Complete Wind Tunnel Endpoint 🌪️

**A unified endpoint that combines wind tunnel mesh generation with CFD simulation configuration in a single JSON-driven workflow.**

## 🎯 Overview

This complete endpoint integrates two powerful systems:
1. **Packaged Wind Tunnel Endpoint** - Generates wind tunnel meshes from object files
2. **Production CFD Endpoint** - Creates SU2 simulation configurations

The result is a seamless workflow that takes an object mesh file and produces a complete, ready-to-run CFD simulation setup.

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Basic Usage
```bash
python complete_wind_tunnel_endpoint.py example_parameters.json
```

### 3. Python API Usage
```python
from complete_wind_tunnel_endpoint import CompleteWindTunnelEndpoint, CompleteWindTunnelRequest

# Create request
request = CompleteWindTunnelRequest(
    object_mesh_file="propeller.vtk",
    mach_number=0.3,
    angle_of_attack=5.0,
    output_directory="my_simulation"
)

# Process request
endpoint = CompleteWindTunnelEndpoint()
result = endpoint.process_complete_request(request)

if result.success:
    print(f"✅ Simulation ready in: {result.simulation_directory}")
else:
    print(f"❌ Failed: {result.message}")
```

## 📋 Complete Parameter Reference

### Required Parameters
- `object_mesh_file` - Path to input object mesh file (SU2, VTK, etc.)

### Wind Tunnel Generation Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tunnel_type` | string | "standard" | Tunnel configuration (compact, standard, research, automotive, aerospace) |
| `flow_direction` | string | "+X" | Flow direction (+X, -X, +Y, -Y, +Z, -Z) |
| `mesh_quality` | string | "medium" | Mesh density (coarse, medium, fine) |
| `domain_scale_factor` | float | 1.0 | Domain scaling factor |
| `generate_vtk` | boolean | true | Generate VTK visualization files |

### CFD Simulation Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `flow_type` | string | "EULER" | Flow solver (EULER, RANS, NAVIER_STOKES) |
| `mach_number` | float | 0.3 | Mach number |
| `reynolds_number` | float | 1000000.0 | Reynolds number |
| `angle_of_attack` | float | 0.0 | Angle of attack (degrees) |
| `sideslip_angle` | float | 0.0 | Sideslip angle (degrees) |
| `max_iterations` | integer | 100 | Maximum solver iterations |
| `turbulence_model` | string | "NONE" | Turbulence model (NONE, SA, SST, KE) |
| `turbulence_intensity` | float | 0.05 | Turbulence intensity |
| `viscosity_ratio` | float | 10.0 | Turbulent to laminar viscosity ratio |

### Advanced CFD Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `freestream_pressure` | float | 101325.0 | Freestream pressure (Pa) |
| `freestream_temperature` | float | 288.15 | Freestream temperature (K) |
| `cfl_number` | float | 1.0 | CFL number for time stepping |
| `convergence_residual` | float | 1e-8 | Convergence criteria |
| `convective_scheme` | string | "JST" | Convective scheme (JST, ROE, etc.) |

### Output Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_directory` | string | "complete_output" | Output directory path |
| `simulation_name` | string | auto-generated | Simulation subdirectory name |
| `prompt` | string | null | Natural language alternative to explicit parameters |

## 📁 Example JSON Files

### Explicit Parameters
```json
{
  "object_mesh_file": "propeller.vtk",
  "tunnel_type": "standard",
  "flow_direction": "+X",
  "mesh_quality": "medium",
  "flow_type": "EULER",
  "mach_number": 0.3,
  "angle_of_attack": 5.0,
  "max_iterations": 100,
  "output_directory": "propeller_analysis",
  "simulation_name": "euler_5deg"
}
```

### Natural Language Prompt
```json
{
  "object_mesh_file": "propeller.vtk",
  "tunnel_type": "standard",
  "mesh_quality": "medium",
  "prompt": "Create an Euler simulation at Mach 0.3 with 5 degrees angle of attack and 150 iterations",
  "output_directory": "propeller_prompt_analysis"
}
```

## 🔄 Processing Workflow

The complete endpoint follows this workflow:

### Step 1: Wind Tunnel Generation 🏗️
- Reads the input object mesh file
- Creates a wind tunnel domain around the object
- Generates tetrahedral mesh elements
- Sets up proper boundary conditions
- Outputs SU2 mesh file and optional VTK visualization

### Step 2: CFD Configuration ⚙️
- Processes CFD parameters (explicit or from natural language prompt)
- Creates SU2 configuration file with all solver settings
- Copies wind tunnel mesh to simulation directory
- Sets up proper boundary markers and reference values

### Step 3: Output Organization 📁
- Creates organized output directory structure
- Generates processing summary JSON file
- Lists all generated files
- Provides ready-to-run simulation setup

## 📊 Output Structure

```
output_directory/
├── processing_summary.json          # Complete processing summary
├── wind_tunnel.su2                  # Generated wind tunnel mesh
├── wind_tunnel.vtk                  # VTK visualization (if enabled)
└── simulation_name/                 # Simulation directory
    ├── config.cfg                   # SU2 configuration file
    └── wind_tunnel.su2              # Mesh file (copy)
```

## 🎯 Supported Input Formats

- **SU2 (.su2)** - Native SU2 mesh format
- **VTK (.vtk)** - VTK legacy format
- **STL (.stl)** - Stereolithography format (via conversion)
- **PLY (.ply)** - Polygon file format (via conversion)

## 🔧 Advanced Usage

### Custom Output Directory Structure
```python
request = CompleteWindTunnelRequest(
    object_mesh_file="wing.vtk",
    output_directory="/path/to/custom/output",
    simulation_name="wing_analysis_v2"
)
```

### High-Fidelity RANS Simulation
```python
request = CompleteWindTunnelRequest(
    object_mesh_file="aircraft.su2",
    flow_type="RANS",
    turbulence_model="SA",
    reynolds_number=5000000.0,
    max_iterations=500,
    mesh_quality="fine"
)
```

### Natural Language Processing
```python
request = CompleteWindTunnelRequest(
    object_mesh_file="propeller.vtk",
    prompt="High Reynolds number turbulent flow analysis at 10 degrees angle of attack with 300 iterations"
)
```

## 🚀 Performance Characteristics

### Wind Tunnel Generation
- **Time**: 10-30 seconds depending on mesh complexity
- **Output**: 5-50 MB SU2 files
- **Memory**: 1-4 GB during processing

### CFD Configuration
- **Time**: < 1 second
- **Output**: 1-5 KB configuration files
- **Memory**: < 100 MB

### Total Processing
- **Typical Time**: 15-45 seconds
- **Output Size**: 10-100 MB total
- **Ready for SU2**: Immediate

## 🛠️ System Requirements

- **Python 3.8+**
- **Memory**: 4GB+ recommended
- **Storage**: 1GB+ for outputs
- **SU2 CFD Suite** (for running simulations)

## 🔍 Troubleshooting

### Common Issues

1. **Input File Not Found**
   ```
   ❌ Input mesh file not found: propeller.vtk
   ```
   **Solution**: Check file path and ensure file exists

2. **Invalid Flow Type**
   ```
   ❌ Invalid flow_type: INVALID
   ```
   **Solution**: Use EULER, RANS, or NAVIER_STOKES

3. **Memory Issues**
   ```
   ❌ Memory error during mesh generation
   ```
   **Solution**: Use coarser mesh quality or smaller domain scale

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Detailed logging will be shown
endpoint = CompleteWindTunnelEndpoint()
```

## 📈 Validation Status

✅ **Tested Configurations:**
- Euler flows (Mach 0.1 - 0.8)
- RANS turbulent flows (Re 1e4 - 1e7)
- Various angles of attack (0° - 15°)
- Multiple mesh qualities
- Different tunnel types

✅ **Validated Outputs:**
- SU2 mesh connectivity
- Boundary condition setup
- Configuration file syntax
- Simulation convergence

## 🤝 Integration Examples

### Batch Processing
```python
import json
from pathlib import Path

# Process multiple configurations
configs = ["config1.json", "config2.json", "config3.json"]

for config_file in configs:
    result = process_json_request(config_file)
    if result.success:
        print(f"✅ {config_file}: {result.simulation_directory}")
    else:
        print(f"❌ {config_file}: {result.message}")
```

### Web API Integration
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
endpoint = CompleteWindTunnelEndpoint()

@app.route('/process_wind_tunnel', methods=['POST'])
def process_wind_tunnel():
    params = request.json
    req = CompleteWindTunnelRequest(**params)
    result = endpoint.process_complete_request(req)
    
    return jsonify({
        'success': result.success,
        'message': result.message,
        'simulation_directory': result.simulation_directory,
        'output_files': result.output_files
    })
```

## 📚 Related Documentation

- [Packaged Wind Tunnel Endpoint](../packaged_wind-tunnel_endpoint/README.md)
- [Production CFD Endpoint](../production_endpoint/README.md)
- [SU2 Documentation](https://su2code.github.io/)

## 🎉 Ready to Use!

The complete endpoint is production-ready and provides a seamless workflow from object mesh to ready-to-run CFD simulation. Perfect for:

- **Automated CFD workflows**
- **Batch processing multiple geometries**
- **Integration with larger simulation pipelines**
- **Educational and research applications**
- **Rapid prototyping and analysis** 