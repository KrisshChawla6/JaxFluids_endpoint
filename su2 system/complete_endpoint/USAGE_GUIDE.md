# Complete Wind Tunnel Endpoint - Usage Guide

This guide provides step-by-step instructions for using the Complete Wind Tunnel Endpoint.

## File Overview

| File | Purpose |
|------|---------|
| `complete_wind_tunnel_endpoint.py` | Main endpoint that combines both systems |
| `example_parameters.json` | Example with explicit CFD parameters |
| `example_with_prompt.json` | Example using natural language prompt |
| `test_complete_endpoint.py` | Test suite to validate functionality |
| `batch_processor.py` | Batch processing for multiple configurations |
| `requirements.txt` | Python dependencies |
| `README.md` | Comprehensive documentation |

## Quick Start

### Step 1: Install Dependencies
```bash
cd complete_endpoint
pip install -r requirements.txt
```

### Step 2: Test the System
```bash
python test_complete_endpoint.py
```

### Step 3: Run Example
```bash
python complete_wind_tunnel_endpoint.py example_parameters.json
```

## Usage Patterns

### Pattern 1: Single Configuration File
```bash
python complete_wind_tunnel_endpoint.py my_config.json
```

### Pattern 2: Batch Processing
```bash
# Create example configurations
python batch_processor.py create-examples

# Process all configurations in parallel
python batch_processor.py example_configs/ --parallel --workers 3
```

### Pattern 3: Python API
```python
from complete_wind_tunnel_endpoint import CompleteWindTunnelEndpoint, CompleteWindTunnelRequest

request = CompleteWindTunnelRequest(
    object_mesh_file="my_object.vtk",
    mach_number=0.3,
    angle_of_attack=5.0,
    output_directory="my_output"
)

endpoint = CompleteWindTunnelEndpoint()
result = endpoint.process_complete_request(request)

if result.success:
    print(f"Ready to run: {result.simulation_directory}")
```

## Common Use Cases

### Propeller Analysis
```json
{
  "object_mesh_file": "propeller.vtk",
  "tunnel_type": "automotive",
  "flow_type": "EULER",
  "mach_number": 0.2,
  "angle_of_attack": 8.0,
  "max_iterations": 150,
  "output_directory": "propeller_study",
  "simulation_name": "prop_8deg_analysis"
}
```

### Wing Section Study
```json
{
  "object_mesh_file": "wing_section.su2",
  "tunnel_type": "aerospace",
  "flow_type": "RANS",
  "turbulence_model": "SA",
  "mach_number": 0.7,
  "reynolds_number": 5000000,
  "angle_of_attack": 2.5,
  "max_iterations": 300,
  "mesh_quality": "fine",
  "output_directory": "wing_analysis"
}
```

### Natural Language Processing
```json
{
  "object_mesh_file": "aircraft_model.vtk",
  "prompt": "High-fidelity RANS simulation at Mach 0.8 with 4 degrees angle of attack, Reynolds number 10 million, and 500 iterations for transonic analysis",
  "output_directory": "transonic_study"
}
```

## Configuration Tips

### Mesh Quality Selection
- **coarse**: Fast processing, lower accuracy
- **medium**: Balanced performance (recommended)
- **fine**: High accuracy, longer processing time

### Flow Type Selection
- **EULER**: Inviscid flow, fastest convergence
- **NAVIER_STOKES**: Viscous flow without turbulence
- **RANS**: Turbulent flow, most realistic

### Tunnel Type Selection
- **compact**: Small domain, fastest processing
- **standard**: Balanced domain size (recommended)
- **research**: Large domain, highest accuracy
- **automotive**: Optimized for ground vehicles
- **aerospace**: Optimized for aircraft

## Output Structure

After processing, you'll get this structure:
```
output_directory/
├── processing_summary.json          # Complete processing metadata
├── wind_tunnel.su2                  # Generated wind tunnel mesh
├── wind_tunnel.vtk                  # VTK visualization file
└── simulation_name/                 # Ready-to-run simulation
    ├── config.cfg                   # SU2 configuration file
    └── wind_tunnel.su2              # Mesh file (copy)
```

### Running the Simulation
```bash
cd output_directory/simulation_name/
SU2_CFD config.cfg
```

## Performance Optimization

### For Speed
```json
{
  "tunnel_type": "compact",
  "mesh_quality": "coarse",
  "flow_type": "EULER",
  "max_iterations": 50
}
```

### For Accuracy
```json
{
  "tunnel_type": "research",
  "mesh_quality": "fine",
  "flow_type": "RANS",
  "turbulence_model": "SA",
  "max_iterations": 500,
  "convergence_residual": 1e-10
}
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: "Input mesh file not found"
Check file path and use absolute path if needed:
```json
{
  "object_mesh_file": "/full/path/to/mesh.vtk"
}
```

#### Issue: "Invalid flow type"
Use correct flow types:
```json
{
  "flow_type": "EULER"        // Correct
}
```

#### Issue: Memory errors during processing
Reduce mesh density:
```json
{
  "mesh_quality": "coarse",
  "domain_scale_factor": 0.8,
  "tunnel_type": "compact"
}
```

#### Issue: Slow convergence
Adjust solver parameters:
```json
{
  "cfl_number": 0.5,
  "max_iterations": 500,
  "convergence_residual": 1e-6
}
```

## Ready to Use!

The Complete Wind Tunnel Endpoint provides a unified workflow from object mesh to ready-to-run CFD simulation with JSON-driven configuration, natural language processing, and batch processing capabilities. 