# Packaged Wind Tunnel Endpoint

This directory contains a complete, self-contained wind tunnel generation and visualization system for CFD analysis.

## 📁 Contents

### Core Files
- `wind_tunnel_api.py` - Main API endpoint for wind tunnel generation
- `generate_cfd_wind_tunnel.py` - CFD-ready wind tunnel mesh generator 
- `smart_cfd_visualizer.py` - Smart visualization with point clouds and boundary visualization

### Conversion & Utilities
- `su2_to_vtk_converter.py` - Professional SU2 to VTK converter
- `simple_su2_to_vtk.py` - Simple backup SU2 to VTK converter

### Output
- `output/` - Directory for generated mesh files

## 🚀 Quick Start

### 1. Generate CFD Wind Tunnel Mesh
```python
from generate_cfd_wind_tunnel import create_cfd_wind_tunnel

# Generate compact CFD-ready wind tunnel
output_file = create_cfd_wind_tunnel()
print(f"Generated: {output_file}")
```

### 2. Visualize the Mesh
```python
from smart_cfd_visualizer import smart_cfd_visualization

# Create smart ANSYS-style visualization
success = smart_cfd_visualization()
```

### 3. Convert SU2 to VTK (if needed)
```python
from su2_to_vtk_converter import convert_su2_to_vtk

# Convert for external visualization
convert_su2_to_vtk("input.su2", "output.vtk")
```

## 🏗️ Wind Tunnel Specifications

### Compact Design
- **Length**: 4 chord lengths total (1.5 upstream + 2.5 downstream)
- **Height**: 2 chord lengths  
- **Width**: 2 chord lengths
- **Mesh**: 50x30x25 = 37,500 domain nodes + object mesh

### Boundaries
- `inlet` - Inflow boundary (upstream face)
- `outlet` - Outflow boundary (downstream face)  
- `slip_wall` - Slip wall boundaries (top, bottom, sides)
- `object_wall` - Object surface (propeller/airfoil)

### CFD Ready Features
- Tetrahedral elements for better convergence
- Proper boundary labels for SU2_CFD
- Compact domain for faster computation
- Fine mesh near object surface

## 📊 Visualization Features

### Smart CFD Visualizer
- **Point Cloud**: Complete wind tunnel domain visualization
- **Detailed Object**: Full resolution propeller/airfoil geometry
- **Boundary Box**: Wind tunnel domain wireframe with labels
- **Color Coding**: Different colors for each boundary type
- **Performance**: Optimized for large meshes (70K+ nodes)

### Interactive Features
- 3D rotation and zoom
- Toggle mesh visibility
- Boundary highlighting
- Flow direction indicators

## 🔧 Dependencies

```python
# Required packages
numpy>=1.20.0
pyvista>=0.37.0  # For visualization
```

## 📝 Usage Examples

### Complete Workflow
```python
# 1. Generate wind tunnel
from generate_cfd_wind_tunnel import create_cfd_wind_tunnel
mesh_file = create_cfd_wind_tunnel()

# 2. Visualize result  
from smart_cfd_visualizer import smart_cfd_visualization
smart_cfd_visualization()

# 3. Export to VTK if needed
from su2_to_vtk_converter import convert_su2_to_vtk
convert_su2_to_vtk(mesh_file, "wind_tunnel.vtk")
```

### API Integration
```python
# Use as API endpoint
from wind_tunnel_api import generate_wind_tunnel_mesh

result = generate_wind_tunnel_mesh(
    airfoil_file="propeller.su2",
    flow_direction="+X",
    mach_number=0.1
)
```

## 🎯 Key Features

### ✅ CFD Optimized
- Compact domain (factor of 2 length ratio)
- Proper boundary conditions
- Tetrahedral mesh elements
- SU2_CFD compatible format

### ✅ Visualization Ready  
- Smart point cloud sampling
- Interactive 3D visualization
- Boundary color coding
- Performance optimized

### ✅ Production Ready
- Complete error handling
- Progress monitoring
- File size optimization
- Clean output management

## 🔍 File Details

| File | Size | Purpose |
|------|------|---------|
| `wind_tunnel_api.py` | 19KB | Main API with full pipeline |
| `generate_cfd_wind_tunnel.py` | 12KB | Core mesh generation |
| `smart_cfd_visualizer.py` | 10KB | Optimized visualization |
| `su2_to_vtk_converter.py` | ~5KB | Professional converter |

## 📈 Performance

### Mesh Generation
- **Time**: ~12 seconds for complete mesh
- **Output**: ~13MB SU2 file
- **Elements**: ~350K tetrahedral elements
- **Nodes**: ~70K total nodes

### Visualization
- **Loading**: ~2-3 seconds for large meshes
- **Rendering**: Real-time interaction
- **Memory**: Optimized point cloud sampling
- **Export**: VTK files ~15-20MB

## 🎨 Visualization Output

The smart visualizer creates:
- **Gold Propeller**: Detailed object geometry with edges
- **Blue Point Cloud**: Complete wind tunnel domain  
- **Green Wireframe**: Boundary box with flow direction
- **Color Boundaries**: Inlet (red), outlet (blue), walls (green)

Perfect for CFD analysis, academic research, and engineering visualization! 