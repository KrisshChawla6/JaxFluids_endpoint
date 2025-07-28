# Specific Case Handling

This directory contains specialized scripts for handling specific geometry cases and custom boundary condition scenarios that go beyond the general-purpose intelligent boundary conditions API.

## Scripts

### `process_rocket_engine.py`

**Purpose**: Processes rocket engine mesh files (specifically `Rocket Engine.msh`) and automatically tags surfaces based on area analysis.

**Features**:
- **Area-based tagging**: Larger surface area → Outlet, Smaller surface area → Inlet
- **3D Visualization**: Interactive PyVista or Matplotlib visualization
- **JAX-Fluids integration**: Generates complete boundary condition configurations
- **Boundary masks**: Creates 3D masks for direct solver usage

**Usage**:
```bash
# From the intelligent_boundary_conditions directory
cd specific_case_handling
python process_rocket_engine.py
```

**What it does**:
1. 📐 Parses the `Rocket Engine.msh` file
2. 🏷️ Analyzes face areas and tags surfaces intelligently
3. 👁️ Shows 3D visualization with color-coded faces:
   - 🔴 **Red** = Inlet (smaller area)
   - 🟢 **Green** = Outlet (larger area) 
   - ⚫ **Gray** = Wall
4. ⚙️ Generates JAX-Fluids configuration and boundary masks
5. 💾 Saves outputs to `./rocket_engine_bc_output/`

**Outputs**:
- `rocket_engine_config.json`: Complete JAX-Fluids case configuration
- `boundary_masks/`: Directory with inlet/outlet/wall masks (.npy files)
- Text summary of tagging results

## Adding New Case-Specific Scripts

When adding new scripts for specific geometries or cases:

1. **Create the script** in this directory
2. **Use proper imports**: Import from the parent directory modules
3. **Follow naming convention**: `process_[geometry_type].py`
4. **Update `__init__.py`**: Add exports for the new script
5. **Document in this README**: Add description and usage instructions

## Example Template

```python
#!/usr/bin/env python3
"""
[Geometry Type] Processor
========================

Description of what this script does for the specific geometry.
"""

import os
import sys
from pathlib import Path

# Add parent directory for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

from geometry_parser import GeometryParser
from face_tagger import FaceTagger
from boundary_condition_generator import BoundaryConditionGenerator

def main():
    """Main processing function"""
    # Your specific processing logic here
    pass

if __name__ == "__main__":
    main()
```

## Requirements

These scripts use the same dependencies as the main intelligent boundary conditions system. Make sure you have installed:

```bash
pip install -r ../requirements.txt
```

For visualization features, also install:
```bash
pip install pyvista matplotlib
``` 