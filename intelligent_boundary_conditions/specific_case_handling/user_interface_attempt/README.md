# Interactive User Interface for Boundary Condition Tagging

This subdirectory contains user interface tools that allow interactive selection and tagging of mesh faces for boundary condition assignment. Instead of relying on automatic algorithms, these tools let users directly click on faces to tag them as inlet, outlet, or wall.

## 🎯 Purpose

The goal is to provide an intuitive way for users to manually specify boundary conditions by:
- **Visualizing** the 3D mesh interactively
- **Clicking** on faces to tag them as inlet, outlet, or wall
- **Seeing** immediate visual feedback with color coding
- **Saving** the selections for boundary condition generation

## 🛠️ Available Tools

### 1. Interactive Face Picker (PyVista) - `interactive_face_picker.py`

**Best option for 3D interaction** - Uses PyVista for professional 3D mesh visualization and interaction.

**Features:**
- **Full 3D interaction:** Rotate, zoom, pan the mesh
- **Direct face clicking:** Click any face to tag it
- **Real-time coloring:** See selections immediately
- **Keyboard shortcuts:** Switch modes quickly
- **Undo/Redo:** Manage selections easily
- **Professional rendering:** High-quality 3D visualization

**Requirements:**
```bash
pip install pyvista
pip install meshio  # optional, for better mesh loading
```

**Usage:**
```bash
cd user_interface_attempt
python interactive_face_picker.py
```

**Controls:**
- **Mouse:** Click on faces to tag them
- **1:** Switch to Inlet mode (red)
- **2:** Switch to Outlet mode (green)  
- **3:** Switch to Wall mode (gray)
- **u:** Undo last selection
- **c:** Clear all selections
- **s:** Save selections to JSON
- **h:** Toggle help text
- **q:** Quit

### 2. Simple Mesh Viewer (Tkinter/Matplotlib) - `simple_mesh_viewer.py`

**Fallback option** - Uses standard Python libraries when PyVista is not available.

**Features:**
- **GUI interface:** User-friendly tkinter window
- **3D visualization:** Matplotlib 3D plotting
- **Face selection:** Click-to-tag functionality
- **Statistics panel:** Real-time selection counts
- **File browser:** Load different mesh files
- **No external dependencies:** Uses standard Python libraries

**Requirements:**
```bash
# Standard libraries only - should work out of the box
pip install matplotlib numpy
```

**Usage:**
```bash
cd user_interface_attempt
python simple_mesh_viewer.py
```

**Features:**
- Load mesh files via file browser
- Select tagging mode (Inlet/Outlet/Wall)
- Click on faces in 3D plot to tag them
- View real-time statistics
- Save selections to JSON

## 📊 Output Format

Both tools save selections in JSON format:

```json
{
  "mesh_file": "path/to/mesh.msh",
  "total_faces": 415164,
  "selections": {
    "inlet": [
      {"face_id": 1234, "position": [0, 0, 0], "timestamp": 1234567890}
    ],
    "outlet": [
      {"face_id": 5678, "position": [100, 0, 0], "timestamp": 1234567891}
    ],
    "wall": [
      {"face_id": 9999, "position": [50, 10, 5], "timestamp": 1234567892}
    ]
  }
}
```

## 🎮 Usage Workflow

### For Rocket Nozzle Boundary Conditions:

1. **Load the mesh:**
   - PyVista: Automatically loads the rocket engine mesh
   - Simple viewer: Use "Load Default" button or browse for file

2. **Identify inlet/outlet regions:**
   - Look for the hollow openings at the ends of the nozzle
   - Inlet: Smaller opening (combustion chamber side)
   - Outlet: Larger opening (nozzle exit side)

3. **Tag faces:**
   - Switch to "Inlet" mode and click on faces around the smaller opening
   - Switch to "Outlet" mode and click on faces around the larger opening
   - Optionally tag specific wall faces if needed

4. **Save selections:**
   - Press 's' (PyVista) or click "Save Selections" (Simple viewer)
   - JSON file is created with your selections

5. **Integration:**
   - The saved selections can be loaded by the boundary condition generator
   - Face IDs are used to apply proper boundary conditions in JAX-Fluids

## 🔧 Integration with Main System

The interactive selections can be integrated with the main intelligent boundary conditions system:

```python
from user_interface_attempt.interactive_face_picker import InteractiveFacePicker

# Create picker
picker = InteractiveFacePicker("mesh_file.msh")
picker.load_mesh()
picker.setup_interactive_plotter()

# Run interactive session
selections = picker.run_interactive_session()

# Use selections in boundary condition generation
from boundary_condition_generator import BoundaryConditionGenerator
bc_gen = BoundaryConditionGenerator(picker.geometry_parser)
config = bc_gen.generate_jaxfluids_config(
    inlet_faces=[s['face_id'] for s in selections if s['tag'] == 'inlet'],
    outlet_faces=[s['face_id'] for s in selections if s['tag'] == 'outlet'],
    wall_faces=[s['face_id'] for s in selections if s['tag'] == 'wall']
)
```

## 🚀 Benefits of Interactive Approach

### Advantages over Automatic Algorithms:

1. **Accuracy:** User knows exactly which faces should be inlet/outlet
2. **Flexibility:** Can handle complex geometries that confuse algorithms  
3. **Visual feedback:** See exactly what's being selected
4. **Context awareness:** User understands the physics and geometry
5. **Reliability:** No guessing or heuristics needed

### Use Cases:

- **Complex nozzle shapes:** Bell nozzles, aerospikes, multi-stage nozzles
- **Irregular meshes:** Meshes with artifacts or unusual topology
- **Specific requirements:** When precise boundary placement is critical
- **Validation:** Double-check automatic algorithm results
- **Education:** Understanding mesh topology and boundary conditions

## 🎯 Example: Rocket Nozzle Tagging

For a typical rocket nozzle:

1. **Identify the flow direction** (usually X-axis in our case)
2. **Find the smaller opening** at X≈0 (combustion chamber) → Tag as **INLET**
3. **Find the larger opening** at X≈max (nozzle exit) → Tag as **OUTLET** 
4. **All other faces** are walls (can be left untagged or explicitly tagged)

The visual feedback helps ensure you're selecting the right faces, especially for hollow openings that might not have explicit mesh faces.

## 📝 Notes

- **Performance:** PyVista handles large meshes better than matplotlib
- **Dependencies:** Simple viewer has fewer dependencies
- **3D interaction:** PyVista provides better 3D navigation
- **File formats:** Both support .msh, .stl, .obj, .ply files
- **Face sampling:** Large meshes may be sampled for display performance

## 🔮 Future Enhancements

Potential improvements:
- **Region selection:** Click and drag to select multiple faces
- **Smart selection:** Click one face, automatically select connected faces
- **Import/Export:** Load selections from previous sessions
- **Preview mode:** See boundary conditions applied before saving
- **Multi-view:** Side-by-side views for complex geometries 