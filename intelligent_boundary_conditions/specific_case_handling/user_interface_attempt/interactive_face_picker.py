#!/usr/bin/env python3
"""
Interactive Face Picker for Boundary Condition Tagging
Uses PyVista's interactive capabilities to let users click on faces to tag them
"""

import os
import sys
import numpy as np
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Optional, Callable

# Add the parent directories to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False

from geometry_parser import GeometryParser

@dataclass
class FaceSelection:
    """Data class to track face selections"""
    face_id: int
    tag: str  # 'inlet', 'outlet', 'wall'
    timestamp: float
    position: List[float]  # centroid of the face

class InteractiveFacePicker:
    """
    Interactive face picker using PyVista for boundary condition tagging
    """
    
    def __init__(self, mesh_file: str):
        """
        Initialize the interactive face picker
        
        Parameters:
        -----------
        mesh_file : str
            Path to the mesh file
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required for interactive face picking. Install with: pip install pyvista")
        
        self.mesh_file = mesh_file
        self.geometry_parser = None
        self.pv_mesh = None
        self.plotter = None
        
        # Face selection tracking
        self.selected_faces: Dict[int, FaceSelection] = {}
        self.current_tag = 'inlet'  # Default tag mode
        
        # Visual styling
        self.colors = {
            'inlet': 'red',
            'outlet': 'green', 
            'wall': 'gray',
            'unselected': 'lightblue'
        }
        
        # UI state
        self.show_face_ids = False
        self.show_help = True
        
    def load_mesh(self):
        """Load and prepare the mesh for interactive picking"""
        
        print("🚀 INTERACTIVE FACE PICKER FOR BOUNDARY TAGGING")
        print("=" * 60)
        print(f"📂 Loading mesh: {self.mesh_file}")
        
        # Parse with our geometry parser first to get faces
        self.geometry_parser = GeometryParser(self.mesh_file)
        geometry_data = self.geometry_parser.parse_geometry()
        
        # Convert to PyVista mesh
        self.pv_mesh = self._convert_to_pyvista()
        
        print(f"✅ Loaded mesh with {self.pv_mesh.n_faces} faces and {self.pv_mesh.n_points} points")
        
    def _convert_to_pyvista(self):
        """Convert our parsed geometry to PyVista mesh"""
        
        if not MESHIO_AVAILABLE:
            print("⚠️  meshio not available. Trying direct MSH to PLY conversion...")
            return self._load_via_ply_conversion()
        
        # Load with meshio and convert to PyVista
        print("🔄 Converting mesh to PyVista format...")
        
        try:
            mesh = meshio.read(self.mesh_file)
            
            # Extract surface faces from volume elements if needed
            vertices = mesh.points
            surface_faces = []
            
            for cell_block in mesh.cells:
                if cell_block.type == 'tetra':
                    # Extract boundary faces from tetrahedra
                    tetra_faces = [
                        [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]
                    ]
                    
                    face_count = {}
                    for tetra in cell_block.data:
                        for face_indices in tetra_faces:
                            face = tuple(sorted([tetra[face_indices[0]], tetra[face_indices[1]], tetra[face_indices[2]]]))
                            face_count[face] = face_count.get(face, 0) + 1
                    
                    # Boundary faces appear only once
                    boundary_faces = [list(face) for face, count in face_count.items() if count == 1]
                    surface_faces.extend(boundary_faces)
                    break
                elif cell_block.type == 'triangle':
                    # Direct surface mesh
                    surface_faces = cell_block.data.tolist()
                    break
            
            if not surface_faces:
                raise ValueError("No surface faces found in mesh")
            
            # Create PyVista mesh
            faces_array = []
            for face in surface_faces:
                faces_array.extend([3] + face)  # PyVista format: [n_points, p1, p2, p3]
            
            pv_mesh = pv.PolyData(vertices, faces_array)
            
            print(f"✅ Converted to PyVista: {pv_mesh.n_faces} faces")
            return pv_mesh
            
        except Exception as e:
            print(f"❌ Meshio conversion failed: {e}")
            return self._load_via_ply_conversion()
    
    def _load_via_ply_conversion(self):
        """Fallback: convert to PLY and load with PyVista"""
        
        print("🔄 Fallback: Using PLY conversion...")
        
        # Use our geometry parser data
        vertices = []
        faces = []
        
        for face in self.geometry_parser.faces:
            for i, vertex in enumerate(face.vertices):
                if i == 0:
                    vertices.extend(vertex)
                elif i < 3:  # Only use first 3 vertices for triangular faces
                    vertices.extend(vertex)
        
        # Create vertex array
        vertices = np.array(vertices).reshape(-1, 3)
        
        # Create face indices (every 3 vertices form a triangle)
        n_faces = len(vertices) // 3
        faces_array = []
        for i in range(n_faces):
            faces_array.extend([3, i*3, i*3+1, i*3+2])
        
        pv_mesh = pv.PolyData(vertices, faces_array)
        print(f"✅ Created PyVista mesh: {pv_mesh.n_faces} faces")
        
        return pv_mesh
    
    def setup_interactive_plotter(self):
        """Setup the interactive PyVista plotter with face picking"""
        
        print("🎮 Setting up interactive plotter...")
        
        # Create plotter
        self.plotter = pv.Plotter()
        
        # Add mesh with initial coloring
        self.plotter.add_mesh(
            self.pv_mesh,
            color=self.colors['unselected'],
            show_edges=True,
            edge_color='white',
            line_width=0.5,
            name='main_mesh',
            pickable=True
        )
        
        # Enable cell picking (faces)
        self.plotter.enable_cell_picking(
            callback=self._face_pick_callback,
            show_message=False,
            style='wireframe',
            line_width=3,
            color='yellow',
            show=True
        )
        
        # Add instructions text
        self._add_instructions()
        
        # Setup key bindings
        self._setup_key_bindings()
        
        print("✅ Interactive plotter ready!")
        
    def _face_pick_callback(self, mesh, cell_id):
        """Callback function when a face is picked"""
        
        if cell_id is None:
            return
        
        print(f"👆 Picked face {cell_id} - tagging as '{self.current_tag}'")
        
        # Get face centroid for position
        face = mesh.extract_cells(cell_id)
        centroid = face.center
        
        # Create selection record
        import time
        selection = FaceSelection(
            face_id=cell_id,
            tag=self.current_tag,
            timestamp=time.time(),
            position=centroid.tolist()
        )
        
        # Store selection
        self.selected_faces[cell_id] = selection
        
        # Update visualization
        self._update_face_colors()
        
        # Print current counts
        self._print_selection_summary()
    
    def _update_face_colors(self):
        """Update the mesh colors based on face selections"""
        
        # Create color array for all faces
        n_faces = self.pv_mesh.n_faces
        colors = np.full(n_faces, 0.8)  # Default gray
        
        # Color selected faces
        for face_id, selection in self.selected_faces.items():
            if face_id < n_faces:  # Safety check
                if selection.tag == 'inlet':
                    colors[face_id] = 0.0  # Red
                elif selection.tag == 'outlet':
                    colors[face_id] = 0.4  # Green
                elif selection.tag == 'wall':
                    colors[face_id] = 0.7  # Dark gray
        
        # Update mesh colors
        self.pv_mesh['face_tags'] = colors
        self.plotter.update_scalars(colors, mesh=self.pv_mesh, render=True)
    
    def _add_instructions(self):
        """Add instruction text to the plotter"""
        
        instructions = [
            "🎯 INTERACTIVE FACE TAGGING",
            "",
            "MOUSE:",
            "  • Click on faces to tag them",
            "",
            "KEYBOARD:",
            "  • '1' - Tag as INLET (red)",
            "  • '2' - Tag as OUTLET (green)", 
            "  • '3' - Tag as WALL (gray)",
            "  • 'u' - Undo last selection",
            "  • 'c' - Clear all selections",
            "  • 's' - Save selections",
            "  • 'h' - Toggle help",
            "  • 'q' - Quit",
            "",
            f"Current mode: {self.current_tag.upper()}"
        ]
        
        self.help_text = self.plotter.add_text(
            "\n".join(instructions),
            position='upper_left',
            font_size=10,
            color='white',
            name='instructions'
        )
    
    def _setup_key_bindings(self):
        """Setup keyboard shortcuts"""
        
        # Tag mode selection
        self.plotter.add_key_event('1', self._set_inlet_mode)
        self.plotter.add_key_event('2', self._set_outlet_mode)
        self.plotter.add_key_event('3', self._set_wall_mode)
        
        # Actions
        self.plotter.add_key_event('u', self._undo_selection)
        self.plotter.add_key_event('c', self._clear_selections)
        self.plotter.add_key_event('s', self._save_selections)
        self.plotter.add_key_event('h', self._toggle_help)
        
    def _set_inlet_mode(self):
        """Set tagging mode to inlet"""
        self.current_tag = 'inlet'
        print("🔴 Mode: INLET (red)")
        self._update_instructions()
    
    def _set_outlet_mode(self):
        """Set tagging mode to outlet"""
        self.current_tag = 'outlet'
        print("🟢 Mode: OUTLET (green)")
        self._update_instructions()
    
    def _set_wall_mode(self):
        """Set tagging mode to wall"""
        self.current_tag = 'wall'
        print("⚫ Mode: WALL (gray)")
        self._update_instructions()
    
    def _undo_selection(self):
        """Undo the last face selection"""
        if not self.selected_faces:
            print("❌ No selections to undo")
            return
        
        # Find the most recent selection
        last_selection = max(self.selected_faces.items(), key=lambda x: x[1].timestamp)
        face_id, selection = last_selection
        
        # Remove it
        del self.selected_faces[face_id]
        print(f"↩️  Undid selection of face {face_id} ({selection.tag})")
        
        # Update visualization
        self._update_face_colors()
        self._print_selection_summary()
    
    def _clear_selections(self):
        """Clear all face selections"""
        count = len(self.selected_faces)
        self.selected_faces.clear()
        print(f"🧹 Cleared {count} selections")
        
        # Update visualization
        self._update_face_colors()
        self._print_selection_summary()
    
    def _save_selections(self):
        """Save face selections to file"""
        
        if not self.selected_faces:
            print("❌ No selections to save")
            return
        
        # Create output data
        output_data = {
            'mesh_file': self.mesh_file,
            'total_faces': self.pv_mesh.n_faces,
            'selections': {}
        }
        
        # Group by tag
        for tag in ['inlet', 'outlet', 'wall']:
            output_data['selections'][tag] = []
        
        for face_id, selection in self.selected_faces.items():
            output_data['selections'][selection.tag].append({
                'face_id': face_id,
                'position': selection.position,
                'timestamp': selection.timestamp
            })
        
        # Save to file
        output_file = 'interactive_face_selections.json'
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Saved {len(self.selected_faces)} selections to: {output_file}")
        self._print_selection_summary()
    
    def _toggle_help(self):
        """Toggle help text visibility"""
        self.show_help = not self.show_help
        
        if self.show_help:
            self.help_text.SetVisibility(True)
        else:
            self.help_text.SetVisibility(False)
        
        self.plotter.render()
    
    def _update_instructions(self):
        """Update the instruction text with current mode"""
        if hasattr(self, 'help_text'):
            instructions = [
                "🎯 INTERACTIVE FACE TAGGING",
                "",
                "MOUSE:",
                "  • Click on faces to tag them",
                "",
                "KEYBOARD:",
                "  • '1' - Tag as INLET (red)",
                "  • '2' - Tag as OUTLET (green)", 
                "  • '3' - Tag as WALL (gray)",
                "  • 'u' - Undo last selection",
                "  • 'c' - Clear all selections",
                "  • 's' - Save selections",
                "  • 'h' - Toggle help",
                "  • 'q' - Quit",
                "",
                f"Current mode: {self.current_tag.upper()}"
            ]
            
            self.help_text.SetText("\n".join(instructions))
            self.plotter.render()
    
    def _print_selection_summary(self):
        """Print current selection summary"""
        
        # Count by tag
        counts = {'inlet': 0, 'outlet': 0, 'wall': 0}
        for selection in self.selected_faces.values():
            counts[selection.tag] += 1
        
        total = sum(counts.values())
        print(f"📊 Selections: Inlet={counts['inlet']}, Outlet={counts['outlet']}, Wall={counts['wall']}, Total={total}")
    
    def run_interactive_session(self):
        """Run the interactive face picking session"""
        
        if not self.pv_mesh:
            raise ValueError("Mesh not loaded. Call load_mesh() first.")
        
        print("\n🎮 Starting interactive session...")
        print("Click on faces to tag them as inlet, outlet, or wall")
        print("Use keyboard shortcuts for different modes")
        print("Press 'q' to quit when done")
        
        # Show the plotter
        self.plotter.show()
        
        print("✅ Interactive session completed!")
        
        # Return final selections
        return self.selected_faces

def main():
    """Main function to run the interactive face picker"""
    
    print("🚀 INTERACTIVE BOUNDARY CONDITION FACE PICKER")
    print("=" * 60)
    
    # Check dependencies
    if not PYVISTA_AVAILABLE:
        print("❌ PyVista is required for this tool.")
        print("Install with: pip install pyvista")
        return
    
    # Initialize picker
    mesh_file = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\mesh\Rocket Engine.msh"
    picker = InteractiveFacePicker(mesh_file)
    
    try:
        # Load mesh
        picker.load_mesh()
        
        # Setup interactive plotter
        picker.setup_interactive_plotter()
        
        # Run interactive session
        selections = picker.run_interactive_session()
        
        # Print final summary
        if selections:
            print("\n🎉 Final Results:")
            counts = {'inlet': 0, 'outlet': 0, 'wall': 0}
            for selection in selections.values():
                counts[selection.tag] += 1
            
            print(f"   Inlet faces: {counts['inlet']}")
            print(f"   Outlet faces: {counts['outlet']}")
            print(f"   Wall faces: {counts['wall']}")
            print(f"   Total tagged: {sum(counts.values())}")
            
            print("\n💾 Selections saved to: interactive_face_selections.json")
        else:
            print("\n❌ No faces were selected")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 