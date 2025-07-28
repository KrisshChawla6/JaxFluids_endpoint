#!/usr/bin/env python3
"""
Interactive Mesh Face Highlighter
Properly loads mesh and allows clicking on faces to highlight them
"""

import os
import sys
import numpy as np

try:
    import pyvista as pv
    print(f"✅ PyVista {pv.__version__} available")
except ImportError:
    print("❌ PyVista not available")
    exit()

try:
    import meshio
    print(f"✅ Meshio available")
except ImportError:
    print("❌ Meshio not available")
    exit()

class InteractiveMeshHighlighter:
    """Interactive mesh viewer with face highlighting"""
    
    def __init__(self, mesh_file):
        self.mesh_file = mesh_file
        self.mesh = None
        self.plotter = None
        self.highlighted_faces = {}
        self.current_color = 'red'
        
        # Color modes
        self.colors = {
            'inlet': 'red',
            'outlet': 'green', 
            'wall': 'gray',
            'highlight': 'yellow'
        }
        
    def load_mesh(self):
        """Load and prepare mesh for visualization"""
        
        print(f"🔄 Loading mesh: {self.mesh_file}")
        
        try:
            # Load with meshio
            meshio_mesh = meshio.read(self.mesh_file)
            print(f"📊 Meshio loaded:")
            print(f"   Points: {len(meshio_mesh.points)}")
            print(f"   Cell blocks: {len(meshio_mesh.cells)}")
            
            for i, block in enumerate(meshio_mesh.cells):
                print(f"   Block {i}: {block.type} ({len(block.data)} elements)")
            
            # Convert to PyVista
            self.mesh = pv.from_meshio(meshio_mesh)
            print(f"✅ PyVista mesh: {self.mesh.n_cells} cells, {self.mesh.n_points} points")
            
            # Extract surface if it's a volume mesh
            if hasattr(self.mesh, 'extract_surface'):
                print("🔄 Extracting surface...")
                self.mesh = self.mesh.extract_surface()
                print(f"📊 Surface extracted: {self.mesh.n_faces} faces")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load mesh: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def setup_plotter(self):
        """Setup the PyVista plotter with interaction"""
        
        print("🎮 Setting up interactive plotter...")
        
        self.plotter = pv.Plotter()
        
        # Add the main mesh
        self.plotter.add_mesh(
            self.mesh,
            color='lightblue',
            show_edges=True,
            edge_color='white',
            line_width=0.3,
            opacity=0.8,
            pickable=True,
            name='main_mesh'
        )
        
        # Setup face picking
        self.plotter.enable_cell_picking(
            callback=self.face_pick_callback,
            show_message=True,
            style='wireframe',
            line_width=2,
            color='yellow'
        )
        
        # Add instructions
        instructions = [
            "🎯 INTERACTIVE FACE HIGHLIGHTER",
            "",
            "CONTROLS:",
            "• Click faces to highlight them",
            "• 1 = Red (Inlet mode)",
            "• 2 = Green (Outlet mode)", 
            "• 3 = Gray (Wall mode)",
            "• c = Clear all highlights",
            "• s = Save selections",
            "• r = Reset view",
            "• q = Quit",
            "",
            f"Current color: {self.current_color}"
        ]
        
        self.text_actor = self.plotter.add_text(
            "\n".join(instructions),
            position='upper_left',
            font_size=10,
            color='white'
        )
        
        # Key bindings
        self.plotter.add_key_event('1', self.set_inlet_mode)
        self.plotter.add_key_event('2', self.set_outlet_mode)
        self.plotter.add_key_event('3', self.set_wall_mode)
        self.plotter.add_key_event('c', self.clear_highlights)
        self.plotter.add_key_event('s', self.save_selections)
        
        # Camera setup
        self.plotter.camera_position = 'iso'
        self.plotter.add_axes()
        
        print("✅ Plotter ready!")
    
    def face_pick_callback(self, mesh, face_id):
        """Callback when a face is picked"""
        
        if face_id is None:
            return
        
        print(f"👆 Picked face {face_id} - highlighting as '{self.current_color}'")
        
        try:
            # Get the picked face
            face = mesh.extract_cells([face_id])
            
            # Store the selection
            self.highlighted_faces[face_id] = {
                'color': self.current_color,
                'mesh': face
            }
            
            # Add highlighted face as separate mesh
            self.plotter.add_mesh(
                face,
                color=self.colors[self.current_color],
                opacity=1.0,
                name=f'highlight_{face_id}',
                reset_camera=False
            )
            
            print(f"✅ Highlighted face {face_id} as {self.current_color}")
            self.update_statistics()
            
        except Exception as e:
            print(f"❌ Failed to highlight face: {e}")
    
    def set_inlet_mode(self):
        """Set color mode to inlet (red)"""
        self.current_color = 'inlet'
        print("🔴 Mode: INLET (red)")
        self.update_instructions()
    
    def set_outlet_mode(self):
        """Set color mode to outlet (green)"""
        self.current_color = 'outlet'
        print("🟢 Mode: OUTLET (green)")
        self.update_instructions()
    
    def set_wall_mode(self):
        """Set color mode to wall (gray)"""
        self.current_color = 'wall'
        print("⚫ Mode: WALL (gray)")
        self.update_instructions()
    
    def clear_highlights(self):
        """Clear all highlighted faces"""
        count = len(self.highlighted_faces)
        
        # Remove highlight meshes from plotter
        for face_id in list(self.highlighted_faces.keys()):
            try:
                self.plotter.remove_actor(f'highlight_{face_id}')
            except:
                pass
        
        self.highlighted_faces.clear()
        print(f"🧹 Cleared {count} highlighted faces")
        self.update_statistics()
    
    def save_selections(self):
        """Save highlighted faces to file"""
        
        if not self.highlighted_faces:
            print("❌ No faces to save")
            return
        
        # Group by color
        selections = {
            'inlet': [],
            'outlet': [],
            'wall': []
        }
        
        for face_id, data in self.highlighted_faces.items():
            color = data['color']
            if color in selections:
                selections[color].append(face_id)
        
        # Create output
        output = {
            'mesh_file': self.mesh_file,
            'total_faces': self.mesh.n_faces,
            'total_highlighted': len(self.highlighted_faces),
            'selections': selections
        }
        
        # Save to file
        import json
        output_file = 'interactive_face_selections.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"💾 Saved {len(self.highlighted_faces)} selections to: {output_file}")
        self.update_statistics()
    
    def update_instructions(self):
        """Update instruction text"""
        instructions = [
            "🎯 INTERACTIVE FACE HIGHLIGHTER",
            "",
            "CONTROLS:",
            "• Click faces to highlight them",
            "• 1 = Red (Inlet mode)",
            "• 2 = Green (Outlet mode)", 
            "• 3 = Gray (Wall mode)",
            "• c = Clear all highlights",
            "• s = Save selections",
            "• r = Reset view",
            "• q = Quit",
            "",
            f"Current color: {self.current_color.upper()}"
        ]
        
        self.text_actor.SetText("\n".join(instructions))
        self.plotter.render()
    
    def update_statistics(self):
        """Print current statistics"""
        counts = {'inlet': 0, 'outlet': 0, 'wall': 0}
        
        for data in self.highlighted_faces.values():
            color = data['color']
            if color in counts:
                counts[color] += 1
        
        total = sum(counts.values())
        print(f"📊 Current: Inlet={counts['inlet']}, Outlet={counts['outlet']}, Wall={counts['wall']}, Total={total}")
    
    def run(self):
        """Run the interactive session"""
        
        if not self.load_mesh():
            return False
        
        self.setup_plotter()
        
        print("\n🎮 Starting interactive session...")
        print("Click on faces to highlight them!")
        print("Use keyboard shortcuts to change modes")
        print("Press 'q' to quit")
        
        # Show the plotter
        self.plotter.show()
        
        print("✅ Interactive session completed!")
        
        # Print final summary
        if self.highlighted_faces:
            print("\n🎉 Final Summary:")
            self.update_statistics()
        else:
            print("\n❌ No faces were highlighted")
        
        return True

def main():
    """Main function"""
    
    print("🚀 INTERACTIVE MESH FACE HIGHLIGHTER")
    print("=" * 50)
    
    # Mesh file
    mesh_file = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\mesh\Rocket Engine.msh"
    
    if not os.path.exists(mesh_file):
        print(f"❌ Mesh file not found: {mesh_file}")
        return
    
    # Create and run highlighter
    highlighter = InteractiveMeshHighlighter(mesh_file)
    highlighter.run()

if __name__ == "__main__":
    main() 