#!/usr/bin/env python3
"""
Direct Mesh Plot with PyVista
Simple script to load and display the rocket engine mesh directly
"""

import os
import sys

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

def main():
    """Load and plot the rocket engine mesh directly"""
    
    print("🚀 DIRECT MESH PLOT - ROCKET ENGINE")
    print("=" * 50)
    
    # Mesh file path
    mesh_file = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\mesh\Rocket Engine.msh"
    
    if not os.path.exists(mesh_file):
        print(f"❌ Mesh file not found: {mesh_file}")
        return
    
    print(f"📂 Loading: {mesh_file}")
    
    try:
        # Load with meshio
        print("🔄 Loading with meshio...")
        mesh = meshio.read(mesh_file)
        
        print(f"📊 Mesh details:")
        print(f"   Points: {len(mesh.points):,}")
        print(f"   Cell blocks: {len(mesh.cells)}")
        
        for i, block in enumerate(mesh.cells):
            print(f"   Block {i}: {block.type} ({len(block.data):,} elements)")
        
        # Convert to PyVista
        print("🔄 Converting to PyVista...")
        pv_mesh = pv.from_meshio(mesh)
        print(f"✅ PyVista mesh: {pv_mesh.n_cells:,} cells, {pv_mesh.n_points:,} points")
        print(f"   Mesh type: {type(pv_mesh).__name__}")
        
        # Extract surface from volume mesh
        print("🔄 Extracting surface...")
        surface_mesh = pv_mesh.extract_surface()
        print(f"📊 Surface: {surface_mesh.n_cells:,} faces, {surface_mesh.n_points:,} points")
        
        # Create plotter
        print("🎮 Setting up plotter...")
        plotter = pv.Plotter(window_size=[1200, 800])
        
        # Add the surface mesh
        plotter.add_mesh(
            surface_mesh,
            color='lightblue',
            show_edges=True,
            edge_color='white',
            line_width=0.2,
            opacity=0.9,
            pickable=True,
            name='rocket_mesh'
        )
        
        # Track highlighted faces
        highlighted_faces = {}
        
        # Face picking callback
        def pick_face(picked_point):
            """Callback for face picking"""
            if hasattr(picked_point, 'picked_cells') and len(picked_point.picked_cells) > 0:
                face_id = picked_point.picked_cells[0]
                print(f"👆 Picked face: {face_id}")
                
                # Extract and highlight the face
                try:
                    picked_face = surface_mesh.extract_cells([face_id])
                    
                    # Remove previous highlight if exists
                    if face_id in highlighted_faces:
                        plotter.remove_actor(f'highlight_{face_id}')
                    
                    # Add new highlight
                    plotter.add_mesh(
                        picked_face,
                        color='red',
                        opacity=1.0,
                        name=f'highlight_{face_id}',
                        reset_camera=False
                    )
                    
                    highlighted_faces[face_id] = picked_face
                    print(f"✅ Highlighted face {face_id}")
                    
                except Exception as e:
                    print(f"❌ Failed to highlight face: {e}")
        
        # Enable picking
        plotter.enable_mesh_picking(callback=pick_face, show_message=True)
        
        # Add instructions
        instructions = [
            "🎯 ROCKET ENGINE MESH VIEWER",
            "",
            "INTERACTIONS:",
            "• Click on faces to highlight them in red",
            "• Drag to rotate view",
            "• Scroll to zoom",
            "• Right-click and drag to pan",
            "",
            "KEYBOARD:",
            "• 'r' - Reset camera view",
            "• 'q' - Quit viewer",
            "",
            f"Total faces: {surface_mesh.n_cells:,}"
        ]
        
        plotter.add_text(
            "\n".join(instructions),
            position='upper_left',
            font_size=10,
            color='white'
        )
        
        # Setup camera
        plotter.camera_position = 'iso'
        plotter.add_axes()
        
        # Show mesh bounds
        bounds = surface_mesh.bounds
        print(f"📐 Mesh bounds:")
        print(f"   X: {bounds[0]:.1f} to {bounds[1]:.1f}")
        print(f"   Y: {bounds[2]:.1f} to {bounds[3]:.1f}")
        print(f"   Z: {bounds[4]:.1f} to {bounds[5]:.1f}")
        
        print("\n🎮 Starting interactive viewer...")
        print("   Click on faces to highlight them!")
        print("   Press 'q' to quit")
        
        # Show the plotter
        plotter.show()
        
        print("✅ Viewer closed")
        
        # Print final stats
        if highlighted_faces:
            print(f"\n📊 You highlighted {len(highlighted_faces)} faces:")
            for face_id in highlighted_faces.keys():
                print(f"   Face {face_id}")
        else:
            print("\n❌ No faces were highlighted")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 