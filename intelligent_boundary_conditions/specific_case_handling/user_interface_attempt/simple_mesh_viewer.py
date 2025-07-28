#!/usr/bin/env python3
"""
Simple Mesh Viewer with Tkinter GUI for Face Selection
Fallback option when PyVista is not available
"""

import os
import sys
import numpy as np
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

# Add the parent directories to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

from geometry_parser import GeometryParser

class SimpleMeshViewer:
    """
    Simple mesh viewer with face selection using matplotlib and tkinter
    """
    
    def __init__(self, root):
        """Initialize the simple mesh viewer"""
        self.root = root
        self.root.title("Simple Mesh Face Picker")
        self.root.geometry("1200x800")
        
        # Data
        self.geometry_parser = None
        self.faces = []
        self.selected_faces = {}
        self.current_tag = 'inlet'
        
        # Colors
        self.colors = {
            'inlet': 'red',
            'outlet': 'green',
            'wall': 'gray',
            'unselected': 'lightblue'
        }
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI layout"""
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        control_frame = ttk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # Right panel for plot
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Setup controls
        self.setup_controls(control_frame)
        
        # Setup plot
        self.setup_plot(plot_frame)
        
    def setup_controls(self, parent):
        """Setup control panel"""
        
        # Title
        title_label = ttk.Label(parent, text="Mesh Face Picker", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # File selection
        file_frame = ttk.LabelFrame(parent, text="Mesh File", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.file_label = ttk.Label(file_frame, text="No file selected", wraplength=250)
        self.file_label.pack(fill=tk.X)
        
        ttk.Button(file_frame, text="Browse...", command=self.browse_file).pack(pady=(5, 0))
        ttk.Button(file_frame, text="Load Default", command=self.load_default_file).pack(pady=(2, 0))
        
        # Tagging mode
        mode_frame = ttk.LabelFrame(parent, text="Tagging Mode", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.tag_var = tk.StringVar(value='inlet')
        
        ttk.Radiobutton(mode_frame, text="Inlet (Red)", variable=self.tag_var, 
                       value='inlet', command=self.update_mode).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Outlet (Green)", variable=self.tag_var, 
                       value='outlet', command=self.update_mode).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Wall (Gray)", variable=self.tag_var, 
                       value='wall', command=self.update_mode).pack(anchor=tk.W)
        
        # Statistics
        stats_frame = ttk.LabelFrame(parent, text="Selection Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=6, width=30)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Actions
        action_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(action_frame, text="Clear All", command=self.clear_selections).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Undo Last", command=self.undo_selection).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Save Selections", command=self.save_selections).pack(fill=tk.X, pady=2)
        
        # Instructions
        instr_frame = ttk.LabelFrame(parent, text="Instructions", padding=10)
        instr_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        instructions = """1. Load a mesh file
2. Select tagging mode (Inlet/Outlet/Wall)
3. Click on faces in the 3D plot to tag them
4. Different colors represent different tags
5. Use actions to manage selections
6. Save when finished"""
        
        ttk.Label(instr_frame, text=instructions, justify=tk.LEFT, wraplength=250).pack(anchor=tk.W)
        
    def setup_plot(self, parent):
        """Setup matplotlib plot"""
        
        # Create figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Setup click handling
        self.canvas.mpl_connect('button_press_event', self.on_face_click)
        
        # Initialize face data for picking
        self.face_data_for_picking = []
        
        # Initial empty plot
        self.ax.set_title("Load a mesh file to begin")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        
    def browse_file(self):
        """Browse for mesh file"""
        
        file_path = filedialog.askopenfilename(
            title="Select Mesh File",
            filetypes=[
                ("Mesh files", "*.msh *.stl *.obj *.ply"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_mesh_file(file_path)
    
    def load_default_file(self):
        """Load the default rocket engine mesh"""
        
        default_file = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\mesh\Rocket Engine.msh"
        if os.path.exists(default_file):
            self.load_mesh_file(default_file)
        else:
            messagebox.showerror("Error", "Default file not found")
    
    def load_mesh_file(self, file_path):
        """Load and display mesh file"""
        
        try:
            print(f"Loading mesh: {file_path}")
            
            # Parse geometry
            self.geometry_parser = GeometryParser(file_path)
            geometry_data = self.geometry_parser.parse_geometry()
            
            # Update file label
            filename = os.path.basename(file_path)
            self.file_label.config(text=f"Loaded: {filename}")
            
            # Prepare face data for plotting
            self.prepare_face_data()
            
            # Plot mesh
            self.plot_mesh()
            
            # Update stats
            self.update_statistics()
            
            print(f"✅ Loaded {len(self.faces)} faces")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load mesh: {e}")
            print(f"❌ Error loading mesh: {e}")
    
    def prepare_face_data(self):
        """Prepare face data for matplotlib plotting"""
        
        self.faces = []
        
        # Sample faces for performance (matplotlib can be slow with many faces)
        max_faces = 5000  # Limit for performance
        total_faces = len(self.geometry_parser.faces)
        
        if total_faces > max_faces:
            # Sample faces evenly
            step = total_faces // max_faces
            sampled_indices = range(0, total_faces, step)
            print(f"📊 Sampling {len(sampled_indices)} faces from {total_faces} total for display")
        else:
            sampled_indices = range(total_faces)
        
        for i in sampled_indices:
            face = self.geometry_parser.faces[i]
            if len(face.vertices) >= 3:
                # Use first 3 vertices for triangle
                triangle = [face.vertices[0], face.vertices[1], face.vertices[2]]
                self.faces.append({
                    'vertices': triangle,
                    'centroid': face.centroid,
                    'id': i,
                    'area': face.area
                })
    
    def plot_mesh(self):
        """Plot the mesh using matplotlib"""
        
        self.ax.clear()
        
        if not self.faces:
            self.ax.set_title("No faces to display")
            return
        
        # Prepare face collections
        face_vertices = []
        face_colors = []
        
        for face in self.faces:
            face_vertices.append(face['vertices'])
            
            # Color based on selection
            face_id = face['id']
            if face_id in self.selected_faces:
                tag = self.selected_faces[face_id]['tag']
                color = self.colors[tag]
            else:
                color = self.colors['unselected']
            
            face_colors.append(color)
        
        # Create 3D polygon collection
        poly_collection = Poly3DCollection(
            face_vertices,
            facecolors=face_colors,
            edgecolors='black',
            linewidths=0.1,
            alpha=0.7
        )
        
        self.ax.add_collection3d(poly_collection)
        
        # Set axis limits
        all_vertices = np.array([v for face in self.faces for v in face['vertices']])
        if len(all_vertices) > 0:
            self.ax.set_xlim(all_vertices[:, 0].min(), all_vertices[:, 0].max())
            self.ax.set_ylim(all_vertices[:, 1].min(), all_vertices[:, 1].max())
            self.ax.set_zlim(all_vertices[:, 2].min(), all_vertices[:, 2].max())
        
        # Labels and title
        self.ax.set_xlabel("X (Flow Direction)")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title(f"Mesh Viewer - {len(self.faces)} faces displayed")
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color=self.colors['inlet'], label='Inlet'),
            mpatches.Patch(color=self.colors['outlet'], label='Outlet'),
            mpatches.Patch(color=self.colors['wall'], label='Wall'),
            mpatches.Patch(color=self.colors['unselected'], label='Unselected')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')
        
        # Store face data for click detection
        self.face_data_for_picking = []
        for face in self.faces:
            centroid = face['centroid']
            self.face_data_for_picking.append({
                'id': face['id'],
                'centroid': centroid,
                'vertices': face['vertices']
            })
        
        self.canvas.draw()
    
    def on_face_click(self, event):
        """Handle mouse click on plot"""
        
        if event.inaxes != self.ax or not self.face_data_for_picking:
            return
        
        # Get click coordinates (this is approximate for 3D)
        if event.xdata is None or event.ydata is None:
            return
        
        click_point = np.array([event.xdata, event.ydata, 0])  # Z is approximated
        
        # Find closest face centroid (simplified approach)
        min_distance = float('inf')
        closest_face_id = None
        
        for face_data in self.face_data_for_picking:
            centroid = np.array(face_data['centroid'])
            # Project to 2D for distance calculation (simplified)
            centroid_2d = centroid[:2]
            click_2d = click_point[:2]
            distance = np.linalg.norm(centroid_2d - click_2d)
            
            if distance < min_distance:
                min_distance = distance
                closest_face_id = face_data['id']
        
        if closest_face_id is not None:
            self.tag_face(closest_face_id)
    
    def tag_face(self, face_id):
        """Tag a face with current mode"""
        
        current_tag = self.tag_var.get()
        
        # Store selection
        import time
        self.selected_faces[face_id] = {
            'tag': current_tag,
            'timestamp': time.time()
        }
        
        print(f"👆 Tagged face {face_id} as '{current_tag}'")
        
        # Update visualization
        self.plot_mesh()
        self.update_statistics()
    
    def update_mode(self):
        """Update current tagging mode"""
        self.current_tag = self.tag_var.get()
        print(f"🎯 Mode changed to: {self.current_tag}")
    
    def clear_selections(self):
        """Clear all face selections"""
        count = len(self.selected_faces)
        self.selected_faces.clear()
        
        if self.faces:
            self.plot_mesh()
        
        self.update_statistics()
        print(f"🧹 Cleared {count} selections")
    
    def undo_selection(self):
        """Undo the last selection"""
        if not self.selected_faces:
            print("❌ No selections to undo")
            return
        
        # Find most recent selection
        last_face_id = max(self.selected_faces.keys(), 
                          key=lambda k: self.selected_faces[k]['timestamp'])
        
        tag = self.selected_faces[last_face_id]['tag']
        del self.selected_faces[last_face_id]
        
        if self.faces:
            self.plot_mesh()
        
        self.update_statistics()
        print(f"↩️ Undid selection of face {last_face_id} ({tag})")
    
    def save_selections(self):
        """Save face selections to JSON file"""
        
        if not self.selected_faces:
            messagebox.showwarning("Warning", "No selections to save")
            return
        
        # Prepare output data
        output_data = {
            'mesh_file': self.geometry_parser.file_path if self.geometry_parser else "unknown",
            'total_displayed_faces': len(self.faces),
            'total_mesh_faces': len(self.geometry_parser.faces) if self.geometry_parser else 0,
            'selections': {}
        }
        
        # Group by tag
        for tag in ['inlet', 'outlet', 'wall']:
            output_data['selections'][tag] = []
        
        for face_id, selection in self.selected_faces.items():
            tag = selection['tag']
            output_data['selections'][tag].append({
                'face_id': face_id,
                'timestamp': selection['timestamp']
            })
        
        # Save to file
        output_file = 'simple_mesh_viewer_selections.json'
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            messagebox.showinfo("Success", f"Selections saved to: {output_file}")
            print(f"💾 Saved {len(self.selected_faces)} selections to: {output_file}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")
    
    def update_statistics(self):
        """Update the statistics display"""
        
        # Count selections by tag
        counts = {'inlet': 0, 'outlet': 0, 'wall': 0}
        for selection in self.selected_faces.values():
            counts[selection['tag']] += 1
        
        total = sum(counts.values())
        
        # Prepare stats text
        stats = f"""Current Selections:
        
Inlet faces: {counts['inlet']}
Outlet faces: {counts['outlet']}
Wall faces: {counts['wall']}
        
Total selected: {total}
Total displayed: {len(self.faces)}"""
        
        if self.geometry_parser:
            total_mesh = len(self.geometry_parser.faces)
            stats += f"\nTotal in mesh: {total_mesh}"
        
        # Update text widget
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats)

def main():
    """Main function to run the simple mesh viewer"""
    
    print("🚀 SIMPLE MESH VIEWER WITH FACE PICKING")
    print("=" * 50)
    
    # Create tkinter app
    root = tk.Tk()
    app = SimpleMeshViewer(root)
    
    print("✅ GUI started - use the interface to load and tag mesh faces")
    
    # Run the app
    root.mainloop()
    
    print("✅ Simple mesh viewer closed")

if __name__ == "__main__":
    main() 