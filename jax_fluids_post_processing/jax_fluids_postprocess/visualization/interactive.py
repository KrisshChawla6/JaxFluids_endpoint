"""
Interactive 3D visualization using PyVista.
Provides professional-grade visualization with mesh embedding and dynamic controls.
"""

from pathlib import Path
from typing import Dict, Optional, Union, Tuple
import numpy as np
import pyvista as pv

from ..utils.grid_utils import create_structured_grid


class InteractiveVisualizer:
    """
    Interactive 3D visualizer for JAX-Fluids data.
    
    Features:
    - Dynamic slice planes with slider control
    - 3D mesh embedding and scaling
    - Flow streamlines
    - Professional black background
    - Interactive controls (keyboard + mouse)
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        self.plotter = None
        self.current_data = None
        self.current_mesh = None
    
    def show_interactive(
        self,
        flow_data: Dict[str, np.ndarray],
        variable: str = "velocity_magnitude",
        mesh_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None
    ) -> None:
        """
        Show interactive 3D visualization.
        
        Args:
            flow_data: Dictionary of flow variables
            variable: Variable to visualize
            mesh_path: Optional path to mesh file
            title: Optional plot title
        """
        if variable not in flow_data:
            available = [k for k in flow_data.keys() if not k.startswith('_')]
            raise ValueError(f"Variable '{variable}' not found. Available: {available}")
        
        # Store current data
        self.current_data = flow_data
        
        # Create structured grid
        metadata = flow_data.get('_metadata', {})
        grid_shape = metadata.get('grid_shape', flow_data[variable].shape)
        domain_bounds = metadata.get('domain_bounds', self._estimate_bounds(grid_shape))
        
        flow_grid = create_structured_grid(
            data_dict=flow_data,
            grid_shape=grid_shape,
            domain_bounds=domain_bounds
        )
        
        # Load mesh if provided
        mesh = None
        if mesh_path and Path(mesh_path).exists():
            try:
                mesh = pv.read(str(mesh_path))
                print(f"✓ Loaded mesh: {mesh.n_points} points, {mesh.n_cells} cells")
            except Exception as e:
                print(f"⚠️  Could not load mesh: {e}")
        
        # Create plotter
        self.plotter = pv.Plotter()
        self.plotter.set_background('black')
        
        # Set up interactive visualization
        self._setup_interactive_visualization(
            flow_grid=flow_grid,
            variable=variable,
            mesh=mesh,
            title=title or f"{variable.replace('_', ' ').title()} Visualization"
        )
        
        # Show
        print("🎯 Interactive visualization opening...")
        print("Controls:")
        print("  • Slider: Change slice position")
        print("  • Keys 1/2/3: Switch between XY/XZ/YZ planes")
        print("  • Key P: Create 2D animation")
        print("  • Mouse: Rotate (left), Zoom (right), Pan (middle)")
        
        self.plotter.show()
    
    def save_screenshot(
        self,
        flow_data: Dict[str, np.ndarray],
        variable: str,
        output_path: Union[str, Path],
        mesh_path: Optional[Union[str, Path]] = None,
        resolution: Tuple[int, int] = (1920, 1080)
    ) -> Path:
        """
        Save a screenshot of the visualization.
        
        Args:
            flow_data: Dictionary of flow variables
            variable: Variable to visualize
            output_path: Path to save screenshot
            mesh_path: Optional path to mesh file
            resolution: Image resolution (width, height)
            
        Returns:
            Path to saved screenshot
        """
        # Create off-screen plotter
        plotter = pv.Plotter(off_screen=True, window_size=resolution)
        plotter.set_background('black')
        
        # Create grid and add data
        metadata = flow_data.get('_metadata', {})
        grid_shape = metadata.get('grid_shape', flow_data[variable].shape)
        domain_bounds = metadata.get('domain_bounds', self._estimate_bounds(grid_shape))
        
        flow_grid = create_structured_grid(
            data_dict=flow_data,
            grid_shape=grid_shape,
            domain_bounds=domain_bounds
        )
        
        # Add slice
        slice_plane = flow_grid.slice(normal='z', origin=flow_grid.center)
        plotter.add_mesh(
            slice_plane,
            scalars=variable,
            cmap='viridis',
            show_edges=False,
            opacity=0.8
        )
        
        # Add mesh if provided
        if mesh_path and Path(mesh_path).exists():
            try:
                mesh = pv.read(str(mesh_path))
                mesh_scaled = self._scale_mesh_to_domain(mesh, domain_bounds)
                plotter.add_mesh(
                    mesh_scaled,
                    color='yellow',
                    style='wireframe',
                    line_width=2,
                    opacity=1.0
                )
            except Exception:
                pass
        
        # Set camera and save
        plotter.camera_position = 'iso'
        plotter.add_text(
            f"{variable.replace('_', ' ').title()}",
            position='upper_left',
            font_size=14,
            color='white'
        )
        
        screenshot_path = Path(output_path)
        plotter.screenshot(str(screenshot_path))
        plotter.close()
        
        print(f"✓ Screenshot saved: {screenshot_path}")
        return screenshot_path
    
    def _setup_interactive_visualization(
        self,
        flow_grid: pv.StructuredGrid,
        variable: str,
        mesh: Optional[pv.PolyData] = None,
        title: str = "Flow Visualization"
    ) -> None:
        """Set up the interactive visualization with controls."""
        # Store data on plotter for callbacks
        self.plotter.flow_grid = flow_grid
        self.plotter.variable = variable
        self.plotter.current_plane = 'z'
        self.plotter.current_position = 0.5
        
        # Get domain bounds
        bounds = flow_grid.bounds
        domain_bounds = [bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]]
        self.plotter.domain_bounds = domain_bounds
        
        # Set up plane configurations
        self.plotter.plane_configs = {
            'z': {'normal': 'z', 'bounds': (domain_bounds[4], domain_bounds[5])},
            'y': {'normal': 'y', 'bounds': (domain_bounds[2], domain_bounds[3])},
            'x': {'normal': 'x', 'bounds': (domain_bounds[0], domain_bounds[1])}
        }
        
        # Add initial slice
        self._update_slice(0.5)
        
        # Add mesh if provided
        if mesh is not None:
            mesh_scaled = self._scale_mesh_to_domain(mesh, domain_bounds)
            self.plotter.add_mesh(
                mesh_scaled,
                color='yellow',
                style='wireframe',
                line_width=3,
                opacity=1.0,
                name='mesh'
            )
        
        # Add streamlines
        self._add_streamlines(flow_grid)
        
        # Add domain outline
        outline = flow_grid.outline()
        self.plotter.add_mesh(outline, color='gray', line_width=1, opacity=0.3)
        
        # Add slider
        self.plotter.add_slider_widget(
            callback=self._update_slice,
            rng=[0.0, 1.0],
            value=0.5,
            title="Slice Position",
            pointa=(0.1, 0.1),
            pointb=(0.4, 0.1),
            style='modern'
        )
        
        # Add keyboard shortcuts
        self.plotter.add_key_event('1', self._switch_to_xy_plane)
        self.plotter.add_key_event('2', self._switch_to_xz_plane) 
        self.plotter.add_key_event('3', self._switch_to_yz_plane)
        self.plotter.add_key_event('p', self._create_animation)
        
        # Add title and instructions
        self.plotter.add_text(
            title,
            position='upper_left',
            font_size=16,
            color='white',
            name='title'
        )
        
        self.plotter.add_text(
            "1=XY  2=XZ  3=YZ  P=Animation",
            position='lower_left', 
            font_size=10,
            color='white'
        )
        
        # Set camera
        self.plotter.camera_position = 'iso'
    
    def _update_slice(self, value: float) -> None:
        """Update slice position based on slider value."""
        plane_config = self.plotter.plane_configs[self.plotter.current_plane]
        bounds = plane_config['bounds']
        position = bounds[0] + value * (bounds[1] - bounds[0])
        
        # Remove old slice
        if 'slice' in self.plotter.renderer._actors:
            self.plotter.remove_actor('slice')
        
        # Create new slice
        slice_plane = self.plotter.flow_grid.slice(
            normal=plane_config['normal'],
            origin=(
                self.plotter.domain_bounds[0] + (self.plotter.domain_bounds[1] - self.plotter.domain_bounds[0])/2,
                self.plotter.domain_bounds[2] + (self.plotter.domain_bounds[3] - self.plotter.domain_bounds[2])/2,
                position
            ) if plane_config['normal'] == 'z' else None
        )
        
        # Add new slice
        self.plotter.add_mesh(
            slice_plane,
            scalars=self.plotter.variable,
            cmap='viridis',
            show_edges=False,
            opacity=0.8,
            name='slice'
        )
        
        # Update title
        plane_name = {'x': 'YZ', 'y': 'XZ', 'z': 'XY'}[self.plotter.current_plane]
        self.plotter.add_text(
            f"{self.plotter.variable.replace('_', ' ').title()} - {plane_name} Plane",
            position='upper_left',
            font_size=16,
            color='white',
            name='title'
        )
        
        self.plotter.current_position = value
    
    def _switch_to_xy_plane(self) -> None:
        """Switch to XY plane (Z-normal)."""
        self.plotter.current_plane = 'z'
        self._update_slice(self.plotter.current_position)
    
    def _switch_to_xz_plane(self) -> None:
        """Switch to XZ plane (Y-normal).""" 
        self.plotter.current_plane = 'y'
        self._update_slice(self.plotter.current_position)
    
    def _switch_to_yz_plane(self) -> None:
        """Switch to YZ plane (X-normal)."""
        self.plotter.current_plane = 'x' 
        self._update_slice(self.plotter.current_position)
    
    def _create_animation(self) -> None:
        """Create 2D animation (placeholder for now)."""
        print(f"🎬 Creating animation for {self.plotter.current_plane.upper()} plane at position {self.plotter.current_position:.2f}...")
        print("Animation creation would be implemented here")
    
    def _add_streamlines(self, flow_grid: pv.StructuredGrid) -> None:
        """Add flow streamlines to visualization."""
        if 'velocity_u' not in flow_grid.array_names:
            return
        
        try:
            # Create seed points
            bounds = flow_grid.bounds
            seed_points = [
                [bounds[0] + 0.1*(bounds[1]-bounds[0]), bounds[2] + 0.5*(bounds[3]-bounds[2]), bounds[4] + 0.5*(bounds[5]-bounds[4])],
                [bounds[0] + 0.1*(bounds[1]-bounds[0]), bounds[2] + 0.3*(bounds[3]-bounds[2]), bounds[4] + 0.7*(bounds[5]-bounds[4])]
            ]
            
            for i, seed in enumerate(seed_points):
                try:
                    streamlines = flow_grid.streamlines_from_source(
                        source=pv.PolyData(seed),
                        vectors='velocity',
                        max_time=100.0,
                        initial_step_length=0.1,
                        max_step_length=1.0
                    )
                    
                    if streamlines.n_points > 0:
                        self.plotter.add_mesh(
                            streamlines,
                            color='white',
                            line_width=2,
                            opacity=0.7,
                            name=f'streamlines_{i}'
                        )
                        print(f"  ✓ Added streamline set {i+1}: {streamlines.n_points} points")
                except Exception:
                    continue
                    
        except Exception as e:
            print(f"⚠️  Could not add streamlines: {e}")
    
    def _scale_mesh_to_domain(
        self, 
        mesh: pv.PolyData, 
        domain_bounds: list
    ) -> pv.PolyData:
        """Scale and position mesh to fit within domain."""
        mesh_bounds = mesh.bounds
        domain_size = max(
            domain_bounds[1] - domain_bounds[0],
            domain_bounds[3] - domain_bounds[2], 
            domain_bounds[5] - domain_bounds[4]
        )
        mesh_size = max(
            mesh_bounds[1] - mesh_bounds[0],
            mesh_bounds[3] - mesh_bounds[2],
            mesh_bounds[5] - mesh_bounds[4]
        )
        
        # Scale factor to make mesh ~1/10 of domain size
        scale_factor = (domain_size * 0.1) / mesh_size
        
        # Center mesh in domain
        mesh_center = [
            (mesh_bounds[0] + mesh_bounds[1]) / 2,
            (mesh_bounds[2] + mesh_bounds[3]) / 2,
            (mesh_bounds[4] + mesh_bounds[5]) / 2
        ]
        domain_center = [
            (domain_bounds[0] + domain_bounds[1]) / 2,
            (domain_bounds[2] + domain_bounds[3]) / 2,
            (domain_bounds[4] + domain_bounds[5]) / 2
        ]
        
        # Apply transformations
        mesh_scaled = mesh.copy()
        mesh_scaled.scale(scale_factor, inplace=True)
        mesh_scaled.translate(
            [domain_center[i] - mesh_center[i] * scale_factor for i in range(3)],
            inplace=True
        )
        
        return mesh_scaled
    
    def _estimate_bounds(self, grid_shape: Tuple[int, int, int]) -> list:
        """Estimate domain bounds if not provided."""
        return [0, grid_shape[0]-1, 0, grid_shape[1]-1, 0, grid_shape[2]-1]