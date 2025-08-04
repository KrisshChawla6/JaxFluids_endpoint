"""
Animation creation for 2D flow visualizations over time.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ..core.processor import FluidProcessor


class AnimationCreator:
    """
    Creates 2D animations of flow variables over time.
    
    Supports multiple output formats (GIF, MP4) and plane orientations.
    """
    
    def __init__(self):
        """Initialize the animation creator."""
        pass
    
    def create_time_series_animation(
        self,
        processor: FluidProcessor,
        variable: str = "velocity_magnitude",
        plane: str = "xy",
        plane_value: float = 0.5,
        time_range: Optional[Tuple[int, int]] = None,
        fps: int = 10,
        format: str = "gif",
        output_path: Union[str, Path] = ".",
        figsize: Tuple[float, float] = (12, 8),
        dpi: int = 100
    ) -> Path:
        """
        Create time series animation of a flow variable.
        
        Args:
            processor: FluidProcessor instance with loaded data
            variable: Variable to animate
            plane: Plane orientation ("xy", "xz", "yz")
            plane_value: Position along plane normal (0.0-1.0)
            time_range: Optional tuple (start_idx, end_idx)
            fps: Frames per second
            format: Output format ("gif" or "mp4")
            output_path: Directory to save animation
            figsize: Figure size (width, height)
            dpi: Resolution in DPI
            
        Returns:
            Path to created animation file
        """
        print(f"🎬 Creating {plane.upper()} plane animation for {variable}...")
        
        # Get time indices
        num_steps = processor.reader.get_num_time_steps()
        if time_range is None:
            time_indices = list(range(num_steps))
        else:
            start_idx, end_idx = time_range
            time_indices = list(range(start_idx, min(end_idx + 1, num_steps)))
        
        if len(time_indices) < 2:
            raise ValueError("Need at least 2 time steps for animation")
        
        # Extract data for all time steps
        print(f"Extracting data for {len(time_indices)} time steps...")
        time_series_data = []
        times = []
        
        for i, t_idx in enumerate(time_indices):
            flow_data = processor.extract_flow_variables(time_index=t_idx)
            
            if variable not in flow_data:
                available = [k for k in flow_data.keys() if not k.startswith('_')]
                raise ValueError(f"Variable '{variable}' not found. Available: {available}")
            
            # Extract 2D slice
            slice_2d = self._extract_2d_slice(
                flow_data[variable], 
                plane, 
                plane_value
            )
            time_series_data.append(slice_2d)
            
            # Get simulation time
            metadata = flow_data.get('_metadata', {})
            sim_time = metadata.get('simulation_time', t_idx)
            times.append(sim_time)
            
            print(f"  ✓ Time step {t_idx}: {slice_2d.shape} slice")
        
        # Create animation
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        animation_file = output_path / f"animation_{variable}_{plane}.{format}"
        
        self._create_matplotlib_animation(
            time_series_data=time_series_data,
            times=times,
            variable=variable,
            plane=plane,
            output_path=animation_file,
            fps=fps,
            figsize=figsize,
            dpi=dpi
        )
        
        print(f"✓ Animation saved: {animation_file}")
        return animation_file
    
    def _extract_2d_slice(
        self, 
        data_3d: np.ndarray, 
        plane: str, 
        plane_value: float
    ) -> np.ndarray:
        """
        Extract 2D slice from 3D data.
        
        Args:
            data_3d: 3D data array
            plane: Plane orientation ("xy", "xz", "yz")
            plane_value: Position along plane normal (0.0-1.0)
            
        Returns:
            2D slice array
        """
        if plane == "xy":
            # Z-slice
            z_idx = int(plane_value * (data_3d.shape[2] - 1))
            return data_3d[:, :, z_idx]
        elif plane == "xz":
            # Y-slice
            y_idx = int(plane_value * (data_3d.shape[1] - 1))
            return data_3d[:, y_idx, :]
        elif plane == "yz":
            # X-slice
            x_idx = int(plane_value * (data_3d.shape[0] - 1))
            return data_3d[x_idx, :, :]
        else:
            raise ValueError(f"Invalid plane '{plane}'. Use 'xy', 'xz', or 'yz'")
    
    def _create_matplotlib_animation(
        self,
        time_series_data: List[np.ndarray],
        times: List[float],
        variable: str,
        plane: str,
        output_path: Path,
        fps: int,
        figsize: Tuple[float, float],
        dpi: int
    ) -> None:
        """Create animation using matplotlib."""
        # Set up figure
        fig, ax = plt.subplots(figsize=figsize, facecolor='black')
        ax.set_facecolor('black')
        
        # Find global min/max for consistent colorbar
        all_data = np.concatenate([data.ravel() for data in time_series_data])
        vmin, vmax = np.percentile(all_data, [1, 99])
        
        # Create initial plot
        im = ax.imshow(
            time_series_data[0].T,  # Transpose for proper orientation
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            aspect='auto',
            origin='lower'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(
            variable.replace('_', ' ').title(),
            color='white',
            fontsize=12
        )
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.label.set_color('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Set labels and title
        plane_labels = {
            'xy': ('X', 'Y'),
            'xz': ('X', 'Z'), 
            'yz': ('Y', 'Z')
        }
        xlabel, ylabel = plane_labels[plane]
        
        ax.set_xlabel(xlabel, color='white', fontsize=12)
        ax.set_ylabel(ylabel, color='white', fontsize=12)
        ax.tick_params(colors='white')
        
        title = ax.set_title(
            f"{variable.replace('_', ' ').title()} - {plane.upper()} Plane\nTime: {times[0]:.3f}",
            color='white',
            fontsize=14
        )
        
        def animate(frame):
            """Animation function."""
            im.set_array(time_series_data[frame].T)
            title.set_text(
                f"{variable.replace('_', ' ').title()} - {plane.upper()} Plane\nTime: {times[frame]:.3f}"
            )
            return [im, title]
        
        # Create animation
        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=len(time_series_data),
            interval=1000//fps,
            blit=False,
            repeat=True
        )
        
        # Save animation
        if output_path.suffix.lower() == '.gif':
            writer = animation.PillowWriter(fps=fps)
        elif output_path.suffix.lower() == '.mp4':
            writer = animation.FFMpegWriter(fps=fps)
        else:
            raise ValueError(f"Unsupported format: {output_path.suffix}")
        
        anim.save(str(output_path), writer=writer, dpi=dpi)
        plt.close(fig)
        
        print(f"  ✓ Created {len(time_series_data)} frame animation at {fps} FPS")