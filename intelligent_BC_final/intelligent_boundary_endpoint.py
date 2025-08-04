#!/usr/bin/env python3
"""
Intelligent Boundary Condition Endpoint
======================================

Main endpoint that processes mesh files and outputs boundary condition masks
for JAX-Fluids internal flow simulations.

Combines functionality from:
- circular_face_creator.py (boundary detection)
- generate_jax_masks.py (mask generation)  
- Working rocket simulation implementation
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime

# Import required libraries with error handling
try:
    import pyvista as pv
    import meshio
    MESH_LIBRARIES_AVAILABLE = True
except ImportError:
    MESH_LIBRARIES_AVAILABLE = False

try:
    from scipy.spatial import ConvexHull
    from sklearn.cluster import DBSCAN
    SCIPY_SKLEARN_AVAILABLE = True
except ImportError:
    SCIPY_SKLEARN_AVAILABLE = False

class IntelligentBoundaryEndpoint:
    """
    Main endpoint for intelligent boundary condition generation
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the endpoint
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Check dependencies
        self._check_dependencies()
        
    def _check_dependencies(self):
        """Check required dependencies"""
        missing_deps = []
        
        if not MESH_LIBRARIES_AVAILABLE:
            missing_deps.append("pyvista/meshio")
        if not SCIPY_SKLEARN_AVAILABLE:
            missing_deps.append("scipy/sklearn")
            
        if missing_deps:
            raise ImportError(f"Missing required dependencies: {', '.join(missing_deps)}")
            
        if self.verbose:
            self.logger.info("✅ All dependencies available")

    def find_circular_boundary_edges(self, mesh_file: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Find circular boundary edges at inlet and outlet openings
        (Based on circular_face_creator.py)
        """
        
        if self.verbose:
            self.logger.info("🔍 Finding circular boundary edges...")
        
        # Load with meshio first
        mesh = meshio.read(mesh_file)
        if self.verbose:
            self.logger.info(f"   Original mesh: {len(mesh.points)} points, {len(mesh.cells)} cell blocks")
        
        pv_mesh = pv.from_meshio(mesh)
        surface = pv_mesh.extract_surface()
        if self.verbose:
            self.logger.info(f"   Surface mesh: {surface.n_points} points, {surface.n_cells} cells")
        
        # Method 1: Try standard boundary edge extraction
        boundary_edges = surface.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=False,
            feature_edges=False,
            manifold_edges=False
        )
        
        if self.verbose:
            self.logger.info(f"   Method 1 - Standard boundary edges: {boundary_edges.n_points} points")
        
        # Method 2: Try with different feature edge settings
        if boundary_edges.n_points == 0:
            if self.verbose:
                self.logger.info("   Trying method 2 - Non-manifold edges...")
            boundary_edges = surface.extract_feature_edges(
                boundary_edges=True,
                non_manifold_edges=True,
                feature_edges=True,
                manifold_edges=False
            )
            if self.verbose:
                self.logger.info(f"   Method 2 - Non-manifold edges: {boundary_edges.n_points} points")
        
        # Method 3: Analyze mesh topology for holes
        if boundary_edges.n_points == 0:
            if self.verbose:
                self.logger.info("   Trying method 3 - Analyzing mesh topology...")
            
            all_points = surface.points
            x_coords = all_points[:, 0]
            x_min, x_max = x_coords.min(), x_coords.max()
            x_range = x_max - x_min
            
            inlet_threshold = x_min + 0.05 * x_range
            outlet_threshold = x_max - 0.05 * x_range
            
            inlet_candidates = all_points[x_coords < inlet_threshold]
            outlet_candidates = all_points[x_coords > outlet_threshold]
            
            inlet_points = self._find_circular_pattern(inlet_candidates, "Inlet")
            outlet_points = self._find_circular_pattern(outlet_candidates, "Outlet")
            
            if inlet_points is not None and outlet_points is not None:
                return inlet_points, outlet_points
        
        # Method 4: Process boundary edges
        if boundary_edges.n_points > 0:
            edge_points = boundary_edges.points
            x_coords = edge_points[:, 0]
            x_min, x_max = x_coords.min(), x_coords.max()
            x_range = x_max - x_min
            
            # Inlet (low X) and outlet (high X)
            inlet_mask = x_coords < (x_min + 0.1 * x_range)
            outlet_mask = x_coords > (x_max - 0.1 * x_range)
            
            inlet_points = edge_points[inlet_mask]
            outlet_points = edge_points[outlet_mask]
            
            if self.verbose:
                self.logger.info(f"   Inlet region: {len(inlet_points)} points at X≈{inlet_points[:, 0].mean():.1f}")
                self.logger.info(f"   Outlet region: {len(outlet_points)} points at X≈{outlet_points[:, 0].mean():.1f}")
            
            return inlet_points, outlet_points
        
        if self.verbose:
            self.logger.error("❌ No boundary edges found with any method")
        return None, None

    def _find_circular_pattern(self, points: np.ndarray, region_name: str) -> Optional[np.ndarray]:
        """Find points that form a circular pattern"""
        if len(points) < 10:
            return None
            
        # Project to Y-Z plane
        yz_points = points[:, 1:]
        center_y = yz_points[:, 0].mean()
        center_z = yz_points[:, 1].mean()
        center = np.array([center_y, center_z])
        
        # Calculate distances from center
        distances = np.linalg.norm(yz_points - center, axis=1)
        
        # Find points that are roughly the same distance from center
        median_dist = np.median(distances)
        tolerance = median_dist * 0.2
        
        circle_mask = np.abs(distances - median_dist) < tolerance
        circle_points = points[circle_mask]
        
        if self.verbose:
            self.logger.info(f"   {region_name}: {len(circle_points)} points form circle pattern")
            self.logger.info(f"   {region_name}: radius ≈ {median_dist:.1f}")
        
        return circle_points if len(circle_points) > 20 else None

    def fit_circle_and_create_face(self, boundary_points: np.ndarray, face_type: str = "inlet") -> Optional[Dict[str, Any]]:
        """
        Fit a circle to boundary points and create face data
        (Based on circular_face_creator.py)
        """
        
        if self.verbose:
            self.logger.info(f"🔧 Creating circular {face_type} face...")
        
        if len(boundary_points) < 10:
            if self.verbose:
                self.logger.error(f"❌ Not enough boundary points for {face_type}")
            return None
        
        # Get the average X position for the face plane
        x_pos = boundary_points[:, 0].mean()
        
        # Project to Y-Z plane for circle fitting
        yz_points = boundary_points[:, 1:]
        
        # Fit circle using least squares
        center_y, center_z, radius = self._fit_circle_least_squares(yz_points)
        
        # Verify the fit quality
        distances = np.sqrt((yz_points[:, 0] - center_y)**2 + (yz_points[:, 1] - center_z)**2)
        fit_error = np.std(distances - radius)
        
        if self.verbose:
            self.logger.info(f"   Circle center: Y={center_y:.1f}, Z={center_z:.1f}")
            self.logger.info(f"   Circle radius: {radius:.1f}")
            self.logger.info(f"   Fit quality: σ={fit_error:.2f}")
        
        face_data = {
            'center': np.array([x_pos, center_y, center_z]),
            'radius': radius,
            'x_position': x_pos,
            'boundary_points': boundary_points,
            'fit_error': fit_error,
            'face_type': face_type
        }
        
        return face_data

    def _fit_circle_least_squares(self, points_2d: np.ndarray) -> Tuple[float, float, float]:
        """Fit circle using least squares method"""
        x, y = points_2d[:, 0], points_2d[:, 1]
        
        A = np.column_stack([x, y, np.ones(len(x))])
        b = x**2 + y**2
        
        try:
            coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
            cx = coeffs[0] / 2
            cy = coeffs[1] / 2
            r = np.sqrt(coeffs[2] + cx**2 + cy**2)
            return cx, cy, r
        except:
            # Fallback to simple center calculation
            cx = np.mean(x)
            cy = np.mean(y)
            r = np.mean(np.sqrt((x - cx)**2 + (y - cy)**2))
            return cx, cy, r

    def generate_boundary_masks(self, 
                               inlet_face: Dict[str, Any], 
                               outlet_face: Dict[str, Any],
                               domain_bounds: list = [-200.0, -800.0, -800.0, 1800.0, 800.0, 800.0],
                               grid_shape: tuple = (128, 64, 64)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 3D boolean masks for inlet/outlet regions
        (Based on generate_jax_masks.py)
        """
        
        if self.verbose:
            self.logger.info("🎯 Generating JAX-Fluids compatible masks...")
        
        # Create structured grid
        nx, ny, nz = grid_shape
        x_min, y_min, z_min, x_max, y_max, z_max = domain_bounds
        
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        z = np.linspace(z_min, z_max, nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Initialize masks
        inlet_mask = np.zeros(grid_shape, dtype=bool)
        outlet_mask = np.zeros(grid_shape, dtype=bool)
        
        # Get face parameters
        inlet_center = inlet_face['center']
        inlet_radius = inlet_face['radius']
        inlet_x = inlet_center[0]
        
        outlet_center = outlet_face['center']
        outlet_radius = outlet_face['radius']
        outlet_x = outlet_center[0]
        
        # Grid spacing for tolerance
        x_tolerance = (x_max - x_min) / nx * 2
        
        if self.verbose:
            self.logger.info(f"   🔵 Inlet: center=({inlet_center[0]:.1f}, {inlet_center[1]:.1f}, {inlet_center[2]:.1f}), radius={inlet_radius:.1f}")
            self.logger.info(f"   🔴 Outlet: center=({outlet_center[0]:.1f}, {outlet_center[1]:.1f}, {outlet_center[2]:.1f}), radius={outlet_radius:.1f}")
        
        # Generate masks using vectorized operations
        inlet_x_mask = np.abs(X - inlet_x) < x_tolerance
        inlet_distance = np.sqrt((Y - inlet_center[1])**2 + (Z - inlet_center[2])**2)
        inlet_mask = inlet_x_mask & (inlet_distance <= inlet_radius)
        
        outlet_x_mask = np.abs(X - outlet_x) < x_tolerance
        outlet_distance = np.sqrt((Y - outlet_center[1])**2 + (Z - outlet_center[2])**2)
        outlet_mask = outlet_x_mask & (outlet_distance <= outlet_radius)
        
        # Validate masks
        inlet_count = np.sum(inlet_mask)
        outlet_count = np.sum(outlet_mask)
        
        if inlet_count == 0:
            raise ValueError("❌ Generated inlet mask is empty")
        if outlet_count == 0:
            raise ValueError("❌ Generated outlet mask is empty")
        
        if self.verbose:
            self.logger.info(f"   ✅ Inlet mask: {inlet_count:,} active grid points")
            self.logger.info(f"   ✅ Outlet mask: {outlet_count:,} active grid points")
        
        return inlet_mask, outlet_mask

    def save_output_data(self, 
                        inlet_face: Dict[str, Any],
                        outlet_face: Dict[str, Any], 
                        inlet_mask: np.ndarray,
                        outlet_mask: np.ndarray,
                        output_dir: str,
                        simulation_name: str = "internal_flow") -> Dict[str, Any]:
        """Save all output data to the specified directory"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            self.logger.info(f"💾 Saving output data to: {output_path}")
        
        # Save masks
        inlet_mask_file = output_path / f"{simulation_name}_inlet_mask.npy"
        outlet_mask_file = output_path / f"{simulation_name}_outlet_mask.npy"
        
        np.save(inlet_mask_file, inlet_mask)
        np.save(outlet_mask_file, outlet_mask)

        # Save face data
        face_data = {
            "inlet_face": {
                "center": inlet_face['center'].tolist(),
                "radius": float(inlet_face['radius']),
                "x_position": float(inlet_face['x_position']),
                "fit_error": float(inlet_face['fit_error'])
            },
            "outlet_face": {
                "center": outlet_face['center'].tolist(), 
                "radius": float(outlet_face['radius']),
                "x_position": float(outlet_face['x_position']),
                "fit_error": float(outlet_face['fit_error'])
            },
            "mask_info": {
                "inlet_points": int(np.sum(inlet_mask)),
                "outlet_points": int(np.sum(outlet_mask)),
                "grid_shape": list(inlet_mask.shape),
                "domain_bounds": [-200.0, -800.0, -800.0, 1800.0, 800.0, 800.0]
            },
            "generation_timestamp": datetime.now().isoformat()
        }
        
        face_data_file = output_path / f"{simulation_name}_boundary_data.json"
        with open(face_data_file, 'w') as f:
            json.dump(face_data, f, indent=2)
        
        result = {
            "inlet_mask_file": str(inlet_mask_file),
            "outlet_mask_file": str(outlet_mask_file),
            "boundary_data_file": str(face_data_file),
            "inlet_points": np.sum(inlet_mask),
            "outlet_points": np.sum(outlet_mask),
            "output_directory": str(output_path)
        }
        
        if self.verbose:
            self.logger.info(f"   ✅ Inlet mask: {inlet_mask_file}")
            self.logger.info(f"   ✅ Outlet mask: {outlet_mask_file}")
            self.logger.info(f"   ✅ Boundary data: {face_data_file}")
        
        return result

    def export_vtk_visualization(self, 
                                mesh_file: str,
                                inlet_face: Dict[str, Any], 
                                outlet_face: Dict[str, Any],
                                output_dir: str,
                                simulation_name: str = "internal_flow") -> str:
        """
        Export a professional VTK visualization with arrows, domain box, and proper labeling
        
        Args:
            mesh_file: Path to original mesh file
            inlet_face: Inlet face data
            outlet_face: Outlet face data
            output_dir: Output directory
            simulation_name: Simulation name for file naming
            
        Returns:
            Path to exported VTK file
        """
        
        if self.verbose:
            self.logger.info("📦 Creating VTK visualization export...")
        
        try:
            import pyvista as pv
            import meshio
            
            # Load and process mesh
            mesh = meshio.read(mesh_file)
            pv_mesh = pv.from_meshio(mesh)
            surface = pv_mesh.extract_surface()
            
            if self.verbose:
                self.logger.info(f"   Loaded mesh: {surface.n_points:,} points, {surface.n_cells:,} cells")
            
            # Create multiblock dataset for organized VTK export
            multiblock = pv.MultiBlock()
            
            # 1. Add main rocket nozzle surface (Euler/slipwall)
            surface_copy = surface.copy()
            surface_copy['boundary_type'] = np.full(surface.n_cells, 1)  # Euler/slipwall = 1
            multiblock['rocket_nozzle'] = surface_copy
            
            # 2. Create and add inlet face
            inlet_face_mesh = self._create_circular_face_vtk(inlet_face, "inlet")
            if inlet_face_mesh:
                inlet_face_mesh['boundary_type'] = np.zeros(inlet_face_mesh.n_cells)  # Inlet = 0
                multiblock['inlet_face'] = inlet_face_mesh
                if self.verbose:
                    self.logger.info(f"   ✅ Added inlet face: {inlet_face_mesh.n_cells} cells")
            
            # 3. Create and add outlet face  
            outlet_face_mesh = self._create_circular_face_vtk(outlet_face, "outlet")
            if outlet_face_mesh:
                outlet_face_mesh['boundary_type'] = np.full(outlet_face_mesh.n_cells, 2)  # Outlet = 2
                multiblock['outlet_face'] = outlet_face_mesh
                if self.verbose:
                    self.logger.info(f"   ✅ Added outlet face: {outlet_face_mesh.n_cells} cells")
                
            # 4. Add flow direction arrows
            try:
                arrows = self._create_flow_arrows(inlet_face, outlet_face)
                if arrows:
                    multiblock['flow_arrows'] = arrows
                    if self.verbose:
                        self.logger.info(f"   ✅ Added flow arrows")
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"   ⚠️  Flow arrows failed: {e}")
                
            # 5. Add domain bounding box
            try:
                domain_box = self._create_domain_box()
                if domain_box:
                    multiblock['domain_box'] = domain_box
                    if self.verbose:
                        self.logger.info(f"   ✅ Added domain box")
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"   ⚠️  Domain box failed: {e}")
            
            # 6. Add text labels (optional)
            try:
                labels = self._create_text_labels(inlet_face, outlet_face)
                if labels:
                    multiblock['labels'] = labels
                    if self.verbose:
                        self.logger.info(f"   ✅ Added text labels")
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"   ⚠️  Text labels failed: {e}")
            
            # Save VTK file (MultiBlock format requires .vtm extension)
            vtk_file = Path(output_dir) / f"{simulation_name}_visualization.vtm"
            
            if self.verbose:
                self.logger.info(f"   📦 Saving MultiBlock with {len(multiblock)} components...")
                
            multiblock.save(str(vtk_file))
            
            if self.verbose:
                self.logger.info(f"   ✅ VTK visualization saved: {vtk_file}")
                self.logger.info(f"   📦 Contains: rocket_nozzle, inlet_face, outlet_face, flow_arrows, domain_box, labels")
            
            # Also create a summary ParaView state file for easy loading
            self._create_paraview_state_file(vtk_file, output_dir, simulation_name)
            
            return str(vtk_file)
            
        except Exception as e:
            if self.verbose:
                self.logger.error(f"❌ VTK export failed: {e}")
            raise
    
    def _create_circular_face_vtk(self, face_data: Dict[str, Any], face_type: str) -> Optional[Any]:
        """Create a circular face mesh for VTK export"""
        try:
            import pyvista as pv
            
            center = np.array(face_data['center'])
            radius = face_data['radius']
            x_pos = face_data['x_position']
            
            # Create high-quality circular face (more segments for VTK)
            n_segments = 128  # Higher resolution for VTK export
            angles = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
            
            # Create circle points
            circle_points = []
            for angle in angles:
                y = center[1] + radius * np.cos(angle)
                z = center[2] + radius * np.sin(angle)
                circle_points.append([x_pos, y, z])
            
            circle_points = np.array(circle_points)
            
            # Create triangular faces (fan triangulation from center)
            points = np.vstack([center.reshape(1, -1), circle_points])
            
            faces = []
            for i in range(n_segments):
                j = (i + 1) % n_segments
                # Triangle: center, point i, point j+1
                faces.extend([3, 0, i+1, j+1])
            
            faces = np.array(faces)
            
            # Create PyVista mesh
            face_mesh = pv.PolyData(points, faces)
            
            # Add metadata
            face_mesh[f'{face_type}_radius'] = np.full(face_mesh.n_points, radius)
            face_mesh[f'{face_type}_center_distance'] = np.linalg.norm(points - center, axis=1)
            
            return face_mesh
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️  Failed to create {face_type} face for VTK: {e}")
            return None
    
    def _create_flow_arrows(self, inlet_face: Dict[str, Any], outlet_face: Dict[str, Any]) -> Any:
        """Create flow direction arrows"""
        try:
            import pyvista as pv
            
            # Create arrow points and vectors
            inlet_center = np.array(inlet_face['center'])
            outlet_center = np.array(outlet_face['center'])
            
            # Flow direction (X-axis for rocket nozzle)
            flow_direction = np.array([1.0, 0.0, 0.0])
            
            # Create inlet arrow (inflow - pointing into nozzle)
            inlet_arrow_start = inlet_center - flow_direction * 200.0  # Start 200 units upstream
            
            # Create outlet arrow (outflow - pointing out of nozzle)  
            outlet_arrow_start = outlet_center
            
            # Create inlet arrow (red for inlet)
            inlet_arrow = pv.Arrow(start=inlet_arrow_start, direction=flow_direction, 
                                 tip_length=0.2, tip_radius=0.05, shaft_radius=0.02, scale=200.0)
            inlet_arrow['arrow_type'] = np.zeros(inlet_arrow.n_cells)  # 0 = inlet
            
            # Create outlet arrow (green for outlet) 
            outlet_arrow = pv.Arrow(start=outlet_arrow_start, direction=flow_direction,
                                  tip_length=0.15, tip_radius=0.04, shaft_radius=0.015, scale=300.0)
            outlet_arrow['arrow_type'] = np.full(outlet_arrow.n_cells, 2)  # 2 = outlet
            
            # Combine arrows into single mesh
            combined_arrows = inlet_arrow.merge(outlet_arrow)
            
            return combined_arrows
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️  Failed to create flow arrows: {e}")
            return None
    
    def _create_domain_box(self) -> Any:
        """Create domain bounding box"""
        try:
            import pyvista as pv
            
            # Domain bounds: [x_min, y_min, z_min, x_max, y_max, z_max]
            bounds = [-200.0, -800.0, -800.0, 1800.0, 800.0, 800.0]
            
            # Create wireframe box
            box = pv.Box(bounds=bounds)
            box_edges = box.extract_feature_edges()
            
            # Add metadata
            box_edges['domain_boundary'] = np.ones(box_edges.n_cells)
            
            return box_edges
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️  Failed to create domain box: {e}")
            return None
    
    def _create_text_labels(self, inlet_face: Dict[str, Any], outlet_face: Dict[str, Any]) -> Any:
        """Create text labels for inlet and outlet"""
        try:
            import pyvista as pv
            
            # Create point cloud for labels
            label_points = []
            label_text = []
            
            # Inlet label
            inlet_center = np.array(inlet_face['center'])
            inlet_label_pos = inlet_center + np.array([0, inlet_face['radius'] * 1.5, 0])
            label_points.append(inlet_label_pos)
            label_text.append(f"INLET\nR={inlet_face['radius']:.1f}mm")
            
            # Outlet label
            outlet_center = np.array(outlet_face['center'])
            outlet_label_pos = outlet_center + np.array([0, outlet_face['radius'] * 1.5, 0])
            label_points.append(outlet_label_pos)
            label_text.append(f"OUTLET\nR={outlet_face['radius']:.1f}mm")
            
            # Create point cloud
            points = pv.PolyData(np.array(label_points))
            points['labels'] = label_text
            
            return points
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️  Failed to create text labels: {e}")
            return None
    
    def _create_paraview_state_file(self, vtk_file: Path, output_dir: str, simulation_name: str):
        """Create a ParaView state file for easy visualization setup"""
        try:
            state_file = Path(output_dir) / f"{simulation_name}_paraview_setup.py"
            
            paraview_script = f'''#!/usr/bin/env python3
"""
ParaView Python Script for Intelligent Boundary Condition Visualization
Generated automatically by intelligent_BC_final endpoint

Usage:
1. Open ParaView
2. Tools -> Python Shell
3. Run Script -> Select this file
   OR
4. pvpython {simulation_name}_paraview_setup.py
"""

import paraview.simple as pvs

# Load VTK file
reader = pvs.OpenDataFile('{vtk_file.name}')

# Get render view
view = pvs.GetActiveViewOrCreate('RenderView')

# Set black background
view.Background = [0, 0, 0]

# Create displays for each component
rocket_display = pvs.Show(reader, view)

# Set up color scheme
rocket_display.ColorArrayName = ['CELLS', 'boundary_type']
rocket_display.LookupTable = pvs.GetColorTransferFunction('boundary_type')

# Configure lookup table for boundary types
lut = rocket_display.LookupTable
lut.RGBPoints = [
    0.0, 1.0, 0.0, 0.0,    # Inlet = Red
    1.0, 0.8, 0.8, 0.8,    # Euler/Slipwall = Light Gray  
    2.0, 0.0, 1.0, 0.0     # Outlet = Green
]
lut.ColorSpace = 'RGB'
lut.NanColor = [0.5, 0.5, 0.5]

# Add color legend
legend = pvs.GetScalarBar(lut, view)
legend.Title = 'Boundary Types'
legend.ComponentTitle = ''
legend.LabelFormat = '%-#6.0f'
legend.RangeLabelFormat = '%-#6.0f'

# Position and style legend
legend.Position = [0.85, 0.2]
legend.ScalarBarLength = 0.6
legend.ScalarBarThickness = 16
legend.TitleColor = [1, 1, 1]
legend.LabelColor = [1, 1, 1]

# Set camera position for good view
view.CameraPosition = [2500, 1500, 1000]
view.CameraFocalPoint = [800, 0, 0]
view.CameraViewUp = [0, 0, 1]

# Reset and render
pvs.ResetCamera()
pvs.Render()

print("ParaView visualization loaded successfully!")
print("Red = Inlet boundary")
print("Gray = Euler/Slipwall boundaries") 
print("Green = Outlet boundary")
print("Domain box and flow arrows included")
'''
            
            with open(state_file, 'w') as f:
                f.write(paraview_script)
                
            if self.verbose:
                self.logger.info(f"   📋 ParaView script saved: {state_file}")
                
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️  Failed to create ParaView state file: {e}")

    def process_mesh(self, mesh_file: str, output_dir: str, simulation_name: str = "internal_flow") -> Dict[str, Any]:
        """
        Main method: Process mesh file and generate boundary condition masks
        
        Args:
            mesh_file: Path to input mesh file (.msh)
            output_dir: Directory to save output files
            simulation_name: Name for output files
            
        Returns:
            Dictionary with results and file paths
        """
        
        if self.verbose:
            self.logger.info("🚀 INTELLIGENT BOUNDARY CONDITION GENERATION")
            self.logger.info("=" * 60)
            self.logger.info(f"Input mesh: {mesh_file}")
            self.logger.info(f"Output directory: {output_dir}")
            self.logger.info(f"Simulation name: {simulation_name}")
        
        try:
            # Check input file
            if not os.path.exists(mesh_file):
                raise FileNotFoundError(f"Mesh file not found: {mesh_file}")
            
            # Step 1: Detect boundary edges
            inlet_points, outlet_points = self.find_circular_boundary_edges(mesh_file)
            
            if inlet_points is None or outlet_points is None:
                raise RuntimeError("Failed to detect inlet/outlet boundaries")
            
            # Step 2: Create virtual faces
            inlet_face = self.fit_circle_and_create_face(inlet_points, "inlet")
            outlet_face = self.fit_circle_and_create_face(outlet_points, "outlet")
            
            if inlet_face is None or outlet_face is None:
                raise RuntimeError("Failed to create virtual boundary faces")
            
            # Step 3: Generate masks
            inlet_mask, outlet_mask = self.generate_boundary_masks(inlet_face, outlet_face)
            
                        # Step 4: Save output data
            result = self.save_output_data(
                inlet_face, outlet_face, inlet_mask, outlet_mask,
                output_dir, simulation_name
            )

            # Step 5: Export VTK visualization
            try:
                vtk_file = self.export_vtk_visualization(
                    mesh_file, inlet_face, outlet_face, output_dir, simulation_name
                )
                result['vtk_visualization'] = vtk_file
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"⚠️  VTK export failed: {e}")
                result['vtk_visualization'] = None

            if self.verbose:
                self.logger.info("🎉 BOUNDARY CONDITION GENERATION COMPLETED SUCCESSFULLY!")
                self.logger.info(f"   Output files saved to: {result['output_directory']}")
                self.logger.info(f"   Inlet points: {result['inlet_points']:,}")
                self.logger.info(f"   Outlet points: {result['outlet_points']:,}")
                if result.get('vtk_visualization'):
                    self.logger.info(f"   VTK visualization: {result['vtk_visualization']}")

            return result
            
        except Exception as e:
            if self.verbose:
                self.logger.error(f"❌ Processing failed: {e}")
            raise

def main():
    """Standalone execution for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate intelligent boundary conditions')
    parser.add_argument('mesh_file', help='Input mesh file (.msh)')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--name', default='internal_flow', help='Simulation name')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Process mesh
    endpoint = IntelligentBoundaryEndpoint()
    result = endpoint.process_mesh(args.mesh_file, args.output_dir, args.name)
    
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Inlet points: {result['inlet_points']:,}")
    print(f"Outlet points: {result['outlet_points']:,}")
    print(f"Output directory: {result['output_directory']}")
    print("="*60)

if __name__ == "__main__":
    main() 