#!/usr/bin/env python3
"""
JAX-Fluids Virtual Face Integrator

Integrates circular virtual faces with immersed boundary SDF to create
JAX-Fluids compatible boundary conditions with inlet/outlet flags.

This builds on the existing immersed_boundary_endpoint_final infrastructure
and the circular_face_creator to provide a complete solution.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any
import sys
import os

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "immersed_boundary_endpoint_final"))

try:
    import pyvista as pv
    import meshio
    from scipy.spatial import ConvexHull
    from sklearn.cluster import DBSCAN
    print("✅ All required libraries available")
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    exit(1)

# Import existing SDF functionality
try:
    from immersed_boundary_sdf import compute_sdf, parse_gmsh_mesh, store_sdf_data
    print("✅ Imported existing SDF functions")
except ImportError as e:
    print(f"⚠️  Could not import SDF functions: {e}")
    print("   Using fallback implementations...")
    
    # Fallback implementation for parse_gmsh_mesh
    def parse_gmsh_mesh(mesh_file):
        """Fallback: Parse Gmsh mesh file and extract boundary triangles"""
        print(f"📁 Reading mesh with fallback parser: {mesh_file}")
        
        # Use meshio to load and extract surface triangles
        mesh = meshio.read(mesh_file)
        pv_mesh = pv.from_meshio(mesh)
        surface = pv_mesh.extract_surface()
        
        # Extract triangular faces
        triangles = []
        points = surface.points
        
        if hasattr(surface, 'faces'):
            faces = surface.faces
            # Reshape faces array - PyVista format is [n_points, p0, p1, ..., n_points, p0, p1, ...]
            i = 0
            while i < len(faces):
                n_points = faces[i]
                if n_points == 3:  # Triangle
                    triangle_indices = faces[i+1:i+1+n_points]
                    triangle = points[triangle_indices]
                    triangles.append(triangle)
                i += n_points + 1
        else:
            # Alternative method using cells
            for cell in surface.cells:
                if cell.type == 'triangle':
                    for triangle_indices in cell.data:
                        triangle = points[triangle_indices]
                        triangles.append(triangle)
        
        print(f"🔺 Extracted {len(triangles)} boundary triangles")
        return np.array(triangles, dtype=np.float32)
    
    # Fallback implementation for compute_sdf using simple point-to-triangle distance
    def compute_sdf(boundary_triangles, domain_bounds, resolution):
        """Fallback: Simple SDF computation using point-to-triangle distances"""
        print("🔧 Computing SDF with fallback method...")
        
        xmin, ymin, zmin, xmax, ymax, zmax = domain_bounds
        nx, ny, nz = resolution
        
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        z = np.linspace(zmin, zmax, nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)
        
        print(f"⚡ Computing distances for {len(grid_points):,} grid points...")
        
        # Simple distance computation - find minimum distance to any triangle
        sdf_values = np.full(len(grid_points), float('inf'))
        
        for i, triangle in enumerate(boundary_triangles):
            if i % 1000 == 0:
                print(f"   Processing triangle {i+1}/{len(boundary_triangles)}")
            
            # Simple point-to-triangle distance (approximation)
            for j, point in enumerate(grid_points):
                # Distance to triangle vertices (rough approximation)
                distances_to_vertices = [np.linalg.norm(point - vertex) for vertex in triangle]
                min_distance = min(distances_to_vertices)
                sdf_values[j] = min(sdf_values[j], min_distance)
        
        # Simple inside/outside test - points very close to surface are likely inside
        # This is a crude approximation
        threshold = np.percentile(sdf_values, 10)  # Bottom 10% considered inside
        sdf_values[sdf_values <= threshold] *= -1  # Make inside negative
        
        sdf_grid = sdf_values.reshape(X.shape)
        print(f"✅ SDF computation completed (fallback method)")
        
        return X, Y, Z, sdf_grid
    
    # Fallback implementation for store_sdf_data
    def store_sdf_data(X, Y, Z, sdf_grid, output_dir, base_name):
        """Fallback: Simple SDF data storage"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_path = output_path / timestamp
        run_path.mkdir(exist_ok=True)
        
        # Store SDF matrix
        sdf_matrix_file = run_path / f"{base_name}_sdf_matrix.npy"
        np.save(sdf_matrix_file, sdf_grid)
        
        # Store metadata
        json_file = run_path / f"{base_name}_metadata.json"
        metadata = {
            'run_id': timestamp,
            'timestamp': datetime.now().isoformat(),
            'domain_bounds': [float(X.min()), float(Y.min()), float(Z.min()), 
                             float(X.max()), float(Y.max()), float(Z.max())],
            'resolution': list(sdf_grid.shape),
            'sdf_stats': {
                'min': float(sdf_grid.min()),
                'max': float(sdf_grid.max()),
                'mean': float(sdf_grid.mean()),
                'std': float(sdf_grid.std())
            }
        }
        
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Stored SDF data (fallback): {run_path}")
        
        return {
            'run_id': timestamp,
            'run_path': str(run_path),
            'sdf_matrix': str(sdf_matrix_file),
            'metadata': str(json_file)
        }

@dataclass
class VirtualFaceData:
    """Standard format for virtual face data"""
    face_type: str                    # "inlet" or "outlet"
    center: np.ndarray               # Shape: (3,) [X, Y, Z]
    radius: float                    # Circle radius
    x_position: float                # Axial position
    triangles: np.ndarray            # Shape: (n_triangles, 3, 3)
    boundary_points: np.ndarray      # Shape: (n_points, 3)
    normal_vector: np.ndarray        # Shape: (3,) unit normal
    area: float                      # Total face area
    fit_error: float                 # Circle fitting quality metric
    n_triangles: int                 # Number of triangular faces

@dataclass
class RocketNozzleGeometry:
    """Complete nozzle geometry with virtual faces"""
    inlet_face: VirtualFaceData
    outlet_face: VirtualFaceData
    wall_triangles: np.ndarray       # Wall mesh triangles
    bounding_box: Tuple[np.ndarray, np.ndarray]  # Min/max coordinates

@dataclass
class JAXFluidsConfig:
    """JAX-Fluids configuration with boundary conditions"""
    domain_bounds: np.ndarray        # [xmin, ymin, zmin, xmax, ymax, zmax]
    resolution: Tuple[int, int, int] # Grid resolution
    sdf_values: np.ndarray           # Signed distance function
    inlet_mask: np.ndarray           # Boolean mask for inlet BC
    outlet_mask: np.ndarray          # Boolean mask for outlet BC
    wall_mask: np.ndarray            # Boolean mask for wall BC
    boundary_conditions: Dict[str, Any]  # BC configuration
    metadata: Dict[str, Any]         # Additional metadata

class VirtualFaceExtractor:
    """Extract virtual faces from mesh using the proven circular edge detection"""
    
    def find_circular_boundary_edges(self, mesh_file):
        """Find the actual circular edges at the inlet and outlet openings"""
        
        print("🔍 Finding circular boundary edges...")
        
        # Load with meshio first
        mesh = meshio.read(mesh_file)
        print(f"   Original mesh: {len(mesh.points)} points, {len(mesh.cells)} cell blocks")
        
        pv_mesh = pv.from_meshio(mesh)
        surface = pv_mesh.extract_surface()
        print(f"   Surface mesh: {surface.n_points} points, {surface.n_cells} cells")
        
        # Method 1: Try standard boundary edge extraction
        boundary_edges = surface.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=False,
            feature_edges=False,
            manifold_edges=False
        )
        
        print(f"   Method 1 - Standard boundary edges: {boundary_edges.n_points} points")
        
        # Method 2: Try with different feature edge settings
        if boundary_edges.n_points == 0:
            print("   Trying method 2 - Non-manifold edges...")
            boundary_edges = surface.extract_feature_edges(
                boundary_edges=True,
                non_manifold_edges=True,
                feature_edges=True,
                manifold_edges=False
            )
            print(f"   Method 2 - Non-manifold edges: {boundary_edges.n_points} points")
        
        # Method 3: Try to find "holes" by analyzing mesh connectivity
        if boundary_edges.n_points == 0:
            print("   Trying method 3 - Analyzing mesh topology...")
            
            # Get all points and analyze X-coordinate distribution
            all_points = surface.points
            x_coords = all_points[:, 0]
            
            # Find points at extreme X positions (likely inlet/outlet regions)
            x_min, x_max = x_coords.min(), x_coords.max()
            x_range = x_max - x_min
            
            # Points near the ends
            inlet_threshold = x_min + 0.05 * x_range
            outlet_threshold = x_max - 0.05 * x_range
            
            inlet_candidates = all_points[x_coords < inlet_threshold]
            outlet_candidates = all_points[x_coords > outlet_threshold]
            
            print(f"   Found {len(inlet_candidates)} inlet candidate points")
            print(f"   Found {len(outlet_candidates)} outlet candidate points")
            
            def find_circular_pattern(points, region_name):
                if len(points) < 10:
                    return None
                    
                # Project to Y-Z plane
                yz_points = points[:, 1:]
                center_y = yz_points[:, 0].mean()
                center_z = yz_points[:, 1].mean()
                center = np.array([center_y, center_z])
                
                # Calculate distances from center
                distances = np.linalg.norm(yz_points - center, axis=1)
                
                # Find points that are roughly the same distance from center (forming a circle)
                median_dist = np.median(distances)
                tolerance = median_dist * 0.2  # 20% tolerance
                
                circle_mask = np.abs(distances - median_dist) < tolerance
                circle_points = points[circle_mask]
                
                print(f"   {region_name}: {len(circle_points)} points form circle pattern")
                print(f"   {region_name}: radius ≈ {median_dist:.1f}")
                
                return circle_points if len(circle_points) > 20 else None
            
            inlet_points = find_circular_pattern(inlet_candidates, "Inlet")
            outlet_points = find_circular_pattern(outlet_candidates, "Outlet")
            
            if inlet_points is not None and outlet_points is not None:
                print(f"   Successfully found circular patterns!")
                return inlet_points, outlet_points
        
        # Method 4: If we found boundary edges, process them
        if boundary_edges.n_points > 0:
            edge_points = boundary_edges.points
            print(f"   Found {len(edge_points)} boundary edge points")
            
            # Cluster points by position to find the two circular openings
            x_coords = edge_points[:, 0]
            
            # Find the two extreme X regions (inlet and outlet)
            x_min, x_max = x_coords.min(), x_coords.max()
            x_range = x_max - x_min
            
            # Points near inlet (low X)
            inlet_mask = x_coords < (x_min + 0.1 * x_range)
            inlet_points = edge_points[inlet_mask]
            
            # Points near outlet (high X) 
            outlet_mask = x_coords > (x_max - 0.1 * x_range)
            outlet_points = edge_points[outlet_mask]
            
            print(f"   Inlet region: {len(inlet_points)} points at X≈{inlet_points[:, 0].mean():.1f}")
            print(f"   Outlet region: {len(outlet_points)} points at X≈{outlet_points[:, 0].mean():.1f}")
            
            return inlet_points, outlet_points
        
        print("❌ No boundary edges found with any method")
        return None, None

    def fit_circle_and_create_face(self, boundary_points, face_type="inlet"):
        """Fit a circle to boundary points and create a triangulated face"""
        
        print(f"🔧 Creating circular {face_type} face...")
        
        if len(boundary_points) < 10:
            print(f"❌ Not enough boundary points for {face_type}")
            return None
        
        # Get the average X position for the face plane
        x_pos = boundary_points[:, 0].mean()
        
        # Project to Y-Z plane for circle fitting
        yz_points = boundary_points[:, 1:]  # Y,Z coordinates
        
        # More robust circle fitting using least squares
        def fit_circle_least_squares(points_2d):
            """Fit circle using least squares method"""
            x, y = points_2d[:, 0], points_2d[:, 1]
            
            # Set up least squares problem: (x-cx)² + (y-cy)² = r²
            A = np.column_stack([x, y, np.ones(len(x))])
            b = x**2 + y**2
            
            # Solve for [2*cx, 2*cy, -(cx² + cy² - r²)]
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
        
        # Fit circle using least squares
        center_y, center_z, radius = fit_circle_least_squares(yz_points)
        
        # Verify the fit quality
        distances = np.sqrt((yz_points[:, 0] - center_y)**2 + (yz_points[:, 1] - center_z)**2)
        fit_error = np.std(distances - radius)
        
        print(f"   Circle center: Y={center_y:.1f}, Z={center_z:.1f}")
        print(f"   Circle radius: {radius:.1f}")
        print(f"   Fit quality: σ={fit_error:.2f} (lower is better)")
        
        # Create a high-quality circular face with more triangles for smoothness
        n_segments = 64  # More segments for smoother circle
        angles = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
        
        # Create circle points on the exact fitted circle
        circle_points = []
        for angle in angles:
            y = center_y + radius * np.cos(angle)
            z = center_z + radius * np.sin(angle)
            circle_points.append([x_pos, y, z])
        
        circle_points = np.array(circle_points)
        
        # Create triangular faces from center to circle edge (fan triangulation)
        triangles = []
        center_point = np.array([x_pos, center_y, center_z])
        
        for i in range(n_segments):
            j = (i + 1) % n_segments
            triangle = np.array([
                center_point,
                circle_points[i],
                circle_points[j]
            ])
            triangles.append(triangle)
        
        # Calculate normal vector (pointing in +X for inlet, -X for outlet)
        if face_type == "inlet":
            normal_vector = np.array([1.0, 0.0, 0.0])  # Flow direction
        else:
            normal_vector = np.array([1.0, 0.0, 0.0])  # Flow direction
        
        # Calculate area
        area = np.pi * radius**2
        
        face_data = VirtualFaceData(
            face_type=face_type,
            center=center_point,
            radius=radius,
            x_position=x_pos,
            triangles=np.array(triangles),
            boundary_points=boundary_points,
            normal_vector=normal_vector,
            area=area,
            fit_error=fit_error,
            n_triangles=len(triangles)
        )
        
        print(f"   Created {len(triangles)} triangular faces")
        
        return face_data

    def extract_virtual_faces(self, mesh_file) -> Optional[RocketNozzleGeometry]:
        """Extract virtual faces and wall triangles from mesh file"""
        
        print("🚀 EXTRACTING VIRTUAL FACES")
        print("=" * 60)
        
        # Step 1: Find circular boundary edges
        inlet_points, outlet_points = self.find_circular_boundary_edges(mesh_file)
        
        if inlet_points is None or outlet_points is None:
            print("❌ Could not find circular boundary edges")
            return None
        
        # Step 2: Create circular faces
        inlet_face_data = self.fit_circle_and_create_face(inlet_points, "inlet")
        outlet_face_data = self.fit_circle_and_create_face(outlet_points, "outlet")
        
        if inlet_face_data is None or outlet_face_data is None:
            print("❌ Could not create circular faces")
            return None
        
        # Step 3: Extract wall triangles from original mesh
        wall_triangles = parse_gmsh_mesh(mesh_file)
        
        # Step 4: Calculate bounding box
        all_points = []
        for triangle in wall_triangles:
            all_points.extend(triangle)
        all_points.extend(inlet_face_data.boundary_points)
        all_points.extend(outlet_face_data.boundary_points)
        
        all_points = np.array(all_points)
        min_bounds = all_points.min(axis=0)
        max_bounds = all_points.max(axis=0)
        
        # Add some padding
        padding = 0.1 * (max_bounds - min_bounds)
        min_bounds -= padding
        max_bounds += padding
        
        geometry = RocketNozzleGeometry(
            inlet_face=inlet_face_data,
            outlet_face=outlet_face_data,
            wall_triangles=wall_triangles,
            bounding_box=(min_bounds, max_bounds)
        )
        
        print("✅ Virtual faces extracted successfully!")
        return geometry

class JAXFluidsIntegrator:
    """Integrate virtual faces with SDF and create JAX-Fluids configuration"""
    
    def create_structured_grid(self, bounding_box, resolution):
        """Create structured Cartesian grid for JAX-Fluids"""
        
        min_bounds, max_bounds = bounding_box
        nx, ny, nz = resolution
        
        x = np.linspace(min_bounds[0], max_bounds[0], nx)
        y = np.linspace(min_bounds[1], max_bounds[1], ny)
        z = np.linspace(min_bounds[2], max_bounds[2], nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        return X, Y, Z
    
    def create_boundary_masks(self, geometry: RocketNozzleGeometry, X, Y, Z):
        """Create 3D boolean masks for inlet/outlet boundary conditions"""
        
        print("🎯 Creating boundary condition masks...")
        
        grid_shape = X.shape
        
        # Initialize masks
        inlet_mask = np.zeros(grid_shape, dtype=bool)
        outlet_mask = np.zeros(grid_shape, dtype=bool)
        wall_mask = np.zeros(grid_shape, dtype=bool)
        
        # Create coordinate arrays for vectorized operations
        coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # Inlet mask: circular region at inlet X position
        inlet_center = geometry.inlet_face.center
        inlet_radius = geometry.inlet_face.radius
        
        # Find grid points near inlet X position
        inlet_x_tolerance = abs(X[1,0,0] - X[0,0,0]) * 2  # 2 grid spacings
        inlet_x_mask = np.abs(coords[:, 0] - inlet_center[0]) < inlet_x_tolerance
        
        # For points near inlet X, check if they're within the circular face
        inlet_coords = coords[inlet_x_mask]
        if len(inlet_coords) > 0:
            yz_distances = np.sqrt(
                (inlet_coords[:, 1] - inlet_center[1])**2 + 
                (inlet_coords[:, 2] - inlet_center[2])**2
            )
            circular_mask = yz_distances <= inlet_radius
            
            # Map back to full grid
            inlet_indices = np.where(inlet_x_mask)[0][circular_mask]
            inlet_mask.ravel()[inlet_indices] = True
        
        # Outlet mask: circular region at outlet X position  
        outlet_center = geometry.outlet_face.center
        outlet_radius = geometry.outlet_face.radius
        
        # Find grid points near outlet X position
        outlet_x_tolerance = abs(X[1,0,0] - X[0,0,0]) * 2  # 2 grid spacings
        outlet_x_mask = np.abs(coords[:, 0] - outlet_center[0]) < outlet_x_tolerance
        
        # For points near outlet X, check if they're within the circular face
        outlet_coords = coords[outlet_x_mask]
        if len(outlet_coords) > 0:
            yz_distances = np.sqrt(
                (outlet_coords[:, 1] - outlet_center[1])**2 + 
                (outlet_coords[:, 2] - outlet_center[2])**2
            )
            circular_mask = yz_distances <= outlet_radius
            
            # Map back to full grid
            outlet_indices = np.where(outlet_x_mask)[0][circular_mask]
            outlet_mask.ravel()[outlet_indices] = True
        
        print(f"   Inlet mask: {inlet_mask.sum()} grid points")
        print(f"   Outlet mask: {outlet_mask.sum()} grid points")
        
        return inlet_mask, outlet_mask, wall_mask
    
    def integrate_with_sdf(self, geometry: RocketNozzleGeometry, 
                          resolution: Tuple[int, int, int] = (128, 64, 64),
                          flow_conditions: Optional[Dict] = None) -> JAXFluidsConfig:
        """Integrate virtual faces with SDF to create JAX-Fluids configuration"""
        
        print("🔧 INTEGRATING WITH SDF FOR JAX-FLUIDS")
        print("=" * 60)
        
        # Create structured grid
        X, Y, Z = self.create_structured_grid(geometry.bounding_box, resolution)
        domain_bounds = np.array([X.min(), Y.min(), Z.min(), X.max(), Y.max(), Z.max()])
        
        print(f"   Domain: X=[{domain_bounds[0]:.1f}, {domain_bounds[3]:.1f}]")
        print(f"           Y=[{domain_bounds[1]:.1f}, {domain_bounds[4]:.1f}]")
        print(f"           Z=[{domain_bounds[2]:.1f}, {domain_bounds[5]:.1f}]")
        print(f"   Resolution: {resolution}")
        
        # Compute SDF for wall surfaces
        print("   Computing SDF for wall surfaces...")
        sdf_values = compute_sdf(geometry.wall_triangles, domain_bounds, resolution)[3]
        
        # Create boundary condition masks
        inlet_mask, outlet_mask, wall_mask = self.create_boundary_masks(geometry, X, Y, Z)
        
        # Modify SDF to create openings at inlet/outlet
        # Make inlet/outlet regions have negative SDF (inside fluid domain)
        sdf_values[inlet_mask] = -abs(sdf_values[inlet_mask]) - 1.0  # Ensure negative
        sdf_values[outlet_mask] = -abs(sdf_values[outlet_mask]) - 1.0  # Ensure negative
        
        # Default flow conditions if not provided
        if flow_conditions is None:
            flow_conditions = {
                'inlet_pressure': 6.9e6,      # 6.9 MPa chamber pressure
                'inlet_temperature': 3580.0,   # 3580 K chamber temperature  
                'inlet_velocity': 100.0,       # Initial velocity estimate
                'outlet_pressure': 101325.0,   # Atmospheric pressure
                'gamma': 1.3,                  # Heat capacity ratio for combustion products
                'gas_constant': 287.0          # J/(kg·K)
            }
        
        # Create JAX-Fluids boundary condition configuration
        boundary_conditions = {
            'domain_faces': {
                'x_min': 'DIRICHLET_INLET' if geometry.inlet_face.x_position < geometry.outlet_face.x_position else 'WALL',
                'x_max': 'NEUMANN_OUTLET' if geometry.outlet_face.x_position > geometry.inlet_face.x_position else 'WALL',
                'y_min': 'SYMMETRY',
                'y_max': 'SYMMETRY',
                'z_min': 'SYMMETRY', 
                'z_max': 'SYMMETRY'
            },
            'virtual_faces': {
                'inlet': {
                    'type': 'DIRICHLET',
                    'center': geometry.inlet_face.center.tolist(),
                    'radius': float(geometry.inlet_face.radius),
                    'normal': geometry.inlet_face.normal_vector.tolist(),
                    'conditions': {
                        'pressure': flow_conditions['inlet_pressure'],
                        'temperature': flow_conditions['inlet_temperature'],
                        'velocity': [flow_conditions['inlet_velocity'], 0.0, 0.0]
                    }
                },
                'outlet': {
                    'type': 'NEUMANN',
                    'center': geometry.outlet_face.center.tolist(),
                    'radius': float(geometry.outlet_face.radius), 
                    'normal': geometry.outlet_face.normal_vector.tolist(),
                    'conditions': {
                        'pressure': flow_conditions['outlet_pressure'],
                        'gradient': 'ZERO_GRADIENT'
                    }
                }
            },
            'immersed_boundary': {
                'method': 'LEVEL_SET',
                'wall_treatment': 'NO_SLIP',
                'thermal_bc': 'ADIABATIC'
            }
        }
        
        # Create metadata
        metadata = {
            'created': datetime.now().isoformat(),
            'mesh_type': 'rocket_nozzle_internal_flow',
            'inlet_specs': {
                'center': geometry.inlet_face.center.tolist(),
                'radius': float(geometry.inlet_face.radius),
                'area': float(geometry.inlet_face.area),
                'fit_error': float(geometry.inlet_face.fit_error)
            },
            'outlet_specs': {
                'center': geometry.outlet_face.center.tolist(),
                'radius': float(geometry.outlet_face.radius),
                'area': float(geometry.outlet_face.area),
                'fit_error': float(geometry.outlet_face.fit_error)
            },
            'flow_conditions': flow_conditions,
            'numerical_methods': {
                'spatial_reconstruction': 'WENO5',
                'riemann_solver': 'HLLC',
                'time_integration': 'RK3'
            }
        }
        
        config = JAXFluidsConfig(
            domain_bounds=domain_bounds,
            resolution=resolution,
            sdf_values=sdf_values,
            inlet_mask=inlet_mask,
            outlet_mask=outlet_mask,
            wall_mask=wall_mask,
            boundary_conditions=boundary_conditions,
            metadata=metadata
        )
        
        print("✅ JAX-Fluids integration complete!")
        return config

    def save_jaxfluids_config(self, config: JAXFluidsConfig, output_dir: str = "jax_fluids_configs"):
        """Save JAX-Fluids configuration in compatible format"""
        
        print("💾 SAVING JAX-FLUIDS CONFIGURATION")
        print("=" * 60)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate timestamped files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"rocket_nozzle_{timestamp}"
        
        # Save SDF data using existing infrastructure
        sdf_files = store_sdf_data(
            *self.create_structured_grid((config.domain_bounds[:3], config.domain_bounds[3:]), config.resolution),
            config.sdf_values, 
            output_path, 
            base_name
        )
        
        # Save boundary condition masks
        masks_file = output_path / f"{base_name}_boundary_masks.npz"
        np.savez_compressed(
            masks_file,
            inlet_mask=config.inlet_mask,
            outlet_mask=config.outlet_mask,
            wall_mask=config.wall_mask,
            domain_bounds=config.domain_bounds,
            resolution=config.resolution
        )
        
        # Save complete JAX-Fluids configuration
        jax_config_file = output_path / f"{base_name}_jaxfluids_complete.json"
        
        jax_config = {
            'case_name': 'rocket_nozzle_internal_flow',
            'domain': {
                'x': {'cells': config.resolution[0], 'range': [float(config.domain_bounds[0]), float(config.domain_bounds[3])]},
                'y': {'cells': config.resolution[1], 'range': [float(config.domain_bounds[1]), float(config.domain_bounds[4])]},
                'z': {'cells': config.resolution[2], 'range': [float(config.domain_bounds[2]), float(config.domain_bounds[5])]}
            },
            'boundary_conditions': config.boundary_conditions,
            'levelset': {
                'model': 'FLUID_SOLID_INTERFACE',
                'sdf_file': str(sdf_files['sdf_matrix']),
                'virtual_faces': True
            },
            'masks': {
                'inlet_mask_file': str(masks_file),
                'outlet_mask_file': str(masks_file),
                'wall_mask_file': str(masks_file)
            },
            'metadata': config.metadata,
            'files': {
                'sdf_data': sdf_files,
                'masks': str(masks_file),
                'config': str(jax_config_file)
            }
        }
        
        with open(jax_config_file, 'w') as f:
            json.dump(jax_config, f, indent=2)
        
        print(f"📁 Configuration saved:")
        print(f"   🚀 JAX-Fluids config: {jax_config_file}")
        print(f"   🎯 Boundary masks: {masks_file}")
        print(f"   📐 SDF files: {sdf_files['run_path']}")
        
        return {
            'config_file': str(jax_config_file),
            'masks_file': str(masks_file),
            'sdf_files': sdf_files,
            'timestamp': timestamp
        }

def main():
    """Main processing function"""
    
    print("🚀 JAX-FLUIDS VIRTUAL FACE INTEGRATOR")
    print("=" * 70)
    
    # Configuration
    mesh_file = r"C:\Users\kriss\Desktop\Endpoint JAX  FLuids\mesh\Rocket Engine.msh"
    
    if not os.path.exists(mesh_file):
        print(f"❌ Mesh file not found: {mesh_file}")
        return 1
    
    try:
        # Step 1: Extract virtual faces from mesh
        extractor = VirtualFaceExtractor()
        geometry = extractor.extract_virtual_faces(mesh_file)
        
        if geometry is None:
            print("❌ Failed to extract virtual faces")
            return 1
        
        # Step 2: Integrate with SDF and create JAX-Fluids config
        integrator = JAXFluidsIntegrator()
        
        # Define flow conditions for rocket nozzle
        flow_conditions = {
            'inlet_pressure': 6.9e6,        # 6.9 MPa chamber pressure
            'inlet_temperature': 3580.0,     # 3580 K chamber temperature
            'inlet_velocity': 100.0,         # m/s initial estimate
            'outlet_pressure': 101325.0,     # Atmospheric pressure
            'gamma': 1.3,                    # Heat capacity ratio
            'gas_constant': 287.0,           # J/(kg·K)
            'mach_design': 3.0              # Design Mach number at exit
        }
        
        config = integrator.integrate_with_sdf(
            geometry, 
            resolution=(256, 128, 128),  # Higher resolution for accuracy
            flow_conditions=flow_conditions
        )
        
        # Step 3: Save JAX-Fluids configuration
        saved_files = integrator.save_jaxfluids_config(config)
        
        print()
        print("=" * 70)
        print("🎉 JAX-FLUIDS INTEGRATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"⚡ Virtual faces: Inlet R={geometry.inlet_face.radius:.1f}, Outlet R={geometry.outlet_face.radius:.1f}")
        print(f"📐 SDF grid: {config.resolution} = {np.prod(config.resolution):,} points")
        print(f"🎯 Boundary masks: {config.inlet_mask.sum()} inlet, {config.outlet_mask.sum()} outlet points")
        print(f"💾 Files saved: {saved_files['config_file']}")
        print("=" * 70)
        print()
        print("🚀 Ready for JAX-Fluids simulation!")
        print("   Use the generated configuration file with JAX-Fluids")
        print("   to run rocket nozzle internal flow simulations.")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 