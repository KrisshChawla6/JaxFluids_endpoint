#!/usr/bin/env python3
"""
Generalized Wind Tunnel API - Main Entry Point
Accepts any mesh file and parameters to create wind tunnels with VTK output
"""

import os
import sys
import time
import logging
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wind_tunnel.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TunnelType(Enum):
    """Available tunnel configurations"""
    COMPACT = "compact"
    STANDARD = "standard" 
    RESEARCH = "research"
    AUTOMOTIVE = "automotive"
    AEROSPACE = "aerospace"

class FlowDirection(Enum):
    """Flow direction options"""
    POSITIVE_X = "+X"
    NEGATIVE_X = "-X"
    POSITIVE_Y = "+Y"
    NEGATIVE_Y = "-Y"
    POSITIVE_Z = "+Z"
    NEGATIVE_Z = "-Z"

@dataclass
class WindTunnelRequest:
    """Wind tunnel generation request"""
    object_mesh_file: str
    tunnel_type: str = "standard"
    flow_direction: str = "+X"
    output_file: Optional[str] = None
    output_directory: str = "output"
    mesh_quality: str = "medium"  # coarse, medium, fine
    custom_scaling: Optional[Dict] = None
    generate_vtk: bool = False
    flow_velocity: float = 10.0  # m/s
    domain_scale_factor: float = 1.0  # Scale factor for fixed sizing (1.0 = original proven dimensions)

@dataclass
class WindTunnelResponse:
    """Wind tunnel generation response"""
    success: bool
    message: str
    output_file: Optional[str] = None
    vtk_file: Optional[str] = None
    mesh_stats: Optional[Dict] = None
    generation_time: Optional[float] = None
    domain_info: Optional[Dict] = None
    error_details: Optional[str] = None

class GeneralizedWindTunnelGenerator:
    """Generalized wind tunnel generator that accepts any mesh file"""
    
    def __init__(self):
        self.logger = logger
        self.temp_dir = Path("tmp")
        self.output_dir = Path("output")
        
        # Create directories
        for directory in [self.temp_dir, self.output_dir]:
            directory.mkdir(exist_ok=True)

    def generate_wind_tunnel(self, request: WindTunnelRequest) -> WindTunnelResponse:
        """
        Main entry point for wind tunnel generation
        
        Args:
            request: WindTunnelRequest with all parameters
            
        Returns:
            WindTunnelResponse with results
        """
        
        start_time = time.time()
        
        try:
            print("🌪️ GENERALIZED WIND TUNNEL GENERATOR")
            print("=" * 50)
            print(f"📁 Input mesh: {request.object_mesh_file}")
            print(f"🏗️ Tunnel type: {request.tunnel_type}")
            print(f"➡️ Flow direction: {request.flow_direction}")
            print(f"⚡ Flow velocity: {request.flow_velocity} m/s")
            print(f"📏 Domain scale: {request.domain_scale_factor}x")
            
            # Validate input file
            if not os.path.exists(request.object_mesh_file):
                return WindTunnelResponse(
                    success=False,
                    message=f"Input mesh file not found: {request.object_mesh_file}",
                    generation_time=time.time() - start_time
                )
            
            # Set output file if not provided
            if request.output_file is None:
                base_name = Path(request.object_mesh_file).stem
                request.output_file = str(self.output_dir / f"{base_name}_wind_tunnel.su2")
            
            # Step 1: Read and analyze object mesh
            print("\n📖 STEP 1: Reading object mesh...")
            object_data = self.read_object_mesh(request.object_mesh_file)
            
            # Step 2: Create wind tunnel domain
            print("\n🏗️ STEP 2: Creating wind tunnel domain...")
            domain_data = self.create_wind_tunnel_domain(object_data, request)
            
            # Step 3: Extract object surface boundaries
            print("\n🔍 STEP 3: Extracting object surface boundaries...")
            object_boundaries = self.extract_object_surface_boundaries(object_data)
            
            # Step 4: Write SU2 mesh
            print("\n💾 STEP 4: Writing SU2 mesh...")
            self.write_su2_mesh_with_object_boundaries(object_data, domain_data, object_boundaries, request.output_file, request)
            
            # Step 5: Generate VTK if requested
            vtk_file = None
            if request.generate_vtk:
                print("\n🎨 STEP 5: Converting to VTK...")
                vtk_file = self.convert_to_vtk(request.output_file)
            
            # Step 6: Validate output
            print("\n✅ STEP 6: Validating output...")
            validation_result = self.validate_mesh(request.output_file)
            
            generation_time = time.time() - start_time
            file_size = os.path.getsize(request.output_file) / (1024 * 1024)
            
            mesh_stats = {
                'total_nodes': len(object_data['nodes']) + len(domain_data['nodes']),
                'total_elements': len(object_data['elements']) + len(domain_data['elements']),
                'object_nodes': len(object_data['nodes']),
                'domain_nodes': len(domain_data['nodes']),
                'file_size_mb': file_size,
                'boundaries': list(domain_data['boundaries'].keys())
            }
            
            print(f"\n🎉 Wind tunnel generated successfully!")
            print(f"   📁 SU2 file: {request.output_file}")
            if vtk_file:
                print(f"   🎨 VTK file: {vtk_file}")
            print(f"   ⏱️ Time: {generation_time:.2f}s")
            print(f"   📊 Size: {file_size:.1f} MB")
            print(f"   🔢 Nodes: {mesh_stats['total_nodes']:,}")
            print(f"   🔷 Elements: {mesh_stats['total_elements']:,}")
            
            return WindTunnelResponse(
                success=True,
                message="Wind tunnel generated successfully",
                output_file=request.output_file,
                vtk_file=vtk_file,
                mesh_stats=mesh_stats,
                generation_time=generation_time,
                domain_info=domain_data['info']
            )
            
        except Exception as e:
            error_msg = f"Wind tunnel generation failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            return WindTunnelResponse(
                success=False,
                message=error_msg,
                error_details=traceback.format_exc(),
                generation_time=time.time() - start_time
            )

    def read_object_mesh(self, mesh_file: str) -> Dict:
        """Read object mesh from any supported format"""
        
        print(f"   📖 Reading: {os.path.basename(mesh_file)}")
        
        # Determine file format
        file_ext = Path(mesh_file).suffix.lower()
        
        if file_ext == '.su2':
            return self.read_su2_mesh(mesh_file)
        else:
            raise ValueError(f"Unsupported mesh format: {file_ext}")
    
    def read_su2_mesh(self, mesh_file: str) -> Dict:
        """Read SU2 mesh file"""
        
        with open(mesh_file, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Parse mesh components
        nodes = []
        elements = []
        boundaries = {}
        
        current_section = None
        current_boundary = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('NELEM='):
                current_section = 'elements'
                continue
            elif line.startswith('NPOIN='):
                current_section = 'nodes'
                continue
            elif line.startswith('NMARK='):
                current_section = 'boundaries'
                continue
            elif line.startswith('MARKER_TAG='):
                boundary_name = line.split('=')[1].strip()
                current_boundary = boundary_name
                boundaries[boundary_name] = []
                continue
            elif line.startswith('MARKER_ELEMS='):
                continue
            
            if current_section == 'elements' and line and not line.startswith('N'):
                parts = line.split()
                if len(parts) >= 5:  # Tetrahedral: type + 4 nodes + element_id
                    try:
                        element = [int(x) for x in parts]
                        elements.append(element)
                    except ValueError:
                        continue
                    
            elif current_section == 'nodes' and line and not line.startswith('N'):
                parts = line.split()
                if len(parts) >= 4:  # x, y, z, node_id
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        nodes.append([x, y, z])
                    except ValueError:
                        continue
                    
            elif current_section == 'boundaries' and current_boundary and line and not line.startswith('MARKER_'):
                parts = line.split()
                if len(parts) > 1:
                    boundaries[current_boundary].append([int(x) for x in parts])
        
        nodes_array = np.array(nodes)
        
        # Calculate bounding box
        bbox = {
            'min_x': nodes_array[:, 0].min(),
            'max_x': nodes_array[:, 0].max(),
            'min_y': nodes_array[:, 1].min(),
            'max_y': nodes_array[:, 1].max(),
            'min_z': nodes_array[:, 2].min(),
            'max_z': nodes_array[:, 2].max(),
        }
        
        # Calculate characteristic dimensions
        length = bbox['max_x'] - bbox['min_x']
        width = bbox['max_y'] - bbox['min_y']
        height = bbox['max_z'] - bbox['min_z']
        
        print(f"   📊 Object: {len(nodes)} nodes, {len(elements)} elements")
        print(f"   📏 Dimensions: {length:.3f} x {width:.3f} x {height:.3f}")
        print(f"   🏷️ Boundaries: {list(boundaries.keys())}")
        
        return {
            'nodes': nodes_array,
            'elements': elements,
            'boundaries': boundaries,
            'bbox': bbox,
            'dimensions': {'length': length, 'width': width, 'height': height}
        }

    def create_wind_tunnel_domain(self, object_data: Dict, request: WindTunnelRequest) -> Dict:
        """Create wind tunnel domain around object"""
        
        bbox = object_data['bbox']
        scale = request.domain_scale_factor
        
        # Calculate domain dimensions based on flow direction and tunnel type
        domain_bounds = self.calculate_domain_bounds(bbox, request.flow_direction, 
                                                   request.tunnel_type, scale)
        
        # Create domain mesh
        domain_mesh = self.create_domain_mesh(domain_bounds, request.mesh_quality)
        
        # Create actual boundary conditions from tetrahedral mesh
        boundaries = self.create_boundary_conditions_from_mesh(
            domain_mesh['elements'], 
            domain_bounds, 
            request.flow_direction, 
            domain_mesh['grid_dims']
        )
        
        print(f"   📏 Domain: {domain_bounds['length']:.1f} x {domain_bounds['width']:.1f} x {domain_bounds['height']:.1f}")
        print(f"   🔢 Domain nodes: {len(domain_mesh['nodes']):,}")
        print(f"   🏷️ Boundaries: {list(boundaries.keys())}")
        
        return {
            'nodes': domain_mesh['nodes'],
            'elements': domain_mesh['elements'],
            'boundaries': boundaries,
            'bounds': domain_bounds,
            'mesh_size': domain_mesh['grid_dims'],
            'info': {
                'domain_type': request.tunnel_type,
                'flow_direction': request.flow_direction,
                'scale_factor': scale,
                'mesh_quality': request.mesh_quality
            }
        }

    def calculate_domain_bounds(self, object_bbox: Dict, flow_direction: str, 
                              tunnel_type: str, scale: float) -> Dict:
        """Calculate wind tunnel domain bounds using simplified scaling algorithm"""
        
        # Object dimensions
        obj_x = object_bbox['max_x'] - object_bbox['min_x']
        obj_y = object_bbox['max_y'] - object_bbox['min_y']
        obj_z = object_bbox['max_z'] - object_bbox['min_z']
        
        # Find the longest axis of the object
        longest_axis = max(obj_x, obj_y, obj_z)
        
        # Scaling algorithm:
        # - Cross-section axes: 1.75x total scaling based on longest axis
        # - Flow direction axis: 3.0x scaling based on longest axis
        cross_section_scaling = 1.75  # 1.75x total based on longest axis
        flow_scaling = 3.0  # 3.0x total based on longest axis
        
        # Apply scale factor
        cross_section_scaling *= scale
        flow_scaling *= scale
        
        # Calculate distances for cross-section axes (based on longest axis, distributed evenly)
        cross_section_dist = cross_section_scaling * longest_axis / 2  # Divide by 2 for each side
        
        # Override flow direction axis with longest-axis-based scaling
        if flow_direction in ['+X', '-X']:
            # Flow in X direction: use longest axis for X scaling
            flow_dist = flow_scaling * longest_axis
            upstream_dist = flow_dist * 0.4  # 40% upstream
            downstream_dist = flow_dist * 0.6  # 60% downstream
            
            if flow_direction == '+X':
                min_x = object_bbox['min_x'] - upstream_dist
                max_x = object_bbox['max_x'] + downstream_dist
            else:
                min_x = object_bbox['min_x'] - downstream_dist
                max_x = object_bbox['max_x'] + upstream_dist
                
            min_y = object_bbox['min_y'] - cross_section_dist
            max_y = object_bbox['max_y'] + cross_section_dist
            min_z = object_bbox['min_z'] - cross_section_dist
            max_z = object_bbox['max_z'] + cross_section_dist
            
        elif flow_direction in ['+Y', '-Y']:
            # Flow in Y direction: use longest axis for Y scaling
            flow_dist = flow_scaling * longest_axis
            upstream_dist = flow_dist * 0.4  # 40% upstream
            downstream_dist = flow_dist * 0.6  # 60% downstream
            
            if flow_direction == '+Y':
                min_y = object_bbox['min_y'] - upstream_dist
                max_y = object_bbox['max_y'] + downstream_dist
            else:
                min_y = object_bbox['min_y'] - downstream_dist
                max_y = object_bbox['max_y'] + upstream_dist
                
            min_x = object_bbox['min_x'] - cross_section_dist
            max_x = object_bbox['max_x'] + cross_section_dist
            min_z = object_bbox['min_z'] - cross_section_dist
            max_z = object_bbox['max_z'] + cross_section_dist
            
        else:  # Z direction
            # Flow in Z direction: use longest axis for Z scaling
            flow_dist = flow_scaling * longest_axis
            upstream_dist = flow_dist * 0.4  # 40% upstream
            downstream_dist = flow_dist * 0.6  # 60% downstream
            
            if flow_direction == '+Z':
                min_z = object_bbox['min_z'] - upstream_dist
                max_z = object_bbox['max_z'] + downstream_dist
            else:
                min_z = object_bbox['min_z'] - downstream_dist
                max_z = object_bbox['max_z'] + upstream_dist
                
            min_x = object_bbox['min_x'] - cross_section_dist
            max_x = object_bbox['max_x'] + cross_section_dist
            min_y = object_bbox['min_y'] - cross_section_dist
            max_y = object_bbox['max_y'] + cross_section_dist
        
        return {
            'min_x': min_x, 'max_x': max_x,
            'min_y': min_y, 'max_y': max_y,
            'min_z': min_z, 'max_z': max_z,
            'length': max_x - min_x,
            'width': max_y - min_y,
            'height': max_z - min_z
        }

    def create_domain_mesh(self, bounds: Dict, quality: str) -> Dict:
        """Create structured domain mesh with tetrahedral elements"""
        
        # Mesh density based on quality
        density_factors = {
            'coarse': 0.5,
            'medium': 1.0,
            'fine': 2.0
        }
        
        factor = density_factors.get(quality, 1.0)
        
        # Calculate grid dimensions (similar to improved generator)
        base_divisions = 20
        nx = int(base_divisions * factor)
        ny = int(base_divisions * factor)
        nz = int(base_divisions * factor)
        
        # Create structured grid
        x = np.linspace(bounds['min_x'], bounds['max_x'], nx)
        y = np.linspace(bounds['min_y'], bounds['max_y'], ny)
        z = np.linspace(bounds['min_z'], bounds['max_z'], nz)
        
        # Generate nodes
        nodes = []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    nodes.append([x[i], y[j], z[k]])
        
        # Generate tetrahedral elements (split each hex into 6 tets)
        elements = []
        
        for k in range(nz-1):
            for j in range(ny-1):
                for i in range(nx-1):
                    # Get 8 corner nodes of hexahedron
                    n0 = k * nx * ny + j * nx + i
                    n1 = k * nx * ny + j * nx + (i + 1)
                    n2 = k * nx * ny + (j + 1) * nx + (i + 1)
                    n3 = k * nx * ny + (j + 1) * nx + i
                    n4 = (k + 1) * nx * ny + j * nx + i
                    n5 = (k + 1) * nx * ny + j * nx + (i + 1)
                    n6 = (k + 1) * nx * ny + (j + 1) * nx + (i + 1)
                    n7 = (k + 1) * nx * ny + (j + 1) * nx + i
                    
                    # Split hexahedron into 6 tetrahedra (VTK type 10)
                    # Standard hex-to-tet decomposition
                    elements.extend([
                        [10, n0, n1, n3, n4],  # Tet 1
                        [10, n1, n2, n3, n6],  # Tet 2  
                        [10, n1, n3, n4, n6],  # Tet 3
                        [10, n3, n4, n6, n7],  # Tet 4
                        [10, n1, n4, n5, n6],  # Tet 5
                        [10, n4, n5, n6, n7]   # Tet 6
                    ])
        
        return {
            'nodes': np.array(nodes),
            'elements': elements,
            'grid_dims': (nx, ny, nz)
        }

    def create_boundary_conditions(self, bounds: Dict, flow_direction: str) -> Dict:
        """Create boundary condition markers - placeholder for compatibility"""
        
        # This method is called before domain mesh is created
        # Return empty boundaries that will be populated later
        boundaries = {
            'inlet': [],
            'outlet': [],
            'slip_wall': [],
            'object_wall': []
        }
        
        return boundaries
    
    def create_boundary_conditions_from_mesh(self, domain_elements: List, bounds: Dict, 
                                           flow_direction: str, mesh_size: Tuple) -> Dict:
        """Create actual boundary conditions from tetrahedral mesh elements"""
        from collections import defaultdict
        
        nx, ny, nz = mesh_size
        
        # Extract all faces from tetrahedra and count occurrences
        face_count = defaultdict(int)
        
        for elem in domain_elements:
            if len(elem) >= 5 and elem[0] == 10:  # Tetrahedron
                nodes = elem[1:5]
                
                # Generate the 4 faces of the tetrahedron
                faces = [
                    tuple(sorted([nodes[0], nodes[1], nodes[2]])),
                    tuple(sorted([nodes[0], nodes[1], nodes[3]])),
                    tuple(sorted([nodes[0], nodes[2], nodes[3]])),
                    tuple(sorted([nodes[1], nodes[2], nodes[3]])),
                ]
                
                for face in faces:
                    face_count[face] += 1
        
        # Boundary faces appear only once (not shared between elements)
        boundary_faces = [face for face, count in face_count.items() if count == 1]
        
        print(f"   🔍 Found {len(boundary_faces)} boundary faces from tetrahedral elements")
        
        # Classify boundary faces based on node positions
        boundaries = {
            'inlet': [],
            'outlet': [],
            'slip_wall': []
        }
        
        # Get node position from structured grid index
        def get_node_position(node_idx):
            k = node_idx // (nx * ny)
            j = (node_idx % (nx * ny)) // nx
            i = node_idx % nx
            
            x = bounds['min_x'] + i * (bounds['max_x'] - bounds['min_x']) / (nx - 1)
            y = bounds['min_y'] + j * (bounds['max_y'] - bounds['min_y']) / (ny - 1)
            z = bounds['min_z'] + k * (bounds['max_z'] - bounds['min_z']) / (nz - 1)
            
            return x, y, z
        
        # Classify each boundary face
        tolerance = 1e-6
        
        for face in boundary_faces:
            # Get positions of the 3 nodes
            positions = [get_node_position(node) for node in face]
            
            # Calculate face center
            center_x = sum(pos[0] for pos in positions) / 3
            center_y = sum(pos[1] for pos in positions) / 3
            center_z = sum(pos[2] for pos in positions) / 3
            
            # Classify based on position and flow direction
            if flow_direction == '+X':
                if abs(center_x - bounds['min_x']) < tolerance:
                    boundaries['inlet'].append([5, face[0], face[1], face[2]])
                elif abs(center_x - bounds['max_x']) < tolerance:
                    boundaries['outlet'].append([5, face[0], face[1], face[2]])
                else:
                    boundaries['slip_wall'].append([5, face[0], face[1], face[2]])
            elif flow_direction == '-X':
                if abs(center_x - bounds['max_x']) < tolerance:
                    boundaries['inlet'].append([5, face[0], face[1], face[2]])
                elif abs(center_x - bounds['min_x']) < tolerance:
                    boundaries['outlet'].append([5, face[0], face[1], face[2]])
                else:
                    boundaries['slip_wall'].append([5, face[0], face[1], face[2]])
            # Add other flow directions as needed
            else:
                # Default to +X behavior
                if abs(center_x - bounds['min_x']) < tolerance:
                    boundaries['inlet'].append([5, face[0], face[1], face[2]])
                elif abs(center_x - bounds['max_x']) < tolerance:
                    boundaries['outlet'].append([5, face[0], face[1], face[2]])
                else:
                    boundaries['slip_wall'].append([5, face[0], face[1], face[2]])
        
        print(f"   🏷️  Domain boundaries created:")
        for name, elems in boundaries.items():
            print(f"      {name}: {len(elems)} faces")
        
        return boundaries
    
    def extract_object_surface_boundaries(self, object_data: Dict) -> List:
        """Extract surface boundaries from object mesh"""
        
        print("   🔍 Extracting object surface boundaries...")
        
        object_boundaries = []
        elements = object_data['elements']
        
        # For tetrahedral elements, extract surface triangles
        # Each tet face that appears only once is on the surface
        face_count = {}
        
        for elem_idx, elem in enumerate(elements):
            if len(elem) >= 5 and elem[0] == 10:  # Tetrahedron
                # Get the 4 nodes (skip element type)
                nodes = elem[1:5]
                
                # Generate 4 triangular faces of the tetrahedron
                faces = [
                    tuple(sorted([nodes[0], nodes[1], nodes[2]])),
                    tuple(sorted([nodes[0], nodes[1], nodes[3]])),
                    tuple(sorted([nodes[0], nodes[2], nodes[3]])),
                    tuple(sorted([nodes[1], nodes[2], nodes[3]]))
                ]
                
                for face in faces:
                    if face in face_count:
                        face_count[face] += 1
                    else:
                        face_count[face] = 1
        
        # Surface faces appear exactly once (not shared with another element)
        surface_faces = [face for face, count in face_count.items() if count == 1]
        
        # Convert to SU2 boundary format (triangle elements)
        for face in surface_faces:
            # Triangle boundary element: [5, node1, node2, node3]
            object_boundaries.append([5, face[0], face[1], face[2]])
        
        print(f"   📊 Extracted {len(object_boundaries)} surface boundary faces")
        
        return object_boundaries
    
    def create_face_elements(self, bounds: Dict, face: str) -> List:
        """Create boundary elements for a domain face"""
        # This method should be called after domain mesh is created
        # For now, return empty list - actual boundary creation happens in create_boundary_conditions_from_mesh
        return []

    def write_su2_mesh_with_object_boundaries(self, object_data: Dict, domain_data: Dict, 
                                            object_boundaries: List, output_file: str, request: WindTunnelRequest):
        """Write complete SU2 mesh file with proper object boundaries"""
        
        print("   💾 Writing SU2 mesh with object boundaries...")
        
        with open(output_file, 'w') as f:
            # Problem dimension
            f.write("NDIME=3\n")
            
            # Elements (domain + object)
            total_elements = len(domain_data['elements']) + len(object_data['elements'])
            f.write(f"NELEM={total_elements}\n")
            
            # Write domain elements first (no element indices needed)
            for elem in domain_data['elements']:
                f.write(" ".join(map(str, elem)) + "\n")
            
            # Write object elements with offset node indices (no element indices needed)
            node_offset = len(domain_data['nodes'])
            
            for elem in object_data['elements']:
                if len(elem) >= 4:
                    # Offset node indices for object elements
                    elem_with_offset = [elem[0]]  # Keep element type
                    for j in range(1, len(elem)):  # Offset all node indices
                        elem_with_offset.append(elem[j] + node_offset)
                    f.write(" ".join(map(str, elem_with_offset)) + "\n")
            
            f.write("\n")
            
            # Nodes (domain + object)
            total_nodes = len(domain_data['nodes']) + len(object_data['nodes'])
            f.write(f"NPOIN={total_nodes}\n")
            
            # Write domain nodes with node indices
            node_index = 0
            for node in domain_data['nodes']:
                f.write(f"{node[0]:.6f} {node[1]:.6f} {node[2]:.6f} {node_index}\n")
                node_index += 1
            
            # Write object nodes with node indices
            for node in object_data['nodes']:
                f.write(f"{node[0]:.6f} {node[1]:.6f} {node[2]:.6f} {node_index}\n")
                node_index += 1
            
            f.write("\n")
            
            # Boundaries (domain + object)
            total_boundaries = len(domain_data['boundaries']) + 1  # +1 for object_wall
            f.write(f"NMARK={total_boundaries}\n")
            
            # Write domain boundaries
            for boundary_name, boundary_elems in domain_data['boundaries'].items():
                f.write(f"MARKER_TAG={boundary_name}\n")
                f.write(f"MARKER_ELEMS={len(boundary_elems)}\n")
                
                for elem in boundary_elems:
                    f.write(" ".join(map(str, elem)) + "\n")
            
            # Write object boundary (offset node indices)
            f.write("MARKER_TAG=object_wall\n")
            f.write(f"MARKER_ELEMS={len(object_boundaries)}\n")
            
            for elem in object_boundaries:
                # Offset node indices for object boundary elements
                elem_with_offset = [elem[0]]  # Keep element type (5 for triangle)
                for j in range(1, len(elem)):
                    elem_with_offset.append(elem[j] + node_offset)
                f.write(" ".join(map(str, elem_with_offset)) + "\n")
        
        print(f"   ✅ Mesh written with {len(object_boundaries)} object boundary elements!")

    def write_su2_mesh(self, object_data: Dict, domain_data: Dict, 
                      output_file: str, request: WindTunnelRequest):
        """Write complete SU2 mesh file (legacy method)"""
        
        with open(output_file, 'w') as f:
            # Problem dimension
            f.write("NDIME=3\n")
            f.write("\n")
            
            # Elements
            total_elements = len(object_data['elements']) + len(domain_data['elements'])
            f.write(f"NELEM={total_elements}\n")
            
            # Write object elements first
            for i, elem in enumerate(object_data['elements']):
                f.write(" ".join(map(str, elem)) + "\n")
            
            # Write domain elements
            node_offset = len(object_data['nodes'])
            for i, elem in enumerate(domain_data['elements']):
                # Adjust node indices for domain elements
                adjusted_elem = [elem[0]]  # Element type
                for j in range(1, len(elem)-1):  # Node indices
                    adjusted_elem.append(elem[j] + node_offset)
                adjusted_elem.append(elem[-1] + len(object_data['elements']))  # Element ID
                f.write(" ".join(map(str, adjusted_elem)) + "\n")
            
            f.write("\n")
            
            # Nodes
            total_nodes = len(object_data['nodes']) + len(domain_data['nodes'])
            f.write(f"NPOIN={total_nodes}\n")
            
            # Write object nodes
            for i, node in enumerate(object_data['nodes']):
                f.write(f"{node[0]:.6f} {node[1]:.6f} {node[2]:.6f} {i}\n")
            
            # Write domain nodes
            for i, node in enumerate(domain_data['nodes']):
                node_id = i + len(object_data['nodes'])
                f.write(f"{node[0]:.6f} {node[1]:.6f} {node[2]:.6f} {node_id}\n")
            
            f.write("\n")
            
            # Boundaries
            boundaries = domain_data['boundaries']
            f.write(f"NMARK={len(boundaries)}\n")
            
            for boundary_name, boundary_elements in boundaries.items():
                f.write(f"MARKER_TAG={boundary_name}\n")
                f.write(f"MARKER_ELEMS={len(boundary_elements)}\n")
                for elem in boundary_elements:
                    f.write(" ".join(map(str, elem)) + "\n")

    def convert_to_vtk(self, su2_file: str) -> Optional[str]:
        """Convert SU2 mesh to VTK format"""
        
        vtk_file = su2_file.replace('.su2', '.vtk')
        
        try:
            # Use the simple SU2 to VTK converter
            return self.simple_su2_to_vtk(su2_file, vtk_file)
        except Exception as e:
            print(f"   ⚠️ VTK conversion failed: {e}")
            return None

    def simple_su2_to_vtk(self, su2_file: str, vtk_file: str) -> str:
        """Simple SU2 to VTK conversion"""
        
        print(f"   🎨 Converting {os.path.basename(su2_file)} to VTK...")
        
        # Read SU2 file
        with open(su2_file, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Parse nodes and elements
        nodes = []
        elements = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('NPOIN='):
                n_points = int(line.split('=')[1])
                i += 1
                
                for _ in range(n_points):
                    if i < len(lines):
                        parts = lines[i].strip().split()
                        if len(parts) >= 3:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            nodes.append([x, y, z])
                        i += 1
                        
            elif line.startswith('NELEM='):
                n_elem = int(line.split('=')[1])
                i += 1
                
                for _ in range(n_elem):
                    if i < len(lines):
                        parts = lines[i].strip().split()
                        if len(parts) > 1:
                            elements.append([int(x) for x in parts])
                        i += 1
            else:
                i += 1
        
        # Write VTK file
        with open(vtk_file, 'w') as f:
            # VTK header
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Wind Tunnel Mesh\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n\n")
            
            # Points
            f.write(f"POINTS {len(nodes)} float\n")
            for node in nodes:
                f.write(f"{node[0]:.6f} {node[1]:.6f} {node[2]:.6f}\n")
            f.write("\n")
            
            # Filter valid elements
            valid_elements = []
            for elem in elements:
                if len(elem) >= 5 and elem[0] in [10, 12]:  # Tetrahedron or Hexahedron
                    valid_elements.append(elem)
            
            if valid_elements:
                # Calculate total size
                total_size = sum(len(elem) - 1 for elem in valid_elements)  # -1 for element type
                
                f.write(f"CELLS {len(valid_elements)} {total_size}\n")
                for elem in valid_elements:
                    if elem[0] == 10:  # Tetrahedron
                        f.write(f"4 {elem[1]} {elem[2]} {elem[3]} {elem[4]}\n")
                    elif elem[0] == 12:  # Hexahedron
                        f.write(f"8 {elem[1]} {elem[2]} {elem[3]} {elem[4]} {elem[5]} {elem[6]} {elem[7]} {elem[8]}\n")
                f.write("\n")
                
                # Cell types
                f.write(f"CELL_TYPES {len(valid_elements)}\n")
                for elem in valid_elements:
                    f.write(f"{elem[0]}\n")
                f.write("\n")
        
        print(f"   ✅ VTK file created: {os.path.basename(vtk_file)}")
        return vtk_file

    def validate_mesh(self, mesh_file: str) -> bool:
        """Validate SU2 mesh format"""
        
        try:
            with open(mesh_file, 'r') as f:
                content = f.read()
            
            # Check for required sections
            required_sections = ['NDIME=', 'NELEM=', 'NPOIN=', 'NMARK=']
            for section in required_sections:
                if section not in content:
                    print(f"   ❌ Missing section: {section}")
                    return False
            
            print(f"   ✅ Mesh validation passed")
            return True
            
        except Exception as e:
            print(f"   ❌ Validation failed: {e}")
            return False

# Main API functions
def create_wind_tunnel(
    object_mesh_file: str,
    tunnel_type: str = "standard",
    flow_direction: str = "+X", 
    output_file: Optional[str] = None,
    output_directory: str = "output",
    mesh_quality: str = "medium",
    generate_vtk: bool = False,
    flow_velocity: float = 10.0,
    domain_scale_factor: float = 1.0
) -> WindTunnelResponse:
    """
    Create wind tunnel mesh from any object mesh file
    
    Args:
        object_mesh_file: Path to object mesh file (.su2)
        tunnel_type: Type of tunnel (compact, standard, research, automotive, aerospace)
        flow_direction: Flow direction (+X, -X, +Y, -Y, +Z, -Z)
        output_file: Output SU2 file path (optional)
        output_directory: Output directory
        mesh_quality: Mesh quality (coarse, medium, fine)
        generate_vtk: Whether to generate VTK file
        flow_velocity: Flow velocity in m/s
        domain_scale_factor: Domain scaling factor
        
    Returns:
        WindTunnelResponse with results
    """
    
    request = WindTunnelRequest(
        object_mesh_file=object_mesh_file,
        tunnel_type=tunnel_type,
        flow_direction=flow_direction,
        output_file=output_file,
        output_directory=output_directory,
        mesh_quality=mesh_quality,
        generate_vtk=generate_vtk,
        flow_velocity=flow_velocity,
        domain_scale_factor=domain_scale_factor
    )
    
    generator = GeneralizedWindTunnelGenerator()
    return generator.generate_wind_tunnel(request)

def get_available_configurations() -> Dict:
    """Get available tunnel configurations"""
    return {
        'tunnel_types': [e.value for e in TunnelType],
        'flow_directions': [e.value for e in FlowDirection],
        'mesh_qualities': ['coarse', 'medium', 'fine'],
        'supported_formats': ['.su2']
    }

def main():
    """Main function for command line usage"""
    
    if len(sys.argv) < 2:
        print("Usage: python wind_tunnel_api.py <object_mesh_file> [options]")
        print("\nOptions:")
        print("  --tunnel-type: compact, standard, research, automotive, aerospace")
        print("  --flow-direction: +X, -X, +Y, -Y, +Z, -Z")
        print("  --mesh-quality: coarse, medium, fine")
        print("  --output: output file path")
        print("  --no-vtk: disable VTK generation")
        return
    
    object_mesh_file = sys.argv[1]
    
    # Parse command line arguments
    tunnel_type = "standard"
    flow_direction = "+X"
    mesh_quality = "medium"
    output_file = None
    generate_vtk = True
    
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--tunnel-type" and i+1 < len(sys.argv):
            tunnel_type = sys.argv[i+1]
        elif arg == "--flow-direction" and i+1 < len(sys.argv):
            flow_direction = sys.argv[i+1]
        elif arg == "--mesh-quality" and i+1 < len(sys.argv):
            mesh_quality = sys.argv[i+1]
        elif arg == "--output" and i+1 < len(sys.argv):
            output_file = sys.argv[i+1]
        elif arg == "--no-vtk":
            generate_vtk = False
    
    # Generate wind tunnel
    response = create_wind_tunnel(
        object_mesh_file=object_mesh_file,
        tunnel_type=tunnel_type,
        flow_direction=flow_direction,
        output_file=output_file,
        mesh_quality=mesh_quality,
        generate_vtk=generate_vtk
    )
    
    if response.success:
        print(f"\n✅ Success: {response.message}")
        if response.vtk_file:
            print(f"🎨 VTK file: {response.vtk_file}")
    else:
        print(f"\n❌ Failed: {response.message}")
        if response.error_details:
            print(f"Details: {response.error_details}")

if __name__ == "__main__":
    main() 