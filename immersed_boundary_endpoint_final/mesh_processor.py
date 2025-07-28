"""
Gmsh Mesh File Processor

This module provides functionality to read and process Gmsh .msh files,
extracting node coordinates and element connectivity for signed distance
function computation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GmshProcessor:
    """
    Processes Gmsh .msh files to extract geometry information.
    
    Supports Gmsh format version 4.1 and extracts:
    - Node coordinates
    - Element connectivity
    - Physical group information
    - Boundary surface triangulation
    """
    
    def __init__(self, mesh_file: str):
        """
        Initialize the Gmsh processor.
        
        Args:
            mesh_file: Path to the .msh file
        """
        self.mesh_file = Path(mesh_file)
        if not self.mesh_file.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_file}")
        
        self.nodes: Dict[int, np.ndarray] = {}
        self.elements: Dict[int, Dict] = {}
        self.physical_names: Dict[int, str] = {}
        self.entities: Dict = {}
        self.surface_triangles: List[np.ndarray] = []
        self.boundary_nodes: np.ndarray = None
        
        logger.info(f"Initialized GmshProcessor for file: {self.mesh_file}")
    
    def read_mesh(self) -> None:
        """
        Read and parse the complete mesh file.
        """
        logger.info("Reading mesh file...")
        
        with open(self.mesh_file, 'r') as f:
            content = f.read()
        
        # Parse different sections
        self._parse_mesh_format(content)
        self._parse_physical_names(content)
        self._parse_entities(content)
        self._parse_nodes(content)
        self._parse_elements(content)
        self._extract_surface_triangles()
        
        logger.info(f"Successfully read mesh with {len(self.nodes)} nodes and {len(self.elements)} elements")
    
    def _parse_mesh_format(self, content: str) -> None:
        """Parse mesh format information."""
        lines = content.split('\n')
        in_format = False
        
        for line in lines:
            if line.strip() == '$MeshFormat':
                in_format = True
                continue
            elif line.strip() == '$EndMeshFormat':
                break
            elif in_format:
                parts = line.split()
                version = float(parts[0])
                if version < 4.0:
                    logger.warning(f"Mesh format version {version} may not be fully supported. Recommended: 4.1+")
                logger.info(f"Mesh format version: {version}")
                break
    
    def _parse_physical_names(self, content: str) -> None:
        """Parse physical names section."""
        lines = content.split('\n')
        in_section = False
        
        for i, line in enumerate(lines):
            if line.strip() == '$PhysicalNames':
                in_section = True
                num_names = int(lines[i + 1])
                continue
            elif line.strip() == '$EndPhysicalNames':
                break
            elif in_section and line.strip() and not line.strip().isdigit():
                parts = line.split()
                if len(parts) >= 3:
                    dim = int(parts[0])
                    tag = int(parts[1])
                    name = parts[2].strip('"')
                    self.physical_names[tag] = name
                    logger.debug(f"Physical name: {tag} -> {name}")
    
    def _parse_entities(self, content: str) -> None:
        """Parse entities section."""
        lines = content.split('\n')
        in_section = False
        
        for i, line in enumerate(lines):
            if line.strip() == '$Entities':
                in_section = True
                continue
            elif line.strip() == '$EndEntities':
                break
            elif in_section and i == lines.index('$Entities') + 1:
                # First line after $Entities contains counts
                counts = [int(x) for x in line.split()]
                logger.debug(f"Entity counts: {counts}")
                # Store for potential future use
                self.entities['counts'] = counts
    
    def _parse_nodes(self, content: str) -> None:
        """Parse nodes section."""
        lines = content.split('\n')
        in_section = False
        
        for i, line in enumerate(lines):
            if line.strip() == '$Nodes':
                in_section = True
                continue
            elif line.strip() == '$EndNodes':
                break
            elif in_section:
                if i == lines.index('$Nodes') + 1:
                    # First line: numEntityBlocks numNodes minNodeTag maxNodeTag
                    info = [int(x) for x in line.split()]
                    num_entity_blocks, num_nodes = info[0], info[1]
                    logger.info(f"Reading {num_nodes} nodes in {num_entity_blocks} entity blocks")
                    continue
                
                # Parse entity blocks
                parts = line.split()
                if len(parts) == 4 and all(x.lstrip('-').replace('.', '').replace('e', '').replace('+', '').replace('-', '').isdigit() or x.lstrip('-').isdigit() for x in parts[:1]):
                    # This might be entity block header: entityDim entityTag parametric numNodesInBlock
                    entity_dim, entity_tag, parametric, num_nodes_in_block = map(int, parts)
                    
                    # Read node tags
                    node_tags = []
                    for j in range(num_nodes_in_block):
                        tag_line = lines[i + 1 + j]
                        node_tags.append(int(tag_line))
                    
                    # Read coordinates
                    coord_start = i + 1 + num_nodes_in_block
                    for j, tag in enumerate(node_tags):
                        coord_line = lines[coord_start + j]
                        coords = [float(x) for x in coord_line.split()]
                        self.nodes[tag] = np.array(coords)
    
    def _parse_elements(self, content: str) -> None:
        """Parse elements section."""
        lines = content.split('\n')
        in_section = False
        
        for i, line in enumerate(lines):
            if line.strip() == '$Elements':
                in_section = True
                continue
            elif line.strip() == '$EndElements':
                break
            elif in_section:
                if i == lines.index('$Elements') + 1:
                    # First line: numEntityBlocks numElements minElementTag maxElementTag
                    info = [int(x) for x in line.split()]
                    num_entity_blocks, num_elements = info[0], info[1]
                    logger.info(f"Reading {num_elements} elements in {num_entity_blocks} entity blocks")
                    continue
                
                # Parse entity blocks
                parts = line.split()
                if len(parts) == 4 and all(x.lstrip('-').isdigit() for x in parts):
                    # Entity block header: entityDim entityTag elementType numElementsInBlock
                    entity_dim, entity_tag, element_type, num_elements_in_block = map(int, parts)
                    
                    # Read elements
                    for j in range(num_elements_in_block):
                        elem_line = lines[i + 1 + j]
                        elem_data = [int(x) for x in elem_line.split()]
                        elem_tag = elem_data[0]
                        node_tags = elem_data[1:]
                        
                        self.elements[elem_tag] = {
                            'type': element_type,
                            'entity_dim': entity_dim,
                            'entity_tag': entity_tag,
                            'nodes': node_tags
                        }
    
    def _extract_surface_triangles(self) -> None:
        """Extract surface triangles for SDF computation."""
        triangles = []
        
        # First, try to find explicit surface triangles (type 2)
        for elem_tag, elem_data in self.elements.items():
            # Element type 2 = triangle, entity_dim 2 = surface
            if elem_data['type'] == 2 and elem_data['entity_dim'] == 2:
                node_tags = elem_data['nodes']
                if len(node_tags) == 3:
                    # Get coordinates of triangle vertices
                    triangle_coords = np.array([
                        self.nodes[node_tags[0]],
                        self.nodes[node_tags[1]], 
                        self.nodes[node_tags[2]]
                    ])
                    triangles.append(triangle_coords)
        
        # If no surface triangles found, extract from tetrahedra boundary
        if len(triangles) == 0:
            logger.info("No explicit surface triangles found. Extracting boundary from tetrahedra...")
            triangles = self._extract_boundary_from_tetrahedra()
        
        self.surface_triangles = triangles
        logger.info(f"Extracted {len(self.surface_triangles)} surface triangles")
    
    def _extract_boundary_from_tetrahedra(self) -> List[np.ndarray]:
        """Extract boundary triangles from tetrahedral mesh."""
        # Dictionary to count face occurrences
        face_count = {}
        face_to_coords = {}
        
        for elem_tag, elem_data in self.elements.items():
            # Element type 4 = tetrahedron
            if elem_data['type'] == 4:
                node_tags = elem_data['nodes']
                if len(node_tags) == 4:
                    # Get the 4 faces of the tetrahedron
                    faces = [
                        tuple(sorted([node_tags[0], node_tags[1], node_tags[2]])),
                        tuple(sorted([node_tags[0], node_tags[1], node_tags[3]])),
                        tuple(sorted([node_tags[0], node_tags[2], node_tags[3]])),
                        tuple(sorted([node_tags[1], node_tags[2], node_tags[3]]))
                    ]
                    
                    for face in faces:
                        face_count[face] = face_count.get(face, 0) + 1
                        if face not in face_to_coords:
                            # Store coordinates for this face
                            face_to_coords[face] = np.array([
                                self.nodes[face[0]],
                                self.nodes[face[1]],
                                self.nodes[face[2]]
                            ])
        
        # Boundary faces appear exactly once
        boundary_triangles = []
        for face, count in face_count.items():
            if count == 1:  # Boundary face
                boundary_triangles.append(face_to_coords[face])
        
        logger.info(f"Extracted {len(boundary_triangles)} boundary triangles from {len(self.elements)} tetrahedra")
        return boundary_triangles
    
    def get_boundary_points(self) -> np.ndarray:
        """
        Get all boundary surface points as a numpy array.
        
        Returns:
            Array of shape (N, 3) containing boundary point coordinates
        """
        if not self.surface_triangles:
            logger.warning("No surface triangles found. Did you call read_mesh()?")
            return np.array([])
        
        # Collect all unique boundary points
        boundary_points = []
        for triangle in self.surface_triangles:
            boundary_points.extend(triangle)
        
        # Remove duplicates
        boundary_points = np.array(boundary_points)
        unique_points = np.unique(boundary_points.view(np.void), return_index=True)[1]
        self.boundary_nodes = boundary_points[unique_points]
        
        logger.info(f"Found {len(self.boundary_nodes)} unique boundary points")
        return self.boundary_nodes
    
    def get_mesh_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the bounding box of the mesh.
        
        Returns:
            Tuple of (min_coords, max_coords) where each is shape (3,)
        """
        if not self.nodes:
            raise ValueError("No nodes loaded. Call read_mesh() first.")
        
        all_coords = np.array(list(self.nodes.values()))
        min_coords = np.min(all_coords, axis=0)
        max_coords = np.max(all_coords, axis=0)
        
        logger.info(f"Mesh bounds: min={min_coords}, max={max_coords}")
        return min_coords, max_coords
    
    def get_mesh_info(self) -> Dict:
        """
        Get comprehensive mesh information.
        
        Returns:
            Dictionary containing mesh statistics and information
        """
        if not self.nodes:
            logger.warning("No mesh data loaded. Call read_mesh() first.")
            return {}
        
        min_coords, max_coords = self.get_mesh_bounds()
        
        info = {
            'num_nodes': len(self.nodes),
            'num_elements': len(self.elements),
            'num_surface_triangles': len(self.surface_triangles),
            'physical_names': self.physical_names,
            'bounds': {
                'min': min_coords.tolist(),
                'max': max_coords.tolist(),
                'size': (max_coords - min_coords).tolist()
            }
        }
        
        return info 