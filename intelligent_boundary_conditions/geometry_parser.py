#!/usr/bin/env python3
"""
Geometry Parser Module
======================

This module handles parsing of various geometry formats (STL, MSH, CAD)
and extracts faces for boundary condition tagging in rocket nozzles and
complex 3D internal flow geometries.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

# Geometry processing libraries
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

logger = logging.getLogger(__name__)

class GeometryFace:
    """Represents a face in the geometry with associated metadata"""
    
    def __init__(self, vertices: np.ndarray, normal: np.ndarray, 
                 centroid: np.ndarray, area: float, face_id: int):
        self.vertices = vertices  # (N, 3) array of vertex coordinates
        self.normal = normal      # (3,) array - face normal vector
        self.centroid = centroid  # (3,) array - face centroid
        self.area = area          # scalar - face area
        self.face_id = face_id    # unique face identifier
        self.tag = None           # boundary condition tag (inlet/outlet/wall)
        
    def __repr__(self):
        return f"GeometryFace(id={self.face_id}, centroid={self.centroid}, area={self.area:.4f})"

class GeometryParser:
    """
    Parses various geometry formats and extracts faces for boundary condition tagging
    """
    
    def __init__(self, geometry_file: str):
        """
        Initialize geometry parser
        
        Args:
            geometry_file: Path to geometry file (STL, MSH, etc.)
        """
        self.geometry_file = Path(geometry_file)
        self.mesh = None
        self.faces = []
        self.geometry_bounds = None
        self.file_format = None
        
        if not self.geometry_file.exists():
            raise FileNotFoundError(f"Geometry file not found: {geometry_file}")
            
        self._detect_file_format()
        
    def _detect_file_format(self):
        """Detect the file format based on extension"""
        ext = self.geometry_file.suffix.lower()
        
        format_map = {
            '.stl': 'stl',
            '.msh': 'msh', 
            '.step': 'step',
            '.stp': 'step',
            '.iges': 'iges',
            '.igs': 'iges',
            '.obj': 'obj',
            '.ply': 'ply'
        }
        
        self.file_format = format_map.get(ext)
        if not self.file_format:
            raise ValueError(f"Unsupported file format: {ext}")
            
        logger.info(f"Detected file format: {self.file_format}")
    
    def parse_geometry(self) -> Dict[str, Any]:
        """
        Parse the geometry file and extract mesh information
        
        Returns:
            Dictionary containing mesh data and metadata
        """
        
        if self.file_format == 'stl':
            return self._parse_stl()
        elif self.file_format == 'msh':
            return self._parse_msh()
        elif self.file_format in ['step', 'iges']:
            return self._parse_cad()
        else:
            return self._parse_generic()
    
    def _parse_stl(self) -> Dict[str, Any]:
        """Parse STL file using trimesh"""
        
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is required for STL parsing. Install with: pip install trimesh")
            
        try:
            logger.info(f"Loading STL file: {self.geometry_file}")
            self.mesh = trimesh.load_mesh(str(self.geometry_file))
            
            # Ensure mesh is a single object
            if isinstance(self.mesh, trimesh.Scene):
                # Convert scene to single mesh
                self.mesh = self.mesh.dump(concatenate=True)
            
            # Extract basic properties
            vertices = self.mesh.vertices
            faces = self.mesh.faces
            face_normals = self.mesh.face_normals
            face_centroids = self.mesh.triangles_center
            
            # Calculate face areas
            face_areas = self.mesh.area_faces
            
            # Store geometry bounds
            self.geometry_bounds = {
                'min': vertices.min(axis=0),
                'max': vertices.max(axis=0),
                'center': vertices.mean(axis=0),
                'extent': vertices.max(axis=0) - vertices.min(axis=0)
            }
            
            # Create GeometryFace objects
            self.faces = []
            for i, face_indices in enumerate(faces):
                face_vertices = vertices[face_indices]
                face_obj = GeometryFace(
                    vertices=face_vertices,
                    normal=face_normals[i],
                    centroid=face_centroids[i],
                    area=face_areas[i],
                    face_id=i
                )
                self.faces.append(face_obj)
            
            logger.info(f"Successfully parsed STL: {len(self.faces)} faces")
            
            return {
                'mesh': self.mesh,
                'vertices': vertices,
                'faces': faces,
                'face_normals': face_normals,
                'face_centroids': face_centroids,
                'face_areas': face_areas,
                'bounds': self.geometry_bounds,
                'num_faces': len(self.faces),
                'format': 'stl'
            }
            
        except Exception as e:
            logger.error(f"Failed to parse STL file: {e}")
            raise
    
    def _parse_msh(self) -> Dict[str, Any]:
        """Parse GMSH .msh file using meshio"""
        
        if not MESHIO_AVAILABLE:
            raise ImportError("meshio is required for MSH parsing. Install with: pip install meshio")
            
        try:
            logger.info(f"Loading MSH file: {self.geometry_file}")
            mesh = meshio.read(str(self.geometry_file))
            
            # Extract surface elements (triangles, quads, or extract from volume)
            surface_elements = []
            volume_elements = []
            
            for cell_block in mesh.cells:
                if cell_block.type in ['triangle', 'quad']:
                    surface_elements.append(cell_block)
                elif cell_block.type in ['tetra', 'hexahedron', 'pyramid', 'wedge']:
                    volume_elements.append(cell_block)
            
            # If no direct surface elements, extract boundary from volume elements
            if not surface_elements and volume_elements:
                logger.info("No direct surface elements found, extracting boundary faces from volume elements...")
                surface_elements = self._extract_boundary_faces_from_volume(mesh, volume_elements)
            
            if not surface_elements:
                raise ValueError("No surface elements found in MSH file")
            
            # Use the first surface element block
            surface_cells = surface_elements[0]
            vertices = mesh.points
            faces = surface_cells.data
            
            # Calculate face properties
            face_centroids = []
            face_normals = []
            face_areas = []
            
            for face_indices in faces:
                face_vertices = vertices[face_indices]
                
                # Calculate centroid
                centroid = face_vertices.mean(axis=0)
                face_centroids.append(centroid)
                
                # Calculate normal (for triangles)
                if len(face_indices) == 3:
                    v1 = face_vertices[1] - face_vertices[0]
                    v2 = face_vertices[2] - face_vertices[0]
                    normal = np.cross(v1, v2)
                    normal = normal / np.linalg.norm(normal)
                    area = 0.5 * np.linalg.norm(np.cross(v1, v2))
                else:
                    # For quads, approximate as two triangles
                    v1 = face_vertices[1] - face_vertices[0]
                    v2 = face_vertices[2] - face_vertices[0]
                    normal = np.cross(v1, v2)
                    normal = normal / np.linalg.norm(normal)
                    # Approximate area for quad
                    area = np.linalg.norm(np.cross(v1, v2)) * 0.5
                    v3 = face_vertices[3] - face_vertices[0]
                    area += np.linalg.norm(np.cross(v2, v3)) * 0.5
                
                face_normals.append(normal)
                face_areas.append(area)
            
            face_normals = np.array(face_normals)
            face_centroids = np.array(face_centroids)
            face_areas = np.array(face_areas)
            
            # Store geometry bounds
            self.geometry_bounds = {
                'min': vertices.min(axis=0),
                'max': vertices.max(axis=0),
                'center': vertices.mean(axis=0),
                'extent': vertices.max(axis=0) - vertices.min(axis=0)
            }
            
            # Create GeometryFace objects
            self.faces = []
            for i, face_indices in enumerate(faces):
                face_vertices = vertices[face_indices]
                face_obj = GeometryFace(
                    vertices=face_vertices,
                    normal=face_normals[i],
                    centroid=face_centroids[i],
                    area=face_areas[i],
                    face_id=i
                )
                self.faces.append(face_obj)
            
            logger.info(f"Successfully parsed MSH: {len(self.faces)} faces")
            
            return {
                'mesh': mesh,
                'vertices': vertices,
                'faces': faces,
                'face_normals': face_normals,
                'face_centroids': face_centroids,
                'face_areas': face_areas,
                'bounds': self.geometry_bounds,
                'num_faces': len(self.faces),
                'format': 'msh'
            }
            
        except Exception as e:
            logger.error(f"Failed to parse MSH file: {e}")
            raise
    
    def _extract_boundary_faces_from_volume(self, mesh, volume_elements):
        """Extract boundary faces from volume elements"""
        
        try:
            import trimesh
            TRIMESH_AVAILABLE = True
        except ImportError:
            TRIMESH_AVAILABLE = False
        
        if not TRIMESH_AVAILABLE:
            logger.warning("trimesh not available for boundary extraction. Trying simple approach.")
            return self._simple_boundary_extraction(mesh, volume_elements)
        
        # Use the first volume element block
        volume_cells = volume_elements[0]
        vertices = mesh.points
        
        # Create a trimesh object from volume elements
        if volume_cells.type == 'tetra':
            # For tetrahedra, extract triangular faces
            faces = []
            tetra_faces = [
                [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]  # 4 faces per tetrahedron
            ]
            
            face_count = {}
            for tetra in volume_cells.data:
                for face_indices in tetra_faces:
                    face = tuple(sorted([tetra[face_indices[0]], tetra[face_indices[1]], tetra[face_indices[2]]]))
                    face_count[face] = face_count.get(face, 0) + 1
            
            # Boundary faces appear only once (not shared between elements)
            boundary_faces = [list(face) for face, count in face_count.items() if count == 1]
            
            # Create a mock cell block for boundary faces
            class MockCellBlock:
                def __init__(self, faces):
                    self.type = 'triangle'
                    self.data = np.array(boundary_faces)
            
            logger.info(f"Extracted {len(boundary_faces)} boundary triangular faces from tetrahedra")
            return [MockCellBlock(boundary_faces)]
        
        else:
            logger.warning(f"Boundary extraction not implemented for element type: {volume_cells.type}")
            return []
    
    def _simple_boundary_extraction(self, mesh, volume_elements):
        """Simple boundary extraction without trimesh"""
        
        # For now, just use all points as a simple approach
        vertices = mesh.points
        n_vertices = len(vertices)
        
        # Create triangular faces by connecting nearby points (rough approximation)
        faces = []
        for i in range(0, min(n_vertices - 2, 100), 3):  # Limit to first 100 points for safety
            faces.append([i, i + 1, i + 2])
        
        class MockCellBlock:
            def __init__(self, faces):
                self.type = 'triangle'
                self.data = np.array(faces)
        
        logger.warning(f"Using simple boundary extraction: {len(faces)} faces")
        return [MockCellBlock(faces)]
    
    def _parse_cad(self) -> Dict[str, Any]:
        """Parse CAD files (STEP, IGES) - requires additional CAD libraries"""
        
        # For CAD files, we need specialized libraries like FreeCAD or OpenCASCADE
        # For now, we'll try to convert to mesh using trimesh if possible
        
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is required for CAD parsing. Install with: pip install trimesh")
        
        try:
            logger.info(f"Loading CAD file: {self.geometry_file}")
            logger.warning("CAD file support is experimental. Consider converting to STL first.")
            
            # Try to load with trimesh (may work for some CAD formats)
            self.mesh = trimesh.load(str(self.geometry_file))
            
            if isinstance(self.mesh, trimesh.Scene):
                self.mesh = self.mesh.dump(concatenate=True)
            
            # If successful, treat as STL
            return self._parse_stl_like_mesh(self.mesh)
            
        except Exception as e:
            logger.error(f"Failed to parse CAD file: {e}")
            raise ValueError(f"CAD file parsing failed. Please convert to STL format first.")
    
    def _parse_generic(self) -> Dict[str, Any]:
        """Parse other mesh formats using trimesh"""
        
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is required for generic mesh parsing")
        
        try:
            logger.info(f"Loading mesh file: {self.geometry_file}")
            self.mesh = trimesh.load(str(self.geometry_file))
            
            if isinstance(self.mesh, trimesh.Scene):
                self.mesh = self.mesh.dump(concatenate=True)
            
            return self._parse_stl_like_mesh(self.mesh)
            
        except Exception as e:
            logger.error(f"Failed to parse mesh file: {e}")
            raise
    
    def _parse_stl_like_mesh(self, mesh) -> Dict[str, Any]:
        """Common parsing logic for trimesh objects"""
        
        vertices = mesh.vertices
        faces = mesh.faces
        face_normals = mesh.face_normals
        face_centroids = mesh.triangles_center
        face_areas = mesh.area_faces
        
        self.geometry_bounds = {
            'min': vertices.min(axis=0),
            'max': vertices.max(axis=0),
            'center': vertices.mean(axis=0),
            'extent': vertices.max(axis=0) - vertices.min(axis=0)
        }
        
        self.faces = []
        for i, face_indices in enumerate(faces):
            face_vertices = vertices[face_indices]
            face_obj = GeometryFace(
                vertices=face_vertices,
                normal=face_normals[i],
                centroid=face_centroids[i],
                area=face_areas[i],
                face_id=i
            )
            self.faces.append(face_obj)
        
        logger.info(f"Successfully parsed mesh: {len(self.faces)} faces")
        
        return {
            'mesh': mesh,
            'vertices': vertices,
            'faces': faces,
            'face_normals': face_normals,
            'face_centroids': face_centroids,
            'face_areas': face_areas,
            'bounds': self.geometry_bounds,
            'num_faces': len(self.faces),
            'format': self.file_format
        }
    
    def get_face_by_id(self, face_id: int) -> Optional[GeometryFace]:
        """Get face by ID"""
        for face in self.faces:
            if face.face_id == face_id:
                return face
        return None
    
    def get_faces_by_tag(self, tag: str) -> List[GeometryFace]:
        """Get all faces with a specific tag"""
        return [face for face in self.faces if face.tag == tag]
    
    def get_geometry_summary(self) -> Dict[str, Any]:
        """Get summary of parsed geometry"""
        
        if not self.faces:
            return {"error": "No geometry parsed yet"}
        
        return {
            'file_path': str(self.geometry_file),
            'format': self.file_format,
            'num_faces': len(self.faces),
            'bounds': self.geometry_bounds,
            'total_area': sum(face.area for face in self.faces),
            'tagged_faces': {
                'inlet': len(self.get_faces_by_tag('inlet')),
                'outlet': len(self.get_faces_by_tag('outlet')),
                'wall': len(self.get_faces_by_tag('wall')),
                'untagged': len([f for f in self.faces if f.tag is None])
            }
        } 