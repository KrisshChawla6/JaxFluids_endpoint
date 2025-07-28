#!/usr/bin/env python3
"""
Intelligent Boundary Conditions Main API
=========================================

This module provides the main API endpoint for intelligent boundary condition
generation for rocket nozzles and complex 3D internal flow geometries.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import time

try:
    from .geometry_parser import GeometryParser
    from .face_tagger import FaceTagger, TaggingMethod, RocketNozzleType
    from .boundary_condition_generator import (
        BoundaryConditionGenerator, 
        RocketEngineConditions,
        BoundaryConditionType
    )
except ImportError:
    from geometry_parser import GeometryParser
    from face_tagger import FaceTagger, TaggingMethod, RocketNozzleType
    from boundary_condition_generator import (
        BoundaryConditionGenerator, 
        RocketEngineConditions,
        BoundaryConditionType
    )

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IntelligentBCRequest:
    """Request structure for intelligent boundary conditions"""
    
    # Required parameters
    geometry_file: str
    output_directory: str
    
    # Flow conditions
    flow_conditions: Optional[Dict[str, float]] = None
    fuel_type: str = "hydrogen"
    chamber_pressure: float = 6.9e6  # Pa
    chamber_temperature: float = 3580  # K
    ambient_pressure: float = 101325  # Pa
    gamma: float = 1.3
    
    # Tagging configuration
    tagging_method: TaggingMethod = TaggingMethod.AUTOMATIC_Z_AXIS
    nozzle_type: RocketNozzleType = RocketNozzleType.CONVERGING_DIVERGING
    flow_axis: str = 'x'
    manual_tagging: bool = False
    visualization: bool = False
    
    # Domain configuration
    domain_resolution: Optional[Tuple[int, int, int]] = None
    domain_bounds: Optional[Dict[str, List[float]]] = None
    
    # Output configuration
    generate_masks: bool = True
    generate_config: bool = True
    output_format: str = "json"
    
    # Integration options
    sdf_integration: bool = False
    sdf_file: Optional[str] = None

@dataclass
class IntelligentBCResponse:
    """Response structure for intelligent boundary conditions"""
    
    success: bool
    message: str
    
    # Generated files
    config_file: Optional[str] = None
    mask_directory: Optional[str] = None
    
    # Summary information
    geometry_summary: Optional[Dict[str, Any]] = None
    tagging_summary: Optional[Dict[str, Any]] = None
    validation_results: Optional[Dict[str, Any]] = None
    generation_summary: Optional[Dict[str, Any]] = None
    
    # Processing metadata
    processing_time: Optional[float] = None
    warnings: Optional[List[str]] = None
    errors: Optional[List[str]] = None

class IntelligentBoundaryConditionsAPI:
    """
    Main API class for intelligent boundary condition generation
    """
    
    def __init__(self):
        """Initialize the API"""
        self.geometry_parser = None
        self.face_tagger = None
        self.bc_generator = None
        
        # Processing state
        self.last_request = None
        self.last_response = None
        
        logger.info("Intelligent Boundary Conditions API initialized")
    
    def process_request(self, request: IntelligentBCRequest) -> IntelligentBCResponse:
        """
        Main processing method for intelligent boundary condition requests
        
        Args:
            request: Boundary condition request
            
        Returns:
            Boundary condition response
        """
        
        start_time = time.time()
        self.last_request = request
        
        response = IntelligentBCResponse(
            success=False,
            message="Processing started...",
            warnings=[],
            errors=[]
        )
        
        try:
            logger.info(f"Processing intelligent boundary conditions for: {request.geometry_file}")
            
            # Step 1: Parse geometry
            logger.info("Step 1: Parsing geometry...")
            self._parse_geometry(request, response)
            
            # Step 2: Tag faces
            logger.info("Step 2: Tagging faces...")
            self._tag_faces(request, response)
            
            # Step 3: Generate boundary conditions
            logger.info("Step 3: Generating boundary conditions...")
            self._generate_boundary_conditions(request, response)
            
            # Step 4: Save outputs
            logger.info("Step 4: Saving outputs...")
            self._save_outputs(request, response)
            
            # Step 5: Generate summary
            self._generate_summary(request, response)
            
            response.success = True
            response.message = "Intelligent boundary conditions generated successfully"
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            response.success = False
            response.message = f"Error: {str(e)}"
            response.errors.append(str(e))
        
        finally:
            response.processing_time = time.time() - start_time
            self.last_response = response
            logger.info(f"Processing completed in {response.processing_time:.2f} seconds")
        
        return response
    
    def _parse_geometry(self, request: IntelligentBCRequest, response: IntelligentBCResponse):
        """Parse the geometry file"""
        
        if not os.path.exists(request.geometry_file):
            raise FileNotFoundError(f"Geometry file not found: {request.geometry_file}")
        
        # Initialize geometry parser
        self.geometry_parser = GeometryParser(request.geometry_file)
        
        # Parse the geometry
        geometry_data = self.geometry_parser.parse_geometry()
        
        # Store summary
        response.geometry_summary = self.geometry_parser.get_geometry_summary()
        
        logger.info(f"Parsed geometry: {response.geometry_summary['num_faces']} faces")
    
    def _tag_faces(self, request: IntelligentBCRequest, response: IntelligentBCResponse):
        """Tag faces for boundary conditions"""
        
        if not self.geometry_parser:
            raise ValueError("Geometry must be parsed first")
        
        # Initialize face tagger
        self.face_tagger = FaceTagger(self.geometry_parser)
        
        # Tag faces based on method
        if request.manual_tagging:
            tagged_faces = self.face_tagger.manual_tag_faces(
                visualization=request.visualization
            )
        else:
            tagged_faces = self.face_tagger.auto_tag_faces(
                method=request.tagging_method,
                nozzle_type=request.nozzle_type,
                flow_axis=request.flow_axis
            )
        
        # Get tagging summary
        response.tagging_summary = self.face_tagger.get_tagging_summary()
        
        # Validate tagging
        response.validation_results = self.face_tagger.validate_tagging()
        
        if not response.validation_results['valid']:
            for error in response.validation_results['errors']:
                response.errors.append(f"Tagging validation: {error}")
        
        for warning in response.validation_results['warnings']:
            response.warnings.append(f"Tagging validation: {warning}")
        
        logger.info(f"Tagged faces: "
                   f"inlet={response.tagging_summary['tagged_counts']['inlet']}, "
                   f"outlet={response.tagging_summary['tagged_counts']['outlet']}, "
                   f"wall={response.tagging_summary['tagged_counts']['wall']}")
    
    def _generate_boundary_conditions(self, request: IntelligentBCRequest, response: IntelligentBCResponse):
        """Generate boundary conditions and configurations"""
        
        if not self.face_tagger:
            raise ValueError("Faces must be tagged first")
        
        # Initialize boundary condition generator
        self.bc_generator = BoundaryConditionGenerator(
            self.geometry_parser, 
            self.face_tagger
        )
        
        # Generate flow conditions
        if request.flow_conditions:
            flow_conditions = request.flow_conditions
        else:
            flow_conditions = RocketEngineConditions.get_rocket_conditions(
                fuel_type=request.fuel_type,
                chamber_pressure=request.chamber_pressure,
                chamber_temperature=request.chamber_temperature,
                ambient_pressure=request.ambient_pressure,
                gamma=request.gamma
            )
        
        # Generate domain configuration
        domain_config = self._generate_domain_config(request)
        
        # Generate JAX-Fluids configuration
        if request.generate_config:
            self.jaxfluids_config = self.bc_generator.generate_jaxfluids_config(
                flow_conditions=flow_conditions,
                domain_config=domain_config
            )
            
            # Add SDF integration if requested
            if request.sdf_integration:
                levelset_config = self.bc_generator.generate_levelset_integration(
                    sdf_file=request.sdf_file
                )
                self.jaxfluids_config.update(levelset_config)
        
        # Generate boundary masks
        if request.generate_masks and request.domain_resolution:
            self.boundary_masks = self.bc_generator.generate_boundary_masks(
                domain_resolution=request.domain_resolution,
                output_format="numpy"
            )
        
        logger.info("Generated boundary conditions and configuration")
    
    def _generate_domain_config(self, request: IntelligentBCRequest) -> Dict[str, Any]:
        """Generate domain configuration"""
        
        bounds = self.geometry_parser.geometry_bounds
        
        if request.domain_bounds:
            # Use provided bounds
            x_range = request.domain_bounds['x']
            y_range = request.domain_bounds['y']
            z_range = request.domain_bounds['z']
        else:
            # Use geometry bounds with padding
            padding = 0.1  # 10% padding
            extent = bounds['extent']
            
            x_range = [bounds['min'][0] - padding * extent[0], 
                      bounds['max'][0] + padding * extent[0]]
            y_range = [bounds['min'][1] - padding * extent[1], 
                      bounds['max'][1] + padding * extent[1]]
            z_range = [bounds['min'][2] - padding * extent[2], 
                      bounds['max'][2] + padding * extent[2]]
        
        if request.domain_resolution:
            nx, ny, nz = request.domain_resolution
        else:
            # Default resolution
            nx, ny, nz = 200, 100, 1
        
        domain_config = {
            "x": {
                "cells": nx,
                "range": x_range
            },
            "y": {
                "cells": ny,
                "range": y_range
            },
            "z": {
                "cells": nz,
                "range": z_range
            },
            "decomposition": {
                "split_x": 1,
                "split_y": 1,
                "split_z": 1
            }
        }
        
        return domain_config
    
    def _save_outputs(self, request: IntelligentBCRequest, response: IntelligentBCResponse):
        """Save generated outputs to files"""
        
        output_dir = Path(request.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        if request.generate_config and hasattr(self, 'jaxfluids_config'):
            config_file = output_dir / f"intelligent_boundary_config.{request.output_format}"
            self.bc_generator.save_configuration(
                self.jaxfluids_config,
                str(config_file),
                format=request.output_format
            )
            response.config_file = str(config_file)
        
        # Save boundary masks
        if request.generate_masks and hasattr(self, 'boundary_masks'):
            mask_dir = output_dir / "boundary_masks"
            self.bc_generator.save_boundary_masks(
                self.boundary_masks,
                str(mask_dir)
            )
            response.mask_directory = str(mask_dir)
        
        logger.info(f"Saved outputs to: {output_dir}")
    
    def _generate_summary(self, request: IntelligentBCRequest, response: IntelligentBCResponse):
        """Generate processing summary"""
        
        if self.bc_generator:
            response.generation_summary = self.bc_generator.get_generation_summary()
        
        logger.info("Generated processing summary")
    
    def create_sample_request(self, 
                            geometry_file: str,
                            output_directory: str = "./intelligent_bc_output") -> IntelligentBCRequest:
        """
        Create a sample request with default parameters
        
        Args:
            geometry_file: Path to geometry file
            output_directory: Output directory path
            
        Returns:
            Sample request object
        """
        
        return IntelligentBCRequest(
            geometry_file=geometry_file,
            output_directory=output_directory,
            tagging_method=TaggingMethod.AUTOMATIC_Z_AXIS,
            nozzle_type=RocketNozzleType.CONVERGING_DIVERGING,
            flow_axis='x',
            domain_resolution=(200, 100, 1),
            generate_masks=True,
            generate_config=True
        )
    
    def get_available_methods(self) -> Dict[str, List[str]]:
        """Get available tagging methods and nozzle types"""
        
        return {
            'tagging_methods': [method.value for method in TaggingMethod],
            'nozzle_types': [nozzle.value for nozzle in RocketNozzleType],
            'boundary_condition_types': [
                BoundaryConditionType.SIMPLE_INFLOW,
                BoundaryConditionType.SIMPLE_OUTFLOW,
                BoundaryConditionType.NOSLIP,
                BoundaryConditionType.SYMMETRY
            ]
        }
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        
        status = {
            'last_request': self.last_request is not None,
            'last_response': self.last_response is not None,
            'geometry_parsed': self.geometry_parser is not None,
            'faces_tagged': self.face_tagger is not None,
            'bc_generated': self.bc_generator is not None
        }
        
        if self.last_response:
            status.update({
                'last_success': self.last_response.success,
                'last_processing_time': self.last_response.processing_time,
                'last_message': self.last_response.message
            })
        
        return status

# Convenience functions for direct usage

def generate_intelligent_boundary_conditions(
    geometry_file: str,
    output_directory: str = "./intelligent_bc_output",
    tagging_method: str = "z_axis_heuristic",
    nozzle_type: str = "converging_diverging",
    flow_axis: str = 'x',
    fuel_type: str = "hydrogen",
    chamber_pressure: float = 6.9e6,
    chamber_temperature: float = 3580,
    ambient_pressure: float = 101325,
    domain_resolution: Tuple[int, int, int] = (200, 100, 1),
    manual_tagging: bool = False,
    visualization: bool = False,
    **kwargs
) -> IntelligentBCResponse:
    """
    Convenience function for generating intelligent boundary conditions
    
    Args:
        geometry_file: Path to geometry file (STL, MSH, etc.)
        output_directory: Directory to save outputs
        tagging_method: Method for face tagging
        nozzle_type: Type of rocket nozzle
        flow_axis: Primary flow direction ('x', 'y', 'z')
        fuel_type: Type of fuel ("hydrogen", "kerosene", "methane")
        chamber_pressure: Chamber pressure in Pa
        chamber_temperature: Chamber temperature in K
        ambient_pressure: Ambient pressure in Pa
        domain_resolution: Grid resolution (nx, ny, nz)
        manual_tagging: Whether to use manual face tagging
        visualization: Whether to show visualization for manual tagging
        **kwargs: Additional parameters
        
    Returns:
        Response object with results
    """
    
    # Convert string enums
    tagging_method_enum = TaggingMethod(tagging_method)
    nozzle_type_enum = RocketNozzleType(nozzle_type)
    
    # Create request
    request = IntelligentBCRequest(
        geometry_file=geometry_file,
        output_directory=output_directory,
        tagging_method=tagging_method_enum,
        nozzle_type=nozzle_type_enum,
        flow_axis=flow_axis,
        fuel_type=fuel_type,
        chamber_pressure=chamber_pressure,
        chamber_temperature=chamber_temperature,
        ambient_pressure=ambient_pressure,
        domain_resolution=domain_resolution,
        manual_tagging=manual_tagging,
        visualization=visualization,
        **kwargs
    )
    
    # Process request
    api = IntelligentBoundaryConditionsAPI()
    response = api.process_request(request)
    
    return response

def quick_rocket_nozzle_bc(
    geometry_file: str,
    output_directory: str = "./rocket_nozzle_bc"
) -> IntelligentBCResponse:
    """
    Quick setup for rocket nozzle boundary conditions with defaults
    
    Args:
        geometry_file: Path to geometry file
        output_directory: Output directory
        
    Returns:
        Response object with results
    """
    
    return generate_intelligent_boundary_conditions(
        geometry_file=geometry_file,
        output_directory=output_directory,
        tagging_method="z_axis_heuristic",
        nozzle_type="converging_diverging",
        flow_axis='x',
        fuel_type="hydrogen",
        domain_resolution=(300, 150, 1)
    ) 