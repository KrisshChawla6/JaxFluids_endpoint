#!/usr/bin/env python3
"""
Integration Adapter for Internal Flow Endpoint
===============================================

This module provides an adapter interface that allows the existing internal flow
endpoint to seamlessly integrate with the intelligent boundary conditions system.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

# Add the intelligent_boundary_conditions to path
sys.path.append(os.path.dirname(__file__))

try:
    from .main_api import (
        IntelligentBoundaryConditionsAPI,
        IntelligentBCRequest,
        IntelligentBCResponse,
        generate_intelligent_boundary_conditions
    )
    from .face_tagger import TaggingMethod, RocketNozzleType
    from .boundary_condition_generator import RocketEngineConditions
except ImportError:
    from main_api import (
        IntelligentBoundaryConditionsAPI,
        IntelligentBCRequest,
        IntelligentBCResponse,
        generate_intelligent_boundary_conditions
    )
    from face_tagger import TaggingMethod, RocketNozzleType
    from boundary_condition_generator import RocketEngineConditions

logger = logging.getLogger(__name__)

class InternalFlowBoundaryAdapter:
    """
    Adapter class for integrating intelligent boundary conditions with internal flow endpoint
    """
    
    def __init__(self):
        """Initialize the adapter"""
        self.api = IntelligentBoundaryConditionsAPI()
        self.last_response = None
        
    def generate_boundary_conditions_for_internal_flow(
        self,
        geometry_file: str,
        output_directory: str,
        internal_flow_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate boundary conditions for internal flow using geometry file
        
        Args:
            geometry_file: Path to geometry file (STL, MSH, etc.)
            output_directory: Directory to save boundary condition outputs
            internal_flow_context: Context from internal flow endpoint
            
        Returns:
            Dictionary containing boundary condition results and integration info
        """
        
        logger.info(f"Generating intelligent boundary conditions for internal flow")
        
        # Extract parameters from internal flow context
        flow_params = self._extract_flow_parameters(internal_flow_context)
        domain_params = self._extract_domain_parameters(internal_flow_context)
        
        # Create intelligent BC request
        request = IntelligentBCRequest(
            geometry_file=geometry_file,
            output_directory=output_directory,
            tagging_method=flow_params['tagging_method'],
            nozzle_type=flow_params['nozzle_type'],
            flow_axis=flow_params['flow_axis'],
            fuel_type=flow_params['fuel_type'],
            chamber_pressure=flow_params['chamber_pressure'],
            chamber_temperature=flow_params['chamber_temperature'],
            ambient_pressure=flow_params['ambient_pressure'],
            gamma=flow_params['gamma'],
            domain_resolution=domain_params['resolution'],
            domain_bounds=domain_params['bounds'],
            generate_masks=True,
            generate_config=True,
            sdf_integration=flow_params.get('sdf_integration', False)
        )
        
        # Process the request
        response = self.api.process_request(request)
        self.last_response = response
        
        # Convert response to internal flow format
        return self._convert_response_to_internal_flow_format(response, internal_flow_context)
    
    def _extract_flow_parameters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract flow parameters from internal flow context"""
        
        # Default parameters
        flow_params = {
            'tagging_method': TaggingMethod.AUTOMATIC_X_AXIS,  # X-axis common for internal flows
            'nozzle_type': RocketNozzleType.CONVERGING_DIVERGING,
            'flow_axis': 'x',
            'fuel_type': 'hydrogen',
            'chamber_pressure': 6.9e6,
            'chamber_temperature': 3580,
            'ambient_pressure': 101325,
            'gamma': 1.3
        }
        
        # Extract from context if available
        if 'flow_type' in context:
            flow_type = context['flow_type']
            if 'rocket' in flow_type.lower():
                flow_params['nozzle_type'] = RocketNozzleType.CONVERGING_DIVERGING
            elif 'bell' in flow_type.lower():
                flow_params['nozzle_type'] = RocketNozzleType.BELL_NOZZLE
        
        # Extract rocket-specific parameters
        if 'chamber_pressure' in context:
            flow_params['chamber_pressure'] = context['chamber_pressure']
        
        if 'chamber_temperature' in context or 'temperature_inlet' in context:
            temp = context.get('chamber_temperature', context.get('temperature_inlet', 3580))
            flow_params['chamber_temperature'] = temp
        
        if 'ambient_pressure' in context:
            flow_params['ambient_pressure'] = context['ambient_pressure']
        
        if 'gamma' in context:
            flow_params['gamma'] = context['gamma']
        
        if 'fuel_type' in context:
            flow_params['fuel_type'] = context['fuel_type']
        
        # Determine flow axis from geometry or context
        if 'primary_flow_direction' in context:
            flow_params['flow_axis'] = context['primary_flow_direction']
        
        logger.info(f"Extracted flow parameters: {flow_params}")
        return flow_params
    
    def _extract_domain_parameters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract domain parameters from internal flow context"""
        
        domain_params = {
            'resolution': None,
            'bounds': None
        }
        
        # Extract domain resolution if available
        if 'domain_resolution' in context:
            domain_params['resolution'] = context['domain_resolution']
        elif 'domain' in context:
            domain = context['domain']
            nx = domain.get('x', {}).get('cells', 200)
            ny = domain.get('y', {}).get('cells', 100)
            nz = domain.get('z', {}).get('cells', 1)
            domain_params['resolution'] = (nx, ny, nz)
        
        # Extract domain bounds if available
        if 'domain_bounds' in context:
            domain_params['bounds'] = context['domain_bounds']
        elif 'domain' in context:
            domain = context['domain']
            if 'x' in domain and 'y' in domain and 'z' in domain:
                bounds = {
                    'x': domain['x'].get('range', [-0.1, 0.3]),
                    'y': domain['y'].get('range', [-0.05, 0.05]),
                    'z': domain['z'].get('range', [0.0, 1.0])
                }
                domain_params['bounds'] = bounds
        
        logger.info(f"Extracted domain parameters: {domain_params}")
        return domain_params
    
    def _convert_response_to_internal_flow_format(
        self,
        response: IntelligentBCResponse,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert intelligent BC response to internal flow format"""
        
        result = {
            'success': response.success,
            'message': response.message,
            'processing_time': response.processing_time,
            'intelligent_bc_enabled': True
        }
        
        if response.success:
            # Add generated files
            result['boundary_config_file'] = response.config_file
            result['boundary_masks_directory'] = response.mask_directory
            
            # Add summary information
            if response.geometry_summary:
                result['geometry_info'] = response.geometry_summary
            
            if response.tagging_summary:
                result['boundary_tagging'] = response.tagging_summary
            
            if response.validation_results:
                result['validation'] = response.validation_results
            
            # Load and return the generated configuration for integration
            if response.config_file and os.path.exists(response.config_file):
                with open(response.config_file, 'r') as f:
                    result['jaxfluids_config'] = json.load(f)
        
        # Add warnings and errors
        if response.warnings:
            result['warnings'] = response.warnings
        
        if response.errors:
            result['errors'] = response.errors
        
        return result
    
    def enhance_internal_flow_config(
        self,
        base_config: Dict[str, Any],
        intelligent_bc_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance an existing internal flow configuration with intelligent boundary conditions
        
        Args:
            base_config: Existing JAX-Fluids configuration
            intelligent_bc_result: Result from intelligent boundary conditions
            
        Returns:
            Enhanced configuration
        """
        
        if not intelligent_bc_result.get('success', False):
            logger.warning("Intelligent BC generation failed, using base config")
            return base_config
        
        enhanced_config = base_config.copy()
        
        # Replace boundary conditions with intelligent ones
        if 'jaxfluids_config' in intelligent_bc_result:
            intelligent_config = intelligent_bc_result['jaxfluids_config']
            
            # Update boundary conditions
            if 'boundary_conditions' in intelligent_config:
                enhanced_config['boundary_conditions'] = intelligent_config['boundary_conditions']
                logger.info("Replaced boundary conditions with intelligent ones")
            
            # Optionally update initial conditions for better compatibility
            if 'initial_condition' in intelligent_config:
                # Merge initial conditions intelligently
                base_ic = enhanced_config.get('initial_condition', {})
                intelligent_ic = intelligent_config['initial_condition']
                
                # Keep base values but add any missing ones from intelligent BC
                for key, value in intelligent_ic.items():
                    if key not in base_ic:
                        base_ic[key] = value
                
                enhanced_config['initial_condition'] = base_ic
            
            # Update material properties if compatible
            if 'material_properties' in intelligent_config:
                base_mp = enhanced_config.get('material_properties', {})
                intelligent_mp = intelligent_config['material_properties']
                
                # Update equation of state if compatible
                if 'equation_of_state' in intelligent_mp:
                    if 'equation_of_state' not in base_mp:
                        base_mp['equation_of_state'] = intelligent_mp['equation_of_state']
                    else:
                        # Update specific values that are compatible
                        base_eos = base_mp['equation_of_state']
                        intelligent_eos = intelligent_mp['equation_of_state']
                        
                        for key in ['specific_heat_ratio', 'specific_gas_constant']:
                            if key in intelligent_eos and key not in base_eos:
                                base_eos[key] = intelligent_eos[key]
                
                enhanced_config['material_properties'] = base_mp
        
        # Add metadata about intelligent BC usage
        enhanced_config['intelligent_boundary_conditions'] = {
            'enabled': True,
            'geometry_file': intelligent_bc_result.get('geometry_info', {}).get('file_path'),
            'tagging_method': intelligent_bc_result.get('boundary_tagging', {}).get('tagging_method'),
            'validation_status': intelligent_bc_result.get('validation', {}).get('valid', False),
            'masks_directory': intelligent_bc_result.get('boundary_masks_directory')
        }
        
        logger.info("Enhanced internal flow configuration with intelligent boundary conditions")
        return enhanced_config

def integrate_with_internal_flow_case_setup(
    case_setup_expert,
    geometry_file: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Integration function for internal flow case setup expert
    
    Args:
        case_setup_expert: Internal flow case setup expert instance
        geometry_file: Path to geometry file
        context: Internal flow context
        
    Returns:
        Enhanced case setup with intelligent boundary conditions
    """
    
    logger.info("Integrating intelligent boundary conditions with internal flow case setup")
    
    # Create adapter
    adapter = InternalFlowBoundaryAdapter()
    
    # Generate boundary conditions
    output_dir = Path(context.get('simulation_directory', './simulation')) / 'intelligent_bc'
    
    try:
        bc_result = adapter.generate_boundary_conditions_for_internal_flow(
            geometry_file=geometry_file,
            output_directory=str(output_dir),
            internal_flow_context=context
        )
        
        # Generate base configuration using existing case setup expert
        base_config = case_setup_expert.generate_case_setup(context)
        
        # Enhance with intelligent boundary conditions
        enhanced_config = adapter.enhance_internal_flow_config(base_config, bc_result)
        
        return {
            'config': enhanced_config,
            'intelligent_bc_result': bc_result,
            'success': True,
            'message': 'Successfully integrated intelligent boundary conditions'
        }
        
    except Exception as e:
        logger.error(f"Failed to integrate intelligent boundary conditions: {e}")
        
        # Fall back to base configuration
        base_config = case_setup_expert.generate_case_setup(context)
        
        return {
            'config': base_config,
            'intelligent_bc_result': None,
            'success': False,
            'message': f'Intelligent BC integration failed: {e}. Using base configuration.',
            'error': str(e)
        }

# Convenience function for quick integration
def quick_rocket_nozzle_integration(
    geometry_file: str,
    simulation_directory: str,
    nozzle_type: str = "converging_diverging",
    fuel_type: str = "hydrogen",
    chamber_pressure: float = 6.9e6
) -> Dict[str, Any]:
    """
    Quick integration for rocket nozzle geometries
    
    Args:
        geometry_file: Path to nozzle geometry file
        simulation_directory: Simulation directory
        nozzle_type: Type of nozzle
        fuel_type: Fuel type
        chamber_pressure: Chamber pressure in Pa
        
    Returns:
        Integration result
    """
    
    context = {
        'simulation_directory': simulation_directory,
        'flow_type': 'rocket_engine',
        'nozzle_type': nozzle_type,
        'fuel_type': fuel_type,
        'chamber_pressure': chamber_pressure,
        'geometry_type': 'complex_3d',
        'advanced_physics': {
            'compressible': True,
            'supersonic': True,
            'viscous': True,
            'heat_transfer': True
        }
    }
    
    adapter = InternalFlowBoundaryAdapter()
    bc_result = adapter.generate_boundary_conditions_for_internal_flow(
        geometry_file=geometry_file,
        output_directory=os.path.join(simulation_directory, 'intelligent_bc'),
        internal_flow_context=context
    )
    
    return bc_result 