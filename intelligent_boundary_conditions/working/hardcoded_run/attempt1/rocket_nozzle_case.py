#!/usr/bin/env python3
"""
JAX-Fluids Case Setup for Rocket Nozzle with Virtual Boundary Conditions

This file defines the case setup for internal flow in a rocket nozzle
with virtual inlet/outlet faces detected from mesh hollow openings.
"""

import jax.numpy as jnp
import numpy as np
from jaxfluids import Initializer

class RocketNozzleCase:
    """Case setup for rocket nozzle internal flow with virtual inlet/outlet BCs"""
    
    def __init__(self, virtual_bc_config):
        """Initialize with virtual boundary condition configuration"""
        self.virtual_bc = virtual_bc_config
        
        # Extract flow conditions from config
        inlet_conditions = self.virtual_bc['inlet']['conditions']
        outlet_conditions = self.virtual_bc['outlet']['conditions'] 
        
        self.chamber_pressure = inlet_conditions['pressure']
        self.chamber_temperature = inlet_conditions['temperature']
        self.inlet_velocity = inlet_conditions['velocity'][0]
        self.ambient_pressure = outlet_conditions['pressure']
        
        # Material properties
        self.gamma = 1.3  # Heat capacity ratio
        self.R = 287.0    # Specific gas constant J/(kg*K)
        
    def initial_condition(self, x, y, z):
        """Set initial conditions for the flow field"""
        
        # Initialize with ambient conditions
        rho = self.ambient_pressure / (self.R * 300.0)  # Ambient density
        u = 0.0  # Initial velocity
        v = 0.0
        w = 0.0
        p = self.ambient_pressure
        
        # Energy based on ideal gas
        e = p / (rho * (self.gamma - 1.0))
        
        return jnp.array([rho, u, v, w, e])
    
    def boundary_condition(self, primitives, time):
        """Apply virtual boundary conditions using masks"""
        
        # Load boundary masks
        try:
            inlet_mask = jnp.load("inlet_boundary_mask.npy")
            outlet_mask = jnp.load("outlet_boundary_mask.npy")
            
            # Apply inlet conditions (Dirichlet)
            inlet_density = self.chamber_pressure / (self.R * self.chamber_temperature)
            
            primitives = primitives.at[0].set(  # Density
                jnp.where(inlet_mask, inlet_density, primitives[0])
            )
            primitives = primitives.at[1].set(  # Velocity X
                jnp.where(inlet_mask, self.inlet_velocity, primitives[1])
            )
            primitives = primitives.at[2].set(  # Velocity Y
                jnp.where(inlet_mask, 0.0, primitives[2])
            )
            primitives = primitives.at[3].set(  # Velocity Z
                jnp.where(inlet_mask, 0.0, primitives[3])
            )
            primitives = primitives.at[4].set(  # Pressure
                jnp.where(inlet_mask, self.chamber_pressure, primitives[4])
            )
            
            # Outlet conditions (Neumann) - zero gradient typically handled by JAX-Fluids
            # For explicit outlet BC, we could apply:
            # primitives = primitives.at[4].set(
            #     jnp.where(outlet_mask, self.ambient_pressure, primitives[4])
            # )
            
        except FileNotFoundError:
            print("Warning: Boundary mask files not found, using default BCs")
        
        return primitives
    
    def levelset_function(self, x, y, z):
        """Define the levelset function for immersed boundaries (walls)"""
        
        # This would typically load the SDF from the immersed boundary endpoint
        # For now, return a large positive value (no solid boundaries)
        # The actual SDF will be loaded separately by JAX-Fluids
        return jnp.ones_like(x) * 1000.0
    
    def source_term(self, primitives, time):
        """Define any source terms (e.g., gravity, forcing)"""
        # No source terms for this case
        return jnp.zeros_like(primitives)

# Factory function to create the case with virtual BC configuration
def create_rocket_nozzle_case(config_file="rocket_nozzle_with_virtual_bc.json"):
    """Create rocket nozzle case with virtual boundary conditions from config file"""
    
    import json
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        virtual_bc = config.get('virtual_boundary_conditions', {})
        if not virtual_bc:
            raise ValueError("No virtual boundary conditions found in config")
        
        return RocketNozzleCase(virtual_bc)
    
    except FileNotFoundError:
        print(f"Warning: Config file {config_file} not found")
        # Create default virtual BC config
        default_virtual_bc = {
            'inlet': {
                'conditions': {
                    'pressure': 6.9e6,
                    'temperature': 3580.0,
                    'velocity': [50.0, 0.0, 0.0]
                }
            },
            'outlet': {
                'conditions': {
                    'pressure': 101325.0
                }
            }
        }
        return RocketNozzleCase(default_virtual_bc) 