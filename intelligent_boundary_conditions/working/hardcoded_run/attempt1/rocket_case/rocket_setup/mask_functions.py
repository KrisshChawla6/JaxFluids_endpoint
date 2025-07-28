#!/usr/bin/env python3
"""
JAX-Fluids Mask Functions for Virtual Boundary Conditions

This module loads our precomputed inlet/outlet masks and provides 
JAX-compatible functions that can be used with JAX-Fluids' native 
forcing system to apply boundary conditions inside the domain.

NO FALLBACKS - ONLY REAL MASK FILES.
"""

import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Union, Callable

class VirtualBoundaryMasks:
    """Load and manage virtual boundary masks for JAX-Fluids forcing"""
    
    def __init__(self):
        self.inlet_mask = None
        self.outlet_mask = None
        self.grid_info = None
        self.domain_bounds = None
        self._masks_loaded = False
        
    def load_masks(self):
        """Load the precomputed inlet/outlet masks - FAIL if not found"""
        # Load masks from our rocket simulation
        mask_dir = Path("../../rocket_nozzle_jaxfluids_simulation")
        
        inlet_file = mask_dir / "inlet_boundary_mask.npy"
        outlet_file = mask_dir / "outlet_boundary_mask.npy"
        
        if not inlet_file.exists():
            raise FileNotFoundError(f"❌ CRITICAL: Inlet mask file not found: {inlet_file}")
        
        if not outlet_file.exists():
            raise FileNotFoundError(f"❌ CRITICAL: Outlet mask file not found: {outlet_file}")
        
        # Load the real mask files
        self.inlet_mask = np.load(inlet_file)
        self.outlet_mask = np.load(outlet_file)
        
        print(f"✅ Loaded virtual boundary masks:")
        print(f"   🔵 Inlet: {np.sum(self.inlet_mask):,} active grid points")
        print(f"   🔴 Outlet: {np.sum(self.outlet_mask):,} active grid points")
        
        # Validate masks
        if self.inlet_mask.shape != self.outlet_mask.shape:
            raise ValueError("❌ CRITICAL: Inlet and outlet masks have different shapes")
        
        if np.sum(self.inlet_mask) == 0:
            raise ValueError("❌ CRITICAL: Inlet mask is empty")
            
        if np.sum(self.outlet_mask) == 0:
            raise ValueError("❌ CRITICAL: Outlet mask is empty")
        
        # Store grid information for coordinate mapping
        self.grid_shape = self.inlet_mask.shape
        self.domain_bounds = np.array([-200.0, -800.0, -800.0, 1800.0, 800.0, 800.0])  # x_min, y_min, z_min, x_max, y_max, z_max
        
        self._masks_loaded = True
        print(f"✅ Mask validation complete - Grid shape: {self.grid_shape}")
    
    def create_coordinate_mappers(self):
        """Create functions to map physical coordinates to grid indices"""
        if not self._masks_loaded:
            raise RuntimeError("❌ CRITICAL: Masks not loaded. Call load_masks() first.")
            
        nx, ny, nz = self.grid_shape
        x_min, y_min, z_min, x_max, y_max, z_max = self.domain_bounds
        
        # Grid spacing
        dx = (x_max - x_min) / (nx - 1)
        dy = (y_max - y_min) / (ny - 1) 
        dz = (z_max - z_min) / (nz - 1)
        
        def coord_to_index(x, y, z):
            """Convert physical coordinates to grid indices"""
            i = jnp.clip(jnp.round((x - x_min) / dx).astype(int), 0, nx-1)
            j = jnp.clip(jnp.round((y - y_min) / dy).astype(int), 0, ny-1)
            k = jnp.clip(jnp.round((z - z_min) / dz).astype(int), 0, nz-1)
            return i, j, k
            
        return coord_to_index
    
    def create_inlet_mask_function(self) -> Callable:
        """Create JAX-compatible function for inlet mask"""
        if not self._masks_loaded:
            raise RuntimeError("❌ CRITICAL: Masks not loaded. Call load_masks() first.")
            
        # Convert numpy mask to JAX array
        inlet_mask_jax = jnp.array(self.inlet_mask)
        coord_to_index = self.create_coordinate_mappers()
        
        def inlet_mask_func(x, y, z):
            """JAX-compatible inlet mask function"""
            i, j, k = coord_to_index(x, y, z)
            return inlet_mask_jax[i, j, k]
            
        return inlet_mask_func
    
    def create_outlet_mask_function(self) -> Callable:
        """Create JAX-compatible function for outlet mask"""
        if not self._masks_loaded:
            raise RuntimeError("❌ CRITICAL: Masks not loaded. Call load_masks() first.")
            
        # Convert numpy mask to JAX array  
        outlet_mask_jax = jnp.array(self.outlet_mask)
        coord_to_index = self.create_coordinate_mappers()
        
        def outlet_mask_func(x, y, z):
            """JAX-compatible outlet mask function"""
            i, j, k = coord_to_index(x, y, z)
            return outlet_mask_jax[i, j, k]
            
        return outlet_mask_func
    
    def create_combined_mask_function(self) -> Callable:
        """Create combined inlet+outlet mask function for forcing"""
        if not self._masks_loaded:
            raise RuntimeError("❌ CRITICAL: Masks not loaded. Call load_masks() first.")
            
        inlet_func = self.create_inlet_mask_function()
        outlet_func = self.create_outlet_mask_function()
        
        def combined_mask_func(x, y, z, t=0.0):
            """Combined mask function for JAX-Fluids forcing"""
            return inlet_func(x, y, z) | outlet_func(x, y, z)
            
        return combined_mask_func

# Global instance for easy access
_mask_manager = VirtualBoundaryMasks()

def get_inlet_mask_function():
    """Get the inlet mask function for JAX-Fluids"""
    if not _mask_manager._masks_loaded:
        raise RuntimeError("❌ CRITICAL: Must call initialize_masks() first")
    return _mask_manager.create_inlet_mask_function()

def get_outlet_mask_function():
    """Get the outlet mask function for JAX-Fluids"""
    if not _mask_manager._masks_loaded:
        raise RuntimeError("❌ CRITICAL: Must call initialize_masks() first")
    return _mask_manager.create_outlet_mask_function()

def get_combined_mask_function():
    """Get the combined mask function for JAX-Fluids"""
    if not _mask_manager._masks_loaded:
        raise RuntimeError("❌ CRITICAL: Must call initialize_masks() first")
    return _mask_manager.create_combined_mask_function()

def initialize_masks():
    """Initialize and load all masks - NO FALLBACKS"""
    _mask_manager.load_masks()

if __name__ == "__main__":
    # Test the mask functions
    print("🧪 Testing mask functions...")
    
    try:
        initialize_masks()
        print("✅ Real masks loaded successfully")
        
        # Test the functions
        inlet_func = get_inlet_mask_function()
        outlet_func = get_outlet_mask_function()
        combined_func = get_combined_mask_function()
        
        # Test with sample coordinates
        test_x, test_y, test_z = 0.0, 0.0, 0.0
        
        print(f"Test coordinates ({test_x}, {test_y}, {test_z}):")
        print(f"  Inlet mask: {inlet_func(test_x, test_y, test_z)}")
        print(f"  Outlet mask: {outlet_func(test_x, test_y, test_z)}")
        print(f"  Combined mask: {combined_func(test_x, test_y, test_z)}")
        
        print("✅ Mask functions validated successfully!")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        print("   Fix the mask file paths and try again.")
        raise 