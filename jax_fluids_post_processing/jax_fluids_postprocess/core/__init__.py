"""Core processing functionality for JAX-Fluids post-processing."""

from .processor import FluidProcessor
from .data_reader import DataReader

__all__ = ["FluidProcessor", "DataReader"]