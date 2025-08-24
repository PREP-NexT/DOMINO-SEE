"""
Grid generation and manipulation utilities for climate data analysis.

This module provides various grid types for spatial analysis including:
- Regular grids (lat/lon)
- Gaussian grids
- Fibonacci sphere grids
- Fekete point grids

The grids are designed to work with climate data and support operations like
regridding, spatial sampling, and network analysis.
"""

# Import main grid classes
from .grid import (
    BaseGrid,
    RegularGrid,
    GaussianGrid,
    FibonacciGrid,
    FeketeGrid,
)

# Import utility functions from grid module
from .grid import (
    grid_num_to_distance,
    distance_to_grid_num,
    cart_to_geo,
    geo_to_cart,
    geo_distance,
    deg_to_eq_spacing,
    eq_spacing_to_deg,
    neighbour_distance,
)

# Note: fekete functions are available via .fekete module but not exposed at package level

# Define what gets imported with "from dominosee.grid import *"
__all__ = [
    # Main grid classes
    'BaseGrid',
    'RegularGrid', 
    'GaussianGrid',
    'FibonacciGrid',
    'FeketeGrid',
    # Utility functions
    'grid_num_to_distance',
    'distance_to_grid_num',
    'cart_to_geo',
    'geo_to_cart',
    'geo_distance',
    'deg_to_eq_spacing',
    'eq_spacing_to_deg',
    'neighbour_distance',
]
