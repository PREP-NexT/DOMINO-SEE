"""
Utility functions for the dominosee package.
"""

# Import specific functions from modules
from .blocking import process_blocks, combine_blocks

# Import from dims module
from .dims import *

# Make all modules available
from . import blocking
from . import dims