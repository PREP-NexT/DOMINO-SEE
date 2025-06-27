"""
Block-wise processing utilities for handling large datasets.

These functions enable splitting large-scale computations into manageable
blocks, particularly useful for Event Coincidence Analysis with very large
spatial dimensions.
"""

import os
import glob
import numpy as np
import xarray as xr

# Make tqdm optional
try:
    from tqdm import tqdm
except ImportError:
    # Create a simple replacement if tqdm is not available
    def tqdm(iterable, **kwargs):
        desc = kwargs.get('desc', '')
        if desc:
            print(f"{desc}...")
        return iterable


def process_blocks(func, *args, block_size=1000, output_dir=None, output_pattern=None, **kwargs):
    """
    Apply a function to blocks of data and optionally save the results.
    
    This is designed for functions that operate on spatial data where the 
    full computation would exceed memory constraints.
    
    Parameters
    ----------
    func : callable
        The function to apply to each block. Should accept xarray DataArrays as inputs.
    *args : xarray.DataArray
        The input data arrays. The function assumes the first dimension of each array
        is the spatial dimension that needs to be blocked.
    block_size : int, default=1000
        Size of blocks to process.
    output_dir : str, optional
        Directory to save block files. If None, results are returned as a list.
    output_pattern : str, optional
        Pattern for output filenames, should include {i} and {j} for block indices.
        Default: "block_{i}_{j}.nc"
    **kwargs :
        Additional keyword arguments to pass to func.
    
    Returns
    -------
    list or None
        If output_dir is None, returns a list of block results.
        Otherwise, returns None (results are saved to files).
    """
    
    # Verify inputs are xarray DataArrays
    for i, arg in enumerate(args):
        if not isinstance(arg, xr.DataArray):
            raise TypeError(f"Argument {i} must be an xarray DataArray, got {type(arg)}")
    
    # Setup output directory if provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        if output_pattern is None:
            output_pattern = "block_{i}_{j}.nc"
    
    # Get dimensions for blocking
    dims = [np.setdiff1d(arg.dims, ["time"])[0] for arg in args]
    sizes = [arg.sizes[dim] for arg, dim in zip(args, dims)]
    
    # Calculate blocks
    block_counts = [int(np.ceil(size / block_size)) for size in sizes]
    
    # Process blocks
    results = []
    for i in tqdm(range(block_counts[0]), desc="Processing blocks"):
        start_i = i * block_size
        end_i = min((i + 1) * block_size, sizes[0])
        
        # Get the first block
        block_args = [args[0].isel({dims[0]: slice(start_i, end_i)})]
        
        # For second dimension, if applicable (e.g., for precursor/trigger calculations)
        if len(args) > 1:
            for j in range(block_counts[1]):
                start_j = j * block_size
                end_j = min((j + 1) * block_size, sizes[1])
                
                # Get the second block
                block_args_j = block_args + [args[1].isel({dims[1]: slice(start_j, end_j)})]
                
                # Apply function to block
                block_result = func(*block_args_j, **kwargs)
                
                # Save block if output_dir provided
                if output_dir is not None:
                    block_filename = os.path.join(output_dir, output_pattern.format(i=i, j=j))
                    block_result.to_netcdf(block_filename)
                else:
                    results.append((i, j, block_result))
        else:
            # Single input case
            block_result = func(*block_args, **kwargs)
            
            # Save block if output_dir provided
            if output_dir is not None:
                block_filename = os.path.join(output_dir, output_pattern.format(i=i, j=0))
                block_result.to_netcdf(block_filename)
            else:
                results.append((i, 0, block_result))
    
    # Return results if not saving to files
    if output_dir is None:
        return results
    
    return None


def combine_blocks(input_pattern, region=None, combine_method='by_coords'):
    """
    Combine block files into a single dataset.
    
    Parameters
    ----------
    input_pattern : str
        Glob pattern to match block files (e.g., "/path/to/blocks/block_*.nc")
    region : dict, optional
        Region to select from the combined dataset, e.g., {'lat': slice(0, 10)}
    combine_method : str, default='by_coords'
        Method to combine datasets, passed to xarray.open_mfdataset
    
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Combined dataset from all blocks
    """
    # List all block files
    block_files = sorted(glob.glob(input_pattern))
    
    if not block_files:
        raise FileNotFoundError(f"No block files found matching pattern: {input_pattern}")
    
    # Open all datasets as a multi-file dataset
    ds = xr.open_mfdataset(block_files, combine=combine_method)
    
    # Apply region selection if provided
    if region is not None:
        ds = ds.sel(**region)
    
    # Convert to DataArray if the result contains only one variable
    if len(ds.data_vars) == 1:
        var_name = list(ds.data_vars)[0]
        return ds[var_name]
    
    return ds


def process_eca_blockwise(func, eventA, eventB, output_dir, block_size=1000, output_pattern="eca_block_{i}_{j}.nc", **kwargs):
    """
    Process Event Coincidence Analysis in blocks for very large datasets.
    
    This is a specialized wrapper around process_blocks for ECA functions.
    
    Parameters
    ----------
    func : callable
        The ECA function to apply (e.g., get_eca_precursor_from_events)
    eventA : xarray.DataArray
        Binary event time series at location A
    eventB : xarray.DataArray
        Binary event time series at location B
    output_dir : str
        Directory to save block files
    block_size : int, default=1000
        Size of blocks to process
    output_pattern : str, default="eca_block_{i}_{j}.nc"
        Pattern for output filenames, with {i} and {j} as block indices
    **kwargs :
        Additional arguments to pass to the ECA function
        
    Returns
    -------
    None
        Results are saved to files in output_dir
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process blocks - let the ECA function handle dimension naming
    return process_blocks(
        func, 
        eventA, 
        eventB, 
        block_size=block_size, 
        output_dir=output_dir,
        output_pattern=output_pattern,
        **kwargs
    ) 