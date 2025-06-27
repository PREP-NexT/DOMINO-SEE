import xarray as xr
import pandas as pd
import numpy as np

"""
Dimension rename
"""
def rename_dimensions(xr_obj, suffix='', keep_dims=None):
    """
    Rename dimensions in an xarray object, adding a suffix to each dimension name.
    Handles both standard dimensions and stacked dimensions with MultiIndex.
    
    Parameters
    ----------
    xr_obj : xr.DataArray or xr.Dataset
        The xarray object whose dimensions to rename
    suffix : str, optional
        Suffix to add to dimension names, by default ''
    keep_dims : list, optional
        List of dimension names to keep unchanged, by default None
        
    Returns
    -------
    tuple
        (renamed_obj, spatial_dims)
        renamed_obj: The xarray object with renamed dimensions
        spatial_dims: List of the renamed spatial dimensions
    """
    if keep_dims is None:
        keep_dims = []
    
    # Get all dimensions to rename (excluding those in keep_dims)
    dims_to_rename = [dim for dim in list(xr_obj.dims) if dim not in keep_dims]
    
    # Dictionary to store dimension renames
    rename_dict = {}
    
    # Process each dimension
    for dim in dims_to_rename:
        # Check if it's a stacked dimension (has a MultiIndex)
        if isinstance(xr_obj.indexes.get(dim), pd.MultiIndex):
            # Get the original dimensions before stacking
            original_dims = xr_obj[dim].attrs.get('stacked_dim_names', [])
            if not original_dims:  # Fallback if attrs not available
                original_dims = list(xr_obj[dim].indexes[dim].names)
            
            # Add suffix to stacked dimension name
            rename_dict[dim] = f"{dim}{suffix}"
            
            # Add renamed original dimensions to the dict for reference
            for orig_dim in original_dims:
                rename_dict[orig_dim] = f"{orig_dim}{suffix}"
        else:
            # Regular dimension, just add suffix
            rename_dict[dim] = f"{dim}{suffix}"
    
    # Apply renaming
    renamed_obj = xr_obj.rename(rename_dict)
    
    # Get the renamed spatial dimensions
    spatial_dims = list(np.setdiff1d(list(renamed_obj.dims), keep_dims))
    
    return renamed_obj, spatial_dims


"""
Location flatten
"""
def stack_lonlat(da: xr.DataArray, stack_dims: list=None) -> xr.DataArray:
    """Stack the space dimensions into one dimension, which is needed for network construction

    Args:
        da (xr.DataArray): DataArray with multiple space dimensions
        stack_dims (list, optional): list of dimension names to be stacked. Defaults to None.

    Raises:
        ValueError: `stack_dims` should be list of dimension names if 
        lat/lon or latitude/longitude are not in dims

    Returns:
        xr.DataArray: DataArray with the stacked space dimension
    """   
    # TODO: location may not satisfy the CF-1.6; use "cell" instead?
    #                                            use "node" instead?
    # http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#appendix-cell-methods
    if stack_dims is None:
        if "lat" in da.dims and "lon" in da.dims:
            da_stack = da.stack(location=("lat", "lon"))
        elif "latitude" in da.dims and "longitude" in da.dims:
            da_stack = da.stack(location=("latitude", "longitude"))
        else:
            raise ValueError("stack_dims should not be None if lat/lon or latitude/longitude are not in dims")
    else:
        da_stack = da.stack(location=stack_dims)
    return da_stack


# def remove_nodes(ds: xr.Dataset, drop_nodes: xr.DataArray) -> xr.Dataset:
#     """"""
#     ds_wo_drop = ds
#     return ds_wo_drop

