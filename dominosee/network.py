import xarray as xr


"""
get link from threshold
"""
def get_link_from_threshold(da_sig: xr.DataArray, threshold: float) -> xr.DataArray:
    """
    Get the link from threshold DataArray

    Parameters:
    -----------
    da_sig : xr.DataArray
        DataArray containing the values to compare
    threshold : float
        Threshold value
    
    Returns:
    --------
    da_link : xr.DataArray
        Boolean DataArray where True indicates a link
    """
    da_link = da_sig >= threshold
    return da_link


"""
get link from significance
"""
def get_link_from_significance(da_sig: xr.DataArray, p_threshold: float) -> xr.DataArray:
    """
    Get the link from significance DataArray
    
    Parameters:
    -----------
    da_sig : xr.DataArray
        DataArray containing the values to compare
    p_threshold : float
        Significance level
    
    Returns:
    --------
    da_link : xr.DataArray
        Boolean DataArray where True indicates a link
    """
    # get link
    da_link = da_sig <= p_threshold
    return da_link

def get_link_from_confidence(da_conf: xr.DataArray, confidence_level: float) -> xr.DataArray:
    """
    Get the link from confidence DataArray
    
    Parameters:
    -----------
    da_conf : xr.DataArray
        DataArray containing the values to compare
    confidence_level : float
        Confidence level
    
    Returns:
    --------
    da_link : xr.DataArray
        Boolean DataArray where True indicates a link
    """
    # get link
    da_link = da_conf >= confidence_level
    return da_link


"""
get link from quantile
"""
def get_link_from_quantile(da_quant: xr.DataArray, q: float) -> xr.DataArray:
    """
    Get the link from quantile DataArray
    
    Parameters:
    -----------
    da_quant : xr.DataArray
        DataArray containing the values to compare
    q : float
        Quantile level
    
    Returns:
    --------
    da_link : xr.DataArray
        Boolean DataArray where True indicates a link
    """
    # get global quantile
    quant = da_quant.quantile(q)
    # get link
    da_link = da_quant >= quant
    return da_link


"""
get link from critical values
"""
def get_link_from_critical_values(da_valu: xr.DataArray, critical_value: xr.DataArray, rule="greater") -> xr.DataArray:
    """
    Get the link from critical values DataArray
    
    Parameters:
    -----------
    da_valu : xr.DataArray
        DataArray containing the values to compare
    critical_value : xr.DataArray
        DataArray containing the critical values
    rule : str, optional
        Comparison rule, either "greater" or "greater_equal" (default: "greater")

    Returns:
    --------
    xr.DataArray
        Boolean DataArray where True indicates a link
    """
    if rule == "greater":
        da_link = da_valu > critical_value
    elif rule == "greater_equal":
        da_link = da_valu >= critical_value
    else:
        raise ValueError("rule must be either 'greater' or 'greater_equal'")
    return da_link
