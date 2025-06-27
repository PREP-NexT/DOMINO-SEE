import json
import numpy as np
from scipy.stats import binom
from numba import njit, prange, vectorize, guvectorize
import xarray as xr

@guvectorize(["void(boolean[:], boolean[:], uint16[:])"], "(n),(n)->()", nopython=True)
def _eca_precursor(b1w, b2, KRprec):
    KRprec[0] = np.sum(b1w & b2)

@guvectorize(["void(boolean[:], boolean[:], uint16[:])"], "(n),(n)->()", nopython=True)
def _eca_trigger(b1, b2wr, KRtrig):
    KRtrig[0] = np.sum(b1 & b2wr)

@njit(parallel=True)
def _eca_precursors_pair_njit(b1w, b2):
    result = np.zeros((b2.shape[0], b1w.shape[0]), dtype=np.uint16)
    m = b2.shape[0]     # 第一数组的第一维
    l = b1w.shape[0]    # 第二数组的第一维
    
    for i in prange(m):
        for j in range(l):
            result[i, j] = np.sum(b2[i, :] & b1w[j, :])
    return result

@njit(parallel=True)
def _eca_triggers_pair_njit(b1, b2wr):
    result = np.zeros((b1.shape[0], b2wr.shape[0]), dtype=np.uint16)
    m = b1.shape[0]      # 第一数组的第一维
    l = b2wr.shape[0]    # 第二数组的第一维
    
    for i in prange(m):
        for j in range(l):
            result[i, j] = np.sum(b1[i, :] & b2wr[j, :])
    return result


def _forward_window(time_series, delt=2, sym=True, tau=0):
    """
    Create a forward (precursor) window for Event Coincidence Analysis.
    
    Parameters
    ----------
    time_series : ndarray
        1D array representing a binary event series (0s and 1s)
    delt : int, default=2
        Size of the ECA window
    sym : bool, default=True
        Whether the ECA window is symmetric
    tau : int, default=0
        Delay parameter for the window (must be non-negative)
        
    Returns
    -------
    ndarray
        Binary array indicating presence of events in forward window
        
    Notes
    -----
    This function creates a window used for calculating precursors in ECA.
    It performs a convolution operation followed by thresholding.
    """
    # Input validation
    time_series = np.asarray(time_series)
    if time_series.ndim != 1:
        raise ValueError("time_series must be a 1D array")
    if tau < 0:
        raise ValueError("delay must be non-negative")
    if delt < 0:
        raise ValueError("delt must be non-negative")
        
    # Create window
    window = np.ones((1 + 1 * sym) * delt + 1)
    
    # Apply window
    if delt == 0:
        result = time_series.copy()
    else:
        result = (np.convolve(time_series, window)[sym*delt:-delt] >= 0.5)
    
    # Apply delay
    if tau > 0:
        result = np.roll(result, tau)
        result[:tau] = False
        
    return result

def _backward_window(time_series, delt=2, sym=True, tau=0):
    """
    Create a backward (trigger) window for Event Coincidence Analysis.
    
    Parameters
    ----------
    time_series : ndarray
        1D array representing a binary event series (0s and 1s)
    delt : int, default=2
        Size of the ECA window
    sym : bool, default=True
        Whether the ECA window is symmetric
    tau : int, default=0
        Delay parameter for the window (must be non-negative)
        
    Returns
    -------
    ndarray
        Binary array indicating presence of events in backward window
        
    Notes
    -----
    This function creates a window used for calculating triggers in ECA.
    If symmetric=True, it's the same as forward_window, otherwise it uses
    a different slice of the convolution.
    """
    # Input validation
    time_series = np.asarray(time_series)
    if time_series.ndim != 1:
        raise ValueError("time_series must be a 1D array")
    if tau < 0:
        raise ValueError("tau must be non-negative")
    if delt < 0:
        raise ValueError("delt must be non-negative")
    
    # If symmetric, we can reuse the forward window logic
    if sym:
        result = _forward_window(time_series, delt, sym, tau)
    else:
        # Create window
        window = np.ones(delt + 1)
        
        # Apply window with different slicing
        result = (np.convolve(time_series, window)[delt:] >= 0.5)
    
    # Apply delay in opposite direction for backward window
    if tau > 0:
        result = np.roll(result, -tau)
        result[-tau:] = False
        
    return result

# Keep the original function for backward compatibility
def eca_window_legacy(b, delt=2, sym=True, tau=0):
    """Legacy version of eca_window function for backward compatibility."""
    return _forward_window(b, delt, sym, tau), _backward_window(b, delt, sym, tau)


def get_eca_precursor_window(da: xr.DataArray, delt: int=2, sym: bool=True, tau: int=0) -> xr.DataArray:
    da_precursor_window = xr.apply_ufunc(_forward_window, da, input_core_dims=[["time"]], output_core_dims=[["time"]], 
                                        vectorize=True, dask="parallelized", kwargs={"delt": delt, "sym": sym, "tau": tau})
    da_precursor_window.attrs = {'long_name': 'Precursor Window', 'units': '1', 'description': 'Window for precursor event identification',
                            "eca_params": json.dumps({"delt": delt, "sym": sym, "tau": tau})}
    return da_precursor_window


def get_eca_trigger_window(da: xr.DataArray, delt: int=2, sym: bool=True, tau: int=0) -> xr.DataArray:
    da_trigger_window = xr.apply_ufunc(_backward_window, da, input_core_dims=[["time"]], output_core_dims=[["time"]], 
                                        vectorize=True, dask="parallelized", kwargs={"delt": delt, "sym": sym, "tau": tau})
    da_trigger_window.attrs = {'long_name': 'Trigger Window', 'units': '1', 'description': 'Window for trigger event identification',
                               "eca_params": json.dumps({"delt": delt, "sym": sym, "tau": tau})}
    return da_trigger_window


def get_eca_precursor(eventA_precursor_window: xr.DataArray, eventB: xr.DataArray, func="njit") -> xr.DataArray:
    # if long_name of eventA_precursor_window is not "Precursor Window", raise ValueError
    if eventA_precursor_window.attrs["long_name"] != "Precursor Window":
        raise ValueError("long_name of eventA_precursor_window must be 'Precursor Window'")
    else:
        eca_params = eventA_precursor_window.attrs["eca_params"]

    # append spatial dimension with location order A => B
    eventB = eventB.rename({var: f"{var}B" for var in np.setdiff1d(eventB.dims, "time")})
    spatial_dimB = np.setdiff1d(eventB.dims, "time")
    eventA_precursor_window = eventA_precursor_window.rename({var: f"{var}A" for var in np.setdiff1d(eventA_precursor_window.dims, "time")})
    spatial_dimA = np.setdiff1d(eventA_precursor_window.dims, "time")

    if func == "njit":
        da_precursor = xr.apply_ufunc(_eca_precursors_pair_njit, eventA_precursor_window, eventB, 
                                      vectorize=True,
                                      input_core_dims=[[spatial_dimA[-1], "time"], [spatial_dimB[-1], "time"]], 
                                      output_core_dims=[[spatial_dimA[-1], spatial_dimB[-1]]],
                                      dask="parallelized",
                                      dask_gufunc_kwargs={"allow_rechunk": True},
                                      output_dtypes=[np.uint16],
                                      )
    else:
        da_precursor = xr.apply_ufunc(_eca_precursor, eventA_precursor_window, eventB, 
                                      vectorize=False,
                                      input_core_dims=[["time"], ["time"]], output_core_dims=[[]],  
                                      dask="parallelized",)
    da_precursor.attrs = {'long_name': 'Precursor Events', 'units': 'count', 'description': 'Number of precursor events (from location A to location B) in location B',
                          "eca_params": eca_params}
    return da_precursor

def get_eca_trigger(eventA_trigger_window: xr.DataArray, eventB: xr.DataArray, func="njit") -> xr.DataArray:
    # if long_name of eventA_trigger_window is not "Trigger Window", raise ValueError
    if eventA_trigger_window.attrs["long_name"] != "Trigger Window":
        raise ValueError("long_name of eventA_trigger_window must be 'Trigger Window'")
    else:
        eca_params = eventA_trigger_window.attrs["eca_params"]

    # append spatial dimension with location order A => B
    eventB = eventB.rename({var: f"{var}B" for var in np.setdiff1d(eventB.dims, "time")})
    spatial_dimB = np.setdiff1d(eventB.dims, "time")
    eventA_trigger_window = eventA_trigger_window.rename({var: f"{var}A" for var in np.setdiff1d(eventA_trigger_window.dims, "time")})
    spatial_dimA = np.setdiff1d(eventA_trigger_window.dims, "time") 

    if func == "njit":
        da_trigger = xr.apply_ufunc(_eca_triggers_pair_njit, eventA_trigger_window, eventB, 
                                      vectorize=True,
                                      input_core_dims=[[spatial_dimA[-1], "time"], [spatial_dimB[-1], "time"]], 
                                      output_core_dims=[[spatial_dimA[-1], spatial_dimB[-1]]],
                                      dask="parallelized",
                                      dask_gufunc_kwargs={"allow_rechunk": True},
                                      output_dtypes=[np.uint16],
                                      )
    else:
        da_trigger = xr.apply_ufunc(_eca_trigger, eventA_trigger_window, eventB, 
                                      vectorize=False,
                                      input_core_dims=[["time"], ["time"]], output_core_dims=[[]],  
                                      dask="parallelized",) 
    da_trigger.attrs = {'long_name': 'Trigger Events', 'units': 'count', 'description': 'Number of trigger events (from location A to location B) in location A',
                        "eca_params": eca_params}
    return da_trigger

def get_eca_precursor_from_events(eventA: xr.DataArray, eventB: xr.DataArray, delt: int=2, sym: bool=True, tau: int=0,
                                  func="njit") -> xr.DataArray:
    """
    Calculate precursor events from eventA (events in location A) to eventB (events in location B) based on event coincidence analysis.

    Parameters
    ----------
    eventA : xr.DataArray
        Binary event time series at location A
    eventB : xr.DataArray
        Binary event time series at location B
    delt : int, optional
        Length of the coincidence window, by default 2
    sym : bool, optional
        If True, use symmetric window, by default True
    tau : int, optional
        Time lag from eventA to eventB, by default 0
    func : str, optional
        Function to use for calculation - "njit" for numba optimized, or "guvectorize" for gufunc, by default "njit"

    Returns
    -------
    xr.DataArray
        Number of precursor events from location A to location B
    """
    # get precursor window
    eventA_precursor_window = get_eca_precursor_window(eventA, delt, sym, tau)
    # get precursor
    da_precursor = get_eca_precursor(eventA_precursor_window, eventB, func)
    return da_precursor

def get_eca_trigger_from_events(eventA: xr.DataArray, eventB: xr.DataArray, delt: int=2, sym: bool=True, tau: int=0,
                                func="njit") -> xr.DataArray:
    """
    Calculate trigger events from eventA (events in location A) to eventB (events in location B) based on event coincidence analysis.

    Parameters
    ----------
    eventA : xr.DataArray
        Binary event time series at location A
    eventB : xr.DataArray
        Binary event time series at location B
    delt : int, optional
        Length of the coincidence window, by default 2
    sym : bool, optional
        If True, use symmetric window, by default True
    tau : int, optional
        Time lag from eventA to eventB, by default 0
    func : str, optional
        Function to use for calculation - "njit" for numba optimized, or "guvectorize" for gufunc, by default "njit"
    """
    # get trigger window
    eventA_trigger_window = get_eca_trigger_window(eventA, delt, sym, tau)
    # get trigger
    da_trigger = get_eca_trigger(eventA_trigger_window, eventB, func)
    return da_trigger

def get_eca_precursor_confidence(precursor: xr.DataArray, eventA: xr.DataArray, eventB: xr.DataArray, 
                                  min_eventnum: int=2) -> xr.DataArray:
    # get eca_params from precursor
    eca_params = json.loads(precursor.attrs["eca_params"])
    TOL = eca_params["delt"] * eca_params["sym"] + 1
    tau = eca_params["tau"]
    T = eventA.sizes["time"] # examine 

    # get NA, NB
    NA = eventA.sum(dim="time").rename({f"{var}": f"{var}A" for var in np.setdiff1d(eventA.dims, "time")})
    NB = eventB.sum(dim="time").rename({f"{var}": f"{var}B" for var in np.setdiff1d(eventB.dims, "time")})
    # assert NA, NB are in the coordinates of precursor
    assert np.all(np.isin(NA.dims, precursor.dims))
    assert np.all(np.isin(NB.dims, precursor.dims))

    prec_conf = xr.apply_ufunc(prec_confidence, 
                              precursor,
                              NA, NB,
                              dask="parallelized", kwargs={"TOL": TOL, "T": T, "tau": tau}).rename("prec_conf")
    
    if min_eventnum > 0:
        prec_conf = prec_conf.where(NA >= min_eventnum, 0.0)
        prec_conf = prec_conf.where(NB >= min_eventnum, 0.0)

    prec_conf.attrs = {'long_name': 'Precursor confidence', 'units': "", 'description': 'Confidence of precursor rates (from location A to location B) in location B',
                      "eca_params": precursor.attrs["eca_params"]}
    return prec_conf


def get_eca_trigger_confidence(trigger: xr.DataArray, eventA: xr.DataArray, eventB: xr.DataArray, 
                               min_eventnum: int=2) -> xr.DataArray:
    # get eca_params from trigger
    eca_params = json.loads(trigger.attrs["eca_params"])
    TOL = eca_params["delt"] * eca_params["sym"] + 1
    tau = eca_params["tau"]
    T = eventA.sizes["time"] # examine 
    
    # get NA, NB
    NA = eventA.sum(dim="time").rename({f"{var}": f"{var}A" for var in np.setdiff1d(eventA.dims, "time")})
    NB = eventB.sum(dim="time").rename({f"{var}": f"{var}B" for var in np.setdiff1d(eventB.dims, "time")})

    # calculate confidence
    trigger_conf = xr.apply_ufunc(trig_confidence, 
                              trigger,
                              NA, NB,
                              dask="parallelized", kwargs={"TOL": TOL, "T": T, "tau": tau}).rename("trigger_conf")

    if min_eventnum > 0:
        trigger_conf = trigger_conf.where(NA >= min_eventnum, 0.0)
        trigger_conf = trigger_conf.where(NB >= min_eventnum, 0.0)

    trigger_conf.attrs = {'long_name': 'Trigger confidence', 'units': "", 'description': 'Confidence of trigger rates (from location A to location B) in location A',
                         "eca_params": trigger.attrs["eca_params"]}
    return trigger_conf


"""
Legacy stack-style calculation
"""
# @njit(parallel=False)
# def eca(b1, b2, b1w, b2wr, dtype='uint16'):
#     # TODO: 拆分成precursor & trigger 两个函数；因为有时只需要一种
#     KRprec = np.zeros((b1.shape[0], b2.shape[0]), dtype=dtype)  # precursor rates
#     KRtrig = np.zeros((b1.shape[0], b2.shape[0]), dtype=dtype)  # triggering rates
#     for j in range(b1.shape[0]):
#         for k in range(b2.shape[0]): # TODO: 规范化代码时考虑只在这一层使用多核，对比一下速度；避免MPI，从而避免稀疏要求
#             KRprec[j, k] = np.sum(b2[k, :] & b1w[j, :])   # precursor: b1   => (b2)  
#             KRtrig[j, k] = np.sum(b1[j, :] & b2wr[k, :])  # trigger: (b1) => b2 
#     return KRprec, KRtrig


# @njit(parallel=True)
# def eca_parallel(b1, b2, b1w, b2wr, dtype='uint16'):
#     KRprec = np.zeros((b1.shape[0], b2.shape[0]), dtype=dtype)  # precursor rates
#     KRtrig = np.zeros((b1.shape[0], b2.shape[0]), dtype=dtype)  # triggering rates
#     for j in prange(b1.shape[0]):
#         for k in range(b2.shape[0]): # TODO: 规范化代码时考虑只在这一层使用多核，对比一下速度；避免MPI，从而避免稀疏要求
#             KRprec[j, k] = np.sum(b2[k, :] & b1w[j, :])   # b1   => (b2)  
#             KRtrig[j, k] = np.sum(b1[j, :] & b2wr[k, :])  # (b1) => b2    
#     return KRprec, KRtrig


# def eca_dataset(b1: xr.DataArray, b2: xr.DataArray, b1w: xr.DataArray, b2wr: xr.DataArray, dtype=None, parallel=True):
#     # TODO: 修正loc A/B的命名
#     # infer dtype based on the length of time dimension
#     if dtype is None:
#         dtype = np.uint8 if b1.shape[0] < 256 else np.uint16 if b1.shape[0] < 65536 else np.uint32  # Cannot use string here in numba
#     # get the location name from dims
#     xdim = np.setdiff1d(b1.dims, "time")[0]
#     layernames = [f"{b1.name}_{xdim}A", f"{b2.name}_{xdim}B"]

#     # make sure all DataArray is in ("location", "time") coordinate order if not
#     if b1.dims[0] != 'location':
#         b1 = b1.transpose('location', 'time')
#     if b2.dims[0] != 'location':
#         b2 = b2.transpose('location', 'time')
#     if b1w.dims[0] != 'location':
#         b1w = b1w.transpose('location', 'time')
#     if b2wr.dims[0] != 'location':
#         b2wr = b2wr.transpose('location', 'time')
#     # calculate the ECA
#     if parallel:
#         ECRprec, ECRtrig = eca_parallel(b1.values, b2.values, b1w.values, b2wr.values, dtype=dtype)
#     else:
#         ECRprec, ECRtrig = eca(b1.values, b2.values, b1w.values, b2wr.values, dtype=dtype)
#     # create DataArray
#     coords_locA = b1.indexes['location'].rename(["lat_locA", "lon_locA"])  # 这里一定不能用b1.coords['location'] rename，因为不会作用于MultiIndex
#     coords_locB = b2.indexes['location'].rename(["lat_locB", "lon_locB"])
#     ECRprec = xr.DataArray(ECRprec, coords=[coords_locA, coords_locB], dims=layernames, name="prec_evt",
#                            attrs={'long_name': 'Precursor Events', 'units': 'count', 'dtype': dtype.__name__, 
#                                   'description': 'Number of precursor events (from location A to location B) in location B',
#                                   'eca_params': b1w.attrs["eca_params"]})
#     ECRtrig = xr.DataArray(ECRtrig, coords=[coords_locA, coords_locB], dims=layernames, name="trig_evt",
#                            attrs={'long_name': 'Trigger Events', 'units': 'count', 'dtype': dtype.__name__, 
#                                   'description': 'Number of trigger events (from location A to location B) in location A',
#                                   'eca_params': b2wr.attrs["eca_params"]})

#     return ECRprec, ECRtrig


# def eca_window(b, delt=2, sym=True, tau=0):
#     """
#     b: 一维向量，表示时间序列
#     delt: ECA窗口的大小
#     sym: ECA窗口是否对称
#     tau: ECA窗口的延迟

#     return: 
#     bw: 用于计算precursor的窗口
#     bwr: 用于计算trigger的窗口

#     note:
#     这里窗口的计算服务于后续与之对应的点的ECA计算, 因此此处不是反映precursor/trigger的对应点, 而是另外一侧
#     """
#     if tau < 0:
#         raise ValueError("tau must be non-negative")
#     window = np.ones((1 + 1 * sym) * delt + 1)
#     # 用于precursor计算的窗口
#     if delt == 0:
#         bw = b  #.copy()
#     else:
#         # bw = np.apply_along_axis(lambda x: np.convolve(x, window)[sym*delt:-delt] >= 0.5, 1, b)
#         bw = (np.convolve(b, window)[sym*delt:-delt] >= 0.5)
#     if tau > 0:
#         bw = np.roll(bw, tau)
#         bw[:, :tau] = False

#     # 用于trigger计算的窗口
#     if sym:
#         bwr = bw.copy()
#     else:
#         bwr = np.convolve(b, window)[delt:] >= 0.5
#     if tau > 0:
#         bwr = np.roll(bwr, -tau)
#         bwr[:, -tau:] = False

#     return bw, bwr


# def get_eca_window(da: xr.DataArray, delt: int=2, sym: bool=True, tau: int=0) -> xr.DataArray:
#     eca_params = {'delt': delt, 'sym': sym, 'tau': tau}
#     da_prec_window, da_trig_window = xr.apply_ufunc(eca_window, da, input_core_dims=[["time"]], output_core_dims=[["time"], ["time"]], 
#                                         vectorize=True, dask="parallelized", kwargs=eca_params)
#     da_prec_window.attrs = {'long_name': 'Precursor Window', 'units': '1', 'description': 'Window for precursor event identification',
#                             "eca_params": json.dumps(eca_params)}
#     da_trig_window.attrs = {'long_name': 'Trigger Window', 'units': '1', 'description': 'Window for trigger event identification', 
#                             "eca_params": json.dumps(eca_params)}
#     return da_prec_window, da_trig_window


"""
Confidence calculation
"""
def prec_confidence(kp, na, nb, TOL, T, tau):
    return binom.cdf(kp, n=nb, p=1-(1-TOL/(T-tau))**na).astype(np.float32) #不需要reshape，xarray会自动broadcast


def trig_confidence(kt, na, nb, TOL, T, tau):
    return binom.cdf(kt, n=na, p=1-(1-TOL/(T-tau))**nb).astype(np.float32)


def get_prec_confidence(da_prec: xr.DataArray, da_layerA: xr.DataArray, da_layerB: xr.DataArray, 
                          min_eventnum: int=2, time_period=None) -> xr.DataArray:
    layer_locA = da_prec.dims[0].split("_")[0]
    layer_locB = da_prec.dims[1].split("_")[0]

    da_evN_locA = da_layerA.sum(dim="time").rename({"location": f"{layer_locA}_locationA"}).rename({"lat": "lat_locA", "lon": "lon_locA"})
    da_evN_locA.indexes[f"{layer_locA}_locationA"].rename({"lat": "lat_locA", "lon": "lon_locA"}, inplace=True)

    da_evN_locB = da_layerB.sum(dim="time").rename({"location": f"{layer_locB}_locationB"}).rename({"lat": "lat_locB", "lon": "lon_locB"})
    da_evN_locB.indexes[f"{layer_locB}_locationB"].rename({"lat": "lat_locB", "lon": "lon_locB"}, inplace=True)

    eca_params = json.loads(da_prec.attrs["eca_params"])
    TOL = eca_params["delt"] * eca_params["sym"] + 1
    tau = eca_params["tau"]  
    T = da_layerA.sizes["time"] # examine 
    # print(f"Calculating confidence with TOL={TOL}, tau={tau}; T={T}")

    prec_sig = xr.apply_ufunc(prec_confidence, 
                              da_prec,
                              da_evN_locA, da_evN_locB,
                              dask="parallelized", kwargs={"TOL": TOL, "T": T, "tau": tau}).rename("prec_sig")
    # prec_sig = prec_sig.compute() # for debug
    
    if min_eventnum > 0:
        prec_sig = prec_sig.where(da_evN_locA >= min_eventnum, 0.0)
        prec_sig = prec_sig.where(da_evN_locB >= min_eventnum, 0.0)

    prec_sig.attrs = {'long_name': 'Precursor confidence', 'units': "", 'description': 'Confidence of precursor rates (from location A to location B) in location B',
                      "eca_params": da_prec.attrs["eca_params"]}
    return prec_sig


def get_trig_confidence(da_trig: xr.DataArray, da_layerA: xr.DataArray, da_layerB: xr.DataArray, 
                          min_eventnum: int=2, time_period=None) -> xr.DataArray:
    layer_locA = da_trig.dims[0].split("_")[0]
    layer_locB = da_trig.dims[1].split("_")[0]

    da_evN_locA = da_layerA.sum(dim="time").rename({"location": f"{layer_locA}_locationA"}).rename({"lat": "lat_locA", "lon": "lon_locA"})
    da_evN_locA.indexes[f"{layer_locA}_locationA"].rename({"lat": "lat_locA", "lon": "lon_locA"}, inplace=True)

    da_evN_locB = da_layerB.sum(dim="time").rename({"location": f"{layer_locB}_locationB"}).rename({"lat": "lat_locB", "lon": "lon_locB"})
    da_evN_locB.indexes[f"{layer_locB}_locationB"].rename({"lat": "lat_locB", "lon": "lon_locB"}, inplace=True)

    eca_params = json.loads(da_trig.attrs["eca_params"])
    TOL = eca_params["delt"] * eca_params["sym"] + 1
    tau = eca_params["tau"]  
    T = da_layerA.sizes["time"] # examine 
    # print(f"Calculating confidence with TOL={TOL}, tau={tau}; T={T}")

    trig_sig = xr.apply_ufunc(trig_confidence, 
                              da_trig,
                              da_evN_locA, da_evN_locB,
                              dask="parallelized", kwargs={"TOL": TOL, "T": T, "tau": tau}).rename("trig_sig")
    # trig_sig = trig_sig.compute() # for debug
    
    if min_eventnum > 0:
        trig_sig = trig_sig.where(da_evN_locA >= min_eventnum, 0.0)
        trig_sig = trig_sig.where(da_evN_locB >= min_eventnum, 0.0)

    trig_sig.attrs = {'long_name': 'Trigger confidence', 'units': "", 'description': 'Confidence of trigger rates (from location A to location B) in location A',
                      "eca_params": da_trig.attrs["eca_params"]}
    return trig_sig
    
