Event Analysis
=============

This guide explains how to use dominosee for analyzing time series events.

Introduction
------------

Event analysis is a fundamental part of understanding hydroclimatic phenomena. dominosee provides tools to select, analyze, and visualize event data across temporal and spatial dimensions.

Basic Event Selection
--------------------

For numpy array-based data, the ``select_first_period`` function allows you to extract the initial phases of events based on their duration:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from dominosee import select_first_period
    
    # Create sample data (20 time steps, 5 events)
    events = np.random.randn(20, 5)
    
    # Define event durations
    durations = np.array([5, 10, 7, 12, 6])
    
    # Select the first 3 days of each event
    first_period = select_first_period(events, durations, days=3)
    
    # Visualize the results
    plt.figure(figsize=(10, 6))
    for i in range(5):
        plt.plot(first_period[:, i], label=f'Event {i+1}')
    plt.legend()
    plt.title('First 3 Days of Events')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

Working with Multidimensional Data
--------------------------------

For gridded climate data and other multidimensional datasets, dominosee provides xarray-compatible functions that maintain dimension information:

.. code-block:: python

    import xarray as xr
    import pandas as pd
    from dominosee import select_first_period_xr
    
    # Create a sample xarray DataArray with dimensions (time, event, location)
    times = pd.date_range('2025-01-01', periods=30)
    data = np.random.randn(30, 4, 3)
    
    da = xr.DataArray(
        data,
        dims=('time', 'event', 'location'),
        coords={
            'time': times,
            'event': np.arange(4),
            'location': ['A', 'B', 'C']
        }
    )
    
    # Define durations as an xarray DataArray
    durations = xr.DataArray(
        np.array([7, 12, 5, 10]),
        dims=('event'),
        coords={'event': np.arange(4)}
    )
    
    # Select the first 5 days of each event
    result = select_first_period_xr(da, durations, days=5)
    
    # The result maintains all the dimensions and coordinates of the original DataArray
    print(result.dims)  # ('time', 'event', 'location')
    
    # Plot the results for one location
    result.sel(location='A').plot.line(x='time', hue='event')
    plt.title('First 5 Days of Events at Location A')
    plt.grid(True)
    plt.show()

Handling Edge Cases
------------------

The event selection functions in dominosee are designed to handle various edge cases:

1. **Events shorter than the requested period**: If an event's duration is less than the requested number of days, the entire event is returned.

2. **Missing values**: NaN values in the input data are preserved in the output.

3. **Zero-duration events**: These are handled gracefully by returning empty data.

Example with mixed durations:

.. code-block:: python

    # Events with varying durations
    events = np.random.randn(20, 4)
    durations = np.array([2, 10, 1, 5])
    
    # Select first 4 days
    result = select_first_period(events, durations, days=4)
    
    # Result will contain:
    # - All data for event 0 (2 days)
    # - First 4 days for event 1 (10 days)
    # - All data for event 2 (1 day)
    # - First 4 days for event 3 (5 days)

Using dask for Large Datasets
----------------------------

When working with large datasets, you can leverage dask through the xarray integration:

.. code-block:: python

    import dask.array as da
    
    # Create a large dask array
    dask_data = da.random.random((1000, 20, 50), chunks=(100, 5, 10))
    
    # Convert to xarray with dask backend
    dask_da = xr.DataArray(
        dask_data,
        dims=('time', 'event', 'location'),
        coords={
            'time': pd.date_range('2025-01-01', periods=1000),
            'event': np.arange(20),
            'location': np.arange(50)
        }
    )
    
    # Create duration array
    durations = xr.DataArray(
        np.random.randint(5, 30, size=20),
        dims=('event'),
        coords={'event': np.arange(20)}
    )
    
    # Apply function with dask backend
    result = select_first_period_xr(dask_da, durations, days=7)
    
    # Operations are computed lazily until you request results
    computed_result = result.compute()
