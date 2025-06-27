Quickstart
==========

This quickstart guide will help you get up and running with dominosee for analyzing interconnected hydroclimatic extreme events.

Basic Usage
----------

First, import the necessary packages:

.. code-block:: python

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import dominosee as ds

Event Analysis
-------------

One of the core functionalities of dominosee is selecting and analyzing event periods:

.. code-block:: python

    # Create sample event data
    events = np.random.randn(100, 10)  # 100 time steps, 10 events
    duration = np.arange(1, 11)        # Duration of each event
    
    # Select the first few days of events based on duration
    first_period = ds.select_first_period(events, duration, days=5)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(first_period)
    plt.title('First 5 Days of Events')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

Working with xarray DataArrays
-----------------------------

For multidimensional gridded climate data, dominosee provides xarray-compatible functions:

.. code-block:: python

    # Create a sample xarray DataArray
    times = pd.date_range('2025-01-01', periods=100)
    events = np.random.randn(100, 5, 3)  # time, event, location
    
    da = xr.DataArray(
        events,
        dims=('time', 'event', 'location'),
        coords={
            'time': times,
            'event': np.arange(5),
            'location': ['A', 'B', 'C']
        }
    )
    
    # Create duration array
    duration = xr.DataArray(
        np.array([3, 5, 2, 7, 4]),
        dims=('event'),
        coords={'event': np.arange(5)}
    )
    
    # Select first period using xarray function
    first_period_xr = ds.select_first_period_xr(da, duration, days=3)
    
    # Plot results for one location
    first_period_xr.sel(location='A').plot.line(x='time')
    plt.title('First 3 Days of Events at Location A')
    plt.grid(True)
    plt.show()

Network Analysis
--------------

dominosee can generate and analyze networks from event data:

.. code-block:: python

    # Generate a sample network from event data
    # This is a simplified example
    network = ds.generate_network(da, threshold=0.5)
    
    # Analyze network properties
    centrality = ds.calculate_centrality(network)
    
    # Visualize the network
    ds.plot_network(network, centrality)

Next Steps
----------

To dive deeper into dominosee:

- Explore the :doc:`user_guide/index` for detailed explanations
- Check out the :doc:`examples/index` for practical examples
- Refer to the :doc:`api/index` for complete function documentation
