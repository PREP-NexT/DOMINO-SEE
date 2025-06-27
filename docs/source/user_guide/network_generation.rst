Network Generation
==================

This guide explains how to generate networks from event data using dominosee.

Introduction
------------

dominosee provides tools to build spatial networks from event series among spatial locations and multiple types of climate extreme events. These networks capture the interconnectedness of hydroclimatic events across different spatial and temporal scales.

Building Basic Networks
----------------------

To generate a network from event data, you first need to prepare your data appropriately:

.. code-block:: python

    import numpy as np
    import xarray as xr
    import dominosee as ds
    
    # Example: Generate a simple network from climate events
    # This assumes you have functions like create_network in your package
    
    # Create sample event data (time, location)
    n_times = 100
    n_locations = 20
    
    # Generate random event occurrences (0 = no event, 1 = event)
    events = np.random.choice([0, 1], size=(n_times, n_locations), p=[0.9, 0.1])
    
    # Convert to xarray for better labeling
    locs = [f'loc_{i}' for i in range(n_locations)]
    times = np.arange(n_times)
    
    event_da = xr.DataArray(
        events,
        dims=('time', 'location'),
        coords={
            'time': times,
            'location': locs
        }
    )
    
    # Generate network with a specified threshold
    # Note: this is example code and should be adjusted based on your actual API
    network = ds.create_network(event_da, threshold=0.5)
    
    # The resulting network shows connections between locations
    # where events occur with similar patterns

Multi-layer Networks
-------------------

For analyzing multiple types of extreme events, dominosee supports multi-layer network generation:

.. code-block:: python

    # Create sample multi-event data (time, event_type, location)
    n_event_types = 3
    multi_events = np.random.choice(
        [0, 1], 
        size=(n_times, n_event_types, n_locations), 
        p=[0.9, 0.1]
    )
    
    # Convert to xarray
    event_types = ['temperature', 'precipitation', 'wind']
    
    multi_event_da = xr.DataArray(
        multi_events,
        dims=('time', 'event_type', 'location'),
        coords={
            'time': times,
            'event_type': event_types,
            'location': locs
        }
    )
    
    # Generate multi-layer network
    # Note: this is example code and should be adjusted based on your actual API
    multi_network = ds.create_multilayer_network(
        multi_event_da, 
        threshold=0.5,
        layer_dim='event_type'
    )
    
    # The resulting network contains layers for each event type
    # showing how different extreme events are interconnected

Network Parameters and Thresholds
-------------------------------

When generating networks, several parameters can be adjusted to control the network properties:

- **Threshold**: Determines the minimum strength of connection to include in the network
- **Window Size**: For time-windowed analysis of event correlations
- **Distance Weighting**: Option to incorporate spatial distance in connection strength
- **Significance Testing**: Statistical tests to ensure connections are significant

Example with parameter adjustments:

.. code-block:: python

    # Generate network with custom parameters
    # Note: this is example code and should be adjusted based on your actual API
    custom_network = ds.create_network(
        event_da,
        threshold=0.6,            # Higher threshold for stronger connections
        window_size=10,           # 10-time-step window for correlation
        distance_weighted=True,   # Consider spatial distance
        significance_level=0.05   # 95% confidence for connections
    )

Saving and Loading Networks
--------------------------

Networks generated with dominosee can be saved for later analysis or visualization:

.. code-block:: python

    # Save network to file
    # Note: this is example code and should be adjusted based on your actual API
    ds.save_network(network, 'my_network.nc')
    
    # Load network from file
    loaded_network = ds.load_network('my_network.nc')
