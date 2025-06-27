Advanced Usage
=============

This guide covers advanced usage patterns for dominosee, including performance optimization and working with large datasets.

Using dask for Parallel Processing
--------------------------------

dominosee integrates with dask to enable efficient parallel processing of large datasets:

.. code-block:: python

    import dask.array as da
    import xarray as xr
    import numpy as np
    import dominosee as ds
    
    # Create a large dask array
    shape = (1000, 50, 100)  # time, event, location
    chunks = (100, 10, 20)   # chunk sizes for parallel processing
    
    # Generate random data as a dask array
    data = da.random.random(shape, chunks=chunks)
    
    # Convert to xarray DataArray with dask backend
    times = np.arange(shape[0])
    events = np.arange(shape[1])
    locations = [f'loc_{i}' for i in range(shape[2])]
    
    da_large = xr.DataArray(
        data,
        dims=('time', 'event', 'location'),
        coords={
            'time': times,
            'event': events,
            'location': locations
        }
    )
    
    # Create duration array
    durations = xr.DataArray(
        np.random.randint(10, 100, size=shape[1]),
        dims=('event'),
        coords={'event': events}
    )
    
    # Apply function with dask backend - computation is done lazily
    result = ds.select_first_period_xr(da_large, durations, days=30)
    
    # Compute a small subset first to test
    subset = result.isel(location=slice(0, 5)).compute()
    
    # When ready, compute the full result
    # This will use parallel processing based on your dask settings
    full_result = result.compute()

Blockwise Computation for Large Datasets
--------------------------------------

For extremely large spatial datasets, dominosee provides utilities for blockwise computation:

.. code-block:: python

    import dominosee.utils.blocking as blocking
    
    # Set up blockwise processing for a large dataset
    # Note: this is example code and should be adjusted based on your actual API
    
    # Define input file and output directory
    input_file = "large_climate_data.nc"
    output_dir = "processed_blocks/"
    
    # Define the block size (e.g., 5x5 degree grid cells)
    block_size = {'lat': 5, 'lon': 5}
    
    # Create blocks
    blocks = blocking.create_blocks(input_file, block_size)
    
    # Process each block separately
    for block_id, block_info in blocks.items():
        # Load this block's data
        block_data = blocking.load_block(input_file, block_info)
        
        # Process this block
        processed_data = process_block(block_data)
        
        # Save the processed block
        blocking.save_block(processed_data, f"{output_dir}/block_{block_id}.nc")
    
    # Optionally merge blocks back together
    merged_result = blocking.merge_blocks(output_dir)

Memory Optimization
------------------

When working with large datasets, memory optimization is crucial:

.. code-block:: python

    # Use chunking strategies for optimal memory usage
    optimal_chunks = {'time': 'auto', 'event': -1, 'location': 100}
    
    # Rechunk an existing dask array for better performance
    rechunked_data = da_large.chunk(optimal_chunks)
    
    # Use low-level operations for memory-critical sections
    def memory_efficient_function(data):
        # Process data in manageable chunks
        # Return results
        pass
    
    # Apply your function to each chunk independently
    result = xr.apply_ufunc(
        memory_efficient_function,
        data,
        dask='allowed',
        output_dtypes=[float]
    )

Using with High-Performance Computing
----------------------------------

For deployment on HPC clusters:

.. code-block:: python

    from dask.distributed import Client
    
    # Set up a dask client for an HPC cluster
    # Adjust based on your specific cluster setup
    client = Client(scheduler_file='scheduler.json')
    
    # Now operations will use the cluster's resources
    result = ds.select_first_period_xr(da_large, durations, days=30).compute()
    
    # You can monitor the computation progress
    print(client.dashboard_link)

Performance Profiling
--------------------

Profile your dominosee code to identify bottlenecks:

.. code-block:: python

    import time
    
    # Simple timing
    start = time.time()
    result = ds.select_first_period_xr(da_large, durations, days=30).compute()
    end = time.time()
    print(f"Computation took {end - start:.2f} seconds")
    
    # More detailed profiling with dask
    with dask.diagnostics.ProgressBar():
        result = ds.select_first_period_xr(da_large, durations, days=30).compute()
    
    # Memory profiling
    with dask.diagnostics.ResourceProfiler() as rprof:
        result = ds.select_first_period_xr(da_large, durations, days=30).compute()
    
    # Plot the memory usage profile
    rprof.visualize()
