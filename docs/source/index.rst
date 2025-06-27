Welcome to DOMINO-SEE
=====================

.. image:: _static/images/dominosee_logo.svg
   :align: center
   :alt: dominosee Logo
   :width: 400px

**DOMINO-SEE** is a framework for **D**\etection **O**\f **M**\ulti-layer **I**\nterconnected **N**\etwork **O**\ccurrences for **S**\patial **E**\xtreme **E**\vents, leveraging event coincidences analysis and complex networks. It is built using `xarray <https://xarray.pydata.org/en/stable/>`_ and can seamlessly benefit from the parallelization handling provided by `dask <https://dask.org/>`_. Its objective is to make it as simple as possible for users to construct event-based climate network analysis workflows.

The name **DOMINO-SEE** represents our approach to detecting and analyzing interconnected occurrences of hydroclimatic extreme events across spatial locations, inspired by the cascade effect of **DOMINO**\es falling in a chain reaction. The **SEE** highlights the framework's ability to capture the spatial synchronization and propagation of extreme events, emphasizing the interconnectedness inherent in complex environmental systems.

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: :octicon:`rocket` Getting Started
        :link: getting_started
        :link-type: ref
        :class-card: sd-bg-light

        Get started quickly with installation instructions and a quick overview.

    .. grid-item-card:: :octicon:`book` User Guide
        :link: user_guide
        :link-type: ref
        :class-card: sd-bg-light

        Learn how to use `dominosee`'s features in-depth.

    .. grid-item-card:: :octicon:`code` API Reference
        :link: api_reference
        :link-type: ref
        :class-card: sd-bg-light

        Detailed documentation of functions, classes, and methods.

    .. grid-item-card:: :octicon:`beaker` Examples
        :link: examples
        :link-type: ref
        :class-card: sd-bg-light

        Explore example notebooks and tutorials.

Key Features
-----------

- **Complex Network Generation**: Fast and memory-efficient functions to build spatial networks from event series among spatial locations and multiple types (layers) of climate extreme events
- **Multi-dimensional Support**: Native support for ``xarray`` DataArrays to handle multi-dimensional gridded climate data
- **Parallel Processing**: ``dask`` integration for efficient processing of large-scale climate datasets
- **Blockwise Computation**: Utilities for splitting large spatial datasets into manageable blocks of netCDF datasets

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Contents

   installation
   quickstart
   notebooks/index
   api/index
   contributing
   changelog

.. _getting_started:

Getting Started
--------------

To get started with `dominosee`, check out the :doc:`installation` and :doc:`quickstart` guides.

.. code-block:: python

   import numpy as np
   import xarray as xr
   
   # Create synthetic climate data
   time = xr.cftime_range(start='2000-01-01', periods=365, freq='D')
   lat = np.linspace(-90, 90, 10)
   lon = np.linspace(-180, 180, 10)
   
   # Generate random SPI-like data
   np.random.seed(42)
   spi = xr.DataArray(
       np.random.normal(0, 1, (len(time), len(lat), len(lon))),
       dims=('time', 'lat', 'lon'),
       coords={'time': time, 'lat': lat, 'lon': lon},
       name='SPI1'
   )
   
   # 1. Event detection
   from dominosee.eventorize import get_event
   da_event = get_event(spi, threshold=-1.0, extreme="below", event_name="drought")
   
   # 2a. ECA analysis
   from dominosee.eca import get_eca_precursor_from_events
   da_precursor = get_eca_precursor_from_events(
       eventA=da_event, eventB=da_event, delt=2, sym=True, tau=0
   )
   
   # 2b. ES analysis
   from dominosee.eventorize import get_event_positions
   from dominosee.es import get_event_sync_from_positions, create_null_model_from_indices
   
   ds_event_pos = get_event_positions(da_event)
   da_es = get_event_sync_from_positions(
       positionsA=ds_event_pos.event_positions,
       positionsB=ds_event_pos.event_positions,
       time_dim='time',
       tau_max=10
   )
   da_es_critical_value = create_null_model_from_indices(da_event.time, tau_max=10, 
                                                         max_events=ds_event_pos.event_count.max(), 
                                                         max_tau=10)
   da_es_null = convert_null_model_for_locations(da_es_critical_value, 
                                                 ds_event_pos.event_count, 
                                                 ds_event_pos.event_count)
   
   # 3. Network construction
   from dominosee.network import get_link_from_confidence, get_link_from_critical_value
   
   # For ECA
   da_precursor_conf = get_eca_precursor_confidence(
       precursor=da_precursor, 
       eventA=da_event, 
       eventB=da_event
   )
   da_precursor_link = get_link_from_confidence(da_precursor_conf, confidence_level=0.95)
   
   # For ES (using a critical value threshold)
   da_es_network = get_link_from_critical_values(da_es, critical_value=da_es_null, rule="greater_equal")

.. .. _user_guide:

.. User Guide
.. ----------

.. The User Guide provides in-depth documentation on using `dominosee` for various tasks.

.. _examples:

Examples
--------

Check out the examples to see `dominosee` in action:

- :doc:`notebooks/eca_network`: Examples of using `dominosee` for ECA network analysis
- :doc:`notebooks/es_network`: Examples of using `dominosee` for ES network analysis

.. _api_reference:

API Reference
------------

The API Reference provides detailed documentation for all functions, classes, and methods in `dominosee`.

- :doc:`api/index`: Complete API documentation
- :doc:`api/eventorize`: Event selection
- :doc:`api/eca`: Event Coincidence Analysis (ECA)
- :doc:`api/es`: Event Synchronization (ES)
- :doc:`api/network`: Network creation


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
