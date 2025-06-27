.. _api_es:

Event Synchronization (dominosee.es)
====================================

.. currentmodule:: dominosee.es

This module provides functions for analyzing event synchronization in time series data,
particularly useful for climate science applications like studying extreme events.

Event Position Conversion
-------------------------

.. autosummary::
   :toctree: generated/
   
   get_event_positions
   get_event_time_differences

Event Synchronization Calculation
---------------------------------

.. autosummary::
   :toctree: generated/

   get_event_sync_from_positions

Null Model Generation
---------------------

.. autosummary::
   :toctree: generated/

   create_null_model_from_indices
   convert_null_model_for_locations

Event Position Utilites
-----------------------

.. autosummary::
   :toctree: generated/ 

   _extract_event_positions
   _DataArrayTime_to_timeindex

Event Synchronization Utilites
------------------------------

.. autosummary::
   :toctree: generated/

   _event_sync
   _event_sync_null

Example Usage
-------------

An example of using ES analysis to study the synchronization of drought events can be found in the :doc:`ES Network Example </notebooks/es_network>`.