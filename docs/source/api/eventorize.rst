.. _api_eventorize:

Event Selection (dominosee.eventorize)
======================================

.. currentmodule:: dominosee.eventorize

This module provides tools for detecting events in time series data based on thresholds.

Event Detection
---------------

.. autosummary::
   :toctree: generated/

   get_event

Threshold Utilities
-------------------

.. autosummary::
   :toctree: generated/

   cut_single_threshold

Consecutive Event Utilities
---------------------------

.. autosummary::
   :toctree: generated/

   select_start_consecutive
   select_end_consecutive
   _select_burst
   _select_wane
   _start_consecutive
   _end_consecutive
