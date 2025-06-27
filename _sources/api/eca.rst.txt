.. _api_eca:

Event Coincidence Analysis (dominosee.eca)
==========================================

.. currentmodule:: dominosee.eca

This module provides functions for performing Event Coincidence Analysis (ECA) on time series data,
particularly useful for studying the temporal relationships between different types of events.

Event Coincidence Analysis
--------------------------

.. autosummary::
   :toctree: generated/
   
   get_eca_precursor
   get_eca_trigger
   get_eca_precursor_from_events
   get_eca_trigger_from_events

Window Functions
----------------

.. autosummary::
   :toctree: generated/
   
   get_eca_precursor_window
   get_eca_trigger_window

Confidence Calculation
----------------------

.. autosummary::
   :toctree: generated/
   
   get_eca_precursor_confidence
   get_eca_trigger_confidence
   get_prec_confidence
   get_trig_confidence

Example Usage
-------------

An example of using ECA to study the temporal relationships between climate events can be found in the :doc:`ECA Network Example </notebooks/eca_network>`.
