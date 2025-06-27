.. _api_network:

Network Construction (dominosee.network)
========================================

.. currentmodule:: dominosee.network

This module provides functions for constructing networks from event analysis results,
with various methods to create links based on thresholds, significance, and other criteria.

Link Creation Functions
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_link_from_threshold
   get_link_from_significance
   get_link_from_confidence
   get_link_from_quantile
   get_link_from_critical_values

Example Usage
-------------

Examples of constructing networks from event analysis can be found in the :doc:`ES Network Example </notebooks/es_network>` and :doc:`ECA Network Example </notebooks/eca_network>`.
