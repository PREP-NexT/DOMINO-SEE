"""
Tests for the eventorize module's percentile-based event extraction functionality
"""
import numpy as np
import pytest
import xarray as xr

from dominosee import eventorize


def test_get_event_percentile(ds_example):
    """
    Test percentile-based event extraction on the example dataset
    
    This test verifies that:
    1. get_event_percentile produces a boolean DataArray
    2. The output has the same dimensions as the input
    3. The output has the expected attributes
    4. The output contains the expected boolean mask values
    """
    # Apply percentile-based event detection (90th percentile, above threshold)
    percentile = 0.9
    extreme = "above"
    event_name = "event"
    
    # Run the function
    event_da = eventorize.get_event_percentile(
        ds_example.Tmean, 
        percentile=percentile, 
        extreme=extreme
    )
    
    # Check type and data type
    assert isinstance(event_da, xr.DataArray)
    assert event_da.dtype == bool, "Event array should have boolean data type"
    
    # Check dimensions match input
    assert event_da.dims == ds_example.Tmean.dims
    assert event_da.shape == ds_example.Tmean.shape
    
    # Check coordinates match input
    for dim in ds_example.Tmean.dims:
        np.testing.assert_array_equal(event_da[dim], ds_example.Tmean[dim])
    
    # Check attributes are set correctly
    assert event_da.attrs["percentile"] == percentile
    assert event_da.attrs["extreme"] == extreme
    assert event_da.attrs["event_name"] == event_name
    assert "description" in event_da.attrs
    assert "long_name" in event_da.attrs
    
    # Check for expected boolean values (derived from the notebook)
    # For percentile=0.9, extreme="above" case
    expected_mask = np.array([
        [False, False, False, False, False, False, False, False, False,
         False, True, False, False, False, False, True, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, True, False, False,
         False, False, False, False, True, False, False, False, False,
         False, False, True, False, True, False]
    ])
    
    # Convert to numpy for comparison and ensure the shape is correct
    actual_mask = event_da.values
    
    # Assert the masks are identical
    np.testing.assert_array_equal(
        actual_mask, expected_mask,
        err_msg="Event mask doesn't match expected values for 90th percentile threshold"
    )


def test_get_event_percentile_selection(ds_example):
    """
    Test event extraction with selection options
    
    This tests the select and select_period parameters which control
    whether to select only the start, middle, or end of events and
    how many timesteps to include.
    
    This test focuses on validating the functionality works when selection
    options are applied, not on the specific values of the result.
    """
    # Test with selection options
    for select in ["start", "end"]:
        event_da = eventorize.get_event_percentile(
            ds_example.Tmean,
            percentile=0.9, 
            extreme="above",
            select=select,
            select_period=1
        )
        
        # Assert that result is still a boolean array
        assert event_da.dtype == bool
        
        # Verify the select parameter was applied
        assert event_da.attrs["select"] == select
        assert event_da.attrs["select_period"] == 1
        
        # Since selection modifies the event pattern, we can't compare against
        # a fixed expected pattern, but we can check that some events are detected
        assert event_da.sum() > 0, f"No events detected with select={select}"
        
        # All tests from basic test apply
        assert isinstance(event_da, xr.DataArray)
        assert event_da.dtype == bool
        
        # Check selection parameters were recorded in attributes
        assert event_da.attrs["select"] == select
        assert event_da.attrs["select_period"] == 1
