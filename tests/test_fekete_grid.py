"""
Tests for FeketeGrid generation and functionality
"""
import numpy as np
import pytest

from dominosee.grid import FeketeGrid, deg_to_eq_spacing, distance_to_grid_num, grid_num_to_distance


def test_fekete_grid_5_degree_resolution():
    """
    Test FeketeGrid generation with 5-degree resolution equivalent on equator.
    
    This test creates a FeketeGrid with approximately 5-degree spacing at the equator
    and validates basic properties of the generated grid.
    """
    # Calculate distance equivalent to 5 degrees at equator
    target_resolution_deg = 5.0
    target_distance_km = deg_to_eq_spacing(target_resolution_deg, radius=6371)
    
    # Calculate number of points needed for this resolution
    num_points = distance_to_grid_num(target_distance_km)
    
    # Create FeketeGrid with minimal iterations for testing
    # Using 0 iterations to just get the initial random configuration for quick testing
    grid = FeketeGrid(num_points=num_points, num_iter=0)
    
    # Basic validation tests
    assert grid.grid is not None, "Grid should be created"
    assert 'lat' in grid.grid, "Grid should contain latitude coordinates"
    assert 'lon' in grid.grid, "Grid should contain longitude coordinates"
    
    # Check grid size
    assert len(grid.grid['lat']) == num_points, f"Grid should have {num_points} points"
    assert len(grid.grid['lon']) == num_points, f"Grid should have {num_points} points"
    
    # Check coordinate ranges
    assert np.all(grid.grid['lat'] >= -90), "Latitudes should be >= -90"
    assert np.all(grid.grid['lat'] <= 90), "Latitudes should be <= 90"
    assert np.all(grid.grid['lon'] >= -180), "Longitudes should be >= -180"
    assert np.all(grid.grid['lon'] <= 180), "Longitudes should be <= 180"
    
    # Check that the grid has the expected distance property
    expected_distance = target_distance_km
    actual_distance = grid.get_distance_equator()
    
    # Allow some tolerance due to the empirical relationship between num_points and distance
    tolerance = 0.1 * expected_distance  # 10% tolerance
    assert abs(actual_distance - expected_distance) <= tolerance, \
        f"Grid distance {actual_distance:.1f} km should be close to target {expected_distance:.1f} km"
    
    # Verify grid properties
    assert grid.num_points == num_points, "Number of points should match input"
    assert grid.num_iter == 0, "Number of iterations should match input"
    assert isinstance(grid.dq, list), "dq should be a list"
    
    print(f"Successfully created FeketeGrid with {num_points} points")
    print(f"Target resolution: {target_resolution_deg}° ({target_distance_km:.1f} km)")
    print(f"Actual distance: {actual_distance:.1f} km")


def test_fekete_grid_with_iterations():
    """
    Test FeketeGrid generation with optimization iterations.
    
    This test creates a smaller FeketeGrid with a few optimization iterations
    to verify the improvement functionality works.
    """
    # Use a smaller grid for faster testing with iterations
    target_distance_km = deg_to_eq_spacing(10.0, radius=6371)  # 10-degree resolution
    num_points = distance_to_grid_num(target_distance_km)
    
    # Create grid with a few iterations
    num_iter = 10
    grid = FeketeGrid(num_points=num_points, num_iter=num_iter)
    
    # Basic validation
    assert grid.grid is not None
    assert len(grid.grid['lat']) == num_points
    assert len(grid.grid['lon']) == num_points
    
    # Check that iterations were performed
    assert grid.num_iter == num_iter
    assert len(grid.dq) > 0, "Should have disequilibrium data from iterations"
    
    # Check that dq values are decreasing (grid is improving)
    if len(grid.dq[0]) > 1:  # If we have multiple dq values
        dq_values = grid.dq[0]
        # Check that the last few values are generally decreasing
        if len(dq_values) >= 3:
            assert dq_values[-1] <= dq_values[0], "Final dq should be <= initial dq (grid improved)"
    
    print(f"Successfully created optimized FeketeGrid with {num_points} points and {num_iter} iterations")


def test_fekete_grid_from_dict():
    """
    Test FeketeGrid initialization from a lat/lon dictionary.
    """
    # Create a simple test grid
    test_lats = np.array([0, 30, -30, 60, -60])
    test_lons = np.array([0, 45, -45, 90, -90])
    test_grid = {'lat': test_lats, 'lon': test_lons}
    
    # Initialize FeketeGrid from dictionary
    grid = FeketeGrid(num_points=5, grid=test_grid)
    
    # Verify the grid was loaded correctly
    assert grid.num_points == 5
    np.testing.assert_array_equal(grid.grid['lat'], test_lats)
    np.testing.assert_array_equal(grid.grid['lon'], test_lons)
    assert isinstance(grid.dq, list)
    
    print("Successfully created FeketeGrid from dictionary")


if __name__ == "__main__":
    # Run the main test when script is executed directly
    test_fekete_grid_5_degree_resolution()
    test_fekete_grid_with_iterations()
    test_fekete_grid_from_dict()
    print("All tests passed!")
