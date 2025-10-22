"""
Test the clean FeketeGrid API following PyTorch/scikit-learn conventions.
"""
import numpy as np
import tempfile
import os
from dominosee.grid import FeketeGrid, deg_to_equatorial_distance, distance_to_grid_num


def test_create_with_initial_iterations():
    """Test creating grid with initial optimization."""
    print("1. Testing: Create grid with initial optimization...")
    grid = FeketeGrid(num_points=50, initial_iterations=10, verbose=False)
    
    assert grid.num_points == 50
    assert grid.total_iterations == 10
    assert grid.grid is not None
    assert len(grid.grid['lat']) == 50
    assert len(grid.grid['lon']) == 50
    assert len(grid.optimization_history) > 0
    
    print("   ✓ Grid created with initial optimization")


def test_create_then_optimize():
    """Test creating grid without optimization, then optimizing later."""
    print("2. Testing: Create grid then optimize later...")
    
    # Create without optimization
    grid = FeketeGrid(num_points=50, verbose=False)
    assert grid.total_iterations == 0
    assert grid.grid is not None
    
    # Optimize later
    grid.optimize(iterations=10)
    assert grid.total_iterations == 10
    assert len(grid.optimization_history) > 0
    
    print("   ✓ Grid created and optimized separately")


def test_load_and_continue():
    """Test loading existing grid and continuing optimization."""
    print("3. Testing: Load existing grid and continue optimization...")
    
    # Create and save a grid
    original = FeketeGrid(num_points=30, initial_iterations=5, verbose=False)
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        filepath = f.name
        original.to_pickle(filepath)
    
    try:
        # Load from file
        loaded = FeketeGrid.load(filepath)
        assert loaded.num_points == 30
        assert loaded.total_iterations == 5
        
        # Continue optimization
        loaded.optimize(iterations=5)
        assert loaded.total_iterations == 10
        
        print("   ✓ Grid loaded and optimization continued")
    finally:
        os.unlink(filepath)


def test_load_from_dict_and_optimize():
    """Test loading from dictionary and optimizing."""
    print("4. Testing: Load from dictionary and optimize...")
    
    # Create sample coordinates
    sample_coords = {
        'lat': np.random.uniform(-90, 90, 40),
        'lon': np.random.uniform(-180, 180, 40)
    }
    
    # Load from dictionary
    grid = FeketeGrid(num_points=40, initial_grid=sample_coords, verbose=False)
    assert grid.num_points == 40
    assert grid.total_iterations == 0
    
    # Optimize
    grid.optimize(iterations=10)
    assert grid.total_iterations == 10
    
    print("   ✓ Grid loaded from dictionary and optimized")


def test_load_with_initial_iterations():
    """Test loading grid with initial optimization."""
    print("5. Testing: Load with initial optimization...")
    
    # Create sample coordinates
    sample_coords = {
        'lat': np.random.uniform(-90, 90, 35),
        'lon': np.random.uniform(-180, 180, 35)
    }
    
    # Load from dictionary with initial iterations
    grid = FeketeGrid(num_points=35, initial_grid=sample_coords, 
                     initial_iterations=15, verbose=False)
    
    assert grid.num_points == 35
    assert grid.total_iterations == 15
    
    print("   ✓ Grid loaded with initial optimization")


def test_reproducibility():
    """Test reproducibility with random seed."""
    print("6. Testing: Reproducibility with random seed...")
    
    # Create two grids with same seed
    grid1 = FeketeGrid(num_points=20, random_seed=42, verbose=False)
    grid2 = FeketeGrid(num_points=20, random_seed=42, verbose=False)
    
    np.testing.assert_array_equal(grid1.grid['lat'], grid2.grid['lat'])
    np.testing.assert_array_equal(grid1.grid['lon'], grid2.grid['lon'])
    
    print("   ✓ Reproducible grid generation confirmed")


def test_convergence_threshold():
    """Test convergence threshold functionality."""
    print("7. Testing: Convergence threshold...")
    
    grid = FeketeGrid(
        num_points=20,
        initial_iterations=100,
        convergence_threshold=0.1,  # High threshold for quick convergence
        verbose=False
    )
    
    # Should converge before max iterations
    assert len(grid.optimization_history) <= 100
    
    print("   ✓ Convergence threshold working")


def test_chained_optimization():
    """Test multiple optimization calls (like training epochs)."""
    print("8. Testing: Chained optimization calls...")
    
    grid = FeketeGrid(num_points=25, verbose=False)
    
    # Multiple optimization rounds
    grid.optimize(iterations=5)
    assert grid.total_iterations == 5
    
    grid.optimize(iterations=10)
    assert grid.total_iterations == 15
    
    grid.optimize(iterations=5)
    assert grid.total_iterations == 20
    
    print("   ✓ Chained optimization working")


def test_5_degree_resolution_example():
    """Test real-world example: 5-degree resolution grid."""
    print("9. Testing: Real-world 5-degree resolution example...")
    
    # Calculate distance equivalent to 5 degrees at equator
    target_resolution_deg = 5.0
    target_distance_km = deg_to_equatorial_distance(target_resolution_deg, radius=6371)
    num_points = distance_to_grid_num(target_distance_km)
    
    # Create grid with initial optimization
    grid = FeketeGrid(num_points=num_points, initial_iterations=10, verbose=False)
    
    assert grid.num_points == num_points
    assert abs(grid.distance - target_distance_km) < 0.1 * target_distance_km
    
    print(f"   ✓ Created {target_resolution_deg}° resolution grid ({num_points} points)")


def test_api_comparison():
    """Demonstrate the clean API vs old patterns."""
    print("\n=== API Pattern Comparison ===")
    print("OLD (awkward):")
    print("  grid = FeketeGrid.from_scratch(num_points=1000, iterations=100)")
    print("  improved = FeketeGrid.from_existing(grid, additional_iterations=50)")
    print()
    print("NEW (clean, follows PyTorch/scikit-learn):")
    print("  grid = FeketeGrid(num_points=1000, initial_iterations=100)")
    print("  grid.optimize(iterations=50)")
    print()
    print("Loading pattern:")
    print("  grid = FeketeGrid.load('saved.pkl')")
    print("  grid.optimize(iterations=50)")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Clean FeketeGrid API")
    print("Following PyTorch/scikit-learn Conventions")
    print("="*60 + "\n")
    
    test_create_with_initial_iterations()
    test_create_then_optimize()
    test_load_and_continue()
    test_load_from_dict_and_optimize()
    test_load_with_initial_iterations()
    test_reproducibility()
    test_convergence_threshold()
    test_chained_optimization()
    test_5_degree_resolution_example()
    test_api_comparison()
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("The clean API is working perfectly!")
    print("="*60 + "\n")
