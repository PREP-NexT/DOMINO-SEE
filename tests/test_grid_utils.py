import numpy as np
import pytest

from dominosee.grid import (
    deg_to_equatorial_distance,
    equatorial_distance_to_deg,
    neighbour_distance,
    geo_distance,
)


def test_deg_equatorial_roundtrip_values():
    for deg in [0.5, 1.0, 5.0, 10.0, 30.0]:
        d = deg_to_equatorial_distance(deg)
        back = equatorial_distance_to_deg(d)
        assert pytest.approx(back, rel=1e-6, abs=1e-8) == deg


def test_known_value_5deg_equator():
    # 5 degrees at equator is circumference * 5/360
    d = deg_to_equatorial_distance(5.0)
    # Use expected from formula to avoid hardcoding magic numbers
    R = 6371.0
    expected = 2 * np.pi * R * (5.0 / 360.0)
    assert pytest.approx(d, rel=1e-6, abs=1e-6) == expected
    # And inverse
    back = equatorial_distance_to_deg(d)
    assert pytest.approx(back, rel=1e-9, abs=1e-9) == 5.0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        deg_to_equatorial_distance(0)
    with pytest.raises(ValueError):
        deg_to_equatorial_distance(-1)
    with pytest.raises(ValueError):
        equatorial_distance_to_deg(0)
    with pytest.raises(ValueError):
        equatorial_distance_to_deg(-10)


def test_neighbour_distance_on_equator_line():
    # Three points on equator along longitude: (0,0), (1,0), (2,0)
    grid = {
        'lon': np.array([0.0, 1.0, 2.0]),
        'lat': np.array([0.0, 0.0, 0.0]),
    }
    nd = neighbour_distance(grid)
    assert nd.shape == (3,)
    expected = deg_to_equatorial_distance(1.0)
    assert pytest.approx(nd[0], rel=1e-6) == expected
    assert pytest.approx(nd[1], rel=1e-6) == expected
    assert pytest.approx(nd[2], rel=1e-6) == expected


def test_neighbour_distance_on_unit_rectangle():
    # Four points forming a 1x1 degree rectangle near equator
    # (0,0), (0,1), (1,0), (1,1)
    grid = {
        'lon': np.array([0.0, 0.0, 1.0, 1.0]),
        'lat': np.array([0.0, 1.0, 0.0, 1.0]),
    }
    nd = neighbour_distance(grid)
    assert nd.shape == (4,)
    # Expected nearest distances per point
    # (0,0): min to (0,1) or (1,0) at equator
    e00 = min(geo_distance(0.0, 0.0, 0.0, 1.0), geo_distance(0.0, 0.0, 1.0, 0.0))
    # (0,1): min to (0,0) (lat diff) or (1,1) (lon diff at 1° lat)
    e01 = min(geo_distance(0.0, 1.0, 0.0, 0.0), geo_distance(0.0, 1.0, 1.0, 1.0))
    # (1,0): min to (0,0) or (1,1)
    e10 = min(geo_distance(1.0, 0.0, 0.0, 0.0), geo_distance(1.0, 0.0, 1.0, 1.0))
    # (1,1): min to (1,0) (lat diff) or (0,1) (lon diff at 1° lat)
    e11 = min(geo_distance(1.0, 1.0, 1.0, 0.0), geo_distance(1.0, 1.0, 0.0, 1.0))
    expected = np.array([e00, e01, e10, e11])
    assert np.allclose(nd, expected, rtol=1e-6)
