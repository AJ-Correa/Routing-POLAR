import numpy as np
import pytest

from search.search import POLAR_SCALER, compute_cost_matrix

pytestmark = pytest.mark.unit


def test_cost_matrix_symmetry():
    locs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    dlin = np.array([0.0, 1.0, 1.0])
    dbac = np.zeros(3)
    m = compute_cost_matrix(
        locs, dlin, dbac, open_route=False, coords_scaled=False
    )
    assert np.allclose(m, m.T)


def test_cost_matrix_open_route_zeros_depot_columns():
    locs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    dlin = np.array([0.0, 1.0, 1.0])
    dbac = np.zeros(3)
    m = compute_cost_matrix(
        locs, dlin, dbac, open_route=True, coords_scaled=False
    )
    assert np.allclose(m[:, 0], 0.0)


def test_cost_matrix_backhaul_mask():
    locs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    dlin = np.array([0.0, 1.0, 0.0])
    dbac = np.array([0.0, 0.0, 1.0])
    m = compute_cost_matrix(
        locs, dlin, dbac, open_route=False, mixed_backhaul=False, coords_scaled=False
    )
    assert m[2, 1] > 1e9


def test_polar_scaled_coords_no_extra_multiplier():
    locs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]) * POLAR_SCALER
    dlin = np.array([0.0, 1.0, 1.0])
    m = compute_cost_matrix(
        locs, dlin, np.zeros(3), open_route=False, coords_scaled=True
    )
    expected = np.linalg.norm(locs[1] - locs[0])
    assert m[0, 1] == pytest.approx(expected)
