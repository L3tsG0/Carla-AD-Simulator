import numpy as np
import pytest

from DrivingAgent.src.occupancy_costmap import (
    CLASS_TO_ID,
    build_ego_centered_costmap,
    compute_ego_center_indices,
    recenter_occupancy,
)


PC_RANGE = [-3.0, -3.0, 0.0, 3.0, 3.0, 2.0]
OCC_SIZE = [6, 6, 2]


def make_payload(center_xyz):
    x, y, z = center_xyz
    return {
        "lidar2ego_translation": [-x, -y, -z],
        "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
    }


def test_compute_ego_center_indices_identity():
    payload = make_payload((1.0, -2.0, 0.5))
    row_idx, col_idx = compute_ego_center_indices(payload, OCC_SIZE, PC_RANGE)
    assert pytest.approx(row_idx, rel=1e-5) == 4.0
    assert pytest.approx(col_idx, rel=1e-5) == 1.0


def test_recenter_occupancy_shifts_grid():
    grid = np.zeros((6, 6, 2), dtype=np.uint8)
    grid[4, 1, 0] = 7
    payload = make_payload((1.0, -2.0, 0.5))
    shifted, shift = recenter_occupancy(grid, payload, OCC_SIZE, PC_RANGE)
    assert shift == (-2, 2)
    assert shifted[2, 3, 0] == 7


def test_build_ego_centered_costmap_returns_window():
    grid = np.zeros((6, 6, 2), dtype=np.uint8)
    grid[3, 3, 0] = CLASS_TO_ID["car"]
    payload = make_payload((0.0, 0.0, 0.0))
    result = build_ego_centered_costmap(
        grid,
        payload,
        occ_size=OCC_SIZE,
        pc_range=PC_RANGE,
        window_size=4,
    )
    assert result["fixed_window"].shape == (4, 4)
    assert result["cost_map"].shape == (6, 6)
    assert np.any(result["cost_map"] > 0)
