from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# nuScenes / OpenOccupancy defaults
DEFAULT_PC_RANGE: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
DEFAULT_OCC_SIZE: List[int] = [512, 512, 40]

# OpenOccupancy semantic classes used in cost-map generation
CLASS_NAMES: List[str] = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]
CLASS_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES, start=1)}
DEFAULT_OBSTACLE_CLASSES: Iterable[str] = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
    "manmade",
    "sidewalk",
]


def load_payload_info(path: Path, sample_index: int = 0) -> Dict[str, Any]:
    """Load a payload PKL/JSON and return the infos entry for the sample index."""
    if not path.exists():
        raise FileNotFoundError(f"Payload file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            payload = pickle.load(f)
    else:
        payload = json.loads(path.read_text())
    infos = payload.get("infos") or payload.get("payload", {}).get("infos")
    if not infos:
        raise ValueError(f"No 'infos' list found in payload {path}")
    if not 0 <= sample_index < len(infos):
        raise IndexError(f"sample_index {sample_index} out of range (len={len(infos)})")
    return infos[sample_index]


def quaternion_to_matrix(quat: Sequence[float]) -> np.ndarray:
    if len(quat) != 4:
        raise ValueError("Quaternion must have 4 elements")
    w, x, y, z = quat
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        raise ValueError("Quaternion has zero magnitude")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def infer_occ_size_from_grid(grid: np.ndarray) -> List[int]:
    if grid.ndim < 3:
        raise ValueError(f"Expected occupancy grid with >=3 dims, got shape {grid.shape}")
    return [int(grid.shape[0]), int(grid.shape[1]), int(grid.shape[2])]


def compute_ego_center_indices(
    payload_info: Dict[str, Any],
    occ_size: Sequence[int],
    pc_range: Sequence[float],
) -> Tuple[float, float]:
    lidar2ego_trans = payload_info.get("lidar2ego_translation")
    lidar2ego_rot = payload_info.get("lidar2ego_rotation")
    if lidar2ego_trans is None or lidar2ego_rot is None:
        raise ValueError("Payload sample lacks lidar2ego transformation")
    rot = quaternion_to_matrix(lidar2ego_rot)
    trans = np.asarray(lidar2ego_trans, dtype=np.float32)
    ego_to_lidar = rot.T
    ego_center_lidar = -ego_to_lidar @ trans

    occ_arr = np.asarray(occ_size, dtype=np.float32)
    pc_arr = np.asarray(pc_range, dtype=np.float32)
    if occ_arr.shape[0] < 2 or pc_arr.shape[0] < 6:
        raise ValueError("Invalid occ_size or pc_range")
    voxel_span = pc_arr[3:6] - pc_arr[0:3]
    voxel_span[voxel_span == 0] = 1.0
    voxel_size = voxel_span / occ_arr
    voxel_size[voxel_size == 0] = 1.0

    row_idx = (ego_center_lidar[0] - pc_arr[0]) / voxel_size[0]
    col_idx = (ego_center_lidar[1] - pc_arr[1]) / voxel_size[1]
    row_idx = float(np.clip(row_idx, 0.0, occ_arr[0] - 1.0))
    col_idx = float(np.clip(col_idx, 0.0, occ_arr[1] - 1.0))
    return row_idx, col_idx


def shift_dense_grid(grid: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
    if row_shift == 0 and col_shift == 0:
        return grid.copy()
    rows, cols = grid.shape[:2]
    shifted = np.zeros_like(grid)
    src_row_start = max(0, -row_shift)
    src_row_end = min(rows, rows - row_shift)
    src_col_start = max(0, -col_shift)
    src_col_end = min(cols, cols - col_shift)
    dest_row_start = src_row_start + row_shift
    dest_row_end = src_row_end + row_shift
    dest_col_start = src_col_start + col_shift
    dest_col_end = src_col_end + col_shift
    if src_row_start >= src_row_end or src_col_start >= src_col_end:
        return shifted
    shifted[
        dest_row_start:dest_row_end,
        dest_col_start:dest_col_end,
    ] = grid[src_row_start:src_row_end, src_col_start:src_col_end]
    return shifted


def recenter_occupancy(
    grid: np.ndarray,
    payload_info: Dict[str, Any],
    occ_size: Sequence[int],
    pc_range: Sequence[float],
) -> Tuple[np.ndarray, Tuple[int, int]]:
    row_idx, col_idx = compute_ego_center_indices(payload_info, occ_size, pc_range)
    target_row = (occ_size[0] - 1.0) / 2.0
    target_col = (occ_size[1] - 1.0) / 2.0
    row_shift = int(round(target_row - row_idx))
    col_shift = int(round(target_col - col_idx))
    shifted = shift_dense_grid(grid, row_shift, col_shift)
    return shifted, (row_shift, col_shift)


def build_class_weights(
    obstacle_classes: Iterable[str],
    ignore_classes: Iterable[str],
    custom_weights: Iterable[str],
    default_cost: float,
) -> Dict[int, float]:
    weights: Dict[int, float] = {}
    for name in obstacle_classes:
        if name in ignore_classes:
            continue
        class_id = CLASS_TO_ID.get(name)
        if class_id is not None:
            weights[class_id] = default_cost
    for item in custom_weights:
        if "=" not in item:
            raise ValueError(f"Invalid class weight '{item}', expected name=value")
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in CLASS_TO_ID:
            raise ValueError(f"Unknown class '{name}' in class-weight override")
        weights[CLASS_TO_ID[name]] = float(value)
    return weights


def project_costmap(grid: np.ndarray, class_weights: Dict[int, float], aggregate: str = "max") -> np.ndarray:
    if grid.ndim != 3:
        raise ValueError(f"Expected 3D occupancy grid, got shape {grid.shape}")
    cost = np.zeros(grid.shape[:2], dtype=np.float32)
    for class_id, weight in class_weights.items():
        if weight == 0:
            continue
        mask = grid == class_id
        if not np.any(mask):
            continue
        occupied = mask.any(axis=2).astype(np.float32) * weight
        if aggregate == "max":
            cost = np.maximum(cost, occupied)
        elif aggregate == "sum":
            cost += occupied
        else:
            raise ValueError(f"Unknown aggregate mode '{aggregate}'")
    return cost


def extract_fixed_window(cost: np.ndarray, size: int) -> np.ndarray:
    if size <= 0 or size > min(cost.shape[:2]):
        raise ValueError(f"Invalid window size {size} for shape {cost.shape}")
    half = size // 2
    center_row = cost.shape[0] // 2
    center_col = cost.shape[1] // 2
    row_start = max(center_row - half, 0)
    row_end = row_start + size
    col_start = max(center_col - half, 0)
    col_end = col_start + size
    return cost[row_start:row_end, col_start:col_end]


def build_ego_centered_costmap(
    grid: np.ndarray,
    payload_info: Dict[str, Any],
    *,
    occ_size: Optional[Sequence[int]] = None,
    pc_range: Optional[Sequence[float]] = None,
    obstacle_classes: Iterable[str] = DEFAULT_OBSTACLE_CLASSES,
    ignore_classes: Iterable[str] = ("vegetation",),
    custom_class_weights: Iterable[str] = (),
    default_cost: float = 1.0,
    aggregate: str = "max",
    window_size: Optional[int] = None,
) -> Dict[str, Any]:
    if occ_size is None:
        occ_size = infer_occ_size_from_grid(grid)
    if pc_range is None:
        pc_range = DEFAULT_PC_RANGE
    centered_grid, shift = recenter_occupancy(grid, payload_info, occ_size, pc_range)
    class_weights = build_class_weights(
        obstacle_classes=obstacle_classes,
        ignore_classes=ignore_classes,
        custom_weights=custom_class_weights,
        default_cost=default_cost,
    )
    cost_map = project_costmap(centered_grid, class_weights, aggregate=aggregate)
    window = extract_fixed_window(cost_map, window_size) if window_size else None
    return {
        "centered_grid": centered_grid,
        "cost_map": cost_map,
        "fixed_window": window,
        "shift": shift,
        "occ_size": list(occ_size),
        "pc_range": list(pc_range),
    }


__all__ = [
    "DEFAULT_PC_RANGE",
    "DEFAULT_OCC_SIZE",
    "CLASS_NAMES",
    "CLASS_TO_ID",
    "DEFAULT_OBSTACLE_CLASSES",
    "load_payload_info",
    "quaternion_to_matrix",
    "infer_occ_size_from_grid",
    "compute_ego_center_indices",
    "shift_dense_grid",
    "recenter_occupancy",
    "build_class_weights",
    "project_costmap",
    "extract_fixed_window",
    "build_ego_centered_costmap",
]
