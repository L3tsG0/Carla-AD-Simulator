from __future__ import annotations

import argparse
import contextlib
import json
import math
import pickle
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from run_logger import RunLogger


@dataclass
class PipelineResult:
    payload_path: Path
    prediction_dense_path: Optional[Path] = None
    prediction_ego_path: Optional[Path] = None
    costmap_full_path: Optional[Path] = None
    costmap_window_path: Optional[Path] = None
    costmap_image_path: Optional[Path] = None
    occ_size: Optional[List[int]] = None
    pc_range: Optional[List[float]] = None
    payload_artifact_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    front_camera_image_path: Optional[Path] = None


class AttackSimulator:
    """Apply adversarial occupancy edits such as hiding or appearing cars."""

    def __init__(self, config: Dict[str, Any], *, tmp_dir: Path) -> None:
        self.config = dict(config)
        self.mode = str(self.config.get("mode", "none")).lower()
        self.target_class_id = int(self.config.get("target_class_id", self.config.get("class_id", 4)))
        self.hide_probability = float(self.config.get("hide_probability", self.config.get("probability", 0.0)))
        self.hide_fill_class_id = int(self.config.get("hide_fill_class_id", self.config.get("fill_class_id", 0)))
        self.appear_probability = float(self.config.get("appear_probability", self.config.get("probability", 0.0)))
        self.anchor_distance_m = float(self.config.get("appear_distance_m", self.config.get("anchor_distance_m", 20.0)))
        box = self.config.get("appear_box_m", self.config.get("box_size_m", [4.72, 1.85, 1.44]))
        if not isinstance(box, (list, tuple)) or len(box) != 3:
            box = [4.72, 1.85, 1.44]
        self.appear_box_m = tuple(float(v) for v in box)
        self.anchor_height_offset = float(self.config.get("anchor_height_offset_m", 0.0))
        self.anchor_mode = str(self.config.get("anchor_mode", "relative_follow")).lower()
        self.speed_trigger_mps = float(
            self.config.get("appear_speed_trigger_mps", self.config.get("speed_trigger_mps", 0.0))
        )
        self.speed_trigger_once = bool(
            self.config.get("appear_speed_trigger_once", self.config.get("speed_trigger_once", True))
        )
        override = self.config.get("anchor_global")
        self.anchor_override = (
            np.asarray(override, dtype=np.float32) if isinstance(override, (list, tuple)) and len(override) == 3 else None
        )
        forward_axis = self.config.get("forward_axis")
        if isinstance(forward_axis, (list, tuple)) and len(forward_axis) == 3:
            vec = np.asarray(forward_axis, dtype=np.float32)
        else:
            vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            norm = 1.0
        self.forward_axis = vec / norm
        self.xy_downsample = float(self.config.get("grid_xy_downsample", self.config.get("xy_downsample", 4.0)))
        if self.xy_downsample <= 0.0:
            self.xy_downsample = 1.0
        axis_tokens = self.config.get("grid_axis_order", ["x", "y", "z"])
        if isinstance(axis_tokens, str):
            axis_tokens = [axis_tokens]
        axis_map = {"x": 0, "y": 1, "z": 2}
        parsed: List[int] = []
        for token in axis_tokens:
            token_lower = str(token).lower()
            idx = axis_map.get(token_lower)
            if idx is not None and idx not in parsed:
                parsed.append(idx)
        if len(parsed) != 3:
            parsed = [axis_map["x"], axis_map["y"], axis_map["z"]]
        self.axis_indices = tuple(parsed)
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        seed = self.config.get("seed")
        self.rng = np.random.default_rng(seed)
        self._fixed_anchor_global: Optional[np.ndarray] = None
        self._last_ego_global: Optional[np.ndarray] = None
        self._cumulative_travel_m: float = 0.0
        self.tick_seconds = float(self.config.get("tick_seconds", 0.1))
        self.last_operation: Optional[Dict[str, Any]] = None
        self._speed_trigger_met = self.speed_trigger_mps <= 0.0

    @staticmethod
    def load_from_path(path: Path, *, tmp_dir: Path) -> "AttackSimulator":
        config = AttackSimulator._read_config(path)
        return AttackSimulator(config, tmp_dir=tmp_dir)

    @staticmethod
    def _read_config(path: Path) -> Dict[str, Any]:
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Attack config not found: {path}")
        try:
            if path.suffix in {".yml", ".yaml"}:
                try:
                    import yaml  # type: ignore
                except ImportError as exc:  # pragma: no cover - optional dependency guard
                    raise RuntimeError("PyYAML is required to parse YAML attack configs.") from exc
                with path.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
            else:
                with path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
        except Exception as exc:  # pragma: no cover - config parsing guard
            raise RuntimeError(f"Failed to parse attack config {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Attack config {path} must be a mapping.")
        return data

    def apply(
        self,
        prediction_path: Path,
        payload_info: Optional[Dict[str, Any]],
        occ_size: Optional[List[int]],
        pc_range: Optional[List[float]],
    ) -> Path:
        if self.mode == "none":
            return prediction_path
        grid = np.load(prediction_path)
        changed = False
        extra: Dict[str, Any] = {}
        if self.mode == "hide":
            changed, extra = self._apply_hiding(grid)
        elif self.mode == "appear":
            changed, extra = self._apply_appearing(grid, payload_info, pc_range, occ_size)
        else:
            extra["reason"] = "mode_unsupported"
        log_payload = {"mode": self.mode, "changed": changed}
        log_payload.update(extra)
        self.last_operation = log_payload
        if not changed:
            return prediction_path
        attacked_path = self._write_attacked(prediction_path, grid)
        return attacked_path

    def _apply_hiding(self, grid: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        info: Dict[str, Any] = {
            "target_class_id": self.target_class_id,
            "hide_probability": self.hide_probability,
        }
        if self.hide_probability <= 0.0 or grid.ndim < 3:
            info["reason"] = "probability_zero"
            return False, info
        mask = grid == self.target_class_id
        total = int(mask.sum())
        info["total_voxels"] = total
        if total == 0:
            info["reason"] = "no_target_voxels"
            return False, info
        take = int(round(total * self.hide_probability))
        if take <= 0:
            info["reason"] = "sample_zero"
            return False, info
        take = min(total, take)
        flat_indices = np.flatnonzero(mask)
        if flat_indices.size == 0:
            info["reason"] = "no_indices"
            return False, info
        selected = self.rng.choice(flat_indices, size=take, replace=False)
        flat_grid = grid.reshape(-1)
        flat_grid[selected] = self.hide_fill_class_id
        info["removed_voxels"] = int(take)
        info["fill_class_id"] = self.hide_fill_class_id
        return True, info

    def _apply_appearing(
        self,
        grid: np.ndarray,
        payload_info: Optional[Dict[str, Any]],
        pc_range: Optional[List[float]],
        occ_size: Optional[List[int]],
    ) -> Tuple[bool, Dict[str, Any]]:
        meta: Dict[str, Any] = {
            "target_class_id": self.target_class_id,
            "appear_probability": self.appear_probability,
        }
        if self.appear_probability <= 0.0 or grid.ndim < 3:
            meta["reason"] = "probability_zero"
            return False, meta
        if pc_range is None or payload_info is None:
            print("[AttackSimulator] Appearing attack requires payload info and pc_range.")
            meta["reason"] = "missing_payload"
            return False, meta
        if self.speed_trigger_mps > 0.0 and (not self._speed_trigger_met or not self.speed_trigger_once):
            current_speed = self._estimate_vehicle_speed(payload_info)
            meta["speed_trigger_mps"] = self.speed_trigger_mps
            meta["ego_speed_mps"] = current_speed
            if current_speed is None or current_speed < self.speed_trigger_mps:
                meta["reason"] = "speed_below_trigger"
                return False, meta
            if self.speed_trigger_once:
                self._speed_trigger_met = True
        anchor_global = self._resolve_anchor_global(payload_info)
        if anchor_global is None:
            meta["reason"] = "anchor_unavailable"
            return False, meta
        anchor_lidar = self._global_to_lidar(payload_info, anchor_global)
        if anchor_lidar is None:
            meta["reason"] = "lidar_transform_failed"
            return False, meta
        grid_shape = np.array(grid.shape[:3], dtype=np.float32)
        mins_raw = np.asarray(pc_range[:3], dtype=np.float32)
        spans_raw = np.asarray(pc_range[3:6], dtype=np.float32) - mins_raw
        spans_raw[spans_raw == 0] = 1.0
        order = np.array(self.axis_indices, dtype=np.int64)
        mins = mins_raw[order]
        spans = spans_raw[order]
        anchor_aligned = anchor_lidar[order]
        if occ_size and len(occ_size) >= 3:
            reference_shape = np.array(occ_size[:3], dtype=np.float32)[order]
        else:
            reference_shape = grid_shape.copy()
        reference_shape[reference_shape <= 0.0] = 1.0
        voxel_sizes_ref = spans / reference_shape
        center = (anchor_aligned - mins) / voxel_sizes_ref
        scale = grid_shape / reference_shape
        center *= scale
        if np.any(np.isnan(center)):
            meta["reason"] = "nan_center"
            return False, meta
        center = np.clip(center, 0.0, grid_shape - 1.0)
        box_dims_world = np.maximum(np.asarray(self.appear_box_m, dtype=np.float32), 1e-3)
        box_dims = box_dims_world[order]
        half_extents = box_dims / (2.0 * voxel_sizes_ref)
        half_extents *= scale
        if self.xy_downsample != 1.0:
            center[:2] = center[:2] / self.xy_downsample
            half_extents[:2] = half_extents[:2] / self.xy_downsample
        min_idx = np.maximum(np.floor(center - half_extents), 0.0).astype(int)
        max_idx = np.minimum(np.ceil(center + half_extents).astype(int) + 1, grid_shape.astype(int))
        x0, y0, z0 = min_idx.tolist()
        x1, y1, z1 = max_idx.tolist()
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            meta["reason"] = "empty_region"
            meta["voxel_bounds"] = {"min": min_idx.tolist(), "max": max_idx.tolist()}
            return False, meta
        region = grid[x0:x1, y0:y1, z0:z1]
        random_mask = self.rng.random(region.shape) < self.appear_probability
        if not random_mask.any():
            meta["reason"] = "mask_empty"
            meta["voxel_bounds"] = {"min": min_idx.tolist(), "max": max_idx.tolist()}
            return False, meta
        region[random_mask] = self.target_class_id
        grid[x0:x1, y0:y1, z0:z1] = region
        meta.update(
            {
                "anchor_global": anchor_global.tolist(),
                "anchor_lidar": anchor_lidar.tolist(),
                "voxel_bounds": {"min": min_idx.tolist(), "max": max_idx.tolist()},
                "region_shape": [int(x1 - x0), int(y1 - y0), int(z1 - z0)],
                "placed_voxels": int(random_mask.sum()),
            }
        )
        return True, meta

    def _resolve_anchor_global(self, payload_info: Dict[str, Any]) -> Optional[np.ndarray]:
        if self.anchor_override is not None:
            return self.anchor_override
        if self.anchor_mode in {"relative_follow", "follow", "moving"}:
            anchor = self._compute_follow_anchor(payload_info)
            if anchor is not None:
                return anchor
        candidate = self._compute_forward_anchor(payload_info)
        if candidate is None:
            return None
        if self.anchor_mode in {"relative_start", "fixed_start"}:
            if self._fixed_anchor_global is None:
                self._fixed_anchor_global = candidate
            return self._fixed_anchor_global
        return candidate

    def _compute_forward_anchor(self, payload_info: Dict[str, Any]) -> Optional[np.ndarray]:
        forward = self._get_forward_vector(payload_info)
        ego_trans = payload_info.get("ego2global_translation")
        if forward is None or ego_trans is None:
            return None
        origin = np.asarray(ego_trans, dtype=np.float32)
        anchor = origin + forward * float(self.anchor_distance_m)
        anchor[2] += self.anchor_height_offset
        return anchor

    def _compute_follow_anchor(self, payload_info: Dict[str, Any]) -> Optional[np.ndarray]:
        forward = self._get_forward_vector(payload_info)
        ego_trans = payload_info.get("ego2global_translation")
        if forward is None or ego_trans is None:
            return None
        ego_global = np.asarray(ego_trans, dtype=np.float32)
        if self._last_ego_global is None:
            self._last_ego_global = ego_global
            self._cumulative_travel_m = 0.0
            anchor = ego_global + forward * float(self.anchor_distance_m)
            anchor[2] += self.anchor_height_offset
            return anchor
        travel = self._estimate_forward_travel(ego_global, forward, payload_info)
        if travel > 0.0:
            self._cumulative_travel_m += travel
        self._last_ego_global = ego_global
        remaining = max(0.0, float(self.anchor_distance_m) - self._cumulative_travel_m)
        anchor = ego_global + forward * remaining
        anchor[2] += self.anchor_height_offset
        return anchor

    def _estimate_forward_travel(
        self, ego_global: np.ndarray, forward: np.ndarray, payload_info: Dict[str, Any]
    ) -> float:
        travel = 0.0
        if self._last_ego_global is not None:
            delta = ego_global - self._last_ego_global
            travel = abs(float(np.dot(delta, forward)))
        if abs(travel) < 1e-3:
            can_bus = payload_info.get("can_bus")
            if isinstance(can_bus, (list, tuple, np.ndarray)) and len(can_bus) >= 10:
                vel = np.asarray(can_bus[7:9], dtype=np.float32)
                vel_vec = np.array([vel[0], vel[1], 0.0], dtype=np.float32)
                travel = abs(float(np.dot(vel_vec, forward)) * float(self.tick_seconds))
        return max(0.0, travel)

    def _estimate_vehicle_speed(self, payload_info: Dict[str, Any]) -> Optional[float]:
        can_bus = payload_info.get("can_bus")
        if isinstance(can_bus, (list, tuple, np.ndarray)) and len(can_bus) >= 10:
            vel = np.asarray(can_bus[7:9], dtype=np.float32)
            return float(np.linalg.norm(vel))
        speed = payload_info.get("speed")
        if speed is not None:
            try:
                return float(speed)
            except (TypeError, ValueError):
                return None
        velocity = payload_info.get("velocity") or payload_info.get("ego_velocity")
        if isinstance(velocity, (list, tuple, np.ndarray)) and len(velocity) >= 2:
            vel = np.asarray(velocity[:2], dtype=np.float32)
            return float(np.linalg.norm(vel))
        return None

    def _get_forward_vector(self, payload_info: Dict[str, Any]) -> Optional[np.ndarray]:
        ego_rot = payload_info.get("ego2global_rotation")
        if ego_rot is None:
            return None
        try:
            rot = OccupancyCostmapPipeline._quaternion_to_matrix(ego_rot)
        except Exception:
            return None
        forward = rot @ self.forward_axis
        norm = float(np.linalg.norm(forward))
        if norm == 0.0:
            return None
        return forward / norm

    def _global_to_lidar(self, payload_info: Dict[str, Any], point_global: np.ndarray) -> Optional[np.ndarray]:
        ego_rot = payload_info.get("ego2global_rotation")
        ego_trans = payload_info.get("ego2global_translation")
        lidar_rot = payload_info.get("lidar2ego_rotation")
        lidar_trans = payload_info.get("lidar2ego_translation")
        if None in (ego_rot, ego_trans, lidar_rot, lidar_trans):
            return None
        try:
            ego_rot = OccupancyCostmapPipeline._quaternion_to_matrix(ego_rot)
            lidar_rot = OccupancyCostmapPipeline._quaternion_to_matrix(lidar_rot)
        except Exception:
            return None
        ego_trans = np.asarray(ego_trans, dtype=np.float32)
        lidar_trans = np.asarray(lidar_trans, dtype=np.float32)
        point_ego = ego_rot.T @ (point_global - ego_trans)
        point_lidar = lidar_rot.T @ (point_ego - lidar_trans)
        return point_lidar

    def _write_attacked(self, source_path: Path, grid: np.ndarray) -> Path:
        timestamp = int(time.time() * 1000)
        dest = self.tmp_dir / f"{source_path.stem}_attack_{timestamp}{source_path.suffix}"
        counter = 1
        while dest.exists():
            dest = self.tmp_dir / f"{source_path.stem}_attack_{timestamp}_{counter}{source_path.suffix}"
            counter += 1
        np.save(dest, grid)
        return dest

    def snapshot(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "mode": self.mode,
            "anchor_mode": self.anchor_mode,
            "cumulative_travel_m": self._cumulative_travel_m,
        }
        if self._fixed_anchor_global is not None:
            data["fixed_anchor"] = self._fixed_anchor_global.tolist()
        if self.last_operation is not None:
            data["last_operation"] = self.last_operation
        return data


class OccupancyCostmapPipeline:
    """Run CARLA capture -> OpenOccupancy inference -> costmap generation in one call."""

    def __init__(
        self,
        notebooks_dir: Path,
        api_url: str = "http://127.0.0.1:8888/infer",
        workspace_root: Optional[Path] = None,
        tmp_dir: Optional[Path] = None,
        python_executable: str = sys.executable,
        post_capture_wait: float = 0.5,
        default_occ_size: Optional[List[int]] = None,
        default_pc_range: Optional[List[float]] = None,
        class_weight_json: Optional[Path] = None,
        cost_aggregate: str = "sum",
        attack_config: Optional[Path] = None,
        run_artifact_root: Optional[Path] = None,
        run_logger: Optional[RunLogger] = None,
    ) -> None:
        self.notebooks_dir = notebooks_dir
        self.api_url = api_url
        self.python_exec = python_executable
        self.workspace_root = workspace_root or notebooks_dir.parents[2]
        self.openocc_root = self.workspace_root / "OpenOccupancy"
        if not self.openocc_root.exists():
            raise FileNotFoundError(f"OpenOccupancy root not found at {self.openocc_root}")

        self.tmp_dir = tmp_dir or notebooks_dir / "tmp" / "pipeline"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        if run_artifact_root is not None:
            self.run_artifact_root = Path(run_artifact_root).resolve()
            self.run_artifact_root.mkdir(parents=True, exist_ok=True)
        else:
            self.run_artifact_root = None

        self.capture_script = notebooks_dir / "2d_prepare_single_frame.py"
        self.post_capture_wait = post_capture_wait
        self.default_occ_size = default_occ_size or [512, 512, 40]
        self.default_pc_range = default_pc_range or [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.class_weights = self._load_class_weights(class_weight_json)
        self.cost_aggregate = cost_aggregate
        self.run_logger = run_logger
        self.attack_simulator: Optional[AttackSimulator] = None
        if attack_config is not None:
            try:
                self.attack_simulator = AttackSimulator.load_from_path(attack_config, tmp_dir=self.tmp_dir)
                print(f"[AttackSimulator] Loaded config from {attack_config}")
            except Exception as exc:
                print(f"[AttackSimulator] Failed to initialize ({exc}); attacks disabled.")
                self.attack_simulator = None

        # Ensure OpenOccupancy is importable for the costmap utility.
        if str(self.workspace_root) not in sys.path:
            sys.path.append(str(self.workspace_root))

    # ----------------------------- helpers -----------------------------
    def _copy_prediction_to_tmp(self, source: Path, dest_dir: Optional[Path] = None) -> Path:
        """Copy the OpenOccupancy dense prediction into the working directory."""
        dest_root = dest_dir or self.tmp_dir
        timestamp = int(time.time())
        base_name = f"{source.stem}_{timestamp}"
        dest = dest_root / f"{base_name}{source.suffix}"
        counter = 1
        while dest.exists():
            dest = dest_root / f"{base_name}_{counter}{source.suffix}"
            counter += 1
        shutil.copy2(source, dest)
        return dest

    def _log_attack_event(self, frame_id: Optional[str]) -> None:
        if self.run_logger is None or self.attack_simulator is None:
            return
        info = self.attack_simulator.last_operation
        if not info:
            return
        payload = {"frame_id": frame_id}
        payload.update(info)
        self.run_logger.log("attack", **payload)

    @staticmethod
    def _load_class_weights(path: Optional[Path]) -> Optional[dict[int, float]]:
        if path is None:
            return None
        try:
            with Path(path).open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            weights = {}
            for k, v in data.items():
                try:
                    cls_id = int(k)
                except (TypeError, ValueError):
                    continue
                if isinstance(v, dict) and "weight" in v:
                    weights[cls_id] = float(v["weight"])
                elif isinstance(v, (int, float)):
                    weights[cls_id] = float(v)
            return weights if weights else None
        except Exception as exc:  # pragma: no cover
            print(f"[Costmap] Failed to load class weights from {path}: {exc}")
            return None

    @staticmethod
    def _load_payload_info(payload_path: Path) -> Dict[str, Any]:
        with payload_path.open("rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            if "infos" in payload and isinstance(payload["infos"], list):
                infos = payload["infos"]
            elif "payload" in payload and isinstance(payload["payload"], dict):
                infos = payload["payload"].get("infos", [])
            else:
                infos = []
        else:
            infos = []
        if not infos:
            raise ValueError("Payload does not contain 'infos'.")
        return infos[0]

    @staticmethod
    def _quaternion_to_matrix(quat: List[float]) -> np.ndarray:
        if len(quat) != 4:
            raise ValueError(f"Quaternion must have 4 elements, got {quat}")
        w, x, y, z = quat
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm == 0:
            raise ValueError("Quaternion has zero magnitude.")
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )

    def _compute_ego_indices(
        self, payload_info: Dict[str, Any], occ_size: List[int], pc_range: List[float]
    ) -> Tuple[float, float]:
        lidar2ego_trans = payload_info.get("lidar2ego_translation")
        lidar2ego_rot = payload_info.get("lidar2ego_rotation")
        if lidar2ego_trans is None or lidar2ego_rot is None:
            raise ValueError("Payload sample lacks lidar2ego transformation.")
        lidar2ego_rot = self._quaternion_to_matrix(lidar2ego_rot)
        lidar2ego_trans = np.asarray(lidar2ego_trans, dtype=np.float32)
        ego_center_lidar = -lidar2ego_rot.T @ lidar2ego_trans
        occ_arr = np.asarray(occ_size, dtype=np.float32)
        pc_arr = np.asarray(pc_range, dtype=np.float32)
        voxel_span = pc_arr[3:6] - pc_arr[0:3]
        voxel_span[voxel_span == 0] = 1.0
        voxel_size = voxel_span / occ_arr
        voxel_size[voxel_size == 0] = 1.0
        row_idx = (ego_center_lidar[0] - pc_arr[0]) / voxel_size[0]
        col_idx = (ego_center_lidar[1] - pc_arr[1]) / voxel_size[1]
        row_idx = float(np.clip(row_idx, 0.0, occ_arr[0] - 1.0))
        col_idx = float(np.clip(col_idx, 0.0, occ_arr[1] - 1.0))
        return row_idx, col_idx

    @staticmethod
    def _shift_dense_grid(grid: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
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
            dest_row_start:dest_row_end, dest_col_start:dest_col_end
        ] = grid[src_row_start:src_row_end, src_col_start:src_col_end]
        return shifted

    def _create_centered_prediction(
        self,
        prediction_path: Path,
        payload_path: Path,
        occ_size: Optional[List[int]],
        pc_range: Optional[List[float]],
        run_dir: Path,
        run_identifier: str,
    ) -> Optional[Path]:
        if occ_size is None or pc_range is None:
            return None
        try:
            payload_info = self._load_payload_info(payload_path)
            ego_row, ego_col = self._compute_ego_indices(payload_info, occ_size, pc_range)
        except Exception:
            return None
        target_row = (occ_size[0] - 1.0) / 2.0
        target_col = (occ_size[1] - 1.0) / 2.0
        row_shift = int(round(target_row - ego_row))
        col_shift = int(round(target_col - ego_col))
        if row_shift == 0 and col_shift == 0:
            return None
        grid = np.load(prediction_path)
        centered = self._shift_dense_grid(grid, row_shift, col_shift)
        ego_path = run_dir / f"pred_c_ego_{run_identifier}{prediction_path.suffix}"
        np.save(ego_path, centered)
        return ego_path

    @staticmethod
    def _infer_occ_size_from_npy(path: Path) -> List[int]:
        arr = np.load(path, mmap_mode="r")
        shape = arr.shape
        if len(shape) < 3:
            raise ValueError(f"Unexpected occupancy tensor shape {shape} (need at least 3 dims).")
        return [int(shape[0]), int(shape[1]), int(shape[2])]

    def _run_subprocess(self, args: List[str]) -> None:
        completed = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Command failed: {' '.join(args)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )

    # --------------------------- main stages ---------------------------
    def capture_frame(
        self,
        spawn_index: int,
        record_seconds: float,
        output_pkl: Optional[Path] = None,
        carla_dir: Optional[Path] = None,
        frame_id: Optional[str] = None,
        filename_template: Optional[str] = None,
    ) -> Path:
        output_pkl = output_pkl or self.tmp_dir / f"carla_capture_{int(time.time())}.pkl"
        args = [
            self.python_exec,
            str(self.capture_script),
        ]
        if carla_dir:
            image_dir = self.tmp_dir / f"manual_images_{int(time.time())}"
            args.extend(
                [
                    "--carla-dir",
                    str(carla_dir),
                    "--output-image-dir",
                    str(image_dir),
                ]
            )
            if frame_id:
                args.extend(["--frame-id", frame_id])
            if filename_template:
                args.extend(["--filename-template", filename_template])
        else:
            args.extend(
                [
                    "--spawn-index",
                    str(spawn_index),
                    "--record-seconds",
                    str(record_seconds),
                    "--post-capture-wait",
                    str(self.post_capture_wait),
                ]
            )
        args.extend(
            [
                "--output-pkl",
                str(output_pkl),
            ]
        )
        self._run_subprocess(args)
        return output_pkl

    def request_inference(
        self, payload_path: Path, timeout: float = 60.0
    ) -> tuple[Path, Optional[List[int]], Optional[List[float]]]:
        with payload_path.open("rb") as f:
            response = requests.post(
                self.api_url,
                files={"pkl_file": f},
                timeout=timeout,
            )
        response.raise_for_status()
        data = response.json()
        dense_paths = data.get("pred_c_dense_paths") or []
        if not dense_paths:
            raise RuntimeError(f"API response missing pred_c_dense_paths: {json.dumps(data, indent=2)}")
        dense_path = Path(dense_paths[0])
        if not dense_path.is_absolute():
            dense_path = (self.openocc_root / dense_path).resolve()
        if not dense_path.exists():
            raise FileNotFoundError(f"pred_c_dense.npy not found: {dense_path}")
        dense_path = self._copy_prediction_to_tmp(dense_path)
        occ_size = data.get("occ_size")
        pc_range = data.get("point_cloud_range")
        return dense_path, occ_size, pc_range

    def build_costmap(
        self,
        prediction_path: Path,
        payload_path: Path,
        window_size: int = 128,
        rotate_degrees: int = 0,
        align_carla_y: bool = False,
        mark_ego_arrow: bool = True,
        recenter_mode: str = "bbox",
        occ_size: Optional[List[int]] = None,
        pc_range: Optional[List[float]] = None,
        frame_id: Optional[str] = None,
        carla_dir: Optional[Path] = None,
    ) -> PipelineResult:
        run_dir = self._create_run_dir(frame_id=frame_id)
        run_identifier = run_dir.name.replace("run_", "")
        if prediction_path is not None:
            src_prediction = Path(prediction_path)
            prediction_local = run_dir / f"pred_c_dense_{run_identifier}{src_prediction.suffix}"
            shutil.copy2(src_prediction, prediction_local)
            if src_prediction.parent == self.tmp_dir:
                with contextlib.suppress(OSError):
                    src_prediction.unlink()
            prediction_path = prediction_local
        if occ_size is None and prediction_path is not None:
            with contextlib.suppress(Exception):
                occ_size = self._infer_occ_size_from_npy(prediction_path)
        prediction_ego = self._create_centered_prediction(
            prediction_path,
            payload_path,
            occ_size,
            pc_range,
            run_dir,
            run_identifier,
        )
        output_full = run_dir / f"costmap_full_{run_identifier}.npy"
        output_window = run_dir / f"costmap_ego_{run_identifier}.npy"
        output_image = run_dir / f"costmap_full_{run_identifier}.png"

        # ----------------- New costmap generation via CostmapGenerator -----------------
        from OpenOccupancy.src.utils.costmap import CostmapGenerator

        grid = np.load(prediction_path)
        grid_xy = (int(grid.shape[0]), int(grid.shape[1]))

        # Prefer explicit occ_size; fall back to the grid shape.
        if occ_size is None:
            with contextlib.suppress(Exception):
                occ_size = self._infer_occ_size_from_npy(prediction_path)
        if pc_range is None:
            pc_range = self.default_pc_range

        ego_center = None
        try:
            payload_info = self._load_payload_info(payload_path)
            base_occ_size = occ_size or list(grid.shape[:3])
            ego_center = self._compute_ego_indices(payload_info, base_occ_size, pc_range)
            # If metadata occ_size differs from the grid resolution, rescale the center.
            if base_occ_size and (base_occ_size[0] != grid_xy[0] or base_occ_size[1] != grid_xy[1]):
                scale_x = grid_xy[0] / float(base_occ_size[0])
                scale_y = grid_xy[1] / float(base_occ_size[1])
                ego_center = (ego_center[0] * scale_x, ego_center[1] * scale_y)
            # Notebook parity: coarse grid index is divided by 4 before cropping.
            ego_center = (ego_center[0] / 4.0, ego_center[1] / 4.0)
        except Exception:
            ego_center = None

        # If class weights provided, apply them before projection.
        cost_gen = None
        if self.class_weights:
            weights_arr = np.ones_like(grid, dtype=np.float32)
            for cls_id, w in self.class_weights.items():
                mask = grid == cls_id
                if np.any(mask):
                    weights_arr[mask] = float(w)
            weighted = weights_arr
        else:
            weighted = (grid != 0).astype(np.float32)

        # Projection to 2D:
        # - with class weights: aggregate along Z directly (sum or max)
        # - without weights: use CostmapGenerator reduce=(sum|max)/normalize=False
        agg = self.cost_aggregate.lower()
        if agg not in {"sum", "max"}:
            raise ValueError(f"Unsupported cost_aggregate={self.cost_aggregate}")
        if self.class_weights:
            if agg == "sum":
                full_cost = weighted.sum(axis=2)
            else:
                full_cost = weighted.max(axis=2)
        else:
            cost_gen = CostmapGenerator(reduce=agg, normalize=False)
            full_cost = cost_gen.make_topdown_costmap(weighted)
        output_full.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_full, full_cost.astype(np.float32))

        # Ego-centered crop (window)
        crop_dim = (min(64, grid_xy[0]), min(64, grid_xy[1]))
        if ego_center is None:
            window_cost = full_cost
            window_bounds = (0, full_cost.shape[0], 0, full_cost.shape[1])
        else:
            cx, cy = np.round(ego_center).astype(np.int64)
            hx, hy = int(crop_dim[0]) // 2, int(crop_dim[1]) // 2
            x0 = int(max(0, cx - hx))
            x1 = int(min(full_cost.shape[0], cx + hx))
            y0 = int(max(0, cy - hy))
            y1 = int(min(full_cost.shape[1], cy + hy))
            window_cost = full_cost[x0:x1, y0:y1]
            window_bounds = (x0, x1, y0, y1)

        np.save(output_window, window_cost.astype(np.float32))
        # Use static method to save (avoids needing cost_gen when weights are used)
        CostmapGenerator.save_costmap(window_cost, output_image)

        result = PipelineResult(
            payload_path=payload_path,
            prediction_dense_path=prediction_path,
            prediction_ego_path=prediction_ego,
            costmap_full_path=output_full,
            costmap_window_path=output_window,
            costmap_image_path=output_image,
            occ_size=occ_size,
            pc_range=pc_range,
        )
        result.front_camera_image_path = self._copy_front_camera_image(carla_dir, frame_id, run_dir)
        result.payload_artifact_path = self._copy_payload_artifact(payload_path, output_full)
        result.metadata_path = self._write_metadata(result)
        return result

    FRONT_CAMERA_NAME = "CAM_FRONT"
    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

    @staticmethod
    def _sanitize_identifier(identifier: Optional[str]) -> str:
        if not identifier:
            return str(int(time.time()))
        ident = str(identifier)
        ident = ident.strip()
        ident = re.sub(r"[^A-Za-z0-9_-]+", "_", ident)
        ident = ident.strip("_")
        if ident.isdigit():
            return f"{int(ident):06d}"
        return ident or str(int(time.time()))

    def _create_run_dir(self, frame_id: Optional[str] = None) -> Path:
        """Create a dedicated directory for a single inference run."""
        if self.run_artifact_root is None:
            timestamp = int(time.time())
            run_dir = self.tmp_dir / f"run_{timestamp}"
            counter = 1
            while run_dir.exists():
                run_dir = self.tmp_dir / f"run_{timestamp}_{counter}"
                counter += 1
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        base = self.run_artifact_root
        identifier = self._sanitize_identifier(frame_id)
        if identifier and not identifier.startswith("tick_"):
            dir_name = f"tick_{identifier}"
        else:
            dir_name = identifier or f"tick_{int(time.time())}"
        run_dir = base / dir_name
        counter = 1
        while run_dir.exists():
            run_dir = base / f"{dir_name}_{counter}"
            counter += 1
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _copy_front_camera_image(
        self, carla_dir: Optional[Path], frame_id: Optional[str], run_dir: Path
    ) -> Optional[Path]:
        if carla_dir is None or frame_id is None:
            return None
        cam_dir = Path(carla_dir) / self.FRONT_CAMERA_NAME
        if not cam_dir.exists():
            return None
        source = None
        for ext in self.IMAGE_EXTENSIONS:
            candidate = cam_dir / f"{frame_id}{ext}"
            if candidate.exists():
                source = candidate
                break
        if source is None:
            return None
        dest = run_dir / f"{self.FRONT_CAMERA_NAME}_{frame_id}{source.suffix.lower()}"
        try:
            shutil.copy2(source, dest)
        except Exception:
            return None
        return dest

    def _copy_payload_artifact(self, payload_path: Path, reference_output: Optional[Path]) -> Optional[Path]:
        """Place a copy of the CARLA payload alongside the generated costmap files."""
        if reference_output is None or not payload_path.exists():
            return None
        dest = reference_output.with_name(reference_output.stem + "_payload.pkl")
        shutil.copy2(payload_path, dest)
        return dest

    def _write_metadata(self, result: PipelineResult) -> Path:
        """Save metadata about the most recent costmap/occupancy artifacts."""
        metadata = {
            "payload_path": str(result.payload_path),
            "prediction_dense_path": str(result.prediction_dense_path) if result.prediction_dense_path else None,
            "prediction_ego_path": str(result.prediction_ego_path) if result.prediction_ego_path else None,
            "costmap_full_path": str(result.costmap_full_path) if result.costmap_full_path else None,
            "costmap_window_path": str(result.costmap_window_path) if result.costmap_window_path else None,
            "costmap_image_path": str(result.costmap_image_path) if result.costmap_image_path else None,
            "occ_size": result.occ_size,
            "pc_range": result.pc_range,
            "payload_artifact_path": str(result.payload_artifact_path) if result.payload_artifact_path else None,
            "front_camera_image_path": str(result.front_camera_image_path) if result.front_camera_image_path else None,
            "run_directory": str(result.costmap_full_path.parent) if result.costmap_full_path else None,
            "timestamp": int(time.time()),
        }
        if result.costmap_full_path is not None:
            metadata_path = result.costmap_full_path.with_suffix(".json")
        else:
            metadata_path = self.tmp_dir / f"costmap_metadata_{int(time.time())}.json"
        with metadata_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        return metadata_path

    def run_once(
        self,
        spawn_index: int,
        record_seconds: float,
        window_size: int = 128,
        rotate_degrees: int = 0,
        align_carla_y: bool = False,
        recenter_mode: str = "bbox",
        carla_dir: Optional[Path] = None,
        frame_id: Optional[str] = None,
        filename_template: Optional[str] = None,
        enable_inference: bool = True,
    ) -> PipelineResult:
        payload = self.capture_frame(
            spawn_index=spawn_index,
            record_seconds=record_seconds,
            carla_dir=carla_dir,
            frame_id=frame_id,
            filename_template=filename_template,
        )
        if not enable_inference:
            return PipelineResult(payload_path=payload)
        prediction, occ_size, pc_range = self.request_inference(payload)
        if self.attack_simulator is not None:
            payload_info = None
            try:
                payload_info = self._load_payload_info(payload)
            except Exception as exc:
                print(f"[AttackSimulator] Failed to load payload info ({exc}); skipping attack.")
            if payload_info is not None:
                try:
                    prediction = self.attack_simulator.apply(
                        prediction_path=prediction,
                        payload_info=payload_info,
                        occ_size=occ_size,
                        pc_range=pc_range or self.default_pc_range,
                    )
                    self._log_attack_event(frame_id)
                except Exception as exc:
                    print(f"[AttackSimulator] Attack application failed: {exc}")
        if occ_size is None:
            try:
                occ_size = self._infer_occ_size_from_npy(prediction)
            except Exception:
                occ_size = self.default_occ_size
        pc_range = pc_range or self.default_pc_range
        if recenter_mode == "ego" and (occ_size is None or pc_range is None):
            raise RuntimeError("Ego recentering requires occ_size and point_cloud_range from the inference API.")
        return self.build_costmap(
            prediction_path=prediction,
            payload_path=payload,
            window_size=window_size,
            rotate_degrees=rotate_degrees,
            align_carla_y=align_carla_y,
            recenter_mode=recenter_mode,
            occ_size=occ_size,
            pc_range=pc_range,
            frame_id=frame_id,
            carla_dir=carla_dir,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CARLA -> OpenOccupancy -> costmap pipeline.")
    parser.add_argument("--spawn-index", type=int, default=361)
    parser.add_argument("--record-seconds", type=float, default=1.0)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--rotate-degrees", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--align-carla-y", action="store_true")
    parser.add_argument(
        "--recenter",
        type=str,
        choices=["bbox", "none", "ego"],
        default="bbox",
        help="Recenter mode passed to costmap generation.",
    )
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8888/infer")
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--post-capture-wait", type=float, default=0.5)
    parser.add_argument(
        "--occ-size",
        nargs=3,
        type=int,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Default occupancy grid size used when API does not return occ_size.",
    )
    parser.add_argument(
        "--pc-range",
        nargs=6,
        type=float,
        default=None,
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
        help="Default point cloud range used when API does not return one.",
    )
    parser.add_argument(
        "--carla-dir",
        type=Path,
        default=None,
        help="Use existing CARLA image directory instead of capturing a new frame.",
    )
    parser.add_argument(
        "--frame-id",
        type=str,
        default=None,
        help="Frame identifier to substitute into filename templates when using --carla-dir.",
    )
    parser.add_argument(
        "--filename-template",
        type=str,
        default=None,
        help=(
            "Template describing relative camera image paths (supports {cam}, {cam_lower}, "
            "{cam_short}, {frame})."
        ),
    )
    parser.add_argument(
        "--class-weight-json",
        type=Path,
        default=None,
        help="JSON mapping class id to weight for costmap generation (keys can be id or {id: {weight}}).",
    )
    parser.add_argument(
        "--attack-config",
        type=Path,
        default=None,
        help="Optional JSON/YAML config describing the AttackSimulator behavior.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebooks_dir = Path(__file__).resolve().parent
    pipeline = OccupancyCostmapPipeline(
        notebooks_dir=notebooks_dir,
        api_url=args.api_url,
        workspace_root=args.workspace_root,
        tmp_dir=args.tmp_dir,
        post_capture_wait=args.post_capture_wait,
        default_occ_size=args.occ_size,
        default_pc_range=args.pc_range,
        class_weight_json=args.class_weight_json,
        attack_config=args.attack_config,
    )
    result = pipeline.run_once(
        spawn_index=args.spawn_index,
        record_seconds=args.record_seconds,
        window_size=args.window_size,
        rotate_degrees=args.rotate_degrees,
        align_carla_y=args.align_carla_y,
        recenter_mode=args.recenter,
        carla_dir=args.carla_dir,
        frame_id=args.frame_id,
        filename_template=args.filename_template,
    )
    print("Payload:", result.payload_path)
    print("Prediction (dense):", result.prediction_dense_path)
    print("Costmap (full):", result.costmap_full_path)
    print("Costmap (window):", result.costmap_window_path)
    print("Costmap image:", result.costmap_image_path)
    if result.occ_size:
        print("occ_size:", result.occ_size)
    if result.pc_range:
        print("point_cloud_range:", result.pc_range)


if __name__ == "__main__":
    main()
