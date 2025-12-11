"""
Lightweight longitudinal planner inspired by ST-P3 sampling.

This module is shared so notebooks and driving scripts can import the same logic.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class STP3StyleLongitudinalPlanner:
    """
    Straight-line longitudinal planner that scores sampled constant-acc trajectories
    using a 2D occupancy costmap (ego-centered). Grid resolution is fixed to the
    current setup: 0.8 m / voxel (0.2 m * 4x downsample).

    NOTE: Original ST-P3 planning cost focuses on occupancy / semantic / HD map
    volumes; it does not have an explicit V_ref penalty. The speed cost here is
    an extra bias to pull the solution toward a preferred cruise speed.
    """

    grid_resolution: float = 0.8  # meters per voxel (after 4x compression)
    dt: float = 0.1  # tick time [s]
    horizon_s: float = 2.0  # planning horizon [s]
    v_ref: float = 5.0  # preferred speed [m/s]
    v_max: float = 12.0  # clip speed
    accelerations: Optional[Sequence[float]] = None  # m/s^2 candidates
    weight_occ: float = 6.0
    weight_speed: float = 1.0
    weight_acc: float = 0.15
    weight_jerk: float = 0.05
    bumper_offset_m: float = 0.0  # shift sampling forward to account for bumper/vehicle length
    corridor_half_width_m: float = 1.0  # sample a lateral band around ego center to catch nearby obstacles
    forward_axis: str = "row"  # "row" = +X along axis 0, "col" = +X along axis 1

    horizon_steps: int = field(init=False)
    accel_set: np.ndarray = field(init=False)
    last_scores: list = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.horizon_steps = max(1, int(round(self.horizon_s / self.dt)))
        if self.accelerations is None:
            # Emphasize braking options; ST-P3 typically samples many slowing paths.
            self.accelerations = np.linspace(-6.0, 2.0, 9)
        self.accel_set = np.asarray(self.accelerations, dtype=np.float32)
        if self.forward_axis not in {"row", "col"}:
            raise ValueError(f"forward_axis must be 'row' or 'col', got {self.forward_axis}")

    def _rollout(self, v0: float, accel: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = np.arange(self.horizon_steps + 1, dtype=np.float32) * self.dt
        v = np.clip(v0 + accel * t, 0.0, self.v_max)
        # distance traveled along +X (forward) axis; prepend 0 for ego position
        x = np.concatenate([[0.0], np.cumsum(v[:-1] * self.dt)])
        a = np.diff(v, prepend=v[0]) / self.dt
        j = np.diff(a, prepend=a[0]) / self.dt
        return v, x, j

    def _gather_cost(
        self, costmap: np.ndarray, v: np.ndarray, x: np.ndarray, ego_xy: Optional[Tuple[float, float]] = None
    ) -> Tuple[float, dict]:
        h, w = costmap.shape
        if ego_xy is None:
            ego_xy = (h / 2.0, w / 2.0)
        # Densify the trajectory so we don't skip cells between steps, and sum all sampled costs.
        step_m = max(self.grid_resolution * 0.25, 0.05)  # sample roughly every 0.25 voxel
        dense_x = np.arange(0.0, float(x[-1]) + step_m, step_m)
        offset_x = self.bumper_offset_m / self.grid_resolution if self.bumper_offset_m else 0.0
        corridor_vox = max(0, int(round(self.corridor_half_width_m / self.grid_resolution)))

        occ_along_path = []
        if self.forward_axis == "row":
            ix = np.rint(ego_xy[0] + offset_x + dense_x / self.grid_resolution).astype(np.int32)
            iy_center = int(np.rint(ego_xy[1]))
            offsets_y = np.arange(-corridor_vox, corridor_vox + 1, dtype=np.int32)

            for row in ix:
                rr = int(np.clip(row, 0, h - 1))
                cols = iy_center + offsets_y
                cols_clipped = np.clip(cols, 0, w - 1)
                # take worst obstacle across corridor at this longitudinal sample
                occ_along_path.append(float(costmap[rr, cols_clipped].max()))
        else:  # forward_axis == "col"
            iy = np.rint(ego_xy[1] + offset_x + dense_x / self.grid_resolution).astype(np.int32)
            ix_center = int(np.rint(ego_xy[0]))
            offsets_x = np.arange(-corridor_vox, corridor_vox + 1, dtype=np.int32)

            for col in iy:
                cc = int(np.clip(col, 0, w - 1))
                rows = ix_center + offsets_x
                rows_clipped = np.clip(rows, 0, h - 1)
                occ_along_path.append(float(costmap[rows_clipped, cc].max()))
        if not occ_along_path:
            return np.inf, {"occ": np.inf, "speed": 0.0, "acc": 0.0, "jerk": 0.0}
        occ_cost = float(np.sum(occ_along_path))

        speed_cost = float(np.mean((v - self.v_ref) ** 2))
        acc = np.diff(v, prepend=v[0]) / self.dt
        acc_cost = float(np.mean(np.abs(acc)))
        jerk = np.diff(acc, prepend=acc[0]) / self.dt
        jerk_cost = float(np.mean(np.abs(jerk))) if jerk.size else 0.0

        total = (
            self.weight_occ * occ_cost
            + self.weight_speed * speed_cost
            + self.weight_acc * acc_cost
            + self.weight_jerk * jerk_cost
        )
        return total, {"occ": occ_cost, "speed": speed_cost, "acc": acc_cost, "jerk": jerk_cost}

    def plan(self, costmap: np.ndarray, v0: float, ego_xy: Optional[Tuple[float, float]] = None):
        scores = []
        for a in self.accel_set:
            v, x, _ = self._rollout(v0, a)
            total, terms = self._gather_cost(costmap, v, x, ego_xy=ego_xy)
            scores.append(
                {
                    "accel": float(a),
                    "total_cost": float(total),
                    "terms": terms,
                    "v": v,
                    "x": x,
                    "next_speed": float(v[1]) if v.size > 1 else float(v[-1]),
                }
            )
        scores = sorted(scores, key=lambda s: s["total_cost"])
        self.last_scores = scores
        return scores[0], scores


__all__ = ["STP3StyleLongitudinalPlanner"]
