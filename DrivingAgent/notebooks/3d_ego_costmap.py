from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from DrivingAgent.src.occupancy_costmap import (  # noqa: E402
    DEFAULT_OBSTACLE_CLASSES,
    DEFAULT_PC_RANGE,
    build_class_weights,
    build_ego_centered_costmap,
    extract_fixed_window,
    infer_occ_size_from_grid,
    load_payload_info,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an ego-centered occupancy grid and fixed-size cost map."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to pred_c_dense.npy")
    parser.add_argument("--payload", type=Path, required=True, help="Payload PKL/JSON path")
    parser.add_argument("--sample-index", type=int, default=0, help="Payload sample index")
    parser.add_argument("--output-cost", type=Path, required=True, help="Output cost map .npy path")
    parser.add_argument(
        "--output-centered-grid",
        type=Path,
        default=None,
        help="Optional path to save the ego-centered occupancy grid",
    )
    parser.add_argument(
        "--output-window",
        type=Path,
        default=None,
        help="Optional path to save the fixed-size ego window cost map",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=None,
        help="Optional path to save an 8-bit PNG of the fixed-size window",
    )
    parser.add_argument("--window-size", type=int, default=256, help="Fixed window size (pixels)")
    parser.add_argument(
        "--aggregate",
        choices=["max", "sum"],
        default="max",
        help="Cost aggregation strategy when classes overlap",
    )
    parser.add_argument(
        "--default-obstacle-cost",
        type=float,
        default=1.0,
        help="Default cost weight assigned to obstacle classes",
    )
    parser.add_argument(
        "--class-weight",
        nargs="*",
        default=[],
        help="Override class weights via name=value, e.g., pedestrian=2.0",
    )
    parser.add_argument(
        "--ignore-classes",
        nargs="*",
        default=["vegetation"],
        help="Classes ignored when building the cost map",
    )
    parser.add_argument(
        "--obstacle-classes",
        nargs="*",
        default=list(DEFAULT_OBSTACLE_CLASSES),
        help="Classes treated as obstacles before weighting",
    )
    parser.add_argument(
        "--occ-size",
        nargs=3,
        type=int,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override occupancy grid size",
    )
    parser.add_argument(
        "--pc-range",
        nargs=6,
        type=float,
        default=None,
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
        help="Override point-cloud range",
    )
    parser.add_argument(
        "--max-visual-cost",
        type=float,
        default=1.0,
        help="Cost value mapped to 255 in the visualization",
    )
    parser.add_argument(
        "--dump-metadata",
        type=Path,
        default=None,
        help="Optional JSON file summarizing outputs",
    )
    return parser.parse_args()


def normalize_cost(cost: np.ndarray, max_value: float) -> np.ndarray:
    if max_value <= 0:
        max_value = float(cost.max()) or 1.0
    normalized = np.clip(cost / max_value, 0.0, 1.0)
    return (normalized * 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    grid = np.load(args.input)
    payload_info = load_payload_info(args.payload, sample_index=args.sample_index)
    occ_size = list(args.occ_size) if args.occ_size else infer_occ_size_from_grid(grid)
    pc_range = list(args.pc_range) if args.pc_range else DEFAULT_PC_RANGE

    result = build_ego_centered_costmap(
        grid,
        payload_info,
        occ_size=occ_size,
        pc_range=pc_range,
        obstacle_classes=args.obstacle_classes,
        ignore_classes=args.ignore_classes,
        custom_class_weights=args.class_weight,
        default_cost=args.default_obstacle_cost,
        aggregate=args.aggregate,
        window_size=args.window_size,
    )

    cost_map = result["cost_map"]
    np.save(args.output_cost, cost_map)
    if args.output_centered_grid:
        np.save(args.output_centered_grid, result["centered_grid"])

    if args.output_window:
        window = result["fixed_window"]
        if window is None:
            window = extract_fixed_window(cost_map, args.window_size)
        np.save(args.output_window, window)
        if args.output_image:
            image = normalize_cost(window, args.max_visual_cost)
            args.output_image.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(args.output_image)
    elif args.output_image:
        image = normalize_cost(cost_map, args.max_visual_cost)
        args.output_image.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(args.output_image)

    if args.dump_metadata:
        metadata = {
            "input": str(args.input),
            "payload": str(args.payload),
            "sample_index": args.sample_index,
            "output_cost": str(args.output_cost),
            "output_centered_grid": str(args.output_centered_grid) if args.output_centered_grid else None,
            "output_window": str(args.output_window) if args.output_window else None,
            "output_image": str(args.output_image) if args.output_image else None,
            "window_size": args.window_size,
            "aggregate": args.aggregate,
            "shift": result["shift"],
            "occ_size": result["occ_size"],
            "pc_range": result["pc_range"],
        }
        args.dump_metadata.parent.mkdir(parents=True, exist_ok=True)
        args.dump_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Ego-centered cost map saved to", args.output_cost)
    if args.output_window:
        print("Fixed window saved to", args.output_window)
    print("Applied shift (row, col):", result["shift"])


if __name__ == "__main__":
    main()
