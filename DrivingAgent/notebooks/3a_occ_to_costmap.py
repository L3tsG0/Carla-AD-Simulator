import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

# nuScenes/OpenOccupancy semantic classes.
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

# Default obstacle classes (everything except static surfaces/terrain/vegetation).
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


def parse_class_weights(
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
            raise ValueError(f"Invalid --class-weight '{item}', expected name=value.")
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()
        if name not in CLASS_TO_ID:
            raise ValueError(f"Unknown class '{name}' in --class-weight.")
        weights[CLASS_TO_ID[name]] = float(value)
    return weights


def project_costmap(
    grid: np.ndarray,
    class_weights: Dict[int, float],
    aggregate: str = "max",
) -> np.ndarray:
    if grid.ndim != 3:
        raise ValueError(f"Expected 3D grid (W,H,D) but got shape {grid.shape}")
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


def normalize_cost(cost: np.ndarray, max_value: float) -> np.ndarray:
    if max_value <= 0:
        max_value = float(cost.max()) or 1.0
    normalized = np.clip(cost / max_value, 0.0, 1.0)
    return (normalized * 255).astype(np.uint8)


def crop_costmap(cost: np.ndarray, margin: int = 0):
    nonzero = np.argwhere(cost > 0)
    if nonzero.size == 0:
        return cost.copy(), (0, cost.shape[0], 0, cost.shape[1])
    min_row = max(int(nonzero[:, 0].min()) - margin, 0)
    max_row = min(int(nonzero[:, 0].max()) + margin + 1, cost.shape[0])
    min_col = max(int(nonzero[:, 1].min()) - margin, 0)
    max_col = min(int(nonzero[:, 1].max()) + margin + 1, cost.shape[1])
    cropped = cost[min_row:max_row, min_col:max_col]
    return cropped, (min_row, max_row, min_col, max_col)


def shift_costmap(cost: np.ndarray, shift: Tuple[int, int]) -> np.ndarray:
    """Shift the cost map by integer pixels without wrapping."""
    row_shift, col_shift = shift
    if row_shift == 0 and col_shift == 0:
        return cost.copy()

    rows, cols = cost.shape
    shifted = np.zeros_like(cost)

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

    shifted[dest_row_start:dest_row_end, dest_col_start:dest_col_end] = cost[
        src_row_start:src_row_end, src_col_start:src_col_end
    ]
    return shifted


def recenter_costmap(
    cost: np.ndarray,
    mode: str,
    config_center: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Recenters the cost map according to the selected mode."""
    if mode == "none":
        return cost.copy(), (0, 0)

    if mode == "bbox":
        nonzero = np.argwhere(cost > 0)
        if nonzero.size == 0:
            return cost.copy(), (0, 0)
        bbox_min = nonzero.min(axis=0).astype(np.float64)
        bbox_max = nonzero.max(axis=0).astype(np.float64)
        bbox_center = (bbox_min + bbox_max) / 2.0
        grid_center = (np.array(cost.shape[:2], dtype=np.float64) - 1.0) / 2.0
        shift = np.round(grid_center - bbox_center).astype(int)
        shifted = shift_costmap(cost, (int(shift[0]), int(shift[1])))
        return shifted, (int(shift[0]), int(shift[1]))
    if mode == "ego":
        if config_center is None:
            return cost.copy(), (0, 0)
        current_center = (np.array(cost.shape[:2], dtype=np.float64) - 1.0) / 2.0
        target_center = np.array(config_center, dtype=np.float64)
        shift = np.round(target_center - current_center).astype(int)
        shifted = shift_costmap(cost, (int(shift[0]), int(shift[1])))
        return shifted, (int(shift[0]), int(shift[1]))

    raise ValueError(f"Unknown recentering mode '{mode}'")


def compute_config_center(
    cost_shape: Tuple[int, int],
    occ_size: Optional[Sequence[int]],
    pc_range: Optional[Sequence[float]] = None,
) -> Optional[Tuple[float, float]]:
    if not occ_size or len(occ_size) < 2:
        return None
    occ_h, occ_w = float(occ_size[0]), float(occ_size[1])
    if occ_h <= 1 or occ_w <= 1:
        return None
    scale_row = cost_shape[0] / occ_h
    scale_col = cost_shape[1] / occ_w

    if pc_range and len(pc_range) >= 4:
        x_min, y_min = float(pc_range[0]), float(pc_range[1])
        x_max, y_max = float(pc_range[3]), float(pc_range[4])
        span_x = x_max - x_min
        span_y = y_max - y_min
        if span_x > 0 and span_y > 0:
            voxel_x = span_x / occ_h
            voxel_y = span_y / occ_w
            # determine which voxel contains ego origin (0,0)
            row_idx = ((0.0 - x_min) / voxel_x) - 0.5
            col_idx = ((0.0 - y_min) / voxel_y) - 0.5
            row_idx = max(0.0, min(occ_h - 1.0, row_idx))
            col_idx = max(0.0, min(occ_w - 1.0, col_idx))
            target_row = row_idx * scale_row
            target_col = col_idx * scale_col
            return (target_row, target_col)

    target_row = ((occ_h - 1.0) / 2.0) * scale_row
    target_col = ((occ_w - 1.0) / 2.0) * scale_col
    return (target_row, target_col)


def _quaternion_to_matrix(quat: Sequence[float]) -> np.ndarray:
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


def load_ego_center_from_payload(
    path: Path,
    occ_size: Optional[Sequence[int]] = None,
    pc_range: Optional[Sequence[float]] = None,
    sample_index: int = 0,
) -> Tuple[float, float]:
    if not path.exists():
        raise FileNotFoundError(f"Payload file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            payload = pickle.load(f)
    else:
        payload = json.loads(path.read_text())

    infos = _extract_infos(payload)
    if not infos:
        raise ValueError("Payload contains no infos entries.")
    if not 0 <= sample_index < len(infos):
        raise IndexError(f"Sample index {sample_index} out of range (len={len(infos)}).")
    info = infos[sample_index]
    lidar2ego_trans = info.get("lidar2ego_translation")
    lidar2ego_rot = info.get("lidar2ego_rotation")
    if lidar2ego_trans is None or lidar2ego_rot is None:
        raise ValueError("Payload sample lacks lidar2ego transformation.")

    lidar2ego_rot = _quaternion_to_matrix(lidar2ego_rot)
    lidar2ego_trans = np.array(lidar2ego_trans, dtype=np.float32)
    ego_center_lidar = -lidar2ego_rot.T @ lidar2ego_trans

    if occ_size is None:
        occ_size = info.get("occ_size")
        if occ_size is None:
            raise ValueError("--occ-size not provided and payload lacks occ_size.")
    if pc_range is None:
        pc_range = info.get("pc_range") or info.get("point_cloud_range")
        if pc_range is None:
            raise ValueError("--pc-range not provided and payload lacks pc_range.")

    occ_arr = np.asarray(occ_size, dtype=np.float32)
    pc_range_arr = np.asarray(pc_range, dtype=np.float32)
    if occ_arr.shape[0] < 2 or pc_range_arr.shape[0] < 6:
        raise ValueError("Invalid occ_size or pc_range for ego center computation.")

    voxel_span = pc_range_arr[3:6] - pc_range_arr[0:3]
    voxel_span[voxel_span == 0] = 1.0
    voxel_size = voxel_span / occ_arr
    voxel_size[voxel_size == 0] = 1.0

    row_idx = (ego_center_lidar[0] - pc_range_arr[0]) / voxel_size[0]
    col_idx = (ego_center_lidar[1] - pc_range_arr[1]) / voxel_size[1]
    row_idx = float(np.clip(row_idx, 0.0, occ_arr[0] - 1.0))
    col_idx = float(np.clip(col_idx, 0.0, occ_arr[1] - 1.0))
    return row_idx, col_idx


def _extract_infos(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        if "infos" in payload:
            infos = payload["infos"]
        elif "payload" in payload and isinstance(payload["payload"], dict):
            infos = payload["payload"].get("infos", [])
        else:
            infos = []
        if isinstance(infos, list):
            return infos
    raise ValueError("Could not locate 'infos' list in payload.")


def load_ego_yaw_from_payload(path: Path, sample_index: int = 0) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Payload file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            payload = pickle.load(f)
    else:
        payload = json.loads(path.read_text())

    infos = _extract_infos(payload)
    if not infos:
        raise ValueError("Payload contains no infos entries.")
    if sample_index < 0 or sample_index >= len(infos):
        raise IndexError(f"Sample index {sample_index} out of range (len={len(infos)}).")
    info = infos[sample_index]
    can_bus = info.get("can_bus")
    if isinstance(can_bus, list) and len(can_bus) >= 18:
        return float(can_bus[-1])

    rotation = info.get("ego2global_rotation") or info.get("lidar2ego_rotation")
    if isinstance(rotation, (list, tuple)) and len(rotation) == 4:
        w, x, y, z = rotation
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.degrees(math.atan2(siny, cosy))
        if yaw < 0:
            yaw += 360.0
        return yaw

def load_occ_params_from_payload(
    path: Path, sample_index: int = 0
) -> Tuple[Optional[Sequence[int]], Optional[Sequence[float]]]:
    if not path.exists():
        raise FileNotFoundError(f"Payload file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            payload = pickle.load(f)
    else:
        payload = json.loads(path.read_text())
    infos = _extract_infos(payload)
    if not infos:
        raise ValueError("Payload contains no infos entries.")
    if not 0 <= sample_index < len(infos):
        raise IndexError(f"Sample index {sample_index} out of range (len={len(infos)}).")
    info = infos[sample_index]
    occ_size = info.get("occ_size")
    pc_range = info.get("pc_range") or info.get("point_cloud_range")
    return occ_size, pc_range
    raise ValueError("Selected sample does not contain yaw information.")


def add_ego_arrow(image: Image.Image, color=(255, 0, 0), yaw_deg: Optional[float] = None) -> Image.Image:
    """Overlay a red arrow pointing forward (negative y) at the image center."""
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)

    width, height = img.size
    min_dim = min(width, height)
    length = max(10, int(min_dim * 0.15))
    shaft_width = max(1, int(min_dim * 0.01))
    head_size = max(4, int(shaft_width * 2))

    center_x = width // 2
    center_y = height // 2
    half_len = length / 2.0
    if yaw_deg is None:
        theta = -math.pi / 2  # default forward (negative y)
    else:
        theta = math.radians(yaw_deg)
        theta = -theta + math.pi / 2  # convert to image coordinates

    dx = half_len * math.cos(theta)
    dy = half_len * math.sin(theta)

    start = (center_x - dx, center_y - dy)
    end = (center_x + dx, center_y + dy)

    draw.line([start, end], fill=color, width=shaft_width)
    head_theta_left = theta + math.radians(150)
    head_theta_right = theta - math.radians(150)
    head = [
        end,
        (
            end[0] + head_size * math.cos(head_theta_left),
            end[1] + head_size * math.sin(head_theta_left),
        ),
        (
            end[0] + head_size * math.cos(head_theta_right),
            end[1] + head_size * math.sin(head_theta_right),
        ),
    ]
    draw.polygon(head, fill=color)
    dot_radius = max(2, shaft_width)
    dot_box = [
        (center_x - dot_radius, center_y - dot_radius),
        (center_x + dot_radius, center_y + dot_radius),
    ]
    draw.ellipse(dot_box, fill=color)
    return img


def extract_fixed_window(cost: np.ndarray, size: int) -> np.ndarray:
    if size <= 0 or size > min(cost.shape[:2]):
        raise ValueError(f"Invalid fixed-window size {size} for shape {cost.shape}")
    half = size // 2
    center_row = cost.shape[0] // 2
    center_col = cost.shape[1] // 2
    row_start = max(center_row - half, 0)
    row_end = row_start + size
    col_start = max(center_col - half, 0)
    col_end = col_start + size
    return cost[row_start:row_end, col_start:col_end]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert 3D occupancy grid (pred_c_dense.npy) to a 2D cost map."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to pred_c_dense.npy created by the inference API.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to save the 2D cost map in .npy format.",
    )
    parser.add_argument(
        "--output-cropped",
        type=Path,
        default=None,
        help="Optional path to save a cropped cost map focusing on occupied region.",
    )
    parser.add_argument(
        "--output-fixed-window",
        type=Path,
        default=None,
        help="Optional path to save a fixed-size ego-centric window. "
        "If omitted, the script still extracts the window for visualization.",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=None,
        help="Optional path to save an 8-bit visualization (PNG).",
    )
    parser.add_argument(
        "--aggregate",
        choices=["max", "sum"],
        default="max",
        help="How to combine costs when multiple classes overlap along z-axis.",
    )
    parser.add_argument(
        "--default-obstacle-cost",
        type=float,
        default=1.0,
        help="Default penalty applied to obstacle classes.",
    )
    parser.add_argument(
        "--class-weight",
        nargs="*",
        default=[],
        help="Override weights via name=value, e.g., pedestrian=2.0 vegetation=0.0",
    )
    parser.add_argument(
        "--ignore-classes",
        nargs="*",
        default=["vegetation"],
        help="Classes to ignore entirely when generating the cost map.",
    )
    parser.add_argument(
        "--obstacle-classes",
        nargs="*",
        default=list(DEFAULT_OBSTACLE_CLASSES),
        help="Class names treated as obstacles (before weighting).",
    )
    parser.add_argument(
        "--max-visual-cost",
        type=float,
        default=1.0,
        help="Maximum cost mapped to 255 in the visualization output. "
        "If <= 0, the script uses the max value from the generated cost map.",
    )
    parser.add_argument(
        "--crop-margin",
        type=int,
        default=0,
        help="Margin (in pixels) to include around the occupied bounding box when"
        " saving the cropped output.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=256,
        help="Size (pixels) of the fixed ego-centric window saved when "
        "--output-fixed-window is provided. Must be even and <= grid size.",
    )
    parser.add_argument(
        "--recenter",
        choices=["none", "bbox", "ego"],
        default="none",
        help="Recentering strategy applied before cropping/windowing. "
        "'bbox' aligns the occupied bounding box center to the grid center.",
    )
    parser.add_argument(
        "--align-carla-y",
        action="store_true",
        help="Flip the cost map horizontally so image axes match CARLA camera view (Y right).",
    )
    parser.add_argument(
        "--rotate-degrees",
        type=int,
        choices=[0, 90, 180, 270],
        default=0,
        help="Rotate the cost map counter-clockwise by the specified degrees.",
    )
    parser.add_argument(
        "--mark-ego-arrow",
        action="store_true",
        help="Overlay a red arrow at the ego center in the visualization output.",
    )
    parser.add_argument(
        "--ego-yaw",
        type=float,
        default=None,
        help="Optional ego yaw in degrees (CARLA convention). Used for arrow direction.",
    )
    parser.add_argument(
        "--ego-yaw-from-payload",
        type=Path,
        default=None,
        help="Optional payload file (PKL/JSON) to read yaw from its can_bus entry.",
    )
    parser.add_argument(
        "--ego-yaw-sample-index",
        type=int,
        default=0,
        help="Sample index used together with --ego-yaw-from-payload.",
    )
    parser.add_argument(
        "--ego-center-from-payload",
        type=Path,
        default=None,
        help="Payload file used to extract ego center for recentering.",
    )
    parser.add_argument(
        "--ego-center-sample-index",
        type=int,
        default=0,
        help="Sample index used together with --ego-center-from-payload.",
    )
    parser.add_argument(
        "--pc-range",
        nargs=6,
        type=float,
        default=None,
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
        help="Point cloud range (6 floats) used for inference.",
    )
    parser.add_argument(
        "--occ-size",
        nargs=3,
        type=int,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Occupancy size (3 ints) used for inference.",
    )
    parser.add_argument(
        "--dump-config",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary of parameters and outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = np.load(args.input)
    if grid.ndim < 3:
        raise ValueError(f"Expected occupancy grid with at least 3 dims, got shape {grid.shape}")
    inferred_occ_size = [int(grid.shape[0]), int(grid.shape[1]), int(grid.shape[2])]
    occ_size = list(args.occ_size) if args.occ_size else inferred_occ_size
    class_weights = parse_class_weights(
        obstacle_classes=args.obstacle_classes,
        ignore_classes=args.ignore_classes,
        custom_weights=args.class_weight,
        default_cost=args.default_obstacle_cost,
    )
    if not class_weights:
        raise ValueError("No classes selected for cost mapping. Check your arguments.")

    cost_map = project_costmap(grid, class_weights, aggregate=args.aggregate)
    config_center = None
    if args.recenter == "ego":
        pc_range = args.pc_range
        if args.ego_center_from_payload:
            payload_occ, payload_pc_range = load_occ_params_from_payload(
                args.ego_center_from_payload, sample_index=args.ego_center_sample_index
            )
            if args.occ_size is None and payload_occ is not None:
                occ_size = list(payload_occ)
            if pc_range is None:
                pc_range = payload_pc_range
            if occ_size is None or pc_range is None:
                raise ValueError(
                    "--occ-size and --pc-range are required when using --ego-center-from-payload "
                    "if the payload lacks these fields."
                )
            payload_row, payload_col = load_ego_center_from_payload(
                args.ego_center_from_payload,
                occ_size,
                pc_range,
                sample_index=args.ego_center_sample_index,
            )
            occ_h, occ_w = float(occ_size[0]), float(occ_size[1])
            scale_row = cost_map.shape[0] / occ_h
            scale_col = cost_map.shape[1] / occ_w
            config_center = (payload_row * scale_row, payload_col * scale_col)
        if config_center is None:
            config_center = compute_config_center(cost_map.shape[:2], occ_size, pc_range)
    recentered_shift = (0, 0)
    if args.recenter != "none":
        cost_map, recentered_shift = recenter_costmap(cost_map, args.recenter, config_center=config_center)

    # Align 2D cost map axes with the 3D visualization convention.
    # Projected grid uses (row=x, col=y); transpose + flip makes forward (+x) point right.
    cost_map = np.flipud(cost_map.T)

    if args.align_carla_y:
        cost_map = np.fliplr(cost_map)
    if args.rotate_degrees:
        k = (args.rotate_degrees // 90) % 4
        cost_map = np.rot90(cost_map, k=k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, cost_map)

    cropped = None
    bbox = None
    if args.output_cropped:
        cropped, bbox = crop_costmap(cost_map, margin=args.crop_margin)
        args.output_cropped.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_cropped, cropped)

    fixed_window = extract_fixed_window(cost_map, args.window_size)
    image_source = fixed_window

    if args.output_fixed_window:
        args.output_fixed_window.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_fixed_window, fixed_window)

    ego_yaw = args.ego_yaw
    if ego_yaw is None and args.ego_yaw_from_payload:
        ego_yaw = load_ego_yaw_from_payload(
            args.ego_yaw_from_payload, sample_index=args.ego_yaw_sample_index
        )
    if args.output_image:
        image = normalize_cost(image_source, args.max_visual_cost)
        args.output_image.parent.mkdir(parents=True, exist_ok=True)
        pic = Image.fromarray(image)
        if args.mark_ego_arrow:
            pic = add_ego_arrow(pic, yaw_deg=ego_yaw)
        pic.save(args.output_image)

    if args.dump_config:
        summary = {
            "input": str(args.input),
            "output": str(args.output),
            "output_image": str(args.output_image) if args.output_image else None,
            "output_cropped": str(args.output_cropped) if args.output_cropped else None,
            "output_fixed_window": str(args.output_fixed_window) if args.output_fixed_window else None,
            "crop_bbox": bbox,
            "window_size": args.window_size if args.output_fixed_window else None,
            "aggregate": args.aggregate,
            "class_weights": {CLASS_NAMES[i - 1]: w for i, w in class_weights.items()},
            "recenter_mode": args.recenter,
            "recenter_shift": recentered_shift,
            "align_carla_y": args.align_carla_y,
            "rotate_degrees": args.rotate_degrees,
            "mark_ego_arrow": args.mark_ego_arrow,
            "ego_yaw": ego_yaw,
            "ego_yaw_payload": str(args.ego_yaw_from_payload) if args.ego_yaw_from_payload else None,
            "ego_yaw_sample_index": args.ego_yaw_sample_index if args.ego_yaw_from_payload else None,
            "pc_range": args.pc_range,
            "occ_size": occ_size,
            "ego_center_payload": str(args.ego_center_from_payload) if args.ego_center_from_payload else None,
            "ego_center_sample_index": args.ego_center_sample_index if args.ego_center_from_payload else None,
        }
        args.dump_config.parent.mkdir(parents=True, exist_ok=True)
        args.dump_config.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved cost map to", args.output)
    if args.output_image:
        print("Saved visualization to", args.output_image)


if __name__ == "__main__":
    main()
