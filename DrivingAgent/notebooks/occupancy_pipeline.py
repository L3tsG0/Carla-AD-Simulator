from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests


@dataclass
class PipelineResult:
    payload_path: Path
    prediction_dense_path: Path
    costmap_full_path: Path
    costmap_window_path: Path
    costmap_image_path: Path
    occ_size: Optional[List[int]] = None
    pc_range: Optional[List[float]] = None


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

        self.capture_script = notebooks_dir / "2d_prepare_single_frame.py"
        self.costmap_script = notebooks_dir / "3a_occ_to_costmap.py"
        self.post_capture_wait = post_capture_wait

    # ----------------------------- helpers -----------------------------
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
    ) -> Path:
        output_pkl = output_pkl or self.tmp_dir / f"carla_capture_{int(time.time())}.pkl"
        args = [
            self.python_exec,
            str(self.capture_script),
            "--spawn-index",
            str(spawn_index),
            "--record-seconds",
            str(record_seconds),
            "--post-capture-wait",
            str(self.post_capture_wait),
            "--output-pkl",
            str(output_pkl),
        ]
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
    ) -> PipelineResult:
        output_full = self.tmp_dir / f"costmap_full_{int(time.time())}.npy"
        output_window = output_full.with_name(output_full.stem.replace("full", "ego") + ".npy")
        output_image = output_full.with_suffix(".png")

        args = [
            self.python_exec,
            str(self.costmap_script),
            "--input",
            str(prediction_path),
            "--output",
            str(output_full),
            "--output-fixed-window",
            str(output_window),
            "--output-image",
            str(output_image),
            "--window-size",
            str(window_size),
            "--recenter",
            recenter_mode,
            "--ego-yaw-from-payload",
            str(payload_path),
        ]
        if align_carla_y:
            args.append("--align-carla-y")
        if mark_ego_arrow:
            args.append("--mark-ego-arrow")
        if rotate_degrees:
            args.extend(["--rotate-degrees", str(rotate_degrees)])
        if occ_size:
            args.extend(["--occ-size"] + [str(int(v)) for v in occ_size])
        if pc_range:
            args.extend(["--pc-range"] + [str(float(v)) for v in pc_range])

        self._run_subprocess(args)
        return PipelineResult(
            payload_path=payload_path,
            prediction_dense_path=prediction_path,
            costmap_full_path=output_full,
            costmap_window_path=output_window,
            costmap_image_path=output_image,
            occ_size=occ_size,
            pc_range=pc_range,
        )

    def run_once(
        self,
        spawn_index: int,
        record_seconds: float,
        window_size: int = 128,
        rotate_degrees: int = 0,
        align_carla_y: bool = False,
        recenter_mode: str = "bbox",
    ) -> PipelineResult:
        payload = self.capture_frame(spawn_index=spawn_index, record_seconds=record_seconds)
        prediction, occ_size, pc_range = self.request_inference(payload)
        return self.build_costmap(
            prediction_path=prediction,
            payload_path=payload,
            window_size=window_size,
            rotate_degrees=rotate_degrees,
            align_carla_y=align_carla_y,
            recenter_mode=recenter_mode,
            occ_size=occ_size,
            pc_range=pc_range,
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
    )
    result = pipeline.run_once(
        spawn_index=args.spawn_index,
        record_seconds=args.record_seconds,
        window_size=args.window_size,
        rotate_degrees=args.rotate_degrees,
        align_carla_y=args.align_carla_y,
        recenter_mode=args.recenter,
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
