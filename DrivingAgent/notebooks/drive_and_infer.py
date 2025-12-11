import argparse
import shutil
import time
from pathlib import Path
from typing import List, Optional

from PIL import Image
from straight_line_driver import DriverConfig, StraightLineDriver
from occupancy_pipeline import OccupancyCostmapPipeline

CAM_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _frame_image_path(cam_dir: Path, frame_id: int) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        candidate = cam_dir / f"{frame_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def _can_read_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def wait_for_frame_images(base_dir: Path, frame_id: int, retries: int = 20, delay: float = 0.05) -> bool:
    for _ in range(retries):
        missing: List[str] = []
        for cam in CAM_NAMES:
            cam_dir = base_dir / cam
            if not cam_dir.exists():
                missing.append(cam)
                continue
            path = _frame_image_path(cam_dir, frame_id)
            if path is None or not _can_read_image(path):
                missing.append(cam)
        if not missing:
            return True
        time.sleep(delay)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive straight and run Occupancy inference using captured images.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--town", type=str, default="Town04")
    parser.add_argument("--spawn-index", type=int, default=361)
    parser.add_argument("--duration-seconds", type=float, default=5.0)
    parser.add_argument("--target-speed-mps", type=float, default=5.0)
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8888/infer")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--rotate-degrees", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--recenter", type=str, choices=["bbox", "none", "ego"], default="bbox")
    parser.add_argument("--align-carla-y", action="store_true")
    parser.add_argument("--camera-output-root", type=Path, default=Path("CarlaRunner/DrivingAgent/notebooks/nuscenes_output"))
    parser.add_argument("--workspace-root", type=Path, default=Path("/home/tsuruoka/hdd/BEV"))
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.5)
    parser.add_argument(
        "--async-mode",
        action="store_true",
        help="Disable synchronous ticking and wait for CARLA world updates asynchronously.",
    )
    parser.add_argument(
        "--stabilization-seconds",
        type=float,
        default=5.0,
        help="Wait time after spawning the vehicle/cameras before starting the drive.",
    )
    parser.add_argument(
        "--sensor-warmup-seconds",
        type=float,
        default=1.0,
        help="Additional wait time after spawning cameras to ensure sensors are producing frames.",
    )
    parser.add_argument(
        "--tick-timeout-seconds",
        type=float,
        default=10.0,
        help="Timeout for synchronous world.tick() calls before falling back to async wait.",
    )
    parser.add_argument(
        "--traffic-manager-port",
        type=int,
        default=None,
        help="Optional port for CARLA Traffic Manager; enables TM sync toggling when provided.",
    )
    parser.add_argument(
        "--driver-debug",
        action="store_true",
        help="Enable verbose StraightLineDriver debug logging.",
    )
    parser.add_argument("--use-lane-yaw", action="store_true", help="Replace spawn yaw with lane waypoint yaw.")
    parser.add_argument("--use-lane-following", action="store_true", help="Apply simple lane-following steering.")
    parser.add_argument("--steer-gain-yaw", type=float, default=0.8, help="Gain for yaw error in lane-follow steer.")
    parser.add_argument("--steer-gain-lat", type=float, default=0.1, help="Gain for lateral error in lane-follow steer.")
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Capture frames but skip calls to the inference API and costmap generation.",
    )
    parser.add_argument(
        "--attack-config",
        type=Path,
        default=None,
        help="Optional AttackSimulator config applied before costmap generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = int(time.time())
    camera_dir = (args.camera_output_root / f"run_{timestamp}").resolve()
    if camera_dir.exists():
        shutil.rmtree(camera_dir)
    driver_cfg = DriverConfig(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        town=args.town,
        spawn_index=args.spawn_index,
        target_speed_mps=args.target_speed_mps,
        duration_seconds=args.duration_seconds,
        enable_cameras=True,
        camera_output_dir=camera_dir,
        fixed_delta_seconds=args.fixed_delta_seconds,
        synchronous_mode=not args.async_mode,
        stabilization_seconds=args.stabilization_seconds,
        sensor_warmup_seconds=args.sensor_warmup_seconds,
        tick_timeout_seconds=args.tick_timeout_seconds,
        traffic_manager_port=args.traffic_manager_port,
        debug=args.driver_debug,
        use_lane_yaw=args.use_lane_yaw,
        use_lane_following=args.use_lane_following,
        steer_gain_yaw=args.steer_gain_yaw,
        steer_gain_lat=args.steer_gain_lat,
    )
    notebooks_dir = Path(__file__).resolve().parent
    captured_frames: List[int] = []

    def on_tick(vehicle, frame):
        captured_frames.append(frame)
        print(f"Captured frame {frame}")

    driver = StraightLineDriver(driver_cfg)
    print(f"Starting drive. Saving camera images to {camera_dir}")
    driver.run(on_tick=on_tick)
    print(f"Drive complete. Captured {len(captured_frames)} frames.")

    if not captured_frames:
        return
    if args.skip_inference:
        print(f"Inference skipped. Captured frames remain in {camera_dir}")
        return

    pipeline = OccupancyCostmapPipeline(
        notebooks_dir=notebooks_dir,
        api_url=args.api_url,
        workspace_root=args.workspace_root,
        attack_config=args.attack_config,
    )
    print("Running inference on recorded frames...")
    for frame in captured_frames:
        if not wait_for_frame_images(camera_dir, frame):
            print(f"Frame {frame} not ready; skipping inference.")
            continue
        print(f"Running occupancy inference (frame {frame})...")
        try:
            result = pipeline.run_once(
                spawn_index=args.spawn_index,
                record_seconds=args.fixed_delta_seconds,
                window_size=args.window_size,
                rotate_degrees=args.rotate_degrees,
                align_carla_y=args.align_carla_y,
                recenter_mode=args.recenter,
                carla_dir=camera_dir,
                frame_id=str(frame),
                filename_template="{cam}/{frame}",
            )
            print("  Payload:", result.payload_path)
            print("  Costmap window:", result.costmap_window_path)
            print("  Costmap image:", result.costmap_image_path)
        except RuntimeError as exc:
            print("  Inference failed:", exc)


if __name__ == "__main__":
    main()
