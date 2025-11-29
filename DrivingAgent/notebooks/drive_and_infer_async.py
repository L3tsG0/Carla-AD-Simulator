import argparse
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from straight_line_driver import DriverConfig, StraightLineDriver
from occupancy_pipeline import OccupancyCostmapPipeline, PipelineResult

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


class AsyncInferenceWorker:
    def __init__(
        self,
        pipeline: OccupancyCostmapPipeline,
        camera_dir: Path,
        args: argparse.Namespace,
        max_pending: int = 8,
    ) -> None:
        self.pipeline = pipeline
        self.camera_dir = camera_dir
        self.args = args
        self.tasks: "queue.Queue[int]" = queue.Queue(max_pending)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.lock = threading.Lock()
        self.latest_result: Optional[Tuple[int, PipelineResult]] = None
        self.stage_root = self.camera_dir / "_async_stage"
        if self.stage_root.exists():
            shutil.rmtree(self.stage_root, ignore_errors=True)
        self.stage_root.mkdir(parents=True, exist_ok=True)
        self.used_images: dict[str, set[Path]] = {cam: set() for cam in CAM_NAMES}
        self.thread.start()

    def submit(self, frame_id: int) -> bool:
        try:
            self.tasks.put_nowait(frame_id)
            return True
        except queue.Full:
            return False

    def get_latest_result(self) -> Optional[Tuple[int, PipelineResult]]:
        with self.lock:
            return self.latest_result

    def _run(self) -> None:
        while not self.stop_event.is_set() or not self.tasks.empty():
            try:
                frame = self.tasks.get(timeout=0.2)
            except queue.Empty:
                continue
            prep = self._prepare_frame_inputs(frame)
            if prep is None:
                print(f"[Worker] Frame {frame}: no usable image set; skipping.")
                self.tasks.task_done()
                continue
            staged_frame, base_dir, staged = prep
            try:
                result = self.pipeline.run_once(
                    spawn_index=self.args.spawn_index,
                    record_seconds=self.args.fixed_delta_seconds,
                    window_size=self.args.window_size,
                    rotate_degrees=self.args.rotate_degrees,
                    align_carla_y=self.args.align_carla_y,
                    recenter_mode=self.args.recenter,
                    carla_dir=base_dir,
                    frame_id=staged_frame,
                    filename_template="{cam}/{frame}",
                )
                with self.lock:
                    self.latest_result = (frame, result)
                print(f"[Worker] Frame {frame} inference complete.")
            except RuntimeError as exc:
                print(f"[Worker] Frame {frame} inference failed: {exc}")
            finally:
                if staged:
                    shutil.rmtree(base_dir, ignore_errors=True)
                self.tasks.task_done()

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        self.tasks.join()
        self.thread.join(timeout=timeout)

    def _prepare_frame_inputs(self, frame: int) -> Optional[Tuple[str, Path, bool]]:
        if wait_for_frame_images(self.camera_dir, frame, retries=40, delay=0.05):
            return str(frame), self.camera_dir, False
        if not self.args.async_mode:
            return None
        staged = self._stage_oldest_frame_set()
        if staged is None:
            return None
        return staged[0], staged[1], True

    def _stage_oldest_frame_set(self) -> Optional[Tuple[str, Path]]:
        job_id = f"async_{int(time.time() * 1000)}"
        job_dir = self.stage_root / job_id
        for cam in CAM_NAMES:
            src = self._get_oldest_unused_image(self.camera_dir / cam, cam)
            if src is None:
                shutil.rmtree(job_dir, ignore_errors=True)
                return None
            dest_dir = job_dir / cam
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{job_id}{src.suffix}"
            shutil.copy2(src, dest)
        return job_id, job_dir

    def _get_oldest_unused_image(self, cam_dir: Path, cam_name: str) -> Optional[Path]:
        candidates = sorted(
            [p for p in cam_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda p: p.stat().st_mtime,
        )
        used = self.used_images.get(cam_name, set())
        for path in candidates:
            if path in used:
                continue
            if _can_read_image(path):
                used.add(path)
                self.used_images[cam_name] = used
                return path
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive CARLA and run Occupancy inference asynchronously.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--town", type=str, default="Town04")
    parser.add_argument("--spawn-index", type=int, default=361)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--target-speed-mps", type=float, default=5.0)
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8888/infer")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--rotate-degrees", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--recenter", type=str, choices=["bbox", "none", "ego"], default="bbox")
    parser.add_argument("--align-carla-y", action="store_true")
    parser.add_argument("--camera-output-root", type=Path, default=Path("CarlaRunner/DrivingAgent/notebooks/nuscenes_output"))
    parser.add_argument("--workspace-root", type=Path, default=Path("/home/tsuruoka/hdd/BEV"))
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.1)
    parser.add_argument("--async-mode", action="store_true")
    parser.add_argument("--stabilization-seconds", type=float, default=5.0)
    parser.add_argument("--sensor-warmup-seconds", type=float, default=1.0)
    parser.add_argument("--tick-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--camera-save-queue-size", type=int, default=4096, help="Queue length for background image saving.")
    parser.add_argument("--camera-save-workers", type=int, default=4, help="Number of background threads writing images to disk.")
    parser.add_argument("--traffic-manager-port", type=int, default=None)
    parser.add_argument("--driver-debug", action="store_true")
    parser.add_argument("--max-pending-frames", type=int, default=8, help="Max number of queued frames awaiting inference.")
    parser.add_argument("--skip-inference", action="store_true", help="Capture frames without running asynchronous inference.")
    parser.add_argument("--frame-skip", type=int, default=1, help="Enqueue one frame out of N ticks (default 1 = every tick).")
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
        camera_save_queue_size=args.camera_save_queue_size,
        camera_save_workers=args.camera_save_workers,
    )

    notebooks_dir = Path(__file__).resolve().parent
    worker: Optional[AsyncInferenceWorker] = None
    if not args.skip_inference:
        pipeline = OccupancyCostmapPipeline(
            notebooks_dir=notebooks_dir,
            api_url=args.api_url,
            workspace_root=args.workspace_root,
        )
        worker = AsyncInferenceWorker(
            pipeline=pipeline,
            camera_dir=camera_dir,
            args=args,
            max_pending=args.max_pending_frames,
        )

    state = {"last_reported": None}
    captured_frames: List[int] = []

    def on_tick(vehicle, frame):
        captured_frames.append(frame)
        if (len(captured_frames) - 1) % max(1, args.frame_skip) != 0:
            return
        if worker is None:
            print(f"[Driver] Captured frame {frame}")
            return
        if not worker.submit(frame):
            print(f"[Driver] Frame {frame} dropped (worker queue full).")
        latest = worker.get_latest_result()
        if latest is not None and latest[0] != state["last_reported"]:
            state["last_reported"] = latest[0]
            result = latest[1]
            print(f"[Driver] Latest inference frame {latest[0]} ready. Costmap: {result.costmap_image_path}")

    driver = StraightLineDriver(driver_cfg)
    print(f"Starting drive. Saving camera images to {camera_dir}")
    driver.run(on_tick=on_tick)
    if worker is None:
        print(f"Drive complete. Captured {len(captured_frames)} frames at {camera_dir}")
    else:
        print("Drive complete; waiting for pending inference jobs...")
        worker.stop()
        print("All pending inference jobs finished.")


if __name__ == "__main__":
    main()
