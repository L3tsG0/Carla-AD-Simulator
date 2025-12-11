import argparse
import math
import queue
import shutil
import threading
import time
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import carla
import numpy as np
from PIL import Image

from straight_line_driver import DriverConfig, StraightLineDriver
from occupancy_pipeline import OccupancyCostmapPipeline, PipelineResult

# Ensure DrivingAgent/src is importable for shared planners/utilities.
THIS_DIR = Path(__file__).resolve().parent
DRIVING_SRC = THIS_DIR.parent / "src"
if DRIVING_SRC.exists() and str(DRIVING_SRC) not in sys.path:
    sys.path.append(str(DRIVING_SRC))

from stp3_longitudinal_planner import STP3StyleLongitudinalPlanner

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
        if self._wait_for_frame_ready(frame):
            return str(frame), self.camera_dir, False
        if not self.args.async_mode:
            return None
        staged = self._stage_oldest_frame_set()
        if staged is None:
            return None
        return staged[0], staged[1], True

    def _wait_for_frame_ready(self, frame: int) -> bool:
        """Block until all camera images for the frame are readable or stop is requested."""
        while not self.stop_event.is_set():
            if wait_for_frame_images(self.camera_dir, frame, retries=1, delay=0.05):
                return True
        return False

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
    parser.add_argument("--acceleration-gain", type=float, default=0.3, help="Throttle P gain for speed control.")
    parser.add_argument("--deceleration-gain", type=float, default=0.6, help="Brake P gain for speed control.")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8888/infer")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--rotate-degrees", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--recenter", type=str, choices=["bbox", "none", "ego"], default="bbox")
    parser.add_argument("--align-carla-y", action="store_true")
    parser.add_argument("--camera-output-root", type=Path, default=Path("CarlaRunner/DrivingAgent/notebooks/nuscenes_output"))
    parser.add_argument("--workspace-root", type=Path, default=Path("/home/tsuruoka/hdd/BEV"))
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.1)
    parser.add_argument("--async-mode", action="store_true")
    parser.add_argument(
        "--inference-mode",
        type=str,
        choices=["async", "sync"],
        default="async",
        help="Inference execution mode. async=queue/worker (current behavior), sync=block each tick until inference completes.",
    )
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
    parser.add_argument("--use-lane-yaw", action="store_true", help="Replace spawn yaw with lane waypoint yaw.")
    parser.add_argument("--use-lane-following", action="store_true", help="Apply simple lane-following steering.")
    parser.add_argument("--steer-gain-yaw", type=float, default=0.8, help="Gain for yaw error in lane-follow steer.")
    parser.add_argument("--steer-gain-lat", type=float, default=0.1, help="Gain for lateral error in lane-follow steer.")
    parser.add_argument("--use-occ-planner", action="store_true", help="Use occupancy-based longitudinal planner.")
    parser.add_argument("--planner-v-ref", type=float, default=5.0, help="Preferred speed for occupancy planner [m/s].")
    parser.add_argument("--planner-horizon-s", type=float, default=2.0, help="Planning horizon for occupancy planner [s].")
    parser.add_argument(
        "--planner-grid-resolution",
        type=float,
        default=0.8,
        help="Grid resolution (m/voxel) for occupancy costmap (default 0.8 = 0.2m*4).",
    )
    parser.add_argument(
        "--planner-corridor-half-width-m",
        type=float,
        default=0.0,
        help="Half-width (meters) of the lateral sampling corridor for occupancy cost (captures off-center obstacles).",
    )
    parser.add_argument(
        "--planner-forward-axis",
        choices=["row", "col"],
        default="row",
        help="Forward axis in costmap: 'row' (=axis 0) or 'col' (=axis 1).",
    )
    parser.add_argument(
        "--planner-weight-speed",
        type=float,
        default=1.0,
        help="Cost weight for (v - v_ref)^2 term in occupancy planner.",
    )
    parser.add_argument(
        "--planner-weight-occ",
        type=float,
        default=6.0,
        help="Cost weight for occupancy term in occupancy planner.",
    )
    parser.add_argument(
        "--enable-planner-bumper-offset",
        action="store_true",
        help="Use ego vehicle bounding_box.extent.x as forward offset to sample cost at the bumper instead of ego center.",
    )
    parser.add_argument(
        "--class-weight-json",
        type=Path,
        default=None,
        help="JSON mapping class id to weight for costmap generation (passed to OccupancyCostmapPipeline).",
    )
    parser.add_argument(
        "--cost-aggregate",
        choices=["sum", "max"],
        default="sum",
        help="Z-axis aggregation for costmap generation (sum or max).",
    )
    parser.add_argument(
        "--test-car-distance",
        type=float,
        default=None,
        help="If set, spawns a stopped vehicle this many meters ahead along the lane from the base spawn point.",
    )
    parser.add_argument(
        "--test-car-blueprint",
        type=str,
        default="vehicle.tesla.model3",
        help="Blueprint id for the test car spawned ahead of the route.",
    )
    parser.add_argument(
        "--log-control-debug",
        action="store_true",
        help="Print control debug (target speed, throttle/brake/reverse, velocity, yaw) each tick.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = int(time.time())
    camera_dir = (args.camera_output_root / f"run_{timestamp}").resolve()
    if camera_dir.exists():
        shutil.rmtree(camera_dir)

    occ_planner: Optional[STP3StyleLongitudinalPlanner] = None
    if args.use_occ_planner:
        occ_planner = STP3StyleLongitudinalPlanner(
            grid_resolution=args.planner_grid_resolution,
            dt=args.fixed_delta_seconds,
            horizon_s=args.planner_horizon_s,
            v_ref=args.planner_v_ref,
            bumper_offset_m=0.0,
            corridor_half_width_m=args.planner_corridor_half_width_m,
            weight_speed=args.planner_weight_speed,
            weight_occ=args.planner_weight_occ,
            forward_axis=args.planner_forward_axis,
        )

    driver_cfg = DriverConfig(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        town=args.town,
        spawn_index=args.spawn_index,
        target_speed_mps=args.target_speed_mps,
        duration_seconds=args.duration_seconds,
        acceleration_gain=args.acceleration_gain,
        deceleration_gain=args.deceleration_gain,
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
        use_lane_yaw=args.use_lane_yaw,
        use_lane_following=args.use_lane_following,
        steer_gain_yaw=args.steer_gain_yaw,
        steer_gain_lat=args.steer_gain_lat,
    )

    planner_state = {
        "last_frame": None,
        "costmap": None,
        "last_target": None,
        "bumper_offset_applied": False,
        "bumper_offset_warned": False,
    }
    test_car: Optional[carla.Actor] = None
    collision_sensor: Optional[carla.Actor] = None
    collision_events: List[str] = []

    notebooks_dir = Path(__file__).resolve().parent
    worker: Optional[AsyncInferenceWorker] = None
    pipeline: Optional[OccupancyCostmapPipeline] = None
    if not args.skip_inference:
        pipeline = OccupancyCostmapPipeline(
            notebooks_dir=notebooks_dir,
            api_url=args.api_url,
            workspace_root=args.workspace_root,
            class_weight_json=args.class_weight_json,
            cost_aggregate=args.cost_aggregate,
        )
        if args.inference_mode == "async":
            worker = AsyncInferenceWorker(
                pipeline=pipeline,
                camera_dir=camera_dir,
                args=args,
                max_pending=args.max_pending_frames,
            )

    state = {"last_reported": None}
    captured_frames: List[int] = []
    process_every_n = max(1, args.frame_skip)

    def _plan_with_cached(vehicle, source: str = "tick") -> None:
        """Run planner on cached costmap every tick."""
        nonlocal planner_state
        if not args.use_occ_planner or occ_planner is None:
            return
        costmap = planner_state.get("costmap")
        if costmap is None:
            return
        h, w = costmap.shape
        ego_xy = (h / 2.0, w / 2.0)
        # Optionally sample cost at the bumper instead of ego center.
        if args.enable_planner_bumper_offset and vehicle is not None:
            try:
                extent_x = float(vehicle.bounding_box.extent.x)  # half-length in X (forward)
                if not planner_state.get("bumper_offset_applied") or not math.isclose(
                    occ_planner.bumper_offset_m, extent_x, rel_tol=1e-3, abs_tol=1e-4
                ):
                    occ_planner.bumper_offset_m = extent_x
                    planner_state["bumper_offset_applied"] = True
            except Exception:
                if not planner_state.get("bumper_offset_warned"):
                    print("[Planner] bumper offset enabled, but vehicle bounding_box unavailable; using ego center.")
                    planner_state["bumper_offset_warned"] = True
        # Current signed speed along heading (use this for planner v0)
        vel = vehicle.get_velocity()
        fwd = vehicle.get_transform().get_forward_vector()
        v0_signed = vel.x * fwd.x + vel.y * fwd.y + vel.z * fwd.z
        if abs(v0_signed) < 0.05:
            v0_signed = 0.0
        # If we are rolling backwards, treat current forward speed as 0 for planning so accel > 0 produces forward motion.
        v0 = float(max(0.0, v0_signed))
        best, _ = occ_planner.plan(costmap, v0=v0, ego_xy=ego_xy)
        # Use a short lookahead window (0.5–1.0s) from the planned trajectory as the target speed.
        lookahead_target = best["next_speed"]
        v_traj = best.get("v")
        if isinstance(v_traj, np.ndarray) and v_traj.size > 0:
            t = np.arange(v_traj.size, dtype=np.float32) * float(occ_planner.dt)
            mask = (t >= 0.5) & (t <= 1.0)
            if mask.any():
                lookahead_target = float(np.mean(v_traj[mask]))
            else:
                lookahead_target = float(v_traj[-1])
        new_target = max(0.0, lookahead_target)
        driver_cfg.target_speed_mps = new_target
        planner_state["last_target"] = new_target
        if source != "tick":
            print(
                f"[Planner] frame {planner_state.get('last_frame')} -> "
                f"next_v={best['next_speed']:.2f} m/s accel={best['accel']:.2f} cost={best['total_cost']:.2f}"
            )

    def _apply_occ_planner(result: PipelineResult, vehicle) -> None:
        """Load new costmap from inference and cache it; also run planner once immediately."""
        nonlocal planner_state
        if not args.use_occ_planner or occ_planner is None:
            return
        if result.costmap_window_path is None or not result.costmap_window_path.exists():
            return
        frame_id = getattr(result, "frame_id", None)
        if frame_id is not None and planner_state.get("last_frame") == frame_id:
            return
        try:
            costmap = np.load(result.costmap_window_path)
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"[Planner] Failed to load costmap {result.costmap_window_path}: {exc}")
            return
        planner_state["costmap"] = costmap
        planner_state["last_frame"] = frame_id
        _plan_with_cached(vehicle, source="inference")

    def _ensure_collision_sensor(vehicle: carla.Vehicle) -> None:
        nonlocal collision_sensor
        if collision_sensor is not None:
            return
        try:
            world = vehicle.get_world()
            bp = world.get_blueprint_library().find("sensor.other.collision")
            sensor_tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
            collision_sensor = world.spawn_actor(bp, sensor_tf, attach_to=vehicle)

            def _on_coll(event: carla.CollisionEvent) -> None:
                other = event.other_actor
                msg = (
                    f"[Collision] frame={event.frame} impulse={event.normal_impulse.length():.2f} "
                    f"with actor id={other.id} type={other.type_id}"
                )
                collision_events.append(msg)
                print(msg)

            collision_sensor.listen(_on_coll)
            print("[Collision] Sensor attached to ego vehicle.")
        except Exception as exc:  # pragma: no cover
            print(f"[Collision] Failed to attach sensor: {exc}")

    def on_tick(vehicle, frame):
        captured_frames.append(frame)
        _ensure_collision_sensor(vehicle)
        # Run planner every tick using cached costmap (if any).
        _plan_with_cached(vehicle, source="tick")
        if args.log_control_debug:
            vel = vehicle.get_velocity()
            yaw = vehicle.get_transform().rotation.yaw
            ctrl = vehicle.get_control()
            fwd = vehicle.get_transform().get_forward_vector()
            signed_speed = vel.x * fwd.x + vel.y * fwd.y + vel.z * fwd.z
            speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            print(
                f"[CtrlDebug] frame={frame} target={driver_cfg.target_speed_mps:.2f} "
                f"throttle={ctrl.throttle:.2f} brake={ctrl.brake:.2f} reverse={ctrl.reverse} "
                f"speed={speed:.2f} fwd_speed={signed_speed:.2f} "
                f"vel=({vel.x:.2f},{vel.y:.2f},{vel.z:.2f}) yaw={yaw:.2f}"
            )
        if (len(captured_frames) - 1) % process_every_n != 0:
            return
        if args.skip_inference:
            print(f"[Driver] Captured frame {frame} (inference skipped)")
            return
        if args.inference_mode == "sync":
            if pipeline is None:
                print(f"[Driver] Frame {frame}: pipeline unavailable; skipping.")
                return
            # Wait for all camera images to land before running inference.
            wait_ok = wait_for_frame_images(
                camera_dir,
                frame,
                retries=max(1, int(args.tick_timeout_seconds / 0.05)),
                delay=0.05,
            )
            if not wait_ok:
                print(f"[Driver] Frame {frame}: images not ready within timeout; skipping inference.")
                return
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
                state["last_reported"] = frame
                message = f"[Driver] Frame {frame} inference complete. Costmap: {result.costmap_image_path}"
                if result.prediction_dense_path:
                    message += f" | Occupancy: {result.prediction_dense_path}"
                print(message)
                _apply_occ_planner(result, vehicle)
            except RuntimeError as exc:
                print(f"[Driver] Frame {frame} inference failed: {exc}")
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
            message = f"[Driver] Latest inference frame {latest[0]} ready. Costmap: {result.costmap_image_path}"
            if result.prediction_dense_path:
                message += f" | Occupancy: {result.prediction_dense_path}"
            print(message)
            # Attach frame id for planner bookkeeping
            result.frame_id = latest[0]
            _apply_occ_planner(result, vehicle)

    driver = StraightLineDriver(driver_cfg)

    # Optionally spawn a stopped test car ahead of the base spawn point.
    if args.test_car_distance is not None:
        base_spawn_points = driver.world.get_map().get_spawn_points()
        if not base_spawn_points or args.spawn_index >= len(base_spawn_points):
            print(f"[TestCar] spawn_index {args.spawn_index} invalid for map (len={len(base_spawn_points)})")
        else:
            base_tf = base_spawn_points[args.spawn_index]
            wp = driver.world.get_map().get_waypoint(base_tf.location, project_to_road=True)
            next_wps = wp.next(args.test_car_distance)
            if not next_wps:
                print(f"[TestCar] No waypoint {args.test_car_distance}m ahead of spawn index {args.spawn_index}")
            else:
                target_tf = next_wps[0].transform
                target_tf.location.z += 0.1
                try:
                    test_bp = driver.blueprints.find(args.test_car_blueprint)
                    test_car = driver.world.spawn_actor(test_bp, target_tf)
                    test_car.set_simulate_physics(True)
                    test_car.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                    print(
                        f"[TestCar] Spawned {args.test_car_blueprint} id={test_car.id} "
                        f"{args.test_car_distance}m ahead of spawn {args.spawn_index}"
                    )
                except Exception as exc:  # pragma: no cover - runtime guard
                    print(f"[TestCar] Failed to spawn test car: {exc}")
    if args.inference_mode == "sync" and args.async_mode:
        print("[Driver] Warning: sync inference requested while CARLA async tick mode is enabled.")
        print("         Blocking until inference finishes may not align with simulator ticks.")
    print(f"Starting drive. Saving camera images to {camera_dir}")
    driver.run(on_tick=on_tick)

    if args.skip_inference:
        print(f"Drive complete. Captured {len(captured_frames)} frames at {camera_dir} (inference skipped)")
    elif args.inference_mode == "async":
        print("Drive complete; waiting for pending inference jobs...")
        if worker is not None:
            worker.stop()
            print("All pending inference jobs finished.")
        else:
            print("No worker was started; nothing to drain.")
        print(f"Captured {len(captured_frames)} frames at {camera_dir}")
    else:
        print(f"Drive complete with synchronous inference. Captured {len(captured_frames)} frames at {camera_dir}")

    # Cleanup extra actors
    if collision_sensor is not None:
        try:
            collision_sensor.stop()
            collision_sensor.destroy()
        except Exception:
            pass
    if test_car is not None and test_car.is_alive:
        try:
            test_car.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
