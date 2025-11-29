from __future__ import annotations

import math
from dataclasses import dataclass
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import carla

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from DrivingAgent.src.camera_rig import NuScenesCameraRig


@dataclass
class DriverConfig:
    """Configuration for a simple straight-line CARLA driver."""

    host: str = "localhost"
    port: int = 2000
    timeout: float = 10.0
    town: str = "Town04"
    spawn_index: int = 361
    target_speed_mps: float = 5.0
    acceleration_gain: float = 0.3
    deceleration_gain: float = 0.6
    duration_seconds: float = 10.0
    enable_cameras: bool = False
    camera_output_dir: Path = Path(__file__).resolve().parent / "nuscenes_output"
    synchronous_mode: bool = True
    fixed_delta_seconds: float = 0.5
    stabilization_seconds: float = 5.0
    sensor_warmup_seconds: float = 1.0
    tick_timeout_seconds: float = 10.0
    traffic_manager_port: Optional[int] = None
    debug: bool = False
    camera_save_queue_size: int = 4096
    camera_save_workers: int = 4


class StraightLineDriver:
    """
    Minimal CARLA driver that spawns a vehicle and accelerates straight ahead.

    Usage:
        cfg = DriverConfig()
        driver = StraightLineDriver(cfg)
        driver.run()
    """

    def __init__(self, config: DriverConfig) -> None:
        self.cfg = config
        self.client = carla.Client(config.host, config.port)
        self.client.set_timeout(config.timeout)
        if config.town:
            self.world = self.client.load_world(config.town)
        else:
            self.world = self.client.get_world()
        self._debug(f"Connected to world {self.world.get_map().name}")
        self.blueprints = self.world.get_blueprint_library()
        self.vehicle: Optional[carla.Vehicle] = None
        self.camera_rig: Optional[NuScenesCameraRig] = None
        self._original_settings: Optional[carla.WorldSettings] = None
        self._using_sync: bool = False
        self.traffic_manager: Optional[carla.TrafficManager] = None
        self._debug("Initialized StraightLineDriver")

    def _debug(self, message: str) -> None:
        if self.cfg.debug:
            print(f"[Driver] {message}")

    def _spawn_vehicle(self) -> carla.Vehicle:
        spawn_points = self.world.get_map().get_spawn_points()
        if self.cfg.spawn_index >= len(spawn_points):
            raise ValueError(f"spawn_index {self.cfg.spawn_index} out of range ({len(spawn_points)} spawn points)")
        blueprint = self.blueprints.find("vehicle.audi.a2")
        vehicle = self.world.spawn_actor(blueprint, spawn_points[self.cfg.spawn_index])
        return vehicle

    @staticmethod
    def _current_speed(vehicle: carla.Vehicle) -> float:
        velocity = vehicle.get_velocity()
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def _apply_control(self, vehicle: carla.Vehicle, throttle: float, brake: float) -> None:
        control = carla.VehicleControl()
        control.throttle = max(0.0, min(1.0, throttle))
        control.brake = max(0.0, min(1.0, brake))
        control.steer = 0.0  # straight
        control.hand_brake = False
        control.reverse = False
        vehicle.apply_control(control)

    def _apply_sync_settings(self) -> None:
        if not self.cfg.synchronous_mode:
            self._using_sync = False
            self._debug("Running in asynchronous mode")
            return
        settings = self.world.get_settings()
        self._original_settings = settings
        if (
            settings.synchronous_mode
            and abs((settings.fixed_delta_seconds or 0.0) - self.cfg.fixed_delta_seconds) < 1e-6
        ):
            return
        new_settings = carla.WorldSettings(
            no_rendering_mode=settings.no_rendering_mode,
            synchronous_mode=True,
            fixed_delta_seconds=self.cfg.fixed_delta_seconds,
            substepping=False,
        )
        self.world.apply_settings(new_settings)
        self._using_sync = self.world.get_settings().synchronous_mode
        self._debug(f"Synchronous mode applied: {self._using_sync}, fixed_delta={self.cfg.fixed_delta_seconds}")
        if self.cfg.traffic_manager_port is not None:
            self.traffic_manager = self.client.get_trafficmanager(self.cfg.traffic_manager_port)
            self.traffic_manager.set_synchronous_mode(self._using_sync)
            self._debug(f"TrafficManager sync mode set to {self._using_sync}")

    def _restore_settings(self) -> None:
        if self._original_settings is not None:
            self.world.apply_settings(self._original_settings)
            self._original_settings = None
        self._using_sync = False
        if self.traffic_manager is not None:
            self.traffic_manager.set_synchronous_mode(False)
            self._debug("TrafficManager sync mode disabled")
            self.traffic_manager = None

    def _wait_with_world(self, seconds: float, apply_brake: bool = False) -> None:
        if self.vehicle is None or seconds <= 0:
            return
        use_ticks = self._using_sync
        if not use_ticks:
            if apply_brake:
                self._apply_control(self.vehicle, throttle=0.0, brake=1.0)
            time.sleep(seconds)
            return
        settings = self.world.get_settings()
        step = settings.fixed_delta_seconds or self.cfg.fixed_delta_seconds or 0.05
        ticks = max(1, int(math.ceil(seconds / step)))
        for idx in range(ticks):
            if apply_brake:
                self._apply_control(self.vehicle, throttle=0.0, brake=1.0)
            try:
                self.world.tick(self.cfg.tick_timeout_seconds)
                self._debug(f"Stabilize tick {idx + 1}/{ticks}")
            except RuntimeError:
                self._using_sync = False
                self._debug("Tick timeout during wait; falling back to sleep")
                remaining = seconds - ((idx + 1) * step)
                if remaining > 0:
                    time.sleep(remaining)
                break

    def _compute_control(self) -> carla.VehicleControl:
        speed = self._current_speed(self.vehicle)
        error = self.cfg.target_speed_mps - speed
        if error >= 0:
            throttle = error * self.cfg.acceleration_gain
            brake = 0.0
        else:
            throttle = 0.0
            brake = -error * self.cfg.deceleration_gain
        control = carla.VehicleControl()
        control.throttle = max(0.0, min(1.0, throttle))
        control.brake = max(0.0, min(1.0, brake))
        control.steer = 0.0
        control.hand_brake = False
        control.reverse = False
        return control

    def run(self, on_tick: Optional[Callable[[carla.Vehicle, int], None]] = None) -> None:
        """Drive straight ahead for cfg.duration_seconds in synchronous ticks."""
        output_dir = Path(self.cfg.camera_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        total_ticks = max(
            1, int(math.ceil(self.cfg.duration_seconds / self.cfg.fixed_delta_seconds))
        )
        try:
            self._apply_sync_settings()
            self.vehicle = self._spawn_vehicle()
            self.vehicle.set_simulate_physics(True)
            self._debug(f"Spawned vehicle id={self.vehicle.id}")
            self._wait_with_world(self.cfg.stabilization_seconds, apply_brake=True)
            if self.cfg.enable_cameras:
                self.camera_rig = NuScenesCameraRig(
                    self.world,
                    output_dir,
                    sensor_tick=self.cfg.fixed_delta_seconds,
                    save_queue_size=self.cfg.camera_save_queue_size,
                    save_worker_count=self.cfg.camera_save_workers,
                )
                self.camera_rig.spawn(self.vehicle)
                self._debug("Cameras spawned")
                self._wait_with_world(self.cfg.sensor_warmup_seconds, apply_brake=True)
            for tick in range(total_ticks):
                control = self._compute_control()
                self.vehicle.apply_control(control)
                if self._using_sync:
                    try:
                        frame_id = self.world.tick(self.cfg.tick_timeout_seconds)
                        self._debug(f"synchronous tick -> frame {frame_id}")
                    except RuntimeError:
                        self._using_sync = False
                        self._debug("Tick timeout; switching to wait_for_tick mode")
                        snapshot = self.world.wait_for_tick(seconds=self.cfg.timeout)
                        if snapshot is None:
                            raise RuntimeError("Timed out waiting for CARLA world update in asynchronous mode.")
                        frame_id = snapshot.frame
                else:
                    snapshot = self.world.wait_for_tick(seconds=self.cfg.timeout)
                    if snapshot is None:
                        raise RuntimeError("Timed out waiting for CARLA world update in asynchronous mode.")
                    frame_id = snapshot.frame
                    self._debug(f"wait_for_tick -> frame {frame_id}")
                if on_tick:
                    on_tick(self.vehicle, frame_id)
        finally:
            self._restore_settings()
            if self.vehicle is not None and self.vehicle.is_alive:
                self.vehicle.destroy()
                self.vehicle = None
            if self.camera_rig is not None:
                self.camera_rig.destroy()
                self.camera_rig = None


def parse_args() -> DriverConfig:
    import argparse

    parser = argparse.ArgumentParser(description="Simple straight-line CARLA driver.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--town", type=str, default="Town04")
    parser.add_argument("--spawn-index", type=int, default=361)
    parser.add_argument("--target-speed-mps", type=float, default=5.0)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--acceleration-gain", type=float, default=0.3)
    parser.add_argument("--deceleration-gain", type=float, default=0.6)
    parser.add_argument("--enable-cameras", action="store_true")
    parser.add_argument("--camera-output-dir", type=Path, default=Path(__file__).resolve().parent / "nuscenes_output")
    parser.add_argument("--stabilization-seconds", type=float, default=5.0)
    parser.add_argument("--sensor-warmup-seconds", type=float, default=1.0)
    parser.add_argument("--tick-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--traffic-manager-port", type=int, default=None)
    parser.add_argument("--driver-debug", action="store_true")
    parser.add_argument("--camera-save-queue-size", type=int, default=4096)
    parser.add_argument("--camera-save-workers", type=int, default=4)

    args = parser.parse_args()
    return DriverConfig(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        town=args.town,
        spawn_index=args.spawn_index,
        target_speed_mps=args.target_speed_mps,
        duration_seconds=args.duration_seconds,
        acceleration_gain=args.acceleration_gain,
        deceleration_gain=args.deceleration_gain,
        enable_cameras=args.enable_cameras,
        camera_output_dir=args.camera_output_dir,
        stabilization_seconds=args.stabilization_seconds,
        sensor_warmup_seconds=args.sensor_warmup_seconds,
        tick_timeout_seconds=args.tick_timeout_seconds,
        traffic_manager_port=args.traffic_manager_port,
        debug=args.driver_debug,
        camera_save_queue_size=args.camera_save_queue_size,
        camera_save_workers=args.camera_save_workers,
    )


def main() -> None:
    config = parse_args()
    driver = StraightLineDriver(config)
    driver.run()


if __name__ == "__main__":
    main()
