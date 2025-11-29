from __future__ import annotations

import math
import time
from dataclasses import dataclass
import sys
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
    timeout: float = 5.0
    town: str = "Town04"
    spawn_index: int = 361
    target_speed_mps: float = 5.0
    acceleration_gain: float = 0.3
    deceleration_gain: float = 0.6
    duration_seconds: float = 10.0
    enable_cameras: bool = False
    camera_output_dir: Path = Path(__file__).resolve().parent / "nuscenes_output"


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
        self.world = self.client.load_world(config.town)
        self.blueprints = self.world.get_blueprint_library()
        self.vehicle: Optional[carla.Vehicle] = None
        self.camera_rig: Optional[NuScenesCameraRig] = None

    def _spawn_vehicle(self) -> carla.Vehicle:
        spawn_points = self.world.get_map().get_spawn_points()
        if self.cfg.spawn_index >= len(spawn_points):
            raise ValueError(f"spawn_index {self.cfg.spawn_index} out of range ({len(spawn_points)} spawn points)")
        blueprint = self.blueprints.find("vehicle.audi.a2")
        vehicle = self.world.spawn_actor(blueprint, spawn_points[self.cfg.spawn_index])
        vehicle.set_autopilot(False)
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

    def run(self, on_tick: Optional[Callable[[carla.Vehicle], None]] = None) -> None:
        """Drive straight ahead for cfg.duration_seconds."""
        try:
            self.vehicle = self._spawn_vehicle()
            self.vehicle.set_simulate_physics(True)
            if self.cfg.enable_cameras:
                output_dir = Path(self.cfg.camera_output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                self.camera_rig = NuScenesCameraRig(self.world, output_dir)
                self.camera_rig.spawn(self.vehicle)
            start_time = time.time()
            while time.time() - start_time < self.cfg.duration_seconds:
                speed = self._current_speed(self.vehicle)
                error = self.cfg.target_speed_mps - speed
                if error >= 0:
                    throttle = error * self.cfg.acceleration_gain
                    brake = 0.0
                else:
                    throttle = 0.0
                    brake = -error * self.cfg.deceleration_gain
                self._apply_control(self.vehicle, throttle, brake)
                if on_tick:
                    on_tick(self.vehicle)
                self.world.tick()
        finally:
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
    )


def main() -> None:
    config = parse_args()
    driver = StraightLineDriver(config)
    driver.run()


if __name__ == "__main__":
    main()
