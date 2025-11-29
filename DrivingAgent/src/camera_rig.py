"""Camera rig utilities for CARLA simulations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import queue
import threading

import carla


class _ImageSaveWorker:
    """Background worker that writes CARLA images to disk."""

    def __init__(self, max_queue: int = 512) -> None:
        self._queue: "queue.Queue[tuple[carla.Image, str]]" = queue.Queue(max_queue)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, image: carla.Image, path: Path) -> None:
        try:
            self._queue.put_nowait((image, str(path)))
        except queue.Full:
            print(f"[CameraRig] Save queue full; dropping frame {image.frame} ({path}).")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        self._queue.join()
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                image, path = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                image.save_to_disk(path)
            except RuntimeError as exc:
                print(f"[CameraRig] Failed to save frame {image.frame} to {path}: {exc}")
            finally:
                self._queue.task_done()


@dataclass(frozen=True)
class CameraConfig:
    """Definition of a camera sensor relative to the ego vehicle."""

    name: str
    transform: carla.Transform


@dataclass
class CameraInstance:
    """Concrete sensor instance with its configuration."""

    config: CameraConfig
    sensor: carla.Sensor


class NuScenesCameraRig:
    """Camera rig that mimics nuScenes 6-camera setup."""

    def __init__(
        self,
        world: carla.World,
        output_dir: Path,
        sensor_tick: float = 0.5,
        save_queue_size: int = 512,
    ) -> None:
        self._world = world
        self._output_dir = Path(output_dir)
        self._sensor_tick = sensor_tick
        self._camera_bp = self._create_camera_blueprint()
        self._sensors: List[carla.Sensor] = []
        self._save_worker = _ImageSaveWorker(max_queue=save_queue_size)

    def _create_camera_blueprint(self) -> carla.ActorBlueprint:
        blueprint = self._world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", "1600")
        blueprint.set_attribute("image_size_y", "900")
        blueprint.set_attribute("fov", "70")
        blueprint.set_attribute("sensor_tick", str(self._sensor_tick))
        return blueprint

    def _rear_axle_offset(self, vehicle: carla.Vehicle) -> float:
        physics = vehicle.get_physics_control()
        rear_left = carla.Location(
            x=physics.wheels[2].position.x / 100.0,
            y=physics.wheels[2].position.y / 100.0,
            z=physics.wheels[2].position.z / 100.0,
        )
        rear_right = carla.Location(
            x=physics.wheels[3].position.x / 100.0,
            y=physics.wheels[3].position.y / 100.0,
            z=physics.wheels[3].position.z / 100.0,
        )
        rear_center = carla.Location(
            x=(rear_left.x + rear_right.x) / 2,
            y=(rear_left.y + rear_right.y) / 2,
            z=(rear_left.z + rear_right.z) / 2,
        )
        return rear_center.distance(vehicle.get_transform().location)

    def _camera_configs(self, dist_to_rear_axle: float) -> List[CameraConfig]:
        return [
            CameraConfig(
                name="CAM_FRONT",
                transform=carla.Transform(
                    carla.Location(x=1.70 - dist_to_rear_axle, y=0.0, z=1.5),
                    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
                ),
            ),
            CameraConfig(
                name="CAM_FRONT_LEFT",
                transform=carla.Transform(
                    carla.Location(x=1.50 - dist_to_rear_axle, y=-0.5, z=1.5),
                    carla.Rotation(pitch=0.0, yaw=-55.0, roll=0.0),
                ),
            ),
            CameraConfig(
                name="CAM_FRONT_RIGHT",
                transform=carla.Transform(
                    carla.Location(x=1.50 - dist_to_rear_axle, y=0.5, z=1.5),
                    carla.Rotation(pitch=0.0, yaw=55.0, roll=0.0),
                ),
            ),
            CameraConfig(
                name="CAM_BACK",
                transform=carla.Transform(
                    carla.Location(x=-0.5 - dist_to_rear_axle, y=0.0, z=1.5),
                    carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0),
                ),
            ),
            CameraConfig(
                name="CAM_BACK_LEFT",
                transform=carla.Transform(
                    carla.Location(x=1.0 - dist_to_rear_axle, y=-0.5, z=1.5),
                    carla.Rotation(pitch=0.0, yaw=-110.0, roll=0.0),
                ),
            ),
            CameraConfig(
                name="CAM_BACK_RIGHT",
                transform=carla.Transform(
                    carla.Location(x=1.0 - dist_to_rear_axle, y=0.5, z=1.5),
                    carla.Rotation(pitch=0.0, yaw=110.0, roll=0.0),
                ),
            ),
        ]

    def spawn(self, vehicle: carla.Vehicle) -> List[CameraInstance]:
        """Spawn cameras and start recording images."""

        dist_to_rear_axle = self._rear_axle_offset(vehicle)
        instances: List[CameraInstance] = []

        for config in self._camera_configs(dist_to_rear_axle):
            save_dir = self._output_dir / config.name
            save_dir.mkdir(parents=True, exist_ok=True)
            sensor = self._world.spawn_actor(
                self._camera_bp,
                config.transform,
                attach_to=vehicle,
            )
            sensor.listen(
                lambda image, path=save_dir: self._save_worker.enqueue(
                    image, path / f"{image.frame}.png"
                )
            )
            instances.append(CameraInstance(config=config, sensor=sensor))

        self._sensors.extend(instance.sensor for instance in instances)
        return instances

    def destroy(self) -> None:
        """Destroy spawned sensors."""

        for sensor in self._sensors:
            if sensor.is_alive:
                sensor.destroy()
        self._sensors.clear()
        self._save_worker.stop()
