from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import carla

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from DrivingAgent.src.config_loader import EnvConfig
from DrivingAgent.src.camera_rig import NuScenesCameraRig, CameraInstance

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / "config" / ".env"
CONFIG = EnvConfig(ENV_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Survey CARLA metadata required for the OpenOccupancy API"
    )
    parser.add_argument(
        "--spawn-index",
        type=int,
        default=361,
        help="Index of spawn point to use when placing the ego vehicle",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the metadata JSON (stdout if omitted)",
    )
    return parser.parse_args()


def euler_deg_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float) -> list[float]:
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y, z]


def transform_to_pose(transform: carla.Transform) -> dict:
    location = transform.location
    rotation = transform.rotation
    return {
        "translation": [location.x, location.y, location.z],
        "rotation": euler_deg_to_quaternion(rotation.roll, rotation.pitch, rotation.yaw),
    }


def collect_camera_metadata(instances: list[CameraInstance]) -> list[dict]:
    metadata: list[dict] = []
    for instance in instances:
        extrinsic = transform_to_pose(instance.config.transform)
        extrinsic.update(
            {
                "name": instance.config.name,
                "resolution": [
                    int(instance.sensor.attributes.get("image_size_x", 0)),
                    int(instance.sensor.attributes.get("image_size_y", 0)),
                ],
                "fov": float(instance.sensor.attributes.get("fov", 0.0)),
            }
        )
        metadata.append(extrinsic)
    return metadata


def main(spawn_index: int, output_path: Path | None) -> None:
    carla_host = CONFIG.get("CARLA_HOST", "localhost")
    carla_port = CONFIG.get_int("CARLA_PORT", 2000)
    carla_timeout = CONFIG.get_float("CARLA_TIMEOUT", 10.0)
    carla_town = CONFIG.get("CARLA_TOWN", "Town04")

    client = carla.Client(carla_host, carla_port)
    client.set_timeout(carla_timeout)
    world = client.load_world(carla_town) if carla_town else client.get_world()

    actor_list: list[carla.Actor] = []
    camera_rig: NuScenesCameraRig | None = None

    try:
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find("vehicle.audi.a2")
        spawn_points = world.get_map().get_spawn_points()
        if spawn_index >= len(spawn_points):
            raise IndexError(
                f"spawn index {spawn_index} is out of range (total {len(spawn_points)})"
            )
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[spawn_index])
        actor_list.append(vehicle)
        vehicle.set_autopilot(False)

        output_dir = BASE_DIR / "nuscenes_output"
        camera_rig = NuScenesCameraRig(world, output_dir)
        camera_instances = camera_rig.spawn(vehicle)

        time.sleep(0.5)

        payload = {
            "ego_pose": transform_to_pose(vehicle.get_transform()),
            "camera_extrinsics": collect_camera_metadata(camera_instances),
        }

        payload_json = json.dumps(payload, indent=2)
        if output_path:
            output_path.write_text(payload_json + "\n")
        else:
            print(payload_json)
    finally:
        if camera_rig is not None:
            camera_rig.destroy()
        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()


if __name__ == "__main__":
    args = parse_args()
    main(spawn_index=args.spawn_index, output_path=args.output)
