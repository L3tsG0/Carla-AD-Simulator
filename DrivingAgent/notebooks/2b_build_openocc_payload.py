from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import carla
import uuid
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from DrivingAgent.src.config_loader import EnvConfig
from DrivingAgent.src.camera_rig import CameraInstance, NuScenesCameraRig

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / "config" / ".env"
CONFIG = EnvConfig(ENV_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build nuScenes-like payload for OpenOccupancy")
    parser.add_argument("--spawn-index", type=int, default=361)
    parser.add_argument("--record-seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None, help="Path to write JSON payload")
    parser.add_argument("--pkl-output", type=Path, default=None, help="Path to write PKL payload")
    return parser.parse_args()


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class OpenOccPayloadBuilder:
    def __init__(self, base_dir: Path, config: EnvConfig) -> None:
        self.base_dir = base_dir
        self.config = config

    @staticmethod
    def _euler_deg_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float) -> List[float]:
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

    def _transform_to_pose(self, transform: carla.Transform) -> Dict[str, List[float]]:
        location = transform.location
        rotation = transform.rotation
        return {
            "translation": [location.x, location.y, location.z],
            "rotation": self._euler_deg_to_quaternion(rotation.roll, rotation.pitch, rotation.yaw),
        }

    @staticmethod
    def _quat_to_rot_matrix(quat: List[float]) -> List[List[float]]:
        w, x, y, z = quat
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]

    @staticmethod
    def _latest_image_path(directory: Path) -> Path:
        images = sorted(directory.glob("*.png"), reverse=True)
        for image in images:
            try:
                with Image.open(image) as img:
                    img.verify()
                return image.resolve()
            except Exception:
                continue
        raise FileNotFoundError(f"No valid images found in {directory}")

    @staticmethod
    def _ensure_rgb(image_path: Path) -> Path:
        try:
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
                target = image_path.with_name(image_path.stem + "_rgb.jpg")
                img.save(target, format="JPEG")
                return target.resolve()
        except OSError:
            # broken PNG, fall back to original path
            return image_path
        return image_path

    def _build_cam_entry(
        self,
        instance: CameraInstance,
        ego_pose: Dict[str, List[float]],
        image_path: Path,
    ) -> Dict[str, Any]:
        extrinsic = self._transform_to_pose(instance.config.transform)
        sensor2lidar_rot = self._quat_to_rot_matrix(extrinsic["rotation"])
        entry: Dict[str, Any] = {
            "data_path": str(image_path),
            "type": "camera",
            "sensor2ego_translation": extrinsic["translation"],
            "sensor2ego_rotation": extrinsic["rotation"],
            "sensor2lidar_translation": [0.0, 0.0, 0.0],
            "sensor2lidar_rotation": sensor2lidar_rot,
            "ego2global_translation": ego_pose["translation"],
            "ego2global_rotation": ego_pose["rotation"],
            "timestamp": int(time.time() * 1e6),
            "cam_intrinsic": np.array(
                [
                    [1256.7414812095406, 0.0, 792.1125740759628],
                    [0.0, 1256.7414812095406, 492.7757465151356],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            ),
            "resolution": [
                int(instance.sensor.attributes.get("image_size_x", 0)),
                int(instance.sensor.attributes.get("image_size_y", 0)),
            ],
            "fov": float(instance.sensor.attributes.get("fov", 0.0)),
        }
        return entry

    def build_payload(self, spawn_index: int, record_seconds: float) -> Dict[str, Any]:
        carla_host = self.config.get("CARLA_HOST", "localhost")
        carla_port = self.config.get_int("CARLA_PORT", 2000)
        carla_timeout = self.config.get_float("CARLA_TIMEOUT", 10.0)
        carla_town = self.config.get("CARLA_TOWN", "Town04")

        client = carla.Client(carla_host, carla_port)
        client.set_timeout(carla_timeout)
        world = client.load_world(carla_town) if carla_town else client.get_world()

        actor_list: List[carla.Actor] = []
        camera_rig: NuScenesCameraRig | None = None

        try:
            vehicle_bp = world.get_blueprint_library().find("vehicle.audi.a2")
            spawn_points = world.get_map().get_spawn_points()
            if spawn_index >= len(spawn_points):
                raise ValueError("invalid spawn index")
            vehicle = world.spawn_actor(vehicle_bp, spawn_points[spawn_index])
            actor_list.append(vehicle)
            vehicle.set_autopilot(True)

            output_dir = self.base_dir / "nuscenes_output"
            camera_rig = NuScenesCameraRig(world, output_dir)
            camera_instances = camera_rig.spawn(vehicle)

            time.sleep(record_seconds)

            ego_pose = self._transform_to_pose(vehicle.get_transform())
            cams_dict = {}
            for instance in camera_instances:
                image_dir = self.base_dir / "nuscenes_output" / instance.config.name
                image_path = self._ensure_rgb(self._latest_image_path(image_dir))
                cams_dict[instance.config.name] = self._build_cam_entry(
                    instance, ego_pose, image_path
                )

            scene_token = f"scene-{uuid.uuid4().hex}"
            sample_token = f"sample-{uuid.uuid4().hex}"

            payload = {
                "infos": [
                    {
                        "token": sample_token,
                        "scene_token": scene_token,
                        "frame_idx": 0,
                        "timestamp": int(time.time() * 1e6),
                        "cams": cams_dict,
                        "ego2global_translation": ego_pose["translation"],
                        "ego2global_rotation": ego_pose["rotation"],
                        "lidar_path": "",
                        "lidar_token": "",
                        "lidar2ego_translation": [0, 0, 0],
                        "lidar2ego_rotation": [1, 0, 0, 0],
                        "lidarseg": "",
                        "prev": "",
                        "next": "",
                        "sweeps": [],
                        "can_bus": [],
                    }
                ],
                "metadata": {"version": "carla"},
            }
            return payload
        finally:
            if camera_rig:
                camera_rig.destroy()
            for actor in actor_list:
                if actor.is_alive:
                    actor.destroy()

    @staticmethod
    def make_pkl(payload: Dict[str, Any], path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump(payload, f)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main() -> None:
    args = parse_args()
    builder = OpenOccPayloadBuilder(BASE_DIR, CONFIG)
    payload = builder.build_payload(args.spawn_index, args.record_seconds)

    wrapped = {"payload": payload}
    payload_json = json.dumps(wrapped, indent=2, default=_json_default)
    if args.output:
        args.output.write_text(payload_json + "\n", encoding="utf-8")
    else:
        print(payload_json)

    if args.pkl_output:
        builder.make_pkl(payload, args.pkl_output)


if __name__ == "__main__":
    main()
