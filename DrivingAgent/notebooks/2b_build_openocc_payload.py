from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    parser.add_argument(
        "--post-capture-wait",
        type=float,
        default=0.5,
        help="Extra wait time after recording to ensure sensors finish writing images.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Path to write JSON payload")
    parser.add_argument("--pkl-output", type=Path, default=None, help="Path to write PKL payload")
    parser.add_argument(
        "--template-info",
        type=Path,
        default=None,
        help="Optional nuScenes infos .pkl to use for filling missing metadata",
    )
    parser.add_argument(
        "--template-index",
        type=int,
        default=0,
        help="Index of the nuScenes sample to take as template when --template-info is set",
    )
    return parser.parse_args()


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class OpenOccPayloadBuilder:
    def __init__(self, base_dir: Path, config: EnvConfig) -> None:
        self.base_dir = base_dir
        self.config = config
        self.payload_path_prefix = self.config.get("PAYLOAD_PATH_PREFIX", None)

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
        valid: List[Path] = []
        for image in images:
            try:
                with Image.open(image) as img:
                    img.verify()
                    width, height = img.size
                    if width <= 0 or height <= 0:
                        continue
                valid.append(image.resolve())
            except Exception:
                continue
            if len(valid) >= 2:
                break
        if not valid:
            raise FileNotFoundError(f"No valid images found in {directory}")
        if len(valid) >= 2:
            return valid[1]
        return valid[0]

    @staticmethod
    def _ensure_rgb(image_path: Path) -> Path:
        """Convert images to RGB JPEG and remove the original PNG."""
        try:
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
        except OSError:
            return image_path

        target = image_path.with_suffix(".jpg")
        if image_path.suffix.lower() != ".jpg" or target != image_path:
            img.save(target, format="JPEG")
            try:
                image_path.unlink()
            except FileNotFoundError:
                pass
            return target.resolve()

        img.save(target, format="JPEG")
        return target.resolve()

    def _format_data_path(self, path: Path) -> str:
        """Return a payload-friendly path (relative to base_dir with optional prefix)."""
        try:
            rel = path.relative_to(self.base_dir)
        except ValueError:
            return str(path)
        if self.payload_path_prefix:
            return str(Path(self.payload_path_prefix) / rel).replace("\\", "/")
        return str(rel).replace("\\", "/")

    @staticmethod
    def _pose_to_matrix(pose: Dict[str, List[float]]) -> np.ndarray:
        rotation = np.array(OpenOccPayloadBuilder._quat_to_rot_matrix(pose["rotation"]))
        translation = np.array(pose["translation"])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        return matrix

    @staticmethod
    def _lidar_transform(dist_to_rear_axle: float) -> carla.Transform:
        """Approximate nuScenes LiDAR mount on CARLA vehicle."""
        location = carla.Location(x=1.35 - dist_to_rear_axle, y=0.0, z=1.73)
        rotation = carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0)
        return carla.Transform(location, rotation)

    @staticmethod
    def _quaternion_yaw(quat: List[float]) -> float:
        w, x, y, z = quat
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny, cosy)

    def _build_can_bus(self, ego_pose: Dict[str, List[float]]) -> List[float]:
        can_bus = [0.0] * 18
        can_bus[:3] = ego_pose["translation"]
        can_bus[3:7] = ego_pose["rotation"]
        yaw_deg = math.degrees(self._quaternion_yaw(ego_pose["rotation"]))
        if yaw_deg < 0:
            yaw_deg += 360.0
        can_bus[-2] = math.radians(yaw_deg)
        can_bus[-1] = yaw_deg
        return can_bus

    def _build_cam_entry(
        self,
        instance: CameraInstance,
        ego_pose: Dict[str, List[float]],
        image_path: Path,
        lidar_pose_matrix: np.ndarray,
    ) -> Dict[str, Any]:
        extrinsic = self._transform_to_pose(instance.config.transform)
        sensor_matrix = self._pose_to_matrix(extrinsic)
        sensor2lidar = np.linalg.inv(lidar_pose_matrix) @ sensor_matrix
        sensor2lidar_rot = sensor2lidar[:3, :3]
        sensor2lidar_trans = sensor2lidar[:3, 3]
        entry: Dict[str, Any] = {
            "data_path": self._format_data_path(image_path),
            "type": "camera",
            "sensor2ego_translation": extrinsic["translation"],
            "sensor2ego_rotation": extrinsic["rotation"],
            "sensor2lidar_translation": sensor2lidar_trans.tolist(),
            "sensor2lidar_rotation": sensor2lidar_rot.tolist(),
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

    @staticmethod
    def _needs_template_value(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value == ""
        if isinstance(value, (list, tuple, dict)):
            return len(value) == 0
        return False

    def _merge_with_template(self, info: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
        merged = copy.deepcopy(info)
        template_fields = [
            "lidar_path",
            "lidar_token",
            "sweeps",
            "lidarseg",
            "prev",
            "next",
            "occ_size",
            "pc_range",
        ]
        for key in template_fields:
            if key not in merged or self._needs_template_value(merged[key]):
                if key in template:
                    merged[key] = copy.deepcopy(template[key])
        return merged

    def build_payload(
        self,
        spawn_index: int,
        record_seconds: float,
        template_info: Optional[Dict[str, Any]] = None,
        post_capture_wait: float = 0.5,
    ) -> Dict[str, Any]:
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
            dist_to_rear_axle = camera_rig._rear_axle_offset(vehicle)  # type: ignore[attr-defined]
            camera_instances = camera_rig.spawn(vehicle)

            time.sleep(record_seconds)
            if post_capture_wait > 0:
                time.sleep(post_capture_wait)

            ego_pose = self._transform_to_pose(vehicle.get_transform())
            lidar_transform = self._lidar_transform(dist_to_rear_axle)
            lidar_pose = self._transform_to_pose(lidar_transform)
            lidar_matrix = self._pose_to_matrix(lidar_pose)
            cams_dict = {}
            for instance in camera_instances:
                image_dir = self.base_dir / "nuscenes_output" / instance.config.name
                image_path = self._ensure_rgb(self._latest_image_path(image_dir))
                cams_dict[instance.config.name] = self._build_cam_entry(
                    instance, ego_pose, image_path, lidar_matrix
                )

            scene_token = f"scene-{uuid.uuid4().hex}"
            sample_token = f"sample-{uuid.uuid4().hex}"

            payload: Dict[str, Any] = {
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
                        "lidar2ego_translation": lidar_pose["translation"],
                        "lidar2ego_rotation": lidar_pose["rotation"],
                        "lidarseg": "",
                        "prev": "",
                        "next": "",
                        "sweeps": [],
                        "can_bus": self._build_can_bus(ego_pose),
                    }
                ],
                "metadata": {"version": "carla"},
            }
            if template_info is not None:
                payload["infos"][0] = self._merge_with_template(payload["infos"][0], template_info)
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


def load_template_info(path: Path, index: int) -> Dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "infos" in data:
        infos = data["infos"]
    elif isinstance(data, list):
        infos = data
    else:
        raise ValueError(f"Unsupported template format in {path}")
    if not 0 <= index < len(infos):
        raise IndexError(f"template index {index} out of range (len={len(infos)})")
    return copy.deepcopy(infos[index])


def main() -> None:
    args = parse_args()
    builder = OpenOccPayloadBuilder(BASE_DIR, CONFIG)
    template_info = None
    if args.template_info:
        template_info = load_template_info(args.template_info, args.template_index)
    payload = builder.build_payload(
        args.spawn_index,
        args.record_seconds,
        template_info=template_info,
        post_capture_wait=args.post_capture_wait,
    )

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
