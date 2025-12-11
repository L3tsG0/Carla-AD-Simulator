"""
Spawn a vehicle offset from a specified CARLA spawn point and capture a front image
from the original spawn location.

Usage:
python 5a_spawn_offset_capture.py --spawn-index 361 --distance 10.0 --output /tmp/offset.png
"""

import argparse
import math
import sys
import time
from pathlib import Path

import carla


def _look_at(source: carla.Location, target: carla.Location) -> carla.Rotation:
    """Create a rotation that points from source to target."""
    dx = target.x - source.x
    dy = target.y - source.y
    dz = target.z - source.z
    yaw = math.degrees(math.atan2(dy, dx))
    horiz_dist = math.sqrt(dx * dx + dy * dy)
    pitch = -math.degrees(math.atan2(dz, horiz_dist))
    return carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)


def _apply_sync(world: carla.World, fixed_delta: float) -> carla.WorldSettings:
    settings = world.get_settings()
    original = carla.WorldSettings()
    original.synchronous_mode = settings.synchronous_mode
    original.fixed_delta_seconds = settings.fixed_delta_seconds
    original.no_rendering_mode = settings.no_rendering_mode
    # Some CARLA versions expose substepping; guard with getattr for compatibility.
    if hasattr(settings, "substepping"):
        original.substepping = settings.substepping
    if hasattr(settings, "max_substep_delta_time"):
        original.max_substep_delta_time = settings.max_substep_delta_time
    if hasattr(settings, "max_substeps"):
        original.max_substeps = settings.max_substeps

    new_settings = carla.WorldSettings()
    new_settings.synchronous_mode = True
    new_settings.fixed_delta_seconds = fixed_delta
    new_settings.no_rendering_mode = settings.no_rendering_mode
    if hasattr(new_settings, "substepping") and hasattr(settings, "substepping"):
        new_settings.substepping = settings.substepping
    if hasattr(new_settings, "max_substep_delta_time") and hasattr(settings, "max_substep_delta_time"):
        new_settings.max_substep_delta_time = settings.max_substep_delta_time
    if hasattr(new_settings, "max_substeps") and hasattr(settings, "max_substeps"):
        new_settings.max_substeps = settings.max_substeps
    world.apply_settings(new_settings)
    return original


def main() -> None:
    parser = argparse.ArgumentParser(description="Spawn vehicle offset from a spawn point and capture an image.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--town", type=str, default="Town04")
    parser.add_argument("--spawn-index", type=int, required=True, help="Base spawn point index.")
    parser.add_argument("--distance", type=float, required=True, help="Forward distance (meters) along the lane.")
    parser.add_argument("--output", type=Path, default=Path("offset_capture.png"), help="Output image path.")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--fixed-delta", type=float, default=0.05)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.load_world(args.town) if args.town else client.get_world()
    blueprint_lib = world.get_blueprint_library()

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No spawn points found.", file=sys.stderr)
        return
    if args.spawn_index >= len(spawn_points):
        print(f"spawn_index {args.spawn_index} out of range ({len(spawn_points)} spawn points)", file=sys.stderr)
        return
    base_tf = spawn_points[args.spawn_index]

    # Compute offset transform along the lane using waypoint API.
    wp = world.get_map().get_waypoint(base_tf.location, project_to_road=True)
    next_wps = wp.next(args.distance)
    if not next_wps:
        print(f"No waypoint found {args.distance}m ahead of spawn index {args.spawn_index}", file=sys.stderr)
        return
    target_wp = next_wps[0]
    target_tf = target_wp.transform
    target_tf.location.z += 0.1  # small lift to avoid ground clipping

    vehicle = None
    camera = None
    original_settings = None
    try:
        original_settings = _apply_sync(world, args.fixed_delta)

        vehicle_bp = blueprint_lib.find("vehicle.tesla.model3")
        vehicle = world.spawn_actor(vehicle_bp, target_tf)
        vehicle.set_simulate_physics(True)
        print(f"Spawned vehicle id={vehicle.id} at {target_tf.location}")

        # Camera at base spawn, looking at the vehicle.
        cam_bp = blueprint_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(args.width))
        cam_bp.set_attribute("image_size_y", str(args.height))
        cam_bp.set_attribute("fov", str(args.fov))

        cam_loc = base_tf.location
        cam_loc.z += 2.0  # lift camera a bit
        cam_rot = _look_at(cam_loc, target_tf.location)
        cam_tf = carla.Transform(cam_loc, cam_rot)

        camera = world.spawn_actor(cam_bp, cam_tf)

        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        captured = {"done": False}

        def _on_image(image: carla.Image) -> None:
            if captured["done"]:
                return
            image.save_to_disk(str(output_path))
            captured["done"] = True
            print(f"Saved image to {output_path}")

        camera.listen(_on_image)

        # Run a few ticks to allow capture.
        for _ in range(10):
            world.tick(args.timeout)
            if captured["done"]:
                break
            time.sleep(0.01)
        if not captured["done"]:
            print("Image capture did not complete within ticks.", file=sys.stderr)
    finally:
        if camera is not None:
            camera.stop()
            camera.destroy()
        if vehicle is not None and vehicle.is_alive:
            vehicle.destroy()
        if original_settings is not None:
            world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
