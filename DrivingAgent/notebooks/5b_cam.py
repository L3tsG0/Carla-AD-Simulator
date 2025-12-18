#!/usr/bin/env python3
import argparse
import pathlib
import random
import shutil
import threading
import time

import carla


def main():
    parser = argparse.ArgumentParser(description="Spawn a vehicle, brake, and grab a rear camera frame.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--town", default="Town04", help="CARLA map to load before spawning the vehicle.")
    parser.add_argument("--vehicle", default="vehicle.tesla.model3", help="Blueprint filter for the vehicle.")
    parser.add_argument("--spawn-index", type=int, default=361, help="Index into map spawn points.")
    parser.add_argument("--output", default="brake_snapshot.png")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--back-offset", type=float, default=5.0, help="Camera distance behind vehicle (meters).")
    parser.add_argument("--height-offset", type=float, default=1.4, help="Camera height over ground (meters).")
    parser.add_argument("--pitch", type=float, default=-10.0, help="Camera pitch downwards (degrees).")
    parser.add_argument("--brake-duration", type=float, default=10.0, help="Seconds to hold brakes before cleanup.")
    parser.add_argument("--frames-dir", default=None, help="Directory for saving the captured frame sequence.")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.load_world(args.town)
    blueprints = world.get_blueprint_library()

    vehicle_bp = blueprints.filter(args.vehicle)[0]
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found in map.")
    transform = spawn_points[args.spawn_index] if args.spawn_index is not None else random.choice(spawn_points)

    vehicle = world.try_spawn_actor(vehicle_bp, transform)
    if vehicle is None:
        raise RuntimeError("Failed to spawn vehicle. Try another spawn point or blueprint.")
    try:
        tail_lights = carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.Brake)
        vehicle.set_light_state(tail_lights)
    except AttributeError:
        # Older CARLA builds may not expose VehicleLightState; continue without changing lights.
        pass

    camera_bp = blueprints.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(args.width))
    camera_bp.set_attribute("image_size_y", str(args.height))
    camera_bp.set_attribute("fov", str(args.fov))
    camera_transform = carla.Transform(
        carla.Location(x=-args.back_offset, z=args.height_offset),
        carla.Rotation(pitch=args.pitch),
    )

    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    output_path = pathlib.Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_single_frame = output_path.suffix != ""
    if args.frames_dir is None:
        if save_single_frame:
            frames_dir = output_path.parent / f"{output_path.stem}_frames"
        else:
            frames_dir = output_path
    else:
        frames_dir = pathlib.Path(args.frames_dir).expanduser().resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)

    done = threading.Event()
    latest = {"path": None}

    def save_image(image: carla.Image) -> None:
        frame_path = frames_dir / f"{image.frame:06d}.png"
        image.save_to_disk(str(frame_path))
        latest["path"] = frame_path
        done.set()

    try:
        camera.listen(save_image)
        vehicle.apply_control(carla.VehicleControl(brake=1.0))
        end_time = time.monotonic() + args.brake_duration
        while time.monotonic() < end_time:
            world.wait_for_tick()
        if not done.is_set():
            done.wait(timeout=0.5)
    finally:
        camera.stop()
        camera.destroy()
        vehicle.destroy()

    if not done.is_set():
        raise RuntimeError("Timed out before camera delivered an image.")
    final_frame = latest["path"]
    if save_single_frame and final_frame is not None and final_frame != output_path:
        shutil.copy(final_frame, output_path)
        print(f"Saved latest frame: {output_path}")
    print(f"Captured frames in: {frames_dir}")


if __name__ == "__main__":
    main()
