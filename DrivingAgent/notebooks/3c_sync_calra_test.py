import argparse

import carla

parser = argparse.ArgumentParser(description="Configure CARLA world for synchronous stepping.")
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--timeout", type=float, default=5.0)
args = parser.parse_args()

client = carla.Client(args.host, args.port)
client.set_timeout(args.timeout)

world = client.get_world()

# いったん async 設定に戻す
"""
settings = world.get_settings()
settings.synchronous_mode = False
settings.substepping = False
world.apply_settings(settings)
"""

# 同期モードに切り替え
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.1
settings.substepping = False
settings.max_substeps = 1
settings.max_substep_delta_time = 0.1

world.apply_settings(settings)

traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)
print("Synchronous settings applied!")
