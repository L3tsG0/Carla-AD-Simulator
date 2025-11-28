import carla
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
    
world = client.load_world('Town04')
spawn_points = world.get_map().get_spawn_points()

for i, sp in enumerate(spawn_points):
    loc = sp.location
    yaw = sp.rotation.yaw
    print(f"{i:02d}: (x={loc.x:6.1f}, y={loc.y:6.1f}, z={loc.z:4.1f}), yaw={yaw:6.1f}")
