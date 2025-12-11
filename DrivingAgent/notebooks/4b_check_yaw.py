import carla

PORT: int = 5555
client = carla.Client('localhost', PORT)
world = client.get_world()
map = world.get_map()
spawn_points = map.get_spawn_points()

# ターゲットを固定
target_spawn = spawn_points[361]

# ★修正: ランダム選択を削除し、target_spawn の座標を使う
lane_waypoint = map.get_waypoint(target_spawn.location, project_to_road=True)
perfect_transform = lane_waypoint.transform

print(f"Spawn Point (361) Yaw : {target_spawn.rotation.yaw}")
print(f"Lane Waypoint Yaw     : {perfect_transform.rotation.yaw}")

# 差分確認
diff = abs(target_spawn.rotation.yaw - perfect_transform.rotation.yaw)
print(f"Difference            : {diff}")