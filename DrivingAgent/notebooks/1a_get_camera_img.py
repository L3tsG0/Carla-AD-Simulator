import carla
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from DrivingAgent.src.config_loader import EnvConfig

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / "config" / ".env"
CONFIG = EnvConfig(ENV_PATH)

def main():
    # 保存先ディレクトリの作成
    output_dir = Path(CONFIG.get("OUTPUT_DIR", BASE_DIR / "nuscenes_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. クライアントの初期化とサーバへの接続
    try:
        carla_host = CONFIG.get("CARLA_HOST", "localhost")
        carla_port = CONFIG.get_int("CARLA_PORT", 2000)
        carla_timeout = CONFIG.get_float("CARLA_TIMEOUT", 10.0)
        carla_town = CONFIG.get("CARLA_TOWN", "Town04")

        client = carla.Client(carla_host, carla_port)
        client.set_timeout(carla_timeout)
        world = client.load_world(carla_town) if carla_town else client.get_world()
        
        # 既存のActorをクリーンアップするためのリスト
        actor_list = []
        
        # ブループリントライブラリの取得
        blueprint_library = world.get_blueprint_library()

        # 2. 車両（Ego Vehicle）の配置
        # 記事に従い Audi A2 を使用（nuScenesのルノー・ゾエに近いコンパクトカーとして）
        vehicle_bp = blueprint_library.find("vehicle.audi.a2")
        
        # スポーンポイントの取得と車両の生成
        spawn_point = world.get_map().get_spawn_points()[361]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        vehicle.set_autopilot(True) # オートパイロットで走行させる
        
        print(f"車両をスポーンしました: {vehicle.type_id}")

        # 車両が物理演算により着地するのを少し待つ
        time.sleep(1.0)

        # 3. nuScenes基準点（後輪車軸中心）の計算
        # 記事にあるロジック: 物理コントロールからホイール位置を取得して計算
        physics = vehicle.get_physics_control()
        
        # CARLAのホイールインデックスは通常 0:FL, 1:FR, 2:BL, 3:BR
        # 単位をcmからmに変換
        rear_left_wheel_pos = carla.Location(
            x=physics.wheels[2].position.x / 100.0,
            y=physics.wheels[2].position.y / 100.0,
            z=physics.wheels[2].position.z / 100.0,
        )
        rear_right_wheel_pos = carla.Location(
            x=physics.wheels[3].position.x / 100.0,
            y=physics.wheels[3].position.y / 100.0,
            z=physics.wheels[3].position.z / 100.0,
        )
        
        # 後輪車軸の中心座標 (World Global Coordinates)
        rear_axle_center_loc = carla.Location(
            x=(rear_left_wheel_pos.x + rear_right_wheel_pos.x) / 2,
            y=(rear_left_wheel_pos.y + rear_right_wheel_pos.y) / 2,
            z=(rear_left_wheel_pos.z + rear_right_wheel_pos.z) / 2
        )

        # 車両原点と後輪車軸中心との距離（オフセット）を計算
        # これにより、カメラを「後輪車軸基準」で配置する際の補正値を出す
        vehicle_loc = vehicle.get_transform().location
        dist_to_rear_axle = rear_axle_center_loc.distance(vehicle_loc)
        
        # 記事に基づき、X軸方向（車両前方）のオフセットとして扱う
        # ※CARLAのローカル座標系: X=前, Y=右, Z=上
        
        # 4. カメラの設定
        # 記事に基づく共通設定: 解像度 1600x900, FOV 70度
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', "1600")
        camera_bp.set_attribute('image_size_y', "900")
        camera_bp.set_attribute('fov', "70")
        # モーションブラーなどを切って鮮明にする場合は以下を追加してもよい
        camera_bp.set_attribute('sensor_tick', '0.5') # 0.5秒ごとに撮影

        # nuScenesの6台のカメラ構成
        # 位置(x, y, z)と回転(pitch, yaw, roll)
        # 記事内の変換ロジックを参考に、CARLA座標系(X前, Y右, Z上)における相対位置を定義
        # 注意: 正確な値はcalibration.jsonが必要ですが、ここでは典型的な配置を再現します
        
        # X: 前方への距離 (dist_to_rear_axle を引くことで後輪基準に合わせる)
        # Y: 左右 (左が負、右が正)
        # Z: 高さ
        cameras_config = [
            {
                "name": "CAM_FRONT",
                "trans": carla.Transform(
                    carla.Location(x=1.70 - dist_to_rear_axle, y=0.0, z=1.5),
                    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
                )
            },
            {
                "name": "CAM_FRONT_LEFT",
                "trans": carla.Transform(
                    carla.Location(x=1.50 - dist_to_rear_axle, y=-0.5, z=1.5),
                    carla.Rotation(pitch=0.0, yaw=-55.0, roll=0.0) # 左斜め前
                )
            },
            {
                "name": "CAM_FRONT_RIGHT",
                "trans": carla.Transform(
                    carla.Location(x=1.50 - dist_to_rear_axle, y=0.5, z=1.5),
                    carla.Rotation(pitch=0.0, yaw=55.0, roll=0.0) # 右斜め前
                )
            },
            {
                "name": "CAM_BACK",
                "trans": carla.Transform(
                    carla.Location(x=-0.5 - dist_to_rear_axle, y=0.0, z=1.5),
                    carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0) # 後ろ
                )
            },
            {
                "name": "CAM_BACK_LEFT",
                # 記事で計算されていた例: yaw=約18度(nuScenes系) -> CARLAでは後方を向く設定が必要
                # ここではnuScenesの標準的なカバレッジに合わせて配置
                "trans": carla.Transform(
                    carla.Location(x=1.0 - dist_to_rear_axle, y=-0.5, z=1.5),
                    carla.Rotation(pitch=0.0, yaw=-110.0, roll=0.0) # 左斜め後ろ
                )
            },
            {
                "name": "CAM_BACK_RIGHT",
                "trans": carla.Transform(
                    carla.Location(x=1.0 - dist_to_rear_axle, y=0.5, z=1.5),
                    carla.Rotation(pitch=0.0, yaw=110.0, roll=0.0) # 右斜め後ろ
                )
            }
        ]

        # 5. カメラの生成とListen設定
        for cam_conf in cameras_config:
            camera_dir = output_dir / cam_conf["name"]
            camera_dir.mkdir(parents=True, exist_ok=True)

            # カメラを車両にAttachして生成
            # attach_to=vehicle により、車両の動きに追従する
            sensor = world.spawn_actor(
                camera_bp, 
                cam_conf["trans"], 
                attach_to=vehicle
            )
            actor_list.append(sensor)
            
            # コールバック関数: 画像を保存
            # lambda内で変数を固定するためにデフォルト引数を使用
            sensor.listen(lambda image, save_path=camera_dir: 
                image.save_to_disk(str(save_path / f"{image.frame}.png"))
            )
            
            print(f"カメラ設置完了: {cam_conf['name']}")

        print("データ収集を開始します (10秒間)...")
        
        # 6. シミュレーションループ
        # クライアント側で時間を進める（非同期モードの場合はwaitのみ）
        time.sleep(8)
    finally:
        # 7. クリーンアップ
        print("終了処理中: 生成したActorを破棄します...")
        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()
        print("完了")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")
    except RuntimeError as e:
        print(f"Error: {e}")
