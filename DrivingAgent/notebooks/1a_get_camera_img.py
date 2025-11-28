from __future__ import annotations

import argparse
import carla
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from DrivingAgent.src.config_loader import EnvConfig
from DrivingAgent.src.camera_rig import NuScenesCameraRig

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / "config" / ".env"
CONFIG = EnvConfig(ENV_PATH)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture camera images in CARLA")
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=None,
        help="Recording duration in seconds (overrides env RECORD_SECONDS)",
    )
    return parser.parse_args()


def main(record_seconds: float | None = None):
    # 保存先ディレクトリの作成
    output_dir = Path(CONFIG.get("OUTPUT_DIR", BASE_DIR / "nuscenes_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. クライアントの初期化とサーバへの接続
    actor_list: list[carla.Actor] = []
    camera_rig: NuScenesCameraRig | None = None
    try:
        carla_host = CONFIG.get("CARLA_HOST", "localhost")
        carla_port = CONFIG.get_int("CARLA_PORT", 2000)
        carla_timeout = CONFIG.get_float("CARLA_TIMEOUT", 10.0)
        carla_town = CONFIG.get("CARLA_TOWN", "Town04")
        duration = record_seconds
        if duration is None:
            duration = CONFIG.get_float("RECORD_SECONDS", 10.0)

        client = carla.Client(carla_host, carla_port)
        client.set_timeout(carla_timeout)
        world = client.load_world(carla_town) if carla_town else client.get_world()
        
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

        # 4. カメラ設置
        camera_rig = NuScenesCameraRig(world, output_dir)
        camera_rig.spawn(vehicle)

        print(f"データ収集を開始します ({duration:.1f}秒間)...")
        
        # 6. シミュレーションループ
        # クライアント側で時間を進める（非同期モードの場合はwaitのみ）
        time.sleep(duration)
    finally:
        # 7. クリーンアップ
        print("終了処理中: 生成したActorを破棄します...")
        if camera_rig is not None:
            camera_rig.destroy()
        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()
        print("完了")

if __name__ == '__main__':
    args = parse_args()
    try:
        main(record_seconds=args.record_seconds)
    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")
    except RuntimeError as e:
        print(f"Error: {e}")
