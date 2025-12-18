# CarlaRunner Docs

CARLA Runner は CARLA シミュレータ（0.9.15）と OpenOccupancy ベースの推論/プランナを組み合わせて、自動運転シナリオをバッチ実行・評価するための補助ツール群です。トップレベルは次の 3 つで構成されます。

- `CarlaSimulator/`: Docker 化された CARLA UE4 サーバーの起動スクリプトと接続テスト用ツール。
- `DrivingAgent/`: 走行・推論ロジック一式（notebooks + src + config）。
- `tmp/`: 実験ログや一時出力のデフォルト置き場（Git 管理外）。

よく使うフローと注意点を以下にまとめます。

## 事前準備
1. **GPU/Docker 環境**  
   `carla-server` コンテナを利用するため、Docker (nvidia runtime) を有効にしておく。Ubuntu の場合は `sudo systemctl start docker && sudo systemctl start nvidia-docker`.（GHPC2については設定済みなので不要）
2. **Python 環境**  
   OpenOccupancy と同じ conda 環境（例: `conda activate OpenOccupancy`）を使い、`DrivingAgent/notebooks` から `pip install -r requirements.txt` 等で依存を揃える。
3. **CARLA Python API 対策**  
   CARLA の Python モジュールを読み込む前に **必ず** `LD_PRELOAD` を通す。これを忘れると `carla` import 時に `undefined symbol` が出ます。
   ```bash
   export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
   ```
   - 毎回入力するのが面倒な場合は `.bashrc` や作業用スクリプトに上記を追記。
4. **ログ/データ置き場**  
   デフォルトで `/home/tsuruoka/nfs/BEV/CarlaRunner/...` に画像や結果を吐き出す。別ディスクに保存したいときは `--camera-output-root` 系引数で上書きする。

## ディレクトリ早見表
| パス | 役割 |
| --- | --- |
| `CarlaSimulator/run_carla_simulator.sh` | GPU 付き Docker で CARLA サーバーを起動するスクリプト。引数でポートを切り替え可能。 |
| `CarlaSimulator/carla_connection_test.sh` | `LD_PRELOAD` 済みの状態で CARLA Python API を叩き、接続確認する軽量テスト。 |
| `DrivingAgent/notebooks/drive_and_infer_async.py` | 実際に車両を spawn しつつ Occupancy 推論を並列で回すメインスクリプト。 |
| `DrivingAgent/notebooks/auto_drive_batch.py` | シミュレータ起動 + `drive_and_infer_async.py` を監視しながら連続実行する自動バッチ制御。 |
| `DrivingAgent/notebooks/auto_drive_batch.bash` | 上記 Python ラッパーを異なる Attack Config で順番に呼ぶ便利スクリプト。 |
| `DrivingAgent/config/*.json` | 攻撃設定やプランナ設定。`--attack-config-path` などで参照。 |
| `DrivingAgent/src/*.py` | `drive_and_infer*` から共通利用されるカメラリグ/プランナ/コストマップ処理。 |

## よく使うワークフローと実行例

### 1. CARLA サーバーを起動する
```bash
cd CarlaRunner/CarlaSimulator
# デフォルト 2000 番ポート。シミュレータを 5555 で開きたい場合は引数指定
bash run_carla_simulator.sh 5555
```
- `run_carla_simulator.sh` は `carlasim/carla:0.9.15` イメージを `--net=host` で立てる。
- ポートは以後の `drive_and_infer_async.py --port` と合わせる。

### 2. 単発の自動運転 & Occupancy 推論
1. 先に `export LD_PRELOAD=...` を実施（**重要**）。
2. 別シェルで以下を実行:
```bash
cd CarlaRunner
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
python3 DrivingAgent/notebooks/drive_and_infer_async.py \
  --spawn-index 361 \
  --duration-seconds 10 \
  --traffic-manager-port 60055 \
  --driver-debug \
  --timeout 120 \
  --port 5555 \
  --camera-output-root /home/tsuruoka/hdd/BEV/CarlaRunner/DrivingAgent/data/ \
  --max-pending-frames 128 \
  --tick-timeout-seconds 120 \
  --target-speed-mps 4.0 \
  --use-lane-yaw \
  --use-lane-following \
  --inference-mode sync \
  --log-control-debug \
  --class-weight-json /home/tsuruoka/hdd/BEV/CarlaRunner/DrivingAgent/notebooks/class_weight.json \
  --use-occ-planner \
  --enable-planner-bumper-offset \
  --planner-weight-speed 2.0 \
  --acceleration-gain 1.0 \
  --planner-horizon-s 3.0 \
  --enable-planner-bumper-offset \
  --cost-aggregate max \
  --planner-forward-axis col \
  --planner-weight-occ 1.0 \
  --attack-config ./DrivingAgent/config/attack_hiding.json \
  --test-car-distance 12
```
- `--spawn-index` や `--duration-seconds` で走行条件を変更。
- `--inference-mode {sync,async}` / `--use-lane-following` なども `auto_drive_batch.py` の `DEFAULT_DRIVER_COMMAND` 参照で調整。
- 成功すると `--camera-output-root` 直下に `{CAM_*}/{frame}.png` と `occupancy_pipeline` の結果が保存される。

### 3. 連続実験を自動化する（バッチラン）
CARLA サーバーの起動～ドライバの再立ち上げまでを自動化したい場合は `auto_drive_batch.py` を使う。

```bash
cd CarlaRunner
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
python3 DrivingAgent/notebooks/auto_drive_batch.py \
  --success-target 5 \
  --max-attempts 20 \
  --sim-port 5555 \
  --sim-script CarlaSimulator/run_carla_simulator.sh \
  --attack-config-path DrivingAgent/config/attack_hiding_0.4.json \
  --camera-output-root-path /home/tsuruoka/nfs/BEV/CarlaRunner/20251213_OccupancyAD_SystemEval_result/Hiding_0.4
```
- 指定回数成功するまで CARLA Docker を再起動しながら `drive_and_infer_async.py` を叩き直す。
- Slack 通知を使う場合は `SLACK_URL` を環境変数でセット（空なら送信しない）。
- 複数の設定を順番に回すときは `DrivingAgent/notebooks/auto_drive_batch.bash` を実行すると、Appearing/Hiding で `prob` の異なる構成を一気に回せる。

### 4. コストマップ/占有率のみを後処理する
事後解析だけ行いたい場合は生成済みのカメラ出力に対し `occupancy_pipeline.py` や `3d_ego_costmap.py` など notebook スクリプトを個別に実行可能。
```bash
cd CarlaRunner
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
python3 DrivingAgent/notebooks/occupancy_pipeline.py \
  --camera-root /path/to/CAM_ROOT \
  --spawn-index 361 \
  --window-size 8 \
  --class-weight-json DrivingAgent/notebooks/class_weight.json
```

## トラブルシューティング
- **`carla` import で落ちる** → `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` を忘れていないか確認。
- **シミュレータがポートで衝突** → `run_carla_simulator.sh` の引数と `--port` / `--traffic-manager-port` を合わせる。残っているコンテナは `docker rm -f carla-server`。
- **画像が欠損する/時々フリーズ** → `auto_drive_batch.py --max-pending-frames` を下げるか、`--tick-timeout-seconds` を延長。
- **ログ場所を変えたい** → `--camera-output-root-path` や環境変数 `SLACK_URL` を指定して出力先を変更。NFS が遅いと感じたらローカル SSD に向ける。

## 追加メモ
- 走行設定のテンプレは `DrivingAgent/notebooks/sample_payload*.json`。API 評価パイプライン `2b_build_openocc_payload.py` と連携。
- `DrivingAgent/notebooks/straight_line_driver.py` は最小限のドライバ挙動。`drive_and_infer.py`（同期版）や `drive_and_infer_async.py` がこれを利用する。
- `DrivingAgent/tests/` に ST-P3 プランナのユニットテストがあるのでロジック変更時に `pytest` を回す。

必要に応じてこのドキュメントに実験ログの場所や追加スクリプトを追記していってください。
