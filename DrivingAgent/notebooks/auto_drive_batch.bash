#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/auto_drive_batch.py"
CONFIG_DIR="$(cd "$SCRIPT_DIR/../config" && pwd)"
OUTPUT_BASE="/home/tsuruoka/nfs/BEV/CarlaRunner/20251213_OccupancyAD_SystemEval_result"
SUCCESS_TARGET=5
MAX_ATTEMPTS=15

run_batch() {
  local label="$1"
  local prob="$2"
  local config_path="$3"
  local target_dir="$OUTPUT_BASE/${label}_${prob}"

  if [[ ! -f "$config_path" ]]; then
    echo "Config not found: $config_path" >&2
    exit 1
  fi

  mkdir -p "$target_dir"
  echo "=== Running $label attack (prob=$prob) -> $target_dir ==="
  python3 "$PY_SCRIPT" \
    --attack-config-path "$config_path" \
    --camera-output-root-path "$target_dir" \
    --success-target "$SUCCESS_TARGET" \
    --max-attempts "$MAX_ATTEMPTS"
}



for prob in 0.8 0.4 0.2 0.0; do
  run_batch "Appearing" "$prob" "$CONFIG_DIR/attack_appearing_${prob}.json"
done
for prob in 0.4 0.2 0.0; do
  run_batch "Hiding" "$prob" "$CONFIG_DIR/attack_hiding_${prob}.json"
done