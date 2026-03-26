#!/usr/bin/env bash
set -euo pipefail

# Curriculum + randomized map training for Proly.
# Phase 1: easier map warmup
# Phase 2: medium map stabilization
# Phase 3: random map mixing for robustness

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_PATH="$SCRIPT_DIR/Proly.app"

if [[ ! -d "$APP_PATH" ]]; then
  echo "Proly.app not found at $APP_PATH"
  exit 1
fi

cd "$SCRIPT_DIR"
source ../.venv/bin/activate

run_map () {
  local worker_id="$1"
  local map_id="$2"
  local episodes="$3"
  echo "=== worker ${worker_id} map ${map_id} episodes ${episodes} ==="
  python -m mlgame3d \
    -w "$worker_id" \
    -i rl_play.py -i hidden -i hidden -i hidden \
    -e "$episodes" -ng -ts 5 -dp 10 \
    -gp items 0 -gp audio false \
    -gp map "$map_id" -gp checkpoint 10 -gp max_time 120 \
    "$APP_PATH"
}

pkill -f "python -m mlgame3d" || true
pkill -f "$APP_PATH/Contents/MacOS/Proly" || true
sleep 1

# Phase 1: map 1 warmup
run_map 61 1 25

# Phase 2: map 2 stabilization
run_map 62 2 25

# Phase 3: randomized map order, repeated rounds
for round in 1 2 3; do
  maps=(1 2 3)
  maps=($(python - <<'PY'
import random
maps=[1,2,3]
random.shuffle(maps)
print(' '.join(map(str,maps)))
PY
))

  for map_id in "${maps[@]}"; do
    run_map "$((70 + round * 10 + map_id))" "$map_id" 18
  done

done

echo "Curriculum random training completed."
