#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_PATH="$SCRIPT_DIR/Proly.app"
LOG_DIR="$SCRIPT_DIR/auto_logs"
BEST_MODEL="$SCRIPT_DIR/best_model.zip"
BEST_SCORE_FILE="$LOG_DIR/best_score.txt"

ROUNDS="${ROUNDS:-6}"
TRAIN_EPISODES_PER_MAP="${TRAIN_EPISODES_PER_MAP:-15}"
EVAL_EPISODES_PER_MAP="${EVAL_EPISODES_PER_MAP:-8}"
TIME_SCALE="${TIME_SCALE:-5}"
DECISION_PERIOD="${DECISION_PERIOD:-10}"

mkdir -p "$LOG_DIR"

if [[ ! -d "$APP_PATH" ]]; then
  echo "Proly.app not found at $APP_PATH"
  exit 1
fi

cd "$SCRIPT_DIR"
source ../.venv/bin/activate

if [[ ! -f "$BEST_SCORE_FILE" ]]; then
  echo "-1" > "$BEST_SCORE_FILE"
fi

cleanup_procs() {
  pkill -f "python -m mlgame3d" || true
  pkill -f "$APP_PATH/Contents/MacOS/Proly" || true
  sleep 1
}

train_one() {
  local worker_id="$1"
  local map_id="$2"
  local episodes="$3"

  echo "[train] worker=$worker_id map=$map_id episodes=$episodes"
  python -m mlgame3d \
    -w "$worker_id" \
    -i rl_play.py -i hidden -i hidden -i hidden \
    -e "$episodes" -ng -ts "$TIME_SCALE" -dp "$DECISION_PERIOD" \
    -gp items 0 -gp audio false \
    -gp map "$map_id" -gp checkpoint 10 -gp max_time 120 \
    "$APP_PATH"
}

eval_one() {
  local worker_id="$1"
  local map_id="$2"
  local episodes="$3"
  local csv_out="$4"

  echo "[eval] worker=$worker_id map=$map_id episodes=$episodes"
  rm -f "$csv_out"
  python -m mlgame3d \
    -w "$worker_id" \
    -i eval_play.py -i hidden -i hidden -i hidden \
    -e "$episodes" -ng -ts "$TIME_SCALE" -dp "$DECISION_PERIOD" \
    -gp items 0 -gp audio false \
    -gp map "$map_id" -gp checkpoint 10 -gp max_time 120 \
    -o "$csv_out" \
    "$APP_PATH"
}

score_csv() {
  local csv_file="$1"
  python - "$csv_file" <<'PY'
import csv
import sys

p = sys.argv[1]
rows = []
with open(p, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            rows.append(int(r.get("checkpoint", "0")))
        except ValueError:
            pass

if not rows:
    print("0.0")
else:
    print(sum(rows) / len(rows))
PY
}

cleanup_procs

for round in $(seq 1 "$ROUNDS"); do
  echo "================ ROUND $round / $ROUNDS ================"

  maps=(1 2 3)
  maps=($(python - <<'PY'
import random
maps = [1,2,3]
random.shuffle(maps)
print(' '.join(map(str, maps)))
PY
))

  # Curriculum-style training in random order
  wid_base=$((100 + round * 10))
  i=0
  for m in "${maps[@]}"; do
    train_one "$((wid_base + i))" "$m" "$TRAIN_EPISODES_PER_MAP"
    i=$((i + 1))
  done

  # Deterministic evaluation on each map
  total_score="0.0"
  for m in 1 2 3; do
    eval_csv="$LOG_DIR/eval_round_${round}_map_${m}.csv"
    eval_one "$((200 + round * 10 + m))" "$m" "$EVAL_EPISODES_PER_MAP" "$eval_csv"
    map_score="$(score_csv "$eval_csv")"
    echo "[eval] round=$round map=$m checkpoint_mean=$map_score"
    total_score="$(python - <<PY
print($total_score + $map_score)
PY
)"
  done

  avg_score="$(python - <<PY
print($total_score / 3.0)
PY
)"
  echo "[eval] round=$round average_checkpoint_mean=$avg_score"
  echo "$round,$avg_score" >> "$LOG_DIR/round_scores.csv"

  best_score="$(cat "$BEST_SCORE_FILE")"
  better="$(python - <<PY
print(1 if float($avg_score) > float($best_score) else 0)
PY
)"

  if [[ "$better" == "1" ]]; then
    cp "$SCRIPT_DIR/model.zip" "$BEST_MODEL"
    echo "$avg_score" > "$BEST_SCORE_FILE"
    echo "[best] updated best_model.zip with score=$avg_score"
  else
    echo "[best] keep existing best score=$best_score"
  fi

done

echo "Auto optimization complete. Best score: $(cat "$BEST_SCORE_FILE")"
