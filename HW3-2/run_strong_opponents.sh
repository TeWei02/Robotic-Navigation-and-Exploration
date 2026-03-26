#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="/Users/kedewei/Desktop/ＨＷ3/.venv/bin/python"
APP_PATH="./Proly.app"

# Four strong agents: all players use champion_play.
"$PYTHON_BIN" -m mlgame3d \
  -w 460 \
  -i champion_play.py -i champion_play.py -i champion_play.py -i champion_play.py \
  -e 3 -ts 1 -dp 5 \
  -gp items 0 -gp audio true -gp map 1 -gp checkpoint 3 -gp max_time 25 \
  "$APP_PATH"
