#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
python="$HOME/work/minimax-h3-deployment/venv/bin/python"
for tick in $(seq 1 240); do
  if grep -qx "8191 ready" paint-native-final.log; then break; fi
  sleep 3
done
test "$(find results/runs -type f -path "*paint_color*--M*/receipt.json" | wc -l)" -eq 18
test "$(find results/runs -type f -path "*paint_color*--P[34]/receipt.json" | wc -l)" -eq 18
"$python" prepare_review.py paint --segmented > blind-paint.log 2>&1 &
blind=$!
H3_SEGMENTED=1 bash worker.sh body
wait "$blind"
"$python" prepare_review.py body --segmented > blind-body.log 2>&1
