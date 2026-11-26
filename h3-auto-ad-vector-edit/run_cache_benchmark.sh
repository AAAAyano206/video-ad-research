#!/usr/bin/env bash
set -euo pipefail
B=/home/ma-user/work/lienyu/h3-auto-ad-vector-edit; P=/home/ma-user/work/minimax-h3-deployment/venv/bin/python; R=$B/scripts/run_h3.py
for i in 1 2 3; do "$P" "$R" replay --package pilot-base --run cache-exact-$i; done
for i in 1 2 3; do
  rm -rf "$B/packages/cache-alias-$i"; cp -a "$B/packages/pilot-base" "$B/packages/cache-alias-$i"
  "$P" "$R" replay --package cache-alias-$i --run cache-alias-$i
done
for x in scan-color-blue-g4 scan-car-suv-g4 scan-person-female-g4; do "$P" "$R" replay --package "$x" --run cache-edited-$x; done
