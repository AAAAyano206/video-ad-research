#!/usr/bin/env bash
set -euo pipefail
B=/home/ma-user/work/lienyu/h3-auto-ad-vector-edit; P=/home/ma-user/work/minimax-h3-deployment/venv/bin/python
edit(){ "$P" "$B/scripts/edit_condition_direction.py" "$@"; }
run(){ "$P" "$B/scripts/run_h3.py" replay --package "$1" --run "$1"; }
for alpha in 4 8; do
  edit --base pilot-base --source donor-color-red --target donor-color-blue --output scan-color-blue-g$alpha --alpha $alpha --mode global
  edit --base pilot-base --source donor-car-coupe --target donor-car-suv --output scan-car-suv-g$alpha --alpha $alpha --mode global
  edit --base pilot-base --source donor-scene-coast --target donor-scene-snow --output scan-scene-snow-g$alpha --alpha $alpha --mode global
  edit --base pilot-base --source donor-person-male --target donor-person-female --output scan-person-female-g$alpha --alpha $alpha --mode global
done
for alpha in 4 8; do
  run scan-color-blue-g$alpha; run scan-car-suv-g$alpha; run scan-scene-snow-g$alpha; run scan-person-female-g$alpha
done
