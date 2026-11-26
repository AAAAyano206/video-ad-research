#!/usr/bin/env bash
set -euo pipefail
B=/home/ma-user/work/lienyu/h3-auto-ad-vector-edit; P=/home/ma-user/work/minimax-h3-deployment/venv/bin/python
R="$B/scripts/run_h3.py"; E="$B/scripts/edit_condition_direction.py"
"$P" "$R" normal --duration 15 --prompt 'Cinematic luxury car advertisement: a glossy red sports coupe drives along a dramatic coastal highway at golden hour, a confident male driver visible through the side window, smooth tracking camera, premium commercial lighting, ocean cliffs and realistic engine sound.' --seed 5101 --package ad15-base --run ad15-base
"$P" "$E" --base ad15-base --source donor-color-red --target donor-color-blue --output ad15-color-blue --alpha 4 --mode global
"$P" "$E" --base ad15-base --source donor-car-coupe --target donor-car-suv --output ad15-car-suv --alpha 4 --mode global
"$P" "$E" --base ad15-base --source donor-scene-coast --target donor-scene-snow --output ad15-scene-snow --alpha 8 --mode global
"$P" "$E" --base ad15-base --source donor-person-male --target donor-person-female --output ad15-person-female --alpha 4 --mode global
for x in ad15-base ad15-color-blue ad15-car-suv ad15-scene-snow ad15-person-female; do "$P" "$R" replay --duration 15 --package "$x" --run "${x}-replay"; done
