#!/usr/bin/env bash
set -euo pipefail
BASE=/home/ma-user/work/lienyu/h3-auto-ad-vector-edit
PY=/home/ma-user/work/minimax-h3-deployment/venv/bin/python
RUN="$BASE/scripts/run_h3.py"
run(){ "$PY" "$RUN" "$@"; }

run normal --prompt 'Cinematic luxury car advertisement: a glossy red sports coupe drives along a dramatic coastal highway at golden hour, a confident male driver visible through the side window, smooth tracking camera, premium commercial lighting, ocean cliffs and realistic engine sound.' --seed 4101 --package pilot-base --run pilot-base

run normal --encode-only --prompt 'A studio product photograph of a red cube.' --package donor-color-red --run donor-color-red
run normal --encode-only --prompt 'A studio product photograph of a blue cube.' --package donor-color-blue --run donor-color-blue
run normal --encode-only --prompt 'A vehicle is a low sports coupe.' --package donor-car-coupe --run donor-car-coupe
run normal --encode-only --prompt 'A vehicle is a tall luxury SUV.' --package donor-car-suv --run donor-car-suv
run normal --encode-only --prompt 'A product is displayed on a coastal highway at golden hour.' --package donor-scene-coast --run donor-scene-coast
run normal --encode-only --prompt 'A product is displayed on a snowy mountain road at blue hour.' --package donor-scene-snow --run donor-scene-snow
run normal --encode-only --prompt 'A confident male driver is visible.' --package donor-person-male --run donor-person-male
run normal --encode-only --prompt 'A confident female driver is visible.' --package donor-person-female --run donor-person-female

edit(){ "$PY" "$BASE/scripts/edit_condition_direction.py" "$@"; }
edit --base pilot-base --source donor-color-red --target donor-color-blue --output pilot-color-blue-a1 --alpha 1 --mode single-row
edit --base pilot-base --source donor-car-coupe --target donor-car-suv --output pilot-car-suv-a1 --alpha 1 --mode single-row
edit --base pilot-base --source donor-scene-coast --target donor-scene-snow --output pilot-scene-snow-a1 --alpha 1 --mode global
edit --base pilot-base --source donor-person-male --target donor-person-female --output pilot-person-female-a1 --alpha 1 --mode single-row

run replay --package pilot-base --run pilot-replay-base
run replay --package pilot-color-blue-a1 --run pilot-color-blue-a1
run replay --package pilot-car-suv-a1 --run pilot-car-suv-a1
run replay --package pilot-scene-snow-a1 --run pilot-scene-snow-a1
run replay --package pilot-person-female-a1 --run pilot-person-female-a1
