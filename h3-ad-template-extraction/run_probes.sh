#!/usr/bin/env bash
set -euo pipefail
B=/home/ma-user/work/lienyu/h3-ad-template-extraction; P=/home/ma-user/work/minimax-h3-deployment/venv/bin/python
export H3_EXPERIMENT_BASE="$B"
"$P" "$B/scripts/make_probes.py"
for i in 0 1 2 3; do "$P" "$B/scripts/run_h3.py" replay --package cond-quarter-$i --run cond-quarter-$i; done
for i in 0 1 2 3; do "$P" "$B/scripts/run_h3.py" replay --package noise-time-$i --run noise-time-$i; done
