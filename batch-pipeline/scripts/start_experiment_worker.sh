#!/usr/bin/env bash
set -euo pipefail

BASE=/home/ma-user/work/lienyu/minimax-h3-embedding-validation
PROD=/home/ma-user/work/minimax-h3-deployment

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u

export ASCEND_RT_VISIBLE_DEVICES=6,7
export HCCL_OP_EXPANSION_MODE=AIV
export COMFYUI_MULTI_DEVICE_BACKEND=auto
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export LIENYU_H3_PACKAGE_ROOT="$BASE/packages"
if [[ -f /lib/aarch64-linux-gnu/libgomp.so.1 ]]; then
    export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
fi

cd "$BASE/ComfyUI"
exec "$PROD/venv/bin/python" main.py \
    --listen 127.0.0.1 --port 8192 --disable-auto-launch --cache-classic \
    --extra-model-paths-config "$BASE/configs/extra_model_paths.yaml" \
    --user-directory "$BASE/instances/worker-4/user" \
    --input-directory "$BASE/instances/worker-4/input" \
    --temp-directory "$BASE/instances/worker-4/temp" \
    --output-directory "$BASE/output" \
    --database-url "sqlite:///$BASE/instances/worker-4/user/comfyui.db"
