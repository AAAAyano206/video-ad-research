#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_project="$HOME/work/lienyu/h3-auto-ad-vector-edit"
deployment="$HOME/work/minimax-h3-deployment"
export LIENYU_LOCALIZED_RUN_ROOT="$root"
export LIENYU_H3_PACKAGE_ROOT="$source_project/packages"
export PATH="/usr/local/ffmpeg/bin:$PATH"
ready() { curl -fsS --max-time 2 "http://127.0.0.1:$1/system_stats" >/dev/null 2>&1; }
queue() { curl -fsS --max-time 3 "http://127.0.0.1:$1/queue"; }
wait_ready() {
    for attempt in $(seq 1 90); do
        if ready "$1"; then return 0; fi
        kill -0 "$2" 2>/dev/null || return 1
        sleep 1
    done
    return 1
}

if [[ "${1:-}" == "start-isolated" ]]; then
    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
    export ASCEND_RT_VISIBLE_DEVICES=6,7 HCCL_OP_EXPANSION_MODE=AIV COMFYUI_MULTI_DEVICE_BACKEND=auto
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
    if [[ -f /lib/aarch64-linux-gnu/libgomp.so.1 ]]; then
        export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
    fi
    cd "$root/ComfyUI"
    exec "$deployment/venv/bin/python" main.py --listen 127.0.0.1 --port 8192 \
      --disable-auto-launch --cache-classic --extra-model-paths-config "$source_project/configs/extra_model_paths.yaml" \
      --user-directory "$root/instance/user" --input-directory "$root/instance/input" \
      --temp-directory "$root/instance/temp" --output-directory "$root/worker-output" \
      --database-url "sqlite:///$root/instance/user/comfyui.db"
fi

cd "$root"
previous="$HOME/work/lienyu/h3-hypit-vehicle-edit-lab"
mkdir -p "$previous/outputs/worker-leases" "$root/instance/"{user,input,temp} "$root/worker-output"
exec 9>"$previous/outputs/worker-leases/lock"
flock -n 9
[[ -f "$root/ComfyUI/main.py" ]] || cp -a "$source_project/ComfyUI" "$root/ComfyUI"
mkdir -p "$root/ComfyUI/custom_nodes/lienyu_localized_nodes"
cp "$root/lienyu_localized_nodes/__init__.py" "$root/ComfyUI/custom_nodes/lienyu_localized_nodes/"
runlog="$(mktemp -d "$root/worker-XXXXXX")"
worker_pid=
restore=0

cleanup() {
    local status=$?
    trap - EXIT
    set +e
    if [[ -n "$worker_pid" ]]; then
        kill "$worker_pid" 2>/dev/null
        wait "$worker_pid" 2>/dev/null
    fi
    if (( restore )) && ! ready 8191; then
        (
            cd "$deployment" || exit
            nohup bash runtime/start.sh 4 < /dev/null >> runtime/worker-4.log 2>&1 7>&- 8>&- 9>&- &
            echo $! > runtime/worker-4.pid
        )
        wait_ready 8191 "$(cat "$deployment/runtime/worker-4.pid")" || status=1
    fi
    for port in 8188 8189 8190 8191; do
        if ready "$port"; then printf '%s ready\n' "$port"; else printf '%s unavailable\n' "$port"; status=1; fi
    done
    if ready 8192; then printf 'experimental worker still listening\n' >&2; status=1; fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for port in 8188 8189 8190 8191; do ready "$port"; done
if ready 8192; then echo "experimental port already in use"; exit 1; fi
exec 7>"$deployment/runtime/h3-bridge.dispatch.lock"
exec 8>"$deployment/runtime/submit.lock"
flock -w 8 7
flock -w 8 8
production_pid="$(cat "$deployment/runtime/worker-4.pid")"
[[ "$production_pid" =~ ^[1-9][0-9]*$ ]]
[[ "$(readlink -f "/proc/$production_pid/cwd")" == "$deployment/ComfyUI" ]]
tr '\0' ' ' < "/proc/$production_pid/cmdline" | grep -F -- '--port 8191 ' >/dev/null
queue 8191 | python3 -c 'import json,sys; obj=json.load(sys.stdin); sys.exit(bool(obj["queue_running"] or obj["queue_pending"]))'
restore=1
kill "$production_pid"
for attempt in $(seq 1 45); do
    if ! kill -0 "$production_pid" 2>/dev/null; then break; fi
    sleep 1
done
if kill -0 "$production_pid" 2>/dev/null; then echo "reserved worker is still exiting"; exit 1; fi
flock -u 8
flock -u 7
bash "$root/worker.sh" start-isolated > "$runlog/comfy.log" 2>&1 7>&- 8>&- 9>&- &
worker_pid=$!
wait_ready 8192 "$worker_pid" || { tail -n 40 "$runlog/comfy.log"; exit 1; }
curl -fsS --max-time 10 http://127.0.0.1:8192/object_info/LienyuLocalizedH3Sampler |
    python3 -c 'import json,sys; assert "LienyuLocalizedH3Sampler" in json.load(sys.stdin)'
printf 'isolated worker ready at 8192\n'
for stage in "$@"; do
    "$deployment/venv/bin/python" "$root/run.py" "$stage"
done
