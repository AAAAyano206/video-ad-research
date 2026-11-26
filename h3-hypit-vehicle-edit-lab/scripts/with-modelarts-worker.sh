#!/usr/bin/env bash
set -euo pipefail

lab="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$lab"
export H3_EXPERIMENT_BASE=/home/ma-user/work/lienyu/h3-auto-ad-vector-edit
deployment=/home/ma-user/work/minimax-h3-deployment
export H3_PYTHON="$deployment/venv/bin/python"
mkdir -p "$lab/outputs/worker-leases"
exec 9>"$lab/outputs/worker-leases/lock"
flock -n 9
experiment_pid=
restore=0
lease_log="$(mktemp -d "$lab/outputs/worker-leases/run-XXXXXX")"

ready() {
  curl -fsS --max-time 2 "http://127.0.0.1:$1/system_stats" >/dev/null 2>&1
}

wait_ready() {
  for attempt in $(seq 1 60); do
    if ready "$1"; then return 0; fi
    kill -0 "$2" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}

cleanup() {
  local status=$?
  trap - EXIT
  set +e
  if [[ -n "$experiment_pid" ]]; then
    kill "$experiment_pid" 2>/dev/null
    wait "$experiment_pid" 2>/dev/null
  fi
  if (( restore )) && ! ready 8191; then
    (
      cd "$deployment" || exit
      nohup bash runtime/start.sh 4 < /dev/null >> runtime/worker-4.log 2>&1 8>&- 9>&- &
      echo $! > runtime/worker-4.pid
    )
    wait_ready 8191 "$(cat "$deployment/runtime/worker-4.pid")" || status=1
  fi
  for port in 8188 8189 8190 8191; do
    if ready "$port"; then printf '%s ready\n' "$port"; else printf '%s unavailable\n' "$port"; status=1; fi
  done
  if ready 8192; then printf '8192 still listening\n' >&2; status=1; fi
  exit "$status"
}

if ready 8192; then echo "Experimental port already in use" >&2; exit 1; fi
production_pid="$(cat "$deployment/runtime/worker-4.pid")"
[[ "$production_pid" =~ ^[1-9][0-9]*$ ]]
[[ "$(readlink -f "/proc/$production_pid/cwd")" == "$deployment/ComfyUI" ]]
tr '\0' ' ' < "/proc/$production_pid/cmdline" | grep -F -- '--port 8191 ' >/dev/null
curl -fsS --max-time 5 http://127.0.0.1:8191/queue | python3 -c 'import json, sys
queue = json.load(sys.stdin)
sys.exit("Production worker 4 has active work") if queue["queue_running"] or queue["queue_pending"] else None'

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
restore=1
kill "$production_pid"
for attempt in $(seq 1 45); do
  if ! kill -0 "$production_pid" 2>/dev/null; then break; fi
  sleep 1
done
if kill -0 "$production_pid" 2>/dev/null; then echo "Production worker is still exiting" >&2; exit 1; fi
bash "$H3_EXPERIMENT_BASE/scripts/start_worker.sh" > "$lease_log/worker.log" 2>&1 8>&- 9>&- &
experiment_pid=$!
if ! wait_ready 8192 "$experiment_pid"; then tail -n 50 "$lease_log/worker.log" >&2; exit 1; fi
printf '8192 ready (pid %s)\n' "$experiment_pid"
"$@"
