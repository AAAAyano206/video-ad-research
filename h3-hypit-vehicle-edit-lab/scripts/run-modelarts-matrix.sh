#!/usr/bin/env bash
set -euo pipefail

lab="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$lab"
batch="$1"
mkdir -p "$batch" "$lab/outputs/worker-leases"
exec 8>"$lab/outputs/worker-leases/matrix.lock"
flock -n 8
exit_code=0

run() {
    local tag="$1"
    shift
    if [[ -e "$batch/$tag.status" ]]; then
        if [[ "$(cat "$batch/$tag.status")" != 0 ]]; then exit_code=1; fi
        return
    fi
    printf '%s start %s\n' "$(date -Iseconds)" "$tag"
    local status=0
    bash scripts/with-modelarts-worker.sh /home/ma-user/work/minimax-h3-deployment/venv/bin/python \
        -m vehicle_edit_lab.run_modelarts "$@" > "$batch/$tag.log" 2>&1 8>&- || status=$?
    printf '%s\n' "$status" > "$batch/$tag.status"
    printf '%s finish %s status=%s\n' "$(date -Iseconds)" "$tag" "$status"
    if (( status != 0 )); then exit_code=1; fi
}

run continuity continuity
for attr in body wheel window; do run "pilot-$attr" pilot "$attr"; done
for attr in body wheel window; do
    if [[ "$(cat "$batch/pilot-$attr.status")" == 0 ]]; then
        run "reference-$attr" pilot "$attr" --part refs
    fi
done
run holdout-paint holdout paint
for attr in body wheel window; do
    if [[ "$(cat "$batch/pilot-$attr.status")" == 0 ]]; then
        run "holdout-$attr" holdout "$attr"
    fi
done
exit "$exit_code"
