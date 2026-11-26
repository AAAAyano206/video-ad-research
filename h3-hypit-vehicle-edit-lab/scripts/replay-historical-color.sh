#!/usr/bin/env bash
set -euo pipefail

: "${H3_EXPERIMENT_BASE:?set this to the existing H3 experiment directory}"
: "${H3_PYTHON:?set this to the Python executable from the existing H3 environment}"
package_root="${LIENYU_H3_PACKAGE_ROOT:-$H3_EXPERIMENT_BASE/packages}"
bridge_script="${H3_BRIDGE_SCRIPT:-$H3_EXPERIMENT_BASE/scripts/run_h3.py}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -f "$package_root/gen-performance-s6101/manifest.json" ]]; then
  printf 'Missing historical H3 package: %s\n' "$package_root/gen-performance-s6101" >&2
  exit 1
fi
if [[ ! -f "$bridge_script" || ! -d "$H3_EXPERIMENT_BASE/evidence" ]]; then
  printf 'The H3 bridge or its evidence directory is missing.\n' >&2
  exit 1
fi
if ! curl -fsS --connect-timeout 3 --max-time 5 http://127.0.0.1:8192/system_stats >/dev/null; then
  printf 'The configured replay bridge needs an active worker on this host at port 8192.\n' >&2
  exit 1
fi

work=outputs/historical-color-run
mkdir -p "$work"
plan="$work/plan.json"
if [[ ! -f "$plan" ]]; then
  "$H3_PYTHON" -m vehicle_edit_lab plan --phase pilot \
    --spec experiments/spec.historical-color.json --output "$plan"
fi

for attribute in car scene person color; do
  vector="$work/$attribute.safetensors"
  if [[ -f "$vector" ]]; then
    continue
  fi
  attribute_id="vehicle.$attribute"
  projection=()
  if [[ "$attribute" == color ]]; then
    attribute_id=vehicle.paint_color
    projection=(--nuisance "$work/car.safetensors" --nuisance "$work/scene.safetensors" --nuisance "$work/person.safetensors")
  fi
  "$H3_PYTHON" -m vehicle_edit_lab learn-direction --package-root "$package_root" --attribute "$attribute_id" \
    --pair "gen-donor-$attribute-0-s" "gen-donor-$attribute-0-t" \
    --pair "gen-donor-$attribute-1-s" "gen-donor-$attribute-1-t" \
    --pair "gen-donor-$attribute-2-s" "gen-donor-$attribute-2-t" \
    ${projection[@]+"${projection[@]}"} --output "$vector"
done

if (( $# == 0 )); then
  set -- 6101
fi
for seed in "$@"; do
  case "$seed" in 6101|6102|6103) ;; *) echo "Unknown historical seed: $seed" >&2; exit 2 ;; esac
  case_id="pilot--vehicle.paint_color--coastal-performance--seed-$seed"
  for variant in A0 A1 A2; do
    job="$case_id--$variant"
    if [[ "$variant" != A0 && ! -f "$package_root/$job/manifest.json" ]]; then
      "$H3_PYTHON" -m vehicle_edit_lab prepare --package-root "$package_root" \
        --plan "$plan" --job "$job" --direction "$work/color.safetensors" --alpha 4
    fi
    if [[ ! -f "results/runs/$job/receipt.json" ]]; then
      "$H3_PYTHON" -m vehicle_edit_lab replay --plan "$plan" --job "$job" --package-root "$package_root" \
        --bridge-root "$H3_EXPERIMENT_BASE" --bridge-script "$bridge_script" --worker-python "$H3_PYTHON"
    fi
  done
done
