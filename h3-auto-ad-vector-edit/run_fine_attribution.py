#!/usr/bin/env python3
"""Causal token, DiT block-group and sampler-budget probes on one fixed H3 package."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from safetensors.torch import load_file, save_file

BASE = Path("/home/ma-user/work/lienyu/h3-auto-ad-vector-edit")
PYTHON = "/home/ma-user/work/minimax-h3-deployment/venv/bin/python"
RUNNER = str(BASE / "scripts/run_h3.py")
SOURCE = "gen-performance-s6101"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(package: str, run: str, extra=()):
    command = [PYTHON, RUNNER, "replay", "--duration", "6", "--package", package, "--run", run, *extra]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def zero_token(index: int) -> str:
    destination = BASE / "packages" / f"attr-token-{index:02d}"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(BASE / "packages" / SOURCE, destination)
    tensors = load_file(destination / "conditioning.safetensors")
    output = dict(tensors)
    output["t000000"] = output["t000000"].clone()
    output["t000000"][:, index, :] = 0
    save_file(output, destination / "conditioning.safetensors")
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["conditioning"]["tensor_sha256"] = sha256(destination / "conditioning.safetensors")
    manifest["causal_ablation"] = {"type": "single_condition_row_zero", "row": index}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return destination.name


def main():
    tensor = load_file(BASE / "packages" / SOURCE / "conditioning.safetensors")["t000000"].float()[0]
    top_rows = tensor.norm(dim=1).topk(8).indices.tolist()
    records = [{"probe": "baseline", "result": replay(SOURCE, "attr-baseline") }]
    for row in top_rows:
        package = zero_token(row)
        records.append({"probe": "token", "row": row, "result": replay(package, package)})
    for start in range(0, 50, 10):
        records.append({"probe": "block_group", "start": start, "end": start + 10,
                        "result": replay(SOURCE, f"attr-block-{start:02d}-{start+10:02d}",
                                         ("--skip-start", str(start), "--skip-end", str(start + 10)))})
    records.append({"probe": "steps", "steps": 8,
                    "result": replay(SOURCE, "attr-steps-8", ("--steps", "8"))})
    (BASE / "evidence/fine-attribution.json").write_text(json.dumps({"source": SOURCE, "top_rows": top_rows, "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
