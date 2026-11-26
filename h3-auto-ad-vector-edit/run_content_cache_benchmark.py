#!/usr/bin/env python3
"""Benchmark exact content-addressed condition-package reuse versus edited misses."""
import hashlib
import json
import subprocess
from pathlib import Path

BASE = Path("/home/ma-user/work/lienyu/h3-auto-ad-vector-edit")
PYTHON = "/home/ma-user/work/minimax-h3-deployment/venv/bin/python"
RUNNER = str(BASE / "scripts/run_h3.py")


def content_id(package: str) -> str:
    manifest = json.loads((BASE / "packages" / package / "manifest.json").read_text())
    files = manifest["files"]
    dependencies = {name: files[name]["tensor_sha256"] for name in ("conditioning", "latent", "noise")}
    return hashlib.sha256(json.dumps(dependencies, sort_keys=True).encode()).hexdigest()


def run(package: str, name: str):
    process = subprocess.run(
        [PYTHON, RUNNER, "replay", "--duration", "6", "--package", package, "--run", name],
        check=True, text=True, capture_output=True,
    )
    return json.loads(process.stdout.strip().splitlines()[-1])


def main():
    requested = ["cache-alias-1", "cache-alias-2", "cache-alias-3", "pilot-color-blue-a1"]
    registry = {}
    records = []
    for index, package in enumerate(requested):
        identity = content_id(package)
        canonical = registry.setdefault(identity, package)
        records.append({"requested": package, "content_id": identity, "canonical": canonical,
                        "result": run(canonical, f"content-cache-{index}")})
    (BASE / "evidence/content-cache.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
