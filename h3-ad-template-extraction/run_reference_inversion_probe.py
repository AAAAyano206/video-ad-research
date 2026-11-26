#!/usr/bin/env python3
"""Probe H3 reference-video conditioning as a prompt-free-source reconstruction heuristic."""
import json
import os
import subprocess
from pathlib import Path

BASE = Path("/home/ma-user/work/lienyu/h3-ad-template-extraction")
VECTOR = Path("/home/ma-user/work/lienyu/h3-auto-ad-vector-edit")
PYTHON = "/home/ma-user/work/minimax-h3-deployment/venv/bin/python"
RUNNER = str(VECTOR / "scripts/run_h3.py")


def call(arguments):
    environment = dict(os.environ, H3_EXPERIMENT_BASE=str(BASE))
    process = subprocess.run([PYTHON, RUNNER, *arguments], check=True, text=True, capture_output=True, env=environment)
    return json.loads(process.stdout.strip().splitlines()[-1])


def main():
    source = VECTOR / "outputs/h3_gen-performance-s6101_00001_.mp4"
    generic = "Use <Video 1> as the sole visual advertisement reference and reproduce its subject, action, composition and timing."
    encoded = call(["normal", "--prompt", generic, "--reference-video", str(source), "--duration", "7",
                    "--seed", "6101", "--package", "blind-reference", "--run", "blind-reference-normal"])
    replay = call(["replay", "--duration", "7", "--package", "blind-reference", "--run", "blind-reference-replay-refmodel", "--reference-model"])
    evidence = {"source_video": str(source), "original_prompt_supplied": False,
                "fixed_generic_instruction": generic, "reference_generation": encoded, "package_replay": replay,
                "claim_boundary": "reference-conditioned regeneration, not unique latent inversion"}
    (BASE / "evidence/reference-inversion.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
