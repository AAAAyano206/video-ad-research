#!/usr/bin/env python3
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


BASE = Path("/home/ma-user/work/lienyu/minimax-h3-embedding-validation")
RUNNER = BASE / "scripts/run_phase2.py"
CASES = [
    ("p01", "A blue paper boat drifts across a quiet rain puddle while a sparrow watches from the curb, natural rain ambience.", "red-to-blue"),
    ("p10", "A child in a red coat flies a rainbow kite on a windy beach while a golden retriever chases its shadow, energetic handheld camera.", "blue-to-red"),
    ("p12", "Close-up of a calligrapher painting a red ink dragon on white rice paper; the brush strokes flow confidently while an incense ribbon curls in the background.", "black-to-red"),
]


def make_intervention(source_name, alternate_name, label):
    source = BASE / "packages" / source_name
    alternate = BASE / "packages" / alternate_name
    target_name = f"{source_name}-attribute-{label}"
    target = BASE / "packages" / target_name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    base_tensors = load_file(source / "conditioning.safetensors")
    alt_tensors = load_file(alternate / "conditioning.safetensors")
    base = base_tensors["t000000"].float()
    alt = alt_tensors["t000000"].float()
    if base.shape != alt.shape:
        raise RuntimeError(f"shape mismatch {base.shape} != {alt.shape}")
    row_l2 = (alt - base).square().mean(dim=-1).sqrt()[0]
    index = int(row_l2.argmax())
    edited = base.clone()
    edited[:, index, :] = alt[:, index, :]
    output = dict(base_tensors)
    output["t000000"] = edited.to(base_tensors["t000000"].dtype)
    save_file(output, target / "conditioning.safetensors")
    import hashlib
    digest = hashlib.sha256((target / "conditioning.safetensors").read_bytes()).hexdigest()
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["conditioning"]["tensor_sha256"] = digest
    manifest["attribute_intervention"] = {
        "label": label, "strategy": "replace-single-most-changed-contextual-token-row",
        "row_index": index, "row_l2": float(row_l2[index]), "contains_prompt": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return target_name, manifest["attribute_intervention"]


def invoke(mode, seed, package, run, prompt=None):
    command = [sys.executable, str(RUNNER), mode, "--seed", str(seed), "--package", package,
               "--run", run, "--timeout", "1200"]
    if prompt is not None:
        command += ["--prompt", prompt]
    return json.loads(subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip().splitlines()[-1])


def main():
    out = BASE / "evidence/phase4-attribute.jsonl"
    done = {json.loads(x)["prompt_id"] for x in out.read_text().splitlines()} if out.exists() else set()
    for prompt_id, alternate_prompt, label in CASES:
        if prompt_id in done:
            continue
        source = f"phase3-{prompt_id}-s101"
        alternate = f"phase4-{prompt_id}-alternate-s101"
        alt_result = invoke("normal", 101, alternate, f"phase4-attribute-{prompt_id}-alternate", alternate_prompt)
        edited_package, intervention = make_intervention(source, alternate, label)
        edited_result = invoke("replay", 101, edited_package, f"phase4-attribute-{prompt_id}-edited")
        item = {"prompt_id": prompt_id, "label": label, "source_package": source,
                "alternate_package": alternate, "edited_package": edited_package,
                "alternate": alt_result, "edited": edited_result, "intervention": intervention}
        with out.open("a") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(json.dumps({"completed": prompt_id, "intervention": intervention}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
