#!/usr/bin/env python3
"""Create prompt-free, dequantized replay packages plus measured wire payloads."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


BASE = Path("/home/ma-user/work/lienyu/minimax-h3-embedding-validation")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_manifest(package, metrics):
    path = package / "manifest.json"
    manifest = json.loads(path.read_text())
    tensor_path = package / "conditioning.safetensors"
    manifest["files"]["conditioning"]["tensor_sha256"] = sha256(tensor_path)
    manifest["compression_experiment"] = metrics
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def transform(source_name, method):
    source = BASE / "packages" / source_name
    target_name = f"{source_name}-{method}"
    target = BASE / "packages" / target_name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    tensors = load_file(source / "conditioning.safetensors")
    output = dict(tensors)
    wire = {}
    errors = {}
    for key, value in tensors.items():
        if not value.is_floating_point():
            wire[key] = value
            continue
        original = value.float()
        if method == "fp16":
            encoded = original.half()
            restored = encoded.float()
            wire[key] = encoded
        elif method == "int8-channel":
            reduce_dims = tuple(range(original.ndim - 1))
            scale = original.abs().amax(dim=reduce_dims).clamp_min(1e-12) / 127.0
            encoded = torch.round(original / scale).clamp(-127, 127).to(torch.int8)
            restored = encoded.float() * scale
            wire[key + "__q"] = encoded
            wire[key + "__scale"] = scale
        else:
            raise ValueError(method)
        output[key] = restored.to(value.dtype)
        delta = restored - original
        errors[key] = {
            "max_abs": float(delta.abs().max()),
            "rmse": float(delta.square().mean().sqrt()),
            "cosine": float(torch.nn.functional.cosine_similarity(original.flatten(), restored.flatten(), dim=0)),
        }
    save_file(output, target / "conditioning.safetensors")
    wire_path = target / "conditioning-wire.safetensors"
    save_file(wire, wire_path)
    metrics = {
        "method": method,
        "wire_bytes": wire_path.stat().st_size,
        "original_bytes": (source / "conditioning.safetensors").stat().st_size,
        "ratio": wire_path.stat().st_size / (source / "conditioning.safetensors").stat().st_size,
        "errors": errors,
        "wire_sha256": sha256(wire_path),
        "contains_prompt": False,
    }
    update_manifest(target, metrics)
    return target_name, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="+")
    args = parser.parse_args()
    for source in args.source:
        for method in ("fp16", "int8-channel"):
            name, metrics = transform(source, method)
            print(json.dumps({"package": name, **metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
