import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from .files import LabError, identifier, read_json, require, sha256, write_json, write_new


PACKAGE_FORMAT = "lienyu.h3.condition-package.v1"
PAYLOADS = ("conditioning", "latent", "noise")


def tensor_tools():
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as error:
        raise LabError("condition editing needs PyTorch and safetensors in the H3 Python environment") from error
    return torch, safe_open, save_file


def package_path(root, name):
    root = Path(root).resolve(strict=True)
    path = root / identifier(name, "package name")
    require(path.resolve().parent == root and not path.is_symlink(), f"invalid package path: {path}")
    return path


def open_package(root, name, payloads=PAYLOADS):
    path = package_path(root, name)
    manifest_path = path / "manifest.json"
    require(not manifest_path.is_symlink(), f"symlinked manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    require(manifest.get("format") == PACKAGE_FORMAT, f"unsupported condition package format: {name}")
    require(manifest.get("contains_prompt") is False, f"{name}: package must declare contains_prompt=false")
    for stem in payloads:
        entry = manifest["files"][stem]
        for file_key, hash_key in (("tensor_file", "tensor_sha256"), ("tree_file", "tree_sha256")):
            source = path / identifier(entry[file_key], file_key)
            require(source.is_file() and not source.is_symlink(), f"missing or symlinked payload: {source}")
            require(sha256(source) == entry[hash_key], f"checksum mismatch: {source}")
    return path, manifest


def open_condition(root, name, key, torch, safe_open):
    package, manifest = open_package(root, name, ("conditioning",))
    entry = manifest["files"]["conditioning"]
    with safe_open(package / entry["tensor_file"], framework="pt", device="cpu") as handle:
        require(key in handle.keys(), f"{name}: missing tensor {key}")
        tensor = handle.get_tensor(key)
    require(
        tensor.ndim == 3 and tensor.shape[0] == 1 and tensor.numel() > 0
        and torch.is_floating_point(tensor) and torch.isfinite(tensor).all().item(),
        f"{name}/{key}: expected a finite floating [1, tokens, features] tensor",
    )
    return tensor, entry["tensor_sha256"]


def unit_vector(vector):
    length = vector.norm()
    require(math.isfinite(length.item()) and length.item() > 0, "direction has zero or non-finite norm")
    return vector / length


def open_direction(path, key, safe_open, attribute=None):
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        require(metadata.get("format") == "vehicle-edit-lab.direction.v1", "unsupported direction format")
        require(metadata.get("tensor_key") == key, "direction uses a different tensor key")
        if attribute is not None:
            require(metadata.get("attribute_id") == attribute, "direction attribute does not match job")
        direction = handle.get_tensor("direction").float()
    require(direction.ndim == 1 and direction.numel() > 0, "direction must be a 1-D feature vector")
    return direction


def learn_direction(root, pairs, nuisance_files, attribute, key, output):
    attribute = identifier(attribute, "attribute id", 36)
    key = identifier(key, "tensor key")
    require(len(pairs) >= 3 and len({tuple(pair) for pair in pairs}) == len(pairs), "use at least three distinct donor pairs")
    torch, safe_open, save_file = tensor_tools()
    differences, provenance = [], []
    for before, after in pairs:
        original, before_hash = open_condition(root, before, key, torch, safe_open)
        target, after_hash = open_condition(root, after, key, torch, safe_open)
        require(original.shape[-1] == target.shape[-1], f"feature width differs in {before} and {after}")
        differences.append(target.float().mean(dim=1).flatten() - original.float().mean(dim=1).flatten())
        provenance.append({"source": before, "target": after, "source_sha256": before_hash, "target_sha256": after_hash})
    require(len({vector.shape for vector in differences}) == 1, "donor pairs have different feature widths")
    median_norm = torch.stack([vector.norm() for vector in differences]).median()
    direction = unit_vector(torch.stack([unit_vector(vector) for vector in differences]).mean(dim=0))
    basis = []
    for file in nuisance_files:
        nuisance = open_direction(file, key, safe_open)
        require(nuisance.shape == direction.shape, f"feature width differs in nuisance direction {file}")
        axis = unit_vector(nuisance)
        for previous in basis:
            axis = axis - axis.dot(previous) * previous
        if axis.norm().item() > 1e-6:
            basis.append(unit_vector(axis))
    for axis in basis:
        direction = direction - direction.dot(axis) * axis
    require(direction.norm().item() > 1e-6, "nuisance projection removed the target direction")
    direction = (unit_vector(direction) * median_norm).contiguous()
    metadata = {
        "format": "vehicle-edit-lab.direction.v1",
        "attribute_id": attribute,
        "tensor_key": key,
        "donor_pairs": json.dumps(provenance, ensure_ascii=False),
        "nuisance_sha256": json.dumps([sha256(path) for path in nuisance_files]),
        "torch_version": str(torch.__version__),
    }
    write_new(output, lambda staged: save_file({"direction": direction}, staged, metadata=metadata))


def verify_seed(manifest, job):
    require(str(manifest.get("noise_seed")) == str(job["seed"]), f"{job['id']}: recorded noise seed does not match plan")


def prepare_package(root, job, direction_path, alpha, key):
    require(job["variant"] in {"A1", "A2"}, "only A1 and A2 have a vector package to prepare")
    key = identifier(key, "tensor key")
    require(math.isfinite(alpha) and alpha > 0, "alpha must be finite and positive")
    source_name = job["base_package"]
    source, manifest = open_package(root, source_name)
    verify_seed(manifest, job)
    destination = package_path(root, job["package"])
    require(destination.name == job["id"] and destination != source, "edited package must use the plan job ID")
    require(not destination.exists(), f"package already exists: {destination}")
    direction_digest = sha256(direction_path)
    matching_vector_job = None
    if job["variant"] == "A2":
        matching_vector_job = job["vector_source_job"]
        _, vector_package = open_package(root, matching_vector_job, ("conditioning",))
        vector_edit = vector_package.get("vehicle_edit_lab", {})
        require(
            vector_edit.get("job_id") == matching_vector_job
            and vector_edit.get("variant") == "A1"
            and vector_edit.get("attribute_id") == job["attribute_id"]
            and vector_edit.get("base_package") == source_name
            and vector_edit.get("base_condition_sha256") == manifest["files"]["conditioning"]["tensor_sha256"]
            and vector_edit.get("direction_sha256") == direction_digest
            and vector_edit.get("tensor_key") == key
            and vector_edit.get("alpha") == alpha,
            "prepare the matching A1 first and use its direction, strength and tensor key for A2",
        )
    torch, safe_open, save_file = tensor_tools()
    original, base_hash = open_condition(root, source_name, key, torch, safe_open)
    require(base_hash == manifest["files"]["conditioning"]["tensor_sha256"], "base conditioning changed during preparation")
    direction = open_direction(direction_path, key, safe_open, job["attribute_id"])
    require(direction.shape[0] == original.shape[-1], "direction feature width does not match the condition tensor")
    length = direction.norm().item()
    require(math.isfinite(length) and length > 0, "invalid direction norm")
    random_seed = None
    if job["variant"] == "A2":
        random_seed = job["random_direction_seed"]
        generator = torch.Generator(device="cpu").manual_seed(random_seed)
        direction = unit_vector(torch.randn(direction.shape, generator=generator, dtype=torch.float32)) * length
    delta = alpha * direction
    edited = (original.float() + delta).to(original.dtype).contiguous()
    require(torch.isfinite(edited).all().item() and not torch.equal(original, edited), "edit is non-finite or vanishes in the package dtype")

    with tempfile.TemporaryDirectory(prefix=".vehicle-package-", dir=destination.parent) as staging:
        prepared = Path(staging) / "package"
        prepared.mkdir()
        manifest["files"] = {stem: manifest["files"][stem] for stem in PAYLOADS}
        manifest.pop("generated_latent_saved_at_unix", None)
        for entry in manifest["files"].values():
            for field in ("tree_file", "tensor_file"):
                shutil.copyfile(source / entry[field], prepared / entry[field])
        entry = manifest["files"]["conditioning"]
        staged_tensor = prepared / entry["tensor_file"]
        with safe_open(staged_tensor, framework="pt", device="cpu") as handle:
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}
            tensor_metadata = handle.metadata()
        tensors[key] = edited
        replacement = Path(staging) / "conditioning.safetensors"
        save_file(tensors, replacement, metadata=tensor_metadata)
        os.replace(replacement, staged_tensor)
        entry["tensor_sha256"] = sha256(staged_tensor)
        manifest["vehicle_edit_lab"] = {
            "job_id": job["id"],
            "variant": job["variant"],
            "attribute_id": job["attribute_id"],
            "base_package": source_name,
            "base_condition_sha256": base_hash,
            "tensor_key": key,
            "direction_sha256": direction_digest,
            "alpha": alpha,
            "matching_vector_job": matching_vector_job,
            "random_direction_seed": random_seed,
            "delta_l2_before_cast": delta.norm().item(),
            "actual_change_l2": (edited.float() - original.float()).norm().item(),
            "torch_version": str(torch.__version__),
        }
        write_json(prepared / "manifest.json", manifest)
        require(not destination.exists(), f"package already exists: {destination}")
        prepared.rename(destination)
    return destination


def verify_replay_package(root, job):
    _, baseline = open_package(root, job["base_package"])
    verify_seed(baseline, job)
    if job["variant"] == "A0":
        require(job["package"] == job["base_package"], "baseline must replay the recorded original package")
        return baseline
    require(job["variant"] in {"A1", "A2"}, "this job requires an external generation or reuse step")
    _, prepared = open_package(root, job["package"])
    verify_seed(prepared, job)
    edit = prepared.get("vehicle_edit_lab", {})
    require(
        edit.get("job_id") == job["id"]
        and edit.get("variant") == job["variant"]
        and edit.get("attribute_id") == job["attribute_id"]
        and edit.get("base_package") == job["base_package"]
        and edit.get("base_condition_sha256") == baseline["files"]["conditioning"]["tensor_sha256"],
        "prepared package does not match the current baseline and plan job",
    )
    for stem in ("latent", "noise"):
        require(prepared["files"][stem] == baseline["files"][stem], f"unexpected {stem} change")
    require(
        prepared["files"]["conditioning"]["tree_sha256"] == baseline["files"]["conditioning"]["tree_sha256"],
        "unexpected conditioning tree change",
    )
    return prepared
