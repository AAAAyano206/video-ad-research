import hashlib
import json
import os
import re
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

import comfy.sample


ROOT = Path(os.environ.get(
    "LIENYU_H3_PACKAGE_ROOT",
    "/home/ma-user/work/lienyu/minimax-h3-embedding-validation/packages",
)).resolve()
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _package(name):
    if not NAME.fullmatch(name):
        raise ValueError("package_name must contain only letters, digits, dot, underscore, and dash")
    path = (ROOT / name).resolve()
    if path.parent != ROOT:
        raise ValueError("package path escapes experiment root")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_nested(value):
    return value.__class__.__name__ == "NestedTensor" and hasattr(value, "unbind")


def _encode(value, tensors, prefix="t"):
    if _is_nested(value):
        return {"kind": "nested", "items": [_encode(item, tensors, prefix) for item in value.unbind()]}
    if isinstance(value, torch.Tensor):
        key = f"{prefix}{len(tensors):06d}"
        tensor = value.detach().to("cpu").contiguous()
        tensors[key] = tensor
        return {"kind": "tensor", "key": key}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("condition package only supports string dictionary keys")
        return {"kind": "dict", "items": {key: _encode(item, tensors, prefix) for key, item in value.items()}}
    if isinstance(value, list):
        return {"kind": "list", "items": [_encode(item, tensors, prefix) for item in value]}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_encode(item, tensors, prefix) for item in value]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"kind": "scalar", "value": value}
    raise TypeError(f"unsupported condition package value: {type(value).__module__}.{type(value).__qualname__}")


def _decode(node, tensors):
    kind = node["kind"]
    if kind == "tensor":
        return tensors[node["key"]]
    if kind == "nested":
        from comfy.nested_tensor import NestedTensor
        return NestedTensor(tuple(_decode(item, tensors) for item in node["items"]))
    if kind == "dict":
        return {key: _decode(item, tensors) for key, item in node["items"].items()}
    if kind == "list":
        return [_decode(item, tensors) for item in node["items"]]
    if kind == "tuple":
        return tuple(_decode(item, tensors) for item in node["items"])
    if kind == "scalar":
        return node["value"]
    raise ValueError(f"unknown package node kind: {kind}")


def _tensor_manifest(tensors):
    return {
        key: {"dtype": str(value.dtype), "shape": list(value.shape), "bytes": value.numel() * value.element_size()}
        for key, value in tensors.items()
    }


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_manifest(package):
    path = package / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {
        "format": "lienyu.h3.condition-package.v1",
        "created_at_unix": time.time(),
        "contains_prompt": False,
        "files": {},
    }


def _save_payload(package, stem, value):
    tensors = {}
    tree = _encode(value, tensors)
    tensor_path = package / f"{stem}.safetensors"
    tree_path = package / f"{stem}.json"
    save_file(tensors, tensor_path)
    _write_json(tree_path, tree)
    return {
        "tensor_file": tensor_path.name,
        "tensor_sha256": _sha256(tensor_path),
        "tree_file": tree_path.name,
        "tree_sha256": _sha256(tree_path),
        "tensors": _tensor_manifest(tensors),
    }


def _load_payload(package, stem):
    manifest = _load_manifest(package)
    record = manifest["files"][stem]
    tensor_path = package / record["tensor_file"]
    tree_path = package / record["tree_file"]
    if _sha256(tensor_path) != record["tensor_sha256"] or _sha256(tree_path) != record["tree_sha256"]:
        raise ValueError(f"{stem} checksum mismatch")
    return _decode(json.loads(tree_path.read_text()), load_file(tensor_path, device="cpu"))


class ExportConditionPackage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "latent": ("LATENT",),
            "package_name": ("STRING", {"default": "trial-001"}),
        }}

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("conditioning", "latent")
    FUNCTION = "export"
    CATEGORY = "lienyu/h3"
    OUTPUT_NODE = True

    def export(self, conditioning, latent, package_name):
        package = _package(package_name)
        manifest = _load_manifest(package)
        manifest["files"]["conditioning"] = _save_payload(package, "conditioning", conditioning)
        manifest["files"]["latent"] = _save_payload(package, "latent", latent)
        manifest["exported_at_unix"] = time.time()
        _write_json(package / "manifest.json", manifest)
        return {"ui": {"text": [f"condition package: {package}"]}, "result": (conditioning, latent)}


class LoadConditionPackage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"package_name": ("STRING", {"default": "trial-001"})}}

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("conditioning", "latent")
    FUNCTION = "load"
    CATEGORY = "lienyu/h3"

    def load(self, package_name):
        package = _package(package_name)
        return (_load_payload(package, "conditioning"), _load_payload(package, "latent"))


class RecordedNoise:
    def __init__(self, seed, package_name):
        self.seed = seed
        self.package_name = package_name

    def generate_noise(self, input_latent):
        latent = input_latent["samples"]
        batch = input_latent.get("batch_index")
        noise = comfy.sample.prepare_noise(latent, self.seed, batch)
        package = _package(self.package_name)
        manifest = _load_manifest(package)
        manifest["noise_seed"] = str(self.seed)
        manifest["files"]["noise"] = _save_payload(package, "noise", noise)
        manifest["noise_recorded_at_unix"] = time.time()
        _write_json(package / "manifest.json", manifest)
        return noise


class RecordInitialNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "package_name": ("STRING", {"default": "trial-001"}),
        }}

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "make"
    CATEGORY = "lienyu/h3"

    def make(self, noise_seed, package_name):
        return (RecordedNoise(noise_seed, package_name),)


class ReplayedNoise:
    def __init__(self, package_name):
        self.package_name = package_name
        self.seed = int(_load_manifest(_package(package_name)).get("noise_seed", "0"))

    def generate_noise(self, input_latent):
        noise = _load_payload(_package(self.package_name), "noise")
        expected = input_latent["samples"]
        if _is_nested(noise) != _is_nested(expected):
            raise ValueError("recorded noise and target latent container types differ")
        return noise


class ReplayInitialNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"package_name": ("STRING", {"default": "trial-001"})}}

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "load"
    CATEGORY = "lienyu/h3"

    def load(self, package_name):
        return (ReplayedNoise(package_name),)


class SaveGeneratedLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",),
            "package_name": ("STRING", {"default": "trial-001"}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "save"
    CATEGORY = "lienyu/h3"
    OUTPUT_NODE = True

    def save(self, samples, package_name):
        package = _package(package_name)
        manifest = _load_manifest(package)
        manifest["files"]["generated_latent"] = _save_payload(package, "generated_latent", samples)
        manifest["generated_latent_saved_at_unix"] = time.time()
        _write_json(package / "manifest.json", manifest)
        return {"ui": {"text": [f"generated latent: {package}"]}, "result": (samples,)}


class LoadGeneratedLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"package_name": ("STRING", {"default": "trial-001"})}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "lienyu/h3"

    def load(self, package_name):
        return (_load_payload(_package(package_name), "generated_latent"),)


class SkipH3Blocks:
    """Causal ablation hook: replace selected H3 DiT blocks with identity."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "start_block": ("INT", {"default": 0, "min": 0, "max": 50}),
            "end_block": ("INT", {"default": 0, "min": 0, "max": 50}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "lienyu/h3"

    @staticmethod
    def _identity(args, extra_options):
        return {"img": args["img"]}

    def patch(self, model, start_block, end_block):
        if not 0 <= start_block <= end_block <= 50:
            raise ValueError("H3 block interval must satisfy 0 <= start <= end <= 50")
        patched = model.clone()
        for index in range(start_block, end_block):
            patched.set_model_patch_replace(self._identity, "dit", "double_block", index)
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "LienyuH3ExportConditionPackage": ExportConditionPackage,
    "LienyuH3LoadConditionPackage": LoadConditionPackage,
    "LienyuH3RecordInitialNoise": RecordInitialNoise,
    "LienyuH3ReplayInitialNoise": ReplayInitialNoise,
    "LienyuH3SaveGeneratedLatent": SaveGeneratedLatent,
    "LienyuH3LoadGeneratedLatent": LoadGeneratedLatent,
    "LienyuH3SkipBlocks": SkipH3Blocks,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LienyuH3ExportConditionPackage": "Lienyu H3 Export Condition Package",
    "LienyuH3LoadConditionPackage": "Lienyu H3 Load Condition Package",
    "LienyuH3RecordInitialNoise": "Lienyu H3 Record Initial Noise",
    "LienyuH3ReplayInitialNoise": "Lienyu H3 Replay Initial Noise",
    "LienyuH3SaveGeneratedLatent": "Lienyu H3 Save Generated Latent",
    "LienyuH3LoadGeneratedLatent": "Lienyu H3 Load Generated Latent",
    "LienyuH3SkipBlocks": "Lienyu H3 Skip Blocks",
}
