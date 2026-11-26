"""Execute H3 graphs on the isolated ModelArts worker and collect audit artifacts."""
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .conditioning import open_package
from .files import read_json, require, sha256, write_json, write_new


class ModelArts:
    def __init__(self, lab):
        self.lab = lab
        self.root = Path(os.environ["H3_EXPERIMENT_BASE"]).resolve()
        self.packages = self.root / "packages"
        self.bridge_script = self.root / "scripts/run_h3.py"
        self.python = Path(os.environ["H3_PYTHON"])
        self.results = lab / "results/runs"
        self.inputs = lab / "outputs/modelarts-inputs"
        self.inputs.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(self.root / "scripts"))
        self.runner = importlib.import_module("run_h3")
        self.assets = importlib.import_module("conditioning")

    def graph(self, name, package, seed, settings, prompt=None, media=(), encode_only=False):
        if prompt is None:
            return self.runner.graph(
                "never-submit-this-placeholder", seed, package, name, settings["duration"],
                settings["width"], settings["height"], "replay", steps=settings["steps"],
            )
        from base64 import b64encode

        content = [{"type": "text", "text": prompt}]
        for role, path in media:
            kind, mime = ("video_url", "video/mp4") if path.suffix == ".mp4" else ("image_url", "image/png")
            content.append({"type": kind, "role": role, kind: {
                "url": f"data:{mime};base64,{b64encode(path.read_bytes()).decode('ascii')}",
            }})
        spec = self.runner.validate({
            "model": "MiniMax-H3", "content": content, "duration": settings["duration"],
            "ratio": "16:9", "resolution": "native", "local_options": {
                "preset": "custom", "ratio": "16:9", "width": settings["width"],
                "height": settings["height"], "steps": settings["steps"], "seed": str(seed),
            },
        })
        if media:
            stage = self.inputs / "staged"
            staged = self.assets.stage_assets(spec["assets"], stage)
            spec["assets"] = self.assets.prepare_assets(staged, stage, self.root / "instances/worker/input")
        workflow = self.runner.make_workflow(spec, name)
        nodes = workflow["prompt"]
        nodes["90"] = {"class_type": "LienyuH3ExportConditionPackage", "inputs": {
            "conditioning": ["7", 0], "latent": ["7", 1], "package_name": package,
        }}
        nodes["8"] = {"class_type": "LienyuH3RecordInitialNoise", "inputs": {
            "noise_seed": seed, "package_name": package,
        }}
        nodes["11"]["inputs"]["conditioning"] = ["90", 0]
        nodes["12"]["inputs"]["latent_image"] = ["90", 1]
        if encode_only:
            keep = {"1", "2", "3", "4", "5", "6", "7", "90"}
            workflow["prompt"] = {key: value for key, value in nodes.items() if key in keep or 19 <= int(key) <= 24}
        return workflow

    def execute(self, directory, job, package, *, prompt=None, media=(), encode_only=False, extra=None):
        name = job["id"]
        receipt = directory / "receipt.json"
        if receipt.exists():
            result = read_json(receipt)
            if "video" in result:
                require(sha256(directory / result["video"]) == result["video_sha256"], f"video drift for {name}")
            return result
        directory.mkdir(parents=True, exist_ok=True)
        graph = self.graph(name, package, job["seed"], job["settings"], prompt, media, encode_only)
        request_path = directory / "request.json"
        pid_path = directory / "submission.json"
        history_path = directory / "history.json"
        if request_path.exists():
            require(read_json(request_path) == graph, f"workflow changed for {name}")
            require(pid_path.exists(), f"previous submission outcome unknown for {name}")
        else:
            write_json(request_path, graph)
        if pid_path.exists():
            submission = read_json(pid_path)
        else:
            submission = {"timestamp": time.time(), "prompt_id": self.runner.req(
                "http://127.0.0.1:8192/prompt", graph,
            )["prompt_id"]}
            write_json(pid_path, submission)
        history = read_json(history_path) if history_path.exists() else self.runner.wait(submission["prompt_id"], 240)
        if not history_path.exists():
            write_json(history_path, history)
        _, manifest = open_package(self.packages, package, ("conditioning", "latent") if encode_only else
                                   ("conditioning", "latent", "noise"))
        result = {
            "schema_version": 1, "status": "completed", "job_id": name, "seed": job["seed"],
            "variant": job.get("variant"), "attribute_id": job.get("attribute_id"),
            "settings": job["settings"], "package": package, "base_package": job.get("base_package"),
            "prompt_id": submission["prompt_id"], "elapsed_s": time.time() - submission["timestamp"],
            "payloads": manifest["files"], "replay_received_prompt": False if prompt is None else None,
            "media": [{"role": role, "file": str(path), "sha256": sha256(path)} for role, path in media],
            **(extra or {}),
        }
        if not encode_only:
            files = [item for value in history["outputs"]["16"].values() if isinstance(value, list)
                     for item in value if isinstance(item, dict) and str(item.get("filename", "")).endswith(".mp4")]
            require(len(files) == 1, f"expected exactly one MP4 for {name}")
            output_root = self.root / "outputs"
            video = (output_root / files[0].get("subfolder", "") / files[0]["filename"]).resolve(strict=True)
            require(video.is_relative_to(output_root), "worker returned a video outside the isolated outputs")
            digest = sha256(video)
            if not (directory / "video.mp4").exists():
                write_new(directory / "video.mp4", lambda staged: shutil.copyfile(video, staged))
            require(sha256(directory / "video.mp4") == digest, f"copied video differs for {name}")
            result.update(video="video.mp4", video_sha256=digest)
        write_json(receipt, result)
        print(f"{name}: completed", flush=True)
        return result

    def capture(self, name, source, timestamp=None):
        destination = self.inputs / f"{name}.png"
        if destination.exists():
            return destination
        seek, filters = ["-ss", str(timestamp)], []
        if timestamp is None:
            probe = subprocess.run([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
                "-of", "default=nokey=1:noprint_wrappers=1", str(source),
            ], check=True, capture_output=True, text=True)
            frames = int(probe.stdout.strip())
            require(frames > 0, f"no video frames found in {source}")
            seek, filters = [], ["-vf", f"select=eq(n\\,{frames - 1})"]
        write_new(destination, lambda target: subprocess.run([
            "ffmpeg", "-v", "error", "-nostdin", *seek, "-i", str(source), *filters,
            "-frames:v", "1", "-threads", "1", str(target),
        ], check=True))
        require(destination.stat().st_size > 0, f"failed to capture frame from {source}")
        return destination

    def copy_replay_evidence(self, name):
        directory = self.results / name
        for suffix in ("request.json", "history.json"):
            source = self.root / "evidence" / f"{name}.{suffix}"
            target = directory / suffix
            if source.is_file() and not target.exists():
                write_new(target, lambda staged: shutil.copyfile(source, staged))
