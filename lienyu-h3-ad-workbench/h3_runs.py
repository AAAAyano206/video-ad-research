"""Capture a new mother template and run separately logged prompt-free H3 replays."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
from vehicle_edit_lab.conditioning import open_package, prepare_package, verify_replay_package
from vehicle_edit_lab.files import read_json, require, sha256, write_json
from vehicle_edit_lab.modelarts import ModelArts


EXP = read_json(ROOT / "experiment.json")
DIRECTION = ROOT / "assets" / "color.safetensors"


def settings(duration):
    return {"duration": duration, "width": 384, "height": 256, "steps": 4, "reference_model": False}


def cases():
    for index, (name, prompt) in enumerate(EXP["holdout"].items()):
        for i, seed in enumerate(EXP["holdout_seeds"]):
            yield {"id": f"{name}-{seed + 3 * index}", "seed": seed + 3 * index, "duration": 6,
                   "affected": True, "previous_frame": False, "prompt": prompt, "family": "holdout"}
    for item in EXP["story"]:
        yield {**item, "family": "story"}


def job(item, role, variant=None, package=None):
    name = item["id"] + "--" + role
    return {"id": name, "seed": item["seed"], "settings": settings(item["duration"]),
            "base_package": item["id"] + "--base", "package": package or name,
            "variant": variant, "attribute_id": EXP["attribute"]}


def capture(host):
    previous = None
    for item in cases():
        spec = job(item, "base")
        media = ()
        if item["previous_frame"]:
            require(previous is not None, "missing preceding story shot")
            frame = host.capture(item["id"] + "--source-first", previous)
            media = (("first_frame", frame),)
        result = host.execute(host.results / spec["id"], spec, spec["package"], prompt=item["prompt"], media=media)
        if item["family"] == "story":
            previous = host.results / spec["id"] / result["video"]


def media_hash(path, stream):
    options = ["-pix_fmt", "rgb24", "-vcodec", "rawvideo"] if stream == "v" else ["-acodec", "pcm_s16le"]
    return subprocess.check_output(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", f"0:{stream}:0",
                                    *options, "-f", "hash", "-hash", "sha256", "-"], text=True).strip()


def execute_replay(host, spec, label):
    _, manifest = open_package(host.packages, spec["package"])
    _, baseline = open_package(host.packages, spec["base_package"])
    require(all(manifest["files"][field] == baseline["files"][field] for field in ("noise", "latent")),
            "physical initial noise or input latent changed")
    output = host.results / spec["id"]
    request = host.graph(spec["id"], spec["package"], spec["seed"], spec["settings"])
    nodes = request["prompt"]
    require(nodes["7"]["class_type"] == "LienyuH3LoadConditionPackage" and
            nodes["8"]["class_type"] == "LienyuH3ReplayInitialNoise", "wrong condition or noise loader")
    require(all("prompt" not in node["inputs"] and "TextEncode" not in node["class_type"] for node in nodes.values()),
            "text input unexpectedly reached the replay workflow")
    receipt = host.execute(output, spec, spec["package"], extra={"experiment_arm": label,
        "conditioning_edit": manifest.get("vehicle_edit_lab")})
    require(read_json(output / "request.json") == request, "stored workflow differs from audited replay")
    require(receipt["replay_received_prompt"] is False, "unexpected replay receipt")
    return receipt


def make_variant(host, item, role, variant, matching=None):
    spec = job(item, role, variant)
    if matching:
        spec.update(vector_source_job=matching, random_direction_seed=900000 + item["seed"])
    if not (host.packages / spec["package"] / "manifest.json").exists():
        prepare_package(host.packages, spec, DIRECTION, EXP["direction_alpha"], "t000000")
    verify_replay_package(host.packages, spec)
    return spec


def replay(host):
    for item in cases():
        base = item["id"] + "--base"
        if item["family"] == "holdout":
            a0 = job(item, "replay", "A0", package=base)
            execute_replay(host, a0, "original")
            base_video = host.results / base / "video.mp4"
            replay_video = host.results / a0["id"] / "video.mp4"
            exact = {stream: media_hash(base_video, stream) == media_hash(replay_video, stream) for stream in ("v", "a")}
            require(all(exact.values()), f"original replay mismatch for {item['id']}: {exact}")
            receipt = replay_video.with_name("decoded-hashes.json")
            if not receipt.exists():
                write_json(receipt, exact)
            target = make_variant(host, item, "target", "A1")
            random = make_variant(host, item, "random", "A2", matching=target["id"])
            execute_replay(host, target, "target-vector")
            execute_replay(host, random, "equal-norm-random")
        elif item["affected"]:
            target = make_variant(host, item, "donor", "A1")
            execute_replay(host, target, "selective-donor")
            full = job(item, "full", package=target["package"])
            execute_replay(host, full, "full-regenerated")
        else:
            full = job(item, "full", package=base)
            execute_replay(host, full, "full-regenerated")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("capture", "replay"))
    args = parser.parse_args()
    os.environ["H3_EXPERIMENT_BASE"] = str(ROOT)
    host = ModelArts(ROOT)
    (ROOT / "packages").mkdir(exist_ok=True)
    print(args.stage, time.strftime("%Y-%m-%dT%H:%M:%S%z"), "direction", sha256(DIRECTION), flush=True)
    (capture if args.stage == "capture" else replay)(host)


if __name__ == "__main__":
    main()
