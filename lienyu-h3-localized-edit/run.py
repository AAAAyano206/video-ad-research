"""Run paired H3 replays with a tracked vehicle region and a source trajectory."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from urllib.request import Request, urlopen

import cv2
import numpy as np
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parent
HISTORY = ROOT.parent / "h3-hypit-vehicle-edit-lab"
PORT = 8192


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def video_frames(path, width, height):
    pixels = subprocess.check_output([
        "ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(path),
        "-map", "0:v:0", "-threads", "1", "-pix_fmt", "bgr24", "-f", "rawvideo", "-",
    ])
    return np.frombuffer(pixels, np.uint8).reshape(-1, height, width, 3)


def vehicle_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, light = cv2.split(hsv)
    red, green, blue = frame[:, :, 2].astype(np.int16), frame[:, :, 1], frame[:, :, 0]
    red_hue = (hue <= 14) | (hue >= 163)
    seeds = (red_hue & (saturation >= 60) & (light >= 42) &
             (red - green >= 15) & (red - blue >= 10)).astype(np.uint8)
    joined = cv2.dilate(seeds, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 13)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
    if count == 1:
        return np.zeros_like(seeds)
    component = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    vehicle = (labels == component).astype(np.uint8)
    return (vehicle * seeds).astype(np.uint8)


def prepare_mask(case, source, frames):
    path = ROOT / "masks" / (case + ".safetensors")
    paint = np.stack([vehicle_mask(frame) for frame in frames])
    if min(np.count_nonzero(frame) for frame in paint) < 12:
        raise ValueError(f"review low-coverage car masks for {case}")
    silhouette = []
    for mask in paint:
        hull = cv2.convexHull(cv2.findNonZero(mask))
        region = cv2.fillConvexPoly(np.zeros(mask.shape, np.uint8), hull, 255)
        silhouette.append(region)
    if not path.exists():
        save_file({"pixel_mask": torch.from_numpy(np.stack(silhouette)),
                   "paint_mask": torch.from_numpy(paint)}, str(path))
        write(path.with_suffix(".json"), {"source_video_sha256": digest(source), "mask_sha256": digest(path),
                                         "paint_pixels_min": int(paint.sum((1, 2)).min()),
                                         "paint_pixels_mean": float(paint.sum((1, 2)).mean())})
    else:
        meta = read(path.with_suffix(".json"))
        assert meta["source_video_sha256"] == digest(source) and meta["mask_sha256"] == digest(path)
    return path, paint


def api(endpoint, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(f"http://127.0.0.1:{PORT}/{endpoint}", data, {"Content-Type": "application/json"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def build(source_case, base_case, variant, case, attribute, mask, trajectory):
    request = deepcopy(read(source_case / "request.json"))
    nodes = request["prompt"]
    assert nodes["7"]["class_type"] == "LienyuH3LoadConditionPackage"
    assert nodes["8"]["class_type"] == "LienyuH3ReplayInitialNoise"
    assert not any("prompt" in node["inputs"] or "TextEncode" in node["class_type"] for node in nodes.values())
    assert "never-submit-this-placeholder" not in json.dumps(request)
    src = read(source_case / "receipt.json")
    a0 = read(base_case / "receipt.json")
    assert src["replay_received_prompt"] is False
    assert src["settings"] == a0["settings"] and src["payloads"]["noise"] == a0["payloads"]["noise"]
    assert src["payloads"]["latent"] == a0["payloads"]["latent"]
    assert nodes["7"]["inputs"]["package_name"] == src["package"]
    for stem in ("conditioning", "latent", "noise"):
        for kind in ("tensor", "tree"):
            entry = src["payloads"][stem]
            actual = Path(os.environ["LIENYU_H3_PACKAGE_ROOT"]) / src["package"] / entry[kind + "_file"]
            assert digest(actual) == entry[kind + "_sha256"]
    nodes["95"] = {"class_type": "LienyuLocalizedH3Sampler", "inputs": {
        "sampler": ["9", 0], "latent": ["7", 1], "case_id": case,
        "mode": "record" if variant == "T0" else "lock", "attribute": attribute,
        "mask_sha256": digest(mask), "mask_directory": mask.parent.name, "trajectory_sha256": "" if variant == "T0" else digest(trajectory),
    }}
    nodes["12"]["inputs"]["sampler"] = ["95", 0]
    if "filename_prefix" not in nodes["16"]["inputs"]:
        raise ValueError("snapshot output node changed")
    nodes["16"]["inputs"]["filename_prefix"] = f"{case}--{variant}"
    # Historical requests contain only replay graphs and generic client metadata.
    return request, src


def run_replay(source_case, base_case, case, variant, attribute, mask):
    destination = ROOT / "results/runs" / f"{case}--{variant}"
    destination.mkdir(parents=True, exist_ok=True)
    video, receipt = destination / "video.mp4", destination / "receipt.json"
    request_file = destination / "request.json"
    trajectory = ROOT / "trajectories" / (case + ".safetensors")
    name = destination.name
    if receipt.exists():
        old = read(receipt)
        assert old["video_sha256"] == digest(video)
        if variant == "T0":
            assert old["trajectory_sha256"] == digest(trajectory)
        return video
    request, source = build(source_case, base_case, variant, case, attribute, mask, trajectory)
    write(request_file, request)
    submission_file = destination / "submission.json"
    if submission_file.exists():
        submission = read(submission_file)
        assert submission["request_sha256"] == digest(request_file)
    else:
        start = time.time()
        pid = api("prompt", request)["prompt_id"]
        submission = {"submitted": start, "prompt_id": pid, "request_sha256": digest(request_file)}
        write(submission_file, submission)
    pid = submission["prompt_id"]
    history_path = destination / "history.json"
    if not history_path.exists():
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            history = api("history/" + pid).get(pid)
            if history:
                write(history_path, history)
                break
            time.sleep(0.6)
    history = read(history_path)
    if not history["status"]["completed"] or history["status"]["status_str"] != "success":
        # Comfy's legacy IMAGE serializers can embed info in the status record.
        raise RuntimeError(f"see worker history for {name}: {history['status']['status_str']}")
    output_entries = [item for entries in history["outputs"]["16"].values()
                      for item in entries if isinstance(item, dict) and item.get("filename", "").endswith(".mp4")]
    assert len(output_entries) == 1
    item = output_entries[0]
    output_root = ROOT / "worker-output"
    raw = (output_root / item.get("subfolder", "") / item["filename"]).resolve(strict=True)
    assert raw.is_relative_to(output_root)
    shutil.copyfile(raw, video)
    published = {**source, "job_id": name, "variant": variant, "source_job": source_case.name,
                 "prompt_id": pid, "video": video.name, "video_sha256": digest(video),
                 "request_sha256": digest(request_file), "mask_sha256": digest(mask),
                 "trajectory_sha256": digest(trajectory), "elapsed_s": time.time() - submission["submitted"],
                 "generation_calls": 1, "replay_received_prompt": False}
    write(receipt, published)
    print(name, "completed", flush=True)
    return video


def decoded_hash(path, stream):
    options = ["-pix_fmt", "rgb24", "-vcodec", "rawvideo"] if stream == "v" else ["-acodec", "pcm_s16le"]
    return subprocess.check_output([
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", f"0:{stream}:0",
        *options, "-f", "hash", "-hash", "sha256", "-",
    ], text=True).strip()


def clip_groups(attribute):
    plan = read(HISTORY / "outputs/modelarts-plans" / f"holdout-{attribute}.json")
    groups = {}
    for job in plan["jobs"]:
        group = groups.setdefault(job["baseline_job"].removesuffix("--A0"), {})
        group[job["variant"]] = HISTORY / "results/runs" / job["id"]
    return groups


def verify_source_clip(folder):
    receipt = read(folder / "receipt.json")
    assert digest(folder / "video.mp4") == receipt["video_sha256"]
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("attribute", choices=("paint", "body"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--segmented", action="store_true", default=os.environ.get("H3_SEGMENTED") == "1")
    args = parser.parse_args()
    for folder in ("masks", "trajectories", "results/runs"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    groups = list(clip_groups(args.attribute).items())
    for case, originals in groups[:args.limit]:
        print(case, flush=True)
        source = originals["A0"] / "video.mp4"
        settings = verify_source_clip(originals["A0"])["settings"]
        for variant in ("A1", "A2"):
            verify_source_clip(originals[variant])
        if args.segmented:
            mask = ROOT / "masks-segmented" / f"{case}.safetensors"
            meta = read(mask.with_suffix(".json"))
            assert meta["source_video_sha256"] == digest(source) and meta["mask_sha256"] == digest(mask)
        else:
            frames = video_frames(source, settings["width"], settings["height"])
            mask, _ = prepare_mask(case, source, frames)
        captured = run_replay(originals["A0"], originals["A0"], case, "T0", args.attribute, mask)
        comparison = {stream: decoded_hash(source, stream) == decoded_hash(captured, stream) for stream in ("v", "a")}
        write(captured.parent / "replay-verification.json", comparison)
        if not all(comparison.values()):
            raise ValueError(f"original video/audio replay did not match for {case}: {comparison}")
        variants = ("M1", "M2") if args.segmented else ("L1", "L2")
        for new, old in zip(variants, ("A1", "A2")):
            clip = run_replay(originals[old], originals["A0"], case, new, args.attribute, mask)
            audio_equal = decoded_hash(source, "a") == decoded_hash(clip, "a")
            write(clip.parent / "replay-verification.json", {"audio_exact": audio_equal})
            if not audio_equal:
                raise ValueError(f"protected audio changed for {case} {new}")


if __name__ == "__main__":
    main()
