#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path


BRIDGE = Path("/home/ma-user/work/lienyu/minimax-h3-embedding-validation/source/h3_bridge")
sys.path.insert(0, str(BRIDGE))
from bridge import make_workflow, validate


def request_json(url, payload=None):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def normal_workflow(text, seed, package_name, run_name):
    body = {
        "model": "MiniMax-H3",
        "content": [{"type": "text", "text": text}],
        "duration": 4,
        "ratio": "1:1",
        "resolution": "native",
        "local_options": {
            "preset": "custom", "ratio": "1:1", "width": 256, "height": 256,
            "steps": 4, "seed": str(seed),
        },
    }
    workflow = make_workflow(validate(body), run_name)
    graph = workflow["prompt"]
    graph["8"] = {"class_type": "LienyuH3RecordInitialNoise", "inputs": {
        "noise_seed": seed, "package_name": package_name,
    }}
    graph["19"] = {"class_type": "LienyuH3ExportConditionPackage", "inputs": {
        "conditioning": ["7", 0], "latent": ["7", 1], "package_name": package_name,
    }}
    graph["11"]["inputs"]["conditioning"] = ["19", 0]
    graph["12"]["inputs"]["latent_image"] = ["19", 1]
    return workflow


def replay_workflow(seed, package_name, run_name):
    workflow = normal_workflow("placeholder-never-submitted", seed, package_name, run_name)
    graph = workflow["prompt"]
    graph["7"] = {"class_type": "LienyuH3LoadConditionPackage", "inputs": {
        "package_name": package_name,
    }}
    graph.pop("19")
    graph["8"] = {"class_type": "LienyuH3ReplayInitialNoise", "inputs": {
        "package_name": package_name,
    }}
    graph["11"]["inputs"]["conditioning"] = ["7", 0]
    graph["12"]["inputs"]["latent_image"] = ["7", 1]
    encoded = json.dumps(workflow, ensure_ascii=False)
    if "placeholder-never-submitted" in encoded:
        raise RuntimeError("replay workflow still contains text input")
    return workflow


def wait(base_url, prompt_id, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        history = request_json(f"{base_url}/history/{prompt_id}").get(prompt_id)
        if history:
            status = history.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(status.get("messages", []), ensure_ascii=False))
            if status.get("completed"):
                return history
        time.sleep(2)
    raise TimeoutError(prompt_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["normal", "replay"])
    parser.add_argument("--prompt")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8192")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.mode == "normal" and not args.prompt:
        parser.error("normal mode requires --prompt")
    workflow = (normal_workflow(args.prompt, args.seed, args.package, args.run)
                if args.mode == "normal" else replay_workflow(args.seed, args.package, args.run))
    request_path = Path("/home/ma-user/work/lienyu/minimax-h3-embedding-validation/evidence") / f"{args.run}.request.json"
    request_path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n")
    result = request_json(f"{args.url}/prompt", workflow)
    prompt_id = result["prompt_id"]
    history = wait(args.url, prompt_id, args.timeout)
    history_path = request_path.with_name(f"{args.run}.history.json")
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n")
    videos = []
    for output in history.get("outputs", {}).values():
        for items in output.values():
            if not isinstance(items, list):
                continue
            videos.extend(item for item in items if isinstance(item, dict) and str(item.get("filename", "")).endswith(".mp4"))
    if len(videos) != 1:
        raise RuntimeError(f"expected one video, got {videos}")
    item = videos[0]
    path = Path("/home/ma-user/work/lienyu/minimax-h3-embedding-validation/output") / item.get("subfolder", "") / item["filename"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {"mode": args.mode, "run": args.run, "prompt_id": prompt_id, "video": str(path), "bytes": path.stat().st_size, "sha256": digest}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
