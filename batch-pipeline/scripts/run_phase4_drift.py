#!/usr/bin/env python3
"""Equal-budget comparison: fresh latent vs reuse of a prior generated AV latent."""
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path


BASE = Path("/home/ma-user/work/lienyu/minimax-h3-embedding-validation")
sys.path.insert(0, str(BASE / "source/h3_bridge"))
from bridge import make_workflow, validate


def request_json(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def template(run_name):
    body = {"model": "MiniMax-H3", "content": [{"type": "text", "text": "never-submitted"}],
            "duration": 4, "ratio": "1:1", "resolution": "native",
            "local_options": {"preset": "custom", "ratio": "1:1", "width": 256, "height": 256,
                              "steps": 4, "seed": "202"}}
    workflow = make_workflow(validate(body), run_name)
    graph = workflow["prompt"]
    graph["7"] = {"class_type": "LienyuH3LoadConditionPackage", "inputs": {"package_name": "phase3-p01-s101"}}
    graph["11"]["inputs"]["conditioning"] = ["7", 0]
    graph["12"]["inputs"]["latent_image"] = ["7", 1]
    encoded = json.dumps(workflow)
    if "never-submitted" in encoded:
        raise RuntimeError("prompt leaked into replay workflow")
    return workflow


def execute(workflow, run_name):
    request_path = BASE / "evidence" / f"{run_name}.request.json"
    request_path.write_text(json.dumps(workflow, indent=2) + "\n")
    prompt_id = request_json("http://127.0.0.1:8192/prompt", workflow)["prompt_id"]
    deadline = time.time() + 1200
    while time.time() < deadline:
        history = request_json(f"http://127.0.0.1:8192/history/{prompt_id}").get(prompt_id)
        if history and history.get("status", {}).get("completed"):
            break
        if history and history.get("status", {}).get("status_str") == "error":
            raise RuntimeError(json.dumps(history["status"].get("messages")))
        time.sleep(2)
    else:
        raise TimeoutError(prompt_id)
    videos = []
    for output in history.get("outputs", {}).values():
        for items in output.values():
            if isinstance(items, list):
                videos.extend(x for x in items if isinstance(x, dict) and str(x.get("filename", "")).endswith(".mp4"))
    if len(videos) != 1:
        raise RuntimeError(videos)
    path = BASE / "output" / videos[0].get("subfolder", "") / videos[0]["filename"]
    return {"prompt_id": prompt_id, "video": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    source = template("phase4-drift-source")
    graph = source["prompt"]
    graph["8"] = {"class_type": "LienyuH3ReplayInitialNoise", "inputs": {"package_name": "phase3-p01-s101"}}
    graph["20"] = {"class_type": "LienyuH3SaveGeneratedLatent", "inputs": {"samples": ["12", 0], "package_name": "phase3-p01-s101"}}
    graph["13"]["inputs"]["samples"] = ["20", 0]
    graph["14"]["inputs"]["samples"] = ["20", 0]
    source_result = execute(source, "phase4-drift-source")

    baseline = template("phase4-drift-baseline")
    baseline["prompt"]["8"] = {"class_type": "LienyuH3RecordInitialNoise", "inputs": {"noise_seed": 202, "package_name": "phase4-drift-noise-s202"}}
    baseline_result = execute(baseline, "phase4-drift-baseline")

    reuse = template("phase4-drift-reuse")
    graph = reuse["prompt"]
    graph["8"] = {"class_type": "LienyuH3ReplayInitialNoise", "inputs": {"package_name": "phase4-drift-noise-s202"}}
    graph["21"] = {"class_type": "LienyuH3LoadGeneratedLatent", "inputs": {"package_name": "phase3-p01-s101"}}
    graph["12"]["inputs"]["latent_image"] = ["21", 0]
    reuse_result = execute(reuse, "phase4-drift-reuse")
    result = {"method": "reuse-prior-final-av-latent", "steps_each": 4, "same_noise": True,
              "generator_received_prompt": False, "source": source_result,
              "fresh_baseline": baseline_result, "reused_latent": reuse_result}
    (BASE / "evidence/phase4-drift.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
