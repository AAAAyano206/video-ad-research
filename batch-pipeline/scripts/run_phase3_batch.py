#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


BASE = Path("/home/ma-user/work/lienyu/minimax-h3-embedding-validation")
RUNNER = BASE / "scripts/run_phase2.py"
SEEDS = (101, 202, 303)


def decoded_hash(path, stream):
    command = (["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
               if stream == "video" else
               ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "s32le", "-acodec", "pcm_s32le", "-"])
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    digest = hashlib.sha256()
    while chunk := process.stdout.read(1024 * 1024):
        digest.update(chunk)
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")
    return digest.hexdigest()


def run(mode, prompt, seed, package, name):
    command = [sys.executable, str(RUNNER), mode, "--seed", str(seed), "--package", package,
               "--run", name, "--timeout", "1200"]
    if mode == "normal":
        command.extend(["--prompt", prompt])
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=str(BASE / "inputs/phase3_prompts.json"))
    args = parser.parse_args()
    prompts = json.loads(Path(args.prompts).read_text())
    result_path = BASE / "evidence/phase3-results.jsonl"
    completed = {}
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            item = json.loads(line)
            completed[(item["prompt_id"], item["seed"])] = item
    for entry in prompts:
        for seed in SEEDS:
            key = (entry["id"], seed)
            if key in completed and completed[key].get("transport_exact"):
                print(json.dumps({"skipped": key}, ensure_ascii=False), flush=True)
                continue
            package = f"phase3-{entry['id']}-s{seed}"
            normal_name = f"phase3-{entry['id']}-s{seed}-normal"
            replay_name = f"phase3-{entry['id']}-s{seed}-replay"
            normal = run("normal", entry["prompt"], seed, package, normal_name)
            replay = run("replay", entry["prompt"], seed, package, replay_name)
            normal_path, replay_path = Path(normal["video"]), Path(replay["video"])
            normal_video = decoded_hash(normal_path, "video")
            replay_video = decoded_hash(replay_path, "video")
            normal_audio = decoded_hash(normal_path, "audio")
            replay_audio = decoded_hash(replay_path, "audio")
            manifest = json.loads((BASE / "packages" / package / "manifest.json").read_text())
            item = {
                "prompt_id": entry["id"], "prompt": entry["prompt"], "seed": seed,
                "package": package, "normal": normal, "replay": replay,
                "normal_decoded_video_sha256": normal_video,
                "replay_decoded_video_sha256": replay_video,
                "normal_decoded_audio_sha256": normal_audio,
                "replay_decoded_audio_sha256": replay_audio,
                "conditioning_sha256": manifest["files"]["conditioning"]["tensor_sha256"],
                "latent_sha256": manifest["files"]["latent"]["tensor_sha256"],
                "noise_sha256": manifest["files"]["noise"]["tensor_sha256"],
                "transport_exact": normal_video == replay_video and normal_audio == replay_audio,
            }
            with result_path.open("a") as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(json.dumps({"completed": key, "transport_exact": item["transport_exact"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
