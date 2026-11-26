#!/usr/bin/env python3
import hashlib
import json
import subprocess
import sys
from pathlib import Path


BASE = Path("/home/ma-user/work/lienyu/minimax-h3-embedding-validation")
RUNNER = BASE / "scripts/run_phase2.py"
PROMPTS = ("p01", "p03", "p07", "p09", "p12")


def decoded_hash(path, stream):
    command = (["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
               if stream == "video" else
               ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "s32le", "-acodec", "pcm_s32le", "-"])
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    digest = hashlib.sha256()
    while chunk := process.stdout.read(1024 * 1024):
        digest.update(chunk)
    if process.wait() != 0:
        raise RuntimeError(path)
    return digest.hexdigest()


def main():
    results = BASE / "evidence/phase4-compression.jsonl"
    done = set()
    if results.exists():
        done = {(x["prompt_id"], x["method"]) for x in map(json.loads, results.read_text().splitlines())}
    for prompt_id in PROMPTS:
        source = f"phase3-{prompt_id}-s101"
        subprocess.run([sys.executable, str(BASE / "scripts/make_compressed_packages.py"), source], check=True)
        baseline = next(json.loads(line) for line in (BASE / "evidence/phase3-results.jsonl").read_text().splitlines()
                        if json.loads(line)["prompt_id"] == prompt_id and json.loads(line)["seed"] == 101)
        for method in ("fp16", "int8-channel"):
            if (prompt_id, method) in done:
                continue
            package = f"{source}-{method}"
            command = [sys.executable, str(RUNNER), "replay", "--seed", "101", "--package", package,
                       "--run", f"phase4-compression-{prompt_id}-{method}", "--timeout", "1200"]
            generated = json.loads(subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip().splitlines()[-1])
            path = Path(generated["video"])
            manifest = json.loads((BASE / "packages" / package / "manifest.json").read_text())
            item = {"prompt_id": prompt_id, "method": method, "package": package, "generated": generated,
                    "decoded_video_sha256": decoded_hash(path, "video"), "decoded_audio_sha256": decoded_hash(path, "audio"),
                    "baseline_video_sha256": baseline["normal_decoded_video_sha256"],
                    "baseline_audio_sha256": baseline["normal_decoded_audio_sha256"],
                    "compression": manifest["compression_experiment"]}
            with results.open("a") as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(json.dumps({"completed": [prompt_id, method]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
