"""Verify the delivered MP4s, graphs and source/noise pairing from their receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "h3-hypit-vehicle-edit-lab/results/runs"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def audio_hash(ffmpeg, video):
    return subprocess.check_output([
        ffmpeg, "-nostdin", "-v", "error", "-i", str(video), "-map", "0:a:0",
        "-acodec", "pcm_s16le", "-f", "hash", "-hash", "sha256", "-",
    ], text=True).strip()


def audit_outputs(root, ffmpeg):
    current = root / "runs"
    records = []
    for name, expected in (("paint", {"T0", "M1", "M2", "P3", "P4"}), ("body", {"T0", "M1", "M2"})):
        mapping = read(root / f"review-{name}/group-key.json")["cases"]
        assert len(mapping) == 9
        for row in mapping.values():
            base_name = row["case"]
            base = PREVIOUS / f"{base_name}--A0"
            original = read(base / "receipt.json")
            track = audio_hash(ffmpeg, base / "video.mp4")
            for variant in sorted(expected):
                folder = current / f"{base_name}--{variant}"
                receipt = read(folder / "receipt.json")
                assert receipt["replay_received_prompt"] is False
                assert receipt["payloads"]["noise"] == original["payloads"]["noise"]
                assert receipt["payloads"]["latent"] == original["payloads"]["latent"]
                assert receipt["settings"] == original["settings"]
                assert digest(folder / "video.mp4") == receipt["video_sha256"]
                assert audio_hash(ffmpeg, folder / "video.mp4") == track
                if variant in {"T0", "M1", "M2"}:
                    request_path = folder / "request.json"
                    assert digest(request_path) == receipt["request_sha256"]
                    graph = read(request_path)["prompt"]
                    assert graph["7"]["class_type"] == "LienyuH3LoadConditionPackage"
                    assert graph["8"]["class_type"] == "LienyuH3ReplayInitialNoise"
                    assert not any("prompt" in entry["inputs"] or "TextEncode" in entry["class_type"]
                                   for entry in graph.values())
                    assert graph["7"]["inputs"]["package_name"] == receipt["package"]
                    assert graph["95"]["inputs"]["mask_sha256"] == receipt["mask_sha256"]
                    if variant == "T0":
                        assert all(read(folder / "replay-verification.json").values())
                    else:
                        assert graph["95"]["inputs"]["trajectory_sha256"] == receipt["trajectory_sha256"]
                else:
                    assert receipt["source_video_sha256"] == original["video_sha256"]
                    donor = PREVIOUS / receipt["donor_job"] / "video.mp4"
                    assert digest(donor) == receipt["donor_video_sha256"]
                    assert receipt["new_generation_calls"] == 0
                records.append({"attribute": name, "job": folder.name, "video_sha256": receipt["video_sha256"],
                                "audio_sha256": track, "original_initial_noise": True,
                                "generator_received_prompt": False})
    target = root / "artifact-audit.json"
    target.write_text(json.dumps({"count": len(records), "validated": records}, indent=2) + "\n")
    print(f"Verified {len(records)} original-noise paired MP4s, audio, and replay receipts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    audit_outputs(ROOT / "results", args.ffmpeg)
