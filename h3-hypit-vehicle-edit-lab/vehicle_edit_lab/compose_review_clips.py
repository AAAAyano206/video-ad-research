"""Compose the H3-only seam comparison and record its source hashes and length."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from .files import read_json, require, sha256, write_json, write_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    runs = Path(__file__).resolve().parents[1] / "results/runs"
    comparison = {}
    for group in ("A0", "A1", "A3"):
        clips, hashes = [], []
        for take in (1, 2):
            directory = runs / f"continuity--vehicle.paint_color--story-{take}--coastal-chain--{group}"
            receipt = read_json(directory / "receipt.json")
            clip = directory / receipt["video"]
            require(sha256(clip) == receipt["video_sha256"], f"receipt does not match {clip}")
            clips.append(clip)
            hashes.append(receipt["video_sha256"])
        output = args.destination / f"{group}-two-shots.mp4"
        if not output.exists():
            with tempfile.TemporaryDirectory(prefix="hypit-concat-") as temp:
                # Use fixed safe concat names. The temporary links avoid quoting source paths in the playlist.
                playlist = Path(temp) / "pair.ffconcat"
                for index, clip in enumerate(clips):
                    (playlist.parent / f"shot{index}.mp4").symlink_to(clip.resolve())
                playlist.write_text("file shot0.mp4\nfile shot1.mp4\n", encoding="utf-8")
                write_new(output, lambda target: subprocess.run([
                    "ffmpeg", "-v", "error", "-nostdin", "-f", "concat", "-i", str(playlist),
                    "-c", "copy", str(target),
                ], check=True))
        result = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,nb_frames",
            "-of", "json", str(output),
        ], text=True)
        probe = json.loads(result)
        require({field["codec_type"] for field in probe["streams"]} == {"audio", "video"},
                f"expected audio and video in {output.name}")
        comparison[group] = {"method": "FFmpeg stream copy", "inputs_sha256": hashes,
                             "output": output.name, "sha256": sha256(output), "probe": probe}
    manifest = args.destination / "manifest.json"
    if manifest.exists():
        require(read_json(manifest) == comparison, "composition manifest changed")
    else:
        write_json(manifest, comparison)
    print(f"Saved three seam comparisons to {args.destination}")


if __name__ == "__main__":
    main()
