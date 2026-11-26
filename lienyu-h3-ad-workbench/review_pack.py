"""Freeze a blinded panel of paired video candidates and aligned frames."""
import argparse
from pathlib import Path
import random
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
from vehicle_edit_lab.files import read_json, require, sha256, write_json
from h3_runs import cases


ARMS = ("target", "random", "legacy", "parts", "parts-random")
NOTES = ("Red paint to blue. Inspect complete paint and preserve body shape, scene, camera, lights, glass, "
         "wheels and calipers. Count missing source evidence separately, do not force a pass.")


def stripped(source, target):
    if target.is_file():
        return
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(source), "-map", "0:v:0",
                    "-c", "copy", "-map_metadata", "-1", "-map_chapters", "-1", "-fflags", "+bitexact",
                    str(target)], check=True)


def sample(path):
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-i", str(path), "-vf",
        "select=eq(mod(n\\,12)\\,6),scale=192:128", "-frames:v", "12", "-threads", "1",
        "-vsync", "0", "-pix_fmt", "bgr24", "-f", "rawvideo", "-",
    ])
    values = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 128, 192, 3)
    require(len(values) == 12, f"expected 12 aligned frames from {path}")
    return values


def board(name, clips, output):
    width, height, left, top = 192, 128, 84, 30
    row_height = height + 18
    picture = np.full((top + row_height * 2 * len(clips), left + width * 6, 3), 247, np.uint8)
    cv2.putText(picture, name, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .56, (26, 26, 26), 1, cv2.LINE_AA)
    for index, (label, clip) in enumerate(clips.items()):
        for j, frame in enumerate(sample(clip)):
            row, col = divmod(j, 6)
            y, x = top + row_height * (2 * index + row), left + col * width
            picture[y + 16:y + 16 + height, x:x + width] = frame
            cv2.putText(picture, f"{0.25 + j / 2:.2f}s", (x + 5, y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                        .35, (40, 40, 40), 1, cv2.LINE_AA)
            if col == 0:
                cv2.putText(picture, label, (13, y + 38), cv2.FONT_HERSHEY_SIMPLEX,
                            .5, (33, 33, 33), 1, cv2.LINE_AA)
    require(cv2.imwrite(str(output), picture, [cv2.IMWRITE_JPEG_QUALITY, 94]), f"could not write {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "review")
    parser.add_argument("--partial", action="store_true", help="export completed cases without freezing the full panel")
    args = parser.parse_args()
    output = args.output
    blind = output / "blind"
    require(not (blind / "index.json").exists(), "blind panel is already frozen")
    blind.mkdir(parents=True, exist_ok=True)
    items = [item for item in cases() if item["family"] == "holdout"]
    rng = random.Random(814331583)
    rng.shuffle(items)
    records, private = [], {}
    for i, item in enumerate(items, 1):
        name = f"Q{i:02d}"
        directory = blind / name
        randomized = list(ARMS)
        rng.shuffle(randomized)
        mapping = dict(zip("XYZUV", randomized))
        if args.partial and not all(
            (ROOT / "results/runs" / f"{item['id']}--{role}" / "receipt.json").is_file()
            for role in ("base", *ARMS)
        ):
            continue
        directory.mkdir(exist_ok=True)
        clips, hashes = {}, {}
        for label, role in {"BASE": "base", **mapping}.items():
            folder = ROOT / "results/runs" / f"{item['id']}--{role}"
            source = folder / "video.mp4"
            digest = sha256(source)
            require(read_json(folder / "receipt.json")["video_sha256"] == digest, "candidate checksum mismatch")
            target = directory / f"{label}.mp4"
            stripped(source, target)
            clips[label] = target
            hashes[role] = digest
        board(name, clips, directory / "sequence.jpg")
        records.append({"id": name, "target": NOTES, "candidates": list(mapping),
                        "board": f"{name}/sequence.jpg"})
        private[name] = {"case": item["id"], "mapping": mapping, "sha256": hashes}
    index_path = blind / ("partial-index.json" if args.partial else "index.json")
    if not args.partial:
        write_json(output / "group-key.json", {"cases": private})
    write_json(index_path, {"cases": records, "sample_times_s": [0.25 + i / 2 for i in range(12)]})


if __name__ == "__main__":
    main()
