"""Export metadata-free review clips and aligned frames for independent blind scoring."""
import argparse
import random
import subprocess

import cv2
import numpy as np

from run import ROOT, clip_groups, digest, read, video_frames, write


WIDTH, HEIGHT = 192, 128
SAMPLE_FRAMES = tuple(range(6, 144, 12))


def make_board(case_dir, clips, settings):
    labels = list(clips)
    board = np.full((30 + (HEIGHT + 17) * len(clips) * 2, 76 + 6 * WIDTH, 3), 246, np.uint8)
    cv2.putText(board, case_dir.name, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, .63, (35, 35, 35), 1, cv2.LINE_AA)
    shape = settings["width"], settings["height"]
    for i, label in enumerate(labels):
        frames = video_frames(clips[label], *shape)
        if len(frames) != 144:
            raise ValueError(f"unexpected frame count for {clips[label]}: {len(frames)}")
        for n, frame_index in enumerate(SAMPLE_FRAMES):
            row, col = divmod(n, 6)
            y = 30 + (i * 2 + row) * (HEIGHT + 17)
            x = 76 + col * WIDTH
            piece = cv2.resize(frames[frame_index], (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            board[y+16:y+16+HEIGHT,x:x+WIDTH] = piece
            if col == 0:
                cv2.putText(board,label,(8,y+43),cv2.FONT_HERSHEY_SIMPLEX,.55,(15,15,15),1,cv2.LINE_AA)
            cv2.putText(board,f"{frame_index/24:.2f}s",(x+4,y+12),cv2.FONT_HERSHEY_SIMPLEX,.36,(25,25,25),1,cv2.LINE_AA)
    if not cv2.imwrite(str(case_dir / "sequence.jpg"), board, [cv2.IMWRITE_JPEG_QUALITY, 93]):
        raise RuntimeError("contact sheet failed")


def blind(attribute, segmented):
    plan = list(clip_groups(attribute).items())
    rng = random.Random({"paint": 116670939, "body": 71126635}[attribute])
    rng.shuffle(plan)
    review = ROOT / "results" / f"review-{attribute}"
    masks = review / "blind"
    masks.mkdir(parents=True, exist_ok=True)
    locked = ("M1", "M2") if segmented else ("L1", "L2")
    transfer = ("P3", "P4") if segmented else ("P1", "P2")
    candidates = ("A1", "A2") + locked + (transfer if attribute == "paint" else ())
    key, index = {}, []
    target = "Red car paint to blue. Keep the original body, wheels, windows, scene, framing and camera motion."
    if attribute == "body":
        target = ("Two-door coupe to SUV class. Keep red paint, scene, framing and camera motion. "
                  "Body proportions and wheel arches can change as required by the category.")
    for number, (case, earlier) in enumerate(plan, start=1):
        options = list(candidates)
        rng.shuffle(options)
        mapping = dict(zip("XYZUVW", options))
        name = f"{'C' if attribute == 'paint' else 'S'}{number:02d}"
        case_dir = masks / name
        case_dir.mkdir(exist_ok=True)
        clips, hashes = {}, {}
        for label, variant in {"BASE": "A0", **mapping}.items():
            results = earlier.get(variant, ROOT / "results/runs" / f"{case}--{variant}")
            receipt = read(results / "receipt.json")
            source = results / "video.mp4"
            assert digest(source) == receipt["video_sha256"]
            video = case_dir / f"{label}.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-v", "error", "-nostdin", "-i", str(source),
                "-map", "0:v:0", "-c", "copy", "-map_metadata", "-1", "-map_chapters", "-1",
                "-fflags", "+bitexact", str(video),
            ], check=True)
            clips[label] = video
            hashes[variant] = receipt["video_sha256"]
        make_board(case_dir, clips, read(earlier["A0"] / "receipt.json")["settings"])
        key[name] = {"case": case, "mapping": mapping, "sha256": hashes}
        index.append({"id": name, "target": target, "source": "BASE",
                      "candidates": list(mapping), "board": f"{name}/sequence.jpg"})
    write(review / "group-key.json", {"cases": key})
    write(masks / "index.json", {"cases": index, "timeline_s": [frame/24 for frame in SAMPLE_FRAMES]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("attribute", choices=("paint", "body"))
    parser.add_argument("--segmented", action="store_true")
    args = parser.parse_args()
    blind(args.attribute, args.segmented)
