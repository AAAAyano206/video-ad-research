"""Compare the prior whole-vehicle pipeline with the same H3 donor and part editing."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "instances" / "segmentation"))
(Path(os.environ["YOLO_CONFIG_DIR"]) / "Ultralytics").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "vendor"))
from vehicle_edit_lab.files import read_json, require, sha256, write_json
from h3_runs import cases


RUNS = ROOT / "results" / "runs"


def mp4(case, role):
    folder = RUNS / f"{case}--{role}"
    video = folder / "video.mp4"
    require(read_json(folder / "receipt.json")["video_sha256"] == sha256(video), f"bad video for {folder.name}")
    return video


def parts(case, role, donor_role, *, from_mask=None):
    name = f"{case}--{role}"
    directory = RUNS / name
    if (directory / "receipt.json").exists():
        return mp4(case, role)
    arguments = [
        sys.executable, str(ROOT / "paint_parts.py"), "--source", str(mp4(case, "base")),
        "--donor", str(mp4(case, donor_role)), "--output-dir", str(directory),
    ]
    if from_mask:
        arguments.extend(("--mask-input", str(from_mask)))
    subprocess.run(arguments, check=True, env={**os.environ, "H3_YOLO_WEIGHTS": str(ROOT / "assets/yolo11n-seg.pt")})
    return mp4(case, role)


def legacy(case, width, height, model):
    destination = RUNS / f"{case}--legacy"
    if (destination / "receipt.json").exists():
        return
    import numpy as np
    from safetensors.numpy import save_file

    import paint_transfer
    import segment
    from run import video_frames

    start = time.monotonic()
    source, donor = mp4(case, "base"), mp4(case, "target")
    frames = video_frames(source, width, height)
    masks, _ = segment.masks_for_shot(model, frames)
    paint = segment.get_paint(frames, masks)
    hue, confidence = paint_transfer.palette(video_frames(donor, width, height), paint)
    destination.mkdir(parents=True, exist_ok=True)
    mask_file = destination / "masks.safetensors"
    save_file({"foreground": masks, "paint": paint}, str(mask_file))
    pixels = list(paint_transfer.apply_paint(frames, paint, hue, masks))
    paint_transfer.publish(source, donor, pixels, mask_file, hue, confidence, destination)
    write_json(destination / "timing.json", {"elapsed_s": time.monotonic() - start})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true", help="wait for asynchronous H3 receipts")
    parser.add_argument("--family", choices=("holdout", "story", "all"), default="all")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "vendor" / "legacy"))
    import cv2
    import torch
    from ultralytics import YOLO

    cv2.setNumThreads(2)
    torch.set_num_threads(6)
    model = YOLO(ROOT / "assets" / "yolo11n-seg.pt")
    for item in cases():
        case = item["id"]
        if not item["affected"] or args.family not in ("all", item["family"]):
            continue
        donor_roles = ("target", "random") if item["family"] == "holdout" else ("donor", "full")
        if args.wait:
            deadline = time.monotonic() + 900
            while any(not (RUNS / f"{case}--{r}" / "receipt.json").exists() for r in donor_roles):
                if time.monotonic() > deadline:
                    raise TimeoutError(f"H3 did not publish {case}")
                time.sleep(3)
        if item["family"] == "holdout":
            legacy(case, 384, 256, model)
            parts(case, "parts", "target")
            parts(case, "parts-random", "random", from_mask=RUNS / f"{case}--parts" / "masks.npz")
        else:
            parts(case, "parts", "donor")
            parts(case, "full-parts", "full")
        print(case, "postprocess complete", flush=True)


if __name__ == "__main__":
    main()
