"""Track the source vehicle with per-frame instance masks and brief dropout recovery."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / "instance/yolo"))
os.environ.setdefault("OMP_NUM_THREADS", "8")

import cv2
import numpy as np
import torch
from safetensors.torch import save_file
from ultralytics import YOLO

from run import ROOT, clip_groups, digest, read, video_frames, write


def masks_for_shot(model, frames):
    masks = np.zeros(frames.shape[:3], dtype=np.uint8)
    for start in range(0, len(frames), 8):
        batch = frames[start:start+8]
        predictions = model.predict(list(batch), device="cpu", retina_masks=True, imgsz=448,
                                    conf=0.09, classes=[2, 5, 7], max_det=4, verbose=False)
        for i, (prediction, frame) in enumerate(zip(predictions, batch), start):
            if prediction.masks is None:
                continue
            options = prediction.masks.data.cpu().numpy().astype(bool)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hue, sat, value = cv2.split(hsv)
            color = ((hue < 17) | (hue > 160)) & (sat > 55) & (value > 28)
            scores = []
            for mask, score in zip(options, prediction.boxes.conf.tolist()):
                car_paint = np.count_nonzero(color & mask)
                scores.append(car_paint * (0.45 + score))
            winner = int(np.argmax(scores))
            if scores[winner] >= 20:
                masks[i] = options[winner].astype(np.uint8) * 255
    known = np.flatnonzero(masks.any(axis=(1, 2)))
    if not len(known):
        raise ValueError("no confident red vehicle masks")
    for i, mask in enumerate(masks):
        if mask.any():
            continue
        nearest = int(known[np.abs(known - i).argmin()])
        distance = abs(i - nearest)
        if distance > 8:
            continue
        # Move the nearest reliable mask by frame-to-frame optical flow.
        current = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        anchor = cv2.cvtColor(frames[nearest], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(current, anchor, None, .5, 3, 19, 3, 5, 1.1, 0)
        yy, xx = np.indices(current.shape, dtype=np.float32)
        masks[i] = cv2.remap(masks[nearest], xx + flow[..., 0], yy + flow[..., 1],
                            cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return masks, int(len(known))


def get_paint(frames, masks):
    paint = []
    for frame, mask in zip(frames, masks):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        color = ((hue <= 18) | (hue >= 159)) & (saturation >= 32) & (value >= 25)
        # Keep neutral reflections and lights while admitting dark paint.
        paint.append(((mask > 0) & color).astype(np.uint8))
    return np.stack(paint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("attribute", choices=("paint", "body"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    torch.set_num_threads(8)
    cv2.setNumThreads(2)
    weights = ROOT / "assets/yolo11n-seg.pt"
    weight_hash = digest(weights)
    model = YOLO(weights)
    root = ROOT / "masks-segmented"
    root.mkdir(exist_ok=True)
    for case, originals in list(clip_groups(args.attribute).items())[:args.limit]:
        sidecar = root / f"{case}.json"
        if sidecar.exists():
            continue
        source = originals["A0"] / "video.mp4"
        settings = read(originals["A0"] / "receipt.json")["settings"]
        frames = video_frames(source, settings["width"], settings["height"])
        masks, n = masks_for_shot(model, frames)
        paint = get_paint(frames, masks)
        path = sidecar.with_suffix(".safetensors")
        save_file({"pixel_mask": torch.from_numpy(masks), "paint_mask": torch.from_numpy(paint)}, str(path))
        meta = {"source_video_sha256": digest(source), "mask_sha256": digest(path),
                "segmentation_weights_sha256": weight_hash, "direct_detections": n, "frames": len(frames),
                "propagated_or_direct_frames": int(masks.any(axis=(1, 2)).sum())}
        write(sidecar, meta)
        print(case, meta, flush=True)


if __name__ == "__main__":
    main()
