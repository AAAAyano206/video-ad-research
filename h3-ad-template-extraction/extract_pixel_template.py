#!/usr/bin/env python3
"""Pixel-only shot, colored-product mask and track extraction for a blind ad MP4."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def largest_box(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 300:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    return [int(x), int(y), int(w), int(h)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.video), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames, cuts, track, previous_hist = 0, [0], [], None
    preview = None
    while True:
        ok, image = capture.read()
        if not ok:
            break
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if previous_hist is not None and frames - cuts[-1] > fps:
            distance = cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if distance > 0.48:
                cuts.append(frames)
        previous_hist = hist
        red = cv2.inRange(hsv, (0, 75, 45), (12, 255, 255)) | cv2.inRange(hsv, (168, 75, 45), (180, 255, 255))
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        box = largest_box(red)
        if box:
            track.append({"frame": frames, "time_s": frames / fps, "bbox_xywh": box,
                          "mask_pixels": int((red > 0).sum())})
            if preview is None or frames % max(1, int(fps * 3)) == 0:
                preview = image.copy(); x, y, w, h = box
                preview[red > 0] = preview[red > 0] * 0.45 + np.array([0, 255, 255]) * 0.55
                cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.imwrite(str(output / f"mask-track-{frames:06d}.jpg"), preview)
        frames += 1
    capture.release()
    cuts.append(frames)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    result = {
        "input": {"sha256": digest, "bytes": source.stat().st_size, "fps": fps, "frames": frames},
        "method": {"provenance_used": False, "shot_metric": "HSV histogram Bhattacharyya > 0.48",
                   "product_mask": "red HSV + morphology", "tracker": "largest connected mask per frame"},
        "shots": [{"start_frame": a, "end_frame": b - 1, "start_s": a / fps, "end_s": b / fps}
                  for a, b in zip(cuts, cuts[1:])],
        "editable_slots": {"vehicle": {"track": track, "confidence_scope": "red-dominant vehicle only"},
                           "scene": {"boundaries": cuts},
                           "person": {"status": "not localized by this pixel-only baseline"}},
    }
    (output / "template.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"shots": len(result["shots"]), "tracked_frames": len(track), "output": str(output)}))


if __name__ == "__main__":
    main()
