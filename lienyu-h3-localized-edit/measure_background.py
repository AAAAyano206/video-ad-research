"""Measure frame-aligned changes outside the source vehicle region."""
import argparse

import cv2
import numpy as np
from safetensors.numpy import load_file

from run import ROOT, clip_groups, read, video_frames, write


def measure(source, edited, masks):
    radius = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
    pixels = changed = 0
    abs_total = 0.0
    for src, dst, mask in zip(source, edited, masks):
        outside = cv2.dilate(mask, radius) == 0
        delta = np.abs(src.astype(np.int16) - dst.astype(np.int16))[outside]
        pixels += len(delta)
        changed += int(np.count_nonzero(delta.max(axis=1) > 20))
        abs_total += float(delta.sum())
    return {"background_pixel_channel_mae_0_255": abs_total / (3 * pixels),
            "background_pixels_any_channel_delta_gt_20_pct": 100 * changed / pixels,
            "background_pixels": pixels, "frames": len(source)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("attribute", choices=("paint", "body"))
    args = parser.parse_args()
    summary = {}
    for case, old in clip_groups(args.attribute).items():
        settings = read(old["A0"] / "receipt.json")["settings"]
        size = settings["width"], settings["height"]
        source = video_frames(old["A0"] / "video.mp4", *size)
        masks = load_file(str(ROOT / "masks-segmented" / (case + ".safetensors")))["pixel_mask"]
        groups = ("A1", "A2", "M1", "M2") + (("P3", "P4") if args.attribute == "paint" else ())
        summary[case] = {}
        for group in groups:
            directory = old.get(group, ROOT / "results/runs" / f"{case}--{group}")
            edited = video_frames(directory / "video.mp4", *size)
            assert len(edited) == len(source) == len(masks)
            summary[case][group] = measure(source, edited, masks)
    write(ROOT / "results" / f"background-{args.attribute}.json", {
        "excluded_region": "per-frame source vehicle mask expanded by 11 pixels",
        "criterion": "maximum absolute difference across color channels > 20 of 255",
        "cases": summary,
    })


if __name__ == "__main__":
    main()
