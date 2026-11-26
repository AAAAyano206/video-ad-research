"""Render original, selected paint pixels, and the resulting shot at aligned times."""
import argparse
from pathlib import Path

import av
import cv2
import numpy as np


def take(path, indices):
    with av.open(str(path)) as movie:
        found = {i: f.to_ndarray(format="bgr24") for i, f in enumerate(movie.decode(video=0)) if i in indices}
    return [found[i] for i in indices]


def render(source, edited, mask, output):
    with np.load(mask, allow_pickle=False) as layers:
        paint = layers["paint"]
    selected = tuple(round((len(paint) - 1) * p) for p in (0.08, .35, .68, .94))
    originals = take(source, selected)
    changes = take(edited, selected)
    overlays = []
    for frame, i in zip(originals, selected):
        edited_frame = frame.copy()
        keep = paint[i] != 0
        edited_frame[keep] = (frame[keep] * .45 + np.array([0, 232, 255]) * .55).astype(np.uint8)
        overlays.append(edited_frame)
    frames = np.vstack([np.hstack(row) for row in (originals, overlays, changes)])
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frames, [cv2.IMWRITE_JPEG_QUALITY, 93]):
        raise RuntimeError(f"failed to write {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("edited", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    render(arguments.source, arguments.edited, arguments.mask, arguments.output)
