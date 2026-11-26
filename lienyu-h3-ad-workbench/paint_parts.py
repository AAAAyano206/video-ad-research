#!/usr/bin/env python3
"""Transfer a color from an H3 donor onto a tracked source car's paint."""

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import av
import cv2
import numpy as np


LUMA = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)
MASK_VERSION = "paint-components-v1"


def digest(path):
    with path.open("rb") as handle:
        value = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def decode(path):
    with av.open(str(path)) as clip:
        video = clip.streams.video[0]
        if not video.average_rate:
            raise ValueError(f"missing frame rate in {path}")
        video.codec_context.thread_count = 2
        return [frame.to_ndarray(format="bgr24") for frame in clip.decode(video)], video.average_rate


def kernel(radius):
    width = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))


def red_support(frame):
    hue, saturation, light = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
    blue, green, red = (channel.astype(np.int16) for channel in cv2.split(frame))
    red_hue = (hue <= 16) | (hue >= 163)
    weak = red_hue & (saturation >= 25) & (light >= 14) & (red - green >= 4) & (red - blue >= 3)
    strong = weak & (saturation >= 51) & (red - green >= 12) & (red - blue >= 8)
    return weak, strong


def detect_vehicle(model, frames, source):
    masks = np.zeros((len(frames), *frames[0].shape[:2]), dtype=np.uint8)
    for start in range(0, len(frames), 8):
        batch = frames[start:start + 8]
        predictions = model.predict(batch, device="cpu", retina_masks=True, imgsz=448,
                                    conf=0.09, classes=[2, 5, 7], max_det=4, verbose=False)
        for index, (prediction, frame) in enumerate(zip(predictions, batch), start):
            if prediction.masks is None:
                continue
            candidates = prediction.masks.data.cpu().numpy().astype(bool)
            confidence = prediction.boxes.conf.cpu().numpy()
            if source:
                _, red = red_support(frame)
                area = (candidates & red).sum(axis=(1, 2))
                scores = area * (0.4 + confidence)
            else:
                area = candidates.sum(axis=(1, 2))
                scores = np.sqrt(area) * confidence
            best = int(scores.argmax())
            if area[best] >= 16:
                masks[index] = candidates[best]
    return masks


def warp_mask(mask, previous, current, grid):
    size = (max(8, current.shape[1] // 2), max(8, current.shape[0] // 2))
    current_small = cv2.resize(current, size, interpolation=cv2.INTER_AREA)
    previous_small = cv2.resize(previous, size, interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(current_small, previous_small, None, .5, 3, 13, 3, 5, 1.1, 0)
    full_flow = cv2.resize(flow, current.shape[::-1], interpolation=cv2.INTER_LINEAR)
    full_flow *= (current.shape[1] / size[0], current.shape[0] / size[1])
    return cv2.remap(mask, grid[0] + full_flow[..., 0], grid[1] + full_flow[..., 1],
                     cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)


def fill_detection_gaps(masks, frames, grid):
    known = np.flatnonzero(masks.any(axis=(1, 2)))
    if known.size == 0:
        return 0
    for index, mask in enumerate(masks):
        if mask.any():
            continue
        neighbor = int(known[np.argmin(np.abs(known - index))])
        if abs(neighbor - index) <= 4:
            old = cv2.cvtColor(frames[neighbor], cv2.COLOR_BGR2GRAY)
            current = cv2.cvtColor(frames[index], cv2.COLOR_BGR2GRAY)
            masks[index] = warp_mask(masks[neighbor], old, current, grid)
    return int(known.size)


def annotation_polygons(spec, index, shape):
    output = np.zeros(shape, dtype=np.uint8)
    for item in spec:
        start, end = item["start_frame"], item["end_frame"]
        if not start <= index <= end:
            continue
        initial = np.asarray(item["polygon"], dtype=np.float32)
        final = np.asarray(item.get("polygon_end", initial), dtype=np.float32)
        progress = (index - start) / max(1, end - start)
        vertices = initial + (final - initial) * progress
        vertices *= (shape[1] - 1, shape[0] - 1)
        cv2.fillPoly(output, [np.rint(vertices).astype(np.int32)], 1)
    return output.astype(bool)


def load_annotations(path, source_hash, frame_count):
    if path is None:
        return {}
    data = json.loads(path.read_text())
    if data.get("source_sha256") != source_hash:
        raise ValueError("part annotations must identify the exact source MP4")
    if set(data) - {"source_sha256", "include", "exclude"}:
        raise ValueError("part annotation fields must be source_sha256, include, or exclude")
    for operation in ("include", "exclude"):
        for part in data.get(operation, []):
            start, end = part["start_frame"], part["end_frame"]
            if type(start) is not int or type(end) is not int or not 0 <= start <= end < frame_count:
                raise ValueError("part annotation frame range is invalid")
            polygon = np.asarray(part["polygon"])
            endpoint = np.asarray(part.get("polygon_end", polygon))
            if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
                    or endpoint.shape != polygon.shape
                    or np.any(~np.isfinite(polygon)) or np.any(~np.isfinite(endpoint))
                    or np.any((polygon < 0) | (polygon > 1))
                    or np.any((endpoint < 0) | (endpoint > 1))):
                raise ValueError("part annotations need matching polygons of normalized [x, y] points")
    return data


def paint_components(frame, vehicle, include, exclude):
    """Use source geometry and red chroma to extend the paint mask or reject isolated trim."""
    weak, strong = red_support(frame)
    allowed = (weak & include & ~exclude).astype(np.uint8)
    protected = exclude.copy()
    locations = cv2.findNonZero(vehicle)
    if locations is None:
        return allowed, protected

    x, y, width, height = cv2.boundingRect(locations)
    support = vehicle.astype(bool)
    radius = min(16, max(3, int(round(max(width, height) * .055))))
    near_vehicle = cv2.dilate(vehicle, kernel(radius)).astype(bool)
    red = (weak & near_vehicle).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(red)
    inside_counts = np.bincount(labels[support], minlength=count)
    strong_counts = np.bincount(labels[strong & near_vehicle], minlength=count)
    minimum = max(7, int(width * height * .0006))
    for label, (left, top, part_width, part_height, area) in enumerate(stats[1:], 1):
        if area < minimum or strong_counts[label] == 0:
            continue
        ys, xs = slice(top, top + part_height), slice(left, left + part_width)
        piece = labels[ys, xs] == label
        inside = inside_counts[label] / area
        compact = area / (part_width * part_height) > .32
        # A small compact red island inside the lower car is consistent with a caliper
        # or reflector. A long, flat rocker panel should stay eligible in that band.
        wheel_band = centers[label, 1] > y + height * .70
        trim_sized = area < max(13, width * height * .008)
        if wheel_band and trim_sized and compact and part_width < part_height * 1.8:
            protected[ys, xs] |= piece
            continue
        thin_upper_part = (centers[label, 1] <= y + height * .56 and part_width > part_height * 1.3
                           and area <= width * height * .14)
        overhang = (inside >= .28 and area >= minimum * 2) or thin_upper_part
        if not overhang:
            piece &= support[ys, xs]
        if inside > 0 or thin_upper_part:
            allowed[ys, xs] |= piece.astype(np.uint8)
    allowed[include & weak] = 1
    protected[include & ~exclude] = False
    allowed[protected] = 0
    return allowed, protected


def track_paint(model, frames, annotations):
    foreground = detect_vehicle(model, frames, source=True)
    shape = foreground.shape[1:]
    grid = np.meshgrid(np.arange(shape[1], dtype=np.float32), np.arange(shape[0], dtype=np.float32))
    detected = fill_detection_gaps(foreground, frames, grid)
    paint = np.zeros_like(foreground)
    previous_gray = None
    for index, (frame, vehicle) in enumerate(zip(frames, foreground)):
        include = annotation_polygons(annotations.get("include", ()), index, shape)
        exclude = annotation_polygons(annotations.get("exclude", ()), index, shape)
        selected, protected = paint_components(frame, vehicle, include, exclude)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if index and selected.any() and paint[index - 1].any():
            # Stabilize one-pixel red margins while retaining current-frame color
            # and part gates. Never follow the previous mask into a protected area.
            prior = warp_mask(paint[index - 1], previous_gray, gray, grid) != 0
            boundary = cv2.dilate(selected, kernel(1)).astype(bool)
            red, _ = red_support(frame)
            selected[prior & red & boundary & vehicle.astype(bool) & ~protected] = 1
        paint[index] = selected
        previous_gray = gray
    return foreground, paint, detected


def dominant_hue(model, frames):
    samples = frames[::max(1, (len(frames) + 31) // 32)]
    vehicles = detect_vehicle(model, samples, source=False)
    histogram = np.zeros(180, dtype=np.float64)
    for frame, vehicle in zip(samples, vehicles):
        if not vehicle.any():
            continue
        hue, saturation, light = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        # Use the eroded vehicle mask to avoid sampling a newly generated backdrop.
        core = cv2.erode(vehicle, kernel(2)).astype(bool)
        eligible = core & (saturation >= 66) & (light >= 24)
        weights = (saturation[eligible].astype(np.float32) / 255) ** 2
        histogram += np.bincount(hue[eligible], weights=weights, minlength=180)
    if histogram.sum() < 6:
        return None, 0.0, int(vehicles.any(axis=(1, 2)).sum())
    smoothed = sum((8 - abs(shift)) * np.roll(histogram, shift) for shift in range(-7, 8))
    hue = int(smoothed.argmax())
    support = sum(histogram[(hue + offset) % len(histogram)] for offset in range(-7, 8))
    return hue, float(support / histogram.sum()), int(vehicles.any(axis=(1, 2)).sum())


def transfer(frame, mask, hue):
    if not mask.any():
        return frame
    bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR).reshape(3).astype(np.float32) / 255
    direction = bgr - bgr @ LUMA
    result = frame.copy()
    pixels = frame[mask != 0].astype(np.float32) / 255
    luminance = pixels @ LUMA
    chroma = pixels.max(axis=1) - pixels.min(axis=1)
    strength = chroma * 1.08
    for coefficient in direction:
        if coefficient > .001:
            strength = np.minimum(strength, (1 - luminance) / coefficient)
        if coefficient < -.001:
            strength = np.minimum(strength, luminance / -coefficient)
    color = luminance[:, None] + strength[:, None] * direction
    result[mask != 0] = np.rint(color * 255).clip(0, 255).astype(np.uint8)
    return result


def audio_hash(path):
    with av.open(str(path)) as clip:
        streams = len(clip.streams.audio)
    return [
        subprocess.check_output([
            "ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", f"0:a:{index}",
            "-c:a", "pcm_s16le", "-f", "hash", "-hash", "sha256", "-",
        ], text=True).strip()
        for index in range(streams)
    ]


def render(source, frames, paint, rate, hue, staging):
    video = staging / "video.mp4"
    if hue is None or not paint.any():
        shutil.copyfile(source, video)
        return
    track = staging / "track.mp4"
    height, width = frames[0].shape[:2]
    with av.open(str(track), mode="w") as clip:
        output = clip.add_stream("libx264", rate=rate, options={"preset": "fast", "crf": "12"})
        output.width, output.height, output.pix_fmt = width, height, "yuv420p"
        output.codec_context.thread_count = 2
        for index, (frame, mask) in enumerate(zip(frames, paint)):
            changed = av.VideoFrame.from_ndarray(transfer(frame, mask, hue), format="bgr24")
            changed.pts = index
            changed.time_base = Fraction(1, rate)
            for packet in output.encode(changed):
                clip.mux(packet)
        for packet in output.encode():
            clip.mux(packet)
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(track), "-i", str(source),
        "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "-1", "-c", "copy",
        "-movflags", "+faststart", str(video),
    ], check=True)


def run(args):
    cv2.setNumThreads(2)
    started = time.monotonic()
    source_hash, donor_hash = digest(args.source), digest(args.donor)
    weights = args.weights.resolve(strict=True)
    weights_hash = digest(weights)
    annotations_hash = digest(args.parts_json) if args.parts_json else None
    mask_input_hash = digest(args.mask_input) if args.mask_input else None
    identity = {
        "source_sha256": source_hash, "donor_sha256": donor_hash, "weights_sha256": weights_hash,
        "annotations_sha256": annotations_hash, "mask_input_sha256": mask_input_hash, "mask_version": MASK_VERSION,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt = args.output_dir / "receipt.json"
    if receipt.is_file():
        previous = json.loads(receipt.read_text())
        if (any(previous.get(key) != value for key, value in identity.items())
                or digest(args.output_dir / "video.mp4") != previous["video_sha256"]
                or digest(args.output_dir / "masks.npz") != previous["mask_sha256"]):
            raise ValueError(f"existing result does not match inputs or hashes in {args.output_dir}")
        return previous

    os.environ.setdefault("YOLO_CONFIG_DIR", str(args.output_dir.parent / ".paint-yolo"))
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(min(8, torch.get_num_threads()))
    frames, rate = decode(args.source)
    donors, _ = decode(args.donor)
    if not frames or not donors:
        raise ValueError("both source and donor must contain video frames")
    model = YOLO(str(weights))
    if args.mask_input:
        with np.load(args.mask_input, allow_pickle=False) as saved:
            if (saved["source_sha256"].item() != source_hash or saved["mask_version"].item() != MASK_VERSION
                    or saved["weights_sha256"].item() != weights_hash):
                raise ValueError("reused mask does not match source MP4, mask implementation, or weights")
            foreground = saved["foreground"]
            paint = saved["paint"]
            detected = int(saved["direct_detections"].item())
        expected_shape = (len(frames), *frames[0].shape[:2])
        if paint.shape != expected_shape or foreground.shape != expected_shape:
            raise ValueError("reused mask shape does not match source frames")
    else:
        annotations = load_annotations(args.parts_json, source_hash, len(frames))
        foreground, paint, detected = track_paint(model, frames, annotations)
    masked = time.monotonic()
    hue, confidence, donor_detections = dominant_hue(model, donors)
    palette_end = time.monotonic()
    with tempfile.TemporaryDirectory(dir=args.output_dir) as temporary:
        staging = Path(temporary)
        np.savez_compressed(staging / "masks.npz", paint=paint, foreground=foreground,
                            source_sha256=source_hash, mask_version=MASK_VERSION, weights_sha256=weights_hash,
                            direct_detections=detected)
        render(args.source, frames, paint, rate, hue, staging)
        source_audio = audio_hash(args.source)
        final_audio = audio_hash(staging / "video.mp4")
        if source_audio != final_audio:
            raise RuntimeError("decoded source audio changed during transfer")
        coverage = paint.sum(axis=(1, 2))
        stage = {
            **identity,
            "video_sha256": digest(staging / "video.mp4"),
            "mask_sha256": digest(staging / "masks.npz"),
            "source_audio_preserved": True,
            "source_audio_hashes": source_audio,
            "target_hue": hue,
            "palette_peak_fraction": confidence,
            "donor_detected_sample_frames": donor_detections,
            "source_frames": len(frames),
            "source_rate": str(rate),
            "source_direct_detections": detected,
            "painted_frames": int(np.count_nonzero(coverage)) if hue is not None else 0,
            "median_selected_pixels": float(np.median(coverage)),
            "selection_status": ("no_eligible_paint" if not paint.any()
                                 else "no_usable_donor_hue" if hue is None else "transferred"),
            "method": "YOLO vehicle detection, geometry/chroma paint-component heuristics, temporal mask checks, "
                      "optional source-only include/exclude polygons, luminance-preserving donor color transfer",
            "added_h3_calls": 0,
            "source_mask_reused": bool(args.mask_input),
            "mask_seconds": masked - started,
            "palette_seconds": palette_end - masked,
            "elapsed_s": time.monotonic() - started,
        }
        (staging / "receipt.json").write_text(json.dumps(stage, indent=2) + "\n")
        for name in ("video.mp4", "masks.npz", "receipt.json"):
            (staging / name).replace(args.output_dir / name)
    return stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=os.environ.get("H3_YOLO_WEIGHTS"))
    parts = parser.add_mutually_exclusive_group()
    parts.add_argument("--parts-json", type=Path)
    parts.add_argument("--mask-input", type=Path)
    args = parser.parse_args()
    if args.weights is None:
        parser.error("provide --weights or set H3_YOLO_WEIGHTS")
    print(json.dumps(run(args)))


if __name__ == "__main__":
    main()
