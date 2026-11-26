"""Use a vector-generated H3 clip as a paint-only color donor for the H3 source shot."""
import argparse
import subprocess

import av
import cv2
import numpy as np
from safetensors.numpy import load_file

from run import ROOT, clip_groups, decoded_hash, digest, prepare_mask, read, video_frames, write


LUMA = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)


def palette(frames, source_paint):
    """Find the dominant chromatic hue near the source car without a target-hue term."""
    histogram = np.zeros(180, dtype=np.float64)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 35))
    for frame, mask in zip(frames, source_paint):
        hue, chroma, value = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        near = cv2.dilate(mask, kernel).astype(bool)
        selected = near & (chroma >= 86) & (value >= 45)
        # High-chroma, normally exposed colors closest to the original car are weighted most.
        weights = chroma[selected].astype(float) ** 2 * (1 + 0.4 * mask[selected])
        histogram += np.bincount(hue[selected], weights=weights, minlength=180)
    if not histogram.any():
        raise ValueError("no usable H3 donor colors")
    smoothed = sum(np.roll(histogram, shift) for shift in range(-7, 8))
    peak = int(smoothed.argmax())
    return peak, float(smoothed[peak] / histogram.sum())


def apply_paint(base, source_paint, hue, foreground=None):
    bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR).reshape(3).astype(np.float32) / 255
    direction = bgr - bgr @ LUMA
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 7))
    for i, (original, seeds) in enumerate(zip(base, source_paint)):
        hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        red = ((hsv[..., 0] <= 17) | (hsv[..., 0] >= 160)).astype(np.float32)
        sat = hsv[..., 1].astype(np.float32)
        support = cv2.dilate(seeds, kernel).astype(np.float32)
        alpha = support * red * np.clip((sat - 25) / 31, 0, 1)
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0.55) * support
        if foreground is not None:
            alpha *= foreground[i] != 0
        image = original.astype(np.float32) / 255
        luma = image @ LUMA
        source_chroma = image.max(axis=2) - image.min(axis=2)
        gamut_ceiling = np.ones_like(luma)
        for coefficient in direction:
            if coefficient > 0.001:
                gamut_ceiling = np.minimum(gamut_ceiling, (1 - luma) / coefficient)
            if coefficient < -0.001:
                gamut_ceiling = np.minimum(gamut_ceiling, luma / -coefficient)
        strength = np.minimum(source_chroma * 1.08, gamut_ceiling)
        colored = luma[..., None] + strength[..., None] * direction
        out = np.rint((image + alpha[..., None] * (colored - image)) * 255).clip(0, 255).astype(np.uint8)
        yield out


def publish(source, donor, frames, mask_file, hue, confidence, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "video.mp4"
    if output.exists():
        recorded = read(output_dir / "receipt.json")
        assert digest(output) == recorded["video_sha256"]
        return
    height, width = frames[0].shape[:2]
    temporary = output_dir / "video.tmp.mp4"
    video_only = output_dir / "pixels.tmp.mp4"
    with av.open(str(video_only), mode="w") as container:
        stream = container.add_stream("libx264", rate=24, options={"preset": "fast", "crf": "12"})
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.codec_context.thread_count = 2
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="bgr24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    subprocess.run([
        "ffmpeg", "-y", "-nostdin", "-v", "error", "-i", str(video_only), "-i", str(source),
        "-map", "0:v:0", "-map", "1:a:0", "-map_metadata", "-1", "-c", "copy",
        "-movflags", "+faststart", str(temporary),
    ], check=True)
    video_only.unlink()
    assert decoded_hash(source, "a") == decoded_hash(temporary, "a")
    temporary.rename(output)
    donor_receipt = read(donor.parent / "receipt.json")
    write(output_dir / "receipt.json", {
        **donor_receipt, "job_id": output_dir.name, "variant": output_dir.name.rsplit("--", 1)[-1],
        "video": output.name, "video_sha256": digest(output), "source_video_sha256": digest(source),
        "donor_video_sha256": digest(donor), "donor_job": donor.parent.name, "mask_sha256": digest(mask_file),
        "dominant_hue_opencv": hue, "hue_histogram_peak_fraction": confidence,
        "method": "Source geometry and luminance + H3-generated dominant paint hue; no target hue is supplied",
        "new_generation_calls": 0, "replay_received_prompt": False,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--segmented", action="store_true")
    args = parser.parse_args()
    (ROOT / "masks").mkdir(exist_ok=True)
    for case, originals in list(clip_groups("paint").items())[:args.limit]:
        source = originals["A0"] / "video.mp4"
        settings = read(originals["A0"] / "receipt.json")["settings"]
        size = settings["width"], settings["height"]
        base = video_frames(source, *size)
        foreground = None
        if args.segmented:
            mask_file = ROOT / "masks-segmented" / f"{case}.safetensors"
            reference = read(mask_file.with_suffix(".json"))
            assert reference["mask_sha256"] == digest(mask_file) and reference["source_video_sha256"] == digest(source)
            masks = load_file(str(mask_file))
            source_paint, foreground = masks["paint_mask"], masks["pixel_mask"]
        else:
            mask_file, source_paint = prepare_mask(case, source, base)
        variants = ("P3", "P4") if args.segmented else ("P1", "P2")
        for result, original in zip(variants, ("A1", "A2")):
            donor = originals[original] / "video.mp4"
            candidate = video_frames(donor, *size)
            assert len(candidate) == len(base)
            hue, confidence = palette(candidate, source_paint)
            changed = list(apply_paint(base, source_paint, hue, foreground))
            publish(source, donor, changed, mask_file, hue, confidence, ROOT / "results/runs" / f"{case}--{result}")
            print(case, result, hue, round(confidence, 3), flush=True)


if __name__ == "__main__":
    main()
