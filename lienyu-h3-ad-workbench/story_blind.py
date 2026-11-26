"""Package silent anonymous visual evidence for review of the whole new ad."""
import argparse
import json
from pathlib import Path
import random
import subprocess


ROOT = Path(__file__).resolve().parent


def run(command):
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    story = ROOT / "results" / "story"
    cost = json.loads((story / "cost-and-equivalence.json").read_text())
    review = story / "review"
    blind = review / "blind"
    blind.mkdir(parents=True, exist_ok=True)
    methods = ["full", "selective"]
    random.Random(443318).shuffle(methods)
    collapsed = cost["matched_arm_evidence"]["final_decoded_streams_equal"]
    if collapsed:
        methods = methods[:1]
    candidates = dict(zip("XY", methods))
    frame_ids = sorted(set(range(0, 864, 24)) | {351, 355, 359, 364, 368, 711, 715, 719, 724, 728, 862, 863})
    parts = "+".join(f"eq(n\\,{i})" for i in frame_ids)
    for label, source in {"BASE": "base", **candidates}.items():
        clip = blind / f"{label}.mp4"
        run([args.ffmpeg, "-y", "-v", "error", "-nostdin", "-i", str(story / source / "final.mp4"),
             "-map", "0:v:0", "-c", "copy", "-map_metadata", "-1", "-map_chapters", "-1", str(clip)])
        run([args.ffmpeg, "-y", "-v", "error", "-nostdin", "-i", str(clip),
             "-vf", f"select={parts},scale=210:118,tile=6x8", "-frames:v", "1",
             "-threads", "1", str(clip.with_suffix(".jpg"))])
    public = {"id": "S01", "candidates": list(candidates), "base": "BASE", "sample_times_s": [i / 24 for i in frame_ids],
              "target": "Turn the entire visible red body paint blue and keep scene, camera, body shape, wheels, glass, lights and reflections. Check the same car over the cut at 15 s, and preserve the no-car insert beginning at 30 s."}
    (blind / "index.json").write_text(json.dumps(public, indent=2) + "\n")
    (review / "group-key.json").write_text(json.dumps({"mapping": candidates, "identical_versions_collapsed": collapsed}, indent=2) + "\n")


if __name__ == "__main__":
    main()
