"""Verify H3 receipts and generate aligned visual evidence for vehicle-edit review."""
import argparse
import csv
import hashlib
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from .files import read_json, require, sha256, write_json, write_new


def decoded_hash(path):
    command = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path)]
    frames = subprocess.check_output([*command, "-map", "0:v:0", "-f", "framemd5", "-"])
    audio = subprocess.check_output([*command, "-map", "0:a:0", "-f", "hash", "-hash", "sha256", "-"])
    return {"video_framemd5_sha256": hashlib.sha256(frames).hexdigest(),
            "audio_pcm_sha256": audio.decode().strip().split("=", 1)[1]}


def frame(path, timestamp, target, tail=False):
    seek = ["-sseof", f"-{timestamp}"] if tail else ["-ss", str(timestamp)]
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *seek, "-i", str(path),
                    "-frames:v", "1", "-threads", "1", str(target)], check=True)


def sheet(path, rows, timestamps, raw_directory=None):
    from PIL import Image, ImageDraw, ImageOps

    canvas = Image.new("RGB", (384 * len(timestamps), 245 * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    if raw_directory:
        raw_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hypit-review-") as temp:
        for row, (label, media, tail) in enumerate(rows):
            for col, time in enumerate(timestamps):
                temp_frame = Path(temp) / "frame.jpg"
                frame(media, time, temp_frame, tail)
                image = Image.open(temp_frame).convert("RGB")
                if raw_directory:
                    raw = raw_directory / f"clip-{row + 1}-t{time}.jpg"
                    if not raw.exists():
                        write_new(raw, lambda target: target.write_bytes(temp_frame.read_bytes()))
                x, y = col * 384, row * 245
                canvas.paste(ImageOps.contain(image, (384, 216)), (x, y + 27))
                draw.text((x + 6, y + 7), f"{label}  {'-' if tail else '+'}{time}s", fill="black")
    write_new(path, lambda tmp: canvas.save(tmp, quality=94))


def audit_case(records, files, historical, baseline_inputs):
    data = {}
    for group, record in records.items():
        file = files[group]
        require(record["status"] == "completed" and sha256(file) == record["video_sha256"],
                f"MP4 does not match receipt: {file}")
        data[group] = {"elapsed_s": record["elapsed_s"], "video_sha256": record["video_sha256"]}
    if {"A0", "A1", "A2"} <= records.keys():
        baseline, edited, random = [records[k] for k in ("A0", "A1", "A2")]
        data["paired_payloads_match"] = all(
            baseline["payloads"][stem] == edited["payloads"][stem] == random["payloads"][stem]
            for stem in ("latent", "noise")
        )
        a1, a2 = [r["vector_edit"]["actual_change_l2"] for r in (edited, random)]
        data["edit_l2"] = {"A1": a1, "A2": a2, "relative_difference": abs(a1 - a2) / max(a1, a2)}
        graphs = [read_json(files[group].parent / "request.json")["prompt"] for group in ("A0", "A1", "A2")]
        for group, graph in zip(("A0", "A1", "A2"), graphs):
            require(graph["7"]["class_type"] == "LienyuH3LoadConditionPackage"
                    and graph["8"]["class_type"] == "LienyuH3ReplayInitialNoise"
                    and all("prompt" not in node["inputs"] for node in graph.values()),
                    f"prompt or non-replay input detected: {group}")
        data["replay_without_text"] = True
        common_nodes = [{key: item for key, item in graph.items() if key not in ("7", "8", "16")} for graph in graphs]
        data["paired_model_and_sampling_match"] = common_nodes[0] == common_nodes[1] == common_nodes[2]
    if "A0" in records:
        baseline = records["A0"]
        if historical:
            source = historical / f"h3_gen-performance-s{baseline['seed']}_00001_.mp4"
        else:
            source = baseline_inputs / baseline["package"] / "video.mp4"
        if source.is_file():
            original, replay = decoded_hash(source), decoded_hash(files["A0"])
            data["baseline_decoded_comparison"] = {"match": original == replay, "original": original, "replay": replay}
    if "A1" in records and "A3" in records:
        a1, a3 = records["A1"], records["A3"]
        data["seam_reuses_clip"] = a3.get("reused_job") == a1["job_id"] and a1["video_sha256"] == a3["video_sha256"]
        if "first_frame_sync" in a3:
            data["first_frame_sync"] = a3["first_frame_sync"]
            data["a3_noise_and_latent_match_a1"] = all(a1["payloads"][field] == a3["payloads"][field]
                                                      for field in ("noise", "latent"))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--historical-outputs", type=Path)
    parser.add_argument("--phase", choices=("pilot", "holdout", "continuity"))
    args = parser.parse_args()
    lab = Path(__file__).resolve().parents[1]
    run_root = lab / "results/runs"
    cases, files, table = defaultdict(dict), defaultdict(dict), []
    for receipt in sorted(run_root.glob("*/receipt.json")):
        data = read_json(receipt)
        case, group = data["job_id"].rsplit("--", 1)
        if args.phase and not case.startswith(f"{args.phase}--"):
            continue
        cases[case][group] = data
        files[case][group] = receipt.parent / data["video"]

    args.destination.mkdir(parents=True, exist_ok=True)
    audit = {}
    for case, records in cases.items():
        if any(not (files[case][group].parent / "request.json").exists()
               for group in records if group in ("A0", "A1", "A2")):
            continue
        history = args.historical_outputs if "--coastal-performance--" in case else None
        audit[case] = audit_case(records, files[case], history, lab / "outputs/modelarts-inputs")
        if case.startswith("continuity--"):
            continue
        order = sorted(records, key=lambda group: hashlib.sha256(f"{case}/{group}".encode()).digest())
        blind_labels = [(f"Clip {index}", files[case][group], False) for index, group in enumerate(order, 1)]
        name = args.destination / f"{case}.jpg"
        if not name.exists():
            local_detail = "--vehicle.wheel_design--" in case or "--vehicle.rear_side_window_shape--" in case
            sheet(name, blind_labels, [1, 3, 5], args.destination / "detail-frames" / case if local_detail else None)
        table.extend((case, n, group, files[case][group].name) for n, group in enumerate(order, 1))
    chains = sorted(case for case in cases if case.startswith("continuity--"))
    if len(chains) == 2:
        first, second = chains
        for group in ("A0", "A1", "A3"):
            if group in files[first] and group in files[second]:
                name = args.destination / f"seam-{group}.jpg"
                if not name.exists():
                    sheet(name, [("Previous shot", files[first][group], True),
                                 ("Next shot", files[second][group], False)], [0.46, 0.25, 0.08])
    def save_key(path):
        with path.open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["case", "blind_clip", "variant", "video_filename"])
            writer.writerows(table)
    write_new(args.destination / "review-key.csv", save_key)
    write_json(args.destination / "audit.json", audit)
    print(f"Review evidence for {len(audit)} cases in {args.destination}")


if __name__ == "__main__":
    main()
