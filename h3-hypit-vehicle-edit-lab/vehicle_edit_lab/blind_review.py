"""Package anonymous, time-aligned H3 clips and resolve independent reviewer scorecards."""
import argparse
import csv
import hashlib
import json
import random
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .files import read_json, require, sha256, write_json, write_new


FIELDS = ("id", "candidate", "target", "stable", "body", "body_allow_suv_geometry",
          "scene", "camera", "other", "notes")
DECISIONS = set(FIELDS[2:-1])
PASS = "P"
ATTRIBUTES = {
    "vehicle.paint_color": "Red paint to blue",
    "vehicle.body_class": "Two-door coupe to SUV; keep red paint",
    "vehicle.wheel_design": "Five spokes to a mounted crisscross mesh wheel",
    "vehicle.rear_side_window_shape": "Wedge to quarter window with near-vertical trailing edge",
}
FRAME_WIDTH, FRAME_HEIGHT = 264, 160


def clip_path(root, job):
    directory = root / job
    receipt = read_json(directory / "receipt.json")
    result = directory / receipt["video"]
    require(receipt["status"] == "completed" and sha256(result) == receipt["video_sha256"], f"clip failed audit: {job}")
    return result


def strip_metadata(ffmpeg, source, output):
    write_new(output, lambda path: subprocess.run([
        ffmpeg, "-nostdin", "-v", "error", "-i", str(source), "-map", "0:v:0",
        "-c", "copy", "-map_metadata", "-1", "-map_chapters", "-1", "-fflags", "+bitexact",
        "-flags:v", "+bitexact", str(path),
    ], check=True))


def decode_frames(ffmpeg, path, time_filter, count):
    from PIL import Image

    filters = (f"{time_filter},scale={FRAME_WIDTH}:{FRAME_HEIGHT}:force_original_aspect_ratio=decrease,"
               f"pad={FRAME_WIDTH}:{FRAME_HEIGHT}:(ow-iw)/2:(oh-ih)/2")
    raw = subprocess.check_output([
        ffmpeg, "-nostdin", "-v", "error", "-threads", "1", "-i", str(path), "-vf", filters,
        "-frames:v", str(count), "-threads", "1", "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ])
    size = FRAME_WIDTH * FRAME_HEIGHT * 3
    require(len(raw) == count * size, f"expected {count} frames from {path}, received {len(raw) // size}")
    return [Image.frombytes("RGB", (FRAME_WIDTH, FRAME_HEIGHT), raw[i:i + size]) for i in range(0, len(raw), size)]


def make_board(ffmpeg, directory, films, seam):
    from PIL import Image, ImageDraw

    if seam:
        # The Hypit builds have 720 frames at 24 fps and an exact cut at frame 360.
        timeline = "trim=start_frame=348:end_frame=372,select=not(mod(n\\,2))"
        times = [(index - 6) / 12 for index in range(12)]
    else:
        timeline = "select=eq(mod(n\\,12)\\,6)"
        times = [(6 + 12 * index) / 24 for index in range(12)]
    row_height, header, margin = FRAME_HEIGHT + 20, 34, 85
    board = Image.new("RGB", (margin + FRAME_WIDTH * 6, header + len(films) * 2 * row_height), "#f5f5f5")
    draw = ImageDraw.Draw(board)
    draw.text((9, 9), directory.name + "  ·  " + ("seam (relative time)" if seam else "sequence from start"),
              fill="#181818")
    for group, (label, video) in enumerate(films.items()):
        frames = decode_frames(ffmpeg, video, timeline, 12)
        for index, (frame, time) in enumerate(zip(frames, times)):
            row, col = divmod(index, 6)
            x = margin + col * FRAME_WIDTH
            y = header + (2 * group + row) * row_height
            if col == 0:
                draw.text((8, y + 27), label, fill="#181818")
            draw.text((x + 4, y + 3), f"{time:+.2f}s" if seam else f"{time:.2f}s", fill="#292929")
            board.paste(frame, (x, y + 18))
    write_new(directory / "sequence.jpg", lambda path: board.save(path, quality=90))


def prepare(args):
    root = args.results.resolve()
    destination = args.output.resolve()
    require(not destination.exists(), f"refusing to overwrite blind package {destination}")
    blind = destination / "blind"
    blind.mkdir(parents=True)
    cases = {}
    for folder in sorted((root / "runs").iterdir()):
        if not (folder / "receipt.json").is_file():
            continue
        case, variant = folder.name.rsplit("--", 1)
        if case.startswith(("pilot--", "holdout--")):
            cases.setdefault(case, {})[variant] = clip_path(root / "runs", folder.name)
    require(len(cases) == 48, "expected 48 complete pilot and holdout cases")
    hypit = root / "hypit-continuity"
    seam_manifest = read_json(hypit / "manifest.json")
    for group, receipt in seam_manifest.items():
        require(sha256(hypit / receipt["video"]) == receipt["sha256"], f"Hypit composition digest differs: {group}")
    cases["continuity--vehicle.paint_color--two-shots"] = {
        group: hypit / record["video"] for group, record in seam_manifest.items()
    }
    rng = random.Random(args.seed)
    items = list(cases.items())
    rng.shuffle(items)
    key, index, jobs = {}, [], []
    for i, (case, variants) in enumerate(items, 1):
        label = f"R{i:03}"
        phase, attribute, *_ = case.split("--")
        order = sorted(set(variants) - {"A0"})
        rng.shuffle(order)
        mapped = dict(zip("XYZ", order))
        directory = blind / label
        directory.mkdir()
        listing = {
            "id": label, "phase": phase, "attribute": ATTRIBUTES[attribute], "candidates": list(mapped),
            "evidence": f"{label}/sequence.jpg",
        }
        index.append(listing)
        key[label] = {"case": case, "attribute": attribute, "phase": phase, "mapping": mapped,
                      "source_sha256": {group: sha256(path) for group, path in variants.items()}}
        jobs.append((directory, {"BASE": variants["A0"], **{name: variants[v] for name, v in mapped.items()}},
                     phase == "continuity"))
    write_json(destination / "group-key.json", {"masking_seed": args.seed, "cases": key})

    def render(item):
        directory, sources, seam = item
        films = {}
        for name, source in sources.items():
            films[name] = directory / f"{name}.mp4"
            strip_metadata(args.ffmpeg, source, films[name])
        make_board(args.ffmpeg, directory, films, seam)

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(render, jobs))
    write_json(blind / "index.json", {"cases": index, "source_fps": 24, "anonymous_videos_include_video_stream_only": True})
    print(f"Prepared {len(index)} blind cases and {sum(len(item['candidates']) for item in index)} candidates")


def read_ratings(path, valid):
    with path.open(newline="") as file:
        records = list(csv.DictReader(file))
    require(records and tuple(records[0]) == FIELDS, f"{path} has the wrong header")
    ratings = {}
    for item in records:
        index = item["id"], item["candidate"]
        require(index not in ratings and index in valid, f"duplicate or unknown rating {index} in {path}")
        require(all(item[field] in ("P", "F", "U") for field in DECISIONS), f"invalid result for {index}")
        require(bool(item["notes"].strip()), f"missing visual evidence note for {index}")
        ratings[index] = item
    require(ratings.keys() == valid, f"incomplete ratings in {path}: {sorted(valid - ratings.keys())}")
    return ratings


def temporal_audit(args):
    from PIL import Image, ImageChops, ImageStat

    index = read_json(args.output / "blind/index.json")
    width, height = 96, 56
    stride = width * height * 3

    def measure(path):
        raw = subprocess.check_output([
            args.ffmpeg, "-nostdin", "-threads", "1", "-v", "error", "-i", str(path),
            "-vf", f"scale={width}:{height}", "-fps_mode", "passthrough", "-threads", "1",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
        ])
        count, remainder = divmod(len(raw), stride)
        require(count and not remainder, f"incomplete decode {path}")
        jumps, previous = [], None
        for offset in range(0, len(raw), stride):
            current = Image.frombytes("RGB", (width, height), raw[offset:offset + stride])
            if previous:
                change = ImageStat.Stat(ImageChops.difference(current, previous)).mean
                jumps.append(sum(change) / (3 * 255))
            previous = current
        return count, jumps

    def case(record):
        directory = args.output / "blind" / record["id"]
        reference_frames, reference_changes = measure(directory / "BASE.mp4")
        expected_frames = 720 if record["phase"] == "continuity" else 144
        require(reference_frames == expected_frames, f"wrong frame count for {record['id']}")
        output = {}
        for candidate in record["candidates"]:
            frames, changes = measure(directory / f"{candidate}.mp4")
            require(frames == expected_frames, f"wrong candidate frame count for {record['id']} {candidate}")
            events = []
            for i, change in enumerate(changes):
                lo, hi = max(0, i - 6), min(len(reference_changes), i + 7)
                near_base = max(reference_changes[lo:hi])
                if change >= 0.11 and change >= near_base + 0.045:
                    events.append({"time_s": round((i + 1) / 24, 4), "change": round(change, 4),
                                   "peak_base_nearby": round(near_base, 4)})
            output[candidate] = {"decoded_frames": frames, "candidate_only_change_cues": events}
        return record["id"], output

    with ThreadPoolExecutor(max_workers=6) as executor:
        values = dict(executor.map(case, index["cases"]))
    write_json(args.output / "temporal-audit.json", {
        "definition": "Exhaustive adjacent-frame RGB difference at 96x56, flags advisory visual inspection cues.",
        "cases": values,
    })
    print(f"Scanned every decoded frame across {len(values)} base clips and {sum(map(len, values.values()))} candidates")


def temporal_cues(args):
    from PIL import Image, ImageDraw

    audit = read_json(args.output / "temporal-audit.json")["cases"]
    destination = args.output / "temporal-cues"
    for label, children in audit.items():
        for candidate, audit_row in children.items():
            events = audit_row["candidate_only_change_cues"]
            groups = []
            for event in events:
                if not groups or event["time_s"] - groups[-1][-1]["time_s"] > 0.25:
                    groups.append([])
                groups[-1].append(event)
            for k, group in enumerate(groups, 1):
                peak = max(group, key=lambda e: e["change"])
                center = round(peak["time_s"] * 24)
                start, end = max(0, center - 4), min(audit_row["decoded_frames"], center + 4)
                span = end - start
                # Each frame is kept in order; rows are BASE and the unnamed candidate.
                canvas = Image.new("RGB", (span * FRAME_WIDTH, 2 * (FRAME_HEIGHT + 24)), "white")
                draw = ImageDraw.Draw(canvas)
                for row, name in enumerate(("BASE", candidate)):
                    y = row * (FRAME_HEIGHT + 24)
                    film = args.output / "blind" / label / f"{name}.mp4"
                    selection = f"trim=start_frame={start}:end_frame={end}"
                    for i, frame in enumerate(decode_frames(args.ffmpeg, film, selection, span)):
                        draw.text((i * FRAME_WIDTH + 5, y + 5), f"{name} {(start+i)/24:.3f} s", fill="black")
                        canvas.paste(frame, (i * FRAME_WIDTH, y + 24))
                write_new(destination / f"{label}-{candidate}-{k}.jpg", lambda p: canvas.save(p, quality=92))
    print(f"Generated {len(list(destination.glob('*.jpg')))} consecutive-frame change comparisons")


def read_adjudication(path, first, second):
    overrides = {}
    for item in read_json(path)["reviews"]:
        key = item["id"], item["candidate"]
        require(key in first and key not in overrides, f"duplicate or unknown adjudication {key}")
        fields = {field for field in DECISIONS if first[key][field] != second[key][field]}
        require(fields and fields == item["decisions"].keys(), f"adjudication must cover exactly the disputes in {key}")
        require(all(value in ("P", "F", "U") for value in item["decisions"].values()) and item["notes"].strip(),
                f"invalid adjudication {key}")
        overrides[key] = item
    require(overrides.keys() == {key for key in first if any(first[key][f] != second[key][f] for f in DECISIONS)},
            "some rating differences are missing adjudication")
    return overrides


def record_outcome(cases, reviewer1, reviewer2, reconciled_parent):
    fields = sorted(DECISIONS)
    for index in sorted(reviewer1):
        label, candidate = index
        metadata = cases[label]
        first, second = reviewer1[index], reviewer2[index]
        observed = {field: [first[field], second[field]] for field in fields}
        item = {"id": label, "candidate": candidate, "phase": metadata["phase"],
                "attribute": metadata["attribute"], "case": metadata["case"],
                "variant": metadata["mapping"][candidate], "ratings": observed,
                "disagreements": [field for field in fields if first[field] != second[field]],
                "evidence_notes": [first["notes"], second["notes"]]}
        item["adjudicated"] = {field: values[0] if values[0] == values[1] else "U"
                               for field, values in observed.items()}
        if reconciled_parent and index in reconciled_parent:
            outcome = reconciled_parent[index]
            item["adjudicated"].update(outcome["decisions"])
            item["adjudication_note"] = outcome["notes"]
        yield item


def summarize(args):
    root = args.output
    cases = read_json(root / "group-key.json")["cases"]
    valid = {(key, candidate) for key, value in cases.items() for candidate in value["mapping"]}
    reviewer1 = read_ratings(args.reviewer1, valid)
    reviewer2 = read_ratings(args.reviewer2, valid)
    adjudicator = read_adjudication(args.adjudication, reviewer1, reviewer2) if args.adjudication else None
    records = list(record_outcome(cases, reviewer1, reviewer2, adjudicator))
    statistics = {}
    disagreements = Counter()
    for item in records:
        result = item["adjudicated"]
        group = statistics.setdefault(item["phase"], {}).setdefault(item["attribute"], {}).setdefault(
            item["variant"], {"n": 0, "target_hits": 0, "stable_hits": 0, "strict_original_body": 0,
                              "strict_allowed_suv_geometry": 0, "protected_failures": Counter()})
        group["n"] += 1
        group["target_hits"] += result["target"] == PASS
        stable = result["target"] == result["stable"] == PASS
        group["stable_hits"] += stable
        protected = all(result[key] == PASS for key in ("scene", "camera", "other"))
        group["strict_original_body"] += stable and protected and result["body"] == PASS
        group["strict_allowed_suv_geometry"] += stable and protected and result["body_allow_suv_geometry"] == PASS
        group["protected_failures"].update(field for field in ("body", "scene", "camera", "other")
                                          if result[field] != PASS)
        disagreements.update(item["disagreements"])
    write_json(args.report or root / "report.json", {
        "reviewers": "Two independent AI evaluators with masked intervention identities",
        "resolved_with": "separate visual adjudication" if adjudicator else "consensus, with disagreements indeterminate",
        "outcomes": statistics, "n_candidate_ratings_per_reviewer": len(records),
        "reviewer_disagreements": dict(disagreements),
        "samples": records,
    })
    print(json.dumps(statistics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ffmpeg", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(handler=prepare)
    s = sub.add_parser("summarize")
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--reviewer1", type=Path, required=True)
    s.add_argument("--reviewer2", type=Path, required=True)
    s.add_argument("--adjudication", type=Path)
    s.add_argument("--report", type=Path)
    s.set_defaults(handler=summarize)
    for name, handler in (("temporal-audit", temporal_audit), ("temporal-cues", temporal_cues)):
        t = sub.add_parser(name)
        t.add_argument("--output", type=Path, required=True)
        t.add_argument("--ffmpeg", required=True)
        t.set_defaults(handler=handler)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
