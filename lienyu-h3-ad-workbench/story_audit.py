"""Check exact matched-arm media, record measured end-to-end costs, and extract seam evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
EXPERIMENT = json.loads((ROOT / "experiment.json").read_text())


def read(folder):
    return json.loads((folder / "receipt.json").read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decoded(ffmpeg, path, stream):
    options = ["-pix_fmt", "rgb24", "-vcodec", "rawvideo"] if stream == "v" else ["-acodec", "pcm_s16le"]
    return subprocess.check_output([ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-map", f"0:{stream}:0",
                                    *options, "-f", "hash", "-hash", "sha256", "-"], text=True).strip()


def receipt_and_check(folder):
    record = read(folder)
    video = folder / "video.mp4"
    if digest(video) != record["video_sha256"]:
        raise ValueError(f"video checksum failed for {folder}")
    return video, record


def capture(source, edited, full, args):
    path = ROOT / "results" / "story" / "aligned.jpg"
    times = ("0.25", "4.75", "9.75", "14.50", "14.958333", "15.00",
             "15.50", "20.00", "25.00", "29.50", "30.25", "35.50")
    command = [args.ffmpeg, "-y", "-v", "error", "-nostdin"]
    for video in (source, edited, full):
        for value in times:
            command.extend(("-threads", "1", "-ss", value, "-i", str(video)))
    width, height, columns = 180, 102, 6
    n = len(times) * 3
    layout = "|".join(f"{i%columns*width}_{i//columns*height}" for i in range(n))
    filters = ";".join(f"[{i}:v]scale={width}:{height}[v{i}]" for i in range(n)) + ";"
    filters += "".join(f"[v{i}]" for i in range(n)) + f"xstack=inputs={n}:layout={layout}[picture]"
    subprocess.run(command + ["-filter_complex_threads", "2", "-filter_complex", filters, "-map", "[picture]", "-frames:v", "1", str(path)], check=True)
    return str(path.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    story = ROOT / "results" / "story"
    runs = ROOT / "results" / "runs"
    masks, shots, stats = {}, {}, {}
    for arm in ("base", "full", "selective"):
        assembly_dir = story / arm
        assembly = read(assembly_dir)
        if digest(assembly_dir / "final.mp4") != assembly["final"]["sha256"]:
            raise ValueError("Hypit final hash differs")
        if assembly["hypit_provider_requests"] or assembly["hypit_h3_calls"]:
            raise ValueError("Hypit unexpectedly called a model provider")
        stats[arm] = {"h3_calls": 0, "gpu_requests_elapsed_s": 0, "postprocess_elapsed_s": 0,
                      "hypit_compose_elapsed_s": assembly["timings_s"]["compose"],
                      "hypit_operations": len(assembly["executed_local_operations"]),
                      "composition_audio_checks": [v["audio_check"]["matched"] for v in assembly["shots"]]}
        shots[arm] = {}
        for clip, item in zip(assembly["shots"], EXPERIMENT["story"]):
            name = item["id"]
            if arm == "full":
                edit, generate = ("full-parts" if item["affected"] else None), "full"
            elif arm == "selective":
                edit, generate = ("parts", "donor") if item["affected"] else (None, None)
            else:
                edit, generate = None, "base"

            video, input_receipt = receipt_and_check(runs / f"{name}--{edit or generate or 'base'}")
            if clip["sha256"] != input_receipt["video_sha256"]:
                raise ValueError(f"wrong media bound to Hypit slot {name}")
            if generate:
                folder = runs / f"{name}--{generate}"
                _, gpu = receipt_and_check(folder)
                if arm != "base" and gpu["replay_received_prompt"] is not False:
                    raise ValueError("replay received original prompt")
                history = json.loads((folder / "history.json").read_text())
                events = history["status"]["messages"]
                if any(e[0] == "execution_cached" and "12" in e[1].get("nodes", []) for e in events):
                    raise ValueError(f"cached sampling invalidates cost trial: {folder.name}")
                stats[arm]["h3_calls"] += 1
                stats[arm]["gpu_requests_elapsed_s"] += gpu["elapsed_s"]
            if edit:
                if not input_receipt["source_audio_preserved"]:
                    raise ValueError("part transfer altered source audio")
                masks.setdefault(name, {})[arm] = input_receipt["mask_sha256"]
                stats[arm]["postprocess_elapsed_s"] += input_receipt["elapsed_s"]
            shots[arm][name] = {key: decoded(args.ffmpeg, video, key) for key in ("v", "a")}
        stats[arm]["serial_elapsed_s"] = sum(value for name, value in stats[arm].items() if name.endswith("elapsed_s"))
    finals = {arm: {s: decoded(args.ffmpeg, story / arm / "final.mp4", s) for s in ("v", "a")} for arm in stats}
    paired = {"shot_decoded_streams_equal": shots["full"] == shots["selective"],
              "final_decoded_streams_equal": finals["full"] == finals["selective"],
              "mask_hashes_equal": all(x["full"] == x["selective"] for x in masks.values())}
    result = {"paths": stats, "matched_arm_evidence": paired, "per_shot": shots, "final_hashes": finals,
              "aligned_still_grid": capture(*(story / arm / "final.mp4" for arm in ("base", "selective", "full")), args),
              "elapsed_definition": "Sum of H3 request waits, CPU paint pass, and local Hypit plan/build/export. Excludes base asset preparation, transfers, and human review."}
    (story / "cost-and-equivalence.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"paths": stats, "matched_arm_evidence": paired}, indent=2))


if __name__ == "__main__":
    main()
