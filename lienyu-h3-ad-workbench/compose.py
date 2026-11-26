"""Assemble H3 shot files with Hypit and save auditable output evidence."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from array import array
from fractions import Fraction
from pathlib import Path
from string import Template
from urllib.parse import unquote
from xml.etree import ElementTree as ET


TEMPLATE_DIR = Path(__file__).resolve().parent / "hypit"
KNOWN_HYPIT = Path("/private/tmp/h3-hypit-cli.7mmRoE")
LOCAL_ENDPOINTS = {"media.local", "html.local"}
TARGET = "final.video"


def executable(explicit, name, fallback):
    selected = explicit or shutil.which(name) or str(fallback)
    path = Path(shutil.which(selected) or selected).expanduser().absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Cannot execute {name} at {path}. Supply --{name}.")
    return path


def digest(path):
    hashed = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hashed.update(chunk)
    return hashed.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def probe(ffprobe, path):
    run = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        check=True, capture_output=True, text=True, timeout=40,
    )
    info = json.loads(run.stdout)
    video = next((item for item in info["streams"] if item["codec_type"] == "video"
                  and not item.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        raise ValueError(f"Missing moving video stream in {path}")
    audio = next((item for item in info["streams"] if item["codec_type"] == "audio"), None)
    duration = video.get("duration") or info["format"]["duration"]
    return {
        "path": str(path),
        "sha256": digest(path),
        "duration": float(duration),
        "fps": video.get("avg_frame_rate"),
        "frames": int(video["nb_frames"]) if video.get("nb_frames", "N/A").isdigit() else None,
        "width": video["width"],
        "height": video["height"],
        "audio": None if audio is None else {
            "codec": audio["codec_name"],
            "duration": float(audio.get("duration") or info["format"]["duration"]),
            "rate": int(audio["sample_rate"]),
            "channels": audio["channels"],
        },
    }


def write_project(directory, shots, args, ffmpeg, ffprobe):
    windows, picture, sound, declarations = [], [], [], []
    svrun = ET.Element("svrun", {"version": "1"})
    ET.SubElement(svrun, "author", {"source": "./ad.svml"})
    ET.SubElement(svrun, "target", {"output": TARGET})
    for index, shot in enumerate(shots, start=1):
        sid = f"shot-{index:02}"
        mid = f"media-{index:02}"
        wid = f"window-{index:02}"
        start = "start" if index == 1 else f"window-{index-1:02}.end"
        audio_mode = "default" if shot["audio"] else "none"
        declarations.append(f'  <h3:TextVideo id="{sid}" prompt={{slot-note}} duration="15" aspect-ratio="16:9"/>')
        declarations.append(f'  <mediaop:Normalize id="{mid}" source={{{sid}.video}} clock={{clock}} '
                            f'video="primary-moving" audio="{audio_mode}" span-authority="video"/>')
        windows.append(f'    <time:Window id="{wid}" from="{start}" for={{{mid}.extent}}/>')
        picture.append(f'    <visual:Clip id="{sid}" media={{{mid}.media}} during={{cut.{wid}}} '
                       'frame={full} fit="contain" z="0"/>')
        if shot["audio"]:
            sound.append(f'    <audio:Clip id="{sid}" source={{{mid}.media}} during={{cut.{wid}}}/>')
        ET.SubElement(svrun, "file", {"id": f"source-{index:02}", "type": "@hypit/blob@1#Blob",
                                    "from": os.path.relpath(shot["path"], directory),
                                    "media-type": "video/mp4"})
        ET.SubElement(svrun, "satisfy", {"output": f"{sid}.video", "candidate": f"source-{index:02}"})
    template = Template((TEMPLATE_DIR / "ad.svml.in").read_text(encoding="utf-8"))
    audio_track = ('  <audio:Track id="sound" timeline={cut.timeline}>\n' + "\n".join(sound)
                   + '\n  </audio:Track>') if sound else ""
    (directory / "ad.svml").write_text(
        template.substitute(fps=args.fps, width=args.width, height=args.height,
                            shot_declarations="\n".join(declarations), windows="\n".join(windows),
                            picture_clips="\n".join(picture), audio_track=audio_track,
                            film_audio_track='    <film:Track source={sound.audio}/>' if sound else "",
                            timeline_end=f"window-{len(shots):02}.end"), encoding="utf-8",
    )
    shutil.copyfile(TEMPLATE_DIR / "look.svs", directory / "look.svs")
    ET.indent(svrun, space="  ")
    (directory / "ad.svrun").write_text(
        '<?svml using="@hypit/markup/run@1"?>\n' + ET.tostring(svrun, encoding="unicode") + "\n",
        encoding="utf-8",
    )
    save_json(directory / "package.json", {"name": "h3-ad-workbench-assembly", "private": True,
                                          "hypit": {"project": True}})
    codecs = {"ffmpegPath": str(ffmpeg), "ffprobePath": str(ffprobe)}
    html_config = {**codecs, "workers": 2, "defaultConcurrency": 1}
    if args.chrome:
        html_config["chromePath"] = str(Path(args.chrome).expanduser().resolve(strict=True))
    save_json(directory / "runtime.json", {
        "format": "hypit.runtime-local@1",
        "dataRoot": ".hypit-runtime",
        "credentials": {},
        "endpoints": {
            "media.local": {"use": "@hypit/media-local", "config": {**codecs, "defaultConcurrency": 2}},
            "html.local": {"use": "@hypit/provider-html-local", "config": html_config},
        },
        "bindings": {},
    })


def cli(hypit, directory, name, *arguments, timeout=120):
    started = time.perf_counter()
    command = [hypit, *arguments, "--json"]
    run = subprocess.run(command, cwd=directory, text=True, capture_output=True, timeout=timeout)
    if run.stderr:
        (directory / f"{name}.log").write_text(run.stderr, encoding="utf-8")
    (directory / f"{name}.json").write_text(run.stdout, encoding="utf-8")
    try:
        result = json.loads(run.stdout)
    except json.JSONDecodeError:
        result = {}
    if run.returncode:
        detail = result.get("build", {}).get("failure") or result.get("error", {}).get("message")
        detail = detail or (run.stderr or run.stdout).strip()
        raise RuntimeError(f"Hypit {name} failed ({run.returncode}): {detail}")
    if not result:
        raise RuntimeError(f"Hypit {name} returned no machine-readable result")
    return result, round(time.perf_counter() - started, 3)


def audio_samples(ffmpeg, clip, start, length):
    result = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-i", clip, "-ss", f"{start:.6f}", "-t", f"{length:.6f}",
         "-map", "0:a:0", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
        check=True, capture_output=True, timeout=35,
    )
    samples = array("h", result.stdout)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def audio_match(ffmpeg, shot, final, offset):
    duration = shot["duration"]
    length = min(1, duration / 4)
    at = {}
    for ratio in (.25, .5, .75):
        position = duration * ratio - length / 2
        source = audio_samples(ffmpeg, shot["path"], position, length)
        output = audio_samples(ffmpeg, final, position + offset, length)
        if not source or not output:
            raise ValueError("Audio verification could not decode a comparison window")
        energy_source = sum(x * x for x in source) / len(source)
        energy_output = sum(y * y for y in output) / len(output)
        if energy_source < 35 ** 2:
            at[str(ratio)] = {"source_silent": True, "output_silent": energy_output < 70 ** 2}
            continue
        # AAC re-encoding and resampling can shift waveforms by a few samples.
        # Search +/- 20 ms, then compare normalized correlation at the best offset.
        scores = []
        for shift in range(-160, 161):
            x = source[max(shift, 0)::4]
            y = output[max(-shift, 0)::4]
            size = min(len(x), len(y))
            y = y[:size]
            x = x[:size]
            norm = (sum(a * a for a in x) * sum(b * b for b in y)) ** .5
            if norm:
                scores.append(sum(a * b for a, b in zip(x, y)) / norm)
        at[str(ratio)] = {"source_silent": False, "correlation": round(max(scores, default=0), 4)}
    matched = all(item.get("output_silent", item.get("correlation", 0) >= .8) for item in at.values())
    return {"matched": matched, "samples": at}


def compose(args):
    directory = Path(args.output_dir).expanduser().resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Output directory is not empty: {directory}")
    hypit = executable(args.hypit, "hypit", KNOWN_HYPIT / "node_modules/.bin/hypit")
    ffmpeg = executable(args.ffmpeg, "ffmpeg", KNOWN_HYPIT / "ffmpeg")
    ffprobe = executable(args.ffprobe, "ffprobe",
                         KNOWN_HYPIT / "node_modules/@ffprobe-installer/darwin-arm64/ffprobe")
    paths = [Path(item).expanduser().resolve(strict=True) for item in args.clips]
    if any(path.suffix.lower() != ".mp4" or not path.is_file() for path in paths):
        raise ValueError("Expected an ordered list of MP4 files")
    shots = [probe(ffprobe, path) for path in paths]
    directory.mkdir(parents=True, exist_ok=True)
    write_project(directory, shots, args, ffmpeg, ffprobe)
    started = time.perf_counter()
    profile = ["--runtime", "runtime.json"]
    source_scope = []
    for folder in sorted({path.parent for path in paths}):
        source_scope.extend(["--asset-root", str(folder)])
    version, _ = cli(hypit, directory, "version", "version")
    plan, plan_seconds = cli(hypit, directory, "plan", "plan", "ad.svrun", *profile, *source_scope)
    if (not plan["ok"] or plan["providerRequestCount"] != 0 or plan["unresolvedRequestCount"] != 0
            or plan["choiceCount"] != len(shots)
            or any(item.get("endpoint") not in LOCAL_ENDPOINTS for item in plan["providers"])):
        raise RuntimeError("Plan did not resolve every shot exclusively to local media operations")
    try:
        build, build_seconds = cli(hypit, directory, "build", "build", "ad.svrun", *profile, *source_scope,
                                   "--title", args.title, "--follow", "--max-wait-ms",
                                   str(args.timeout * 1000), timeout=args.timeout + 25)
        record = build["build"]
        build_id = record["id"]
        if record["result"]["state"] != "complete":
            raise RuntimeError(f"Hypit result did not complete: {build_id}")
        execution, _ = cli(hypit, directory, "execution", "logs", build_id, *profile, "--lines", "2000")
        completed = [item for item in execution["records"] if item["kind"] == "completed"]
        if execution["omittedRecords"] or len(completed) != plan["localRequestCount"]:
            raise RuntimeError("Execution log does not contain every planned local operation")
        if any(item["endpoint"] not in LOCAL_ENDPOINTS for item in execution["records"]):
            raise RuntimeError("Composition invoked an unplanned non-local endpoint")
        final = directory / "final.mp4"
        _, export_seconds = cli(hypit, directory, "get", "get", build_id, "--output", TARGET, "--to", "final.mp4")
        compose_seconds = round(time.perf_counter() - started, 3)
        output = probe(ffprobe, final)
        offset_frames = 0
        for shot in shots:
            expected = round(shot["duration"] * args.fps)
            if shot["fps"] and Fraction(shot["fps"]) == args.fps and shot["frames"] is not None:
                expected = shot["frames"]
            shot["timeline_start_s"] = offset_frames / args.fps
            shot["timeline_frames"] = expected
            offset_frames += expected
        dimensions = output["width"], output["height"]
        if (output["frames"] != offset_frames or Fraction(output["fps"]) != args.fps
                or dimensions != (args.width, args.height)):
            raise RuntimeError(f"Output has {output['frames']} frames, {output['fps']} fps and size {dimensions}; "
                               f"expected {offset_frames} frames, {args.fps} fps and {(args.width, args.height)}")
        if output["audio"] is None and any(shot["audio"] for shot in shots):
            raise RuntimeError("Composition output lost input audio")
        if output["audio"] and abs(output["audio"]["duration"] - output["duration"]) > 1 / args.fps:
            raise RuntimeError("Output audio and video durations differ by more than one frame")
        for shot in shots:
            if shot["audio"]:
                shot["audio_check"] = audio_match(ffmpeg, shot, final, shot["timeline_start_s"])
                if not shot["audio_check"]["matched"]:
                    raise RuntimeError(f"Audio signal for shot did not match: {shot['path']}")
        receipt = {
            "schema_version": 1, "tool": "Hypit", "tool_version": version["version"], "title": args.title,
            "build_id": build_id, "final": output, "shots": shots, "planned_local_operations": plan["localRequestCount"],
            "executed_local_operations": [{"endpoint": item["endpoint"], "need": unquote(item["command"])}
                                          for item in completed],
            "hypit_provider_requests": plan["providerRequestCount"], "hypit_h3_calls": 0,
            "timings_s": {"plan": plan_seconds, "build": build_seconds, "export": export_seconds,
                          "compose": compose_seconds,
                          "local_execution": (max(item["time"] for item in completed) - min(
                              item["time"] for item in execution["records"] if item["kind"] == "started"
                          )) / 1000},
        }
        save_json(directory / "receipt.json", receipt)
        return receipt
    finally:
        # The profile is private to this build directory, so it can be stopped after output export.
        result = subprocess.run([hypit, "runtime", "down", "runtime.json", "--json"], cwd=directory,
                                timeout=30, text=True, capture_output=True)
        if result.returncode:
            (directory / "runtime-down.log").write_text(result.stderr or result.stdout, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--hypit", default=os.environ.get("HYPIT_CLI"))
    parser.add_argument("--ffmpeg", default=os.environ.get("HYPIT_FFMPEG"))
    parser.add_argument("--ffprobe", default=os.environ.get("HYPIT_FFPROBE"))
    parser.add_argument("--chrome", default=os.environ.get("HYPIT_CHROME"))
    args = parser.parse_args()
    if not 1 <= len(args.clips) <= 4:
        parser.error("Choose one to four ordered clips")
    if min(args.width, args.height, args.fps, args.timeout) < 1:
        parser.error("Width, height, frame rate and timeout must be positive")
    try:
        receipt = compose(args)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
