"""Resume the vehicle-edit matrix from verified packages and per-job receipts."""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from .conditioning import learn_direction, open_package, prepare_package, tensor_tools, verify_replay_package
from .files import read_json, require, sha256, write_json, write_new
from .modelarts import ModelArts
from .planning import make_plan


LAB = Path(__file__).resolve().parents[1]
MATRIX = read_json(LAB / "experiments/matrix.json")
CONFIG = read_json(LAB / "experiments/modelarts-cases.json")
SETTINGS = {"duration": 6, "width": 384, "height": 256, "steps": 4, "reference_model": False}
KEY = "t000000"


def sample(name, seed, settings):
    return {"id": name, "seed": seed, "settings": settings}


def save_plan(plan, path):
    if path.exists():
        require(read_json(path) == plan, f"existing plan differs from frozen spec: {path}")
    else:
        write_json(path, plan)


def freeze_direction(host, attribute):
    historical = LAB / "outputs/historical-color-run"
    name = CONFIG["attributes"][attribute]["short"]
    if name == "paint":
        return historical / "color.safetensors"
    path = host.inputs / f"direction-{name}.safetensors"
    if path.exists():
        return path
    if name == "body":
        pairs = [(f"gen-donor-car-{n}-s", f"gen-donor-car-{n}-t") for n in range(3)]
        guards = ("color", "scene", "person")
    else:
        pairs = []
        cfg = CONFIG["attributes"][attribute]
        guards = ("color", "car")
        for n, template in enumerate(cfg["donors"]):
            pair = []
            for side in ("from", "to"):
                package = f"hypit-donor-{name}-{n}-{side}"
                host.execute(host.inputs / package, sample(package, 8000 + n, SETTINGS), package,
                             prompt=template.format(attribute=cfg[side]), encode_only=True)
                pair.append(package)
            pairs.append(pair)
    learn_direction(host.packages, pairs, [historical / f"{other}.safetensors" for other in guards],
                    attribute, KEY, path)
    return path


def replay(host, job, direction):
    directory = host.results / job["id"]
    if job["variant"] in ("A1", "A2") and not (host.packages / job["package"] / "manifest.json").is_file():
        prepare_package(host.packages, job, direction, CONFIG["alpha"], KEY)
    manifest = verify_replay_package(host.packages, job)
    receipt = host.execute(directory, job, job["package"], extra={
        "vector_edit": manifest.get("vehicle_edit_lab") if job["variant"] in ("A1", "A2") else None,
    })
    host.copy_replay_evidence(job["id"])
    return receipt


def reference_image(host, attribute):
    cfg = CONFIG["attributes"][attribute]
    name = f"hypit-ref-{cfg['short']}"
    directory = host.inputs / name
    seed = 8610 + list(CONFIG["attributes"]).index(attribute)
    host.execute(directory, sample(name, seed, SETTINGS), name, prompt=cfg["reference"])
    return host.capture(name, directory / "video.mp4", timestamp=2)


def run_reference(host, job):
    baseline = host.results / job["baseline_job"] / "video.mp4"
    cfg = CONFIG["attributes"][job["attribute_id"]]
    clip = host.inputs / f"{job['baseline_job']}-ref.mp4"
    if not clip.exists():
        write_new(clip, lambda target: subprocess.run([
            "ffmpeg", "-v", "error", "-nostdin", "-i", str(baseline), "-t", "2.25",
            "-an", "-vf", "fps=24", "-c:v", "mpeg4", "-q:v", "4", str(target),
        ], check=True))
    image = reference_image(host, job["attribute_id"])
    ref_job = {**job, "settings": {**job["settings"], "reference_model": True}}
    host.execute(host.results / job["id"], ref_job, job["id"], media=(
        ("reference_video", clip), ("reference_image", image),
    ), prompt=(
        f"Preserve the shot, vehicle identity except for the specified change, camera movement and scene from "
        f"<Video 1>. Change only {cfg['from']} to {cfg['to']}, using <Picture 1> as a visual guide for that "
        "attribute. Keep this change stable throughout. Premium photorealistic automotive advertisement."
    ), extra={"method": "Ref2VA with independent H3-generated target still and 2.25 s of baseline video"})


def run_history(host):
    source = read_json(LAB / "experiments/spec.historical-color.json")
    plan = make_plan(MATRIX, source, "pilot")
    for job in plan["jobs"]:
        if job["variant"] == "A4":
            run_reference(host, job)
        else:
            host.copy_replay_evidence(job["id"])


def run_attribute(host, name, phase, part):
    attribute, cfg = next((key, values) for key, values in CONFIG["attributes"].items() if values["short"] == name)
    direction = freeze_direction(host, attribute)
    settings = dict(SETTINGS, width=640, height=384) if name in ("wheel", "window") else SETTINGS
    phase_offset = 100 if phase == "holdout" else 0
    attr_index = list(CONFIG["attributes"]).index(attribute)
    shots, prompts = [], {}
    for index, (shot, template) in enumerate(cfg[phase].items()):
        shots.append({"id": shot, "phase": phase, "attribute_id": attribute, "settings": settings, "takes": []})
        for n in range(3):
            seed = 9101 + phase_offset + 20 * attr_index + 3 * index + n
            package = f"hypit-{phase}-{name}-{shot}-{seed}"
            shots[-1]["takes"].append({"id": f"seed-{seed}", "seed": seed, "package": package})
            prompts[package] = template.format(attribute=cfg["from"])
    plan = make_plan(MATRIX, {"schema_version": 1, "shots": shots}, phase)
    plan["frozen_direction_sha256"] = sha256(direction)
    plan["alpha"] = CONFIG["alpha"]
    plan["baseline_prompts"] = prompts
    save_plan(plan, LAB / f"outputs/modelarts-plans/{phase}-{name}.json")
    for job in plan["jobs"]:
        if job["variant"] == "A4":
            if part == "refs":
                run_reference(host, job)
        elif part == "core":
            if job["variant"] == "A0":
                name = job["base_package"]
                host.execute(host.inputs / name, sample(name, job["seed"], settings), name, prompt=prompts[name])
            replay(host, job, direction)
    if phase == "pilot" and part == "core":
        reference_image(host, attribute)


def first_frame_entry(path, manifest):
    tree = read_json(path / manifest["files"]["conditioning"]["tree_file"])
    keyframes = tree["items"][0]["items"][1]["items"]["minimax_keyframes"]["items"]
    require(len(keyframes) == 1 and keyframes[0]["items"]["resolved_frame_index"]["value"] == 0,
            "expected a single frame-zero keyframe in FL2VA package")
    return keyframes[0]["items"]["latent"]["key"]


def swap_first_frame(host, job, donor, keyframe):
    destination = host.packages / job["id"]
    source, source_manifest = open_package(host.packages, job["vector_source_job"])
    frame_package, frame_manifest = open_package(host.packages, donor, ("conditioning",))
    existing = destination / "manifest.json"
    source_key = first_frame_entry(source, source_manifest)
    frame_key = first_frame_entry(frame_package, frame_manifest)
    _, safe_open, save_file = tensor_tools()
    with safe_open(source / source_manifest["files"]["conditioning"]["tensor_file"], framework="pt",
                   device="cpu") as f:
        tensors, metadata = {key: f.get_tensor(key) for key in f.keys()}, f.metadata()
    with safe_open(frame_package / frame_manifest["files"]["conditioning"]["tensor_file"], framework="pt",
                   device="cpu") as f:
        updated = f.get_tensor(frame_key)
    original = tensors[source_key]
    require(original.shape == updated.shape and original.dtype == updated.dtype, "first-frame shape/dtype changed")
    require(not original.equal(updated), "encoded replacement frame equals the source")
    tensors[source_key] = updated
    provenance = {
        "parent_job": job["first_frame_from_job"], "source_condition_package": source.name,
        "new_frame_sha256": sha256(keyframe), "new_frame_package": donor,
        "replaced_condition_tensor": source_key, "qwen_condition_tensor_unchanged": True,
        "separate_image_keyframe_latent_only": True,
    }
    if not existing.exists():
        with tempfile.TemporaryDirectory(prefix=".hypit-frame-", dir=host.packages) as temporary:
            staged = Path(temporary) / destination.name
            shutil.copytree(source, staged)
            entry = source_manifest["files"]["conditioning"]
            tensor_path = staged / entry["tensor_file"]
            save_file(tensors, tensor_path, metadata=metadata)
            entry["tensor_sha256"] = sha256(tensor_path)
            source_manifest["first_frame_sync"] = provenance
            (staged / "manifest.json").unlink()
            write_json(staged / "manifest.json", source_manifest)
            staged.rename(destination)
    _, repaired = open_package(host.packages, job["id"])
    _, original_manifest = open_package(host.packages, job["vector_source_job"])
    require(repaired["first_frame_sync"] == provenance, "existing A3 package has other provenance")
    for name in ("latent", "noise"):
        require(repaired["files"][name] == original_manifest["files"][name], f"A3 must preserve {name}")
    with safe_open(destination / repaired["files"]["conditioning"]["tensor_file"], framework="pt", device="cpu") as f:
        require(set(f.keys()) == set(tensors) and all(f.get_tensor(key).equal(value) for key, value in tensors.items()),
                "A3 package changed a non-keyframe tensor or has a different replacement")
    return provenance


def run_continuity(host):
    attribute = "vehicle.paint_color"
    settings = dict(SETTINGS, duration=15)
    direction = freeze_direction(host, attribute)
    shots = [
        {"phase": "continuity", "id": f"story-{index}", "attribute_id": attribute, "settings": settings,
         "takes": [{"id": "coastal-chain", "seed": seed, "package": package}],
         **({"first_frame_from": "story-1"} if index == 2 else {})}
        for index, seed, package in ((1, 5101, "ad15-base"), (2, 7002, "story-shot2"))
    ]
    plan = make_plan(MATRIX, {"schema_version": 1, "shots": shots}, "continuity")
    save_plan(plan, LAB / "outputs/modelarts-plans/continuity.json")
    jobs = {job["id"]: job for job in plan["jobs"]}
    for job in jobs.values():
        if job["variant"] in ("A0", "A1"):
            replay(host, job, direction)
        elif job["execution"] == "reuse":
            source = host.results / job["reuse_job"]
            target = host.results / job["id"]
            if not (target / "video.mp4").exists():
                write_new(target / "video.mp4", lambda staged: shutil.copyfile(source / "video.mp4", staged))
            output = read_json(source / "receipt.json")
            require(sha256(target / "video.mp4") == output["video_sha256"], "reused clip hash mismatch")
            if not (target / "receipt.json").exists():
                write_json(target / "receipt.json", {**output, "job_id": job["id"], "variant": "A3",
                           "reused_job": job["reuse_job"], "elapsed_s": 0, "generation_calls": 0})
        else:
            parent = jobs[job["first_frame_from_job"]]
            frame = host.capture("hypit-chain-a3-blue-last", host.results / parent["id"] / "video.mp4")
            donor = "hypit-chain-a3-frame"
            host.execute(host.inputs / donor, sample(donor, job["seed"], settings), donor,
                         prompt="A video frame for VAE keyframe conditioning.",
                         media=(("first_frame", frame),), encode_only=True)
            provenance = swap_first_frame(host, job, donor, frame)
            host.execute(host.results / job["id"], job, job["id"], extra={"first_frame_sync": provenance})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("history", "continuity", "pilot", "holdout"))
    parser.add_argument("attribute", nargs="?", choices=("paint", "body", "wheel", "window"))
    parser.add_argument("--part", choices=("core", "refs"), default="core")
    args = parser.parse_args()
    host = ModelArts(LAB)
    if args.stage == "history":
        run_history(host)
    elif args.stage == "continuity":
        run_continuity(host)
    else:
        if not args.attribute:
            parser.error("pilot and holdout require a vehicle attribute")
        run_attribute(host, args.attribute, args.stage, args.part)


if __name__ == "__main__":
    main()
