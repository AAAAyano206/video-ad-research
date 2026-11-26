import json
import os
import shutil
import subprocess
from pathlib import Path

from .conditioning import PAYLOADS, verify_replay_package
from .files import LabError, identifier, integer, require, sha256, write_json, write_new


def replay(job, package_root, bridge_script, bridge_root, worker_python, results_dir, timeout):
    job_id = identifier(job["id"], "job id")
    require(job["variant"] in {"A0", "A1", "A2"}, f"{job_id}: see the plan for its required external step")
    manifest = verify_replay_package(package_root, job)
    settings = job["settings"]
    duration = integer(settings["duration"], "duration", 4, 15)
    width = integer(settings["width"], "width")
    height = integer(settings["height"], "height")
    steps = integer(settings["steps"], "steps")
    timeout = integer(timeout, "bridge timeout")
    require(type(settings["reference_model"]) is bool, "invalid reference_model")
    bridge_root = Path(bridge_root).resolve(strict=True)
    bridge_script = Path(bridge_script).resolve(strict=True)
    worker_python = Path(worker_python).absolute()
    require(worker_python.is_file() and os.access(worker_python, os.X_OK), f"Python isn't executable: {worker_python}")
    evidence = bridge_root / "evidence"
    require(evidence.is_dir(), f"missing bridge evidence directory: {evidence}")
    for suffix in ("request.json", "history.json"):
        require(not (evidence / f"{job_id}.{suffix}").exists(), f"bridge evidence already exists for {job_id}")
    command = [
        str(worker_python), str(bridge_script), "replay",
        "--package", job["package"], "--run", job_id, "--seed", str(job["seed"]),
        "--duration", str(duration), "--width", str(width), "--height", str(height),
        "--steps", str(steps), "--timeout", str(timeout),
    ]
    if settings["reference_model"]:
        command.append("--reference-model")
    env = dict(os.environ, H3_EXPERIMENT_BASE=str(bridge_root), LIENYU_H3_PACKAGE_ROOT=str(Path(package_root).resolve()))
    result = Path(results_dir).resolve() / job_id
    result.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": 1,
        "job_id": job_id,
        "attribute_id": job["attribute_id"],
        "variant": job["variant"],
        "seed": job["seed"],
        "settings": settings,
        "package": job["package"],
        "base_package": job["base_package"],
        "payloads": {stem: manifest["files"][stem] for stem in PAYLOADS},
        "vector_edit": manifest.get("vehicle_edit_lab") if job["variant"] != "A0" else None,
    }
    write_json(result / "started.json", record)

    try:
        run = subprocess.run(
            command, cwd=bridge_root, env=env, capture_output=True, text=True, timeout=timeout + 60,
        )
        (result / "bridge.stdout.log").write_text(run.stdout, encoding="utf-8")
        (result / "bridge.stderr.log").write_text(run.stderr, encoding="utf-8")
        require(run.returncode == 0, f"bridge exited with code {run.returncode}")
        lines = run.stdout.splitlines()
        require(bool(lines), "bridge returned no JSON receipt")
        output = json.loads(lines[-1])
        require(output.get("run") == job_id and output.get("mode") == "replay", "bridge returned a different run")
        source = Path(output["video"]).resolve(strict=True)
        require(source.is_file() and source.suffix.lower() == ".mp4", "bridge returned an invalid MP4 path")
        video = result / "video.mp4"
        write_new(video, lambda staged: shutil.copyfile(source, staged))
        digest = sha256(video)
        require(digest == output["sha256"], "copied video does not match the bridge receipt")
        receipt = result / "receipt.json"
        record.update(
            status="completed", video=video.name, video_sha256=digest,
            elapsed_s=output["elapsed_s"], prompt_id=output["prompt_id"],
        )
        write_json(receipt, record)
        return receipt
    except (OSError, ValueError, KeyError, TypeError, AttributeError, subprocess.TimeoutExpired) as error:
        write_json(result / "failure.json", {"job_id": job_id, "error": str(error)})
        raise LabError(f"{error}; inspect {result} and the worker before retrying") from error
