import hashlib
from collections import defaultdict

from .files import identifier, integer, require


VARIANTS = {
    "A0": "baseline_replay",
    "A1": "vector_only",
    "A2": "norm_matched_random_direction",
    "A3": "vector_with_first_frame_sync",
    "A4": "reference_media_regeneration",
}


def take_specs(shot, phase):
    takes = {}
    seeds = set()
    for take in shot["takes"]:
        name = identifier(take["id"], "take id", 24)
        seed = integer(take["seed"], "H3 seed", 0, 2**64 - 1)
        require(name not in takes and seed not in seeds, f"duplicate take id or seed in {shot['id']}")
        seeds.add(seed)
        takes[name] = {"seed": seed, "package": identifier(take["package"], "base package")}
    require(bool(takes), f"no takes in {shot['id']}")
    if "seeds_per_shot" in phase:
        require(len(takes) == phase["seeds_per_shot"], f"unexpected take count in {shot['id']}")
    return takes


def shot_specs(shots, phase_name, phase):
    result = {}
    for shot in shots:
        name = identifier(shot["id"], "shot id", 32)
        require(name not in result, f"duplicate shot id: {name}")
        parent = shot.get("first_frame_from")
        require(
            parent is None or phase_name == "continuity" and parent in result,
            f"first_frame_from in {name} must name an earlier continuity shot of the same attribute",
        )
        takes = take_specs(shot, phase)
        if parent:
            require(takes.keys() == result[parent]["takes"].keys(), f"unpaired takes for {parent} and {name}")
        settings = shot["settings"]
        reference_model = settings.get("reference_model", False)
        require(type(reference_model) is bool, f"{name}: reference_model must be a boolean")
        result[name] = {
            "parent": parent,
            "takes": takes,
            "settings": {
                "duration": integer(settings["duration"], "duration", 4, 15),
                "width": integer(settings["width"], "width"),
                "height": integer(settings["height"], "height"),
                "steps": integer(settings["steps"], "steps"),
                "reference_model": reference_model,
            },
        }
    count = phase.get("shot_count", phase.get("shots_per_attribute"))
    require(count is None or len(result) == count, f"{phase_name} requires {count} shots per selected attribute")
    if phase_name == "continuity":
        names = list(result)
        for previous, current in zip(names, names[1:]):
            require(result[current]["parent"] == previous, f"{current} must depend on {previous}'s last frame")
    return result


def make_plan(matrix, spec, phase_name, attribute_filter=None):
    require(matrix["schema_version"] == 1 and spec["schema_version"] == 1, "unsupported schema version")
    identifier(phase_name, "phase", 16)
    require(phase_name in matrix["phases"], f"unknown phase: {phase_name}")
    phase = matrix["phases"][phase_name]
    variants = phase["variants"]
    require(
        len(set(variants)) == len(variants)
        and all(matrix["variants"].get(v) == VARIANTS.get(v) and v in VARIANTS for v in variants)
        and "A0" in variants and ("A3" not in variants or "A1" in variants),
        "unsupported variant configuration",
    )
    attributes = {a["id"]: a for a in matrix["attributes"]}
    require(attribute_filter is None or attribute_filter in attributes, f"unknown attribute: {attribute_filter}")
    by_attribute = defaultdict(list)
    for shot in spec["shots"]:
        if shot["phase"] != phase_name:
            continue
        attribute = identifier(shot["attribute_id"], "attribute id", 36)
        require(attribute in attributes, f"unknown attribute in shots: {attribute}")
        if attribute_filter is None or attribute == attribute_filter:
            by_attribute[attribute].append(shot)
    require(bool(by_attribute), f"no matching shots configured for {phase_name}")

    jobs = []
    for attribute, shots in by_attribute.items():
        shots_by_id = shot_specs(shots, phase_name, phase)
        prefix = f"{phase_name}--{attribute}"
        for shot_id, shot in shots_by_id.items():
            parent = shot["parent"]
            for take_id, take in shot["takes"].items():
                case = f"{prefix}--{shot_id}--{take_id}"
                for variant in variants:
                    job_id = identifier(f"{case}--{variant}", "job id")
                    job = {
                        "id": job_id, "case_id": case, "attribute_id": attribute, "variant": variant,
                        "strategy": VARIANTS[variant], "shot_id": shot_id, "take_id": take_id,
                        "seed": take["seed"], "base_package": take["package"], "settings": shot["settings"],
                        "baseline_job": f"{case}--A0",
                    }
                    if variant == "A0":
                        job.update(execution="replay", package=take["package"])
                    elif variant in {"A1", "A2"}:
                        job.update(execution="prepare_then_replay", package=job_id)
                        if variant == "A2":
                            job["vector_source_job"] = f"{case}--A1"
                            job["random_direction_seed"] = int.from_bytes(
                                hashlib.sha256(job_id.encode()).digest()[:8], "big",
                            )
                    elif variant == "A3" and parent is None:
                        job.update(execution="reuse", reuse_job=f"{case}--A1")
                    elif variant == "A3":
                        job.update(
                            execution="external_first_frame_sync",
                            first_frame_from_job=f"{prefix}--{parent}--{take_id}--A3",
                            vector_source_job=f"{case}--A1",
                        )
                    else:
                        job.update(execution="external_reference_generation")
                    jobs.append(job)
    require(len({job["id"] for job in jobs}) == len(jobs), "job IDs collide; use distinct shot and take IDs")
    return {"schema_version": 1, "phase": phase_name, "attributes": list(by_attribute), "jobs": jobs}


def get_job(plan, job_id):
    require(plan["schema_version"] == 1, "unsupported plan version")
    matches = [job for job in plan["jobs"] if job["id"] == job_id]
    require(len(matches) == 1, f"expected one job with id {job_id}")
    return matches[0]
