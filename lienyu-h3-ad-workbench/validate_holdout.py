"""Audit each holdout pairing before unblinding."""
import argparse
import hashlib
import json
from pathlib import Path

from story_audit import decoded


ROOT = Path(__file__).resolve().parent
EXPECTED = ("base", "replay", "target", "random", "legacy", "parts", "parts-random")


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def load(folder):
    video = folder / "video.mp4"
    receipt = json.loads((folder / "receipt.json").read_text())
    if digest(video) != receipt["video_sha256"]:
        raise ValueError(f"bad output hash for {folder.name}")
    return video, receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    key = json.loads((ROOT / "results/review/group-key.json").read_text())["cases"]
    folders = ROOT / "results/runs"
    evidence = {}
    for data in key.values():
        case = data["case"]
        inputs = {role: folders / f"{case}--{role}" for role in EXPECTED}
        entries = {role: load(folder) for role, folder in inputs.items()}
        base_video, base = entries["base"]
        reference_audio = decoded(args.ffmpeg, base_video, "a")
        for role in ("replay", "target", "random"):
            receipt = entries[role][1]
            if receipt["replay_received_prompt"] is not False:
                raise ValueError(f"unexpected prompt receipt for {case}/{role}")
            graph = json.loads((inputs[role] / "request.json").read_text())["prompt"]
            if any("prompt" in node["inputs"] or "TextEncode" in node["class_type"] for node in graph.values()):
                raise ValueError(f"unexpected text encoder or prompt input for {case}/{role}")
            if any(receipt["payloads"][k] != base["payloads"][k] for k in ("noise", "latent")):
                raise ValueError(f"changed initial state for {case}/{role}")
        replay = entries["replay"][0]
        if any(decoded(args.ffmpeg, base_video, s) != decoded(args.ffmpeg, replay, s) for s in ("v", "a")):
            raise ValueError(f"non-exact replay for {case}")
        lengths = [entries[role][1]["conditioning_edit"]["actual_change_l2"] for role in ("target", "random")]
        relative_error = abs(lengths[0] - lengths[1]) / lengths[0]
        if relative_error > .001:
            raise ValueError(f"unmatched edit strengths for {case}")
        for role in ("legacy", "parts", "parts-random"):
            video, receipt = entries[role]
            source_field = "source_video_sha256" if role == "legacy" else "source_sha256"
            if (receipt[source_field] != base["video_sha256"]
                    or decoded(args.ffmpeg, video, "a") != reference_audio):
                raise ValueError(f"source dependency or audio changed for {case}/{role}")
            mask = inputs[role] / ("masks.safetensors" if role == "legacy" else "masks.npz")
            if digest(mask) != receipt["mask_sha256"]:
                raise ValueError(f"bad source mask for {case}/{role}")
        parts, random = (entries[role][1] for role in ("parts", "parts-random"))
        if not random["source_mask_reused"] or parts["mask_sha256"] != random["mask_sha256"]:
            raise ValueError(f"part mask not identical in same-source control for {case}")
        evidence[case] = {"exact_replay": True, "original_noise_and_latent": True, "generator_prompt": False,
                          "relative_target_random_edit_norm_error": relative_error, "parts_audio_exact": True,
                          "target_hue": parts["target_hue"], "random_hue": random["target_hue"]}
    out = ROOT / "results" / "holdout-audit.json"
    out.write_text(json.dumps({"verified_cases": len(evidence), "cases": evidence}, indent=2) + "\n")
    print(f"Verified all {len(evidence)} held-out case runs and their physical pairings.")


if __name__ == "__main__":
    main()
