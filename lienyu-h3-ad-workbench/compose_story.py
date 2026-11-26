"""Bind this experiment's planned media edits to the Hypit template."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SHOT_ROLES = {
    "base": ("base", "base"),
    "selective": ("parts", "base"),
    "full": ("full-parts", "full"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=SHOT_ROLES)
    args = parser.parse_args()
    experiment = json.loads((ROOT / "experiment.json").read_text())
    affected, unaffected = SHOT_ROLES[args.arm]
    clips = [
        ROOT / "results/runs" / f"{shot['id']}--{affected if shot['affected'] else unaffected}" / "video.mp4"
        for shot in experiment["story"]
    ]
    subprocess.run([
        sys.executable, str(ROOT / "compose.py"),
        "--output-dir", str(ROOT / "results" / "story" / args.arm),
        "--title", f"River Causeway - {args.arm}", *(str(path) for path in clips),
    ], check=True)


if __name__ == "__main__":
    main()
