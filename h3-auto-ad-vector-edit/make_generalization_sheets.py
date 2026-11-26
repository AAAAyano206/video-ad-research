#!/usr/bin/env python3
"""Build one auditable 3-seed contact sheet per story from JSONL results."""
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

BASE = Path("/home/ma-user/work/lienyu/h3-auto-ad-vector-edit")
ROWS = ["base", "color", "car", "scene", "person"]


def frame(video: str, output: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", "3", "-i", video,
         "-frames:v", "1", "-vf", "scale=384:256", str(output)],
        check=True,
    )


def main() -> None:
    records = [json.loads(line) for line in (BASE / "evidence/generalization.jsonl").read_text().splitlines()]
    by_key = {(r["story"], r["seed"], r["variant"]): r for r in records}
    stories = sorted({r["story"] for r in records})
    seeds = sorted({r["seed"] for r in records})
    for story in stories:
        sheet = Image.new("RGB", (3 * 384, 5 * 286), "white")
        draw = ImageDraw.Draw(sheet)
        for row, variant in enumerate(ROWS):
            for col, seed in enumerate(seeds):
                rec = by_key[(story, seed, variant)]
                png = BASE / "evidence" / f"frame-{story}-{seed}-{variant}.jpg"
                frame(rec["result"]["video"], png)
                sheet.paste(Image.open(png).convert("RGB"), (col * 384, row * 286 + 30))
                draw.text((col * 384 + 6, row * 286 + 6), f"{variant} / seed {seed}", fill="black")
        sheet.save(BASE / "evidence" / f"generalization-{story}.jpg", quality=92)


if __name__ == "__main__":
    main()
