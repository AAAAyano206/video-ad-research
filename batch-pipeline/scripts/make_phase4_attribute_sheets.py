#!/usr/bin/env python3
import io
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


BASE = Path("/home/ma-user/work/lienyu/minimax-h3-embedding-validation")
phase3 = [json.loads(x) for x in (BASE / "evidence/phase3-results.jsonl").read_text().splitlines()]
rows = [json.loads(x) for x in (BASE / "evidence/phase4-attribute.jsonl").read_text().splitlines()]


def frame(path, timestamp):
    data = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", str(timestamp), "-i", str(path),
                                    "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"])
    return Image.open(io.BytesIO(data)).convert("RGB")


target = BASE / "evidence/phase4-attribute-sheets"
target.mkdir(parents=True, exist_ok=True)
for row in rows:
    baseline = next(x for x in phase3 if x["prompt_id"] == row["prompt_id"] and x["seed"] == 101)
    videos = {"source": baseline["normal"]["video"], "full alternate": row["alternate"]["video"], "one-row edit": row["edited"]["video"]}
    sheet = Image.new("RGB", (1024, 858), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, path) in enumerate(videos.items()):
        y = i * 286
        draw.text((8, y + 4), f"{label}: {row['label']}", fill="black")
        for column, timestamp in enumerate((0.5, 1.5, 2.5, 3.5)):
            sheet.paste(frame(path, timestamp), (column * 256, y + 30))
    sheet.save(target / f"{row['prompt_id']}.jpg", quality=92)
print(target)
