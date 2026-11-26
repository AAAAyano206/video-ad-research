#!/usr/bin/env python3
import io
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


BASE = Path("/home/ma-user/work/lienyu/minimax-h3-embedding-validation")
phase3 = [json.loads(x) for x in (BASE / "evidence/phase3-results.jsonl").read_text().splitlines()]
phase4 = [json.loads(x) for x in (BASE / "evidence/phase4-compression.jsonl").read_text().splitlines()]
grouped = defaultdict(dict)
for row in phase4:
    grouped[row["prompt_id"]][row["method"]] = Path(row["generated"]["video"])
for row in phase3:
    if row["seed"] == 101 and row["prompt_id"] in grouped:
        grouped[row["prompt_id"]]["baseline"] = Path(row["normal"]["video"])


def frame(path, timestamp):
    data = subprocess.check_output([
        "ffmpeg", "-v", "error", "-ss", str(timestamp), "-i", str(path),
        "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",
    ])
    return Image.open(io.BytesIO(data)).convert("RGB")


target = BASE / "evidence/phase4-compression-sheets"
target.mkdir(parents=True, exist_ok=True)
times = (0.5, 1.5, 2.5, 3.5)
for prompt_id, videos in sorted(grouped.items()):
    sheet = Image.new("RGB", (1024, 3 * 286), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, label in enumerate(("baseline", "fp16", "int8-channel")):
        y = row_index * 286
        draw.text((8, y + 4), label, fill="black")
        for column, timestamp in enumerate(times):
            image = frame(videos[label], timestamp)
            sheet.paste(image, (column * 256, y + 30))
            draw.text((column * 256 + 205, y + 34), f"{timestamp:.1f}s", fill="white", stroke_width=1, stroke_fill="black")
    sheet.save(target / f"{prompt_id}.jpg", quality=92)
print(target)
