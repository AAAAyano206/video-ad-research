#!/usr/bin/env python3
import io
import json
import subprocess
import textwrap
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


BASE = Path("/home/ma-user/work/lienyu/minimax-h3-embedding-validation")
rows = [json.loads(line) for line in (BASE / "evidence/phase3-results.jsonl").read_text().splitlines()]
grouped = defaultdict(list)
for row in rows:
    grouped[row["prompt_id"]].append(row)


def frame(path, timestamp):
    data = subprocess.check_output([
        "ffmpeg", "-v", "error", "-ss", str(timestamp), "-i", str(path),
        "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",
    ])
    return Image.open(io.BytesIO(data)).convert("RGB")


target = BASE / "evidence/contact-sheets"
target.mkdir(parents=True, exist_ok=True)
times = (0.5, 1.5, 2.5, 3.5)
for prompt_id, items in sorted(grouped.items()):
    items.sort(key=lambda item: item["seed"])
    title_lines = textwrap.wrap(items[0]["prompt"], width=110)
    title_height = 24 + 18 * len(title_lines)
    sheet = Image.new("RGB", (4 * 256, title_height + 3 * 286), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), f"{prompt_id}: " + "\n".join(title_lines), fill="black")
    for row_index, item in enumerate(items):
        video = Path(item["normal"]["video"])
        y = title_height + row_index * 286
        draw.text((8, y + 4), f"seed {item['seed']}", fill="black")
        for column, timestamp in enumerate(times):
            image = frame(video, timestamp)
            sheet.paste(image, (column * 256, y + 30))
            draw.text((column * 256 + 205, y + 34), f"{timestamp:.1f}s", fill="white", stroke_width=1, stroke_fill="black")
    sheet.save(target / f"{prompt_id}.jpg", quality=92)
print(target)
