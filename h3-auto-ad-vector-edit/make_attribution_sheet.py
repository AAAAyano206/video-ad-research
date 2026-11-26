#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw

BASE = Path("/home/ma-user/work/lienyu/h3-auto-ad-vector-edit")
data = json.loads((BASE / "evidence/fine-attribution.json").read_text())
sheet = Image.new("RGB", (5 * 384, 3 * 286), "white")
draw = ImageDraw.Draw(sheet)
for index, record in enumerate(data["records"]):
    video = record["result"]["video"]
    frame = BASE / "evidence" / f"attr-frame-{index:02d}.jpg"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "3", "-i", video,
                    "-frames:v", "1", "-vf", "scale=384:256", str(frame)], check=True)
    row, column = divmod(index, 5)
    label = record["probe"]
    if label == "token": label += f" row={record['row']}"
    if label == "block_group": label += f" {record['start']}-{record['end']}"
    if label == "steps": label += f"={record['steps']}"
    draw.text((column * 384 + 5, row * 286 + 6), label, fill="black")
    sheet.paste(Image.open(frame).convert("RGB"), (column * 384, row * 286 + 30))
sheet.save(BASE / "evidence/fine-attribution-sheet.jpg", quality=92)
