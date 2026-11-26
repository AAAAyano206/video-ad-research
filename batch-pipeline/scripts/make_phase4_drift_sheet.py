#!/usr/bin/env python3
import io
import json
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw

base = Path("/home/ma-user/work/lienyu/minimax-h3-embedding-validation")
row = json.loads((base / "evidence/phase4-drift.json").read_text())
videos = [("source detail (seed101)", row["source"]["video"]),
          ("fresh baseline (seed202, 4 steps)", row["fresh_baseline"]["video"]),
          ("reused latent (same seed202 noise, 4 steps)", row["reused_latent"]["video"])]
sheet = Image.new("RGB", (1024, 858), "white")
draw = ImageDraw.Draw(sheet)
for i, (label, path) in enumerate(videos):
    y = i * 286
    draw.text((8, y + 4), label, fill="black")
    for j, timestamp in enumerate((0.5, 1.5, 2.5, 3.5)):
        data = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", str(timestamp), "-i", path,
                                        "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"])
        sheet.paste(Image.open(io.BytesIO(data)).convert("RGB"), (j * 256, y + 30))
target = base / "evidence/phase4-drift-sheet.jpg"
sheet.save(target, quality=92)
print(target)
