#!/usr/bin/env python3
import json
import pathlib
import statistics

base = pathlib.Path("/home/ma-user/work/lienyu/minimax-h3-embedding-validation")
rows = [
    json.loads(line)
    for line in (base / "evidence/phase3-results.jsonl").read_text().splitlines()
    if line.strip()
]
print("rows", len(rows), "transport_exact", sum(bool(x.get("transport_exact")) for x in rows))
print(
    "video_exact",
    sum(x.get("video_raw_sha256_normal") == x.get("video_raw_sha256_replay") for x in rows),
)
print(
    "audio_exact",
    sum(x.get("audio_raw_sha256_normal") == x.get("audio_raw_sha256_replay") for x in rows),
)
print("keys", sorted(rows[0]))
for name in ("conditioning", "latent", "noise"):
    values = []
    for row in rows:
        path = base / "packages" / row["package"] / f"{name}.safetensors"
        values.append(path.stat().st_size)
    print(name, min(values), round(statistics.mean(values), 1), max(values), sum(values))
print("output_mp4", len(list((base / "output").glob("h3_phase3-*.mp4"))))
