"""Build an unblinded overview from the per-case randomized review sheets."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from .files import require, write_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("review", type=Path)
    args = parser.parse_args()
    entries = defaultdict(lambda: defaultdict(dict))
    with (args.review / "review-key.csv").open(newline="") as stream:
        for record in csv.DictReader(stream):
            case = record["case"]
            attribute = case.split("--")[1]
            entries[attribute][case][record["variant"]] = int(record["blind_clip"]) - 1
    for attribute, cases in sorted(entries.items()):
        variants = tuple(sorted(set().union(*(case.keys() for case in cases.values()))))
        width, height, label, header = 160, 116, 132, 30
        board = Image.new("RGB", (label + len(variants) * 3 * width, header + height * len(cases)), "white")
        draw = ImageDraw.Draw(board)
        for i, variant in enumerate(variants):
            for j, t in enumerate((1, 3, 5)):
                draw.text((label + (3 * i + j) * width + 4, 7), f"{variant} @ {t}s", fill="black")
        local_detail = attribute in ("vehicle.wheel_design", "vehicle.rear_side_window_shape")
        painted_width = 360 if local_detail else 324
        for i, (case, order) in enumerate(sorted(cases.items())):
            _, _, shot, seed = case.split("--")
            y = header + i * height
            draw.text((3, y + 8), shot, fill="black")
            draw.text((3, y + 27), seed, fill="black")
            source = args.review / f"{case}.jpg"
            require(source.is_file(), f"missing visual source for {case}")
            with Image.open(source) as sheet:
                for v, variant in enumerate(variants):
                    if variant not in order:
                        continue
                    row = order[variant] * 245 + 27
                    for j in range(3):
                        frame = sheet.crop((j * 384, row, j * 384 + painted_width, row + 216))
                        frame.thumbnail((width - 3, height - 3))
                        board.paste(frame, (label + (3 * v + j) * width, y))
        output = args.review / f"overview-{attribute}.jpg"
        if not output.exists():
            write_new(output, lambda target: board.save(target, quality=96))


if __name__ == "__main__":
    main()
