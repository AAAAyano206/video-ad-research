import os
import xml.etree.ElementTree as ET
from pathlib import Path

from .files import identifier, read_json, require, sha256, write_new


def relative_path(path, run_path):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), f"expected a file: {path}")
    return Path(os.path.relpath(path, run_path.parent)).as_posix()


def assign(value):
    output, separator, argument = value.partition("=")
    require(separator and argument, f"expected OUTPUT=VALUE: {value}")
    return identifier(output, "Hypit output"), argument


def receipt_video(path):
    path = Path(path)
    receipt = read_json(path)
    require(receipt["schema_version"] == 1 and receipt["status"] == "completed", f"run has no completed video: {path}")
    video = path.parent / identifier(receipt["video"], "receipt video filename")
    require(sha256(video) == receipt["video_sha256"], f"receipt video checksum mismatch: {video}")
    return video


def export_run(author, output, target, clips, receipts, reuse):
    output = Path(output).resolve()
    require(output.suffix == ".svrun", "output must be a .svrun file")
    run = ET.Element("svrun", {"version": "1"})
    ET.SubElement(run, "author", {"source": relative_path(author, output)})
    ET.SubElement(run, "target", {"output": identifier(target, "Hypit target")})
    bindings = {}

    def candidate(slot):
        require(slot not in bindings, f"more than one replacement for {slot}")
        name = f"candidate-{len(bindings) + 1}"
        bindings[slot] = name
        return name

    media = [assign(item) for item in clips]
    for item in receipts:
        slot, receipt = assign(item)
        media.append((slot, receipt_video(receipt)))
    for slot, path in media:
        require(Path(path).suffix.lower() == ".mp4", f"expected an MP4 for {slot}")
        ET.SubElement(run, "file", {
            "id": candidate(slot), "type": "@hypit/blob@1#Blob",
            "from": relative_path(path, output), "media-type": "video/mp4",
        })
    for item in reuse:
        slot, spec = assign(item)
        build, separator, previous = spec.partition(":")
        require(separator and previous, f"expected OUTPUT=BUILD_ID:PREVIOUS_OUTPUT: {item}")
        ET.SubElement(run, "build-record", {
            "id": candidate(slot), "build": identifier(build, "build id"),
            "output": identifier(previous, "previous output"),
        })
    require(bool(bindings), "provide at least one replacement or reused result")
    for slot, name in bindings.items():
        ET.SubElement(run, "satisfy", {"output": slot, "candidate": name})
    ET.indent(run, space="  ")
    write_new(output, lambda staged: staged.write_text(
        '<?svml using="@hypit/markup/run@1"?>\n' + ET.tostring(run, encoding="unicode") + "\n", encoding="utf-8",
    ))
