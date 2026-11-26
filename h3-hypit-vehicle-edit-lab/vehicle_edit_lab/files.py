import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


class LabError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise LabError(message)


def identifier(value, label, max_length=128):
    require(
        isinstance(value, str)
        and 0 < len(value) <= max_length
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value),
        f"{label} must be an identifier of up to {max_length} characters",
    )
    return value


def integer(value, label, minimum=1, maximum=None):
    require(
        type(value) is int and value >= minimum and (maximum is None or value <= maximum),
        f"invalid {label}: {value!r}",
    )
    return value


def read_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(data, dict), f"expected a JSON object in {path}")
    return data


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lab-", dir=path.parent) as stage:
        staged = Path(stage) / path.name
        write(staged)
        os.link(staged, path)


def write_json(path, data):
    write_new(path, lambda staged: staged.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
    ))
