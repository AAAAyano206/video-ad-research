"""Check independent scorecards and calculate target and strict rates before reporting."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path


FIELDS = ("id", "candidate", "target", "stable", "body", "scene", "camera", "other", "notes")
DECISIONS = FIELDS[2:-1]


def read_json(path):
    return json.loads(path.read_text())


def ratings(path, expected):
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames) != FIELDS:
            raise ValueError(f"unexpected header in {path}")
        data = {}
        for row in reader:
            key = row["id"], row["candidate"]
            if key in data or not row["notes"].strip() or any(row[f] not in {"P", "F", "U"} for f in DECISIONS):
                raise ValueError(f"invalid review row {key} in {path}")
            data[key] = row
    if data.keys() != expected:
        raise ValueError(f"{path} covered {len(data)} of {len(expected)} required candidate rows")
    return data


def summarize(folder):
    key = read_json(folder / "group-key.json")["cases"]
    expected = {(case, label) for case, record in key.items() for label in record["mapping"]}
    first, second = (ratings(folder / f"reviewer-{i}.csv", expected) for i in (1, 2))
    adjudication_path = folder / "adjudication.json"
    adjudication = read_json(adjudication_path) if adjudication_path.exists() else {}
    rows, groups = [], {}
    differences = Counter()
    for case, label in sorted(expected):
        scores = {}
        reasons = {}
        for field in DECISIONS:
            a, b = first[(case, label)][field], second[(case, label)][field]
            if a == b:
                scores[field] = a
            else:
                differences[field] += 1
                entry = adjudication.get(f"{case}/{label}", {}).get(field)
                if entry:
                    if entry["score"] not in {"P", "F", "U"} or not entry["reason"]:
                        raise ValueError(f"invalid adjudication for {case}/{label}/{field}")
                    scores[field] = entry["score"]
                    reasons[field] = entry["reason"]
                else:
                    scores[field] = "P" if a == b == "P" else "U"
        method = key[case]["mapping"][label]
        group = groups.setdefault(method, Counter(n=0, target=0, stable=0, strict=0))
        group["n"] += 1
        hit = scores["target"] == "P"
        stable = hit and scores["stable"] == "P"
        strict = stable and all(scores[f] == "P" for f in DECISIONS[2:])
        for field in DECISIONS[2:]:
            group[f"{field}_pass"] += scores[field] == "P"
        group.update(target=hit, stable=stable, strict=strict)
        rows.append({"blind_id": case, "candidate": label, "source_case": key[case]["case"], "method": method,
                     **scores, "strict": strict, "adjudication": reasons,
                     "reviewer_notes": [first[(case, label)]["notes"], second[(case, label)]["notes"]]})
    unresolved = sum(differences.values()) - sum(len(row["adjudication"]) for row in rows)
    result = {"reviewers": "two independent AI assessors on anonymous paired case clips", "ratings_per_reviewer": len(expected),
              "disagreed_fields": dict(differences), "unresolved_fields": unresolved,
              "policy": "U is never counted as a pass; strict requires all six fields to pass.",
              "groups": {group: dict(counter) for group, counter in groups.items()}, "rows": rows}
    output = folder / "report.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"groups": result["groups"], "disagreed_fields": dict(differences), "unresolved": unresolved},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    summarize(parser.parse_args().folder)
