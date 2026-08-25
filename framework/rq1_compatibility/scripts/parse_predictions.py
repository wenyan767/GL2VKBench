#!/usr/bin/env python3
"""Parse user-provided RQ1 model responses without guessing labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

VALID = {"compatible", "incompatible", "uncertain"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def response_text(row: dict[str, Any]) -> str:
    value = row.get("raw_response", row.get("response", ""))
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def parse_one(text: str) -> tuple[str | None, str, str | None]:
    try:
        value = json.loads(text.strip())
        if isinstance(value, dict):
            prediction = str(value.get("compatibility", "")).strip().lower()
            reason = str(value.get("reason", "")).strip() or None
            if prediction in VALID and reason:
                status = "uncertain" if prediction == "uncertain" else "valid"
                return prediction, status, reason
            return None, "invalid", reason
    except json.JSONDecodeError:
        pass
    return None, "unparsed", None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = {str(row["record_id"]): row for row in read_jsonl(args.manifest)}
    responses = {str(row.get("record_id")): row for row in read_jsonl(args.responses) if row.get("record_id") is not None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with args.output.open("w", encoding="utf-8") as stream:
        for record_id, source in manifest.items():
            response = responses.get(record_id)
            if response is None:
                prediction, status, reason = None, "missing_response", None
            else:
                prediction, status, reason = parse_one(response_text(response))
            row = {
                "record_id": record_id,
                "label": source.get("label"),
                "prediction": prediction,
                "status": status,
                "reason": reason,
            }
            counts[status] = counts.get(status, 0) + 1
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(manifest), "status_counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
