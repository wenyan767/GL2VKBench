#!/usr/bin/env python3
"""Build a disposable RQ1 runtime manifest from the two source datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RQ1_ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_ROOT = RQ1_ROOT.parent
GITHUB_ROOT = FRAMEWORK_ROOT.parent
DEFAULT_POSITIVE = GITHUB_ROOT / "benchmark" / "benchmark_data.json"
DEFAULT_NEGATIVE = RQ1_ROOT / "inputs" / "neg_406.json"


def load_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path} must be a JSON array of objects")
    return value


def assert_no_semantic_description(value: Any, path: Path) -> None:
    if isinstance(value, dict):
        if "semantic_description" in value:
            raise ValueError(f"{path} contains semantic_description; remove derived fields first")
        for child in value.values():
            assert_no_semantic_description(child, path)
    elif isinstance(value, list):
        for child in value:
            assert_no_semantic_description(child, path)


def source_parts(record: dict[str, Any], negative: bool) -> tuple[str, str, str, str]:
    case = record.get("gl_subclass") if isinstance(record.get("gl_subclass"), dict) else record
    bases = record.get("gl_base_classes") if not negative else record.get("base_classes")
    bases = bases if isinstance(bases, list) else []
    base_code = "\n\n".join(str(item.get("code", "")) for item in bases if isinstance(item, dict))
    if negative:
        record_id = str(case.get("qualified_name") or "")
        label = "incompatible"
    else:
        record_id = str(record.get("row_id") or case.get("class_id") or "")
        label = "compatible"
    code = str(case.get("code", ""))
    if not record_id or not code.strip():
        raise ValueError(f"record has no stable ID or test-case code: {record_id!r}")
    return record_id, code, base_code, label


def build_manifest(positive: list[dict[str, Any]], negative: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_name, records, is_negative in (("benchmark_data.json", positive, False), ("neg_406.json", negative, True)):
        for index, record in enumerate(records):
            record_id, code, base_code, label = source_parts(record, is_negative)
            if record_id in seen:
                raise ValueError(f"duplicate runtime record_id: {record_id}")
            seen.add(record_id)
            rows.append({
                "record_id": record_id,
                "source": source_name,
                "source_index": index,
                "label": label,
                "semantic_description": None,
                "semantic_generation_inputs": {
                    "test_case_code": code,
                    "base_class_code": base_code,
                },
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive", type=Path, default=DEFAULT_POSITIVE)
    parser.add_argument("--negative", type=Path, default=DEFAULT_NEGATIVE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    positive = load_array(args.positive.resolve())
    negative = load_array(args.negative.resolve())
    assert_no_semantic_description(positive, args.positive)
    assert_no_semantic_description(negative, args.negative)
    rows = build_manifest(positive, negative)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} runtime requests to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
