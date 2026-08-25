"""Prepare model-independent RQ2 migration prompts from BGE Top-1 output.

This script deliberately does not import a model client. It validates the retrieval
contract and writes plain-text prompts that users can send through any model of
their choice.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL = ("row_id", "gl_subclass", "gl_base_classes", "similar_vulkan_case")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def as_records(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"{path} must contain a JSON object or a list of objects.")
    return payload


def require_text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")


def validate_record(record: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_TOP_LEVEL if key not in record]
    if missing:
        raise ValueError(f"row {record.get('row_id', '?')} is missing: {', '.join(missing)}")
    subclass = record["gl_subclass"]
    if not isinstance(subclass, dict):
        raise ValueError(f"row {record['row_id']}: gl_subclass must be an object.")
    require_text(subclass.get("code"), f"row {record['row_id']}.gl_subclass.code")
    candidate = record["similar_vulkan_case"]
    if not isinstance(candidate, dict):
        raise ValueError(
            f"row {record['row_id']} has no eligible similar_vulkan_case; rerun retrieval or inspect candidates."
        )
    require_text(candidate.get("code"), f"row {record['row_id']}.similar_vulkan_case.code")
    require_text(candidate.get("class_id"), f"row {record['row_id']}.similar_vulkan_case.class_id")
    class_type = str(candidate.get("class_type") or "").strip().lower()
    if class_type not in {"child", "class", "function"}:
        raise ValueError(
            f"row {record['row_id']}.similar_vulkan_case.class_type must be 'child', 'class', or 'function'."
        )
    if "similarity_score" not in candidate:
        raise ValueError(f"row {record['row_id']}.similar_vulkan_case.similarity_score is required.")


def render(template: str, record: dict[str, Any]) -> str:
    marker = "{{RETRIEVAL_RECORD_JSON}}"
    template_markers = set(re.findall(r"\{\{[^{}\r\n]+\}\}", template))
    if marker not in template_markers:
        raise ValueError(f"Prompt template must contain {marker}.")
    unknown_markers = sorted(template_markers - {marker})
    if unknown_markers:
        raise ValueError(
            "Prompt template contains unsupported placeholders: "
            + ", ".join(unknown_markers)
        )
    rendered = template.replace(
        marker, json.dumps(record, ensure_ascii=False, indent=2)
    )
    return rendered.rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RQ2 migration prompts from BGE Top-1 JSON.")
    parser.add_argument("--input", required=True, type=Path, help="BGE retrieval output JSON.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Runtime directory for generated prompts.")
    parser.add_argument("--template", type=Path, default=Path(__file__).with_name("migration_prompt.txt"))
    parser.add_argument("--row-id", type=int, help="Prepare only one row.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = as_records(load_json(args.input), args.input)
    if args.row_id is not None:
        records = [record for record in records if record.get("row_id") == args.row_id]
        if not records:
            raise ValueError(f"No row_id={args.row_id} found in {args.input}.")
    template = args.template.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for record in records:
        validate_record(record)
        row_id = record["row_id"]
        prompt_path = args.output_dir / f"case_{row_id}.prompt.txt"
        prompt_path.write_text(render(template, record), encoding="utf-8")
        manifest.append({"row_id": row_id, "prompt_file": prompt_path.name})
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(manifest)} prompt(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
