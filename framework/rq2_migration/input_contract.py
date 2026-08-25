"""Validation for user-generated RQ2 retrieval output consumed by Framework."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL = (
    "row_id",
    "ground_truth_vk_class_id",
    "gl_subclass",
    "gl_base_classes",
    "similar_vulkan_case",
)


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def reference_template_type(value: Any) -> str:
    """Normalize the retrieval label to the generated CTS template type."""
    raw_type = str(value or "").strip().lower()
    if raw_type in {"child", "class"}:
        return "class"
    if raw_type == "function":
        return "function"
    raise ValueError(
        "similar_vulkan_case.class_type must be 'child', 'class', or 'function'"
    )


def validate_records(payload: Any, *, source: str = "Framework input") -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError(f"{source} must contain a JSON list")
    seen_row_ids: set[int] = set()
    records: list[dict[str, Any]] = []
    for index, record in enumerate(payload):
        label = f"{source} record {index}"
        if not isinstance(record, dict):
            raise ValueError(f"{label} must be an object")
        missing = [key for key in REQUIRED_TOP_LEVEL if key not in record]
        if missing:
            raise ValueError(f"{label} is missing: {', '.join(missing)}")
        try:
            row_id = int(record["row_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}.row_id must be an integer") from exc
        if row_id in seen_row_ids:
            raise ValueError(f"{source} has duplicate row_id={row_id}")
        seen_row_ids.add(row_id)

        gl_case = record["gl_subclass"]
        if not isinstance(gl_case, dict):
            raise ValueError(f"row {row_id}.gl_subclass must be an object")
        for key in ("class_id", "code", "semantic_description"):
            if not _nonempty_text(gl_case.get(key)):
                raise ValueError(f"row {row_id}.gl_subclass.{key} must be non-empty")
        if not isinstance(record["gl_base_classes"], list):
            raise ValueError(f"row {row_id}.gl_base_classes must be a list")

        ground_truth = record["ground_truth_vk_class_id"]
        if not _nonempty_text(ground_truth):
            raise ValueError(f"row {row_id}.ground_truth_vk_class_id must be non-empty")
        candidate = record["similar_vulkan_case"]
        if not isinstance(candidate, dict):
            raise ValueError(f"row {row_id}.similar_vulkan_case must be a selected candidate object")
        for key in ("class_id", "class_type", "code", "semantic_description"):
            if not _nonempty_text(candidate.get(key)):
                raise ValueError(f"row {row_id}.similar_vulkan_case.{key} must be non-empty")
        try:
            reference_template_type(candidate["class_type"])
        except ValueError as exc:
            raise ValueError(f"row {row_id}.{exc}") from exc
        if candidate["class_id"] == ground_truth:
            raise ValueError(f"row {row_id} retrieved its own ground-truth Vulkan class")
        score = candidate.get("similarity_score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError(f"row {row_id}.similar_vulkan_case.similarity_score must be numeric")
        records.append(record)
    return records


def load_records(path: Path) -> list[dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Framework input does not exist: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    return validate_records(payload, source=str(resolved))


__all__ = [
    "REQUIRED_TOP_LEVEL",
    "load_records",
    "reference_template_type",
    "validate_records",
]
