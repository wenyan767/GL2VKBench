#!/usr/bin/env python3
"""Calculate the thesis-defined RQ2 migration metrics from runtime records."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

CLASSES = ("Passed", "Failed", "Unsupported")
CTS_RESULT_MAP = {"Pass": "Passed", "Fail": "Failed", "NotSupported": "Unsupported"}
TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+\.\d+|\d+|==|!=|<=|>=|->|::|\+\+|--|&&|\|\||<<|>>|[{}\[\]();,.:<>+\-*/%&|^~=!?#]")
REQUIRED_FIELDS = (
    "row_id",
    "reference_code",
    "migrated_code",
    "compilation_success",
    "execution_success",
    "reference_result",
    "migrated_result",
    "reference_api_sequence",
    "migrated_api_sequence",
)


def tokens(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return TOKEN_RE.findall(str(value or ""))


def bleu4(reference: Any, hypothesis: Any) -> float:
    ref, hyp = tokens(reference), tokens(hypothesis)
    if not ref or not hyp:
        return 0.0
    precisions = []
    for n in range(1, 5):
        ref_counts = Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1))
        hyp_counts = Counter(tuple(hyp[i:i+n]) for i in range(len(hyp)-n+1))
        total = sum(hyp_counts.values())
        clipped = sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
        precisions.append((clipped + 1.0) / (total + 1.0))
    brevity = 1.0 if len(hyp) > len(ref) else math.exp(1.0 - len(ref) / max(len(hyp), 1))
    return brevity * math.exp(sum(math.log(p) for p in precisions) / 4.0)


def f1_for(label: str, refs: list[str], preds: list[str]) -> float:
    tp = sum(r == label and p == label for r, p in zip(refs, preds))
    fp = sum(r != label and p == label for r, p in zip(refs, preds))
    fn = sum(r == label and p != label for r, p in zip(refs, preds))
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def codebleu(reference: Any, hypothesis: Any) -> float:
    try:
        from codebleu import calc_codebleu  # type: ignore
    except ImportError as exc:
        raise RuntimeError("CB4 requires the optional 'codebleu' package; install it before calculating CodeBLEU-4.") from exc
    result = calc_codebleu([str(reference or "")], [str(hypothesis or "")], lang="cpp")
    return float(result.get("codebleu", 0.0))


def validate_rows(rows: list[Any]) -> list[dict[str, Any]]:
    """Reject incomplete runtime records before calculating aggregate scores."""
    validated: list[dict[str, Any]] = []
    seen_row_ids: set[Any] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} must be a JSON object")
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"row {index} is missing required field(s): {', '.join(missing)}")
        for field in ("reference_code", "migrated_code"):
            if not isinstance(row[field], str):
                raise ValueError(f"row {index}.{field} must be a string")
        for field in ("reference_api_sequence", "migrated_api_sequence"):
            if not isinstance(row[field], list):
                raise ValueError(f"row {index}.{field} must be a list")
        for field in ("compilation_success", "execution_success"):
            if not isinstance(row[field], bool):
                raise ValueError(f"row {index}.{field} must be a JSON boolean")
        row_id = row["row_id"]
        if not isinstance(row_id, (int, str)) or isinstance(row_id, bool) or str(row_id).strip() == "":
            raise ValueError(f"row {index}.row_id must be a non-empty integer or string")
        if row_id in seen_row_ids:
            raise ValueError(f"row {index}.row_id duplicates {row_id!r}")
        seen_row_ids.add(row_id)
        normalized = dict(row)
        normalized["reference_result"] = normalize_result(row["reference_result"], f"row {index}.reference_result", allow_empty=False)
        normalized["migrated_result"] = normalize_result(row["migrated_result"], f"row {index}.migrated_result", allow_empty=True)
        if row["execution_success"] and not row["compilation_success"]:
            raise ValueError(f"row {index} cannot execute successfully after compilation failure")
        if row["execution_success"] != bool(normalized["migrated_result"]):
            raise ValueError(f"row {index}.execution_success must be true exactly when migrated_result contains a valid CTS verdict")
        validated.append(normalized)
    return validated


def normalize_result(value: Any, field: str, *, allow_empty: bool) -> str:
    """Map the normal CTS verdicts to the thesis result classes."""
    if value in ("", None):
        if allow_empty:
            return ""
        raise ValueError(f"{field} cannot be empty")
    result = str(value)
    if result in CLASSES:
        return result
    if result in CTS_RESULT_MAP:
        return CTS_RESULT_MAP[result]
    raise ValueError(f"{field} has no three-class mapping for {result!r}")


def calculate_metrics(rows: list[Any], *, include_codebleu: bool = True) -> dict[str, Any]:
    """Validate records and calculate aggregate RQ2 metrics."""
    rows = validate_rows(rows)
    refs = [str(row["reference_result"]) for row in rows]
    compilable = [bool(row["compilation_success"]) for row in rows]
    executable = [compilable[i] and bool(row["execution_success"]) and row["migrated_result"] in CLASSES for i, row in enumerate(rows)]
    preds = [row["migrated_result"] if executable[i] else "" for i, row in enumerate(rows)]
    api_scores = [bleu4(row["reference_api_sequence"], row["migrated_api_sequence"]) if executable[i] else 0.0 for i, row in enumerate(rows)]
    denominator = len(rows)
    metrics: dict[str, float] = {
        "B4": sum(bleu4(row["reference_code"], row["migrated_code"]) for row in rows) / denominator if rows else 0.0,
        "CSR": sum(compilable) / denominator if rows else 0.0,
        "ESR": sum(executable) / denominator if rows else 0.0,
        "TRC": sum(f1_for(label, refs, preds) for label in CLASSES) / len(CLASSES) if rows else 0.0,
        "APISim": sum(api_scores) / denominator if rows else 0.0,
    }
    if include_codebleu:
        metrics["CB4"] = sum(codebleu(row["reference_code"], row["migrated_code"]) for row in rows) / denominator if rows else 0.0
    return {
        "total_cases": denominator,
        "metrics": metrics,
        "result_classes": list(CLASSES),
        "cts_result_mapping": CTS_RESULT_MAP,
        "denominator_policy": "All supplied RQ2 cases; missing or failed runtime evidence contributes zero.",
        "note": "execution_success means CTS produced Pass/Fail/NotSupported, not that the test passed; ESR and APISim require a valid verdict.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="User-owned runtime migration result JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-codebleu", action="store_true", help="Omit CB4 when codebleu is unavailable")
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list):
        raise ValueError("input must contain a JSON list")
    report = calculate_metrics(rows, include_codebleu=not args.skip_codebleu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
