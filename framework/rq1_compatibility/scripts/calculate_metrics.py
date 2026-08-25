#!/usr/bin/env python3
"""Calculate labeled binary RQ1 metrics and explicit status counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    with args.predictions.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    binary = {"compatible", "incompatible"}
    valid = [row for row in rows if row.get("status") == "valid" and row.get("prediction") in binary and row.get("label") in binary]
    tp = sum(row.get("label") == "compatible" and row.get("prediction") == "compatible" for row in valid)
    fn = sum(row.get("label") == "compatible" and row.get("prediction") == "incompatible" for row in valid)
    fp = sum(row.get("label") == "incompatible" and row.get("prediction") == "compatible" for row in valid)
    tn = sum(row.get("label") == "incompatible" and row.get("prediction") == "incompatible" for row in valid)
    total = len(valid)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {
        "total_records": len(rows),
        "valid_labeled_records": total,
        "status_counts": {status: sum(row.get("status") == status for row in rows) for status in ("valid", "uncertain", "invalid", "unparsed", "missing_response")},
        "invalid_label_records": sum(row.get("label") not in binary for row in rows),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "metrics": {"accuracy": (tp + tn) / total if total else 0.0, "precision": precision, "recall": recall, "f1": f1},
        "note": "Only explicitly labeled records with valid binary predictions are included in metrics; uncertain and malformed outputs are not mapped to incompatible.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
