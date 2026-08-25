#!/usr/bin/env python3
"""Run a Vulkan CTS case and materialize its runtime API sequence.

This model-independent tool combines the two RQ2 runtime steps that are easy
to accidentally separate: enabling ``VK_LAYER_LUNARG_api_dump`` while a test
executes and converting the resulting dump into an ordered API-name stream.
It can also parse an existing dump with ``--dump-only``.

The generated ``api_sequence.json`` is suitable for converting into the
``reference_api_sequence``/``migrated_api_sequence`` fields consumed by the
RQ2 metric calculator. No model API is called.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API_CALL_PATTERN = re.compile(r"\b(vk[A-Za-z0-9_]+)\s*\(")
DEFAULT_LAYER = "VK_LAYER_LUNARG_api_dump"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def extract_api_sequence(text: str) -> list[str]:
    """Extract every Vulkan call name in dump order."""
    return [match.group(1) for match in API_CALL_PATTERN.finditer(text)]


def _write_gzip(path: Path, data: bytes) -> Path:
    gz_path = Path(str(path) + ".gz")
    with gzip.open(gz_path, "wb") as stream:
        stream.write(data)
    return gz_path


def _text_output(value: Any) -> str:
    """Normalize subprocess output from both text and bytes exception paths."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def build_payload(
    dump_path: Path,
    *,
    case_name: str = "",
    execution_status: str = "",
    returncode: int | None = None,
    command: list[str] | None = None,
    row_id: int | None = None,
    class_id: str = "",
    test_path: str = "",
    runtime_name: str = "",
    qpa_status_code: str = "",
) -> dict[str, Any]:
    exists = dump_path.exists()
    text = dump_path.read_text(encoding="utf-8", errors="replace") if exists else ""
    sequence = extract_api_sequence(text)
    normalized = "\n".join(sequence)
    return {
        "generated_at": now(),
        "case_name": case_name,
        "row_id": row_id,
        "ground_truth_vk_class_id": class_id,
        "test_path": test_path,
        "runtime_name": runtime_name,
        "execution_status": execution_status,
        "qpa_status_code": qpa_status_code,
        "returncode": returncode,
        "command": command or [],
        "api_dump_path": str(dump_path),
        "api_dump_exists": exists,
        "api_dump_size_bytes": dump_path.stat().st_size if exists else 0,
        "api_sequence": sequence,
        "api_sequence_length": len(sequence),
        "api_sequence_hash": hashlib.sha256(normalized.encode("utf-8")).hexdigest() if sequence else "",
        "api_call_counts": {name: sequence.count(name) for name in sorted(set(sequence))},
    }


def materialize(
    dump_path: Path,
    output_dir: Path,
    *,
    case_name: str = "",
    execution_status: str = "",
    returncode: int | None = None,
    command: list[str] | None = None,
    gzip_output: bool = False,
    row_id: int | None = None,
    class_id: str = "",
    test_path: str = "",
    runtime_name: str = "",
    qpa_status_code: str = "",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(
        dump_path,
        case_name=case_name,
        execution_status=execution_status,
        returncode=returncode,
        command=command,
        row_id=row_id,
        class_id=class_id,
        test_path=test_path,
        runtime_name=runtime_name,
        qpa_status_code=qpa_status_code,
    )
    json_bytes = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    names_bytes = ("\n".join(payload["api_sequence"]) + ("\n" if payload["api_sequence"] else "")).encode("utf-8")
    json_path = output_dir / "api_sequence.json"
    names_path = output_dir / "api_names.txt"
    json_path.write_bytes(json_bytes)
    names_path.write_bytes(names_bytes)
    if gzip_output:
        _write_gzip(json_path, json_bytes)
        _write_gzip(names_path, names_bytes)
    return payload


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_path = args.api_dump or (output_dir / "vk_apidump.txt")
    command = [str(args.executable), f"--deqp-case={args.case}"] + list(args.extra_arg)
    qpa_path = output_dir / "TestResults.qpa"
    if not any(arg.startswith("--deqp-log-filename=") for arg in command):
        command.insert(2, f"--deqp-log-filename={qpa_path}")
    env = os.environ.copy()
    env["VK_INSTANCE_LAYERS"] = args.layer_name
    env["VK_APIDUMP_OUTPUT_FORMAT"] = "text"
    env["VK_APIDUMP_LOG_FILENAME"] = str(dump_path)
    env["VK_APIDUMP_DETAILED"] = "false"
    env["VK_APIDUMP_NO_ADDR"] = "true"
    env["VK_APIDUMP_FLUSH"] = "true"
    layer_path = args.layer_path
    if layer_path:
        env["VK_LAYER_PATH"] = str(layer_path)
    elif os.environ.get("VULKAN_SDK"):
        env["VK_LAYER_PATH"] = str(Path(os.environ["VULKAN_SDK"]) / "Bin")

    stdout_path = output_dir / "run_stdout.txt"
    stderr_path = output_dir / "run_stderr.txt"
    started = time.monotonic()
    status = "failed"
    returncode: int | None = None
    timeout_hit = False
    try:
        result = subprocess.run(
            command,
            cwd=str(args.cwd) if args.cwd else None,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=args.timeout,
        )
        returncode = result.returncode
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        status = "passed" if returncode == 0 else "failed"
    except subprocess.TimeoutExpired as exc:
        timeout_hit = True
        stdout_path.write_text(_text_output(exc.stdout), encoding="utf-8")
        stderr_path.write_text(_text_output(exc.stderr), encoding="utf-8")
        status = "timeout"
    except OSError as exc:
        stderr_path.write_text(str(exc), encoding="utf-8")
        status = "error"

    payload = materialize(
        dump_path,
        output_dir,
        case_name=args.case,
        execution_status=status,
        returncode=returncode,
        command=command,
        gzip_output=args.gzip,
        row_id=args.row_id,
        class_id=args.class_id,
        test_path=args.test_path,
        runtime_name=args.runtime_name,
        qpa_status_code=args.qpa_status_code,
    )
    summary = {
        "generated_at": now(),
        "case_name": args.case,
        "status": status,
        "returncode": returncode,
        "timeout_hit": timeout_hit,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "command": command,
        "api_sequence_path": str(output_dir / "api_sequence.json"),
        "api_names_path": str(output_dir / "api_names.txt"),
        "api_sequence_length": payload["api_sequence_length"],
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, help="Path to deqp-vk executable (required unless --dump-only).")
    parser.add_argument("--case", default="", help="Exact dEQP case name.")
    parser.add_argument("--row-id", type=int, default=None, help="Optional benchmark row identifier.")
    parser.add_argument("--class-id", default="", help="Optional ground-truth Vulkan class identifier.")
    parser.add_argument("--test-path", default="", help="Optional source test path.")
    parser.add_argument("--runtime-name", default="", help="Optional runtime label (for example vk_gl_cts).")
    parser.add_argument("--execution-status", default="", help="Status to attach in --dump-only mode.")
    parser.add_argument("--qpa-status-code", default="", help="Optional QPA status code.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, default=None)
    parser.add_argument("--api-dump", type=Path, default=None, help="Existing dump path or run output path.")
    parser.add_argument("--layer-name", default=DEFAULT_LAYER)
    parser.add_argument("--layer-path", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--gzip", action="store_true", help="Also write api_sequence.json.gz and api_names.txt.gz.")
    parser.add_argument("--dump-only", action="store_true", help="Parse an existing --api-dump without executing a test.")
    args = parser.parse_args()
    if args.dump_only:
        if not args.api_dump:
            parser.error("--dump-only requires --api-dump")
        payload = materialize(
            args.api_dump,
            args.output_dir,
            execution_status=args.execution_status,
            gzip_output=args.gzip,
            row_id=args.row_id,
            class_id=args.class_id,
            test_path=args.test_path,
            runtime_name=args.runtime_name,
            qpa_status_code=args.qpa_status_code,
        )
    else:
        if not args.executable or not args.case:
            parser.error("--executable and --case are required unless --dump-only is used")
        payload = run_case(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
