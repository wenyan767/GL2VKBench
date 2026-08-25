"""Step 3 —— 修复循环（三条方法线）。

源自 agent_runner.py。三条线共用本文件：

    woCot         non_cot seed + feedback repair
    singleAgent   cot     seed + feedback repair
    multiAgent    non_cot seed + multi-agent orchestration

attempt 从 2 起；attempt_1 就是种子本身（Step 2 已编译，判据落盘共用），
不在这里重复生成，也不复制文件。

改动：
  * 删除 stage2 LLM 池分支（--llm-channel / --llm-worker-index）与连通性探测。
  * 客户端改为环境变量驱动，删除 CONFIG_DIR_MAP 与全部配置文件加载。
  * 路径锚点改到 paths.py。
"""

from __future__ import annotations


import argparse

import json

import os

import random

import re

import shutil

import subprocess

import sys

import time

import traceback

from dataclasses import dataclass

from datetime import datetime

from pathlib import Path

from typing import Any

CURRENT_FILE = Path(__file__).resolve()


from cts import build_case_dir, resolve_cts_root, reset_module_generated_state, run_case_compile

from core import acquire_build_lock, build_lock_scope_for_cts_root

from core import (

    clear_execution_overrides,
    get_execution_overrides,
    resolve_config_dir_override,
    resolve_defer_compile,


    resolve_output_root_override,
    resolve_run_record_path_override,
    resolve_run_after_compile,
    resolve_row_ids_override,
    resolve_package_name,
    resolve_runtime_root_group,
    resolve_target_name,
    set_execution_overrides,

)

from cts import summarize_compile_logs, summary_error_count, write_summary

from core import (

    CLASS_REPAIR_SCHEMA,
    CHECKER_SCHEMA,
    EXPLAINER_SCHEMA,
    DIRECTOR_DECISION_SCHEMA,
    FUNCTION_REPAIR_SCHEMA,
    REGENERATION_CLASS_SCHEMA,
    REGENERATION_FUNCTION_SCHEMA,
    VERIFIER_SCHEMA,
    STRUCTURE_REJUDGE_SCHEMA,

)

from core import load_gl_sim_vk_row, load_initial_candidate, load_json, resolve_gl_sim_vk_path

import model_adapter
from paths import (
    DEFAULT_RUNS_DIR,
    DIRECTOR_PROMPT_DIR as _DIRECTOR_PROMPT_DIR,
    FEEDBACK_PROMPT_DIR as _FEEDBACK_PROMPT_DIR,
)
from methods import METHOD_INPUT_NAMES, get_method, resolve_repair_strategy

from core import prepare_case, write_json

from core import (

    _build_locked_classification,
    _build_repaired_body,
    _normalize_payload_consistency,
    apply_repair_payload,
    validate_payload,

)

from core import clean_generated_code

from core import ClassifiedCode, classify_candidate_code

from core import validate_case_artifact

from core import assemble_cpp


FEEDBACK_PROMPT_DIR = _FEEDBACK_PROMPT_DIR

PROMPT_ROOT_DIR = _DIRECTOR_PROMPT_DIR


def baseline_check_log_dir() -> Path:
    """Keep baseline evidence under the caller-selected artifact root."""
    return (resolve_output_root_override() or DEFAULT_RUNS_DIR) / "_baseline_checks"

PROMPT_INPUT_TOKEN_BUDGET = 48000

PROMPT_TOKEN_CHAR_RATIO = 2

PROMPT_SLOT_BUDGETS = {

    "{{GL_SUBCLASS_JSON}}": 12000,
    "{{GL_BASE_CLASSES_JSON}}": 6000,
    "{{COMPILE_RAW_OUTPUT}}": 30000,
    "{{CHECKER_OUTPUT}}": 24000,
    "{{VERIFIER_OUTPUT}}": 12000,
    "{{EXPLAINER_OUTPUT}}": 12000,
    # {{COMPILE_ERRORS_JSON}} 故意不在这张表里 —— 编译错误**任何情况下都不截断**,
    # 即使整条提示词超预算也不动它。没有预算项 = 压缩循环取不到 budget 就跳过。
    "{{CURRENT_STRUCTURE_JSON}}": 8000,
    "{{DIRECTOR_OUTPUT}}": 8000,
    "{{PREVIOUS_VK_CODE}}": 24000,
    "{{ASSEMBLED_CANDIDATE_CODE}}": 24000,
    "{{CURRENT_VK_CODE}}": 24000,
    "{{CLEANED_CANDIDATE_BODY}}": 24000,

}


BUILD_CONFLICT_MAX_RETRIES = 5

BUILD_CONFLICT_WAIT_MIN_SECONDS = 4.0

BUILD_CONFLICT_WAIT_MAX_SECONDS = 12.0


def _configure_stdio_for_live_progress() -> None:

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


def _get_all_row_ids() -> list[int]:

    """Load all available row_ids from the explicitly supplied Framework input."""
    rows = load_json(resolve_gl_sim_vk_path())
    return sorted([int(row["row_id"]) for row in rows])


def _read_text(path: Path) -> str:

    return path.read_text(encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:

    return json.loads(_read_text(path))


def _json_dump(data: Any) -> str:

    return json.dumps(data, ensure_ascii=False, indent=2)


def _json_safe(value: Any) -> Any:

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=lambda item: str(item))]
    return value


def _is_build_conflict(stdout_text: str, stderr_text: str) -> bool:

    combined = f"{stdout_text}\n{stderr_text}".lower()
    conflict_markers = [
        "permission denied",
        "cannot restore timestamp",
        "unable to open cache file for save",
        "could not open file for write",
        "build files cannot be regenerated correctly",
        "configure step failed",
        "system error: permission denied",
    ]
    return any(marker in combined for marker in conflict_markers)


def _clip_text_middle(text: str, max_chars: int, marker: str) -> str:

    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= len(marker) + 16:
        return text[:max_chars]
    remaining = max_chars - len(marker)
    head = int(remaining * 0.6)
    tail = remaining - head
    return text[:head].rstrip() + marker + text[-tail:].lstrip()


def _estimate_text_tokens(text: str) -> int:

    if not text:
        return 0
    return (len(text) + PROMPT_TOKEN_CHAR_RATIO - 1) // PROMPT_TOKEN_CHAR_RATIO


# 编译错误摘要曾经在这里被二次压缩到 20 条 (errors/top_errors/error_groups), 在整条

# 提示词超 48000 est. tokens 且本槽位超 12000 字符时触发。该函数已删除, 调用点也已移除:

# 编译错误是修复阶段唯一的定位依据, 截断它会让模型改完看得见的部分、对剩下的一无所知,

# 下一轮必然再失败。超预算时压缩别的槽位(代码、GL 上下文、各阶段输出), 不动错误。


def _trim_jsonish_text(text: str, max_chars: int, marker: str) -> str:

    if len(text) <= max_chars:
        return text
    try:
        payload = json.loads(text)
    except Exception:
        return _clip_text_middle(text, max_chars, marker)


    if isinstance(payload, dict):
        shrunk: dict[str, Any] = {}
        per_field_budget = max(512, max_chars // max(1, len(payload)))
        for key, value in payload.items():
            if isinstance(value, str):
                shrunk[key] = _clip_text_middle(value, min(len(value), per_field_budget), marker)
            elif isinstance(value, list) and len(value) > 8:
                shrunk[key] = value[:8]
            else:
                shrunk[key] = value
        candidate = _json_dump(shrunk)
        if len(candidate) <= max_chars:
            return candidate
    return _clip_text_middle(text, max_chars, marker)


@dataclass

class LLMConnectivityError(RuntimeError):

    stage: str
    attempt_index: int
    original_error: str


@dataclass

class LLMContentError(RuntimeError):

    stage: str
    attempt_index: int
    original_error: str


@dataclass

class CachedCompileResult:

    returncode: int
    stdout_path: str
    stderr_path: str


@dataclass

class ModuleStateSnapshot:

    final_status: str | None
    resume_attempt_index: int | None
    attempts: list[dict[str, Any]]
    source: str


_FATAL_BUILD_LOG_PATTERNS = (

    re.compile(r"^\s*CMake Error\b", re.IGNORECASE),
    re.compile(r"\bBuild files cannot be regenerated correctly\b", re.IGNORECASE),
    re.compile(r"\bNo SOURCES given to target\b", re.IGNORECASE),
    re.compile(r"\bCannot find source file:\b", re.IGNORECASE),
    re.compile(r"\bMSBUILD\s*:\s*error\b", re.IGNORECASE),
    re.compile(r"\bninja:\s*error:\b", re.IGNORECASE),
    re.compile(r"\berror C\d+:", re.IGNORECASE),
    re.compile(r"\bfatal error\b", re.IGNORECASE),
    re.compile(r"\berror:\b", re.IGNORECASE),

)


def _resume_state_path(case_dir: Path) -> Path:

    return case_dir / "meta" / "agent_resume_state.json"


def _load_resume_state(case_dir: Path) -> dict[str, Any] | None:

    path = _resume_state_path(case_dir)
    if not path.exists():
        return None
    return _load_json(path)


def _write_resume_state(case_dir: Path, payload: dict[str, Any]) -> None:

    write_json(_resume_state_path(case_dir), payload)


def _clear_resume_state(case_dir: Path) -> None:

    path = _resume_state_path(case_dir)
    if path.exists():
        path.unlink()


def _looks_like_connectivity_error(error_text: str) -> bool:

    lowered = error_text.lower()
    markers = (
        "httpsconnectionpool",
        "max retries exceeded",
        "failed to establish a new connection",
        "sslzeroreturnerror",
        "connection aborted",
        "connection reset",
        "timed out",
        "read timed out",
        "connect timeout",
        "proxyerror",
        "temporarily unavailable",
        "name or service not known",
        "nodename nor servname provided",
        "eof occurred in violation of protocol",
        "rate limit",
        "429 too many requests",
        "502 bad gateway",
        "503 service unavailable",
        "504 gateway timeout",
    )
    return any(marker in lowered for marker in markers)


def _looks_like_service_error_response(raw_text: str) -> bool:

    lowered = raw_text.strip().lower()
    if not lowered:
        return False


    head = lowered[:600]
    markers = (
        "<html",
        "<!doctype html",
        "502 bad gateway",
        "503 service unavailable",
        "504 gateway timeout",
        "429 too many requests",
        "rate limit exceeded",
        "upstream connect error",
        "proxy error",
        "temporarily unavailable",
    )
    return any(marker in head for marker in markers)


def _extract_schema_matching_json_object(raw_text: str, schema: dict[str, Any], stage: str) -> dict[str, Any]:

    stripped = raw_text.strip()
    candidates: list[dict[str, Any]] = []
    prefer_salvage = False
    if stage in {"compile_only_fix", "regeneration"}:
        if len(_extract_all_json_like_string_fields(raw_text, "class_code", allow_unterminated=True)) > 1:
            prefer_salvage = True
        if len(_extract_all_json_like_string_fields(raw_text, "function_code", allow_unterminated=True)) > 1:
            prefer_salvage = True


    if prefer_salvage:
        payload = _salvage_structured_payload(raw_text)
        if payload is not None:
            effective_schema = _select_schema_for_payload(payload, schema, stage)
            _validate_schema(payload, effective_schema)
            return payload


    if stripped:
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                candidates.append(payload)
        except json.JSONDecodeError:
            pass


    for candidate_text in _iter_json_object_candidates(raw_text):
        try:
            payload = json.loads(candidate_text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            candidates.append(payload)


    last_error: ValueError | None = None
    seen: set[str] = set()
    for payload in candidates:
        marker = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if marker in seen:
            continue
        seen.add(marker)
        try:
            effective_schema = _select_schema_for_payload(payload, schema, stage)
            _validate_schema(payload, effective_schema)
            return payload
        except ValueError as exc:
            last_error = exc


    if last_error is not None:
        raise last_error
    raise ValueError("unable to locate JSON object in LLM response")


def _iter_json_object_candidates(raw_text: str) -> list[str]:

    candidates: list[str] = []


    for fenced in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text, re.IGNORECASE):
        candidate = _extract_first_complete_json_object(fenced)
        if candidate:
            candidates.append(candidate)


    candidate = _extract_first_complete_json_object(raw_text)
    if candidate:
        candidates.append(candidate)


    for start_index in _find_likely_json_start_indices(raw_text):
        candidate = _extract_first_complete_json_object_from_index(raw_text, start_index)
        if candidate:
            candidates.append(candidate)


    return _dedupe_candidates(candidates)


def _extract_first_complete_json_object(text: str) -> str | None:

    start_index = text.find("{")
    if start_index == -1:
        return None
    return _extract_first_complete_json_object_from_index(text, start_index)


def _extract_first_complete_json_object_from_index(text: str, start_index: int) -> str | None:

    if start_index < 0 or start_index >= len(text) or text[start_index] != "{":
        return None


    depth = 0
    in_string = False
    escape = False
    started = False
    for index in range(start_index, len(text)):
        ch = text[index]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue


        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            started = True
            continue
        if ch == "}":
            if not started:
                continue
            depth -= 1
            if depth == 0:
                return text[start_index : index + 1]


    return None


def _find_likely_json_start_indices(text: str) -> list[int]:

    patterns = (
        r'\{\s*"template_type"\s*:',
        r'\{\s*"use_agents"\s*:',
        r'\{\s*"test_objective"\s*:',
        r'\{\s*"semantics_aligned"\s*:',
        r'\{\s*"error_groups"\s*:',
    )
    starts: list[int] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            starts.append(match.start())
    return sorted(set(starts))


def _dedupe_candidates(candidates: list[str]) -> list[str]:

    seen: set[str] = set()
    kept: list[str] = []
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        kept.append(normalized)
    return kept


def _build_result_payload(

    case_dir: Path,
    mode_name: str,
    model_name: str,
    row_id: int,
    max_attempts: int,
    final_status: str,
    attempts: list[dict[str, Any]],

) -> dict[str, Any]:

    return {
        "mode_name": mode_name,
        "model_name": model_name,
        "row_id": row_id,
        "case_dir": str(case_dir),
        "module_name": resolve_package_name(mode_name, model_name),
        "target_name": resolve_target_name(mode_name, model_name),
        "max_attempts": max_attempts,
        "final_status": final_status,
        "attempts": attempts,
        "runtime_case": _build_runtime_case_manifest(mode_name, model_name, row_id),
        "runtime_execution": _load_latest_runtime_execution(case_dir),
        "execution_overrides": _json_safe(get_execution_overrides()),
    }


def _load_latest_runtime_execution(case_dir: Path) -> dict[str, Any] | None:

    compile_root = case_dir / "compile"
    if not compile_root.exists():
        return None
    candidates = sorted(
        compile_root.glob("attempt_*/runtime/runtime_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            payload = _load_json(path)
        except Exception:
            continue
        if isinstance(payload, dict):
            return {**payload, "summary_path": str(path)}
    return None


def _build_local_failure_result(

    *,
    case_dir: Path,
    mode_name: str,
    model_name: str,
    row_id: int,
    max_attempts: int,
    attempts: list[dict[str, Any]],
    stage: str,
    error: str,

) -> dict[str, Any]:

    if stage:
        attempts = [
            *attempts,
            {
                "attempt_index": 0,
                "mode_name": mode_name,
                "model_name": model_name,
                "status": "local_exception",
                "failed_stage": stage,
                "local_error": error,
                "repaired": False,
            },
        ]
    result = _build_result_payload(
        case_dir=case_dir,
        mode_name=mode_name,
        model_name=model_name,
        row_id=row_id,
        max_attempts=max_attempts,
        final_status="failed_local_exception",
        attempts=attempts,
    )
    write_json(case_dir / "meta" / "agent_run_summary.json", result)
    return result


def _load_previous_attempts(case_dir: Path, resume_attempt_index: int) -> list[dict[str, Any]]:

    summary_path = case_dir / "meta" / "agent_run_summary.json"
    if not summary_path.exists():
        return []
    summary = _load_json(summary_path)
    attempts = summary.get("attempts", [])
    if not isinstance(attempts, list):
        return []
    return [attempt for attempt in attempts if int(attempt.get("attempt_index", 0)) < resume_attempt_index]


def _compile_source_artifact_paths(case_dir: Path, attempt_index: int) -> list[Path]:

    if attempt_index <= 1:
        base_dir = case_dir
    else:
        base_dir = case_dir / "attempts" / f"attempt_{attempt_index}"
    return [
        base_dir / "normalized" / "assembled_candidate.cpp",
        base_dir / "meta" / "classification.json",
        base_dir / "meta" / "structure_validation.json",
    ]


def _is_cached_compile_result_current(

    case_dir: Path,
    attempt_index: int,
    stdout_path: Path,
    stderr_path: Path,
    error_summary_path: Path | None,

) -> bool:

    compile_outputs = [stdout_path, stderr_path]
    if error_summary_path is not None:
        compile_outputs.append(error_summary_path)
    if any(not path.exists() for path in compile_outputs):
        return False


    source_paths = [path for path in _compile_source_artifact_paths(case_dir, attempt_index) if path.exists()]
    if not source_paths:
        return True


    newest_source_mtime = max(path.stat().st_mtime for path in source_paths)
    oldest_compile_mtime = min(path.stat().st_mtime for path in compile_outputs)
    return oldest_compile_mtime >= newest_source_mtime


def _load_cached_compile_result(case_dir: Path, attempt_index: int) -> CachedCompileResult | None:

    compile_dir = case_dir / "compile" / f"attempt_{attempt_index}"
    stdout_path = compile_dir / "build_stdout.txt"
    stderr_path = compile_dir / "build_stderr.txt"
    error_summary_path = compile_dir / "error_summary.json"
    stage2_record_path = compile_dir / "compile_record.json"
    if stage2_record_path.exists():
        try:
            stage2_record = _load_json(stage2_record_path)
            stage2_summary = _load_json(error_summary_path)
            recorded_attempt = int(stage2_record.get("attempt_index", -1))
            recorded_returncode = int(stage2_record["returncode"])
            # 摘要现在只有 errors, 不再落 build_returncode。原先这里读那个字段做交叉
            # 校验, 字段一撤就是 KeyError —— 而它被下面的 except Exception 吞掉, 表现
            # 为每个 Stage 2 缓存命中都退化成慢路径并打一行 Invalid, 不会报错只会变慢。
            # 交叉校验改用「有没有错误」: 有错必然非 0, 没错必然 0。
            summary_failed = summary_error_count(stage2_summary) > 0
            summary_consistent = summary_failed == (recorded_returncode != 0)
            recorded_outcome = str(stage2_record.get("compile_outcome") or "")
            outcome_matches = (
                (recorded_returncode == 0 and recorded_outcome == "compiled")
                or (recorded_returncode != 0 and recorded_outcome == "failed")
            )
            if (
                recorded_attempt == int(attempt_index)
                and summary_consistent
                and outcome_matches
                and stdout_path.exists()
                and stderr_path.exists()
            ):
                # Stage 2 writes compile evidence and then refreshes structure
                # metadata.  That metadata refresh can legitimately have a
                # newer mtime than the build logs, so compile_record.json is
                # the authoritative freshness/ownership marker here.
                return CachedCompileResult(
                    returncode=recorded_returncode,
                    stdout_path=str(stdout_path),
                    stderr_path=str(stderr_path),
                )
            print(
                f"[Cache] Ignoring inconsistent Stage 2 compile record for "

                f"attempt {attempt_index}"
            )
        except Exception as exc:
            print(
                f"[Cache] Invalid Stage 2 compile record for attempt "

                f"{attempt_index}: {type(exc).__name__}: {exc}"
            )
    if stdout_path.exists() and stderr_path.exists():
        if not _is_cached_compile_result_current(case_dir, attempt_index, stdout_path, stderr_path, error_summary_path):
            print(f"[Cache] Ignoring stale cached compile result for attempt {attempt_index}")
            return None
        returncode = _infer_compile_result_from_artifacts(stdout_path, stderr_path, error_summary_path)
        if (returncode == 0 and resolve_run_after_compile()
                and not (compile_dir / "runtime" / "runtime_summary.json").exists()):
            print(f"[Cache] Compile succeeded but runtime evidence is missing for attempt {attempt_index}")
            return None
        return CachedCompileResult(
            returncode=returncode,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )


    summary_path = case_dir / "meta" / "agent_run_summary.json"
    if not summary_path.exists():
        return None
    summary = _load_json(summary_path)
    for attempt in summary.get("attempts", []):
        if int(attempt.get("attempt_index", -1)) != int(attempt_index):
            continue
        summary_stdout = Path(str(attempt.get("compile_stdout_path") or ""))
        summary_stderr = Path(str(attempt.get("compile_stderr_path") or ""))
        summary_error = Path(str(attempt.get("compile_error_summary_path") or ""))
        if not _is_cached_compile_result_current(case_dir, attempt_index, summary_stdout, summary_stderr, summary_error):
            print(f"[Cache] Ignoring stale summary compile result for attempt {attempt_index}")
            return None
        returncode = int(attempt["compile_returncode"])
        if (returncode == 0 and resolve_run_after_compile()
                and not (compile_dir / "runtime" / "runtime_summary.json").exists()):
            print(f"[Cache] Summary compile succeeded but runtime evidence is missing for attempt {attempt_index}")
            return None
        return CachedCompileResult(
            returncode=returncode,
            stdout_path=str(summary_stdout),
            stderr_path=str(summary_stderr),
        )
    return None


def _load_latest_compile_evidence(case_dir: Path) -> tuple[int, int, Path] | None:

    compile_root = case_dir / "compile"
    if not compile_root.exists():
        return None


    latest: tuple[int, int, Path] | None = None
    for attempt_dir in compile_root.glob("attempt_*"):
        if not attempt_dir.is_dir():
            continue
        try:
            attempt_index = int(attempt_dir.name.split("_")[1])
        except Exception:
            continue
        summary_path = attempt_dir / "error_summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = _load_json(summary_path)
            error_count = summary_error_count(summary)
        except Exception:
            continue
        if latest is None or attempt_index > latest[0]:
            latest = (attempt_index, error_count, summary_path)
    return latest


def _read_compile_error_count(summary_path: Path) -> int | None:

    if not summary_path.exists():
        return None
    try:
        summary = _load_json(summary_path)
    except Exception:
        return None
    try:
        return summary_error_count(summary)
    except Exception:
        return None


def _load_summary_compile_evidence(summary: dict[str, Any], case_dir: Path) -> tuple[int, int, Path] | None:

    attempts = summary.get("attempts", [])
    if not isinstance(attempts, list):
        return None


    ranked_attempts = sorted(
        (attempt for attempt in attempts if isinstance(attempt, dict)),
        key=lambda attempt: int(attempt.get("attempt_index", -1)),
        reverse=True,
    )
    for attempt in ranked_attempts:
        attempt_index = int(attempt.get("attempt_index", -1))
        candidate_paths = [
            attempt.get("compile_error_summary_path"),
            attempt.get("input_compile_error_summary_path"),
        ]
        for raw_path in candidate_paths:
            if not raw_path:
                continue
            summary_path = Path(str(raw_path))
            if not summary_path.is_absolute():
                summary_path = case_dir / summary_path
            error_count = _read_compile_error_count(summary_path)
            if error_count is not None:
                return attempt_index, error_count, summary_path
    return None


def _resolve_effective_final_status(summary: dict[str, Any], case_dir: Path) -> str:

    final_status = str(summary.get("final_status") or "")
    latest_compile = _load_summary_compile_evidence(summary, case_dir)
    if latest_compile is None:
        latest_compile = _load_latest_compile_evidence(case_dir)
    if latest_compile is None:
        return final_status


    attempt_index, _, _ = latest_compile
    inferred_returncode = _infer_compile_stage_returncode(case_dir, attempt_index)
    if inferred_returncode is None:
        return final_status
    compile_succeeded = inferred_returncode == 0


    if final_status == "interrupted_llm_connectivity":
        return final_status
    if compile_succeeded and final_status in {
        "failed",
        "failed_llm_content",
        "failed_local_exception",
        "recovered_retry_ready",
        "compiled",
    }:
        return "compiled"
    if not compile_succeeded and final_status == "compiled":
        return "failed"
    return final_status


def _matches_allowed_types(value: Any, allowed_types: list[str]) -> bool:

    for allowed_type in allowed_types:
        if allowed_type == "string" and isinstance(value, str):
            return True
        if allowed_type == "boolean" and isinstance(value, bool):
            return True
        if allowed_type == "object" and isinstance(value, dict):
            return True
        if allowed_type == "array" and isinstance(value, list):
            return True
        if allowed_type == "null" and value is None:
            return True
    return False


def _validate_schema(payload: dict[str, Any], schema: dict[str, Any]) -> None:

    required = schema.get("required", [])
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"payload missing required fields: {missing}")


    properties = schema.get("properties", {})
    for name, spec in properties.items():
        if name not in payload:
            continue
        allowed_types = spec.get("type")
        if allowed_types is None:
            continue
        if not isinstance(allowed_types, list):
            allowed_types = [allowed_types]
        if not _matches_allowed_types(payload[name], allowed_types):
            raise ValueError(f"field {name} has unexpected type: {type(payload[name]).__name__}")
        enum_values = spec.get("enum")
        if enum_values is not None and payload[name] not in enum_values:
            raise ValueError(f"field {name} has unexpected value: {payload[name]}")


def _extract_json_like_string_field(text: str, key: str, *, allow_unterminated: bool = False) -> str | None:

    marker = f'"{key}"'
    key_index = text.find(marker)
    if key_index == -1:
        return None


    colon_index = text.find(":", key_index + len(marker))
    if colon_index == -1:
        return None


    value_start = _find_next_non_whitespace(text, colon_index + 1)
    if value_start == -1 or text[value_start] != '"':
        return None


    chars: list[str] = []
    escape = False
    for index in range(value_start + 1, len(text)):
        ch = text[index]
        if escape:
            chars.append("\\" + ch)
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            raw = "".join(chars)
            try:
                return json.loads(f'"{raw}"')
            except json.JSONDecodeError:
                return raw
        chars.append(ch)


    if allow_unterminated:
        raw = "".join(chars)
        raw = raw.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
        return raw
    return None


def _extract_all_json_like_string_fields(text: str, key: str, *, allow_unterminated: bool = False) -> list[str]:

    values: list[str] = []
    marker = f'"{key}"'
    search_start = 0
    while True:
        key_index = text.find(marker, search_start)
        if key_index == -1:
            break
        candidate = _extract_json_like_string_field(text[key_index:], key, allow_unterminated=allow_unterminated)
        if isinstance(candidate, str):
            stripped = candidate.strip()
            if stripped:
                values.append(candidate)
        search_start = key_index + len(marker)
    return values


def _find_next_non_whitespace(text: str, start: int) -> int:

    for index in range(start, len(text)):
        if not text[index].isspace():
            return index
    return -1


def _extract_json_like_bool_field(text: str, key: str) -> bool | None:

    match = re.search(rf'"{re.escape(key)}"\s*:\s*(true|false)', text)
    if not match:
        return None
    return match.group(1) == "true"


def _extract_json_like_nullable_string_field(text: str, key: str) -> str | None:

    null_match = re.search(rf'"{re.escape(key)}"\s*:\s*null', text)
    if null_match:
        return None
    return _extract_json_like_string_field(text, key)


def _extract_json_like_array_field(text: str, key: str) -> list[Any] | None:

    marker_match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', text)
    if not marker_match:
        return None
    start_index = marker_match.end() - 1


    depth = 0
    in_string = False
    escape = False
    for index in range(start_index, len(text)):
        ch = text[index]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue


        if ch == '"':
            in_string = True
            continue
        if ch == "[":
            depth += 1
            continue
        if ch == "]":
            depth -= 1
            if depth == 0:
                candidate = text[start_index : index + 1]
                try:
                    value = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, list) else None
    return None


def _salvage_structured_payload(raw_text: str) -> dict[str, Any] | None:

    template_type = _extract_json_like_string_field(raw_text, "template_type")
    if template_type not in {"class", "function"}:
        return None


    payload: dict[str, Any] = {"template_type": template_type}
    if template_type == "class":
        class_code_candidates = _extract_all_json_like_string_fields(raw_text, "class_code", allow_unterminated=True)
        if not class_code_candidates:
            return None
        payload["params_decl_code"] = _extract_json_like_string_field(raw_text, "params_decl_code") or ""
        payload["has_params"] = _extract_json_like_bool_field(raw_text, "has_params")
        payload["params_type"] = _extract_json_like_nullable_string_field(raw_text, "params_type")
        payload["register_cases"] = _extract_json_like_array_field(raw_text, "register_cases") or []
        if payload["has_params"] is None:
            payload["has_params"] = bool(payload["params_type"] or payload["register_cases"])
        for class_code in class_code_candidates:
            candidate_payload = dict(payload)
            candidate_payload["class_code"] = class_code
            try:
                normalized_candidate = _normalize_payload_before_schema_validation(
                    candidate_payload,
                    stage="compile_only_fix",
                    fallback_schema=CLASS_REPAIR_SCHEMA,
                )
                repaired_body = _build_repaired_body(_normalize_payload_consistency(normalized_candidate))
                _build_locked_classification(
                    mode_name="only-fix",
                    model_name="",
                    row_id=-1,
                    payload=_normalize_payload_consistency(normalized_candidate),
                    repaired_body=repaired_body,
                )
                return normalized_candidate
            except Exception:
                continue
        payload["class_code"] = max(class_code_candidates, key=len)
        return payload


    function_code_candidates = _extract_all_json_like_string_fields(raw_text, "function_code", allow_unterminated=True)
    if not function_code_candidates:
        return None
    payload["check_support_code"] = _extract_json_like_string_field(raw_text, "check_support_code") or ""
    payload["init_programs_code"] = _extract_json_like_string_field(raw_text, "init_programs_code") or ""
    payload["params_decl_code"] = _extract_json_like_string_field(raw_text, "params_decl_code") or ""
    payload["has_params"] = _extract_json_like_bool_field(raw_text, "has_params")
    payload["has_check_support"] = _extract_json_like_bool_field(raw_text, "has_check_support")
    payload["has_init_programs"] = _extract_json_like_bool_field(raw_text, "has_init_programs")
    payload["params_type"] = _extract_json_like_nullable_string_field(raw_text, "params_type")
    if payload["has_params"] is None:
        payload["has_params"] = bool(payload["params_type"])
    if payload["has_check_support"] is None:
        payload["has_check_support"] = bool(payload["check_support_code"].strip())
    if payload["has_init_programs"] is None:
        payload["has_init_programs"] = bool(payload["init_programs_code"].strip())
    for function_code in function_code_candidates:
        candidate_payload = dict(payload)
        candidate_payload["function_code"] = function_code
        try:
            normalized_candidate = _normalize_payload_before_schema_validation(
                candidate_payload,
                stage="compile_only_fix",
                fallback_schema=FUNCTION_REPAIR_SCHEMA,
            )
            repaired_body = _build_repaired_body(_normalize_payload_consistency(normalized_candidate))
            _build_locked_classification(
                mode_name="only-fix",
                model_name="",
                row_id=-1,
                payload=_normalize_payload_consistency(normalized_candidate),
                repaired_body=repaired_body,
            )
            return normalized_candidate
        except Exception:
            continue
    payload["function_code"] = max(function_code_candidates, key=len)
    return payload


def _extract_first_fenced_code_block(raw_text: str) -> str | None:

    match = re.search(r"```(?:cpp|c\+\+|c)?\s*([\s\S]*?)\s*```", raw_text, re.IGNORECASE)
    if not match:
        return None
    code = match.group(1).strip()
    return code or None


def _trim_trailing_non_code_sections(code: str) -> str:

    trimmed = code.strip()
    if not trimmed:
        return trimmed


    trailing_markers = (
        "\n1. ",
        "\n2. ",
        "\n3. ",
        "\n4. ",
        "\n5. ",
        "\n6. ",
        "\n7. ",
        "\n8. ",
        "\n9. ",
        "\n10. ",
        "\n- ",
        "\n* ",
        "\n**",
        "\n### ",
        "\n## ",
        "\nAnalyze the Request",
        "\nFinal Code Assembly",
    )
    cut_positions = [trimmed.find(marker) for marker in trailing_markers if trimmed.find(marker) != -1]
    if cut_positions:
        trimmed = trimmed[: min(cut_positions)].rstrip()


    namespace_comment = re.search(r"\n\}\s*//\s*\w[\w\s:.-]*\n\}\s*//\s*\w[\w\s:.-]*", trimmed)
    if namespace_comment:
        end = namespace_comment.end()
        trimmed = trimmed[:end].rstrip()


    return trimmed


def _prepare_raw_cpp_candidate(raw_text: str) -> str:

    candidate = _extract_first_fenced_code_block(raw_text) or raw_text
    candidate = _trim_trailing_non_code_sections(candidate)
    return candidate.strip()


def _is_placeholder_code_text(code_text: str) -> bool:

    stripped = code_text.strip()
    if not stripped:
        return True
    if stripped in {".", "..", "..."}:
        return True
    collapsed = re.sub(r"\s+", " ", stripped)
    return collapsed in {"...", "class ...", "function ..."}


def _validate_repair_payload_usability(

    payload: dict[str, Any],
    *,
    stage: str,
    mode_name: str,
    model_name: str,
    row_id: int,

) -> None:

    if stage not in {"compile_only_fix", "regeneration"}:
        return


    template_type = payload.get("template_type")
    code_text = payload.get("class_code") if template_type == "class" else payload.get("function_code")
    if not isinstance(code_text, str) or _is_placeholder_code_text(code_text):
        raise ValueError("repair payload did not contain usable generated code")


    cleaned = clean_generated_code(code_text, decode_escaped_newlines=False)
    cleaned_code = str(cleaned.get("cleaned_code") or "").strip()
    if _is_placeholder_code_text(cleaned_code):
        raise ValueError("repair payload did not contain usable generated code")


def _validate_repair_payload_applicability(

    payload: dict[str, Any],
    *,
    mode_name: str,
    model_name: str,
    row_id: int,

) -> None:

    normalized_payload = _normalize_payload_consistency(payload)
    validate_payload(normalized_payload)
    repaired_body = _build_repaired_body(normalized_payload)
    _build_locked_classification(
        mode_name=mode_name,
        model_name=model_name,
        row_id=row_id,
        payload=normalized_payload,
        repaired_body=repaired_body,
    )


def _recover_code_payload_from_raw_cpp(

    *,
    raw_text: str,
    stage: str,
    mode_name: str,
    model_name: str,
    row_id: int,

) -> dict[str, Any] | None:

    if stage not in {"compile_only_fix", "regeneration"}:
        return None


    raw_cpp_candidate = _prepare_raw_cpp_candidate(raw_text)
    if not raw_cpp_candidate:
        return None


    cleaned = clean_generated_code(raw_cpp_candidate, decode_escaped_newlines=False)
    cleaned_code = str(cleaned.get("cleaned_code") or "").strip()
    if not cleaned_code:
        return None


    try:
        classification = classify_candidate_code(
            cleaned_code,
            mode_name=mode_name,
            model_name=model_name,
            row_id=row_id,
        )
    except Exception:
        return None


    if classification.template_type == "class":
        return {
            "template_type": "class",
            "class_code": cleaned_code,
            "params_decl_code": "",
            "has_params": classification.has_params,
            "params_type": classification.params_type,
            "register_cases": classification.register_cases or [],
        }


    return {
        "template_type": "function",
        "function_code": cleaned_code,
        "check_support_code": "",
        "init_programs_code": "",
        "params_decl_code": "",
        "has_params": classification.has_params,
        "has_check_support": classification.has_check_support,
        "has_init_programs": classification.has_init_programs,
        "params_type": classification.params_type,
        # 只第一项有效, 供 params_init 用; 叶子名仍由脚本定
        "register_cases": (classification.register_cases or [])[:1],
    }


def _select_schema_for_payload(payload: dict[str, Any], fallback_schema: dict[str, Any], stage: str) -> dict[str, Any]:

    if stage not in {"compile_only_fix", "regeneration"}:
        return fallback_schema
    template_type = payload.get("template_type")
    if template_type == "class":
        return CLASS_REPAIR_SCHEMA if stage == "compile_only_fix" else REGENERATION_CLASS_SCHEMA
    if template_type == "function":
        return FUNCTION_REPAIR_SCHEMA if stage == "compile_only_fix" else REGENERATION_FUNCTION_SCHEMA
    return fallback_schema


def _normalize_register_cases_payload(raw_cases: Any) -> list[dict[str, str]]:

    if raw_cases is None:
        return []
    if isinstance(raw_cases, dict):
        raw_cases = [raw_cases]
    if not isinstance(raw_cases, list):
        return []


    normalized_cases: list[dict[str, str]] = []
    for index, case in enumerate(raw_cases, start=1):
        if isinstance(case, str):
            params_init = case.strip()
            if params_init:
                normalized_cases.append({"case_name": f"case_{index}", "params_init": params_init})
            continue
        if not isinstance(case, dict):
            continue
        case_name = str(case.get("case_name") or f"case_{index}").strip()
        params_init = str(case.get("params_init") or "").strip()
        if not params_init:
            continue
        normalized_cases.append({"case_name": case_name or f"case_{index}", "params_init": params_init})
    return normalized_cases


def _normalize_recovered_repair_payload(payload: dict[str, Any]) -> dict[str, Any]:

    normalized = dict(payload)
    template_type = normalized.get("template_type")
    if template_type not in {"class", "function"}:
        if isinstance(normalized.get("class_code"), str) and normalized.get("class_code", "").strip():
            template_type = "class"
        elif isinstance(normalized.get("function_code"), str) and normalized.get("function_code", "").strip():
            template_type = "function"
        normalized["template_type"] = template_type


    if template_type == "class":
        normalized["class_code"] = str(normalized.get("class_code") or "")
        normalized["params_decl_code"] = str(normalized.get("params_decl_code") or "")
        has_params = normalized.get("has_params")
        if not isinstance(has_params, bool):
            has_params = bool(
                normalized.get("params_type")
                or normalized["params_decl_code"].strip()
                or normalized.get("register_cases")
            )
        normalized["has_params"] = has_params
        params_type = normalized.get("params_type")
        normalized["params_type"] = params_type if isinstance(params_type, str) else None
        normalized["register_cases"] = _normalize_register_cases_payload(normalized.get("register_cases"))
        return normalized


    if template_type == "function":
        normalized["function_code"] = str(normalized.get("function_code") or "")
        normalized["check_support_code"] = str(normalized.get("check_support_code") or "")
        normalized["init_programs_code"] = str(normalized.get("init_programs_code") or "")
        normalized["params_decl_code"] = str(normalized.get("params_decl_code") or "")
        has_params = normalized.get("has_params")
        if not isinstance(has_params, bool):
            has_params = bool(
                normalized.get("params_type")
                or normalized["params_decl_code"].strip()
                or normalized.get("register_cases")
            )
        normalized["has_params"] = has_params
        has_check_support = normalized.get("has_check_support")
        if not isinstance(has_check_support, bool):
            has_check_support = bool(normalized["check_support_code"].strip())
        normalized["has_check_support"] = has_check_support
        has_init_programs = normalized.get("has_init_programs")
        if not isinstance(has_init_programs, bool):
            has_init_programs = bool(normalized["init_programs_code"].strip())
        normalized["has_init_programs"] = has_init_programs
        params_type = normalized.get("params_type")
        normalized["params_type"] = params_type if isinstance(params_type, str) else None
        # function 挂一个叶子, 只留第一项供 params_init 使用, 见 template_manager
        # ._function_params_init。整体丢弃会让参数退化成全零的 {}。
        normalized["register_cases"] = _normalize_register_cases_payload(
            normalized.get("register_cases")
        )[:1]
        return normalized


    return normalized


def _normalize_payload_before_schema_validation(

    payload: dict[str, Any],
    *,
    stage: str,
    fallback_schema: dict[str, Any],

) -> dict[str, Any]:

    if stage not in {"compile_only_fix", "regeneration"}:
        return payload
    normalized = _normalize_recovered_repair_payload(payload)
    effective_schema = _select_schema_for_payload(normalized, fallback_schema, stage)
    for field in effective_schema.get("required", []):
        if field not in normalized:
            if field in {"params_decl_code", "check_support_code", "init_programs_code"}:
                normalized[field] = ""
            elif field == "register_cases":
                normalized[field] = []
            elif field == "has_params":
                normalized[field] = bool(
                    normalized.get("params_type")
                    or str(normalized.get("params_decl_code") or "").strip()
                    or normalized.get("register_cases")
                )
            elif field == "template_type":
                if isinstance(normalized.get("class_code"), str) and normalized.get("class_code", "").strip():
                    normalized[field] = "class"
                elif isinstance(normalized.get("function_code"), str) and normalized.get("function_code", "").strip():
                    normalized[field] = "function"
    return normalized


def _coerce_repair_payload_for_compile(

    payload: dict[str, Any] | None,
    *,
    stage: str,
    fallback_schema: dict[str, Any],

) -> dict[str, Any]:

    normalized = _normalize_payload_before_schema_validation(dict(payload or {}), stage=stage, fallback_schema=fallback_schema)
    template_type = normalized.get("template_type")
    if template_type not in {"class", "function"}:
        template_spec = fallback_schema.get("properties", {}).get("template_type", {})
        enum_values = template_spec.get("enum") if isinstance(template_spec, dict) else None
        if isinstance(enum_values, list) and len(enum_values) == 1 and enum_values[0] in {"class", "function"}:
            template_type = enum_values[0]
        else:
            template_type = "class" if "class_code" in normalized else "function"
        normalized["template_type"] = template_type


    if template_type == "class":
        normalized["class_code"] = str(normalized.get("class_code") or "")
        normalized["params_decl_code"] = str(normalized.get("params_decl_code") or "")
        if not isinstance(normalized.get("has_params"), bool):
            normalized["has_params"] = bool(
                normalized.get("params_type")
                or normalized["params_decl_code"].strip()
                or normalized.get("register_cases")
            )
        normalized["register_cases"] = _normalize_register_cases_payload(normalized.get("register_cases"))
        return normalized


    normalized["function_code"] = str(normalized.get("function_code") or "")
    normalized["check_support_code"] = str(normalized.get("check_support_code") or "")
    normalized["init_programs_code"] = str(normalized.get("init_programs_code") or "")
    normalized["params_decl_code"] = str(normalized.get("params_decl_code") or "")
    if not isinstance(normalized.get("has_params"), bool):
        normalized["has_params"] = bool(normalized.get("params_type") or normalized["params_decl_code"].strip())
    if not isinstance(normalized.get("has_check_support"), bool):
        normalized["has_check_support"] = bool(normalized["check_support_code"].strip())
    if not isinstance(normalized.get("has_init_programs"), bool):
        normalized["has_init_programs"] = bool(normalized["init_programs_code"].strip())
    # function 挂一个叶子, 只留第一项供 params_init 使用, 见 template_manager
    # ._function_params_init。整体丢弃会让参数退化成全零的 {}。
    normalized["register_cases"] = _normalize_register_cases_payload(
        normalized.get("register_cases")
    )[:1]
    return normalized


def _recover_payload_from_raw_response(

    raw_path: Path,
    schema: dict[str, Any],
    stage: str,
    *,
    mode_name: str,
    model_name: str,
    row_id: int,

) -> dict[str, Any]:

    raw_text = _read_text(raw_path)
    try:
        payload = _extract_schema_matching_json_object(raw_text, schema, stage)
    except ValueError:
        payload = _salvage_structured_payload(raw_text)
        if payload is None:
            payload = _recover_code_payload_from_raw_cpp(
                raw_text=raw_text,
                stage=stage,
                mode_name=mode_name,
                model_name=model_name,
                row_id=row_id,
            )
            if payload is None:
                raise
    payload = _normalize_payload_before_schema_validation(payload, stage=stage, fallback_schema=schema)
    effective_schema = _select_schema_for_payload(payload, schema, stage)
    _validate_schema(payload, effective_schema)
    return payload


def _load_existing_summary(case_dir: Path) -> dict[str, Any] | None:

    summary_path = case_dir / "meta" / "agent_run_summary.json"
    if not summary_path.exists():
        return None
    return _load_json(summary_path)


def _build_attempt_compile_record(

    case_dir: Path,
    attempt_index: int,
    *,
    input_compile_index: int,
    output_compile_index: int,
    status: str,

) -> dict[str, Any]:

    input_stdout_path, input_stderr_path, input_summary_path = _compile_stage_paths(case_dir, input_compile_index)
    output_stdout_path, output_stderr_path, output_summary_path = _compile_stage_paths(case_dir, output_compile_index)
    input_compile_returncode = _infer_compile_stage_returncode(case_dir, input_compile_index)
    output_compile_returncode = _infer_compile_stage_returncode(case_dir, output_compile_index)
    return {
        "attempt_index": attempt_index,
        "status": status,
        "input_compile_returncode": input_compile_returncode,
        "input_compile_stdout_path": str(input_stdout_path),
        "input_compile_stderr_path": str(input_stderr_path),
        "input_compile_error_summary_path": str(input_summary_path),
        "compile_returncode": output_compile_returncode,
        "compile_stdout_path": str(output_stdout_path),
        "compile_stderr_path": str(output_stderr_path),
        "compile_error_summary_path": str(output_summary_path),
        "repaired": output_compile_index != input_compile_index,
    }


def _scan_case_modules(case_dir: Path, mode_name: str, max_attempts: int) -> ModuleStateSnapshot | None:

    if not _is_compile_stage_complete(case_dir, 1):
        return None


    attempts: list[dict[str, Any]] = []
    for attempt_index in range(1, max_attempts + 1):
        input_compile_index = attempt_index
        input_compile_returncode = _infer_compile_stage_returncode(case_dir, input_compile_index)
        if input_compile_returncode is None:
            break


        if input_compile_returncode == 0:
            attempts.append(
                _build_attempt_compile_record(
                    case_dir,
                    attempt_index,
                    input_compile_index=input_compile_index,
                    output_compile_index=input_compile_index,
                    status="compiled",
                )
            )
            return ModuleStateSnapshot(
                final_status="compiled",
                resume_attempt_index=None,
                attempts=attempts,
                source="attempt_modules",
            )


        has_feedback = _has_feedback_outputs_for_attempt(case_dir, mode_name, attempt_index)
        has_next_attempt_artifacts = _is_applied_attempt_complete(case_dir, attempt_index + 1)
        repaired_compile_index = attempt_index + 1
        has_repaired_compile = _is_compile_stage_complete(case_dir, repaired_compile_index)
        if not has_feedback or not has_next_attempt_artifacts or not has_repaired_compile:
            attempts.append(
                _build_attempt_compile_record(
                    case_dir,
                    attempt_index,
                    input_compile_index=input_compile_index,
                    output_compile_index=input_compile_index,
                    status="repaired_for_next_attempt",
                )
            )
            return ModuleStateSnapshot(
                final_status="recovered_retry_ready",
                resume_attempt_index=attempt_index,
                attempts=attempts,
                source="attempt_modules",
            )


        repaired_compile_returncode = _infer_compile_stage_returncode(case_dir, repaired_compile_index)
        is_final_attempt = attempt_index >= max_attempts
        if repaired_compile_returncode == 0:
            attempts.append(
                _build_attempt_compile_record(
                    case_dir,
                    attempt_index,
                    input_compile_index=input_compile_index,
                    output_compile_index=repaired_compile_index,
                    status="compiled",
                )
            )
            return ModuleStateSnapshot(
                final_status="compiled",
                resume_attempt_index=None,
                attempts=attempts,
                source="attempt_modules",
            )


        attempts.append(
            _build_attempt_compile_record(
                case_dir,
                attempt_index,
                input_compile_index=input_compile_index,
                output_compile_index=repaired_compile_index,
                status="compile_failed_max_attempts" if is_final_attempt else "repaired_for_next_attempt",
            )
        )
        if is_final_attempt:
            return ModuleStateSnapshot(
                final_status="failed",
                resume_attempt_index=None,
                attempts=attempts,
                source="attempt_modules",
            )


    return None


def _apply_module_state_summary(

    *,
    case_dir: Path,
    mode_name: str,
    model_name: str,
    row_id: int,
    max_attempts: int,
    module_state: ModuleStateSnapshot,

) -> dict[str, Any]:

    result = _build_result_payload(
        case_dir=case_dir,
        mode_name=mode_name,
        model_name=model_name,
        row_id=row_id,
        max_attempts=max_attempts,
        final_status=module_state.final_status or "failed",
        attempts=module_state.attempts,
    )
    result["reconstructed_from"] = module_state.source
    if module_state.resume_attempt_index is not None:
        result["resume_attempt_index"] = module_state.resume_attempt_index
    write_json(case_dir / "meta" / "agent_run_summary.json", result)
    return result


def _refresh_summary_from_modules(

    *,
    case_dir: Path,
    mode_name: str,
    model_name: str,
    row_id: int,
    max_attempts: int,

) -> tuple[dict[str, Any] | None, ModuleStateSnapshot | None]:

    existing_summary = _load_existing_summary(case_dir)
    module_state = _scan_case_modules(case_dir, mode_name, max_attempts)
    if module_state is None:
        return existing_summary, None


    existing_final_status = str(existing_summary.get("final_status") or "") if isinstance(existing_summary, dict) else ""
    existing_attempt_count = len(existing_summary.get("attempts", [])) if isinstance(existing_summary, dict) else -1
    if (
        existing_summary is None
        or existing_final_status != (module_state.final_status or "")
        or existing_attempt_count != len(module_state.attempts)
        or existing_summary.get("reconstructed_from") == "attempt_modules"
    ):
        existing_summary = _apply_module_state_summary(
            case_dir=case_dir,
            mode_name=mode_name,
            model_name=model_name,
            row_id=row_id,
            max_attempts=max_attempts,
            module_state=module_state,
        )
    return existing_summary, module_state


def _has_feedback_outputs_for_attempt(case_dir: Path, mode_name: str, attempt_index: int) -> bool:

    feedback_dir = case_dir / "feedback" / f"attempt_{attempt_index}"
    if resolve_repair_strategy(mode_name) == "feedback":
        return (feedback_dir / "repair_response_extracted.json").exists()
    return (
        (feedback_dir / "director_response_extracted.json").exists()
        and (feedback_dir / "regeneration_response_extracted.json").exists()
    )


def _resolve_repair_stage_schema(case_dir: Path, attempt_record: dict[str, Any], failed_stage: str) -> tuple[str, dict[str, Any]]:

    prefix = "repair" if failed_stage == "compile_only_fix" else "regeneration"


    if prefix == "repair":
        rejudge_path = attempt_record.get("rejudge_path")
        if rejudge_path and Path(str(rejudge_path)).exists():
            rejudge_payload = _load_json(Path(str(rejudge_path)))
            if rejudge_payload.get("template_type") == "class":
                return prefix, CLASS_REPAIR_SCHEMA
        return prefix, FUNCTION_REPAIR_SCHEMA


    attempt_index = int(attempt_record.get("attempt_index", 1))
    classification_path = (
        case_dir / "meta" / "classification.json"
        if attempt_index <= 1
        else case_dir / "attempts" / f"attempt_{attempt_index}" / "meta" / "classification.json"
    )
    if classification_path.exists() and _load_json(classification_path).get("template_type") == "class":
        return prefix, REGENERATION_CLASS_SCHEMA
    return prefix, REGENERATION_FUNCTION_SCHEMA


def _try_recover_failed_stage_payload(

    *,
    case_dir: Path,
    mode_name: str,
    model_name: str,
    row_id: int,
    attempt_record: dict[str, Any],
    failed_stage: str,
    max_attempts: int,
    develop: bool,

) -> tuple[dict[str, Any], int | None] | None:

    attempt_index = int(attempt_record.get("attempt_index", 0))
    if attempt_index <= 0:
        return None


    prefix, schema = _resolve_repair_stage_schema(case_dir, attempt_record, failed_stage)
    feedback_dir = case_dir / "feedback" / f"attempt_{attempt_index}"
    raw_path = feedback_dir / f"{prefix}_response_raw.txt"
    if not raw_path.exists():
        return None


    payload = _recover_payload_from_raw_response(
        raw_path,
        schema,
        failed_stage,
        mode_name=mode_name,
        model_name=model_name,
        row_id=row_id,
    )
    extracted_path = feedback_dir / f"{prefix}_response_extracted.json"
    write_json(extracted_path, payload)


    recovered_record = dict(attempt_record)
    recovered_record[f"{prefix}_path"] = str(extracted_path)
    recovered_record["recovered_from_raw_response"] = True
    recovered_record["recovered_source_raw_path"] = str(raw_path)


    next_attempt = attempt_index + 1
    if next_attempt <= max_attempts:
        apply_repair_payload(
            mode_name=mode_name,
            model_name=model_name,
            row_id=row_id,
            payload=payload,
            attempt_index=next_attempt,
            develop=develop,
        )
        recovered_record["status"] = "recovered_repair_payload_for_retry"
        recovered_record["repaired"] = True
        recovered_record["recovery_resume_attempt"] = next_attempt
        return recovered_record, next_attempt


    return None


def _recover_failed_llm_content_attempt(

    *,
    case_dir: Path,
    mode_name: str,
    model_name: str,
    row_id: int,
    max_attempts: int,
    develop: bool,

) -> tuple[list[dict[str, Any]], int] | None:

    summary = _load_existing_summary(case_dir)
    if not summary:
        return None


    if summary.get("final_status") == "recovered_retry_ready":
        attempts = summary.get("attempts", [])
        if not isinstance(attempts, list) or not attempts:
            return None
        recovery_resume_attempt = attempts[-1].get("recovery_resume_attempt")
        if isinstance(recovery_resume_attempt, int):
            attempt_cpp = (
                case_dir / "attempts" / f"attempt_{recovery_resume_attempt}" / "normalized" / "assembled_candidate.cpp"
            )
            if attempt_cpp.exists():
                return attempts, recovery_resume_attempt
        return None


    attempts = summary.get("attempts", [])
    if not isinstance(attempts, list) or not attempts:
        return None


    if summary.get("final_status") not in {"failed_llm_content", "failed"}:
        return None


    attempt_record = attempts[-1]
    attempt_index = int(attempt_record.get("attempt_index", 0))
    failed_stage = str(attempt_record.get("failed_stage") or "")
    llm_error = str(attempt_record.get("llm_error") or "")
    retryable_error_markers = (
        "payload missing required fields",
        "unable to locate JSON object",
        "repair payload did not contain usable generated code",
        "unable to classify candidate as class or function",
        "register_cases must be an array when provided",
    )
    if failed_stage not in {"compile_only_fix", "regeneration"} and not any(
        marker in llm_error for marker in retryable_error_markers
    ):
        return None


    recovered = _try_recover_failed_stage_payload(
        case_dir=case_dir,
        mode_name=mode_name,
        model_name=model_name,
        row_id=row_id,
        attempt_record=attempt_record,
        failed_stage=failed_stage,
        max_attempts=max_attempts,
        develop=develop,
    )
    if recovered is None:
        return None


    recovered_record, resume_attempt = recovered


    recovered_result = _build_result_payload(
        case_dir=case_dir,
        mode_name=mode_name,
        model_name=model_name,
        row_id=row_id,
        max_attempts=max_attempts,
        final_status="recovered_retry_ready",
        attempts=[*attempts[:-1], recovered_record],
    )
    write_json(case_dir / "meta" / "agent_run_summary.json", recovered_result)
    return [*attempts[:-1], recovered_record], resume_attempt


def _recover_legacy_attempt_semantics(

    *,
    case_dir: Path,
    mode_name: str,
    model_name: str,
    row_id: int,
    max_attempts: int,

) -> tuple[list[dict[str, Any]], int] | None:

    summary = _load_existing_summary(case_dir)
    if not summary:
        return None


    attempts = summary.get("attempts", [])
    if not isinstance(attempts, list) or len(attempts) < 2:
        return None


    if summary.get("final_status") not in {"failed", "recovered_retry_ready"}:
        return None


    last_attempt = attempts[-1]
    last_attempt_index = int(last_attempt.get("attempt_index", 0))
    last_status = str(last_attempt.get("status") or "")
    if last_attempt_index != max_attempts:
        return None
    if last_status not in {"compile_failed_max_attempts", "compile_failed_after_final_repair"}:
        return None


    previous_attempt = attempts[-2]
    if str(previous_attempt.get("status") or "") != "repaired_for_next_attempt":
        return None


    compile_stdout_path = Path(str(last_attempt.get("compile_stdout_path") or ""))
    compile_stderr_path = Path(str(last_attempt.get("compile_stderr_path") or ""))
    summary_path = Path(str(last_attempt.get("compile_error_summary_path") or ""))
    if not compile_stdout_path.exists() or not compile_stderr_path.exists() or not summary_path.exists():
        return None


    source_attempt_dir = case_dir / "attempts" / f"attempt_{last_attempt_index}"
    if not (source_attempt_dir / "normalized" / "assembled_candidate.cpp").exists():
        return None
    if not (source_attempt_dir / "meta" / "classification.json").exists():
        return None
    if _has_feedback_outputs_for_attempt(case_dir, mode_name, last_attempt_index):
        return None


    trimmed_attempts = [attempt for attempt in attempts if int(attempt.get("attempt_index", 0)) < last_attempt_index]
    recovered_result = _build_result_payload(
        case_dir=case_dir,
        mode_name=mode_name,
        model_name=model_name,
        row_id=row_id,
        max_attempts=max_attempts,
        final_status="recovered_retry_ready",
        attempts=trimmed_attempts,
    )
    write_json(case_dir / "meta" / "agent_run_summary.json", recovered_result)
    return trimmed_attempts, last_attempt_index


def _classify_retryable_failed_row(

    mode_name: str,
    model_name: str,
    row_id: int,
    develop: bool,

) -> str | None:

    case_dir = build_case_dir(mode_name, model_name, row_id, develop=develop)
    summary, module_state = _refresh_summary_from_modules(
        case_dir=case_dir,
        mode_name=mode_name,
        model_name=model_name,
        row_id=row_id,
        max_attempts=2,
    )
    if module_state is not None and module_state.resume_attempt_index is not None:
        return "module_resume"
    module_resume_attempt = _detect_resume_attempt_from_modules(case_dir, max_attempts=2, resume_state=_load_resume_state(case_dir))
    if module_resume_attempt is not None:
        return "module_resume"


    legacy_recovered_attempt = _recover_legacy_attempt_semantics(
        case_dir=case_dir,
        mode_name=mode_name,
        model_name=model_name,
        row_id=row_id,
        max_attempts=2,
    )
    if legacy_recovered_attempt is not None:
        return "module_resume"


    if summary is None:
        return None


    if summary.get("final_status") == "recovered_retry_ready":
        attempts = summary.get("attempts", [])
        if not isinstance(attempts, list) or not attempts:
            return None
        recovery_resume_attempt = attempts[-1].get("recovery_resume_attempt")
        if isinstance(recovery_resume_attempt, int):
            attempt_cpp = case_dir / "attempts" / f"attempt_{recovery_resume_attempt}" / "normalized" / "assembled_candidate.cpp"
            return "llm_content_recovery" if attempt_cpp.exists() else None


        legacy_resume_attempt = len(attempts) + 1
        if legacy_resume_attempt <= 2 and not _has_feedback_outputs_for_attempt(case_dir, mode_name, legacy_resume_attempt):
            source_compile = case_dir / "compile" / f"attempt_{legacy_resume_attempt}" / "error_summary.json"
            source_attempt = case_dir / "attempts" / f"attempt_{legacy_resume_attempt}" / "meta" / "classification.json"
            if source_compile.exists() and source_attempt.exists():
                return "module_resume"
        return None


    attempts = summary.get("attempts", [])
    if not isinstance(attempts, list) or not attempts:
        return None


    if summary.get("final_status") not in {"failed_llm_content", "failed"}:
        return None


    last_attempt = attempts[-1]
    failed_stage = str(last_attempt.get("failed_stage") or "")
    llm_error = str(last_attempt.get("llm_error") or "")
    retryable_error_markers = (
        "payload missing required fields",
        "unable to locate JSON object",
        "repair payload did not contain usable generated code",
        "unable to classify candidate as class or function",
        "register_cases must be an array when provided",
    )
    if failed_stage not in {"compile_only_fix", "regeneration"} and not any(
        marker in llm_error for marker in retryable_error_markers
    ):
        return None


    resolved_stage = failed_stage if failed_stage in {"compile_only_fix", "regeneration"} else "regeneration"
    prefix, _ = _resolve_repair_stage_schema(case_dir, last_attempt, resolved_stage)
    raw_path = case_dir / "feedback" / f"attempt_{int(last_attempt.get('attempt_index', 1))}" / f"{prefix}_response_raw.txt"
    return "llm_content_recovery" if raw_path.exists() else None


def _fill_prompt(template_path: Path, replacements: dict[str, str]) -> str:

    prompt = _read_text(template_path)
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def _compress_prompt_replacements(

    template_path: Path,
    replacements: dict[str, str],
    *,
    stage: str,

) -> dict[str, str]:

    prompt = _fill_prompt(template_path, replacements)
    if _estimate_text_tokens(prompt) <= PROMPT_INPUT_TOKEN_BUDGET:
        return replacements


    adjusted = dict(replacements)
    print(
        f"[Prompt] Oversized prompt detected for {template_path.name} "

        f"(est_tokens={_estimate_text_tokens(prompt)} > input_budget={PROMPT_INPUT_TOKEN_BUDGET}), compressing context..."
    )
    ordered_keys = [
        "{{GL_SUBCLASS_JSON}}",
        "{{GL_BASE_CLASSES_JSON}}",
        "{{COMPILE_RAW_OUTPUT}}",
        "{{CHECKER_OUTPUT}}",
        "{{VERIFIER_OUTPUT}}",
        "{{EXPLAINER_OUTPUT}}",
        # {{COMPILE_ERRORS_JSON}} 不列入 —— 编译错误不参与压缩, 见 PROMPT_SLOT_BUDGETS 注释
        "{{CURRENT_STRUCTURE_JSON}}",
        "{{DIRECTOR_OUTPUT}}",
        "{{PREVIOUS_VK_CODE}}",
        "{{ASSEMBLED_CANDIDATE_CODE}}",
        "{{CURRENT_VK_CODE}}",
        "{{CLEANED_CANDIDATE_BODY}}",
    ]


    for key in ordered_keys:
        current = adjusted.get(key)
        if not current:
            continue
        budget = PROMPT_SLOT_BUDGETS.get(key)
        if budget is None or len(current) <= budget:
            continue
        if key in {"{{DIRECTOR_OUTPUT}}", "{{EXPLAINER_OUTPUT}}", "{{VERIFIER_OUTPUT}}", "{{CHECKER_OUTPUT}}"}:
            adjusted[key] = _trim_jsonish_text(current, budget, f"\n\n[{key.strip('{}')}_TRUNCATED]\n\n")
        else:
            adjusted[key] = _clip_text_middle(current, budget, f"\n\n[{key.strip('{}')}_TRUNCATED]\n\n")
        prompt = _fill_prompt(template_path, adjusted)
        if _estimate_text_tokens(prompt) <= PROMPT_INPUT_TOKEN_BUDGET:
            break


    if _estimate_text_tokens(prompt) > PROMPT_INPUT_TOKEN_BUDGET:
        for key in (
            "{{GL_SUBCLASS_JSON}}",
            "{{GL_BASE_CLASSES_JSON}}",
            "{{PREVIOUS_VK_CODE}}",
            "{{ASSEMBLED_CANDIDATE_CODE}}",
            "{{CURRENT_VK_CODE}}",
            "{{CLEANED_CANDIDATE_BODY}}",
            # {{COMPILE_ERRORS_JSON}} 不列入 —— 这是最后的兜底压缩, 依然不动编译错误
        ):
            current = adjusted.get(key)
            if not current:
                continue
            if key in {"{{GL_SUBCLASS_JSON}}", "{{GL_BASE_CLASSES_JSON}}"}:
                adjusted[key] = _trim_jsonish_text(current, 8000, f"\n\n[{key.strip('{}')}_TRUNCATED_MORE]\n\n")
            else:
                adjusted[key] = _clip_text_middle(current, 16000, f"\n\n[{key.strip('{}')}_TRUNCATED_MORE]\n\n")
            prompt = _fill_prompt(template_path, adjusted)
            if _estimate_text_tokens(prompt) <= PROMPT_INPUT_TOKEN_BUDGET:
                break


    print(
        f"[Prompt] Final prompt size for {template_path.name}: "

        f"{len(prompt)} chars, est_tokens={_estimate_text_tokens(prompt)}, input_budget={PROMPT_INPUT_TOKEN_BUDGET}"
    )
    return adjusted


def _stage_case_dirs(case_dir: Path, attempt_index: int) -> tuple[Path, Path]:

    feedback_attempt_dir = case_dir / "feedback" / f"attempt_{attempt_index}"
    compile_attempt_dir = case_dir / "compile" / f"attempt_{attempt_index}"
    feedback_attempt_dir.mkdir(parents=True, exist_ok=True)
    compile_attempt_dir.mkdir(parents=True, exist_ok=True)
    return feedback_attempt_dir, compile_attempt_dir


def _artifact_base_dir(case_dir: Path, attempt_index: int) -> Path:

    if attempt_index <= 1:
        return case_dir
    return case_dir / "attempts" / f"attempt_{attempt_index}"


def _load_current_artifacts(case_dir: Path, attempt_index: int) -> tuple[str, dict[str, Any], dict[str, Any]]:

    base_dir = _artifact_base_dir(case_dir, attempt_index)
    normalized_dir = base_dir / "normalized" if attempt_index > 1 else case_dir / "normalized"
    meta_dir = base_dir / "meta" if attempt_index > 1 else case_dir / "meta"
    return (
        _read_text(normalized_dir / "cleaned_candidate_body.cpp"),
        _load_json(meta_dir / "classification.json"),
        _load_json(meta_dir / "structure_validation.json"),
    )


def _load_assembled_cpp(case_dir: Path, attempt_index: int) -> str:

    base_dir = _artifact_base_dir(case_dir, attempt_index)
    normalized_dir = base_dir / "normalized" if attempt_index > 1 else case_dir / "normalized"
    return _read_text(normalized_dir / "assembled_candidate.cpp")


def _ensure_attempt_artifacts_present(

    case_dir: Path,
    *,
    mode_name: str,
    model_name: str,
    row_id: int,
    attempt_index: int,
    develop: bool,

) -> None:

    if attempt_index <= 1:
        normalized_dir = case_dir / "normalized"
        meta_dir = case_dir / "meta"
        required_paths = (
            normalized_dir / "cleaned_candidate_body.cpp",
            normalized_dir / "assembled_candidate.cpp",
            meta_dir / "classification.json",
            meta_dir / "structure_validation.json",
        )
        if all(path.exists() for path in required_paths):
            return
        print(f"[Recover] Initial artifacts missing, regenerating via prepare_case...")
        prepare_case(
            mode_name=mode_name,
            model_name=model_name,
            row_id=row_id,
            force=False,
            develop=develop,
        )
        return


    attempt_dir = case_dir / "attempts" / f"attempt_{attempt_index}"
    normalized_dir = attempt_dir / "normalized"
    meta_dir = attempt_dir / "meta"
    body_path = normalized_dir / "cleaned_candidate_body.cpp"
    assembled_path = normalized_dir / "assembled_candidate.cpp"
    classification_path = meta_dir / "classification.json"
    structure_validation_path = meta_dir / "structure_validation.json"


    if body_path.exists() and classification_path.exists():
        try:
            classification = ClassifiedCode(**_load_json(classification_path))
            if not assembled_path.exists():
                assembled_path.parent.mkdir(parents=True, exist_ok=True)
                assembled_path.write_text(assemble_cpp(classification), encoding="utf-8")
                print(f"[Recover] Rebuilt missing assembled_candidate.cpp for attempt {attempt_index}")
            if not structure_validation_path.exists():
                validate_case_artifact(case_dir, attempt_index)
                print(f"[Recover] Rebuilt missing structure_validation.json for attempt {attempt_index}")
            if (
                body_path.exists()
                and assembled_path.exists()
                and classification_path.exists()
                and structure_validation_path.exists()
            ):
                return
        except Exception as exc:
            print(f"[Recover] Lightweight artifact rebuild failed for attempt {attempt_index}: {exc}")


    cached_feedback = _load_cached_feedback_payload(case_dir, attempt_index - 1)
    if cached_feedback is None:
        return


    _, payload, _ = cached_feedback
    print(f"[Recover] Reapplying cached repair payload to rebuild attempt {attempt_index} artifacts...")
    apply_repair_payload(
        mode_name=mode_name,
        model_name=model_name,
        row_id=row_id,
        payload=payload,
        attempt_index=attempt_index,
        develop=develop,
    )


def _load_compile_raw_output(stdout_path: str, stderr_path: str) -> str:

    stdout_text = _read_text(Path(stdout_path)) if Path(stdout_path).exists() else ""
    stderr_text = _read_text(Path(stderr_path)) if Path(stderr_path).exists() else ""
    combined = []
    if stdout_text.strip():
        combined.append("[stdout]\n" + stdout_text.strip())
    if stderr_text.strip():
        combined.append("[stderr]\n" + stderr_text.strip())
    return "\n\n".join(combined)


def _create_client(model_name: str, mode_name: str, case_dir: Path,
                   network_mode: str | None = None, proxy_url: str | None = None):
    """Create the vendor-neutral client backed by the user's model runner."""
    return model_adapter.create_client(
        model_name,
        stats_path=case_dir / "meta" / "llm_usage.jsonl",
        retry=1,
    )


def _call_json_prompt(

    client,
    prompt: str,
    output_dir: Path,
    prefix: str,
    schema: dict[str, Any],
    meta: dict[str, Any],

) -> dict[str, Any]:

    prompt_path = output_dir / f"{prefix}_prompt.txt"
    raw_path = output_dir / f"{prefix}_response_raw.txt"
    extracted_path = output_dir / f"{prefix}_response_extracted.json"
    if extracted_path.exists():
        try:
            payload = _load_json(extracted_path)
            _validate_schema(payload, schema)
            _validate_repair_payload_usability(
                payload,
                stage=str(meta.get("stage", prefix)),
                mode_name=str(meta.get("mode_name", "")),
                model_name=str(meta.get("model_name", "")),
                row_id=int(meta.get("row_id", -1)),
            )
            return payload
        except ValueError:
            pass
    prompt_path.write_text(prompt, encoding="utf-8")
    stage = str(meta.get("stage", prefix))
    mode_name = str(meta.get("mode_name", ""))
    model_name = str(meta.get("model_name", ""))
    row_id = int(meta.get("row_id", -1))
    prompt_tokens_est = _estimate_text_tokens(prompt)
    print(f"  [PromptBudget] stage={stage} prompt_est={prompt_tokens_est}/{PROMPT_INPUT_TOKEN_BUDGET}")
    repair_retry_limit = 2 if stage in {"compile_only_fix", "regeneration"} else 1
    last_value_error: ValueError | None = None


    for generation_index in range(1, repair_retry_limit + 1):
        current_prompt = prompt
        if generation_index > 1:
            retry_reason = str(last_value_error or "the previous response could not be parsed/applied")
            current_prompt = (
                prompt
                + f"\n\nPrevious response was unusable because: {retry_reason}"
                + "\nTreat this as a repair hint and fix that exact issue in the next JSON payload."
                + "\nReturn one complete repair payload JSON object only."
                + "\nThe code field must contain the full compilable class/function body, not `...`, summaries, or partial snippets."
            )


        try:
            raw_response = client.generate(
                current_prompt,
                description=prefix if generation_index == 1 else f"{prefix}_retry_{generation_index}",
                meta={**meta, "generation_index": generation_index},
            )
        except Exception as exc:
            traceback_path = output_dir / f"{prefix}_client_exception.txt"
            traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
            error_text = str(exc)
            if _looks_like_connectivity_error(error_text):
                raise LLMConnectivityError(
                    stage=stage,
                    attempt_index=int(meta.get("attempt_index", -1)),
                    original_error=error_text,
                ) from exc
            if isinstance(exc, KeyError):
                raise LLMContentError(
                    stage=stage,
                    attempt_index=int(meta.get("attempt_index", -1)),
                    original_error=f"internal key error during {stage}: missing {exc}",
                ) from exc
            raise


        if generation_index == 1:
            raw_path.write_text(raw_response, encoding="utf-8")
        else:
            retry_raw_path = output_dir / f"{prefix}_response_raw_retry_{generation_index}.txt"
            retry_raw_path.write_text(raw_response, encoding="utf-8")
            raw_path.write_text(raw_response, encoding="utf-8")


        if _looks_like_service_error_response(raw_response):
            raise LLMConnectivityError(
                stage=stage,
                attempt_index=int(meta.get("attempt_index", -1)),
                original_error=f"Service error in raw response: {raw_response[:200]}",
            )


        payload: dict[str, Any] | None = None
        try:
            if stage not in {"compile_only_fix", "regeneration"}:
                payload = _extract_schema_matching_json_object(raw_response, schema, stage)
                _validate_schema(payload, schema)
                write_json(extracted_path, payload)
                return payload


            try:
                payload = _extract_schema_matching_json_object(raw_response, schema, stage)
            except ValueError:
                payload = _salvage_structured_payload(raw_response)
                if payload is None:
                    payload = _recover_code_payload_from_raw_cpp(
                        raw_text=raw_response,
                        stage=stage,
                        mode_name=mode_name,
                        model_name=model_name,
                        row_id=row_id,
                    )
                    if payload is None:
                        raise
            payload = _normalize_payload_before_schema_validation(payload, stage=stage, fallback_schema=schema)
            effective_schema = _select_schema_for_payload(payload, schema, stage)
            _validate_schema(payload, effective_schema)
            _validate_repair_payload_usability(
                payload,
                stage=stage,
                mode_name=mode_name,
                model_name=model_name,
                row_id=row_id,
            )
            _validate_repair_payload_applicability(
                payload,
                mode_name=mode_name,
                model_name=model_name,
                row_id=row_id,
            )
            write_json(extracted_path, payload)
            return payload
        except ValueError as exc:
            last_value_error = exc
            if generation_index >= repair_retry_limit and stage in {"compile_only_fix", "regeneration"}:
                coerced_payload = _coerce_repair_payload_for_compile(
                    payload,
                    stage=stage,
                    fallback_schema=schema,
                )
                write_json(extracted_path, coerced_payload)
                return coerced_payload
            if generation_index >= repair_retry_limit:
                break


    raise LLMContentError(
        stage=stage,
        attempt_index=int(meta.get("attempt_index", -1)),
        original_error=str(last_value_error or ValueError("unknown LLM content error")),
    )


def _common_prompt_replacements(

    gl_row: dict[str, Any],
    cleaned_body: str,
    assembled_cpp: str,
    current_structure: dict[str, Any],
    summary: dict[str, Any],

) -> dict[str, str]:

    prompt_current_structure = dict(current_structure)
    prompt_current_structure.pop("code", None)
    return {
        "{{GL_SUBCLASS_JSON}}": _json_dump(gl_row.get("gl_subclass")),
        "{{GL_BASE_CLASSES_JSON}}": _json_dump(gl_row.get("gl_base_classes")),
        "{{CLEANED_CANDIDATE_BODY}}": cleaned_body,
        "{{ASSEMBLED_CANDIDATE_CODE}}": assembled_cpp,
        "{{CURRENT_STRUCTURE_JSON}}": _json_dump(prompt_current_structure),
        "{{CURRENT_VK_CODE}}": assembled_cpp,
        "{{PREVIOUS_VK_CODE}}": assembled_cpp,
        "{{COMPILE_ERRORS_JSON}}": _json_dump(summary),
        "{{STRUCTURE_REJUDGE_JSON}}": _json_dump({}),
    }


def _augment_structure_with_hints(

    current_structure: dict[str, Any],
    structure_validation: dict[str, Any] | None,
    *,
    compile_summary: dict[str, Any],

) -> dict[str, Any]:

    augmented = dict(current_structure)
    hints: list[str] = []
    if isinstance(structure_validation, dict):
        for warning in structure_validation.get("warnings", []) or []:
            if isinstance(warning, str) and warning.strip():
                hints.append(warning.strip())
    # 只在确实拿到编译证据时才提示条数, 否则空 summary 会被说成「还有 0 个错误」
    if isinstance(compile_summary, dict) and (
        isinstance(compile_summary.get("errors"), list)
        or isinstance(compile_summary.get("error_count"), int)
    ):
        hints.append(f"current compile still has {summary_error_count(compile_summary)} errors")
    if hints:
        augmented["soft_hints"] = hints[:8]
    return augmented


def _build_runtime_case_manifest(mode_name: str, model_name: str, row_id: int) -> dict[str, Any]:

    root_group = resolve_runtime_root_group(mode_name, model_name)
    case_name = f"case_{row_id}"
    candidate_paths = [
        f"dEQP-VK.{root_group}.smoke.{case_name}",
        f"dEQP-VK.{root_group}.{case_name}",
    ]
    return {
        "package": "dEQP-VK",
        "root_group": root_group,
        "case_name": case_name,
        "case_filter": f"dEQP-VK.{root_group}*{row_id}*",
        "candidate_paths": candidate_paths,
    }


def _select_repair_contract(template_type: str, prefix: str) -> tuple[Path, dict[str, Any]]:

    if prefix == "repair":
        if template_type == "class":
            return FEEDBACK_PROMPT_DIR / "feedback_repair_class.txt", CLASS_REPAIR_SCHEMA
        return FEEDBACK_PROMPT_DIR / "feedback_repair_function.txt", FUNCTION_REPAIR_SCHEMA
    if template_type == "class":
        return PROMPT_ROOT_DIR / "generator_agent.txt", REGENERATION_CLASS_SCHEMA
    return PROMPT_ROOT_DIR / "generator_agent.txt", REGENERATION_FUNCTION_SCHEMA


def _run_only_fix_flow(

    client,
    feedback_dir: Path,
    gl_row: dict[str, Any],
    cleaned_body: str,
    assembled_cpp: str,
    current_structure: dict[str, Any],
    structure_validation: dict[str, Any],
    summary: dict[str, Any],
    mode_name: str,
    model_name: str,
    row_id: int,
    attempt_index: int,

) -> dict[str, Any]:

    prompt_structure = _augment_structure_with_hints(current_structure, structure_validation, compile_summary=summary)
    replacements = _common_prompt_replacements(gl_row, cleaned_body, assembled_cpp, prompt_structure, summary)


    print(f"  [LLM] Calling feedback structural re-evaluation agent...")
    rejudge_template = FEEDBACK_PROMPT_DIR / "feedback_rejudge_agent.txt"
    rejudge_prompt = _fill_prompt(
        rejudge_template,
        _compress_prompt_replacements(
            rejudge_template,
            replacements,
            stage="feedback_rejudge",
        ),
    )
    rejudge_payload = _call_json_prompt(
        client=client,
        prompt=rejudge_prompt,
        output_dir=feedback_dir,
        prefix="rejudge",
        schema=STRUCTURE_REJUDGE_SCHEMA,
        meta={
            "mode_name": mode_name,
            "model_name": model_name,
            "row_id": row_id,
            "attempt_index": attempt_index,
            "stage": "feedback_rejudge",
        },
    )
    print(f"  [Rejudge] {rejudge_payload.get('reason')} (template_type={rejudge_payload.get('template_type')})")


    replacements["{{STRUCTURE_REJUDGE_JSON}}"] = _json_dump(rejudge_payload)
    print(f"  [LLM] Calling compile_only_fix agent...")
    repair_template = FEEDBACK_PROMPT_DIR / "compile_only_fix.txt"
    repair_prompt = _fill_prompt(repair_template, _compress_prompt_replacements(repair_template, replacements, stage="compile_only_fix"))
    repair_schema = CLASS_REPAIR_SCHEMA if rejudge_payload["template_type"] == "class" else FUNCTION_REPAIR_SCHEMA
    repair_payload = _call_json_prompt(
        client=client,
        prompt=repair_prompt,
        output_dir=feedback_dir,
        prefix="repair",
        schema=repair_schema,
        meta={
            "mode_name": mode_name,
            "model_name": model_name,
            "row_id": row_id,
            "attempt_index": attempt_index,
            "stage": "compile_only_fix",
        },
    )
    print(f"  [LLM] OK Repair completed")


    write_json(feedback_dir / "structure_validation_snapshot.json", structure_validation)
    return {
        "flow": "singleAgent",
        "rejudge_payload": rejudge_payload,
        "repair_payload": repair_payload,
        "record": {
            "rejudge_path": str(feedback_dir / "rejudge_response_extracted.json"),
            "repair_path": str(feedback_dir / "repair_response_extracted.json"),
        },
    }


def _run_director_flow(

    client,
    feedback_dir: Path,
    gl_row: dict[str, Any],
    cleaned_body: str,
    assembled_cpp: str,
    current_structure: dict[str, Any],
    summary: dict[str, Any],
    compile_result,
    mode_name: str,
    model_name: str,
    row_id: int,
    attempt_index: int,
    max_attempts: int,

) -> dict[str, Any]:

    target_name = resolve_target_name(mode_name, model_name)
    prompt_structure = _augment_structure_with_hints(current_structure, None, compile_summary=summary)
    replacements = _common_prompt_replacements(gl_row, cleaned_body, assembled_cpp, prompt_structure, summary)
    replacements.update(
        {
            "{{MODE_NAME}}": mode_name,
            "{{MODEL_NAME}}": model_name,
            "{{COMPILE_TARGET}}": target_name,
            "{{ATTEMPT_INDEX}}": str(attempt_index),
            "{{MAX_REGENS}}": str(max_attempts - 1),
        }
    )


    print(f"  [LLM] Calling Director agent...")
    director_template = PROMPT_ROOT_DIR / "director_agent.txt"
    director_prompt = _fill_prompt(director_template, _compress_prompt_replacements(director_template, replacements, stage="director"))
    director_payload = _call_json_prompt(
        client=client,
        prompt=director_prompt,
        output_dir=feedback_dir,
        prefix="director",
        schema=DIRECTOR_DECISION_SCHEMA,
        meta={
            "mode_name": mode_name,
            "model_name": model_name,
            "row_id": row_id,
            "attempt_index": attempt_index,
            "stage": "director",
        },
    )
    use_agents = director_payload.get("use_agents", [])
    if isinstance(use_agents, str):
        use_agents = [use_agents]
    if isinstance(use_agents, list):
        legacy_agent_names = {
            "concept": "explainer",
            "semantic": "verifier",
            "code_explainer": "checker",
        }
        use_agents = [legacy_agent_names.get(name, name) for name in use_agents]
        director_payload["use_agents"] = use_agents
        write_json(feedback_dir / "director_response_extracted.json", director_payload)
    elif not isinstance(use_agents, list):
        use_agents = []
        director_payload["use_agents"] = use_agents
        write_json(feedback_dir / "director_response_extracted.json", director_payload)
    print(f"  [LLM] OK Director completed (use_agents={use_agents})")


    agent_outputs: dict[str, Any] = {}


    if "explainer" in use_agents:
        print(f"  [LLM] Calling Explainer agent...")
        explainer_template = PROMPT_ROOT_DIR / "explainer_agent.txt"
        explainer_prompt = _fill_prompt(explainer_template, _compress_prompt_replacements(explainer_template, replacements, stage="explainer"))
        agent_outputs["explainer"] = _call_json_prompt(
            client=client,
            prompt=explainer_prompt,
            output_dir=feedback_dir,
            prefix="explainer",
            schema=EXPLAINER_SCHEMA,
            meta={
                "mode_name": mode_name,
                "model_name": model_name,
                "row_id": row_id,
                "attempt_index": attempt_index,
                "stage": "explainer",
            },
        )
        print(f"  [LLM] OK Explainer completed")


    verifier_replacements = dict(replacements)
    verifier_replacements["{{EXPLAINER_BLUEPRINT}}"] = _json_dump(agent_outputs.get("explainer", {}))
    if "verifier" in use_agents:
        print(f"  [LLM] Calling Verifier agent...")
        verifier_template = PROMPT_ROOT_DIR / "verifier_agent.txt"
        verifier_prompt = _fill_prompt(
            verifier_template,
            _compress_prompt_replacements(verifier_template, verifier_replacements, stage="verifier"),
        )
        agent_outputs["verifier"] = _call_json_prompt(
            client=client,
            prompt=verifier_prompt,
            output_dir=feedback_dir,
            prefix="verifier",
            schema=VERIFIER_SCHEMA,
            meta={
                "mode_name": mode_name,
                "model_name": model_name,
                "row_id": row_id,
                "attempt_index": attempt_index,
                "stage": "verifier",
            },
        )
        print(f"  [LLM] OK Verifier completed")


    checker_replacements = dict(replacements)
    checker_replacements["{{COMPILE_RAW_OUTPUT}}"] = _load_compile_raw_output(
        compile_result.stdout_path,
        compile_result.stderr_path,
    )
    if "checker" in use_agents:
        print(f"  [LLM] Calling Checker agent...")
        checker_template = PROMPT_ROOT_DIR / "checker_agent.txt"
        checker_prompt = _fill_prompt(
            checker_template,
            _compress_prompt_replacements(checker_template, checker_replacements, stage="checker"),
        )
        agent_outputs["checker"] = _call_json_prompt(
            client=client,
            prompt=checker_prompt,
            output_dir=feedback_dir,
            prefix="checker",
            schema=CHECKER_SCHEMA,
            meta={
                "mode_name": mode_name,
                "model_name": model_name,
                "row_id": row_id,
                "attempt_index": attempt_index,
                "stage": "checker",
            },
        )
        print(f"  [LLM] OK Checker completed")


    print(f"  [LLM] Calling Generator agent...")
    regen_replacements = dict(replacements)
    regen_replacements["{{DIRECTOR_OUTPUT}}"] = _json_dump(director_payload)
    regen_replacements["{{EXPLAINER_OUTPUT}}"] = _json_dump(agent_outputs.get("explainer", {}))
    regen_replacements["{{VERIFIER_OUTPUT}}"] = _json_dump(agent_outputs.get("verifier", {}))
    regen_replacements["{{CHECKER_OUTPUT}}"] = _json_dump(agent_outputs.get("checker", {}))
    regen_prompt_path, regen_schema = _select_repair_contract(current_structure.get("template_type", "function"), "regenerate")
    generator_prompt = _fill_prompt(
        regen_prompt_path,
        _compress_prompt_replacements(regen_prompt_path, regen_replacements, stage="regeneration"),
    )
    regeneration_payload = _call_json_prompt(
        client=client,
        prompt=generator_prompt,
        output_dir=feedback_dir,
        prefix="regeneration",
        schema=regen_schema,
        meta={
            "mode_name": mode_name,
            "model_name": model_name,
            "row_id": row_id,
            "attempt_index": attempt_index,
            "stage": "regeneration",
        },
    )
    print(f"  [LLM] OK Generator completed")


    normalized_regeneration_payload = _normalize_payload_consistency(regeneration_payload)
    validate_payload(normalized_regeneration_payload)
    repaired_body = _build_repaired_body(normalized_regeneration_payload)
    _build_locked_classification(
        mode_name=mode_name,
        model_name=model_name,
        row_id=row_id,
        payload=normalized_regeneration_payload,
        repaired_body=repaired_body,
    )


    return {
        "flow": "multiAgent",
        "director_payload": director_payload,
        "repair_payload": normalized_regeneration_payload,
        "record": {
            "director_path": str(feedback_dir / "director_response_extracted.json"),
            "explainer_path": str(feedback_dir / "explainer_response_extracted.json") if "explainer" in agent_outputs else None,
            "verifier_path": str(feedback_dir / "verifier_response_extracted.json") if "verifier" in agent_outputs else None,
            "checker_path": (
                str(feedback_dir / "checker_response_extracted.json") if "checker" in agent_outputs else None
            ),
            "regeneration_path": str(feedback_dir / "regeneration_response_extracted.json"),
        },
    }


def _verify_baseline_compile(mode_name: str, model_name: str) -> bool:

    """

    Verify that the baseline target can compile successfully.

    This ensures the CTS module is in a clean state before processing cases.



    Returns:

        True if baseline compiles successfully, False otherwise

    """
    target_name = resolve_target_name(mode_name, model_name)
    cts_root = resolve_cts_root(mode_name, model_name)
    build_dir = cts_root / "build"
    lock_scope = build_lock_scope_for_cts_root(cts_root)
    module_name = resolve_package_name(mode_name, model_name)


    print(f"[Baseline Check] Verifying {target_name} can compile...")


    command = ["cmake", "--build", str(build_dir), "--target", target_name]


    max_polls = 5
    timeout_seconds = 300
    log_root = baseline_check_log_dir()
    log_root.mkdir(parents=True, exist_ok=True)
    failure_records: list[dict[str, Any]] = []
    auto_repair_attempted = False


    for poll_index in range(1, max_polls + 1):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        attempt_dir = log_root / f"{timestamp}_{mode_name}_{model_name}_poll_{poll_index}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = attempt_dir / "baseline_check_stdout.txt"
        stderr_path = attempt_dir / "baseline_check_stderr.txt"
        summary_path = attempt_dir / "baseline_check_summary.json"
        summary: dict[str, Any] = {
            "mode_name": mode_name,
            "model_name": model_name,
            "target_name": target_name,
            "command": command,
            "poll_index": poll_index,
            "max_polls": max_polls,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }


        try:
            started_at = time.time()
            result = None
            stdout_text = ""
            stderr_text = ""
            with acquire_build_lock(lock_scope, f"baseline:{target_name}"):
                print(f"[Baseline Check] Resetting module source state for {module_name} before baseline compile...")
                reset_module_generated_state(mode_name, model_name)
                for build_attempt in range(1, BUILD_CONFLICT_MAX_RETRIES + 1):
                    result = subprocess.run(
                        command,
                        cwd=str(cts_root),
                        capture_output=True,
                        timeout=timeout_seconds,
                    )
                    stdout_text = result.stdout.decode("utf-8", errors="replace")
                    stderr_text = result.stderr.decode("utf-8", errors="replace")
                    if result.returncode == 0 or not _is_build_conflict(stdout_text, stderr_text):
                        break
                    if build_attempt >= BUILD_CONFLICT_MAX_RETRIES:
                        break
                    delay_seconds = random.uniform(BUILD_CONFLICT_WAIT_MIN_SECONDS, BUILD_CONFLICT_WAIT_MAX_SECONDS)
                    print(
                        f"[Baseline Check] Build conflict detected during poll {poll_index}/{max_polls} "

                        f"(retry {build_attempt}/{BUILD_CONFLICT_MAX_RETRIES}), waiting {delay_seconds:.1f}s"
                    )
                    time.sleep(delay_seconds)
            assert result is not None
            duration = time.time() - started_at
            stdout_path.write_text(stdout_text, encoding="utf-8")
            stderr_path.write_text(stderr_text, encoding="utf-8")
            summary.update(
                {
                    "status": "passed" if result.returncode == 0 else "failed",
                    "returncode": result.returncode,
                    "duration_seconds": duration,
                }
            )
            write_json(summary_path, summary)


            if result.returncode == 0:
                print(f"[Baseline Check] PASS on poll {poll_index}/{max_polls} ({duration:.1f}s)")
                if poll_index > 1:
                    print("[Baseline Check] Earlier failure did not persist; treating it as transient.")
                print(f"[Baseline Check] Log: {summary_path}")
                return True


            if not auto_repair_attempted and _maybe_auto_repair_baseline_module_state(
                mode_name=mode_name,
                model_name=model_name,
                cts_root=cts_root,
                build_dir=build_dir,
                stdout_text=stdout_text,
                stderr_text=stderr_text,
            ):
                auto_repair_attempted = True
                print("[Baseline Check] Auto-repair completed; retrying baseline compile immediately.")
                continue


            failure_records.append(
                {
                    "poll_index": poll_index,
                    "summary_path": str(summary_path),
                    "returncode": result.returncode,
                }
            )
            print(f"[Baseline Check] FAIL on poll {poll_index}/{max_polls} (returncode={result.returncode})")
            print(f"[Baseline Check] Log: {summary_path}")
        except subprocess.TimeoutExpired as exc:
            stdout_text = (exc.stdout or b"").decode("utf-8", errors="replace")
            stderr_text = (exc.stderr or b"").decode("utf-8", errors="replace")
            stdout_path.write_text(stdout_text, encoding="utf-8")
            stderr_path.write_text(stderr_text, encoding="utf-8")
            summary.update(
                {
                    "status": "timeout",
                    "timeout_seconds": timeout_seconds,
                }
            )
            write_json(summary_path, summary)
            failure_records.append(
                {
                    "poll_index": poll_index,
                    "summary_path": str(summary_path),
                    "returncode": "timeout",
                }
            )
            print(f"[Baseline Check] TIMEOUT on poll {poll_index}/{max_polls} (>{timeout_seconds}s)")
            print(f"[Baseline Check] Log: {summary_path}")
        except Exception as exc:
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(str(exc), encoding="utf-8")
            summary.update(
                {
                    "status": "error",
                    "error": str(exc),
                }
            )
            write_json(summary_path, summary)
            failure_records.append(
                {
                    "poll_index": poll_index,
                    "summary_path": str(summary_path),
                    "returncode": "error",
                }
            )
            print(f"[Baseline Check] ERROR on poll {poll_index}/{max_polls}: {exc}")
            print(f"[Baseline Check] Log: {summary_path}")


        if poll_index < max_polls:
            delay_seconds = random.uniform(3.0, 7.0)
            print(f"[Baseline Check] Waiting {delay_seconds:.1f}s before next poll...")
            time.sleep(delay_seconds)


    print("[Baseline Check] Persistent failure across all polls.")
    for failure in failure_records:
        print(f"[Baseline Check] Poll {failure['poll_index']} log: {failure['summary_path']}")
    return False


def _extract_compiler_file_paths(stdout_text: str, stderr_text: str) -> list[str]:

    combined = f"{stdout_text}\n{stderr_text}"
    matches = re.findall(r"([A-Za-z]:\\[^\r\n\"]+\.(?:cpp|hpp))", combined)
    ordered: list[str] = []
    seen: set[str] = set()
    for match in matches:
        candidate = match.strip()
        marker = candidate.lower()
        if marker in seen:
            continue
        seen.add(marker)
        ordered.append(candidate)
    return ordered


def _extract_stale_generated_module_paths(

    mode_name: str,
    model_name: str,
    stdout_text: str,
    stderr_text: str,

) -> list[Path]:

    module_name = resolve_package_name(mode_name, model_name).lower()
    expected_module_fragment = f"\\modules\\vulkan\\{module_name}\\"
    stale_paths: list[Path] = []
    seen: set[str] = set()
    for raw_path in _extract_compiler_file_paths(stdout_text, stderr_text):
        normalized = raw_path.lower().replace("/", "\\")
        if expected_module_fragment not in normalized:
            continue
        if not Path(raw_path).name.lower().startswith("vktcustomcase_"):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        stale_paths.append(Path(raw_path))
    return stale_paths


def _clean_module_build_subtree(build_dir: Path, module_name: str) -> bool:

    module_build_dir = build_dir / "external" / "vulkancts" / "modules" / "vulkan" / module_name
    if not module_build_dir.exists():
        return False
    print(f"[Baseline Check] Cleaning build subtree: {module_build_dir}")
    shutil.rmtree(module_build_dir, ignore_errors=False)
    return True


def _maybe_auto_repair_baseline_module_state(

    *,
    mode_name: str,
    model_name: str,
    cts_root: Path,
    build_dir: Path,
    stdout_text: str,
    stderr_text: str,

) -> bool:

    stale_paths = _extract_stale_generated_module_paths(mode_name, model_name, stdout_text, stderr_text)
    if not stale_paths:
        return False


    module_name = resolve_package_name(mode_name, model_name).lower()
    print("[Baseline Check] Detected stale generated module state:")
    for stale_path in stale_paths:
        print(f"[Baseline Check]   - {stale_path}")
    print(f"[Baseline Check] Resetting module source state for {module_name}...")
    reset_module_generated_state(mode_name, model_name)
    if not _clean_module_build_subtree(build_dir, module_name):
        return False


    print("[Baseline Check] Re-running CMake configure after targeted cleanup...")
    configure_result = subprocess.run(
        ["cmake", "-S", str(cts_root), "-B", str(build_dir)],
        cwd=str(cts_root),
        capture_output=True,
        timeout=300,
    )
    if configure_result.returncode != 0:
        print("[Baseline Check] Auto-repair configure step failed.")
        return False
    print("[Baseline Check] Targeted cleanup + configure completed.")
    return True


def _compile_stage_paths(case_dir: Path, compile_index: int) -> tuple[Path, Path, Path]:

    compile_dir = case_dir / "compile" / f"attempt_{compile_index}"
    return (
        compile_dir / "build_stdout.txt",
        compile_dir / "build_stderr.txt",
        compile_dir / "error_summary.json",
    )


def _is_compile_stage_complete(case_dir: Path, compile_index: int) -> bool:

    stdout_path, stderr_path, summary_path = _compile_stage_paths(case_dir, compile_index)
    return stdout_path.exists() and stderr_path.exists() and summary_path.exists()


def _infer_compile_stage_returncode(case_dir: Path, compile_index: int) -> int | None:

    if not _is_compile_stage_complete(case_dir, compile_index):
        return None
    stdout_path, stderr_path, summary_path = _compile_stage_paths(case_dir, compile_index)
    return _infer_compile_result_from_artifacts(stdout_path, stderr_path, summary_path)


def _load_compile_summary(summary_path: Path) -> dict[str, Any] | None:

    try:
        return _load_json(summary_path)
    except Exception:
        return None


def _extract_recorded_compile_returncode(summary: dict[str, Any] | None) -> int | None:

    if not isinstance(summary, dict):
        return None
    for key in ("build_returncode", "returncode", "compile_returncode", "subprocess_returncode"):
        value = summary.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def _read_text_if_exists(path: Path) -> str:

    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _logs_indicate_compile_failure(stdout_text: str, stderr_text: str) -> bool:

    combined = f"{stdout_text}\n{stderr_text}"
    return any(pattern.search(combined) for pattern in _FATAL_BUILD_LOG_PATTERNS)


def _infer_compile_result_from_artifacts(stdout_path: Path, stderr_path: Path, summary_path: Path) -> int:

    summary = _load_compile_summary(summary_path) if summary_path.exists() else None
    recorded_returncode = _extract_recorded_compile_returncode(summary)
    if recorded_returncode is not None:
        return 0 if recorded_returncode == 0 else 1


    if isinstance(summary, dict):
        try:
            error_count = summary_error_count(summary)
            if error_count > 0:
                return 1
        except Exception:
            return 1


    stdout_text = _read_text_if_exists(stdout_path)
    stderr_text = _read_text_if_exists(stderr_path)
    if _logs_indicate_compile_failure(stdout_text, stderr_text):
        return 1


    return 0 if isinstance(summary, dict) else 1


def _detect_resume_attempt_from_modules(

    case_dir: Path,
    max_attempts: int,
    resume_state: dict[str, Any] | None,

) -> int | None:

    if resume_state and resume_state.get("error_type") == "llm_connectivity":
        return int(resume_state["attempt_index"])
    if not _is_compile_stage_complete(case_dir, 1):
        return None
    for attempt_index in range(1, max_attempts + 1):
        source_compile_returncode = _infer_compile_stage_returncode(case_dir, attempt_index)
        if source_compile_returncode == 0:
            return None
        if attempt_index >= max_attempts:
            return None
        if not _is_compile_stage_complete(case_dir, attempt_index + 1):
            return attempt_index
        next_compile_returncode = _infer_compile_stage_returncode(case_dir, attempt_index + 1)
        if next_compile_returncode == 0:
            return None
    return None


def _is_terminal_summary_without_resume(summary: dict[str, Any] | None, max_attempts: int) -> bool:

    if not isinstance(summary, dict):
        return False


    final_status = str(summary.get("final_status") or "")
    if final_status == "compiled":
        return True
    if final_status != "failed":
        return False


    attempts = summary.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return False
    last_attempt = attempts[-1]
    last_attempt_index = int(last_attempt.get("attempt_index", 0))
    last_status = str(last_attempt.get("status") or "")
    if last_attempt_index != max_attempts:
        return False
    if last_status not in {"compile_failed_max_attempts", "compile_failed_after_final_repair"}:
        return False
    case_dir_raw = summary.get("case_dir")
    mode_name = str(summary.get("mode_name") or "")
    if isinstance(case_dir_raw, str) and case_dir_raw.strip() and mode_name in {"only-fix", "director"}:
        case_dir = Path(case_dir_raw)
        if case_dir.exists() and not _has_feedback_outputs_for_attempt(case_dir, mode_name, last_attempt_index):
            return False


    compile_stdout_path = Path(str(last_attempt.get("compile_stdout_path") or ""))
    compile_stderr_path = Path(str(last_attempt.get("compile_stderr_path") or ""))
    compile_summary_path = Path(str(last_attempt.get("compile_error_summary_path") or ""))
    return compile_stdout_path.exists() and compile_stderr_path.exists() and compile_summary_path.exists()


def _load_cached_feedback_payload(case_dir: Path, attempt_index: int) -> tuple[str, dict[str, Any], dict[str, Any]] | None:

    feedback_dir = case_dir / "feedback" / f"attempt_{attempt_index}"
    candidates = (
        ("singleAgent", feedback_dir / "repair_response_extracted.json"),
        ("multiAgent", feedback_dir / "regeneration_response_extracted.json"),
    )
    for flow_name, path in candidates:
        if path.exists():
            try:
                payload = _load_json(path)
                stage = "compile_only_fix" if path.name == "repair_response_extracted.json" else "regeneration"
                _validate_repair_payload_usability(
                    payload,
                    stage=stage,
                    mode_name=str(case_dir.parent.parent.name),
                    model_name=str(case_dir.parent.name),
                    row_id=int(str(case_dir.name).replace("case_", "")),
                )
                _validate_repair_payload_applicability(
                    payload,
                    mode_name=str(case_dir.parent.parent.name),
                    model_name=str(case_dir.parent.name),
                    row_id=int(str(case_dir.name).replace("case_", "")),
                )
            except Exception:
                print(f"[Cache] Ignoring unusable cached feedback payload: {path}")
                continue
            return flow_name, _normalize_payload_consistency(payload), {"cached_feedback_path": str(path)}
    return None


def _is_applied_attempt_complete(case_dir: Path, attempt_index: int) -> bool:

    attempt_dir = case_dir / "attempts" / f"attempt_{attempt_index}"
    return (
        (attempt_dir / "normalized" / "cleaned_candidate_body.cpp").exists()
        and (attempt_dir / "normalized" / "assembled_candidate.cpp").exists()
        and (attempt_dir / "meta" / "classification.json").exists()
        and (attempt_dir / "meta" / "structure_validation.json").exists()
    )


def _run_agent_case_v2(

    mode_name: str,
    model_name: str,
    row_id: int,
    max_attempts: int = 2,
    force_prepare: bool = False,
    develop: bool = False,
    skip_baseline_check: bool = False,

) -> dict[str, Any]:

    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")


    print(f"\n{'='*80}")
    print(f"Starting agent case: {mode_name}/{model_name}/row_id={row_id}")
    print(f"{'='*80}")


    if not skip_baseline_check:
        if not _verify_baseline_compile(mode_name, model_name):
            error_msg = (
                f"Baseline compile check repeatedly failed for {mode_name}/{model_name}. "

                f"Please inspect logs under {baseline_check_log_dir()} and run clean_all_test_cases.py if needed."
            )
            print(f"\n{'='*80}")
            print(f"[CRITICAL ERROR] {error_msg}")
            print(f"{'='*80}\n")
            raise RuntimeError(error_msg)


    case_dir = build_case_dir(mode_name, model_name, row_id, develop=develop)
    if not case_dir.exists() or force_prepare:
        print(f"[Prepare] Preparing case directory...")
        try:
            prepare_case(mode_name=mode_name, model_name=model_name, row_id=row_id, force=force_prepare, develop=develop)
        except Exception as exc:
            return _build_local_failure_result(
                case_dir=case_dir,
                mode_name=mode_name,
                model_name=model_name,
                row_id=row_id,
                max_attempts=max_attempts,
                attempts=[],
                stage="prepare_case",
                error=str(exc),
            )
        print(f"[Prepare] Case prepared at: {case_dir}")
    else:
        print(f"[Prepare] Using existing case directory: {case_dir}")


    print(f"[Load] Loading GL metadata for row_id={row_id}...")
    try:
        gl_row = load_gl_sim_vk_row(row_id)
    except Exception as exc:
        return _build_local_failure_result(
            case_dir=case_dir,
            mode_name=mode_name,
            model_name=model_name,
            row_id=row_id,
            max_attempts=max_attempts,
            attempts=[],
            stage="load_gl_metadata",
            error=str(exc),
        )
    print(f"[Load] Loaded GL subclass: {gl_row.get('gl_subclass', {}).get('name', 'unknown')}")


    if resolve_defer_compile() and not _is_compile_stage_complete(case_dir, 1):
        return _build_local_failure_result(
            case_dir=case_dir,
            mode_name=mode_name,
            model_name=model_name,
            row_id=row_id,
            max_attempts=max_attempts,
            attempts=[],
            stage="external_compile_attempt_1_required",
            error=(
                "--defer-compile requires durable attempt_1 compile evidence; "

                "the Stage 2 compile worker must run before the LLM worker"
            ),
        )


    probe = {
        "success": None,
        "skipped": True,
        "reason": "probe disabled; create client directly",
        "network_mode": None,
        "proxy_url": None,
    }
    write_json(case_dir / "meta" / "llm_probe.json", probe)
    print(f"[Probe] Skipped LLM connectivity probe; creating client directly")


    print(f"[Client] Creating LLM client...")
    try:
        client = _create_client(
            model_name=model_name,
            mode_name=mode_name,
            case_dir=case_dir,
            network_mode=None,
            proxy_url=None,
        )
    except Exception as exc:
        stage = "create_client"
        error_text = str(exc)
        if _looks_like_connectivity_error(error_text):
            result = _build_result_payload(
                case_dir=case_dir,
                mode_name=mode_name,
                model_name=model_name,
                row_id=row_id,
                max_attempts=max_attempts,
                final_status="interrupted_llm_connectivity",
                attempts=[
                    {
                        "attempt_index": 0,
                        "mode_name": mode_name,
                        "model_name": model_name,
                        "status": "llm_connectivity_interrupted",
                        "failed_stage": stage,
                        "llm_error": error_text,
                        "repaired": False,
                    }
                ],
            )
            write_json(case_dir / "meta" / "agent_run_summary.json", result)
            return result
        return _build_local_failure_result(
            case_dir=case_dir,
            mode_name=mode_name,
            model_name=model_name,
            row_id=row_id,
            max_attempts=max_attempts,
            attempts=[],
            stage=stage,
            error=error_text,
        )
    print(f"[Client] LLM client created")


    existing_summary, module_state = _refresh_summary_from_modules(
        case_dir=case_dir,
        mode_name=mode_name,
        model_name=model_name,
        row_id=row_id,
        max_attempts=max_attempts,
    )
    resume_state = _load_resume_state(case_dir)
    resume_attempt_index = _detect_resume_attempt_from_modules(case_dir, max_attempts, resume_state)
    if module_state is not None and module_state.resume_attempt_index is not None:
        resume_attempt_index = module_state.resume_attempt_index
    attempts: list[dict[str, Any]] = _load_previous_attempts(case_dir, resume_attempt_index) if resume_attempt_index else []
    if not resume_attempt_index:
        legacy_recovered_attempt = _recover_legacy_attempt_semantics(
            case_dir=case_dir,
            mode_name=mode_name,
            model_name=model_name,
            row_id=row_id,
            max_attempts=max_attempts,
        )
        if legacy_recovered_attempt is not None:
            attempts, resume_attempt_index = legacy_recovered_attempt
            existing_summary = _load_existing_summary(case_dir)
            print(f"[Resume] Recovered legacy three-step case, resuming at attempt {resume_attempt_index}")
    if (
        not force_prepare
        and resume_attempt_index is None
        and (resume_state is None or resume_state.get("error_type") != "llm_connectivity")
        and _is_terminal_summary_without_resume(existing_summary, max_attempts)
    ):
        print(f"[Resume] Existing terminal summary detected; skipping rerun")
        return existing_summary
    if resume_attempt_index and not attempts:
        attempts = _load_previous_attempts(case_dir, resume_attempt_index)


    final_status = "failed"
    target_name = resolve_target_name(mode_name, model_name)
    module_name = resolve_package_name(mode_name, model_name)
    runtime_case_manifest = _build_runtime_case_manifest(mode_name, model_name, row_id)


    if not _is_compile_stage_complete(case_dir, 1):
        try:
            _ensure_attempt_artifacts_present(
                case_dir,
                mode_name=mode_name,
                model_name=model_name,
                row_id=row_id,
                attempt_index=1,
                develop=develop,
            )
        except Exception as exc:
            return _build_local_failure_result(
                case_dir=case_dir,
                mode_name=mode_name,
                model_name=model_name,
                row_id=row_id,
                max_attempts=max_attempts,
                attempts=attempts,
                stage="recover_initial_artifacts",
                error=str(exc),
            )
        print(f"\n[Initial] Validating case structure...")
        try:
            initial_structure_result = validate_case_artifact(case_dir, None)
        except Exception as exc:
            return _build_local_failure_result(
                case_dir=case_dir,
                mode_name=mode_name,
                model_name=model_name,
                row_id=row_id,
                max_attempts=max_attempts,
                attempts=attempts,
                stage="validate_initial_artifacts",
                error=str(exc),
            )
        if initial_structure_result["valid"]:
            print(f"[Initial] Structure validation passed")
        else:
            print(f"[Initial] Structure validation warnings: {initial_structure_result.get('warnings', [])}")


        print(f"[Initial] Compiling target: {target_name}...")
        try:
            initial_compile_result = run_case_compile(
                mode_name=mode_name,
                model_name=model_name,
                row_id=row_id,
                attempt_index=1,
                develop=develop,
            )
        except Exception as exc:
            print(f"[Initial] Compile raised exception: {type(exc).__name__}: {exc}")
            return _build_local_failure_result(
                case_dir=case_dir,
                mode_name=mode_name,
                model_name=model_name,
                row_id=row_id,
                max_attempts=max_attempts,
                attempts=attempts,
                stage="initial_compile",
                error=f"{type(exc).__name__}: {exc}",
            )
        print(f"[Initial] Compile finished (returncode={initial_compile_result.returncode})")
        initial_summary_path = Path(initial_compile_result.stdout_path).parent / "error_summary.json"
        initial_summary = summarize_compile_logs(initial_compile_result.stdout_path, initial_compile_result.stderr_path)
        initial_summary["build_returncode"] = int(initial_compile_result.returncode)
        write_summary(initial_summary, initial_summary_path)
        if initial_compile_result.returncode == 0:
            final_status = "compiled"
            write_json(case_dir / "meta" / "runtime_case.json", runtime_case_manifest)
            result = _build_result_payload(
                case_dir=case_dir,
                mode_name=mode_name,
                model_name=model_name,
                row_id=row_id,
                max_attempts=max_attempts,
                final_status=final_status,
                attempts=[],
            )
            write_json(case_dir / "meta" / "agent_run_summary.json", result)
            return result


    start_attempt = resume_attempt_index or 1
    print(f"\n[Loop] Starting module loop (max_attempts={max_attempts})")
    for attempt_index in range(start_attempt, max_attempts + 1):
        print(f"\n{'-'*80}")
        print(f"[Attempt {attempt_index}/{max_attempts}] Starting...")
        print(f"{'-'*80}")


        try:
            _ensure_attempt_artifacts_present(
                case_dir,
                mode_name=mode_name,
                model_name=model_name,
                row_id=row_id,
                attempt_index=attempt_index,
                develop=develop,
            )
        except Exception as exc:
            return _build_local_failure_result(
                case_dir=case_dir,
                mode_name=mode_name,
                model_name=model_name,
                row_id=row_id,
                max_attempts=max_attempts,
                attempts=attempts,
                stage=f"recover_attempt_{attempt_index}_artifacts",
                error=str(exc),
            )


        compile_result = _load_cached_compile_result(case_dir, attempt_index)
        if compile_result is None:
            if resolve_defer_compile():
                return _build_local_failure_result(
                    case_dir=case_dir,
                    mode_name=mode_name,
                    model_name=model_name,
                    row_id=row_id,
                    max_attempts=max_attempts,
                    attempts=attempts,
                    stage=f"external_compile_attempt_{attempt_index}_invalid",
                    error=(
                        "--defer-compile forbids an in-run compile fallback; "

                        "Stage 2 compile evidence is missing or inconsistent"
                    ),
                )
            try:
                compile_result = run_case_compile(
                    mode_name=mode_name,
                    model_name=model_name,
                    row_id=row_id,
                    attempt_index=attempt_index,
                    develop=develop,
                )
            except Exception as exc:
                print(f"[Attempt {attempt_index}] Compile raised exception: {type(exc).__name__}: {exc}")
                return _build_local_failure_result(
                    case_dir=case_dir,
                    mode_name=mode_name,
                    model_name=model_name,
                    row_id=row_id,
                    max_attempts=max_attempts,
                    attempts=attempts,
                    stage=f"attempt_{attempt_index}_compile",
                    error=f"{type(exc).__name__}: {exc}",
                )
        summary_path = Path(compile_result.stdout_path).parent / "error_summary.json"
        summary = summarize_compile_logs(compile_result.stdout_path, compile_result.stderr_path)
        summary["build_returncode"] = int(compile_result.returncode)
        write_summary(summary, summary_path)


        source_artifact_attempt = None if attempt_index == 1 else attempt_index
        print(f"[Attempt {attempt_index}] Validating case structure...")
        try:
            structure_result = validate_case_artifact(case_dir, source_artifact_attempt)
        except Exception as exc:
            return _build_local_failure_result(
                case_dir=case_dir,
                mode_name=mode_name,
                model_name=model_name,
                row_id=row_id,
                max_attempts=max_attempts,
                attempts=attempts,
                stage=f"validate_attempt_{attempt_index}_artifacts",
                error=str(exc),
            )
        if structure_result["valid"]:
            print(f"[Attempt {attempt_index}] Structure validation passed")
        else:
            print(f"[Attempt {attempt_index}] Structure validation warnings: {structure_result.get('warnings', [])}")


        print(f"[Attempt {attempt_index}] Found {summary_error_count(summary)} errors")
        try:
            cleaned_body, current_structure, structure_validation = _load_current_artifacts(
                case_dir,
                1 if attempt_index == 1 else attempt_index,
            )
            assembled_cpp = _load_assembled_cpp(case_dir, 1 if attempt_index == 1 else attempt_index)
        except Exception as exc:
            return _build_local_failure_result(
                case_dir=case_dir,
                mode_name=mode_name,
                model_name=model_name,
                row_id=row_id,
                max_attempts=max_attempts,
                attempts=attempts,
                stage=f"load_attempt_{attempt_index}_artifacts",
                error=str(exc),
            )
        feedback_dir, _ = _stage_case_dirs(case_dir, attempt_index)


        attempt_record: dict[str, Any] = {
            "attempt_index": attempt_index,
            "mode_name": mode_name,
            "model_name": model_name,
            "module_name": module_name,
            "target_name": target_name,
            "structure_valid": structure_result["valid"],
            "input_compile_returncode": compile_result.returncode,
            "input_compile_stdout_path": compile_result.stdout_path,
            "input_compile_stderr_path": compile_result.stderr_path,
            "input_compile_error_summary_path": str(summary_path),
            "repaired": False,
        }


        if compile_result.returncode == 0:
            print(f"[Attempt {attempt_index}] Compile already succeeded, skipping feedback")
            attempt_record["status"] = "compiled"
            attempt_record["compile_returncode"] = compile_result.returncode
            attempt_record["compile_stdout_path"] = compile_result.stdout_path
            attempt_record["compile_stderr_path"] = compile_result.stderr_path
            attempt_record["compile_error_summary_path"] = str(summary_path)
            attempts.append(attempt_record)
            final_status = "compiled"
            write_json(case_dir / "meta" / "runtime_case.json", runtime_case_manifest)
            break


        cached_feedback = _load_cached_feedback_payload(case_dir, attempt_index)
        if cached_feedback is not None:
            cached_flow_name, cached_payload, cached_record = cached_feedback
            flow_result = {
                "flow": cached_flow_name,
                "record": cached_record,
                "repair_payload": cached_payload,
            }
            print(f"[Attempt {attempt_index}] Reusing cached feedback payload")
        else:
            try:
                if resolve_repair_strategy(mode_name) == "feedback":
                    flow_result = _run_only_fix_flow(
                        client=client,
                        feedback_dir=feedback_dir,
                        gl_row=gl_row,
                        cleaned_body=cleaned_body,
                        assembled_cpp=assembled_cpp,
                        current_structure=current_structure,
                        structure_validation=structure_validation,
                        summary=summary,
                        mode_name=mode_name,
                        model_name=model_name,
                        row_id=row_id,
                        attempt_index=attempt_index,
                    )
                else:
                    flow_result = _run_director_flow(
                        client=client,
                        feedback_dir=feedback_dir,
                        gl_row=gl_row,
                        cleaned_body=cleaned_body,
                        assembled_cpp=assembled_cpp,
                        current_structure=current_structure,
                        summary=summary,
                        compile_result=compile_result,
                        mode_name=mode_name,
                        model_name=model_name,
                        row_id=row_id,
                        attempt_index=attempt_index,
                        max_attempts=max_attempts,
                    )
                print(f"[Attempt {attempt_index}] Feedback flow completed")
            except LLMConnectivityError as exc:
                attempt_record["status"] = "llm_connectivity_interrupted"
                attempt_record["flow"] = mode_name
                attempt_record["resume_stage"] = exc.stage
                attempt_record["llm_error"] = exc.original_error
                attempts.append(attempt_record)
                final_status = "interrupted_llm_connectivity"
                _write_resume_state(
                    case_dir,
                    {
                        "error_type": "llm_connectivity",
                        "mode_name": mode_name,
                        "model_name": model_name,
                        "row_id": row_id,
                        "attempt_index": attempt_index,
                        "stage": exc.stage,
                    },
                )
                result = _build_result_payload(
                    case_dir=case_dir,
                    mode_name=mode_name,
                    model_name=model_name,
                    row_id=row_id,
                    max_attempts=max_attempts,
                    final_status=final_status,
                    attempts=attempts,
                )
                write_json(case_dir / "meta" / "agent_run_summary.json", result)
                return result
            except LLMContentError as exc:
                rejudge_path = feedback_dir / "rejudge_response_extracted.json"
                if rejudge_path.exists():
                    attempt_record["rejudge_path"] = str(rejudge_path)


                recovered = None
                if exc.stage in {"compile_only_fix", "regeneration"}:
                    try:
                        recovered = _try_recover_failed_stage_payload(
                            case_dir=case_dir,
                            mode_name=mode_name,
                            model_name=model_name,
                            row_id=row_id,
                            attempt_record={**attempt_record, "attempt_index": attempt_index},
                            failed_stage=exc.stage,
                            max_attempts=max_attempts,
                            develop=develop,
                        )
                    except Exception as recovery_exc:
                        print(f"[Attempt {attempt_index}] Recovery from raw response failed: {recovery_exc}")


                if recovered is not None:
                    recovered_record, resume_attempt = recovered
                    attempts.append(recovered_record)
                    if resume_attempt is not None:
                        print(f"[Attempt {attempt_index}] Recovered payload from raw response, continuing at attempt {resume_attempt}")
                        continue


                attempt_record["status"] = "llm_content_generation_failed"
                attempt_record["flow"] = mode_name
                attempt_record["failed_stage"] = exc.stage
                attempt_record["llm_error"] = exc.original_error
                attempts.append(attempt_record)
                final_status = "failed_llm_content"
                _clear_resume_state(case_dir)
                result = _build_result_payload(
                    case_dir=case_dir,
                    mode_name=mode_name,
                    model_name=model_name,
                    row_id=row_id,
                    max_attempts=max_attempts,
                    final_status=final_status,
                    attempts=attempts,
                )
                write_json(case_dir / "meta" / "agent_run_summary.json", result)
                return result


        write_json(feedback_dir / "structure_validation_snapshot.json", structure_validation)
        attempt_record["flow"] = flow_result["flow"]
        attempt_record.update({k: v for k, v in flow_result["record"].items() if v is not None})


        is_final_attempt = attempt_index >= max_attempts
        next_compile_label = (
            f"final repaired compile for attempt {attempt_index}"
            if is_final_attempt
            else f"attempt {attempt_index + 1}"
        )


        if _is_applied_attempt_complete(case_dir, attempt_index + 1):
            print(f"[Attempt {attempt_index}] Reusing applied repair artifacts for {next_compile_label}")
        else:
            try:
                apply_repair_payload(
                    mode_name=mode_name,
                    model_name=model_name,
                    row_id=row_id,
                    payload=flow_result["repair_payload"],
                    attempt_index=attempt_index + 1,
                    develop=develop,
                )
            except Exception as exc:
                return _build_local_failure_result(
                    case_dir=case_dir,
                    mode_name=mode_name,
                    model_name=model_name,
                    row_id=row_id,
                    max_attempts=max_attempts,
                    attempts=attempts,
                    stage=f"apply_repair_attempt_{attempt_index + 1}",
                    error=str(exc),
                )


        if (
            resolve_defer_compile()
            and not _is_compile_stage_complete(case_dir, attempt_index + 1)
        ):
            next_compile_attempt = attempt_index + 1
            print(
                f"[Attempt {attempt_index}] Repair artifacts are ready; "

                f"deferring compile attempt {next_compile_attempt} to Stage 2"
            )
            attempt_record["status"] = "deferred_compile_ready"
            attempt_record["repaired"] = True
            attempt_record["next_compile_attempt"] = next_compile_attempt
            attempt_record["next_attempt_artifact_dir"] = str(
                case_dir / "attempts" / f"attempt_{next_compile_attempt}"
            )
            attempts.append(attempt_record)
            _write_resume_state(
                case_dir,
                {
                    "state": "deferred_compile",
                    "mode_name": mode_name,
                    "model_name": model_name,
                    "row_id": row_id,
                    "attempt_index": attempt_index,
                    "next_compile_attempt": next_compile_attempt,
                },
            )
            result = _build_result_payload(
                case_dir=case_dir,
                mode_name=mode_name,
                model_name=model_name,
                row_id=row_id,
                max_attempts=max_attempts,
                final_status="deferred_compile_ready",
                attempts=attempts,
            )
            result["deferred_compile"] = {
                "source_attempt": attempt_index,
                "compile_attempt": next_compile_attempt,
                "artifact_dir": attempt_record["next_attempt_artifact_dir"],
            }
            write_json(case_dir / "meta" / "agent_run_summary.json", result)
            return result


        if _is_compile_stage_complete(case_dir, attempt_index + 1):
            print(f"[Attempt {attempt_index}] Reusing cached compile result for {next_compile_label}")
            next_compile_result = _load_cached_compile_result(case_dir, attempt_index + 1)
            if next_compile_result is None:
                if resolve_defer_compile():
                    return _build_local_failure_result(
                        case_dir=case_dir,
                        mode_name=mode_name,
                        model_name=model_name,
                        row_id=row_id,
                        max_attempts=max_attempts,
                        attempts=attempts,
                        stage=f"external_compile_attempt_{attempt_index + 1}_invalid",
                        error=(
                            "--defer-compile forbids an in-run compile fallback; "

                            "Stage 2 compile evidence is missing or inconsistent"
                        ),
                    )
                print(f"[Attempt {attempt_index}] Cached compile result missing, recompiling {next_compile_label}")
                try:
                    next_compile_result = run_case_compile(
                        mode_name=mode_name,
                        model_name=model_name,
                        row_id=row_id,
                        attempt_index=attempt_index + 1,
                        develop=develop,
                    )
                except Exception as exc:
                    print(
                        f"[Attempt {attempt_index}] {next_compile_label} raised exception: "

                        f"{type(exc).__name__}: {exc}"
                    )
                    return _build_local_failure_result(
                        case_dir=case_dir,
                        mode_name=mode_name,
                        model_name=model_name,
                        row_id=row_id,
                        max_attempts=max_attempts,
                        attempts=attempts,
                        stage=f"attempt_{attempt_index + 1}_compile",
                        error=f"{type(exc).__name__}: {exc}",
                    )
        else:
            if resolve_defer_compile():
                return _build_local_failure_result(
                    case_dir=case_dir,
                    mode_name=mode_name,
                    model_name=model_name,
                    row_id=row_id,
                    max_attempts=max_attempts,
                    attempts=attempts,
                    stage=f"external_compile_attempt_{attempt_index + 1}_required",
                    error=(
                        "--defer-compile requires Stage 2 to compile the "

                        "prepared repair before the runner resumes"
                    ),
                )
            print(f"[Attempt {attempt_index}] Running {next_compile_label}...")
            try:
                next_compile_result = run_case_compile(
                    mode_name=mode_name,
                    model_name=model_name,
                    row_id=row_id,
                    attempt_index=attempt_index + 1,
                    develop=develop,
                )
            except Exception as exc:
                print(
                    f"[Attempt {attempt_index}] {next_compile_label} raised exception: "

                    f"{type(exc).__name__}: {exc}"
                )
                return _build_local_failure_result(
                    case_dir=case_dir,
                    mode_name=mode_name,
                    model_name=model_name,
                    row_id=row_id,
                    max_attempts=max_attempts,
                    attempts=attempts,
                    stage=f"attempt_{attempt_index + 1}_compile",
                    error=f"{type(exc).__name__}: {exc}",
                )
        next_summary_path = Path(next_compile_result.stdout_path).parent / "error_summary.json"
        next_summary = summarize_compile_logs(next_compile_result.stdout_path, next_compile_result.stderr_path)
        next_summary["build_returncode"] = int(next_compile_result.returncode)
        write_summary(next_summary, next_summary_path)


        attempt_record["compile_returncode"] = next_compile_result.returncode
        attempt_record["compile_stdout_path"] = next_compile_result.stdout_path
        attempt_record["compile_stderr_path"] = next_compile_result.stderr_path
        attempt_record["compile_error_summary_path"] = str(next_summary_path)
        attempt_record["repaired"] = True


        if next_compile_result.returncode == 0:
            attempt_record["status"] = "compiled"
            attempts.append(attempt_record)
            final_status = "compiled"
            write_json(case_dir / "meta" / "runtime_case.json", runtime_case_manifest)
            break


        if attempt_index >= max_attempts:
            attempt_record["status"] = "compile_failed_max_attempts"
            attempts.append(attempt_record)
            final_status = "failed"
            break


        attempt_record["status"] = "repaired_for_next_attempt"
        attempts.append(attempt_record)


    _clear_resume_state(case_dir)
    result = _build_result_payload(
        case_dir=case_dir,
        mode_name=mode_name,
        model_name=model_name,
        row_id=row_id,
        max_attempts=max_attempts,
        final_status=final_status,
        attempts=attempts,
    )
    write_json(case_dir / "meta" / "agent_run_summary.json", result)
    return result


def run_agent_case(

    mode_name: str,
    model_name: str,
    row_id: int,
    max_attempts: int = 2,
    force_prepare: bool = False,
    develop: bool = False,
    skip_baseline_check: bool = False,

) -> dict[str, Any]:

    return _run_agent_case_v2(
        mode_name=mode_name,
        model_name=model_name,
        row_id=row_id,
        max_attempts=max_attempts,
        force_prepare=force_prepare,
        develop=develop,
        skip_baseline_check=skip_baseline_check,
    )


def _get_batch_progress_path(mode_name: str, model_name: str, develop: bool) -> Path:

    """Get the path to the batch progress tracking file"""
    output_root_override = resolve_output_root_override()
    if output_root_override is not None:
        progress_dir = Path(output_root_override)
        progress_dir.mkdir(parents=True, exist_ok=True)
        return progress_dir / "batch_progress.json"
    if develop:
        base_dir = DEFAULT_RUNS_DIR
    else:
        base_dir = DEFAULT_RUNS_DIR
    progress_dir = base_dir / mode_name / model_name
    progress_dir.mkdir(parents=True, exist_ok=True)
    return progress_dir / "batch_progress.json"


def _load_batch_progress(mode_name: str, model_name: str, develop: bool) -> dict[str, Any]:

    """Load batch processing progress"""
    progress_path = _get_batch_progress_path(mode_name, model_name, develop)
    if not progress_path.exists():
        return {"completed": [], "failed": [], "interrupted": [], "last_row_id": None}
    return _load_json(progress_path)


def _save_batch_progress(mode_name: str, model_name: str, develop: bool, progress: dict[str, Any]) -> None:

    """Save batch processing progress"""
    progress_path = _get_batch_progress_path(mode_name, model_name, develop)
    write_json(progress_path, progress)


def _get_run_record_path() -> Path | None:

    explicit = resolve_run_record_path_override()
    if explicit is not None:
        return explicit
    output_root_override = resolve_output_root_override()
    if output_root_override is None:
        return None
    run_record_path = Path(output_root_override) / "run_record.json"
    if not run_record_path.exists():
        return None
    return run_record_path


def _update_run_record(**updates: Any) -> None:

    run_record_path = _get_run_record_path()
    if run_record_path is None:
        return
    try:
        payload = _load_json(run_record_path)
    except Exception:
        payload = {}
    payload.update(_json_safe(updates))
    run_record_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(run_record_path, payload)


def _sync_batch_progress_with_case_summaries(

    mode_name: str,
    model_name: str,
    develop: bool,
    progress: dict[str, Any],

) -> dict[str, Any]:

    completed = {int(row_id) for row_id in progress.get("completed", [])}
    failed = {int(row_id) for row_id in progress.get("failed", [])}
    interrupted = {int(row_id) for row_id in progress.get("interrupted", [])}
    case_root = _get_batch_progress_path(mode_name, model_name, develop).parent
    discovered_row_ids = set(completed | failed | interrupted)
    for case_dir in case_root.glob("case_*"):
        if not case_dir.is_dir():
            continue
        try:
            discovered_row_ids.add(int(case_dir.name.replace("case_", "")))
        except ValueError:
            continue


    for row_id in sorted(discovered_row_ids):
        case_dir = case_root / f"case_{row_id}"
        if not case_dir.exists():
            case_dir = build_case_dir(mode_name, model_name, row_id, develop=develop)
        summary, module_state = _refresh_summary_from_modules(
            case_dir=case_dir,
            mode_name=mode_name,
            model_name=model_name,
            row_id=row_id,
            max_attempts=2,
        )
        if module_state is not None:
            final_status = module_state.final_status or ""
        elif summary is not None:
            final_status = _resolve_effective_final_status(summary, case_dir)
        else:
            continue
        completed.discard(row_id)
        failed.discard(row_id)
        interrupted.discard(row_id)
        if final_status == "compiled":
            completed.add(row_id)
        elif final_status == "recovered_retry_ready":
            interrupted.add(row_id)
        elif final_status == "interrupted_llm_connectivity":
            interrupted.add(row_id)
        elif final_status:
            failed.add(row_id)


    synced = dict(progress)
    synced["completed"] = sorted(completed)
    synced["failed"] = sorted(failed)
    synced["interrupted"] = sorted(interrupted)
    return synced


def _can_prepare_case_now(mode_name: str, model_name: str, row_id: int) -> bool:

    """Check whether the current prepare/classify pipeline can now accept this row."""
    try:
        candidate = load_initial_candidate(mode_name, model_name, row_id)
        cleaned = clean_generated_code(candidate["generated_cpp"], decode_escaped_newlines=False)
        classify_candidate_code(
            cleaned["cleaned_code"],
            mode_name=mode_name,
            model_name=model_name,
            row_id=row_id,
        )
        return True
    except Exception:
        return False


def _has_agent_run_summary(mode_name: str, model_name: str, row_id: int, develop: bool) -> bool:

    case_dir = build_case_dir(mode_name, model_name, row_id, develop=develop)
    return (case_dir / "meta" / "agent_run_summary.json").exists()


def _collect_retryable_failed_rows(

    mode_name: str,
    model_name: str,
    failed: set[int],
    start_from: int | None,
    develop: bool,

) -> tuple[set[int], dict[str, set[int]]]:

    retryable: set[int] = set()
    retryable_by_type: dict[str, set[int]] = {}
    for row_id in sorted(failed):
        if start_from is not None and row_id < start_from:
            continue
        retry_type = _classify_retryable_failed_row(mode_name, model_name, row_id, develop)
        if retry_type is not None:
            retryable.add(row_id)
            retryable_by_type.setdefault(retry_type, set()).add(row_id)
    return retryable, retryable_by_type


def _build_batch_row_queue(

    mode_name: str,
    model_name: str,
    all_row_ids: list[int],
    completed: set[int],
    failed: set[int],
    interrupted: set[int],
    start_from: int | None,
    last_row_id: int | None,
    develop: bool,

) -> tuple[list[int], int, set[int]]:

    """Build batch execution order.



    Default behavior:

    1. Retry interrupted rows first.

    2. Retry previously failed rows that now pass the current prepare/classify gate.

    3. Then process every row that has never been handled yet.



    If start_from is provided, both interrupted retries and new rows are filtered

    to row_id >= start_from.



    all_row_ids is the authoritative row set for this invocation -- the caller has

    already applied --row-ids-file to it (run_agent_batch:3740-3742) -- so the retry

    lists, which come from batch_progress.json, are intersected with it below.



    This matters only when --row-ids-file is used *without* a matching --output-root.

    glm_groups/launch_limit1_all_groups.py:114-143 always pairs the two, giving each

    group its own cases/ dir, so its batch_progress.json only ever holds that group's

    rows and the intersection is a no-op. stage2 instead slices one shared output_root

    into chunks, where batch_progress.json spans the whole (mode,model); without the

    intersection a 5-row chunk expands back to every interrupted/retryable row on

    record (measured 2026-08-01: 5-row chunk -> 40-row queue, 11 cases processed),

    and two concurrent chunks would contend for the same rows.

    """
    allowed_rows = set(all_row_ids)
    # Filter *before* _collect_retryable_failed_rows: it does a per-row disk scan
    # (_classify_retryable_failed_row), so screening the input also avoids paying
    # for rows this invocation would discard anyway.
    failed = {row_id for row_id in failed if row_id in allowed_rows}
    interrupted = {row_id for row_id in interrupted if row_id in allowed_rows}
    retryable_failed_rows, retryable_failed_rows_by_type = _collect_retryable_failed_rows(mode_name, model_name, failed, start_from, develop)


    if start_from is not None:
        start_index = next((i for i, rid in enumerate(all_row_ids) if rid >= start_from), 0)
        interrupted_rows = sorted(
            row_id for row_id in interrupted if row_id >= start_from and row_id not in completed and row_id not in failed
        )
        failed_retry_rows = sorted(row_id for row_id in retryable_failed_rows if row_id >= start_from and row_id not in completed)
    else:
        start_index = next((i for i, rid in enumerate(all_row_ids) if rid > last_row_id), 0) if last_row_id is not None else 0
        interrupted_rows = sorted(row_id for row_id in interrupted if row_id not in completed and row_id not in failed)
        failed_retry_rows = sorted(row_id for row_id in retryable_failed_rows if row_id not in completed)


    queued_rows = set(interrupted_rows) | set(failed_retry_rows)
    remaining_rows = [
        row_id
        for row_id in all_row_ids
        if (start_from is None or row_id >= start_from)
        and row_id not in completed
        and row_id not in failed
        and row_id not in interrupted
        and row_id not in queued_rows
    ]
    if remaining_rows:
        start_index = next((i for i, rid in enumerate(all_row_ids) if rid == remaining_rows[0]), start_index)
    elif interrupted_rows or failed_retry_rows:
        first_queued_row = (interrupted_rows + failed_retry_rows)[0]
        start_index = next((i for i, rid in enumerate(all_row_ids) if rid == first_queued_row), start_index)
    elif start_from is None and last_row_id is not None:
        start_index = next((i for i, rid in enumerate(all_row_ids) if rid > last_row_id), len(all_row_ids))
    execution_queue = interrupted_rows + failed_retry_rows + remaining_rows
    setattr(_build_batch_row_queue, "_last_retryable_failed_rows_by_type", retryable_failed_rows_by_type)
    return execution_queue, start_index, retryable_failed_rows


def _advance_last_row_id(previous_last_row_id: int | None, row_id: int) -> int:

    if previous_last_row_id is None:
        return row_id
    return max(previous_last_row_id, row_id)


def run_batch_cases(

    mode_name: str,
    model_name: str,
    max_attempts: int = 2,
    force_prepare: bool = False,
    develop: bool = False,
    start_from: int | None = None,
    limit: int | None = None,

) -> dict[str, Any]:

    """Run agent cases in batch mode with resume support



    Args:

        limit: Maximum number of cases to process (excluding skipped cases)

    """
    all_row_ids = _get_all_row_ids()
    row_ids_override = resolve_row_ids_override()
    if row_ids_override is not None:
        allowed = {int(v) for v in row_ids_override}
        all_row_ids = [row_id for row_id in all_row_ids if row_id in allowed]
    progress = _load_batch_progress(mode_name, model_name, develop)
    progress = _sync_batch_progress_with_case_summaries(mode_name, model_name, develop, progress)
    _save_batch_progress(mode_name, model_name, develop, progress)


    completed = set(progress.get("completed", []))
    failed = set(progress.get("failed", []))
    interrupted = set(progress.get("interrupted", []))  # Network/LLM errors - should retry
    last_row_id = progress.get("last_row_id")
    execution_queue, start_index, retryable_failed_rows = _build_batch_row_queue(
        mode_name=mode_name,
        model_name=model_name,
        all_row_ids=all_row_ids,
        completed=completed,
        failed=failed,
        interrupted=interrupted,
        start_from=start_from,
        last_row_id=last_row_id,
        develop=develop,
    )
    retryable_failed_rows_by_type = getattr(_build_batch_row_queue, "_last_retryable_failed_rows_by_type", {})


    results = {
        "mode_name": mode_name,
        "model_name": model_name,
        "total_cases": len(all_row_ids),
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "interrupted": 0,
        "skipped": 0,
        "cases": [],
    }


    print(f"Starting batch processing: {mode_name}/{model_name}")
    print(f"Total cases: {len(all_row_ids)}, Starting from index: {start_index}")
    if limit is not None:
        print(f"Limit: will process at most {limit} cases (excluding skipped)")
    print(f"Previously completed: {len(completed)}, Previously failed: {len(failed)}, Previously interrupted: {len(interrupted)}")
    if interrupted:
        print(f"Pending retries: {len([row_id for row_id in execution_queue if row_id in interrupted])} interrupted cases will be retried first")
    if retryable_failed_rows:
        print(f"Retryable failed rows under current recovery logic: {len(retryable_failed_rows)}")
        for retry_type in sorted(retryable_failed_rows_by_type):
            print(f"  - {retry_type}: {len(retryable_failed_rows_by_type[retry_type])}")
    print(f"Current execution queue size: {len(execution_queue)}")


    # Verify baseline compile once at the start of batch processing
    print(f"\n[Batch] Performing initial baseline compile check...")
    if not _verify_baseline_compile(mode_name, model_name):
        error_msg = (
            f"Baseline compile check repeatedly failed for {mode_name}/{model_name}. "

            f"Please inspect logs under {baseline_check_log_dir()}."
        )
        print(f"\n{'='*80}")
        print(f"[CRITICAL ERROR] {error_msg}")
        print(f"[CRITICAL ERROR] Please run: python 2_Baseline/agent/src/clean_all_test_cases.py")
        print(f"{'='*80}\n")
        raise RuntimeError(error_msg)
    print(f"[Batch] OK Initial baseline check passed, proceeding with batch processing\n")


    processed_count = 0  # Track actually processed cases (not skipped)
    baseline_check_interval = 5  # Check baseline every 5 processed cases


    for queue_index, row_id in enumerate(execution_queue, start=1):
        # Check limit before processing
        if limit is not None and processed_count >= limit:
            print(f"\n[Limit] Reached processing limit of {limit} cases, stopping")
            break


        # Periodic baseline check every 5 processed cases
        if processed_count > 0 and processed_count % baseline_check_interval == 0:
            print(f"\n[Batch] Periodic baseline check (after {processed_count} cases)...")
            if not _verify_baseline_compile(mode_name, model_name):
                error_msg = (
                    f"Baseline compile check repeatedly failed after processing {processed_count} cases "

                    f"for {mode_name}/{model_name}. Please inspect logs under {baseline_check_log_dir()}."
                )
                print(f"\n{'='*80}")
                print(f"[CRITICAL ERROR] {error_msg}")
                print(f"[CRITICAL ERROR] Stopping batch processing until the persistent baseline failure is understood.")
                print(f"[CRITICAL ERROR] Please run: python 2_Baseline/agent/src/clean_all_test_cases.py")
                print(f"{'='*80}\n")
                raise RuntimeError(error_msg)
            print(f"[Batch] OK Periodic baseline check passed\n")


        # Don't skip interrupted cases - they should be retried
        if row_id in interrupted:
            print(f"[{queue_index}/{len(execution_queue)}] Retrying row_id={row_id} (previously interrupted by network/LLM error)")
            interrupted.discard(row_id)  # Remove from interrupted set
        elif row_id in retryable_failed_rows:
            print(f"[{queue_index}/{len(execution_queue)}] Retrying row_id={row_id} (previously failed, but now passes current prepare/classify gate)")
            failed.discard(row_id)


        print(f"\n[{queue_index}/{len(execution_queue)}] Processing row_id={row_id}... (processed: {processed_count + 1}/{limit if limit else 'unlimited'})")


        try:
            result = run_agent_case(
                mode_name=mode_name,
                model_name=model_name,
                row_id=row_id,
                max_attempts=max_attempts,
                force_prepare=force_prepare,
                develop=develop,
                skip_baseline_check=True,
            )


            results["processed"] += 1
            processed_count += 1  # Increment processed count
            final_status = result.get("final_status")


            results["cases"].append({
                "row_id": row_id,
                "status": final_status,
                "summary_path": str(build_case_dir(mode_name, model_name, row_id, develop) / "meta" / "agent_run_summary.json"),
            })


            # Categorize based on final_status
            if final_status == "compiled":
                completed.add(row_id)
                results["succeeded"] += 1
                print(f"OK row_id={row_id} succeeded")
            elif final_status == "interrupted_llm_connectivity":
                # Network/LLM connectivity issues - should retry next time
                interrupted.add(row_id)
                results["interrupted"] += 1
                print(f"WARN row_id={row_id} interrupted (network/LLM error - will retry next time)")
            else:
                # Real failures: failed_llm_content, failed, etc.
                failed.add(row_id)
                results["failed"] += 1
                print(f"FAIL row_id={row_id} failed: {final_status}")


            # Update progress after each case
            progress["completed"] = sorted(list(completed))
            progress["failed"] = sorted(list(failed))
            progress["interrupted"] = sorted(list(interrupted))
            progress["last_row_id"] = _advance_last_row_id(progress.get("last_row_id"), row_id)
            _save_batch_progress(mode_name, model_name, develop, progress)


        except Exception as exc:
            print(f"FAIL row_id={row_id} exception: {exc}")
            # Exceptions are treated as interruptions (might be temporary)
            interrupted.add(row_id)
            results["interrupted"] += 1
            results["processed"] += 1
            processed_count += 1  # Increment processed count even on exception
            results["cases"].append({
                "row_id": row_id,
                "status": "exception",
                "error": str(exc),
            })


            # Update progress even on exception
            progress["interrupted"] = sorted(list(interrupted))
            progress["last_row_id"] = _advance_last_row_id(progress.get("last_row_id"), row_id)
            _save_batch_progress(mode_name, model_name, develop, progress)


    print(f"\nBatch processing complete:")
    print(f"  Processed: {results['processed']}")
    print(f"  Succeeded: {results['succeeded']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Interrupted (will retry): {results['interrupted']}")
    print(f"  Skipped: {results['skipped']}")


    return results


def main() -> None:

    _configure_stdio_for_live_progress()
    parser = argparse.ArgumentParser(description="Run agent feedback-repair loop for cases.")
    parser.add_argument("--mode", required=True, choices=METHOD_INPUT_NAMES)
    parser.add_argument("--model", required=True, choices=sorted(model_adapter.MODELS))
    parser.add_argument("--row-id", type=int, help="Process single row_id. If not specified, process all cases in batch mode.")
    parser.add_argument("--start-from", type=int, help="In batch mode, start from this row_id (inclusive)")
    parser.add_argument("--limit", type=int, help="In batch mode, maximum number of cases to process (excluding skipped cases)")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--skip-baseline-check", action="store_true")
    parser.add_argument("--develop", action="store_true")
    parser.add_argument("--package-name", help="Override CTS package/module name, e.g. director_glm_0")
    parser.add_argument("--cts-root", type=Path, help="Override CTS root for this execution")
    parser.add_argument("--output-root", type=Path, help="Override output root for this execution")
    parser.add_argument("--run-record-path", type=Path, help="Per-worker Supervisor record path")
    parser.add_argument("--gl-sim-vk-path", required=True, type=Path,
                        help="User-generated RQ2 retrieval output JSON")
    parser.add_argument("--candidate-root", type=Path, help="Override generated.cpp candidate root directory")
    parser.add_argument("--row-ids-file", type=Path, help="Optional JSON file containing row_ids for grouped batch execution")
    parser.add_argument(
        "--defer-compile",
        action="store_true",
        help=(
            "Generate/apply exactly the next repair, then return successfully "

            "without modifying or compiling a CTS tree"
        ),
    )
    parser.add_argument("--run-after-compile", action="store_true",
                        help="Execute deqp-vk after a successful compile and save runtime evidence")
    parser.add_argument("--runtime-timeout-seconds", type=int, default=300)
    model_adapter.add_runner_arguments(parser)
    args = parser.parse_args()
    # Accept paper-facing method names while retaining legacy internal mode
    # identifiers for CTS package layout and historical artifact paths.
    args.mode = get_method(args.mode).legacy


    if args.runtime_timeout_seconds < 1:
        parser.error("--runtime-timeout-seconds must be positive")
    if args.defer_compile and args.row_id is None:
        parser.error("--defer-compile requires --row-id")


    row_ids_override = None
    if args.row_ids_file is not None:
        row_ids_payload = _load_json(args.row_ids_file)
        if isinstance(row_ids_payload, dict):
            row_ids_override = row_ids_payload.get("row_ids")
        elif isinstance(row_ids_payload, list):
            row_ids_override = row_ids_payload
        else:
            raise ValueError("row-ids-file must be a JSON list or an object containing 'row_ids'")


    model_adapter.configure_from_args(args, required=True)
    set_execution_overrides(
        package_name=args.package_name,
        cts_root=args.cts_root,
        output_root=args.output_root,
        run_record_path=args.run_record_path,
        row_ids=row_ids_override,
        gl_sim_vk_path=args.gl_sim_vk_path,
        candidate_root=args.candidate_root,
        defer_compile=args.defer_compile,
        run_after_compile=args.run_after_compile,
        runtime_timeout_seconds=args.runtime_timeout_seconds,
    )
    try:
        _update_run_record(
            status="running",
            pid=os.getpid(),
            started_at=datetime.now().isoformat(timespec="seconds"),
            finished_at=None,
            error=None,
        )
        if args.row_id is not None:
            # Single case mode
            result = run_agent_case(
                mode_name=args.mode,
                model_name=args.model,
                row_id=args.row_id,
                max_attempts=args.max_attempts,
                force_prepare=args.force_prepare,
                develop=args.develop,
                skip_baseline_check=args.skip_baseline_check,
            )
            print(_json_dump(result))
            _update_run_record(
                status="finished",
                finished_at=datetime.now().isoformat(timespec="seconds"),
                result_summary={
                    "mode_name": args.mode,
                    "model_name": args.model,
                    "row_id": args.row_id,
                    "final_status": result.get("final_status"),
                },
            )
        else:
            # Batch mode with resume support
            result = run_batch_cases(
                mode_name=args.mode,
                model_name=args.model,
                max_attempts=args.max_attempts,
                force_prepare=args.force_prepare,
                develop=args.develop,
                start_from=args.start_from,
                limit=args.limit,
            )
            print(_json_dump(result))
            _update_run_record(
                status="finished",
                finished_at=datetime.now().isoformat(timespec="seconds"),
                result_summary={
                    "mode_name": args.mode,
                    "model_name": args.model,
                    "processed": result.get("processed"),
                    "succeeded": result.get("succeeded"),
                    "failed": result.get("failed"),
                    "interrupted": result.get("interrupted"),
                    "skipped": result.get("skipped"),
                },
            )
    except Exception as exc:
        _update_run_record(
            status="error",
            finished_at=datetime.now().isoformat(timespec="seconds"),
            error=str(exc),
        )
        raise
    finally:
        clear_execution_overrides()


if __name__ == "__main__":

    main()
