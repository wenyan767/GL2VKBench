"""CTS 编译与运行期验证。

源自 compiler_adapter.py，去掉三处不可公开的耦合：

  * 删除 CTS_ROOT_BY_MODE / CTS_ROOT_BY_MODE_MODEL 两张硬编码映射表
    （共 12 个本机绝对路径）。CTS 根现在必须由 --cts-root 显式传入。
  * module_pristine 快照落到产物根下；它是运行期自动生成的，不随包分发
    （原目录名由本机路径派生，会泄露开发机布局）。
  * 本地模块导入改为从 core / paths 取。

namespace 重写在这里：assemble_cpp 的输出内层 namespace 恒为模板字面量
custom_migration，挂载前替换成 {package}_{model}。这是编译侧职责，不在 core。
"""

from __future__ import annotations

import json
import locale
import os
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from paths import DEFAULT_RUNS_DIR
from core import acquire_build_lock, build_lock_scope_for_cts_root
from core import (
    resolve_cts_root_override,
    resolve_package_name,
    resolve_run_after_compile,
    resolve_runtime_root_group,
    resolve_runtime_timeout_seconds,
    resolve_target_name,
)
from core import build_output_dir




BUILD_CONFLICT_MAX_RETRIES = 5
BUILD_CONFLICT_WAIT_MIN_SECONDS = 4.0
BUILD_CONFLICT_WAIT_MAX_SECONDS = 12.0
COMPILE_SUBPROCESS_TIMEOUT_SECONDS = 900
COMPILE_RECOVERY_MAX_RETRIES = 3
COMPILE_RECOVERY_WAIT_MIN_SECONDS = 2.0
COMPILE_RECOVERY_WAIT_MAX_SECONDS = 6.0
MODULE_FILE_OP_MAX_RETRIES = 5
MODULE_FILE_OP_WAIT_MIN_SECONDS = 4.0
MODULE_FILE_OP_WAIT_MAX_SECONDS = 12.0
MODULE_PRISTINE_ROOT = DEFAULT_RUNS_DIR / "_module_pristine"


@dataclass
class CompileResult:
    returncode: int
    command: list[str]
    target: str
    stdout_path: str
    stderr_path: str
    injected_cpp_path: str
    restored: bool
    duration_seconds: float
    attempt_index: int
    runtime_summary_path: str | None = None


@dataclass
class ModuleDriftReport:
    module_dir: str
    module_name: str
    canonical_prefix: str
    discovered_prefixes: list[str]
    cmake_prefixes: list[str]
    drift_files: list[str]
    custom_case_files: list[str]
    backup_files: list[str]
    has_mixed_family_drift: bool
    has_cmake_drift: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "module_dir": self.module_dir,
            "module_name": self.module_name,
            "canonical_prefix": self.canonical_prefix,
            "discovered_prefixes": self.discovered_prefixes,
            "cmake_prefixes": self.cmake_prefixes,
            "drift_files": self.drift_files,
            "custom_case_files": self.custom_case_files,
            "backup_files": self.backup_files,
            "has_mixed_family_drift": self.has_mixed_family_drift,
            "has_cmake_drift": self.has_cmake_drift,
        }


def _write_compile_runtime_status(
    compile_dir: Path,
    *,
    row_id: int,
    attempt_index: int,
    module_name: str,
    target: str,
    command: list[str],
    stage: str,
    started_at: float,
    extra: dict | None = None,
) -> None:
    payload = {
        "row_id": row_id,
        "attempt_index": attempt_index,
        "module_name": module_name,
        "target": target,
        "command": command,
        "pid": os.getpid(),
        "stage": stage,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started_at)),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }
    if extra:
        payload.update(extra)
    (compile_dir / "compile_runtime_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _decode_process_output(raw: bytes) -> str:
    if not raw:
        return ""

    candidates: list[str] = []
    preferred = locale.getpreferredencoding(False)
    if preferred:
        candidates.append(preferred)
    candidates.extend(["utf-8", "gb18030", "cp936", "utf-16le"])

    seen: set[str] = set()
    best_text = raw.decode("utf-8", errors="replace")
    best_score = (best_text.count("\ufffd"), best_text.count("鈥"), best_text.count("鍙"))

    for encoding in candidates:
        normalized = encoding.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            decoded = raw.decode(encoding)
        except UnicodeDecodeError:
            decoded = raw.decode(encoding, errors="replace")
        score = (decoded.count("\ufffd"), decoded.count("鈥"), decoded.count("鍙"))
        if score < best_score:
            best_text = decoded
            best_score = score

    return best_text


def _find_deqp_vk_executable(cts_build_dir: Path) -> Path:
    candidates = [
        cts_build_dir / "external" / "vulkancts" / "modules" / "vulkan" / "Debug" / "deqp-vk.exe",
        cts_build_dir / "external" / "vulkancts" / "modules" / "vulkan" / "Release" / "deqp-vk.exe",
        cts_build_dir / "external" / "vulkancts" / "modules" / "deqp-vk.exe",
        cts_build_dir / "external" / "vulkancts" / "modules" / "vulkan" / "deqp-vk",
        cts_build_dir / "external" / "vulkancts" / "modules" / "deqp-vk",
    ]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(f"deqp-vk executable not found under {cts_build_dir}")
    return max(existing, key=lambda path: path.stat().st_mtime)


def _run_runtime_command(
    command: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
) -> tuple[int, str, str, float]:
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=stdout_handle,
            stderr=stderr_handle,
            timeout=timeout_seconds,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    return completed.returncode, stdout_text, stderr_text, time.time() - started


def _parse_caselist(stdout_text: str, candidates: list[str]) -> list[str]:
    discovered: list[str] = []
    for raw_line in stdout_text.splitlines():
        line = raw_line.strip()
        if line.startswith("TEST: "):
            line = line[len("TEST: "):].strip()
        if line.startswith("dEQP-VK."):
            discovered.append(line)
    candidate_set = set(candidates)
    exact = [path for path in discovered if path in candidate_set]
    return list(dict.fromkeys(exact or discovered))


def build_runtime_execution_evidence(
    returncode: int,
    qpa_path: Path,
) -> dict[str, object]:
    """Classify a completed deqp invocation without changing compile success."""
    qpa_text = (
        qpa_path.read_text(encoding="utf-8", errors="replace")
        if qpa_path.is_file()
        else ""
    )
    status_codes = re.findall(r'<Result\s+StatusCode="([^"]+)"', qpa_text)
    normalized = [status.strip().lower() for status in status_codes]
    unsigned_returncode = returncode & 0xFFFFFFFF
    looks_like_crash = returncode < 0 or unsigned_returncode >= 0x80000000

    if returncode != 0:
        outcome = "crashed" if looks_like_crash else "failed"
    elif not normalized:
        outcome = "incomplete"
    elif all(status == "pass" for status in normalized):
        outcome = "passed"
    elif all(status == "notsupported" for status in normalized):
        outcome = "not_supported"
    elif any(status in {"crash", "timeout"} for status in normalized):
        outcome = "crashed"
    elif any(status in {"fail", "internalerror", "resourceerror"}
             for status in normalized):
        outcome = "failed"
    else:
        outcome = "completed_with_nonpass"

    return {
        "execution_outcome": outcome,
        "execute_returncode_hex": f"0x{unsigned_returncode:08X}",
        "qpa_status_codes": status_codes,
        "qpa_complete": "#endSession" in qpa_text,
    }


def _inject_runtime_registry(cts_root: Path, mode_name: str,
                             model_name: str) -> tuple[Path, str]:
    """临时把当前自定义模块注册进 dEQP-VK TestPackage，返回原文用于恢复。"""
    registry_path = (
        cts_root / "external" / "vulkancts" / "modules" / "vulkan"
        / "vktTestPackage.cpp"
    )
    original = registry_path.read_text(encoding="utf-8")
    prefix = _module_class_prefix(mode_name, model_name)
    namespace_name = _build_generated_namespace(mode_name, model_name)
    root_group = resolve_runtime_root_group(mode_name, model_name)
    include_line = f'#include "vkt{prefix}Tests.hpp"'
    registration_line = (
        f'    addRootChild("{root_group}", m_caseListFilter, '
        f'{namespace_name}::createTests);'
    )
    patched = original
    if include_line not in patched:
        include_anchor = next((
            anchor for anchor in (
                '#include "vktCustomMigrationTests.hpp"',
                '#include "vktDataGraphTests.hpp"',
            )
            if anchor in patched
        ), None)
        if include_anchor is None:
            raise RuntimeError(f"runtime registry include anchor missing: {registry_path}")
        patched = patched.replace(include_anchor, include_anchor + "\n" + include_line, 1)
    if registration_line not in patched:
        registration_anchor = next((
            anchor for anchor in (
                (
                    '    addRootChild("custom_migration", m_caseListFilter, '
                    'custom_migration::createTests);'
                ),
                (
                    '    addRootChild("fragment_shading_rate", m_caseListFilter, '
                    'FragmentShadingRate::createTests);'
                ),
            )
            if anchor in patched
        ), None)
        if registration_anchor is None:
            raise RuntimeError(f"runtime registry registration anchor missing: {registry_path}")
        # Vulkan 与 Vulkan SC 分支各有一个 anchor；两边都注册以保持同源文件可编译。
        patched = patched.replace(
            registration_anchor,
            registration_anchor + "\n" + registration_line,
        )
    registry_path.write_text(patched, encoding="utf-8")
    _force_refresh_timestamp(registry_path)
    return registry_path, original


def _run_runtime_after_successful_compile(
    *,
    mode_name: str,
    model_name: str,
    row_id: int,
    compile_dir: Path,
    cts_root: Path,
    cts_build_dir: Path,
) -> Path:
    """在模块仍注入且 build lock 仍持有时执行，并只返回证据路径。

    运行构建、case 解析或执行失败都不会改变模块编译的 returncode；失败状态写入
    runtime_summary.json，外层 finally 随后无条件恢复模块源码。
    """
    runtime_dir = compile_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    summary_path = runtime_dir / "runtime_summary.json"
    timeout_seconds = resolve_runtime_timeout_seconds()
    started = time.time()
    root_group = resolve_runtime_root_group(mode_name, model_name)
    case_name = f"case_{row_id}"
    candidate_paths = [
        f"dEQP-VK.{root_group}.smoke.{case_name}",
        f"dEQP-VK.{root_group}.{case_name}",
    ]
    payload: dict[str, object] = {
        "status": "runtime_initialized",
        "mode_name": mode_name,
        "model_name": model_name,
        "row_id": row_id,
        "root_group": root_group,
        "candidate_paths": candidate_paths,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "source_restored": False,
        "runtime_registry_restored": False,
    }
    registry_path: Path | None = None
    registry_original: str | None = None
    try:
        registry_path, registry_original = _inject_runtime_registry(
            cts_root, mode_name, model_name
        )
        payload["runtime_registry_path"] = str(registry_path)
        runtime_build_command = ["cmake", "--build", str(cts_build_dir), "--target", "deqp-vk"]
        build_rc, _, _ = _run_cmake_build_with_backoff(
            runtime_build_command,
            cts_root,
            stdout_path=runtime_dir / "build_stdout.txt",
            stderr_path=runtime_dir / "build_stderr.txt",
        )
        payload.update({
            "runtime_build_command": runtime_build_command,
            "runtime_build_returncode": build_rc,
        })
        if build_rc != 0:
            payload["status"] = "runtime_build_failed"
            return summary_path

        executable = _find_deqp_vk_executable(cts_build_dir)
        case_filter = f"dEQP-VK.{root_group}.case_{row_id}.*"
        caselist_command = [
            str(executable),
            "--deqp-runmode=stdout-caselist",
            f"--deqp-case={case_filter}",
            f"--deqp-log-filename={runtime_dir / 'Caselist.qpa'}",
        ]
        caselist_rc, caselist_stdout, _, caselist_seconds = _run_runtime_command(
            caselist_command,
            cwd=executable.parent,
            stdout_path=runtime_dir / "caselist_stdout.txt",
            stderr_path=runtime_dir / "caselist_stderr.txt",
            timeout_seconds=timeout_seconds,
        )
        matches = _parse_caselist(caselist_stdout, candidate_paths)
        payload.update({
            "executable": str(executable),
            "case_filter": case_filter,
            "caselist_command": caselist_command,
            "caselist_returncode": caselist_rc,
            "caselist_duration_seconds": round(caselist_seconds, 3),
            "matches": matches,
        })
        if caselist_rc != 0 or not matches:
            payload["status"] = "case_resolution_failed"
            return summary_path

        resolved_case_path = matches[0] if len(matches) == 1 else None
        qpa_path = runtime_dir / "TestResults.qpa"
        run_command = [
            str(executable),
            f"--deqp-case={case_filter}",
            "--deqp-watchdog=disable",
            "--deqp-log-images=disable",
            "--deqp-log-shader-sources=disable",
            f"--deqp-log-filename={qpa_path}",
        ]
        run_rc, _, _, run_seconds = _run_runtime_command(
            run_command,
            cwd=executable.parent,
            stdout_path=runtime_dir / "run_stdout.txt",
            stderr_path=runtime_dir / "run_stderr.txt",
            timeout_seconds=timeout_seconds,
        )
        payload.update({
            "status": "executed",
            "resolved_case_path": resolved_case_path,
            "resolved_case_paths": matches,
            "run_command": run_command,
            "execute_returncode": run_rc,
            "run_duration_seconds": round(run_seconds, 3),
            "qpa_path": str(qpa_path),
            "qpa_exists": qpa_path.exists(),
            **build_runtime_execution_evidence(run_rc, qpa_path),
        })
        return summary_path
    except Exception as exc:
        payload.update({
            "status": "runtime_exception",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        return summary_path
    finally:
        if registry_path is not None and registry_original is not None:
            try:
                registry_path.write_text(registry_original, encoding="utf-8")
                _force_refresh_timestamp(registry_path)
                payload["runtime_registry_restored"] = True
            except Exception as exc:
                payload["runtime_registry_restore_error"] = str(exc)
        payload["elapsed_seconds"] = round(time.time() - started, 3)
        summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


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


def _run_cmake_build_with_backoff(
    command: list[str],
    cwd: Path,
    *,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, str, str]:
    last_result: tuple[int, str, str] | None = None
    for attempt_index in range(1, BUILD_CONFLICT_MAX_RETRIES + 1):
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                stdout=stdout_handle,
                stderr=stderr_handle,
                timeout=COMPILE_SUBPROCESS_TIMEOUT_SECONDS,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        last_result = (completed.returncode, stdout_text, stderr_text)
        if completed.returncode == 0 or not _is_build_conflict(stdout_text, stderr_text):
            return last_result
        if attempt_index >= BUILD_CONFLICT_MAX_RETRIES:
            return last_result
        delay_seconds = random.uniform(BUILD_CONFLICT_WAIT_MIN_SECONDS, BUILD_CONFLICT_WAIT_MAX_SECONDS)
        print(
            f"[Build Retry] Conflict detected for {' '.join(command)} "
            f"(attempt {attempt_index}/{BUILD_CONFLICT_MAX_RETRIES}), waiting {delay_seconds:.1f}s"
        )
        time.sleep(delay_seconds)
    assert last_result is not None
    return last_result


def build_case_dir(mode_name: str, model_name: str, row_id: int, develop: bool = False) -> Path:
    return build_output_dir(mode_name, model_name, row_id, develop=develop)


def resolve_cts_root(mode_name: str, model_name: str | None = None) -> Path:
    """返回 CTS 树根。必须已由 --cts-root 显式设置，没有默认值。

    原实现在这里回落到两张硬编码映射表（E:\VK-GL-CTS_02 等）。那些树同时被
    其它流程占用，猜错会把代码注入正在使用的树——实测有过残留挂载导致整批
    候选被拒的先例。所以这里宁可报错也不猜。
    """
    override = resolve_cts_root_override()
    if override is None:
        raise ValueError(
            "未设置 CTS 根。请显式传 --cts-root <VK-GL-CTS 检出路径>；"
            "本包不猜测 CTS 位置，也没有内置默认值。")
    return override


def resolve_cts_build_dir(mode_name: str, model_name: str | None = None) -> Path:
    return resolve_cts_root(mode_name, model_name) / "build"


def resolve_cts_vulkan_dir(mode_name: str, model_name: str | None = None) -> Path:
    return resolve_cts_root(mode_name, model_name) / "external" / "vulkancts" / "modules" / "vulkan"


def _safe_token(value: str) -> str:
    return value.replace("-", "_")


def _camel_case(*parts: str) -> str:
    tokens: list[str] = []
    for part in parts:
        tokens.extend(token for token in part.split("_") if token)
    return "".join(token.capitalize() for token in tokens)


def _module_name(mode_name: str, model_name: str) -> str:
    return resolve_package_name(mode_name, model_name)


def _module_target(mode_name: str, model_name: str) -> str:
    return resolve_target_name(mode_name, model_name)


def _module_class_prefix(mode_name: str, model_name: str) -> str:
    return _camel_case(*resolve_package_name(mode_name, model_name).split("_"))


def _module_dir(mode_name: str, model_name: str) -> Path:
    return resolve_cts_vulkan_dir(mode_name, model_name) / _module_name(mode_name, model_name)


def _module_cmake_path(mode_name: str, model_name: str) -> Path:
    return _module_dir(mode_name, model_name) / "CMakeLists.txt"


def _module_group_cpp_path(mode_name: str, model_name: str) -> Path:
    prefix = _module_class_prefix(mode_name, model_name)
    return _module_dir(mode_name, model_name) / "group" / f"vkt{prefix}Group.cpp"


def _module_group_hpp_name(mode_name: str, model_name: str) -> str:
    prefix = _module_class_prefix(mode_name, model_name)
    return f"group/vkt{prefix}Group.hpp"


def _module_smoke_cpp_path(mode_name: str, model_name: str) -> Path:
    prefix = _module_class_prefix(mode_name, model_name)
    return _module_dir(mode_name, model_name) / "test_case" / f"vkt{prefix}SmokeTests.cpp"


def _module_smoke_hpp_name(mode_name: str, model_name: str) -> str:
    prefix = _module_class_prefix(mode_name, model_name)
    return f"test_case/vkt{prefix}SmokeTests.hpp"


def _relative_to_module(module_dir: Path, path: Path) -> str:
    return str(path.relative_to(module_dir)).replace("\\", "/")


def _extract_vkt_family_prefix(name: str) -> str | None:
    match = re.match(r"^vkt([A-Za-z0-9]+?)(Tests|Group|SmokeTests)(?:\.(?:cpp|hpp))?(?:\.bak)?$", name)
    if not match:
        return None
    return match.group(1)


def _cmake_family_prefixes(cmake_text: str) -> set[str]:
    prefixes: set[str] = set()
    for match in re.finditer(r"vkt([A-Za-z0-9]+?)(Tests|Group|SmokeTests)\.(?:cpp|hpp)", cmake_text):
        prefixes.add(match.group(1))
    return prefixes


def diagnose_module_drift(module_dir: Path, mode_name: str, model_name: str) -> ModuleDriftReport:
    canonical_prefix = _module_class_prefix(mode_name, model_name)
    discovered_prefixes: set[str] = set()
    drift_files: list[str] = []
    custom_case_files: list[str] = []
    backup_files: list[str] = []

    for relative_dir in (Path("."), Path("group"), Path("test_case")):
        scan_dir = module_dir / relative_dir
        if not scan_dir.exists():
            continue
        for path in scan_dir.iterdir():
            if not path.is_file():
                continue
            relpath = _relative_to_module(module_dir, path)
            name = path.name
            if name.startswith("vktCustomCase_"):
                custom_case_files.append(relpath)
                continue
            if name.endswith(".bak"):
                backup_files.append(relpath)
            prefix = _extract_vkt_family_prefix(name)
            if prefix is None:
                continue
            discovered_prefixes.add(prefix)
            if prefix != canonical_prefix:
                drift_files.append(relpath)

    cmake_prefixes: set[str] = set()
    cmake_path = module_dir / "CMakeLists.txt"
    if cmake_path.exists():
        cmake_prefixes = _cmake_family_prefixes(cmake_path.read_text(encoding="utf-8", errors="replace"))

    return ModuleDriftReport(
        module_dir=str(module_dir),
        module_name=module_dir.name,
        canonical_prefix=canonical_prefix,
        discovered_prefixes=sorted(discovered_prefixes),
        cmake_prefixes=sorted(cmake_prefixes),
        drift_files=sorted(drift_files),
        custom_case_files=sorted(custom_case_files),
        backup_files=sorted(backup_files),
        has_mixed_family_drift=any(prefix != canonical_prefix for prefix in discovered_prefixes),
        has_cmake_drift=any(prefix != canonical_prefix for prefix in cmake_prefixes),
    )


def _cleanup_module_drift_files(module_dir: Path, mode_name: str, model_name: str) -> ModuleDriftReport:
    report = diagnose_module_drift(module_dir, mode_name, model_name)
    for relpath in report.custom_case_files + report.backup_files + report.drift_files:
        target_path = module_dir / relpath
        if target_path.exists():
            _run_module_file_op_with_retry("cleanup_unlink", target_path, target_path.unlink)
    return report


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_attempt_dir(case_dir: Path, attempt_index: int | None) -> Path:
    if attempt_index is None or attempt_index <= 1:
        return case_dir
    return case_dir / "attempts" / f"attempt_{attempt_index}"


def _resolve_assembled_path(case_dir: Path, attempt_index: int | None) -> Path:
    attempt_dir = _resolve_attempt_dir(case_dir, attempt_index)
    return attempt_dir / "normalized" / "assembled_candidate.cpp"


def _resolve_classification_path(case_dir: Path, attempt_index: int | None) -> Path:
    attempt_dir = _resolve_attempt_dir(case_dir, attempt_index)
    return attempt_dir / "meta" / "classification.json"


def _build_generated_basename(mode_name: str, model_name: str, row_id: int) -> str:
    return f"vktCustomCase_{_safe_token(mode_name)}_{_safe_token(model_name)}_{row_id}"


def _build_populate_symbol(mode_name: str, model_name: str, row_id: int) -> str:
    return f"populate_{_safe_token(mode_name)}_{_safe_token(model_name)}_{row_id}"


def _build_generated_namespace(mode_name: str, model_name: str) -> str:
    return _module_name(mode_name, model_name)


def _render_generated_compile_unit(
    assembled_candidate: str,
    register_name: str,
    case_name: str,
    namespace_name: str,
    populate_symbol: str,
) -> str:
    bridge = f"""

void {populate_symbol}(tcu::TestCaseGroup* group)
{{
    {register_name}(group);
}}
"""
    code = assembled_candidate.replace("namespace custom_migration", f"namespace {namespace_name}", 1)
    namespace_close = f"}} // {namespace_name}"
    code = code.replace("} // custom_migration", namespace_close)
    if namespace_close in code:
        return code.replace(namespace_close, bridge.rstrip() + "\n\n" + namespace_close, 1).rstrip() + "\n"
    return code.rstrip() + "\n" + bridge.lstrip()


def _render_generated_header(basename: str, namespace_name: str, populate_symbol: str) -> str:
    include_guard = f"_{basename.upper()}_HPP"
    return f"""#ifndef {include_guard}
#define {include_guard}

#include "tcuDefs.hpp"
#include "tcuTestCase.hpp"

namespace tcu
{{
class TestCaseGroup;
}} // namespace tcu

namespace vkt
{{
namespace {namespace_name}
{{

void {populate_symbol}(tcu::TestCaseGroup* group);

}} // namespace {namespace_name}
}} // namespace vkt

#endif // {include_guard}
"""


def _render_group_cpp(mode_name: str, model_name: str, header_include: str,
                      populate_symbol: str, case_name: str) -> str:
    prefix = _module_class_prefix(mode_name, model_name)
    namespace_name = _build_generated_namespace(mode_name, model_name)
    if mode_name == "only-fix":
        return f"""#include "{_module_group_hpp_name(mode_name, model_name)}"

#include "{_module_smoke_hpp_name(mode_name, model_name)}"
#include "{header_include}"
#include "vktTestGroupUtil.hpp"

namespace vkt
{{
namespace {namespace_name}
{{

void create{prefix}Groups(tcu::TestCaseGroup* group)
{{
    addTestGroup(group, "smoke", populate{prefix}SmokeTests);
    addTestGroup(group, "{case_name}", {populate_symbol});
}}

}} // namespace {namespace_name}
}} // namespace vkt
"""
    return f"""#include "{_module_group_hpp_name(mode_name, model_name)}"

#include "{header_include}"
#include "vktTestGroupUtil.hpp"

namespace vkt
{{
namespace {namespace_name}
{{

void create{prefix}Groups(tcu::TestCaseGroup* group)
{{
    addTestGroup(group, "{case_name}", {populate_symbol});
}}

}} // namespace {namespace_name}
}} // namespace vkt
"""


def _render_smoke_cpp(mode_name: str, model_name: str, header_include: str | None, populate_symbol: str | None) -> str:
    prefix = _module_class_prefix(mode_name, model_name)
    namespace_name = _build_generated_namespace(mode_name, model_name)
    # A caller may keep only-fix repair semantics while selecting a distinct
    # physical package (shot_only_fix_*).  Symbols must follow that physical
    # identity or they disagree with the package's generated header.
    smoke_model_name = _safe_token(_module_name(mode_name, model_name))
    if mode_name == "only-fix":
        smoke_model_name = re.sub(r"_\d+$", "", smoke_model_name)
    smoke_symbol = f"{smoke_model_name}_smoke"
    extra_include = f'#include "{header_include}"\n' if header_include else ""
    extra_call = f"    {populate_symbol}(group);\n" if populate_symbol else ""
    return f"""#include "{_module_smoke_hpp_name(mode_name, model_name)}"
{extra_include}
#include "deDefs.hpp"
#include "vktTestCaseUtil.hpp"

namespace vkt
{{
namespace {namespace_name}
{{

void populate{prefix}SmokeTests(tcu::TestCaseGroup* group)
{{
    addFunctionCase(group, "basic", {smoke_symbol});
{extra_call}}}

tcu::TestStatus {smoke_symbol}(Context& context)
{{
    DE_UNREF(context);
    return tcu::TestStatus::pass("{smoke_symbol}");
}}

}} // namespace {namespace_name}
}} // namespace vkt
"""


def _render_cmake(mode_name: str, model_name: str, generated_cpp_relpath: str) -> str:
    prefix = _module_class_prefix(mode_name, model_name)
    var_name = prefix.upper()
    target = _module_target(mode_name, model_name)
    target_sc = target.replace("deqp-vk-", "deqp-vksc-")
    return f"""include_directories(
    ..
    .
    group
    test_case
    ${{DEQP_INL_DIR}}
    )

set({var_name}_SRCS
    vkt{prefix}Tests.cpp
    group/vkt{prefix}Group.cpp
    test_case/vkt{prefix}SmokeTests.cpp
    {generated_cpp_relpath}
    )

PCH({var_name}_SRCS ../pch.cpp)

add_library({target} STATIC ${{{var_name}_SRCS}})
target_link_libraries({target} PUBLIC tcutil vkutil)

add_library({target_sc} STATIC ${{{var_name}_SRCS}})
target_include_directories({target_sc} BEFORE PUBLIC ${{CMAKE_SOURCE_DIR}}/external/vulkancts/framework/vulkan/generated/vulkansc)
target_link_libraries({target_sc} PUBLIC deqp-vksc-util tcutil vkscutil)
"""


def _render_initial_group_cpp(mode_name: str, model_name: str) -> str:
    prefix = _module_class_prefix(mode_name, model_name)
    namespace_name = _build_generated_namespace(mode_name, model_name)
    if mode_name == "only-fix":
        return f"""#include "vkt{prefix}Group.hpp"
#include "vkt{prefix}SmokeTests.hpp"
#include "vktTestGroupUtil.hpp"

namespace vkt
{{
namespace {namespace_name}
{{

void create{prefix}Groups(tcu::TestCaseGroup* group)
{{
    addTestGroup(group, "smoke", populate{prefix}SmokeTests);
}}

}} // namespace {namespace_name}
}} // namespace vkt
"""
    return f"""#include "vkt{prefix}Group.hpp"
#include "vkt{prefix}SmokeTests.hpp"
#include "vktTestGroupUtil.hpp"

namespace vkt
{{
namespace {namespace_name}
{{

void create{prefix}Groups(tcu::TestCaseGroup* group)
{{
    // Custom test cases will be added here during compilation
}}

}} // namespace {namespace_name}
}} // namespace vkt
"""


def _render_initial_cmake(mode_name: str, model_name: str) -> str:
    prefix = _module_class_prefix(mode_name, model_name)
    var_name = prefix.upper()
    target = _module_target(mode_name, model_name)
    target_sc = target.replace("deqp-vk-", "deqp-vksc-")
    return f"""include_directories(
    ..
    .
    group
    test_case
    ${{DEQP_INL_DIR}}
    )

set({var_name}_SRCS
    vkt{prefix}Tests.cpp
    group/vkt{prefix}Group.cpp
    test_case/vkt{prefix}SmokeTests.cpp
    )

PCH({var_name}_SRCS ../pch.cpp)

add_library({target} STATIC ${{{var_name}_SRCS}})
target_link_libraries({target} PUBLIC tcutil vkutil)

add_library({target_sc} STATIC ${{{var_name}_SRCS}})
target_include_directories({target_sc} BEFORE PUBLIC ${{CMAKE_SOURCE_DIR}}/external/vulkancts/framework/vulkan/generated/vulkansc)
target_link_libraries({target_sc} PUBLIC deqp-vksc-util tcutil vkscutil)
"""


def _sanitize_snapshot_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower() or "default"


def _module_pristine_dir(module_dir: Path) -> Path:
    cts_root = module_dir.parents[4]
    return MODULE_PRISTINE_ROOT / _sanitize_snapshot_token(str(cts_root)) / module_dir.name


def _baseline_module_files(mode_name: str, model_name: str) -> dict[str, str]:
    prefix = _module_class_prefix(mode_name, model_name)
    files = {
        "CMakeLists.txt": _render_initial_cmake(mode_name, model_name),
        f"group/vkt{prefix}Group.cpp": _render_initial_group_cpp(mode_name, model_name),
    }
    if mode_name == "only-fix":
        files[f"test_case/vkt{prefix}SmokeTests.cpp"] = _render_smoke_cpp(mode_name, model_name, None, None)
    return files


def _ensure_pristine_module_files(module_dir: Path, mode_name: str, model_name: str) -> Path:
    pristine_dir = _module_pristine_dir(module_dir)
    pristine_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, text in _baseline_module_files(mode_name, model_name).items():
        snapshot_path = pristine_dir / relative_path
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(text, encoding="utf-8")
    return pristine_dir


def _backup_file(src: Path, dst: Path) -> None:
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _is_permission_denied_error(exc: BaseException) -> bool:
    return isinstance(exc, PermissionError) or "permission denied" in str(exc).lower()


def _run_module_file_op_with_retry(action: str, path: Path, func) -> None:
    last_error: BaseException | None = None
    for attempt_index in range(1, MODULE_FILE_OP_MAX_RETRIES + 1):
        try:
            func()
            return
        except Exception as exc:
            last_error = exc
            if not _is_permission_denied_error(exc) or attempt_index >= MODULE_FILE_OP_MAX_RETRIES:
                raise
            delay_seconds = random.uniform(MODULE_FILE_OP_WAIT_MIN_SECONDS, MODULE_FILE_OP_WAIT_MAX_SECONDS)
            print(
                f"[Module File Retry] action={action} path={path} "
                f"attempt={attempt_index}/{MODULE_FILE_OP_MAX_RETRIES} waiting {delay_seconds:.1f}s "
                f"after {type(exc).__name__}: {exc}"
            )
            time.sleep(delay_seconds)
    assert last_error is not None
    raise last_error


def _restore_or_remove(backup_path: Path, target_path: Path) -> None:
    if backup_path.exists():
        _run_module_file_op_with_retry(
            "restore_copyfile",
            target_path,
            lambda: shutil.copyfile(backup_path, target_path),
        )
    elif target_path.exists():
        _run_module_file_op_with_retry(
            "restore_unlink",
            target_path,
            target_path.unlink,
        )


def _force_refresh_timestamp(path: Path) -> None:
    now = time.time()
    os.utime(path, (now, now))


def _clean_old_test_cases(module_dir: Path, mode_name: str, model_name: str) -> ModuleDriftReport:
    """
    Remove all old vktCustomCase_* files from test_case directory.
    Also resets CMakeLists.txt and group.cpp to initial state (no test cases).
    """
    test_case_dir = module_dir / "test_case"
    if not test_case_dir.exists():
        return diagnose_module_drift(module_dir, mode_name, model_name)

    pristine_dir = _ensure_pristine_module_files(module_dir, mode_name, model_name)
    report = _cleanup_module_drift_files(module_dir, mode_name, model_name)

    for relative_path in _baseline_module_files(mode_name, model_name):
        source_path = pristine_dir / relative_path
        target_path = module_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        _run_module_file_op_with_retry(
            "cleanup_restore_copyfile",
            target_path,
            lambda source_path=source_path, target_path=target_path: shutil.copyfile(source_path, target_path),
        )
    return report


def reset_module_generated_state(mode_name: str, model_name: str) -> None:
    """
    Reset one CTS module back to its baseline generated-file state.

    This removes any injected vktCustomCase_* files and restores the
    module-owned generated files that may reference them.
    """
    module_dir = _module_dir(mode_name, model_name)
    reset_module_generated_state_in_dir(module_dir, mode_name, model_name)


def reset_module_generated_state_in_dir(module_dir: Path, mode_name: str, model_name: str) -> None:
    """
    Reset one CTS module directory back to its baseline generated-file state.

    Use this when the caller already knows the exact module directory, such as
    grouped packages mounted on non-default CTS roots.
    """
    if not module_dir.exists():
        raise FileNotFoundError(f"module dir not found: {module_dir}")
    _clean_old_test_cases(module_dir, mode_name, model_name)


def _run_case_compile_once(
    mode_name: str,
    model_name: str,
    row_id: int,
    target: str | None = None,
    attempt_index: int | None = None,
    develop: bool = False,
) -> CompileResult:
    case_dir = build_case_dir(mode_name, model_name, row_id, develop=develop)
    effective_attempt = 1 if attempt_index is None else attempt_index
    compile_dir = case_dir / "compile" / f"attempt_{effective_attempt}"
    attempts_dir = case_dir / "attempts" / f"attempt_{effective_attempt}"
    compile_dir.mkdir(parents=True, exist_ok=True)
    attempts_dir.mkdir(parents=True, exist_ok=True)

    assembled_path = _resolve_assembled_path(case_dir, effective_attempt)
    classification_path = _resolve_classification_path(case_dir, effective_attempt)
    if not assembled_path.exists():
        raise FileNotFoundError(f"missing assembled candidate: {assembled_path}")
    if not classification_path.exists():
        raise FileNotFoundError(f"missing classification: {classification_path}")

    module_dir = _module_dir(mode_name, model_name)
    module_cmake = _module_cmake_path(mode_name, model_name)
    module_group_cpp = _module_group_cpp_path(mode_name, model_name)
    module_smoke_cpp = _module_smoke_cpp_path(mode_name, model_name)
    module_test_case_dir = module_dir / "test_case"
    if not module_dir.exists():
        raise FileNotFoundError(f"module dir not found: {module_dir}")

    # Clean all old test case files and reset CMakeLists.txt/group.cpp before injecting new one
    print(
        f"[CompileDiag] row_id={row_id} attempt={effective_attempt} "
        f"module={module_dir.name} stage=reset_module_start"
    )
    _clean_old_test_cases(module_dir, mode_name, model_name)
    print(
        f"[CompileDiag] row_id={row_id} attempt={effective_attempt} "
        f"module={module_dir.name} stage=reset_module_done"
    )

    classification = _load_json(classification_path)
    assembled_candidate = assembled_path.read_text(encoding="utf-8")
    basename = _build_generated_basename(mode_name, model_name, row_id)
    populate_symbol = _build_populate_symbol(mode_name, model_name, row_id)
    namespace_name = _build_generated_namespace(mode_name, model_name)

    generated_cpp_relpath = f"test_case/{basename}.cpp"
    generated_hpp_relpath = f"test_case/{basename}.hpp"
    generated_cpp_path = module_test_case_dir / f"{basename}.cpp"
    generated_hpp_path = module_test_case_dir / f"{basename}.hpp"

    compile_unit = _render_generated_compile_unit(
        assembled_candidate=assembled_candidate,
        register_name=classification["register_name"],
        case_name=classification["case_name"],
        namespace_name=namespace_name,
        populate_symbol=populate_symbol,
    )
    header_text = _render_generated_header(basename, namespace_name, populate_symbol)
    group_cpp = _render_group_cpp(
        mode_name,
        model_name,
        generated_hpp_relpath,
        populate_symbol,
        str(classification["case_name"]),
    )
    smoke_cpp = None
    if mode_name == "only-fix":
        smoke_cpp = _render_smoke_cpp(mode_name, model_name, None, None)
    cmake_text = _render_cmake(mode_name, model_name, generated_cpp_relpath)

    injected_cpp_path = attempts_dir / f"{basename}.cpp"
    injected_hpp_path = attempts_dir / f"{basename}.hpp"
    injected_group_cpp_path = attempts_dir / module_group_cpp.name
    injected_smoke_cpp_path = attempts_dir / module_smoke_cpp.name
    injected_cmake_path = attempts_dir / "CMakeLists.txt"
    print(
        f"[CompileDiag] row_id={row_id} attempt={effective_attempt} "
        f"module={module_dir.name} stage=write_attempt_artifacts_start"
    )
    injected_cpp_path.write_text(compile_unit, encoding="utf-8")
    injected_hpp_path.write_text(header_text, encoding="utf-8")
    injected_group_cpp_path.write_text(group_cpp, encoding="utf-8")
    if smoke_cpp is not None:
        injected_smoke_cpp_path.write_text(smoke_cpp, encoding="utf-8")
    injected_cmake_path.write_text(cmake_text, encoding="utf-8")
    print(
        f"[CompileDiag] row_id={row_id} attempt={effective_attempt} "
        f"module={module_dir.name} stage=write_attempt_artifacts_done"
    )

    generated_cpp_backup = attempts_dir / f"{basename}.cpp.bak"
    generated_hpp_backup = attempts_dir / f"{basename}.hpp.bak"
    group_cpp_backup = attempts_dir / f"{module_group_cpp.name}.bak"
    smoke_cpp_backup = attempts_dir / f"{module_smoke_cpp.name}.bak"
    cmake_backup = attempts_dir / "CMakeLists.txt.bak"

    if generated_cpp_path.exists():
        _backup_file(generated_cpp_path, generated_cpp_backup)
    if generated_hpp_path.exists():
        _backup_file(generated_hpp_path, generated_hpp_backup)
    if module_group_cpp.exists():
        _backup_file(module_group_cpp, group_cpp_backup)
    if module_smoke_cpp.exists():
        _backup_file(module_smoke_cpp, smoke_cpp_backup)
    if module_cmake.exists():
        _backup_file(module_cmake, cmake_backup)

    stdout_path = compile_dir / "build_stdout.txt"
    stderr_path = compile_dir / "build_stderr.txt"
    resolved_target = target or _module_target(mode_name, model_name)
    cts_root = resolve_cts_root(mode_name, model_name)
    cts_build_dir = resolve_cts_build_dir(mode_name, model_name)
    lock_scope = build_lock_scope_for_cts_root(cts_root)
    command = ["cmake", "--build", str(cts_build_dir), "--target", resolved_target]

    started_at = time.time()
    restored = False
    returncode = -1
    runtime_summary_path: Path | None = None
    _write_compile_runtime_status(
        compile_dir,
        row_id=row_id,
        attempt_index=effective_attempt,
        module_name=module_dir.name,
        target=resolved_target,
        command=command,
        stage="compile_initialized",
        started_at=started_at,
    )
    with acquire_build_lock(lock_scope, f"compile:{resolved_target}:row_{row_id}:attempt_{effective_attempt}"):
        try:
            print(
                f"[CompileDiag] row_id={row_id} attempt={effective_attempt} "
                f"module={module_dir.name} stage=inject_module_files_start"
            )
            _write_compile_runtime_status(
                compile_dir,
                row_id=row_id,
                attempt_index=effective_attempt,
                module_name=module_dir.name,
                target=resolved_target,
                command=command,
                stage="inject_module_files_start",
                started_at=started_at,
            )
            generated_cpp_path.write_text(compile_unit, encoding="utf-8")
            generated_hpp_path.write_text(header_text, encoding="utf-8")
            module_group_cpp.write_text(group_cpp, encoding="utf-8")
            if smoke_cpp is not None:
                module_smoke_cpp.write_text(smoke_cpp, encoding="utf-8")
            module_cmake.write_text(cmake_text, encoding="utf-8")
            print(
                f"[CompileDiag] row_id={row_id} attempt={effective_attempt} "
                f"module={module_dir.name} stage=inject_module_files_done"
            )
            _write_compile_runtime_status(
                compile_dir,
                row_id=row_id,
                attempt_index=effective_attempt,
                module_name=module_dir.name,
                target=resolved_target,
                command=command,
                stage="inject_module_files_done",
                started_at=started_at,
            )
            _force_refresh_timestamp(generated_cpp_path)
            _force_refresh_timestamp(generated_hpp_path)
            _force_refresh_timestamp(module_group_cpp)
            if smoke_cpp is not None:
                _force_refresh_timestamp(module_smoke_cpp)
            _force_refresh_timestamp(module_cmake)
            print(
                f"[CompileDiag] row_id={row_id} attempt={effective_attempt} "
                f"module={module_dir.name} stage=cmake_build_start target={resolved_target}"
            )
            _write_compile_runtime_status(
                compile_dir,
                row_id=row_id,
                attempt_index=effective_attempt,
                module_name=module_dir.name,
                target=resolved_target,
                command=command,
                stage="cmake_build_start",
                started_at=started_at,
            )
            returncode, stdout_text, stderr_text = _run_cmake_build_with_backoff(
                command,
                cts_root,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
            print(
                f"[CompileDiag] row_id={row_id} attempt={effective_attempt} "
                f"module={module_dir.name} stage=cmake_build_done returncode={returncode}"
            )
            _write_compile_runtime_status(
                compile_dir,
                row_id=row_id,
                attempt_index=effective_attempt,
                module_name=module_dir.name,
                target=resolved_target,
                command=command,
                stage="cmake_build_done",
                started_at=started_at,
                extra={"returncode": returncode},
            )
            if returncode == 0 and resolve_run_after_compile():
                print(
                    f"[Runtime] row_id={row_id} attempt={effective_attempt} "
                    f"module={module_dir.name} stage=runtime_start"
                )
                runtime_summary_path = _run_runtime_after_successful_compile(
                    mode_name=mode_name,
                    model_name=model_name,
                    row_id=row_id,
                    compile_dir=compile_dir,
                    cts_root=cts_root,
                    cts_build_dir=cts_build_dir,
                )
                print(
                    f"[Runtime] row_id={row_id} attempt={effective_attempt} "
                    f"summary={runtime_summary_path}"
                )
        except Exception as exc:
            print(
                f"[CompileDiag] row_id={row_id} attempt={effective_attempt} "
                f"module={module_dir.name} stage=compile_exception "
                f"type={type(exc).__name__} error={exc}"
            )
            _write_compile_runtime_status(
                compile_dir,
                row_id=row_id,
                attempt_index=effective_attempt,
                module_name=module_dir.name,
                target=resolved_target,
                command=command,
                stage="compile_exception",
                started_at=started_at,
                extra={"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise
        finally:
            print(
                f"[CompileDiag] row_id={row_id} attempt={effective_attempt} "
                f"module={module_dir.name} stage=restore_module_files_start"
            )
            _write_compile_runtime_status(
                compile_dir,
                row_id=row_id,
                attempt_index=effective_attempt,
                module_name=module_dir.name,
                target=resolved_target,
                command=command,
                stage="restore_module_files_start",
                started_at=started_at,
                extra={"returncode": returncode},
            )
            _restore_or_remove(generated_cpp_backup, generated_cpp_path)
            _restore_or_remove(generated_hpp_backup, generated_hpp_path)
            _restore_or_remove(group_cpp_backup, module_group_cpp)
            _restore_or_remove(smoke_cpp_backup, module_smoke_cpp)
            _restore_or_remove(cmake_backup, module_cmake)
            restored = True
            if runtime_summary_path is not None and runtime_summary_path.exists():
                try:
                    runtime_payload = json.loads(runtime_summary_path.read_text(encoding="utf-8"))
                    runtime_payload["source_restored"] = True
                    runtime_payload["source_restored_at"] = time.strftime(
                        "%Y-%m-%dT%H:%M:%S", time.localtime()
                    )
                    runtime_summary_path.write_text(
                        json.dumps(runtime_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception as exc:
                    print(f"[Runtime] unable to annotate restoration evidence: {exc}")
            print(
                f"[CompileDiag] row_id={row_id} attempt={effective_attempt} "
                f"module={module_dir.name} stage=restore_module_files_done"
            )
            _write_compile_runtime_status(
                compile_dir,
                row_id=row_id,
                attempt_index=effective_attempt,
                module_name=module_dir.name,
                target=resolved_target,
                command=command,
                stage="restore_module_files_done",
                started_at=started_at,
                extra={"returncode": returncode, "restored": restored},
            )

    duration_seconds = time.time() - started_at
    return CompileResult(
        returncode=returncode,
        command=command,
        target=resolved_target,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        injected_cpp_path=str(injected_cpp_path),
        restored=restored,
        duration_seconds=duration_seconds,
        attempt_index=effective_attempt,
        runtime_summary_path=(str(runtime_summary_path)
                              if runtime_summary_path is not None else None),
    )


def run_case_compile(
    mode_name: str,
    model_name: str,
    row_id: int,
    target: str | None = None,
    attempt_index: int | None = None,
    develop: bool = False,
) -> CompileResult:
    last_error: Exception | None = None
    effective_attempt = 1 if attempt_index is None else attempt_index

    for recovery_index in range(1, COMPILE_RECOVERY_MAX_RETRIES + 1):
        try:
            if recovery_index > 1:
                print(
                    f"[Compile Retry] row_id={row_id} attempt={effective_attempt} "
                    f"retry={recovery_index}/{COMPILE_RECOVERY_MAX_RETRIES}"
                )
            return _run_case_compile_once(
                mode_name=mode_name,
                model_name=model_name,
                row_id=row_id,
                target=target,
                attempt_index=attempt_index,
                develop=develop,
            )
        except Exception as exc:
            last_error = exc
            print(
                f"[Compile Retry] row_id={row_id} attempt={effective_attempt} "
                f"caught {type(exc).__name__}: {exc}"
            )
            if recovery_index >= COMPILE_RECOVERY_MAX_RETRIES:
                raise
            print(
                f"[Compile Retry] row_id={row_id} attempt={effective_attempt} "
                f"resetting module state before retry"
            )
            reset_module_generated_state(mode_name, model_name)
            delay_seconds = random.uniform(COMPILE_RECOVERY_WAIT_MIN_SECONDS, COMPILE_RECOVERY_WAIT_MAX_SECONDS)
            print(
                f"[Compile Retry] row_id={row_id} attempt={effective_attempt} "
                f"waiting {delay_seconds:.1f}s before retry {recovery_index + 1}/{COMPILE_RECOVERY_MAX_RETRIES}"
            )
            time.sleep(delay_seconds)

    assert last_error is not None
    raise last_error


# ========================================================================
# 源自 error_summarizer.py —— 编译日志摘要
# ========================================================================

ERROR_LINE_PATTERNS = (
    # 原先写死 `error C\d+:`, 于是只认编译器诊断 —— MSBuild 的 `error MSB3073:`/
    # `error MSB4181:` 和链接器的 `error LNK2019:` 一条都匹配不上, 这些行里恰好是
    # 「构建为什么整体失败」的唯一线索。改成任意 `error <字母码><数字>:`。
    # 每条都去掉了原来的 `.+` 头尾包裹。这些模式都用 search, 头尾的 `.+` 不改变
    # 「命中/不命中」, 只是让不命中的长行退化成 O(n²) 回溯 —— CMake 那几行几千字符的
    # include 路径清单正好触发它, 全树扫一遍要按小时计。
    re.compile(r"\berror\s+[A-Za-z]{1,6}\d+\s*:"),
    re.compile(r"\berror:\b"),
    re.compile(r"\bfatal error\b", re.IGNORECASE),
    re.compile(r"^\s*CMake Error\b", re.IGNORECASE),
    re.compile(r"\bMSBUILD\s*:\s*error\b", re.IGNORECASE),
    re.compile(r"\bninja:\s*error:", re.IGNORECASE),
    re.compile(r"\bBuild files cannot be regenerated correctly\b", re.IGNORECASE),
    re.compile(r"\bNo SOURCES given to target\b", re.IGNORECASE),
    re.compile(r"\bCannot find source file:", re.IGNORECASE),
)

WARNING_LINE_PATTERNS = (
    re.compile(r"\bwarning\s+[A-Za-z]{1,6}\d+\s*:"),
    re.compile(r"\bwarning:\b"),
)

_DIAGNOSTIC_HINTS = ("error", "warning", "no sources given", "cannot find source file", "build files cannot be")

DETAIL_NOISE_PATTERNS = (
    # 「(编译源文件 ..\..\xxx.cpp)」/「(compiling source file ...)」纯属重复头行里的路径
    re.compile(r"^\s*\(\s*(?:编译源文件|compiling source file)\b", re.IGNORECASE),
)

DETAIL_LINE_RE = re.compile(r"^(?:\d+>)?[ \t]{2,}\S")

DETAIL_STOP_PATTERNS = (
    re.compile(r"\s->\s"),
    re.compile(r"\.(?:cpp|cxx|cc|c|vcxproj|lib|exe|dll|obj)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:\d+>)?\s*(?:已完成生成项目|Done Building Project)\b", re.IGNORECASE),
)

NOISE_PATTERNS = (
    re.compile(r"^\s*$"),
    re.compile(r"^\s*Microsoft \(R\) Build Engine", re.IGNORECASE),
    re.compile(r"^\s*Copyright \(C\)", re.IGNORECASE),
    re.compile(r"^\s*Build started", re.IGNORECASE),
    re.compile(r"^\s*Project ", re.IGNORECASE),
    re.compile(r"^\s*Done Building Project", re.IGNORECASE),
    re.compile(r"^\s*[0-9]+>\s*Generating Code\.\.\.", re.IGNORECASE),
    re.compile(r"^\s*[0-9]+>\s*Compiling\.\.\.", re.IGNORECASE),
    re.compile(r"^\s*[0-9]+>\s*Linking\.\.\.", re.IGNORECASE),
)

CASCADE_ERROR_CODES = {"C2641", "C2780", "C2783", "C2923"}

PRIMARY_ERROR_CODES = {"C2065", "C2143", "C2146", "C2653", "C2882", "C3861", "C4430"}

def summarize_compile_logs(stdout_path: str | Path, stderr_path: str | Path) -> dict:
    stdout_text = Path(stdout_path).read_text(encoding="utf-8", errors="replace")
    stderr_text = Path(stderr_path).read_text(encoding="utf-8", errors="replace")

    lines = _dedupe_preserve_order(
        _extract_signal_lines(stdout_text.splitlines()) + _extract_signal_lines(stderr_text.splitlines())
    )

    # 只看头行判 error/warning: 每条现在是「头行 + 缩进补充行」, 补充行里出现
    # 「<error type>」这类字样不该把一条 warning 提升成 error。
    errors = [entry for entry in lines if _matches_any(entry.partition("\n")[0], ERROR_LINE_PATTERNS)]

    # 只保留 errors, 且**从不截断**。原先这里是 errors[:50] —— 一个与提示词预算无关
    # 的硬截断, 写文件时就把错误丢掉了, 模型永远看不到第 50 条之后的报错(重载决议类
    # 错误 C2664/C2665/C2660/C2661 基本都排在 50 条之后)。现在全量带出, 条数由
    # len(errors) 表达, 不再单独记 error_count。
    #
    # warnings / top_errors / error_groups 一并从返回值里摘掉: warning 与编译失败无
    # 因果关系; top_errors 是 errors 的子集且脱敏后条目重复; error_groups 的聚类依赖
    # _extract_symbols, 而后者的引号正则匹配不到中文 MSVC 的全角引号, 产出不可靠。
    # 相关函数**保留在本文件里不删**, 便于对照实验随时回退。
    return {"errors": [_to_prompt_message(line) for line in errors]}

def write_summary(summary: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

def summary_error_count(summary: dict) -> int:
    """从摘要里取错误条数 —— 所有消费方都该走这里, 不要直接读字段。

    errors 已不再截断, 所以 len(errors) 就是真值, error_count 字段不再产出。
    存量实验输出树里的旧文件仍带 error_count, 这里做兼容回退 —— 否则旧 case 会被
    读成「0 个错误」, 也就是「编译成功」, 那是个静默的错误结论。
    """
    errors = summary.get("errors")
    if isinstance(errors, list):
        return len(errors)
    try:
        return int(summary.get("error_count", 0) or 0)
    except (TypeError, ValueError):
        return 0

def _extract_signal_lines(lines: list[str]) -> list[str]:
    """按「诊断头 + 其后缩进的补充行」成条抽取, 一条不漏也不截断。

    过去这里是逐行 filter, 只留下匹配 ERROR/WARNING 模式的头行。MSVC 的诊断是多行的:
    头行说「没有重载」, 紧跟的缩进行才说清有哪些候选、试的是什么参数列表、哪个参数
    转换不过去。丢掉缩进行等于只给模型结论不给依据, 它只能猜着改。
    """
    kept: list[str] = []
    index = 0
    total = len(lines)
    while index < total:
        line = lines[index].rstrip()
        index += 1
        if not _may_be_diagnostic(line):
            continue
        if _matches_any(line, NOISE_PATTERNS):
            continue
        if not _is_diagnostic_head(line):
            continue
        details, index = _collect_detail_lines(lines, index)
        kept.append("\n".join([line, *details]) if details else line)
    return kept

def _collect_detail_lines(lines: list[str], start: int) -> tuple[list[str], int]:
    """收集 start 处开始、属于上一条诊断的缩进补充行, 返回 (补充行, 新游标)。

    边界: 遇到下一条诊断头、或缩进不足的非空行就停。空行不收但也不停 —— MSVC 的
    构建日志是隔行输出的, 遇空即停会导致一条补充行都收不到。**不设条数上限**。
    """
    details: list[str] = []
    index = start
    while index < len(lines):
        line = lines[index].rstrip()
        if not line.strip():
            index += 1
            continue
        if _is_diagnostic_head(line):
            break
        if not DETAIL_LINE_RE.match(line) or _matches_any(line, DETAIL_STOP_PATTERNS):
            break
        index += 1
        if _matches_any(line, DETAIL_NOISE_PATTERNS):
            continue
        details.append(line)
    while details and not details[-1].strip():
        details.pop()
    return details, index

def _matches_any(text: str, patterns: tuple[re.Pattern, ...]) -> bool:
    return any(pattern.search(text) for pattern in patterns)

def _may_be_diagnostic(line: str) -> bool:
    """前置闸门, 见 _DIAGNOSTIC_HINTS。判否即可跳过全部正则。"""
    lowered = line.lower()
    return any(hint in lowered for hint in _DIAGNOSTIC_HINTS)

def _is_diagnostic_head(line: str) -> bool:
    if not _may_be_diagnostic(line):
        return False
    return _matches_any(line, ERROR_LINE_PATTERNS) or _matches_any(line, WARNING_LINE_PATTERNS)

def _dedupe_preserve_order(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    kept: list[str] = []
    for line in lines:
        normalized = re.sub(r"\s+", " ", line).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        kept.append(line)
    return kept

def _group_error_clusters(errors: list[str]) -> list[dict]:
    parsed = [_parse_error_line(error) for error in errors]
    groups: list[dict] = []

    for item in parsed:
        if item is None:
            continue

        current_group = groups[-1] if groups else None
        if current_group is None or not _belongs_to_group(current_group, item):
            groups.append(
                {
                    "file": item["file"],
                    "start_line": item["line"],
                    "end_line": item["line"],
                    "error_codes": [item["code"]],
                    "symbols": list(item["symbols"]),
                    "representative_error": item["raw"],
                    "errors": [item["raw"]],
                    "kind": _classify_error_kind(item["code"], item["message"]),
                }
            )
            continue

        current_group["end_line"] = item["line"]
        current_group["errors"].append(item["raw"])
        if item["code"] not in current_group["error_codes"]:
            current_group["error_codes"].append(item["code"])
        for symbol in item["symbols"]:
            if symbol not in current_group["symbols"]:
                current_group["symbols"].append(symbol)

        current_kind = current_group["kind"]
        next_kind = _classify_error_kind(item["code"], item["message"])
        if current_kind == "cascade" and next_kind != "cascade":
            current_group["kind"] = next_kind
            current_group["representative_error"] = item["raw"]

    summarized: list[dict] = []
    for group in groups:
        summarized.append(
            {
                "file": group["file"],
                "line_span": [group["start_line"], group["end_line"]],
                "kind": group["kind"],
                "error_codes": group["error_codes"],
                "symbols": group["symbols"][:8],
                "representative_error": group["representative_error"],
                "collapsed_count": len(group["errors"]),
            }
        )
    return summarized

def _sanitize_error_group(group: dict) -> dict:
    return {
        "kind": group["kind"],
        "error_codes": group["error_codes"],
        "symbols": group["symbols"][:8],
        "representative_error": _to_prompt_message(str(group["representative_error"])),
        "collapsed_count": group["collapsed_count"],
    }

def _parse_error_line(error: str) -> dict | None:
    match = re.search(
        r"(?P<file>[A-Za-z]:\\[^()]+)\((?P<line>\d+),(?P<column>\d+)\): error (?P<code>[A-Z]+\d+): (?P<message>.+)",
        error,
    )
    if not match:
        return None

    message = match.group("message").strip()
    message = re.sub(r"\s+\[[^\]]+\]\s*$", "", message)
    return {
        "raw": error,
        "file": match.group("file"),
        "line": int(match.group("line")),
        "column": int(match.group("column")),
        "code": match.group("code"),
        "message": message,
        "symbols": _extract_symbols(message),
    }

def _sanitize_log_line(line: str) -> str:
    return re.sub(
        r"(?P<path>[A-Za-z]:\\[^():]+)",
        lambda match: _sanitize_path(match.group("path")),
        line,
    )

def _sanitize_path(path: str) -> str:
    normalized = path.replace("/", "\\")
    lowered = normalized.lower()
    anchor_markers = (
        "external\\vulkancts\\",
        "framework\\",
        "modules\\",
        "executor\\",
        "build\\",
        "2_baseline\\",
        "01_analysis\\",
    )
    for marker in anchor_markers:
        index = lowered.find(marker)
        if index != -1:
            return normalized[index:]

    parts = [part for part in normalized.split("\\") if part]
    if len(parts) <= 4:
        return normalized
    return "\\".join(parts[-4:])

def _to_prompt_message(entry: str) -> str:
    """整条诊断(头行 + 缩进补充行)脱敏。

    逐行处理是必须的: 下面几个 `^...` 规则里的 `[^:]+` 会跨过换行符, 一旦把多行诊断
    当成一个字符串喂进去, 它会从头行一路吃到补充行中间, 把内容截掉。
    """
    head, _, rest = entry.partition("\n")
    lines = [_to_prompt_head(head)]
    for detail in rest.splitlines():
        sanitized = _sanitize_log_line(detail).rstrip()
        if sanitized.strip():
            lines.append(sanitized)
    return "\n".join(lines).strip()

def _to_prompt_head(line: str) -> str:
    sanitized = _sanitize_log_line(line).strip()
    sanitized = re.sub(r"^\d+>\s*", "", sanitized)
    sanitized = re.sub(r"^[^:]+?\(\d+,\d+\):\s*", "", sanitized)
    sanitized = re.sub(r"^[^:]+:\s*(?=(?:fatal\s+)?error\b|warning\b)", "", sanitized, count=1, flags=re.IGNORECASE)
    return sanitized.strip()

def _extract_symbols(message: str) -> list[str]:
    symbols: list[str] = []
    for quoted in re.findall(r'"([^"]+)"', message):
        token = quoted.strip()
        if token and token not in symbols:
            symbols.append(token)

    if not symbols:
        for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_:]*\b", message):
            if token.startswith("error") or token in {"int", "void"}:
                continue
            if token not in symbols:
                symbols.append(token)
            if len(symbols) >= 4:
                break
    return symbols

def _classify_error_kind(code: str, message: str) -> str:
    if code in CASCADE_ERROR_CODES:
        return "cascade"
    if code in PRIMARY_ERROR_CODES:
        return "primary"

    lowered = message.lower()
    if "missing" in lowered or "syntax error" in lowered or "undeclared" in lowered:
        return "primary"
    return "secondary"

def _belongs_to_group(group: dict, item: dict) -> bool:
    if item["file"] != group["file"]:
        return False
    if item["line"] - group["end_line"] > 3:
        return False

    same_symbol = bool(set(group["symbols"]) & set(item["symbols"]))
    current_kind = group["kind"]
    next_kind = _classify_error_kind(item["code"], item["message"])

    if same_symbol:
        return True
    if current_kind == "primary" and next_kind in {"primary", "cascade"}:
        return True
    if current_kind == "secondary" and next_kind == "cascade":
        return True
    return False
