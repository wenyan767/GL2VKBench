"""可选：运行期验证 —— 把编译通过的用例真的跑起来。

编译通过不等于用例正确：编译只进静态库、不链接，声明而未定义也能 rc==0。
既有产物里，编译通过的候选含无条件 pass("Pass") 的比例按模型为 57.4%~90.7%
（gemini 632 例中 573 例、gpt 738 例中 633 例、claude 223 例中 128 例），编译率高的
模型空壳占比也最高。要拿运行结论，必须真的挂进 CTS 树、构建 deqp-vk、执行、读 qpa。

    python runtime.py --stage plan    --cts-root <PATH>
    python runtime.py --stage mount   --cts-root <PATH>
    python runtime.py --stage build   --cts-root <PATH>
    python runtime.py --stage execute --cts-root <PATH>
    python runtime.py --restore
    python runtime.py --verify-clean  --cts-root <PATH>

这一步会往 CTS 树里写文件。写入前记台账（runs/_runtime/state/mount_manifest.json），
结束后按台账还原并校验树干净。台账没还原完不允许再次挂载 —— 残留挂载实测会让
后续批次持续报 refusing to inject drifted，几分钟烧掉 61 个候选。

与原实现的三处差异：

  1. 候选来源从 stage2 sqlite 换成扫盘。判据用 error_summary.json 的 errors
     是否为空：与 compile_record.json 同时存在的 4274 份结论完全一致、零分歧，
     另有 12572 份产物只有 error_summary.json（其中 1257 份判通过）。用前者能
     多覆盖近三倍产物，也不再依赖那张表。
  2. 删掉 check_slot —— 它读 stage2 的并发槽位表，本包没有那套调度。
  3. CTS 根必须 --cts-root 显式传入，没有默认值。

前置：Vulkan SDK（VULKAN_SDK 环境变量）、可用 GPU、已构建的 CTS 树。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import core
import cts as cts_mod
import methods
import paths

# 这些路径必须跟随 --output-root；不能在 import 时永远锚到公开包内 runs/。
STATE_DIR: Path
BACKUP_DIR: Path
MOUNT_MANIFEST: Path
ALIVE_PLAN: Path
LOG_DIR: Path
DIAG_DIR: Path
RUN_DATA_ROOT: Path


def configure_artifact_root(output_root: Path | None = None) -> None:
    global STATE_DIR, BACKUP_DIR, MOUNT_MANIFEST, ALIVE_PLAN
    global LOG_DIR, DIAG_DIR, RUN_DATA_ROOT
    root = paths.runs_root(output_root) if output_root is not None else paths.runs_root()
    runtime_root = root / "_runtime"
    STATE_DIR = runtime_root / "state"
    BACKUP_DIR = STATE_DIR / "backup"
    MOUNT_MANIFEST = STATE_DIR / "mount_manifest.json"
    ALIVE_PLAN = STATE_DIR / "alive_plan.json"
    LOG_DIR = runtime_root / "logs"
    DIAG_DIR = runtime_root / "diagnostics"
    RUN_DATA_ROOT = runtime_root / "data"


configure_artifact_root()

MODELS = ("claude", "deepseek", "gemini", "glm", "gpt")

# 从注册表派生，不写字面量：改名时这里跟着变，verify_tree_clean 才不会漏检
# 某条线的残留（漏检等于把残留留在树上，后续批次会持续报 drifted）。
MODES = tuple(m.slug for m in methods.METHODS.values())
SEED_VARIANTS = tuple(methods.SEED_KINDS)
VARIANTS = SEED_VARIANTS + MODES


def normalize_variant(value: str) -> str:
    """Normalize paper-facing and legacy method names to storage slugs."""
    if value in SEED_VARIANTS:
        return value
    try:
        return methods.get_method(value).slug
    except ValueError as exc:
        raise ValueError(
            f"Unknown runtime method {value!r}; available: "
            f"{', '.join(methods.METHOD_INPUT_NAMES)}"
        ) from exc

# 从编译输出里解析肇事 case 的 (package_token, row_id)。两个分支分别对应
# 「文件名在前、error 在后」与「error 在前、文件名在后」两种 MSVC 排布。
FAILED_ROW_RE = re.compile(
    r"vktCustomCase_([a-z_]+?)_(\d+)\.(?:cpp|obj)\b[^\n\r]*?"
    r"(?:\(\d+[^)]*\)\s*:\s*(?:fatal\s+)?error|\)[^\n\r]*?(?:fatal\s+)?error)"
    r"|(?:fatal\s+)?error[^\n\r]*?vktCustomCase_([a-z_]+?)_(\d+)\.(?:cpp|obj)\b",
    re.IGNORECASE)

# api_dump 层与原实现一致，保持运行证据格式可比。
VULKAN_SDK_ROOT = Path(os.environ.get("VULKAN_SDK", ""))
VK_APIDUMP_LAYER_NAME = "VK_LAYER_LUNARG_api_dump"

# 判据是 qpa 里 <Result StatusCode="..."> 的值。
# true = dEQP 给出了正常结论；false = 拿不到正常结论。
TRUE_VERDICTS = frozenset({
    "Pass", "Fail", "NotSupported", "QualityWarning", "CompatibilityWarning",
})
FALSE_VERDICTS = frozenset({
    "InternalError", "Crash", "Timeout", "ResourceError", "DeviceLost", "Waiver",
})
STATUS_CODE_RE = re.compile(r'<Result\s+StatusCode="([^"]+)"')


def require_cts_root(value: str | None) -> Path:
    """CTS 根必须显式给。没有默认值，也不从环境变量猜。"""
    if not value:
        raise SystemExit(
            "--cts-root <VK-GL-CTS 检出路径> 是必需的。\n"
            "本包不猜测 CTS 位置：那些树可能正被其它流程占用，"
            "猜错会把代码注入正在使用的树。")
    root = Path(value).resolve()
    if not root.is_dir():
        raise SystemExit(f"--cts-root 不存在或不是目录: {root}")
    return root


def compile_passed(attempt_dir: Path) -> bool | None:
    """判定某一轮编译是否通过。返回 None 表示这一轮没有判据。

    主判据是 error_summary.json 的 errors 是否为空。compile_record.json 存在时
    作交叉验证 —— 实测 4274 份同时存在的产物结论完全一致；若出现分歧以
    compile_record 为准并告警，因为那是编译器直接返回的 returncode。
    """
    summary_path = attempt_dir / "error_summary.json"
    verdict: bool | None = None
    if summary_path.is_file():
        try:
            errors = json.loads(summary_path.read_text(encoding="utf-8")).get("errors", [])
            verdict = (len(errors) == 0)
        except (OSError, ValueError):
            verdict = None

    record_path = attempt_dir / "compile_record.json"
    if record_path.is_file():
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            rc_pass = (record.get("returncode") == 0 or record.get("success") is True)
            if verdict is not None and verdict != rc_pass:
                print(f"  [WARN] 判据分歧 {attempt_dir}: "
                      f"error_summary={verdict} compile_record={rc_pass}，取后者")
            return rc_pass
        except (OSError, ValueError):
            pass
    return verdict


def parse_rows(spec: str | None) -> List[int] | None:
    if not spec:
        return None
    out: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def scan_compiled_candidates(
    modes: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    row_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """扫 runs/ 找编译通过的候选，取代原实现的 stage2 sqlite 查询。

    cpp/hpp 必须来自**真正编译成功的那一轮**：实测成功轮次分布是
    attempt_1=163 / attempt_2=852 / attempt_3=763（1778 个编译通过的候选，三条线 ×
    五个模型），无脑取首轮或末轮都是错的。
    这里取通过的最大轮次 —— 与原实现 resolve_success_attempt 同一口径。
    """
    wanted_variants = (
        [normalize_variant(item) for item in modes]
        if modes else list(VARIANTS)
    )
    wanted_models = list(models) if models else list(MODELS)
    out: List[Dict[str, Any]] = []

    for variant in wanted_variants:
        for model in wanted_models:
            if variant in SEED_VARIANTS:
                model_dir = paths.runs_root() / "compile" / variant / model
                module_name = methods.seed_package_name(variant, model)
            else:
                method = methods.get_method(variant)
                model_dir = paths.runs_root() / method.slug / model
                module_name = methods.package_name(method.slug, model)
            if not model_dir.is_dir():
                continue
            for case_dir in sorted(model_dir.glob("case_*")):
                try:
                    row_id = int(case_dir.name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                if row_ids and row_id not in row_ids:
                    continue
                best: Optional[int] = None
                for attempt_dir in sorted(case_dir.glob("compile/attempt_*")):
                    try:
                        idx = int(attempt_dir.name.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                    if compile_passed(attempt_dir):
                        best = idx if best is None else max(best, idx)
                if best is None:
                    continue
                out.append({
                    "mode": variant,
                    "model": model,
                    "row_id": row_id,
                    "final_case_dir": str(case_dir),
                    "current_attempt": best,
                    "module_name": module_name,
                })
    out.sort(key=lambda e: (e["mode"], e["model"], e["row_id"]))
    return out


def _iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def _safe_token(value: str) -> str:
    return value.replace("-", "_")

def _vulkan_dir(cts_root: Path) -> Path:
    return cts_root / "external" / "vulkancts" / "modules" / "vulkan"

def _registry_path(cts_root: Path) -> Path:
    return _vulkan_dir(cts_root) / "vktTestPackage.cpp"

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")

def _decode(raw: bytes) -> str:
    for encoding in ("utf-8", "gb18030", "cp936", "utf-16le"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")

def _find_deqp_vk(cts_root: Path) -> Optional[Path]:
    base = cts_root / "build" / "external" / "vulkancts" / "modules" / "vulkan"
    for candidate in (base / "Debug" / "deqp-vk.exe",
                      base / "Release" / "deqp-vk.exe",
                      base / "deqp-vk.exe"):
        if candidate.is_file():
            return candidate
    return None

def _run(command: List[str], *, cwd: Path, timeout_seconds: int,
         env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    started = time.time()
    try:
        completed = subprocess.run(
            command, cwd=str(cwd), capture_output=True,
            timeout=timeout_seconds, check=False, env=env)
        stdout, stderr = completed.stdout, completed.stderr
        returncode, timed_out = completed.returncode, False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        returncode, timed_out = 124, True
    return {
        "command": command,
        "returncode": returncode,
        "stdout": _decode(stdout),
        "stderr": _decode(stderr),
        "timed_out": timed_out,
        "elapsed_seconds": round(time.time() - started, 3),
    }

def resolve_success_attempt(case_dir: Path) -> Optional[int]:
    """返回编译成功的那个 attempt 号；找不到返回 None。"""
    best: Optional[int] = None
    for record_path in sorted(case_dir.glob("compile/attempt_*/compile_record.json")):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if record.get("returncode") == 0 or record.get("success") is True:
            try:
                best = int(record_path.parent.name.split("_")[1])
            except (IndexError, ValueError):
                continue
    return best

def build_case_plan(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """把一个候选变成挂载计划条目，附带跳过原因（若不可用）。"""
    mode = str(candidate["mode"])
    model = str(candidate["model"])
    row_id = int(candidate["row_id"])
    entry: Dict[str, Any] = {
        "mode": mode, "model": model, "row_id": row_id,
        "case_dir": candidate.get("final_case_dir") or "",
        "skip_reason": None,
    }
    raw_dir = candidate.get("final_case_dir")
    if not raw_dir:
        entry["skip_reason"] = "final_case_dir 为空"
        return entry
    case_dir = Path(raw_dir)
    if not case_dir.is_dir():
        entry["skip_reason"] = "final_case_dir 不存在"
        return entry

    attempt = candidate.get("current_attempt")
    if attempt is None:
        attempt = resolve_success_attempt(case_dir)
    if attempt is None:
        entry["skip_reason"] = "找不到 returncode==0 的 compile_record.json"
        return entry
    attempt = int(attempt)
    entry["attempt"] = attempt

    attempt_dir = case_dir / "attempts" / f"attempt_{attempt}"
    cpp_matches = sorted(attempt_dir.glob(f"vktCustomCase_*_{row_id}.cpp"))
    paired = [
        (cpp, cpp.with_suffix(".hpp"))
        for cpp in cpp_matches
        if cpp.with_suffix(".hpp").is_file()
    ]
    if len(paired) != 1:
        entry["skip_reason"] = (
            f"attempt_{attempt} 的 cpp/hpp 配对数不是 1（实际 {len(paired)}）"
        )
        return entry
    src_cpp, src_hpp = paired[0]
    basename = src_cpp.stem
    header_text = _read_text(src_hpp)
    populate_matches = list(dict.fromkeys(re.findall(
        r"\b(populate_[A-Za-z0-9_]+)\s*\(", header_text
    )))
    if len(populate_matches) != 1:
        entry["skip_reason"] = (
            f"attempt_{attempt} 无法唯一解析 populate 符号（实际 {populate_matches}）"
        )
        return entry
    populate_symbol = populate_matches[0]

    runtime_case_path = case_dir / "meta" / "runtime_case.json"
    runtime_case: Dict[str, Any] = {}
    if runtime_case_path.is_file():
        try:
            runtime_case = json.loads(runtime_case_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            runtime_case = {}

    module_name = str(candidate.get("module_name") or "")
    if not module_name:
        entry["skip_reason"] = "候选缺 module_name"
        return entry
    class_prefix = "".join(
        part[:1].upper() + part[1:]
        for part in module_name.split("_")
        if part
    )
    # CTS 注册名由当前 package/module 契约决定。历史 metadata 可能使用
    # 连字符（例如 director-claude），但生成的包实际注册为下划线名称；
    # 历史值只留作溯源，不能拿来拼当前执行路径。
    source_root_group = str(runtime_case.get("root_group") or "")
    root_group = module_name
    classification_path = case_dir / "meta" / "classification.json"
    expected_case_name = f"case_{row_id}"
    if classification_path.is_file():
        try:
            classification = json.loads(classification_path.read_text(encoding="utf-8"))
            expected_case_name = str(
                classification.get("case_name") or expected_case_name
            )
        except (OSError, ValueError):
            pass
    entry.update({
        "basename": basename,
        "source_cpp": str(src_cpp),
        "source_hpp": str(src_hpp),
        "module_name": module_name,
        "target": f"deqp-vk-{module_name.replace('_', '-')}",
        "class_prefix": class_prefix,
        "root_group": root_group,
        "source_root_group": source_root_group or None,
        # 这里只保存预期叶子名；完整路径必须在 deqp-vk 链接成功后从 caselist 发现。
        "expected_case_name": expected_case_name,
        "case_path": None,
        "case_filter": f"dEQP-VK.{root_group}.*",
        "candidate_paths": list(runtime_case.get("candidate_paths") or []),
        # 挂载时将追加的三行
        "cmake_src_line": f"    test_case/{basename}.cpp",
        "include_line": f'#include "{basename}.hpp"',
        "populate_symbol": populate_symbol,
        "populate_line": f"    {populate_symbol}(group);",
    })
    return entry

class MountLedger:
    """记录"改了哪些文件"和"建了哪些文件"，落盘后即可脱离进程还原。

    stage2 靠 `finally` 还原，进程被杀就漏（本轮停机前刚验证过一次侥幸没漏）。
    stage3 要动 15 个包共 45+ 个文件，必须能在进程死后还原，所以台账先落盘再动手。
    """

    def __init__(self, cts_root: Path):
        self.cts_root = cts_root
        self.modified: Dict[str, str] = {}   # 原文件绝对路径 -> 备份文件名
        self.created: List[str] = []         # 由本轮创建的文件绝对路径
        self.meta: Dict[str, Any] = {}

    def backup(self, path: Path) -> None:
        """把一个"将被修改"的既有文件原文另存。同一文件只备份第一次。"""
        key = str(path)
        if key in self.modified:
            return
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        name = f"{digest}_{path.name}"
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, BACKUP_DIR / name)
        self.modified[key] = name

    def note_created(self, path: Path) -> None:
        self.created.append(str(path))

    def save(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        MOUNT_MANIFEST.write_text(json.dumps({
            "mounted_at": _iso_now(),
            "cts_root": str(self.cts_root),
            "backup_dir": str(BACKUP_DIR),
            "modified": self.modified,
            "created": self.created,
            **self.meta,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

def restore_from_manifest(*, quiet: bool = False) -> int:
    """按台账还原：改过的写回原文，建过的删掉。幂等。"""
    if not MOUNT_MANIFEST.is_file():
        if not quiet:
            print(f"  (没有挂载台账 {MOUNT_MANIFEST}，无需还原)")
        return 0
    payload = json.loads(_read_text(MOUNT_MANIFEST))
    modified: Dict[str, str] = payload.get("modified", {})
    created: List[str] = payload.get("created", [])
    backup_dir = Path(payload.get("backup_dir") or BACKUP_DIR)

    if not quiet:
        print(f"=== 还原（台账 {MOUNT_MANIFEST}）===")
        print(f"  挂载于 {payload.get('mounted_at')}   root={payload.get('cts_root')}")
        print(f"  待写回 = {len(modified)}   待删除 = {len(created)}")

    failures: List[str] = []
    removed = 0
    for raw in created:
        path = Path(raw)
        try:
            if path.is_file():
                path.unlink()
                removed += 1
        except OSError as exc:
            failures.append(f"删除失败 {path}: {exc}")

    written = 0
    for raw, name in modified.items():
        source = backup_dir / name
        target = Path(raw)
        if not source.is_file():
            failures.append(f"备份缺失，无法写回 {target}（缺 {source}）")
            continue
        try:
            shutil.copyfile(source, target)
            written += 1
        except OSError as exc:
            failures.append(f"写回失败 {target}: {exc}")

    if failures:
        print(f"  [FAIL] 还原有 {len(failures)} 处问题，台账保留以便重试:")
        for line in failures[:20]:
            print(f"    {line}")
        return 2

    # 台账消费掉，避免二次还原把后续正常修改覆盖回旧版本。
    done_dir = STATE_DIR / "restored"
    done_dir.mkdir(parents=True, exist_ok=True)
    stamp = _iso_now().replace(":", "").replace("-", "")
    shutil.move(str(MOUNT_MANIFEST), str(done_dir / f"mount_manifest_{stamp}.json"))
    if ALIVE_PLAN.is_file():
        ALIVE_PLAN.unlink()
    if not quiet:
        print(f"  [OK] 写回 {written} 个文件，删除 {removed} 个文件。")
        print(f"       台账归档到 {done_dir}")
    return 0

def verify_tree_clean(cts_root: Path) -> Tuple[bool, List[str]]:
    """确认树上没有 stage3 挂载残留。还原后调用。"""
    residue: List[str] = []
    vulkan_dir = _vulkan_dir(cts_root)
    for mode in VARIANTS:
        for model in MODELS:
            module_name = (
                methods.seed_package_name(mode, model)
                if mode in SEED_VARIANTS
                else methods.package_name(mode, model)
            )
            module_dir = vulkan_dir / module_name
            test_case_dir = module_dir / "test_case"
            if test_case_dir.is_dir():
                residue += [str(p) for p in test_case_dir.glob("vktCustomCase_*")]
            group_dir = module_dir / "group"
            if group_dir.is_dir():
                for path in group_dir.glob("*.cpp"):
                    if "populate_" in _read_text(path):
                        residue.append(f"{path} (含 populate_ 调用)")
            cmake_path = module_dir / "CMakeLists.txt"
            if cmake_path.is_file() and "vktCustomCase_" in _read_text(cmake_path):
                residue.append(f"{cmake_path} (含 vktCustomCase_ SRCS)")
    return (not residue), residue


def _patch_runtime_factory(
    tests_cpp_text: str,
    prefix: str,
    items: List[Dict[str, Any]],
) -> str:
    """Make the package factory directly own every mounted case reference.

    CTS packages are static libraries.  On MSVC, relying on a second archive
    object (group.cpp -> populate_case) can leave the leaf object out of the
    final deqp-vk link even though the package library itself built cleanly.
    The factory therefore calls each generated populate function directly.
    """
    include_lines = [
        f'#include "test_case/{item["basename"]}.hpp"'
        for item in items
    ]
    missing = [line for line in include_lines if line not in tests_cpp_text]
    if missing:
        anchor = '#include "vktTestGroupUtil.hpp"'
        if anchor not in tests_cpp_text:
            raise RuntimeError(f"tests.cpp 缺少 include 锚点: {anchor}")
        tests_cpp_text = tests_cpp_text.replace(
            anchor, anchor + "\n" + "\n".join(missing), 1
        )

    helper_name = f"create{prefix}RuntimeGroups"
    if f"void {helper_name}(tcu::TestCaseGroup* group)" not in tests_cpp_text:
        calls = "\n".join(
            f"    {item['populate_symbol']}(group);" for item in items
        )
        helper = (
            f"\nvoid {helper_name}(tcu::TestCaseGroup* group)\n{{\n"
            f"    create{prefix}Groups(group);\n{calls}\n}}\n"
        )
        declaration_anchor = f"tcu::TestCaseGroup* createTests(tcu::TestContext& testCtx, const std::string& name)"
        if declaration_anchor not in tests_cpp_text:
            raise RuntimeError("tests.cpp 缺少 createTests 声明锚点")
        tests_cpp_text = tests_cpp_text.replace(
            declaration_anchor,
            f"void {helper_name}(tcu::TestCaseGroup* group);\n\n"
            + declaration_anchor,
            1,
        )
        namespace_close = f"}} // namespace {items[0]['module_name']}"
        if namespace_close not in tests_cpp_text:
            raise RuntimeError(
                f"tests.cpp 缺少 namespace 结束锚点: {namespace_close}"
            )
        tests_cpp_text = tests_cpp_text.replace(
            namespace_close, helper + "\n" + namespace_close, 1
        )

    old = f"return createTestGroup(testCtx, name, create{prefix}Groups);"
    new = f"return createTestGroup(testCtx, name, {helper_name});"
    if old in tests_cpp_text:
        tests_cpp_text = tests_cpp_text.replace(old, new, 1)
    elif new not in tests_cpp_text:
        raise RuntimeError("tests.cpp 缺少 createTestGroup factory 锚点")
    return tests_cpp_text

def mount_cases(cts_root: Path, usable: List[Dict[str, Any]]) -> MountLedger:
    """把 N 个 case 挂进已由 ctspkg 完整注册的独立 package。

    每个 package 修改 CMakeLists.txt 和 vkt*Tests.cpp：
      CMakeLists.txt                   追加 N 行 `    test_case/{basename}.cpp`
      vkt*Tests.cpp                    直接引用 N 个 populate_X(group) 函数，
                                       确保静态库 leaf 对象进入最终链接。
    smoke 组保持原样不动 —— generated case 直接挂在 root group 下，路径是
    `dEQP-VK.{root_group}.case_{row}`，与 stage2 的 case_filter 一致。
    """
    if MOUNT_MANIFEST.is_file():
        raise RuntimeError(
            f"已存在未还原的挂载台账 {MOUNT_MANIFEST}。先跑 --restore，"
            f"否则这次的备份会覆盖上次的，两轮都还原不回去。")

    ledger = MountLedger(cts_root)
    vulkan_dir = _vulkan_dir(cts_root)
    by_module: Dict[str, List[Dict[str, Any]]] = {}
    for item in usable:
        by_module.setdefault(item["module_name"], []).append(item)

    # 先把台账占位落盘，之后每个包处理完刷新一次。中途被杀也有可还原的记录。
    ledger.meta = {"case_count": len(usable), "module_count": len(by_module),
                   "status": "mounting"}
    ledger.save()

    for module_name in sorted(by_module):
        group = by_module[module_name]
        head = group[0]
        module_dir = vulkan_dir / module_name
        test_case_dir = module_dir / "test_case"
        cmake_path = module_dir / "CMakeLists.txt"
        tests_cpp = module_dir / f"vkt{head['class_prefix']}Tests.cpp"
        group_cpp = module_dir / "group" / f"vkt{head['class_prefix']}Group.cpp"
        for required in (module_dir, test_case_dir, cmake_path, tests_cpp, group_cpp):
            if not required.exists():
                raise FileNotFoundError(f"package 结构不完整: {required}")
        group_text = _read_text(group_cpp)
        if "addTestGroup(" in group_text:
            raise RuntimeError(
                f"package root 已注册 smoke/group child，不能直接挂 executable leaf: {group_cpp}"
            )

        ledger.backup(cmake_path)
        ledger.backup(tests_cpp)
        cmake_text = _read_text(cmake_path)
        tests_text = _read_text(tests_cpp)

        # SRCS 追加锚点：既有的 SmokeTests.cpp 行。
        cmake_anchor = f"    test_case/vkt{head['class_prefix']}SmokeTests.cpp"
        if cmake_anchor not in cmake_text:
            raise RuntimeError(f"CMakeLists 缺 SRCS 锚点 {cmake_anchor}: {cmake_path}")
        src_lines = []
        for item in group:
            dst_cpp = test_case_dir / f"{item['basename']}.cpp"
            dst_hpp = test_case_dir / f"{item['basename']}.hpp"
            for dst in (dst_cpp, dst_hpp):
                if dst.exists():
                    raise FileExistsError(
                        f"目标已存在，拒绝覆盖: {dst}。树不干净，先跑 --restore。")
            shutil.copyfile(item["source_cpp"], dst_cpp)
            ledger.note_created(dst_cpp)
            shutil.copyfile(item["source_hpp"], dst_hpp)
            ledger.note_created(dst_hpp)
            src_lines.append(f"    test_case/{item['basename']}.cpp")

        cmake_text = cmake_text.replace(
            cmake_anchor, cmake_anchor + "\n" + "\n".join(src_lines), 1)
        tests_text = _patch_runtime_factory(
            tests_text, head["class_prefix"], group
        )

        _write_text(cmake_path, cmake_text)
        _write_text(tests_cpp, tests_text)
        ledger.save()
        print(f"  {module_name:24s} n={len(group):4d}  "
              f"SRCS+{len(src_lines)}  factory_refs+{len(group)}")

    # Package 的 root 注册属于 ctspkg 的持久集成职责；runtime 只验证，不再临时改
    # vktTestPackage.cpp，否则 restore 会把建包阶段的合法注册一起撤销。
    registry = _registry_path(cts_root)
    registered: List[str] = []
    for module_name in sorted(by_module):
        head = by_module[module_name][0]
        registry_text = _read_text(registry)
        registration = (
            f'addRootChild("{head["root_group"]}", m_caseListFilter, '
            f'{head["module_name"]}::createTests);'
        )
        if registration not in registry_text:
            raise RuntimeError(
                f"Package 尚未完整注册到 vktTestPackage.cpp: {registration}"
            )
        registered.append(head["root_group"])
    ledger.meta["registered_root_groups"] = registered
    ledger.meta["status"] = "mounted"
    ledger.save()
    print(f"\n  root group 已注册 {len(registered)} 个 -> {registry.name}")
    print(f"  台账 -> {MOUNT_MANIFEST}")
    return ledger

def unmount_rows(cts_root: Path, items: List[Dict[str, Any]]) -> None:
    """从树上剔除若干 case（编译失败归因后使用）。不动台账里的备份。"""
    vulkan_dir = _vulkan_dir(cts_root)
    by_module: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        by_module.setdefault(item["module_name"], []).append(item)
    for module_name, group in sorted(by_module.items()):
        head = group[0]
        module_dir = vulkan_dir / module_name
        cmake_path = module_dir / "CMakeLists.txt"
        tests_cpp = module_dir / f"vkt{head['class_prefix']}Tests.cpp"
        cmake_text = _read_text(cmake_path)
        tests_text = _read_text(tests_cpp)
        for item in group:
            basename = item["basename"]
            populate = f"{item['populate_symbol']}(group);"
            cmake_text = cmake_text.replace(f"    test_case/{basename}.cpp\n", "")
            tests_text = tests_text.replace(f'#include "test_case/{basename}.hpp"\n', "")
            tests_text = tests_text.replace(f"    {populate}\n", "")
            for suffix in (".cpp", ".hpp"):
                path = module_dir / "test_case" / f"{basename}{suffix}"
                if path.is_file():
                    path.unlink()
        _write_text(cmake_path, cmake_text)
        _write_text(tests_cpp, tests_text)

def verify_baseline(cts_root: Path, timeout_seconds: int) -> int:
    """在当前（未挂载）状态下编一次 deqp-vk。

    必须先确认干净树可编：否则批量挂载后一旦编译失败，无法区分是某个 case 的问题
    还是这棵树本来就编不过。CMake 里另有 `director_glm_5/6/7` 三个历史遗留 package，
    它们不属于 stage2 的 15 个组合但会一起被编进去。
    """
    build_dir = cts_root / "build"
    print(f"=== 基线编译验证（不挂载任何 case）===")
    print(f"  root  = {cts_root}")
    print(f"  build = {build_dir}")
    if not build_dir.is_dir():
        print("  [FAIL] build 目录不存在，需要先 cmake 配置")
        return 2
    cache = build_dir / "CMakeCache.txt"
    print(f"  CMakeCache.txt 存在 = {cache.is_file()}")

    command = ["cmake", "--build", str(build_dir), "--target", "deqp-vk",
               "--config", "Debug"]
    print(f"\n  执行: {' '.join(command)}")
    print(f"  超时: {timeout_seconds}s\n")
    started = time.time()
    try:
        completed = subprocess.run(
            command, cwd=str(cts_root), capture_output=True,
            timeout=timeout_seconds, check=False)
    except FileNotFoundError:
        print("  [FAIL] 找不到 cmake。cmake 只在 PowerShell 的 PATH 上，"
              "不要从 bash 启动本命令。")
        return 2
    except subprocess.TimeoutExpired:
        print(f"  [FAIL] 编译超时（>{timeout_seconds}s）")
        return 2
    elapsed = time.time() - started

    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    print(f"  returncode = {completed.returncode}   耗时 = {elapsed:.1f}s")

    log_dir = LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = _iso_now().replace(":", "").replace("-", "")
    (log_dir / f"baseline_build_{stamp}_stdout.txt").write_text(
        stdout, encoding="utf-8")
    (log_dir / f"baseline_build_{stamp}_stderr.txt").write_text(
        stderr, encoding="utf-8")
    print(f"  日志 -> {log_dir}")

    if completed.returncode != 0:
        print("\n  [FAIL] 基线编译失败。挂载前必须先修好，否则挂载后的失败无法归因。")
        tail = [line for line in stdout.splitlines()
                if "error" in line.lower()][:15]
        for line in tail:
            print(f"    {line.strip()[:160]}")
        return 2

    executable = None
    for pattern in ("build/external/vulkancts/modules/vulkan/Debug/deqp-vk.exe",
                    "build/external/vulkancts/modules/vulkan/deqp-vk.exe"):
        candidate = cts_root / pattern
        if candidate.is_file():
            executable = candidate
            break
    print(f"\n  [OK] 基线可编译。")
    print(f"  deqp-vk = {executable or '(未找到可执行文件，需确认输出路径)'}")
    return 0

def parse_failed_cases(build_text: str) -> set:
    """从编译输出里解析肇事 case 的 (package_token, row_id)。

    沿用 3_Run/src/3_batch_mount_only_fix_compiled_cases.py:753 的思路，但那份把
    model 名写死在正则里；这里 package token 也是捕获组，因为 15 个包一起编。
    """
    found = set()
    for match in FAILED_ROW_RE.finditer(build_text):
        token = match.group(1) or match.group(3)
        row = match.group(2) or match.group(4)
        if token and row:
            found.add((token, int(row)))
    return found

def build_once(cts_root: Path, timeout_seconds: int, tag: str) -> Dict[str, Any]:
    build_dir = cts_root / "build"
    result = _run(["cmake", "--build", str(build_dir), "--target", "deqp-vk",
                   "--config", "Debug"],
                  cwd=cts_root, timeout_seconds=timeout_seconds)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _iso_now().replace(":", "").replace("-", "")
    for stream in ("stdout", "stderr"):
        (LOG_DIR / f"build_{tag}_{stamp}_{stream}.txt").write_text(
            result[stream], encoding="utf-8")
    return result

def command_build(args: argparse.Namespace,
                  usable: Optional[List[Dict[str, Any]]] = None) -> Tuple[int, List[Dict[str, Any]]]:
    """编一次；失败就解析肇事 row、剔除、重编。返回 (rc, 仍在树上的 case)。

    "一次编译 = 一次全赌"：335 个 case 从未共存过，任一个有符号重定义、宏冲突或
    `using namespace` 污染，整个链接就失败。所以剔除循环是必需的，不是可选优化。
    """
    if usable is None:
        usable = load_plan_or_die(args)
    alive = list(usable)
    ejected: List[Dict[str, Any]] = []

    for attempt in range(1, args.max_build_rounds + 1):
        print(f"\n=== 编译第 {attempt} 轮   树上 case = {len(alive)} ===")
        result = build_once(args.cts_root, args.build_timeout_seconds, f"r{attempt}")
        print(f"  returncode = {result['returncode']}   "
              f"耗时 = {result['elapsed_seconds']:.1f}s"
              + ("   [超时]" if result["timed_out"] else ""))
        if result["returncode"] == 0:
            if not alive:
                print("  [FAIL] 编译虽然通过，但所有候选都已被剔除，禁止把空执行当成成功。")
                record_ejected(ejected)
                return 2, alive
            print(f"  [OK] 编译通过。剔除累计 = {len(ejected)}")
            record_ejected(ejected)
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            ALIVE_PLAN.write_text(
                json.dumps(alive, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return 0, alive

        merged = result["stdout"] + "\n" + result["stderr"]
        failed = parse_failed_cases(merged)
        if not failed:
            print("  [FAIL] 编译失败但无法归因到具体 case —— 可能是链接期冲突或树的问题。")
            for line in [l.strip() for l in merged.splitlines()
                         if "error" in l.lower()][:15]:
                print(f"    {line[:170]}")
            print("  不做盲目剔除。人工看日志后再决定。")
            return 2, alive

        victims = [item for item in alive
                   if (_safe_token(item["mode"]) + "_" + item["model"],
                       item["row_id"]) in failed]
        if not victims:
            print(f"  [FAIL] 解析出 {len(failed)} 个肇事标识，但没匹配上树上的 case。")
            print(f"    解析结果样例: {sorted(failed)[:10]}")
            return 2, alive

        print(f"  归因到 {len(victims)} 个 case，剔除后重编:")
        for item in victims[:20]:
            print(f"    {item['mode']}/{item['model']}/row={item['row_id']}")
        if len(victims) > 20:
            print(f"    ... 其余 {len(victims) - 20} 个")
        unmount_rows(args.cts_root, victims)
        for item in victims:
            item["eject_reason"] = "link_failed_in_batch"
            item["eject_round"] = attempt
            item["eject_diagnostics"] = _eject_diagnostics(merged, item)
        ejected += victims
        alive = [item for item in alive if item not in victims]
        record_ejected(ejected)

    print(f"\n  [FAIL] {args.max_build_rounds} 轮仍未编过。剔除累计 = {len(ejected)}")
    return 2, alive


def _parse_caselist_paths(stdout_text: str) -> List[str]:
    discovered: List[str] = []
    for raw_line in stdout_text.splitlines():
        line = raw_line.strip()
        if line.startswith("TEST: "):
            line = line[len("TEST: "):].strip()
        if line.startswith("dEQP-VK."):
            discovered.append(line)
    return list(dict.fromkeys(discovered))


def discover_case_paths(
    args: argparse.Namespace,
    alive: List[Dict[str, Any]],
) -> Tuple[int, List[Dict[str, Any]]]:
    """Resolve every executable case path from deqp-vk's real caselist."""
    executable = _find_deqp_vk(args.cts_root)
    if executable is None:
        print("  [FAIL] 找不到 deqp-vk.exe，无法发现 case path。")
        return 2, alive

    by_root: Dict[str, List[Dict[str, Any]]] = {}
    for item in alive:
        by_root.setdefault(item["root_group"], []).append(item)

    failed = False
    for root_group, items in sorted(by_root.items()):
        case_filter = f"dEQP-VK.{root_group}.*"
        result = _run(
            [
                str(executable),
                "--deqp-runmode=stdout-caselist",
                f"--deqp-case={case_filter}",
            ],
            cwd=executable.parent,
            timeout_seconds=args.run_timeout_seconds,
        )
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        token = re.sub(r"[^A-Za-z0-9_.-]", "_", root_group)
        (LOG_DIR / f"caselist_{token}_stdout.txt").write_text(
            result["stdout"], encoding="utf-8"
        )
        (LOG_DIR / f"caselist_{token}_stderr.txt").write_text(
            result["stderr"], encoding="utf-8"
        )
        discovered = _parse_caselist_paths(result["stdout"])
        if result["returncode"] != 0:
            print(
                f"  [FAIL] caselist {root_group} returncode={result['returncode']}"
            )
            failed = True
            continue

        for item in items:
            expected = str(item["expected_case_name"])
            exact = [
                path for path in discovered
                if path.rsplit(".", 1)[-1] == expected
            ]
            if len(exact) != 1:
                item["case_resolution_error"] = (
                    f"expected leaf {expected!r}, matches={exact}"
                )
                print(
                    f"  [FAIL] {root_group} row={item['row_id']} "
                    f"无法唯一发现叶子 {expected!r}（匹配 {len(exact)}）"
                )
                failed = True
                continue
            item["case_path"] = exact[0]
            item["caselist_filter"] = case_filter
            print(f"  [OK] row={item['row_id']} -> {exact[0]}")

    if not failed:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        ALIVE_PLAN.write_text(
            json.dumps(alive, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return (2 if failed else 0), alive

def _eject_diagnostics(build_text: str, item: Dict[str, Any]) -> Dict[str, Any]:
    """摘出该 case 的链接错误行，并区分"共存产物"与"该 case 自身不完整"。

    LNK2005 = 同一 package 里两个 case 在 namespace 作用域定义了同名类（没放进匿名
    namespace）。单独编不会撞 —— 这是**共存产物，不是该 case 的缺陷**，指标上不该
    记成编译失败。
    LNK2001/LNK2019 = 声明了成员函数却没有定义。与共存无关：stage2 只编静态库
    (`--target deqp-vk-{mode}-{model}`)，编进 .lib 不解析外部符号，所以这类残缺代码在
    stage2 拿到 returncode==0，只有 stage3 链接 deqp-vk.exe 时才暴露。
    """
    token = f"_{item['row_id']}.obj"
    lines = [line.strip() for line in build_text.splitlines()
             if token in line and "error" in line.lower()]
    codes = set(re.findall(r"\bLNK\d{4}\b", "\n".join(lines)))
    if codes & {"LNK2001", "LNK2019"}:
        kind = "case_incomplete_undefined_symbol"
        note = ("该 case 声明了成员函数却未定义。stage2 只编静态库不链接，"
                "所以 terminal_status='compiled' 没能发现。属于该 case 自身的缺陷。")
    elif "LNK2005" in codes:
        kind = "coexistence_duplicate_symbol"
        note = ("与同 package 的其他 case 符号重定义（类未放进匿名 namespace）。"
                "单独编译不会失败，是共存产物，不是该 case 的缺陷。")
    else:
        kind = "unclassified_link_error"
        note = "未归入已知类别，需人工看日志。"
    return {"kind": kind, "note": note, "link_error_codes": sorted(codes),
            "error_lines": lines[:8]}

def record_ejected(ejected: List[Dict[str, Any]]) -> None:
    """把被剔除的 case 落盘。

    不落盘就等于静默截断：指标脚本扫 3_Run/data 只会看到执行成功的那些，
    永远不知道有几个被排除、为什么被排除。
    """
    if not ejected:
        return
    path = RUN_DATA_ROOT / "stage3_ejected_cases.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    kinds = Counter(item.get("eject_diagnostics", {}).get("kind", "?")
                    for item in ejected)
    path.write_text(json.dumps({
        "generated_at": _iso_now(),
        "note": ("这些 case 在 stage2 判定为 compiled，但在 stage3 的整体链接里失败被剔除，"
                 "因此没有运行期结论。统计时不能当作『未执行』忽略。"),
        "ejected_count": len(ejected),
        "by_kind": dict(kinds),
        "cases": [{
            "mode": item["mode"], "model": item["model"], "row_id": item["row_id"],
            "compiled_attempt": item.get("attempt"),
            "source_case_dir": item.get("case_dir"),
            "eject_round": item.get("eject_round"),
            "eject_reason": item.get("eject_reason"),
            "diagnostics": item.get("eject_diagnostics", {}),
        } for item in ejected],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  剔除台账 -> {path}")

def _apidump_env(api_dump_path: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env["VK_LAYER_PATH"] = str(VULKAN_SDK_ROOT / "Bin")
    env["VK_INSTANCE_LAYERS"] = VK_APIDUMP_LAYER_NAME
    env["VK_APIDUMP_OUTPUT_FORMAT"] = "text"
    env["VK_APIDUMP_LOG_FILENAME"] = str(api_dump_path)
    env["VK_APIDUMP_DETAILED"] = "false"
    env["VK_APIDUMP_NO_ADDR"] = "true"
    env["VK_APIDUMP_FLUSH"] = "true"
    return env

def classify_qpa(qpa_path: Path, returncode: int | None = None) -> Dict[str, Any]:
    """按 <Result StatusCode="..."> 判桶。

    true  = dEQP 给出了正常结论 (Pass/Fail/NotSupported/QualityWarning/CompatibilityWarning)
    false = 拿不到正常结论：InternalError 等、qpa 截断（有 #beginTestCaseResult 但无
            StatusCode，进程死在测试中途）、qpa 缺失。

    ⚠ 未修缺陷（2026-08-03）：下面的 no-codes 分支把三种性质完全不同的情况混成
    (truncated) / (no_verdict) 两类，全进 false 桶：

      1. 过滤器没命中     rc=0 + 有 #endSession + 无 #beginTestCaseResult + stdout 0/0
                          -> **工具错（case_path 拼错名字），不是方法失败**
      2. 跑到中途崩       有 #beginTestCaseResult 但无 StatusCode  -> 方法的运行期缺陷
      3. 注册期崩溃       rc!=0 (0xC0000005) + 无 #endSession
                          -> **共存产物**：一个坏 populate_* 会毒死整个 package 的
                             全部 case，与 LNK2005 同类，不是单 case 的缺陷

    第 1 类的成因见本文件 case_path 处的注释；实测对照见
    E:\\Temp\\claude\\filter_miss\\summary.json。三者在 qpa + returncode 上可判别，
    所以能加独立 verdict 值区分，不必重跑（但第 1 类要拿到结论仍需用真名重跑）。

    本函数只拿到 qpa 路径、看不到 returncode，所以要完整区分需要调用方传入 rc ——
    修法与三个待定口径见 README.md 的「⚠ 未解决」一节。在修掉之前，
    **不要按 true/false 桶直接做 RQ 统计**。
    """
    if not qpa_path.is_file():
        return {"bucket": "false", "deqp_verdict": "(no_qpa)",
                "has_begin_marker": False, "is_complete": False}
    text = qpa_path.read_text(encoding="utf-8", errors="replace")
    codes = STATUS_CODE_RE.findall(text)
    has_begin = "#beginTestCaseResult" in text
    complete = "#endSession" in text
    if not codes:
        if has_begin:
            verdict = "(truncated)"
            failure_kind = "execution_incomplete"
        elif returncode == 0 and complete:
            verdict = "(filter_miss)"
            failure_kind = "case_filter_miss"
        elif returncode not in (None, 0):
            verdict = "(pre_verdict_crash)"
            failure_kind = "registration_or_startup_failure"
        else:
            verdict = "(no_verdict)"
            failure_kind = "unknown_no_verdict"
        return {"bucket": "false", "deqp_verdict": verdict,
                "failure_kind": failure_kind,
                "has_begin_marker": has_begin, "is_complete": complete}
    verdict = codes[0]
    if verdict in TRUE_VERDICTS:
        bucket = "true"
    elif verdict in FALSE_VERDICTS:
        bucket = "false"
    else:
        # 未见过的 StatusCode 归 false 并保留原值，宁可少算 true 也不要把未知当正常。
        bucket = "false"
    return {"bucket": bucket, "deqp_verdict": verdict,
            "all_status_codes": codes,
            "has_begin_marker": has_begin, "is_complete": complete}

def command_execute(args: argparse.Namespace,
                    alive: Optional[List[Dict[str, Any]]] = None) -> int:
    if alive is None:
        if ALIVE_PLAN.is_file():
            alive = json.loads(ALIVE_PLAN.read_text(encoding="utf-8"))
        else:
            alive = load_plan_or_die(args)
    unresolved = [item for item in alive if not item.get("case_path")]
    if unresolved:
        print(
            f"  [FAIL] {len(unresolved)} 个候选尚未通过真实 caselist 解析 case path。"
        )
        return 2
    executable = _find_deqp_vk(args.cts_root)
    if executable is None:
        print(f"  [FAIL] 找不到 deqp-vk.exe。先跑 --build。")
        return 2
    print(f"=== 逐个执行   n={len(alive)} ===")
    print(f"  可执行文件 {executable}")
    print(f"  输出根     {RUN_DATA_ROOT}\n")

    buckets = Counter()
    verdicts = Counter()
    records: List[Dict[str, Any]] = []
    for index, item in enumerate(alive, start=1):
        out_dir = (RUN_DATA_ROOT / item["mode"] / item["model"]
                   / "_pending" / f"case_{item['row_id']}")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        qpa_path = out_dir / "TestResults.qpa"
        api_dump_path = out_dir / "vk_apidump.txt"
        case_path = item["case_path"]

        result = _run([str(executable), f"--deqp-case={case_path}",
                       "--deqp-watchdog=disable", "--deqp-log-images=disable",
                       "--deqp-log-shader-sources=disable",
                       f"--deqp-log-filename={qpa_path}"],
                      cwd=executable.parent,
                      timeout_seconds=args.run_timeout_seconds,
                      env=_apidump_env(api_dump_path))
        verdict = classify_qpa(qpa_path, result["returncode"])
        bucket = verdict["bucket"]
        buckets[bucket] += 1
        verdicts[verdict["deqp_verdict"]] += 1

        summary = {
            "recorded_at": _iso_now(),
            "stage": "stage3_run",
            "mode_name": item["mode"],
            "model_name": item["model"],
            "row_id": item["row_id"],
            "source_case_dir": item["case_dir"],
            "compiled_attempt": item["attempt"],
            "runtime_root": str(args.cts_root),
            "executable": str(executable),
            "case_path": case_path,
            "result_bucket": bucket,
            "deqp_verdict": verdict["deqp_verdict"],
            "failure_kind": verdict.get("failure_kind"),
            "qpa": {
                "path": str(qpa_path),
                "exists": qpa_path.is_file(),
                "size_bytes": qpa_path.stat().st_size if qpa_path.is_file() else 0,
                "has_begin_marker": verdict["has_begin_marker"],
                "is_complete": verdict["is_complete"],
                "all_status_codes": verdict.get("all_status_codes", []),
            },
            "api_dump": {
                "path": str(api_dump_path),
                "exists": api_dump_path.is_file(),
                "size_bytes": (api_dump_path.stat().st_size
                               if api_dump_path.is_file() else 0),
            },
            "run": {
                "returncode": result["returncode"],
                "returncode_hex": f"0x{result['returncode'] & 0xFFFFFFFF:08X}",
                "elapsed_seconds": result["elapsed_seconds"],
                "timed_out": result["timed_out"],
            },
        }
        (out_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        for stream in ("stdout", "stderr"):
            (out_dir / f"run_{stream}.txt").write_text(result[stream], encoding="utf-8")

        # 分桶落位：_pending -> {true|false}。不覆盖既有目录。
        final_dir = (RUN_DATA_ROOT / item["mode"] / item["model"] / bucket
                     / f"case_{item['row_id']}")
        if final_dir.exists():
            print(f"  [{index}/{len(alive)}] {case_path}  {verdict['deqp_verdict']}"
                  f"  -> [SKIP] 目标已存在，产物留在 {out_dir}")
            item["output_dir"] = str(out_dir)
        else:
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(out_dir), str(final_dir))
            item["output_dir"] = str(final_dir)
            print(f"  [{index}/{len(alive)}] {case_path}  "
                  f"{verdict['deqp_verdict']:22s} -> {bucket}"
                  f"  {result['elapsed_seconds']:.1f}s")
        item["result_bucket"] = bucket
        item["deqp_verdict"] = verdict["deqp_verdict"]
        records.append(summary)

    # 清掉空的 _pending 目录
    for mode in MODES:
        for model in MODELS:
            pending = RUN_DATA_ROOT / mode / model / "_pending"
            if pending.is_dir() and not any(pending.iterdir()):
                pending.rmdir()

    print(f"\n=== 分桶 ===")
    for bucket, count in buckets.most_common():
        print(f"  {bucket:6s} {count:5d}")
    print(f"\n=== dEQP 原始结论（指标计算要读这个，不要只按桶）===")
    for verdict, count in verdicts.most_common():
        print(f"  {verdict:24s} {count:5d}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _iso_now().replace(":", "").replace("-", "")
    ledger_path = LOG_DIR / f"execute_ledger_{stamp}.json"
    ledger_path.write_text(json.dumps({
        "generated_at": _iso_now(),
        "cts_root": str(args.cts_root),
        "executed": len(records),
        "buckets": dict(buckets),
        "verdicts": dict(verdicts),
        "cases": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  台账 -> {ledger_path}")
    return 0

def load_plan_or_die(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """从扫盘结果构造挂载计划。原实现在这里读 stage2 sqlite（args.database）。"""
    candidates = scan_compiled_candidates(
        modes=args.modes or None,
        models=args.models or None,
        row_ids=parse_rows(getattr(args, "rows", None)))
    plans = [build_case_plan(item) for item in candidates]
    usable = [item for item in plans if item["skip_reason"] is None]
    skipped = [item for item in plans if item["skip_reason"] is not None]
    if skipped:
        print(f"  [WARN] 跳过 {len(skipped)} 个不可用候选:")
        for reason, count in Counter(i["skip_reason"] for i in skipped).most_common():
            print(f"    {reason}: {count}")
    if not usable:
        raise RuntimeError("没有可挂载的候选")
    limit = getattr(args, "limit", None)
    if limit:
        usable = usable[:limit]
        print(f"  [LIMIT] 只取前 {len(usable)} 个")
    return usable

# ── CLI ─────────────────────────────────────────────────────────────────

def command_plan(args: argparse.Namespace) -> int:
    """列出可挂载的候选并落盘计划。不写 CTS 树。"""
    require_cts_root(args.cts_root)
    candidates = scan_compiled_candidates(
        modes=args.modes or None,
        models=args.models or None,
        row_ids=parse_rows(args.rows))
    print(f"扫描 {paths.runs_root()} -> 编译通过候选 {len(candidates)} 个")
    if not candidates:
        print("没有编译通过的候选。先跑 pipeline.py --stage compile / --stage repair。")
        return 1

    plan = [build_case_plan(c) for c in candidates]
    usable = [e for e in plan if not e.get("skip_reason")]
    skipped = [e for e in plan if e.get("skip_reason")]
    print(f"可挂载 {len(usable)}   跳过 {len(skipped)}")
    if skipped:
        for reason, count in Counter(e["skip_reason"] for e in skipped).most_common():
            print(f"    跳过 {count:4d}: {reason}")

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = STATE_DIR / "plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"计划 -> {plan_path}")
    return 0


def command_mount(args: argparse.Namespace) -> int:
    """按计划挂载进 CTS 树。写入前记台账。"""
    cts_root = require_cts_root(args.cts_root)
    plan = load_plan_or_die(args)
    usable = [e for e in plan if not e.get("skip_reason")]
    if not usable:
        print("计划里没有可挂载条目。")
        return 1
    print(f"挂载 {len(usable)} 个候选进 {cts_root}")
    print("  这一步会往 CTS 树里写文件；结束后跑 --restore 还原。")
    mount_cases(cts_root, usable)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GL2VKFrame 运行期验证（可选）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--stage",
                   choices=("plan", "mount", "build", "execute", "all"),
                   default="plan",
                   help="plan 只扫描不写树；mount/build/execute 会改 CTS 树")
    p.add_argument("--cts-root", default=None,
                   help="VK-GL-CTS 检出路径。必需，无默认值")
    p.add_argument("--output-root", type=Path, default=None,
                   help="运行产物根；本地实验应显式指向不上传的测试目录")
    p.add_argument("--modes", nargs="*", default=None,
                   help="限定方法线（slug），缺省全部三条")
    p.add_argument("--models", nargs="*", default=None, choices=list(MODELS),
                   help="限定模型，缺省全部五个")
    p.add_argument("--rows", default=None, help="限定 row_id，如 44,59 或 100-120")
    p.add_argument("--limit", type=int, default=None,
                   help="只取前 N 个可挂载候选")
    p.add_argument("--build-timeout-seconds", type=int, default=3600,
                   help="整棵树构建超时")
    p.add_argument("--run-timeout-seconds", type=int, default=300,
                   help="单个 case 的执行超时")
    p.add_argument("--max-build-rounds", type=int, default=6,
                   help="编译失败后归因剔除并重编的最大轮数")
    p.add_argument("--restore", action="store_true",
                   help="按台账还原 CTS 树并退出")
    p.add_argument("--verify-clean", action="store_true",
                   help="校验 CTS 树没有残留挂载并退出")
    return p


def main() -> int:
    args = build_parser().parse_args()
    configure_artifact_root(args.output_root)
    if args.modes:
        try:
            args.modes = [normalize_variant(item) for item in args.modes]
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    if args.restore:
        return restore_from_manifest()

    if args.verify_clean:
        cts_root = require_cts_root(args.cts_root)
        clean, leftovers = verify_tree_clean(cts_root)
        if clean:
            print(f"CTS 树干净: {cts_root}")
            return 0
        print(f"发现 {len(leftovers)} 处残留:")
        for item in leftovers[:20]:
            print(f"    {item}")
        print("残留挂载会让后续批次持续报 refusing to inject drifted。先跑 --restore。")
        return 1

    args.cts_root = require_cts_root(args.cts_root)
    stage = args.stage

    if stage == "plan":
        return command_plan(args)

    if stage == "mount":
        baseline_rc = verify_baseline(args.cts_root, args.build_timeout_seconds)
        if baseline_rc:
            return baseline_rc
        return command_mount(args)

    if stage == "build":
        build_rc, _ = command_build(args, load_plan_or_die(args))
        return build_rc

    if stage == "execute":
        alive = (
            json.loads(ALIVE_PLAN.read_text(encoding="utf-8"))
            if ALIVE_PLAN.is_file()
            else load_plan_or_die(args)
        )
        discover_rc, alive = discover_case_paths(args, alive)
        if discover_rc:
            return discover_rc
        return command_execute(args, alive)

    # `all` is the safe local-test transaction: baseline -> mount -> full link ->
    # caselist discovery -> execute -> unconditional candidate unmount.
    plan_rc = command_plan(args)
    if plan_rc:
        return plan_rc
    baseline_rc = verify_baseline(args.cts_root, args.build_timeout_seconds)
    if baseline_rc:
        return baseline_rc
    usable = load_plan_or_die(args)
    mount_cases(args.cts_root, usable)
    rc = 0
    try:
        build_rc, alive = command_build(args, usable)
        rc |= build_rc
        if build_rc == 0:
            discover_rc, alive = discover_case_paths(args, alive)
            rc |= discover_rc
            if discover_rc == 0:
                rc |= command_execute(args, alive)
    finally:
        rc |= restore_from_manifest()
        clean, leftovers = verify_tree_clean(args.cts_root)
        if not clean:
            print(f"  [FAIL] restore 后仍有 {len(leftovers)} 处候选挂载残留。")
            rc |= 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
