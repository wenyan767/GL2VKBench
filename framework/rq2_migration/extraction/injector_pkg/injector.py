"""
injector.py
===========
总入口：对单条 record（class_based 或 function_based）注入 4 个新字段：
  - registrations[*].registration_class
  - registrations[*].mount_code_id
  - mount_codes   (dict: id → 函数体源码)
  - cross_registrations  (list)

用法：
  from cts_extractor_incremental.injector import inject_4_fields
  inject_4_fields(record, source_file, ast_utils, side="vk", repo_root=Path("..."))

record 是 mutate-in-place，返回 None。

约定：
  - 不修改 record 的现有字段（name, qualified_name, registrations 等不动）
  - 只在 registrations[*] 里加 registration_class 和 mount_code_id
  - 在 record 顶层加 mount_codes 和 cross_registrations
  - libclang 解析失败时全部标 None / []，不抛异常
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional


# env 必须干净
for name in ("INCLUDE", "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
    os.environ.pop(name, None)


from .mount_code_extractor import find_enclosing_function_extent, make_mount_code_id  # noqa: E402
from .registration_classifier import classify_registration, quick_classify_by_call  # noqa: E402
from .cross_file_scan import find_cross_registrations  # noqa: E402


def inject_4_fields(
    record: Dict,
    source_file,
    ast_utils,
    side: str = "vk",
    repo_root: Optional[Path] = None,
) -> None:
    """对单条 record 注入 4 字段（mutate-in-place）。

    Args:
        record: class_based 或 function_based record
        source_file: 注册点所在 cpp 文件绝对路径
        ast_utils: ASTUtils 实例
        side: "vk" 或 "gl"
        repo_root: 仓库根绝对路径（用于 cross_registrations.file 字段）
    """
    source_file = Path(source_file)
    if repo_root is None:
        repo_root = source_file.parent.parent.parent

    registrations: List[Dict] = record.get("registrations", [])

    # 1. 给每条 registration 打 4 大类标签 + mount_code_id
    mount_codes: Dict[str, str] = {}
    mount_code_source: str = "libclang"  # 默认；fallback 时会被覆盖

    for reg in registrations:
        kind = reg.get("kind", "")
        line_no = reg.get("line")

        # 1a. registration_class（基于 call 文本的 ast-free 分类 —— v3.5 bug 修复）
        reg["registration_class"] = quick_classify_by_call(kind, reg.get("call", ""))

        # 1b. mount_code_id + mount_codes
        if line_no is None:
            reg["mount_code_id"] = None
            continue
        ext = find_enclosing_function_extent(source_file, line_no, ast_utils=ast_utils)
        if ext is None:
            reg["mount_code_id"] = None
            continue
        start_line, end_line, body_text, source = ext  # 4-tuple now
        # 优先级降级：text_only < tree_sitter < libclang；用最低的标注整条 record
        if source == "text_only":
            mount_code_source = "text_only"
        elif source == "tree_sitter" and mount_code_source != "text_only":
            mount_code_source = "tree_sitter"
        mid = make_mount_code_id(start_line, end_line)
        reg["mount_code_id"] = mid
        if mid not in mount_codes:
            mount_codes[mid] = body_text

    record["mount_codes"] = mount_codes
    record["mount_code_source"] = mount_code_source  # v3.6 NEW —— 顶层标注来源

    # 2. cross_registrations
    if record.get("test_type") == "class_based":
        target_class_name = record.get("name", "")
        target_qualified = record.get("qualified_name", "")
        cross = find_cross_registrations(
            target_class_name, target_qualified,
            source_file, ast_utils, repo_root,
        )
        record["cross_registrations"] = cross
    else:
        record["cross_registrations"] = []


def _quick_classify(kind: str, record: Dict) -> str:
    """DEPRECATED: 保留为 backward-compatible stub，路由到 quick_classify_by_call。
    建议改用 quick_classify_by_call(kind, reg.get("call", ""))。
    """
    return quick_classify_by_call(kind, "")


def inject_4_fields_with_classifier(
    record: Dict,
    source_file,
    ast_utils,
    side: str = "vk",
    repo_root: Optional[Path] = None,
    call_cursors_by_line: Optional[Dict[int, tuple]] = None,
) -> None:
    """完整版：分类时调 libclang 拿 call_cursor（更精确，区分 模板/非模板/直接构造）。

    Args:
        call_cursors_by_line: {line_no: (call_cursor, call_name)} 预解析的 call cursor 映射；
                              不传则只用 quick_classify_by_call（基于 call 文本）
    """
    source_file = Path(source_file)
    if repo_root is None:
        repo_root = source_file.parent.parent.parent

    registrations: List[Dict] = record.get("registrations", [])
    mount_codes: Dict[str, str] = {}
    mount_code_source: str = "libclang"

    for reg in registrations:
        kind = reg.get("kind", "")
        line_no = reg.get("line")

        # 1a. registration_class（完整版）
        if call_cursors_by_line and line_no in call_cursors_by_line:
            cursor, call_name = call_cursors_by_line[line_no]
            reg["registration_class"] = classify_registration(cursor, call_name, source_file, ast_utils)
        else:
            reg["registration_class"] = quick_classify_by_call(kind, reg.get("call", ""))

        # 1b. mount_code_id
        if line_no is None:
            reg["mount_code_id"] = None
            continue
        ext = find_enclosing_function_extent(source_file, line_no, ast_utils=ast_utils)
        if ext is None:
            reg["mount_code_id"] = None
            continue
        start_line, end_line, body_text, source = ext
        if source == "text_only":
            mount_code_source = "text_only"
        elif source == "tree_sitter" and mount_code_source != "text_only":
            mount_code_source = "tree_sitter"
        mid = make_mount_code_id(start_line, end_line)
        reg["mount_code_id"] = mid
        if mid not in mount_codes:
            mount_codes[mid] = body_text

    record["mount_codes"] = mount_codes
    record["mount_code_source"] = mount_code_source

    if record.get("test_type") == "class_based":
        target_class_name = record.get("name", "")
        target_qualified = record.get("qualified_name", "")
        cross = find_cross_registrations(
            target_class_name, target_qualified,
            source_file, ast_utils, repo_root,
        )
        record["cross_registrations"] = cross
    else:
        record["cross_registrations"] = []


def collect_call_cursors(source_file, ast_utils) -> Dict[int, tuple]:
    """预解析 source_file 收集所有 addChild / addFunctionCase 的 call_cursor。

    Returns:
        {line_no: (call_cursor, call_name)}
    """
    source_file = Path(source_file)
    out: Dict[int, tuple] = {}
    try:
        analysis = ast_utils._analyze_file(source_file)
    except Exception:
        return out
    for usr, regs in analysis.get("class_registrations", {}).items():
        for r in regs:
            line = r.get("line")
            kind = r.get("kind", "")
            if line is not None:
                # call_cursor 没法直接从 reg dict 反查；标记 None 让 classifier 走 quick 路径
                out[line] = (None, kind)
    for usr, regs in analysis.get("function_registrations", {}).items():
        for r in regs:
            line = r.get("line")
            kind = r.get("kind", "")
            if line is not None:
                out[line] = (None, kind)
    return out