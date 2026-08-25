"""
mount_code_extractor.py
========================
找注册点所在函数体（mount_code），生成 mount_code_id 和 mount_codes 字段。

策略：
  1. 主路径：libclang ASTUtils._analyze_file（已经遍历整个 TU）
     从 analysis["function_defs_by_name"] + walk_preorder 找 FUNCTION_DECL/CXX_METHOD/CONSTRUCTOR
     看哪个函数节点的 extent.start.line <= target_line <= end.line
  2. 回退：libclang 的 cursor.extent.end.line 对 inline 函数有时给错（停在右括号所在行）
     用 _brace_match_lines 反推真正的右括号所在行号
  3. 最终兜底：03_extract_full_repo/scripts/utils.py 的 find_enclosing_function（纯文本）

返回：
  (start_line, end_line, body_text)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

# libclang 必须在干净 env 下 import
for name in ("INCLUDE", "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
    os.environ.pop(name, None)

import clang.cindex  # noqa: E402


# 函数体边界——以 brace 配对为准
_FN_KINDS = {
    clang.cindex.CursorKind.FUNCTION_DECL,
    clang.cindex.CursorKind.CXX_METHOD,
    clang.cindex.CursorKind.CONSTRUCTOR,
    clang.cindex.CursorKind.DESTRUCTOR,
    clang.cindex.CursorKind.FUNCTION_TEMPLATE,
}


def _find_enclosing_function_via_analysis(
    ast_utils,
    source_file: Path,
    target_line: int,
) -> Optional[Tuple[int, int]]:
    """从 ASTUtils 缓存的 analysis 找包含 target_line 的函数定义。"""
    try:
        analysis = ast_utils._analyze_file(source_file)
    except Exception:
        return None

    candidates = []
    # 1. 自由函数 + 模板函数（ASTUtils._build_function_info 字段是 line_start/line_end）
    for usr, func_info in analysis.get("functions", {}).items():
        start = func_info.get("line_start")
        end = func_info.get("line_end")
        if start and end and start <= target_line <= end:
            candidates.append((start, end, func_info))
    # 2. 类成员方法（methods_by_parent）—— 字段名同样是 line_start/line_end
    for parent_usr, methods in analysis.get("methods_by_parent", {}).items():
        for m in methods:
            start = m.get("line_start")
            end = m.get("line_end")
            if start and end and start <= target_line <= end:
                candidates.append((start, end, m))

    if not candidates:
        return None
    # 取最近的（最小 start_line，但 start 最大且 target 仍落在内）
    candidates.sort(key=lambda x: (x[0], -x[1]))
    # 优先选最小的（最近一层包裹）
    return (candidates[0][0], candidates[0][1])


def _find_enclosing_function_via_walk(
    source_file: Path,
    target_line: int,
    ast_utils=None,
) -> Optional[Tuple[int, int]]:
    """独立 walk_preorder 找函数节点。"""
    if ast_utils is None:
        return None
    tu = ast_utils.parse_file(source_file)
    if tu is None:
        return None
    candidates = []
    for cursor in tu.cursor.walk_preorder():
        if cursor.kind not in _FN_KINDS:
            continue
        if cursor.location.file is None:
            continue
        try:
            if Path(str(cursor.location.file)).resolve() != source_file.resolve():
                continue
        except Exception:
            continue
        start = cursor.extent.start.line
        # 推断 end_line：先信 cursor.extent.end.line，对 inline 函数再用 brace 反推
        end = cursor.extent.end.line
        if start <= target_line <= end + 50:  # 容忍 end 行号偏差
            # brace 反推：拿 cursor text 找最后一个 } 所在行
            text = _cursor_text_safe(ast_utils, source_file, cursor)
            if text and "{" in text:
                real_end = _brace_match_line(source_file, start, text)
                if real_end:
                    end = real_end
            if start <= target_line <= end:
                candidates.append((start, end))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], -x[1]))
    return candidates[0]


def _cursor_text_safe(ast_utils, source_file: Path, cursor) -> Optional[str]:
    """拿 cursor 节点的源码。ast_utils._cursor_text 已处理；回退到 raw tokenization。"""
    if ast_utils is not None and hasattr(ast_utils, "_cursor_text"):
        try:
            return ast_utils._cursor_text(source_file, cursor)
        except Exception:
            pass
    # 兜底：自己 tokenRange
    try:
        tokens = list(cursor.get_tokens())
        return " ".join(t.spelling for t in tokens)
    except Exception:
        return None


def _brace_match_line(source_file: Path, start_line: int, text: str) -> Optional[int]:
    """在 text 中找 { 对应的 } 所在行号。text 是 cursor 节点的整段源码（含 \n）。"""
    # 找第一个 { 的位置（找最外层）
    brace_start = text.find("{")
    if brace_start == -1:
        return None
    depth = 0
    i = brace_start
    in_string = False
    string_char = ""
    in_line_comment = False
    in_block_comment = False
    in_raw_string = False
    raw_delim = ""
    while i < len(text):
        ch = text[i]
        # 行注释
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        # 块注释
        if in_block_comment:
            if ch == "*" and i + 1 < len(text) and text[i + 1] == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        # 原始字符串 R"delim(...)delim"
        if in_raw_string:
            if ch == ")" and text[i:i + 1 + len(raw_delim)] == ")" + raw_delim:
                in_raw_string = False
                raw_delim = ""
                i += 1 + len(raw_delim)
                continue
            i += 1
            continue
        # 普通字符串
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == string_char:
                in_string = False
            i += 1
            continue
        # 注释 / 字符串开始
        if ch == "/" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt == "/":
                in_line_comment = True
                i += 2
                continue
            if nxt == "*":
                in_block_comment = True
                i += 2
                continue
        if ch == "R" and i + 1 < len(text) and text[i + 1] == '"':
            # 找 R"delim(
            j = i + 2
            delim = ""
            while j < len(text) and text[j] != "(":
                delim += text[j]
                j += 1
            if j < len(text) and text[j] == "(":
                in_raw_string = True
                raw_delim = delim
                i = j + 1
                continue
        if ch == '"' or ch == "'":
            in_string = True
            string_char = ch
            i += 1
            continue
        # 大括号
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                # 算 } 所在行号
                lines_before = text[:i].count("\n")
                return start_line + lines_before
        i += 1
    return None


def find_enclosing_function_extent(
    source_file: Path,
    target_line: int,
    ast_utils=None,
) -> Optional[Tuple[int, int, str, str]]:
    """找 target_line 所在函数定义节点的 (start_line, end_line, body_text, source)。

    source 标识 mount_code 抽取来源："libclang" | "tree_sitter" | "text_only"。

    Args:
        source_file: 源 cpp 文件绝对路径
        target_line: 注册点所在行号（1-based）
        ast_utils: vulkan_extractor.ast_utils.ASTUtils 实例（优先用其缓存）

    Returns:
        (start_line, end_line, body_text, source) 或 None
    """
    source_file = Path(source_file)

    ext = None
    source = "libclang"

    if ast_utils is not None:
        # 主路径：复用 analysis 缓存
        ext = _find_enclosing_function_via_analysis(ast_utils, source_file, target_line)
        if ext is None:
            ext = _find_enclosing_function_via_walk(source_file, target_line, ast_utils)

    # 第二路径：tree-sitter（libclang 死锁/失败时的 fallback，不死锁 + 100x 快）
    if ext is None:
        source = "tree_sitter"
        try:
            from .mount_code_treesitter import find_enclosing_function_treesitter
            ext = find_enclosing_function_treesitter(source_file, target_line)
        except Exception:
            ext = None

    # 第三路径：纯文本扫描（最差情况下，仍能给个近似函数体）
    if ext is None:
        source = "text_only"
        ext = _find_enclosing_function_text_only(source_file, target_line)

    if ext is None:
        return None
    start, end = ext
    try:
        text = source_file.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        body = "\n".join(lines[start - 1:end])
    except Exception:
        return None
    return (start, end, body, source)


def _find_enclosing_function_text_only(
    source_file: Path,
    target_line: int,
) -> Optional[Tuple[int, int]]:
    """纯文本兜底：用 brace 配对找最近函数定义。

    适用于 libclang 解析失败的 cpp。
    """
    try:
        text = source_file.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    lines = text.splitlines()
    if target_line < 1 or target_line > len(lines):
        return None

    # 从 target_line 向上找最近的 `{` 所在行（按缩进推断函数头）
    # 启发式：从 target_line 向上扫描，找最近的、左侧是 identifier+`(` 或 `{` 单独成对的行
    # 然后从那一行的 { 开始 brace 配对
    target_text = lines[target_line - 1] if target_line - 1 < len(lines) else ""
    # 找 target_line 之前最近的 function header：含 `(` 且不含 `;` 的行
    func_header_line = None
    for i in range(target_line - 1, max(0, target_line - 200), -1):
        line = lines[i]
        if "(" in line and ";" not in line and ")" in line:
            func_header_line = i + 1
            break
    if func_header_line is None:
        # 全局作用域（namespace 顶层）：直接配对函数体
        # 从 target_line 向下找第一个 { 所在行
        for i in range(target_line - 1, min(len(lines), target_line + 200)):
            if "{" in lines[i]:
                func_header_line = i + 1
                break
    if func_header_line is None:
        return None

    # 从 func_header_line 开始 brace 配对
    start = func_header_line
    depth = 0
    started = False
    for i in range(start - 1, len(lines)):
        line = lines[i]
        # 简化版：跳过注释/字符串内容里的 { 和 }
        clean = _strip_comments_and_strings(line)
        for ch in clean:
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}":
                depth -= 1
                if started and depth == 0:
                    return (start, i + 1)
    return None


def _strip_comments_and_strings(line: str) -> str:
    """移除行内注释 + 字符串字面量后返回（用于粗略 brace 计数）。"""
    out = []
    i = 0
    in_string = False
    string_char = ""
    while i < len(line):
        ch = line[i]
        if not in_string:
            if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                break
            if ch == '"' or ch == "'":
                in_string = True
                string_char = ch
                i += 1
                continue
            out.append(ch)
        else:
            if ch == "\\":
                i += 2
                continue
            if ch == string_char:
                in_string = False
            i += 1
            continue
        i += 1
    return "".join(out)


def make_mount_code_id(start_line: int, end_line: int) -> str:
    """生成 `fn_<start>_<end>` 形式的稳定 id。"""
    return f"fn_{int(start_line)}_{int(end_line)}"