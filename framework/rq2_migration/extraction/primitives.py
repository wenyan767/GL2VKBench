"""primitives.py — 纯文本 C++ 解析原语（从 step4v8_backfill.py 收编）。

只做零依赖的字符级处理：去注释/字符串、命名空间栈、花括号配对、按完整
ns 抽 class 声明块。base_class.py 用它们从上游源码切出基类定义+成员实现。
路径解析改为读 config（env 可覆盖），不再硬编码盘符。
"""
from __future__ import annotations

import bisect
import re
from pathlib import Path

import config

UP = config.SOURCE_ROOT
EXT = UP / "external"

_file_cache: dict = {}
_clean_cache: dict = {}


def read_rel(rel) -> str:
    """读源码文件。约定：file_path 以 external/ 开头则相对 SOURCE_ROOT，否则相对 external/。"""
    key = str(rel)
    if key in _file_cache:
        return _file_cache[key]
    if isinstance(rel, str):
        r = rel.replace("\\", "/")
        p = (UP / r) if r.startswith("external/") else (EXT / r)
    else:
        p = rel
    t = p.read_text(encoding="utf-8", errors="replace") if Path(p).exists() else ""
    _file_cache[key] = t
    return t


def get_clean(rel_or_text, is_text: bool = False) -> str:
    """按文件路径缓存去注释文本。"""
    if is_text:
        return strip_comments_strings(rel_or_text)
    key = str(rel_or_text)
    if key in _clean_cache:
        return _clean_cache[key]
    c = strip_comments_strings(read_rel(rel_or_text))
    _clean_cache[key] = c
    return c


def strip_comments_strings(t: str) -> str:
    """剔除块注释/行注释/字符串，用等长空白替换以保持字符偏移不变。"""
    out = []
    i, n = 0, len(t)
    while i < n:
        c = t[i]
        if c == "/" and i + 1 < n and t[i + 1] == "*":
            j = t.find("*/", i + 2)
            j = (j + 2) if j >= 0 else n
            out.append(" " * (j - i)); i = j
        elif c == "/" and i + 1 < n and t[i + 1] == "/":
            j = t.find("\n", i)
            j = j if j >= 0 else n
            out.append(" " * (j - i)); i = j
        elif c == '"' or c == "'":
            q = c; j = i + 1
            while j < n and t[j] != q:
                if t[j] == "\\":
                    j += 1
                j += 1
            j += 1
            out.append(" " * (j - i)); i = j
        else:
            out.append(c); i += 1
    return "".join(out)


_nsidx_cache: dict = {}


def _ns_events(clean: str):
    """一次扫描，返回 [(pos, ns_stack_tuple)] 事件序列，便于按位置查栈。"""
    key = id(clean)
    if key in _nsidx_cache:
        return _nsidx_cache[key]
    events = []
    ns, dns, d = [], [], 0
    for mm in re.finditer(r"namespace\s+(\w+)\s*\{|\{|\}", clean):
        tok = mm.group(0)
        if tok.startswith("namespace"):
            ns.append(mm.group(1)); dns.append(d); d += 1
        elif tok == "{":
            d += 1
        else:
            d -= 1
            while dns and d <= dns[-1]:
                dns.pop(); ns.pop()
        events.append((mm.end(), tuple(ns)))
    _nsidx_cache[key] = events
    return events


def ns_stack_at(clean: str, pos: int):
    """clean 中 pos 处的 namespace 栈（一次扫描事件序列 + 二分）。"""
    events = _ns_events(clean)
    if not events:
        return ()
    positions = [e[0] for e in events]
    i = bisect.bisect_right(positions, pos) - 1
    return events[i][1] if i >= 0 else ()


def brace_end(clean: str, open_pos: int) -> int:
    """从 open_pos 处的 `{` 找配对 `}` 的下一个位置。"""
    depth, i, n = 1, open_pos + 1, len(clean)
    while i < n and depth > 0:
        if clean[i] == "{":
            depth += 1
        elif clean[i] == "}":
            depth -= 1
        i += 1
    return i
