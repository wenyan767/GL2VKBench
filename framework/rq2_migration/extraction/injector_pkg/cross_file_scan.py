"""
cross_file_scan.py
==================
同目录 + 父目录的兄弟 cpp，找跨文件对目标类的注册点。

策略：
  1. 主路径：复用 ASTUtils._analyze_file（已 cached）+ analysis["class_registrations"]
     通过 USR 匹配：usr -> class_info -> qualified_name/name -> match target
  2. 回退：纯正则扫兄弟 cpp 的 addChild(new ClassName(...)) / addFunctionCase(...)

性能：
  ASTUtils._analyze_file 有 cache key = 归一化文件路径，全跑一次后重复访问 < 50ms

返回：
  list[dict] —— 每条 {"file", "line", "call", "kind", "registration_class"}
  最多 5 条（每个外文件），跨多文件聚合
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional


# env 必须干净
for name in ("INCLUDE", "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
    os.environ.pop(name, None)


# 回退正则（libclang 解析失败时使用）
_FALLBACK_RE = [
    # addChild(new Cls(...))
    (re.compile(r"addChild\s*\(\s*new\s+(?:[\w:]+::)?(\w+)\s*[({]"), "addChild"),
    # addFunctionCase(func, "...")
    (re.compile(r"addFunctionCase(?:WithPrograms)?\s*\(\s*&?(\w+)\s*[,)]"), "addFunctionCase"),
]


def find_cross_registrations(
    target_class_name: str,
    target_qualified_name: str,
    source_file: Path,
    ast_utils,
    repo_root: Path,
    *,
    max_per_file: int = 5,
    max_total: int = 10,
) -> List[Dict]:
    """同目录 + 父目录的兄弟 cpp，找本类的注册点。

    Args:
        target_class_name: 类短名（"BlitImageTestCase"）
        target_qualified_name: 类限定名（"api::BlitImageTestCase"）
        source_file: 注册点的源文件绝对路径（self 不计入）
        ast_utils: ASTUtils 实例（用于 libclang analysis cache）
        repo_root: 仓库根，用于 file 字段的相对路径（反斜杠、含 external 前缀）
    """
    source_file = Path(source_file).resolve()
    src_dir = source_file.parent
    parent = src_dir.parent

    seen_usrs = set()
    candidates: List[Dict] = []

    # 收集要扫的目录：同目录 + 父目录
    scan_dirs = []
    if src_dir.exists():
        scan_dirs.append(src_dir)
    if parent.exists() and parent != src_dir:
        scan_dirs.append(parent)

    # 先拿 target 在 source_file 自己里的 USR 列表（避免自匹配；正常情况下应该不会在别的文件出现同名）
    own_qualified = target_qualified_name
    own_short = target_class_name

    for scan_dir in scan_dirs:
        if len(candidates) >= max_total:
            break
        try:
            siblings = sorted(scan_dir.glob("*.cpp"))
        except Exception:
            continue
        for sib in siblings:
            if len(candidates) >= max_total:
                break
            try:
                if sib.resolve() == source_file:
                    continue
            except Exception:
                continue

            # 主路径：libclang analysis
            regs = _scan_via_astutils(sib, ast_utils, own_short, own_qualified, seen_usrs)
            if regs:
                candidates.extend(regs[:max_per_file])
                continue

            # 回退：纯正则
            regs = _scan_via_regex(sib, own_short, own_qualified)
            if regs:
                candidates.extend(regs[:max_per_file])

    return candidates


def _scan_via_astutils(
    sib: Path,
    ast_utils,
    own_short: str,
    own_qualified: str,
    seen_usrs: set,
) -> List[Dict]:
    """用 ASTUtils 缓存的 analysis 找匹配。"""
    out: List[Dict] = []
    try:
        analysis = ast_utils._analyze_file(sib)
    except Exception:
        return out
    classes = analysis.get("classes", {})
    class_registrations = analysis.get("class_registrations", {})
    qualified_index = analysis.get("qualified_name_index", {})

    # 用 qualified_name_index 直接定位目标 usr
    target_usrs = set()
    if own_qualified and own_qualified in qualified_index:
        for usr in qualified_index[own_qualified]:
            target_usrs.add(usr)
    # 也允许 short name 匹配（同一类可能在不同 namespace 重名）
    for usr in qualified_index.get(own_short, []):
        target_usrs.add(usr)

    for usr in target_usrs:
        if usr in seen_usrs:
            continue
        seen_usrs.add(usr)
        regs = class_registrations.get(usr, [])
        for r in regs:
            out.append({
                "file": _rel_path_str(sib),
                "line": r.get("line"),
                "call": r.get("call", "").strip(),
                "kind": r.get("kind", "addChild"),
                "registration_class": r.get("registration_class", "未分类"),
            })
    return out


def _scan_via_regex(
    sib: Path,
    own_short: str,
    own_qualified: str,
) -> List[Dict]:
    """纯正则 fallback。"""
    out: List[Dict] = []
    try:
        text = sib.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return out
    short_q = own_qualified.split("::")[-1] if own_qualified else own_short
    target = short_q or own_short
    for pat, kind in _FALLBACK_RE:
        for m in pat.finditer(text):
            captured = m.group(1)
            if captured == target:
                # 算 line
                line_no = text[: m.start()].count("\n") + 1
                out.append({
                    "file": _rel_path_str(sib),
                    "line": line_no,
                    "call": m.group(0),
                    "kind": kind,
                    "registration_class": "未分类",
                })
                if len(out) >= 5:
                    return out
    return out


def _rel_path_str(p: Path) -> str:
    """把绝对路径转成相对仓库根的反斜杠路径（含 external 前缀）。"""
    try:
        # 尝试匹配 CTS 源码树下的 external/... 相对路径。
        s = str(p)
        for marker in ("external\\openglcts\\", "external\\vulkancts\\"):
            idx = s.lower().find(marker)
            if idx != -1:
                return s[idx:].replace("/", "\\")
        # 否则只返回文件名
        return p.name
    except Exception:
        return str(p)
