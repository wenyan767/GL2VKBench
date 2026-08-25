"""base_class.py — 给每个 case_dir 补一级基类完整定义 base_class.json。

在 extract.py 产出 child_class.json / function_tests.json 之后运行。

设计（与数据集一致）：
- 递归深度 = 1 级：只补 record 直系父类；父类的父类仅留 name 引用。
- 存储：每个 case_dir 同级 base_class.json（自包含，跨 dir 重复存）。
- 只收本地(非框架)基类；class_id = <module>_<stem>__base_<i>，i 仅在非框架里顺序编号。
  framework 根(TestCase/TestNode/TestCaseGroup)不进 base_class.json。
- 字段：name/qualified_name/namespace/file_path/code/class_id/is_framework/base_classes。
- code：上游 class 定义 + 全部成员实现（用 primitives 抽取，保留原文注释）。
- --write 时顺带回填同级 child/function 的 base_classes[].id：
  本地基类→本地 __base_i；canonical framework 根→确定性 fw__ id(见 CANON_FW_ID)；
  跨库共享基类的 fw__ 提升属 v3.33 全库聚合,超出本包范围,保持原值不猜。

用法：
  python base_class.py            # dry-run（只统计）
  python base_class.py --write    # 落盘 base_class.json + 回填 id
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import config
from primitives import strip_comments_strings, ns_stack_at, brace_end

UP = config.SOURCE_ROOT
EXT = UP / "external"
DATASETS = [config.VK_OUTPUT_ROOT]
FRAMEWORK = {"TestCase", "TestNode", "TestCaseGroup"}

# v3.33 framework 层：外部 deqp 根类的确定性 fw__ id（qualified_name.replace('::','_')）。
# 仅收录无歧义、可自包含解析的 canonical 根；跨库共享基类的 fw__ 提升属 v3.33 全库聚合，超出本包范围。
CANON_FW_ID = {
    "TestCase": "fw__deqp_TestCase",
    "TestNode": "fw__deqp_TestNode",
    "TestCaseGroup": "fw__deqp_TestCaseGroup",
}

_idx = None


def cls_index():
    """全库 class 名 → 文件列表 索引（含 framework）。"""
    global _idx
    if _idx is not None:
        return _idx
    idx: dict = {}
    for root in [EXT / "vulkancts/modules", UP / "framework"]:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix not in (".hpp", ".cpp"):
                continue
            try:
                t = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(p.relative_to(UP)).replace("/", "\\")
            for m in re.finditer(r"\b(?:class|struct)\s+(\w+)\s*(?:<[^>]*>)?\s*(?:\{|:(?!:))", t):
                idx.setdefault(m.group(1), []).append(rel)
    _idx = idx
    return idx


def rd_up(rel: str) -> str:
    p = UP / rel.replace("\\", "/")
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def extract_base_def(simple: str):
    """抽基类 simple 的 decl + 全部成员实现 + 直系父类引用 + file_path（取第一个定义文件）。"""
    files = cls_index().get(simple, [])
    if not files:
        return None
    hpp = next((f for f in files if f.endswith(".hpp")), files[0])
    raw = rd_up(hpp)
    clean = strip_comments_strings(raw)
    m = re.search(rf"\b(?:class|struct)\s+{re.escape(simple)}(?:<[^>]*>)?\s*(?:\{{|:(?!:))", clean)
    if not m:
        return None
    # 父类（: public X, Y ...）
    brace_pos = raw.find("{", m.start())
    seg = raw[m.start(): brace_pos if brace_pos > 0 else m.end()]
    parents = []
    pm = re.search(r":\s*(.+)$", seg, re.S)
    if pm:
        for part in pm.group(1).split(","):
            pn = re.sub(r"\b(public|protected|private|virtual)\b", "", part).strip().rstrip("{").strip()
            pn = re.sub(r"<.*", "", pn).strip()
            if pn:
                parents.append({"name": pn, "id": "", "relationship": "inheritance"})
    # decl 块
    k = clean.find("{", m.start())
    end = brace_end(clean, k)
    while end < len(raw) and raw[end] in " \n\t;":
        end += 1
    decl = raw[m.start():end]
    # impls（在 hpp 及其 .cpp 兄弟）
    impls = []
    for src in [hpp, hpp[:-4] + ".cpp" if hpp.endswith(".hpp") else hpp]:
        t2 = rd_up(src)
        c2 = strip_comments_strings(t2)
        pat = rf"(?:^|\n)[\w:<>,\s\*&]*?\b{re.escape(simple)}(?:<[^>]*>)?\s*::\s*~?\w+\s*(?:<[^>]*>)?\s*\([^;{{}}]*\)\s*(?:const)?\s*\{{"
        for im in re.finditer(pat, c2):
            kk = c2.find("{", im.end() - 1)
            ee = brace_end(c2, kk)
            impls.append(t2[im.start():ee].strip())
    code = decl
    if impls:
        code += "\n\n// === Member function implementations ===\n\n" + "\n\n".join(impls)
    ns = ns_stack_at(clean, m.start())
    file_path = "external\\" + hpp[len("external\\"):] if hpp.startswith("external\\") else hpp
    return {"code": code, "file_path": file_path, "namespace": "::".join(ns),
            "parents": parents, "nimpl": len(impls)}


def _backfill_base_ids(cd: Path, base_recs: list) -> int:
    """把 case_dir 内 child/function 记录的 base_classes[].id 补上。
    本地基类用同级 base_class.json 的 name→class_id；canonical framework 根用 CANON_FW_ID。
    其余(跨库共享基类)保持原值,不猜。返回补/改的条数。
    """
    name2id = {b["name"]: b["class_id"] for b in base_recs}
    n = 0
    for fn in ("child_class.json", "function_tests.json"):
        p = cd / fn
        if not p.exists():
            continue
        data = json.load(open(p, encoding="utf-8"))
        changed = False
        for r in data:
            for b in (r.get("base_classes") or []):
                nm = b.get("name")
                simple = (nm or "").split("::")[-1].split("<")[0]
                newid = name2id.get(nm) or CANON_FW_ID.get(simple)
                if newid and b.get("id") != newid:
                    b["id"] = newid
                    changed = True
                    n += 1
        if changed:
            json.dump(data, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return n


def main() -> int:
    write = "--write" in sys.argv
    total_files = total_recs = fail = filled = 0
    _defcache: dict = {}

    for base in DATASETS:
        if not base.exists():
            continue
        cds = set()
        for jf in list(base.rglob("child_class.json")) + list(base.rglob("function_tests.json")):
            cds.add(jf.parent)
        for cd in sorted(cds):
            module = cd.parent.name
            stem = cd.name
            # 收集该 dir 的一级基类名（保序去重）
            names, seen = [], set()
            for fn in ("child_class.json", "function_tests.json"):
                p = cd / fn
                if not p.exists():
                    continue
                for r in json.load(open(p, encoding="utf-8")):
                    for b in (r.get("base_classes") or []):
                        nm = b.get("name")
                        if nm and nm not in seen:
                            seen.add(nm); names.append(nm)
            if not names:
                continue
            # base_class.json 只收本地(非框架)基类；__base_i 仅在非框架里顺序编号。
            recs = []
            for nm in names:
                simple = nm.split("::")[-1].split("<")[0]
                if simple in FRAMEWORK:
                    continue
                d = _defcache.get(simple)
                if d is None:
                    d = extract_base_def(simple)
                    _defcache[simple] = d or {}
                if not d:
                    fail += 1
                    continue
                recs.append({
                    "name": nm,
                    "qualified_name": (d.get("namespace") + "::" + simple) if d.get("namespace") else nm,
                    "namespace": d.get("namespace", ""),
                    "file_path": d["file_path"],
                    "code": d["code"],
                    "base_classes": d["parents"],
                    "class_id": f"{module}_{stem}__base_{len(recs)}",
                    "is_framework": False,
                })
            if recs:
                total_files += 1
                total_recs += len(recs)
                if write:
                    json.dump(recs, open(cd / "base_class.json", "w", encoding="utf-8"),
                              ensure_ascii=False, indent=2)
            # v3.32 回填：同级 child/function 的 base_classes[].id
            # 本地基类按 name→class_id；canonical framework 根按确定性 fw__ id。
            if write:
                filled += _backfill_base_ids(cd, recs)

    print(f"生成 base_class.json 文件: {total_files}")
    print(f"基类记录总数: {total_recs}")
    print(f"抽取失败(上游定位不到): {fail}")
    if write:
        print(f"回填 base_classes[].id: {filled}")
    print("[WRITE] 已写入。" if write else "[DRY-RUN] 加 --write 执行。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
