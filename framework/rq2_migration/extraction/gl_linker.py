"""
gl_hpp_cpp_linker.py
====================
v3.9 include-aware GL linker.

解决问题：1,160 个 unclassified records（hpp 声明 + cpp addChild 的跨文件实现）。

核心策略（v3.9 修订）：
  - hpp 走 libclang（精确：class 声明 + base_classes + code）
  - cpp 走 regex（libclang 解析 cpp 整个 codebase 失败，必须用 regex）
  - 按 class name 链接

CLI:
  --dry-run              只打印 plan（默认）
  --apply                实际执行
  --cpp <path>           只处理单个 cpp（调试用）
  --libclang-timeout N   每个 libclang parse 的超时秒数（默认 30）
  --limit N              只跑前 N 个 cpp（0 = all）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# env: libclang needs clean env
for name in ("INCLUDE", "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
    os.environ.pop(name, None)

# === 路径（glue: 全部来自 config.py，env 可覆盖）===
SCRIPT_DIR = Path(__file__).resolve().parent
from config import SOURCE_ROOT as VK_GL_CTS  # noqa: E402
from config import GL_MODULES_ROOT, GL_OUTPUT_ROOT  # noqa: E402
# hpp 子进程 sys.path 注入本包目录（内含 ast_engine.py），不再依赖外部 src。
SRC_ROOT = SCRIPT_DIR
OUT_ROOT = GL_OUTPUT_ROOT

# === 缓存 ===
hpp_class_cache: Dict[Path, Dict[str, Dict]] = {}

# === 统计 ===
n_libclang_hang = 0


# ============================================================
# 1. #include 解析（复用 run_opengl_ast_extract.py:585-625）
# ============================================================

def _collect_local_includes(cpp: Path, module_dir: Path, modules_root: Path) -> List[Path]:
    """Return CTS-local headers #include'd by this cpp, resolved to real files.

    v3.11 改动：除了 #include 的 hpp，也加入 cpp 同目录的所有 hpp
    （因为 cpp 里 addChild 的类可能在同名 hpp 或共享的 utility hpp 里）
    """
    out: List[Path] = []
    try:
        text = cpp.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return out
    seen: Set[str] = set()
    cache: Dict[str, Dict[str, Optional[Path]]] = {}

    # v3.11 新增：先加入 cpp 同目录的所有 hpp（共享 utility 可能不被 #include）
    for hpp in cpp.parent.glob("*.hpp"):
        if hpp not in out:
            out.append(hpp)
            seen.add(hpp.name)

    # 原 #include 解析
    for m in re.finditer(r'#\s*include\s+"([^"]+)"', text):
        inc = m.group(1).strip()
        base = inc.replace("\\", "/").split("/")[-1]
        if not base.lower().endswith((".hpp", ".h")):
            continue
        if base in seen:
            continue
        seen.add(base)
        cand = cpp.with_name(base)
        if cand.exists():
            out.append(cand)
            continue
        resolved: Optional[Path] = None
        for search_root in (module_dir, modules_root):
            sub_cache = cache.setdefault(str(search_root), {})
            if base in sub_cache:
                resolved = sub_cache[base]
            else:
                hits = list(search_root.rglob(base))
                sub_cache[base] = hits[0] if hits else None
                resolved = sub_cache[base]
            if resolved is not None:
                break
        if resolved is not None:
            out.append(resolved)
    return out


# ============================================================
# 2. hpp 端：libclang 扫 hpp 拿 class 声明
# ============================================================

LIBCLANG_BODY_TPL = """
import sys, json, re, os
from pathlib import Path
sys.path.insert(0, r"{src_root}")
for _n in ("INCLUDE", "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
    os.environ.pop(_n, None)
from ast_engine import ASTUtils
import clang.cindex

target = Path(r"{target}")
au = ASTUtils()
try:
    a = au._analyze_file(target)
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
    sys.exit(0)

# 从 class code 文本里 regex 提取 base class
# （libclang get_children 和 walk_preorder 都不返回 nested class 的 CXX_BASE_SPECIFIER）
def extract_bases_from_code(code):
    m = re.search(r'class\s+\w+\s*:\s*([^{{]+)', code)
    if m:
        bases_str = m.group(1)
        parts = re.split(r',', bases_str)
        out = []
        for p in parts:
            p = p.strip()
            p = re.sub(r'^(public|private|protected|virtual)\s+', '', p)
            p = p.strip()
            if p:
                out.append(p)
        return out
    return []

out = {{}}
for usr, cls in a.get("classes", {{}}).items():
    code = cls.get("code", "")
    base_classes = extract_bases_from_code(code) or cls.get("base_classes", [])
    qn = cls.get("qualified_name", "") or cls["name"]
    # v3.11 修复：用 qualified_name 作 key 避免 namespace 嵌套的同名类覆盖
    out[qn] = {{
        "name": cls["name"],
        "usr": usr,
        "code": code,
        "base_classes": base_classes,
        "qualified_name": qn,
        "namespace": cls.get("namespace", ""),
        "line_start": cls.get("line_start"),
        "line_end": cls.get("line_end"),
        "file_path": str(cls.get("file_path", target)),
    }}
print(json.dumps({{"classes": out}}))
"""


def _run_libclang_in_subprocess(target: Path, timeout: int) -> Optional[Dict]:
    global n_libclang_hang
    body = LIBCLANG_BODY_TPL.format(
        src_root=str(SRC_ROOT),
        target=str(target),
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", body],
            capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        n_libclang_hang += 1
        return None
    except Exception:
        n_libclang_hang += 1
        return None
    if proc.returncode != 0:
        return None
    out_text = proc.stdout.strip()
    if not out_text:
        return None
    try:
        result = json.loads(out_text)
    except Exception:
        return None
    if "error" in result:
        return None
    return result


def _analyze_hpp(hpp_path: Path, libclang_timeout: int) -> Dict[str, Dict]:
    """libclang 扫 hpp → {class_name: {code, base_classes, ...}}（带缓存）。"""
    hpp_path = hpp_path.resolve()
    if hpp_path in hpp_class_cache:
        return hpp_class_cache[hpp_path]
    result = _run_libclang_in_subprocess(hpp_path, libclang_timeout)
    classes = result.get("classes", {}) if result else {}
    hpp_class_cache[hpp_path] = classes
    return classes


# ============================================================
# 3. cpp 端：regex 扫 addChild（v3.11 支持 qualified_name）
# ============================================================

# v3.11 改进：捕获完整 qualified_name（含 namespace 嵌套）
# 例如 addChild(new BufferStorage::DynamicStorageTest(...))  → qn = "BufferStorage::DynamicStorageTest"
CPP_FALLBACK_PATTERN = re.compile(
    r"addChild\s*\(\s*new\s+([\w]+(?:::\w+)+|\w+)\b"
)


def _extract_full_addChild_call(text: str, m_start: int, max_len: int = 600) -> Optional[str]:
    """从 m_start（addChild 起点）开始抓完整 addChild(new X(...)); 调用。

    配对括号深度计数，找到第一个匹配的 ); 终止。
    Returns: 完整调用字符串（不含尾随空白），或 None（不匹配）。
    """
    # 从 m_start 找 addChild 头的结尾（class name + 第一个 (）
    head_end = m_start
    # 跳过 addChild
    head_end = text.find('(', m_start)
    if head_end == -1:
        return None
    head_end += 1  # 跳过第一个 (
    # 跳过空白
    while head_end < len(text) and text[head_end].isspace():
        head_end += 1
    # 跳过 new
    if not text[head_end:head_end+4].startswith("new "):
        return None
    head_end += 4
    while head_end < len(text) and text[head_end].isspace():
        head_end += 1
    # 跳过 class name（可能含 ::）
    while head_end < len(text) and (text[head_end].isalnum() or text[head_end] in "_:"):
        head_end += 1
    # 跳过空白
    while head_end < len(text) and text[head_end].isspace():
        head_end += 1
    # 现在应该在 class 后的 ( 位置
    if head_end >= len(text) or text[head_end] != '(':
        return None
    # 配对括号深度
    depth = 1
    j = head_end + 1
    while j < len(text) and depth > 0:
        c = text[j]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == ';' and depth == 0:
            break
        j += 1
    if depth != 0:
        return None
    # j 现在在匹配的 ) 之后
    semi = text.find(';', j)
    if semi == -1 or semi - m_start > max_len:
        end_pos = min(j + 1, m_start + max_len)
    else:
        end_pos = semi + 1
    return text[m_start:end_pos].strip()


def _extract_namespace_from_call(call_text: str) -> str:
    """从 addChild(new ns::X(...)) 提取命名空间。"""
    m = re.match(r"addChild\s*\(\s*new\s+([\w]+(?:::\w+)*)\s*\(", call_text)
    if m:
        qn = m.group(1)
        if "::" in qn:
            return "::".join(qn.split("::")[:-1])
        return ""
    return ""


def _looks_like_gl_test_class(name: str) -> bool:
    """GL 测试类名特征。"""
    if not name:
        return False
    if name in ("TestSubcase", "TestCase", "TestCaseGroup"):
        return False
    # v3.11 改进：也接受含 :: 的 qualified_name
    bare = name.split("::")[-1] if "::" in name else name
    # v3.11 改进：去掉大小写强制（更宽容）
    return bare[0].isupper() and (
        bare.endswith("Test") or bare.endswith("TestCase") or bare.endswith("Tests")
        or bare.endswith("Instance") or bare.endswith("Group") or "Test" in bare or "Case" in bare
        or bare.endswith("Base")  # v3.11 改进：接受 TestBase 类
        or bare.endswith("Case")  # v3.11 改进
    )


def _is_test_case_class(cls_info: Dict) -> bool:
    """v3.11 改进：检查 class 是否是 TestCase 的子孙（直接或间接）。"""
    if not cls_info:
        return False
    # 直接继承
    if cls_info.get("base_classes"):
        return True
    # 看 code 文本里有没有 public TestCase 链
    code = cls_info.get("code", "")
    if "public deqp::TestCase" in code or "public glcts::TestCase" in code or "public TestCase" in code:
        return True
    if "public deqp::TestCaseGroup" in code or "public TestCaseGroup" in code:
        return True
    return False


def _analyze_cpp_for_addChild(cpp_path: Path) -> Dict[str, List[Dict]]:
    """regex 扫 cpp 拿 addChild，按 qualified_name 分组（v3.11）。

    Returns: {qualified_name: [addChild_reg, ...]}
    addChild_reg = {kind, line, call, namespace, source}

    v3.11 改动：key 改用 qualified_name 而非裸 class_name，以匹配 hpp 里的嵌套 namespace 类
    """
    out: Dict[str, List[Dict]] = {}
    try:
        text = cpp_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return out
    for m in CPP_FALLBACK_PATTERN.finditer(text):
        qn = m.group(1)  # 完整 qualified_name
        if not _looks_like_gl_test_class(qn):
            continue
        line_no = text[:m.start()].count("\n") + 1
        # 抓完整 addChild(new X(...)); 调用
        full_call = _extract_full_addChild_call(text, m.start())
        if not full_call:
            # 兜底：仍用旧 200 字符截断
            full_call = text[m.start():m.end()][:200]
        ns = _extract_namespace_from_call(full_call)
        reg = {
            "kind": "addChild",
            "line": line_no,
            "call": full_call,
            "namespace": ns,
            "source": "regex",
        }
        out.setdefault(qn, []).append(reg)
    return out


# ============================================================
# 4. linker merge：per cpp
# ============================================================

def link_one_cpp(
    cpp_path: Path,
    hpp_paths: List[Path],
    libclang_timeout: int,
) -> Tuple[List[Dict], int]:
    """对一个 cpp 做完整 linker：拿 hpp classes + cpp addChild → merged records。

    Returns: (records, n_unmatched_class_names)

    v3.11 改进：用 namespace endswith 匹配
    - cpp 里 addChild 的类名是相对引用（如 TransformFeedback::CreationTest）
    - hpp 里 qualified_name 是全限定名（如 gl4cts::DirectStateAccess::TransformFeedback::CreationTest）
    - 匹配策略：hpp_namespace.endswith(cpp_namespace) 且 class_name 相同
    """
    # 1. 拿 hpp 端所有 class（cache 共享）
    hpp_classes = []  # list of (qualified_name, hpp_path, class_info)
    for hpp in hpp_paths:
        for cls_name, cls_info in _analyze_hpp(hpp, libclang_timeout).items():
            qn = cls_info.get("qualified_name") or cls_info["name"]
            hpp_classes.append((qn, hpp.resolve(), cls_info))

    # 2. 拿 cpp 端 addChild（regex 可靠，libclang 在 cpp 上全军覆没）
    cpp_regs = _analyze_cpp_for_addChild(cpp_path)

    # 3. merge：用 namespace endswith 匹配
    records: List[Dict] = []
    n_unmatched = 0
    matched_keys = set()  # 防止同一 hpp_class 被多次匹配

    for cpp_qn, regs in cpp_regs.items():
        # cpp_qn = "TransformFeedback::CreationTest"
        # 拆 namespace 和 class name
        if "::" in cpp_qn:
            cpp_ns, cpp_class = "::".join(cpp_qn.split("::")[:-1]), cpp_qn.split("::")[-1]
        else:
            cpp_ns, cpp_class = "", cpp_qn

        # 找匹配的 hpp class
        matched = None
        for (hpp_qn, hpp_p, cls_info) in hpp_classes:
            # 拆 hpp namespace 和 class name
            if "::" in hpp_qn:
                hpp_ns, hpp_class = "::".join(hpp_qn.split("::")[:-1]), hpp_qn.split("::")[-1]
            else:
                hpp_ns, hpp_class = "", hpp_qn

            # class name 必须相同
            if hpp_class != cpp_class:
                continue

            # hpp namespace 必须以 cpp namespace 结尾
            # v3.11: 修复 - 裸名 cpp 调用（cpp_ns=""）应该匹配任何 hpp namespace
            ns_match = False
            if cpp_ns == "":  # 裸名调用
                ns_match = True  # 任何 hpp namespace 都算匹配
            elif hpp_ns == cpp_ns:
                ns_match = True
            elif hpp_ns.endswith("::" + cpp_ns):
                ns_match = True

            if ns_match:
                # v3.11: 接受任何 base class（包括 TestBase 等中间类）
                key = (str(hpp_p), hpp_qn)
                if key in matched_keys:
                    continue
                if matched is None:
                    matched = (hpp_p, cls_info, hpp_qn)

        if matched is None:
            n_unmatched += 1
            continue
        hpp_p, cls_info, hpp_qn = matched
        matched_keys.add((str(hpp_p), hpp_qn))

        # file_path 用 hpp 的相对路径（与 v2.0 ref 格式一致：external\openglcts\modules\gl\...）
        try:
            hpp_rel = Path(cls_info["file_path"]).relative_to(VK_GL_CTS)
            hpp_file_path = str(hpp_rel)
        except ValueError:
            hpp_file_path = cls_info["file_path"]

        rec = {
            "name": cls_info["name"],
            "qualified_name": cls_info.get("qualified_name", cls_info["name"]),
            "namespace": cls_info.get("namespace", ""),
            "test_type": "class_based",
            "test_level": "class",
            "file_path": hpp_file_path,
            "code": cls_info.get("code", ""),
            "base_classes": [{"name": b, "id": "", "relationship": "inheritance"}
                             for b in cls_info.get("base_classes", [])],
            "registrations": regs,
            "linked_in_v311": True,  # v3.11 marker
            "linker_hpp": str(hpp_p),
            "linker_cpp": str(cpp_path),
            "merged_from_v2": False,
        }
        records.append(rec)
    return records, n_unmatched


# ============================================================
# 5. 写回 dataset_v2
# ============================================================

def write_records_to_dataset(cpp_path: Path, new_records: List[Dict], out_root: Path) -> Tuple[int, int]:
    """把 linker 产生的 records 写回 dataset_v2 对应目录。

    策略：
      - 如果 (name, file_path) 已在 existing 里 → **update** 该 record 的 registrations/base_classes/code
      - 否则 → append 新 record

    Returns: (n_updated, n_appended)
    """
    try:
        rel = cpp_path.relative_to(GL_MODULES_ROOT)
    except ValueError:
        return 0, 0
    parts = rel.parts
    if len(parts) < 2:
        return 0, 0
    cat = parts[0]
    stem = cpp_path.stem  # 不要 .cpp 后缀！例如 'gl4cDirectStateAccessTests'
    module_dir = out_root / cat / stem
    module_dir.mkdir(parents=True, exist_ok=True)
    json_path = module_dir / "child_class.json"

    if json_path.exists():
        existing = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(existing, list):
            existing = []
    else:
        existing = []

    # 索引 existing by (name, file_path)
    existing_index: Dict[Tuple[str, str], int] = {}
    for i, r in enumerate(existing):
        if isinstance(r, dict):
            key = (r.get("name", ""), r.get("file_path", ""))
            existing_index[key] = i

    n_updated = 0
    n_appended = 0
    for r in new_records:
        key = (r.get("name", ""), r.get("file_path", ""))
        if key in existing_index:
            # Update existing record
            idx = existing_index[key]
            old = existing[idx]
            # 保留 v2.0 ref 的 trace/class_id 等元数据
            for keep_key in ("class_id", "class_type", "trace", "line_start", "line_end"):
                if keep_key in old and keep_key not in r:
                    r[keep_key] = old[keep_key]
            # 替换空 registrations 和 base_classes
            if not old.get("registrations") and r.get("registrations"):
                old["registrations"] = r["registrations"]
                old["linked_in_v39"] = True
                n_updated += 1
            # 保留 better base_classes
            if not old.get("base_classes") and r.get("base_classes"):
                old["base_classes"] = r["base_classes"]
            existing[idx] = old
        else:
            existing.append(r)
            existing_index[key] = len(existing) - 1
            n_appended += 1

    json_path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return n_updated, n_appended


# ============================================================
# 6. main
# ============================================================

def _module_dir_of(cpp: Path) -> Path:
    try:
        rel = cpp.relative_to(GL_MODULES_ROOT)
    except ValueError:
        return GL_MODULES_ROOT
    if rel.parts:
        return GL_MODULES_ROOT / rel.parts[0]
    return GL_MODULES_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--cpp", type=str, default=None)
    ap.add_argument("--libclang-timeout", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.apply:
        args.dry_run = False

    print(f"[linker] mode: {'DRY-RUN' if args.dry_run else 'APPLY'}")
    print(f"[linker] libclang timeout: {args.libclang_timeout}s (hpp only)")
    print(f"[linker] cpp scan: regex (libclang 解析 cpp 在此 codebase 失败)")
    print(f"[linker] GL modules root: {GL_MODULES_ROOT}")
    print(f"[linker] out root: {OUT_ROOT}")
    print()

    if args.cpp:
        cpp_target = (GL_MODULES_ROOT / args.cpp).resolve()
        if not cpp_target.exists():
            print(f"!! cpp not found: {cpp_target}")
            return 1
        all_cpps = [cpp_target]
    else:
        all_cpps = sorted(GL_MODULES_ROOT.rglob("*.cpp"))

    if args.limit and args.limit > 0:
        all_cpps = all_cpps[:args.limit]

    print(f"[linker] total cpp: {len(all_cpps)}")

    n_with_hpps = 0
    n_skipped_no_hpp = 0
    n_records_linked = 0
    n_records_unique_written = 0
    n_unmatched = 0
    t0 = time.time()

    for idx, cpp in enumerate(all_cpps):
        module_dir = _module_dir_of(cpp)
        hpps = _collect_local_includes(cpp, module_dir, GL_MODULES_ROOT)
        if not hpps:
            n_skipped_no_hpp += 1
            continue
        n_with_hpps += 1

        if args.dry_run:
            rel = cpp.relative_to(GL_MODULES_ROOT)
            print(f"  [{idx+1:4d}/{len(all_cpps)}] {rel}  includes {len(hpps)} hpp(s)")
            for h in hpps[:3]:
                try:
                    h_rel = h.relative_to(GL_MODULES_ROOT)
                    print(f"           → {h_rel}")
                except ValueError:
                    print(f"           → {h}")
            continue

        try:
            records, n_um = link_one_cpp(cpp, hpps, args.libclang_timeout)
        except Exception as e:
            import traceback
            print(f"  [ERROR] {cpp}: {e}")
            traceback.print_exc()
            continue

        n_unmatched += n_um
        if not records:
            continue

        # 去重（per linker run, per hpp+class）
        seen_keys: Set[Tuple[str, str]] = set()
        uniq_records = []
        for r in records:
            key = (r.get("linker_hpp", ""), r.get("name", ""))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            uniq_records.append(r)

        n_updated, n_appended = write_records_to_dataset(cpp, uniq_records, OUT_ROOT)
        n_records_linked += len(records)
        n_records_unique_written += n_updated + n_appended

        if (idx + 1) % 20 == 0 or idx == len(all_cpps) - 1:
            elapsed = time.time() - t0
            print(f"  [{idx+1:4d}/{len(all_cpps)}] processed {n_records_linked} records "
                  f"({n_records_linked/max(1, elapsed):.1f}/s), "
                  f"updated={n_updated}, appended={n_appended}, "
                  f"with hpps: {n_with_hpps}, skipped: {n_skipped_no_hpp}, "
                  f"unmatched: {n_unmatched}, hpp libclang hang: {n_libclang_hang}, "
                  f"hpp cache: {len(hpp_class_cache)}",
                  flush=True)

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"[linker] DONE in {elapsed:.1f}s")
    print(f"[linker] total cpp: {len(all_cpps)}")
    print(f"[linker]   with local includes: {n_with_hpps}")
    print(f"[linker]   skipped (no hpp): {n_skipped_no_hpp}")
    print(f"[linker]   records linked: {n_records_linked}")
    print(f"[linker]   unique records written: {n_records_unique_written}")
    print(f"[linker]   unmatched class names: {n_unmatched}")
    print(f"[linker]   hpp libclang timeouts: {n_libclang_hang}")
    print(f"[linker]   hpp cache size: {len(hpp_class_cache)}")
    if args.dry_run:
        print(f"[linker] (DRY-RUN, no records written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
