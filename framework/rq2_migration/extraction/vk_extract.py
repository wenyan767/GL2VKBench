"""
vk_processor.py
===============
单 cpp 处理逻辑（从 phase1_smoke.py 抽出，便于被阶段 3 全量脚本复用）。

对外提供：
  process_vk_cpp(cpp_path, ast_utils) → dict（含 class_records / function_records）
  process_gl_cpp(cpp_path, ast_utils) → dict

v3.20.2 改进：_has_test_case_base 升级为继承链 DFS — 直系父类不含
TestCase/SubcaseBase 等关键字的 2 层继承类（如 `MyTest : public MyBase`，
其中 `MyBase : public TestCase`）原本会被错误过滤；现在沿继承链向上
找直到命中框架终端（v3.20.2 与 opengl 提取器 chain_reaches_terminal 对齐）。
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# env 必须干净
for _n in ("INCLUDE", "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
    os.environ.pop(_n, None)


# glue: paths now come from config.py (env-overridable) instead of hardcoding.
from config import SOURCE_ROOT as VK_GL_CTS  # noqa: E402
from config import VK_MODULES_ROOT as _VK_MODROOT  # noqa: E402
# The public RQ2 extractor targets the Vulkan module tree only.
# The source layout places Vulkan cases below .../modules/vulkan.
VK_MODULES_ROOT = _VK_MODROOT / "vulkan"


_TEST_CASE_BASES = ("TestCase", "TestInstance", "SubcaseBase", "TestCaseGroup", "TestNode", "TestPackage")


def _is_terminal_base(name: str) -> bool:
    """Check if a base class NAME contains a framework terminal keyword.

    Loose containment match (vs exact) so e.g. `deqp::TestCase` and
    `tcu::TestInstance` both match the keyword `TestCase` / `TestInstance`.
    """
    if not name:
        return False
    return any(kw in name for kw in _TEST_CASE_BASES)


def _has_test_case_base(base_names) -> bool:
    """v3.20.2 backward-compat: 直系父类匹配即 true（保 1 层链原有行为）。"""
    for b in base_names or []:
        if _is_terminal_base(b):
            return True
    return False


# --- v3.20.2: 2-layer inheritance chain DFS ----------------------

# Build a class-name → base-classes lookup.  Injected by process_vk_cpp
# (one module's worth) so we can resolve `MyTest : public MyBase` chains.
_CLASS_BASE_INDEX: Dict[str, List[str]] = {}


def register_class_base(qname: str, base_names: List[str]) -> None:
    """Index a class (resolved in module) for chain DFS."""
    if qname and base_names is not None:
        _CLASS_BASE_INDEX[qname] = list(base_names)


def clear_class_base_index() -> None:
    """Reset between cpp / module processing."""
    _CLASS_BASE_INDEX.clear()


def _chain_reaches_terminal(class_qn: str,
                            max_depth: int = 8) -> Tuple[bool, int]:
    """DFS up the inheritance chain toward a framework terminal.

    Returns (reached, depth).  depth = number of hops (0 = direct base is terminal).
    Mirrors opengl extractor's chain_reaches_terminal semantics.

    v3.20.2 fix: 此前 VK 处理器只查直系 (`_has_test_case_base`)，2 层链
    `Leaf : Mid : TestCase` 会被错误过滤。现在按链向上找，最深 8 层防环。
    """
    if not class_qn:
        return False, 0
    seen = {class_qn}
    current = class_qn
    for depth in range(max_depth + 1):
        bases = _CLASS_BASE_INDEX.get(current, [])
        for b in bases:
            if _is_terminal_base(b):
                return True, depth
        # advance to any base we know about
        nxt = next((b for b in bases if b in _CLASS_BASE_INDEX and b not in seen), None)
        if nxt is None:
            return False, depth
        seen.add(nxt)
        current = nxt
    return False, max_depth


def _has_test_case_base_chain(qname: str) -> Tuple[bool, int]:
    """v3.20.2: 通过继承链 DFS 判断是否为测试叶子。
    返回 (reached, depth)。depth>=1 表示 2 层链。
    """
    return _chain_reaches_terminal(qname)


def _looks_like_gl_test_class(name: str) -> bool:
    if not name or not name[0].isupper():
        return False
    return any(kw in name for kw in ("Test", "Case", "Group"))


def process_vk_cpp(cpp_path: Path, ast_utils) -> Dict:
    """用 ASTUtils 解析单 cpp，提取所有 class_based / function_based records（已注入 4 字段）。"""
    from injector_pkg.injector import inject_4_fields

    text = cpp_path.read_text(encoding="utf-8", errors="replace")
    analysis = ast_utils._analyze_file(cpp_path)
    real_class_tests: List[Dict] = []
    function_tests: List[Dict] = []

    # v3.20.2: clear & rebuild the class→bases index for this cpp's module
    # so the chain DFS below can resolve 2-layer inheritance like
    # `MyTest : public SomeBase` where SomeBase itself derives from TestCase.
    clear_class_base_index()
    for usr, cls in analysis["classes"].items():
        qn = cls.get("qualified_name") or cls.get("name") or ""
        if qn and cls.get("base_classes"):
            register_class_base(qn, cls["base_classes"])

    # 类基记录：libclang 已经绑定 class + registration
    for usr, cls in analysis["classes"].items():
        base_names = cls.get("base_classes", [])
        qn = cls.get("qualified_name") or cls.get("name") or ""
        # v3.20.2 fix: use inheritance-chain DFS instead of direct base check,
        # so 2-layer leaves (e.g. `Leaf : public Mid : public TestCase`) are not
        # wrongly filtered out. The depth value lets us also emit an explicit
        # `two_layer_inheritance` flag in the record.
        if not base_names:
            continue
        reached, chain_depth = _has_test_case_base_chain(qn)
        if not reached:
            # back-compat: also accept the original direct-base match in case
            # the chain DFS misses (e.g. cross-module parent not in index).
            if not _has_test_case_base(base_names):
                continue
            chain_depth = 0
        regs = analysis.get("class_registrations", {}).get(usr, [])
        if not regs:
            continue
        # 关联 TestInstance
        instance_qname = _find_associated_test_instance_qname(cls, analysis)
        record = {
            "name": cls["name"],
            "qualified_name": qn,
            "namespace": cls.get("namespace", ""),
            "test_type": "class_based",
            "test_level": "class",
            "file_path": str(cpp_path.relative_to(VK_MODULES_ROOT)),
            "code": cls.get("code", ""),
            "base_classes": [{"name": b, "id": "", "relationship": "inheritance"} for b in base_names],
            "associated_test_instance": instance_qname,
            "registrations": regs,
            "two_layer_inheritance": chain_depth >= 1,  # v3.20.2 explicit flag
        }
        real_class_tests.append(record)

    # 函数基记录
    for usr, func in analysis.get("functions", {}).items():
        regs = analysis.get("function_registrations", {}).get(usr, [])
        if not regs:
            continue
        # 只保留 addFunctionCase 家族的 registration。
        # 修复：addChild(createXxx().release()) 这类「类工厂」调用会在
        # _resolve_registered_functions 递归时被误绑到 enclosing 函数体里的
        # 测试函数上（例如 createSemaphoreTests 的 addChild 被塞进 testSemaphoreQueries）。
        # 函数用例只能由 addFunctionCase* 注册；addChild 属于类注册，应留在 class_registrations。
        regs = [r for r in regs if r.get("kind", "").startswith("addFunctionCase")]
        if not regs:
            continue
        record = {
            "name": func["name"],
            "qualified_name": func.get("qualified_name") or func["name"],
            "namespace": func.get("namespace", ""),
            "test_type": "function_based",
            "test_level": "function",
            "file_path": str(cpp_path.relative_to(VK_MODULES_ROOT)),
            "code": func.get("code", ""),
            "base_classes": [],
            "registrations": regs,
        }
        function_tests.append(record)

    all_records = real_class_tests + function_tests
    for r in all_records:
        inject_4_fields(r, cpp_path, ast_utils, side="vk", repo_root=VK_GL_CTS)

    return {
        "cpp": str(cpp_path),
        "side": "vk",
        "category": cpp_path.relative_to(VK_MODULES_ROOT).parts[0],
        "file_basename": cpp_path.stem,
        "class_records": real_class_tests,
        "function_records": function_tests,
    }


def process_gl_cpp(cpp_path: Path, ast_utils) -> Dict:
    """GL cpp：libclang 主路径 + 正则 fallback 抓 addChild（class 多在 .hpp）。"""
    from injector_pkg.injector import inject_4_fields

    analysis = ast_utils._analyze_file(cpp_path)
    real_class_tests: List[Dict] = []
    function_tests: List[Dict] = []

    seen_class_names = set()
    for usr, cls in analysis["classes"].items():
        base_names = cls.get("base_classes", [])
        if not _has_test_case_base(base_names):
            continue
        regs = analysis.get("class_registrations", {}).get(usr, [])
        if not regs:
            continue
        instance_qname = _find_associated_test_instance_qname(cls, analysis)
        record = {
            "name": cls["name"],
            "qualified_name": cls.get("qualified_name") or cls["name"],
            "namespace": cls.get("namespace", ""),
            "test_type": "class_based",
            "test_level": "class",
            "file_path": str(cpp_path.relative_to(GL_MODULES_ROOT)),
            "code": cls.get("code", ""),
            "base_classes": [{"name": b, "id": "", "relationship": "inheritance"} for b in base_names],
            "associated_test_instance": instance_qname,
            "registrations": regs,
        }
        real_class_tests.append(record)
        seen_class_names.add(cls["name"])

    # GL fallback：正则扫 addChild(new ClassName(...))——含命名空间
    text = cpp_path.read_text(encoding="utf-8", errors="replace")
    fallback_pattern = re.compile(r"addChild\s*\(\s*new\s+(?:[\w]+::)*(\w+)\s*\(")
    for m in fallback_pattern.finditer(text):
        cls_name = m.group(1)
        if cls_name in seen_class_names:
            continue
        if cls_name in ("TestSubcase", "TestCase", "TestCaseGroup"):
            continue
        if not _looks_like_gl_test_class(cls_name):
            continue
        line_no = text[:m.start()].count("\n") + 1
        # 提取 namespace
        full_call = text[m.start():m.end()][:200]
        # 从 m.group(0) 提取 ns（如果存在）
        ns_match = re.match(r"addChild\s*\(\s*new\s+([\w]+)::(\w+)\s*\(", m.group(0))
        ns = ns_match.group(1) if ns_match else ""
        regs = [{"kind": "addChild", "line": line_no, "call": full_call}]
        record = {
            "name": cls_name,
            "qualified_name": f"{ns}::{cls_name}" if ns else cls_name,
            "namespace": ns,
            "test_type": "class_based",
            "test_level": "class",
            "file_path": str(cpp_path.relative_to(GL_MODULES_ROOT)),
            "code": f"/* Class {cls_name} defined elsewhere (in .hpp) */",
            "base_classes": [],
            "associated_test_instance": None,
            "registrations": regs,
        }
        real_class_tests.append(record)
        seen_class_names.add(cls_name)

    for usr, func in analysis.get("functions", {}).items():
        regs = analysis.get("function_registrations", {}).get(usr, [])
        if not regs:
            continue
        # 只保留 addFunctionCase 家族（见 process_vk_cpp 同处修复说明）：
        # addChild(createXxx().release()) 类工厂会被误绑到 enclosing 测试函数。
        regs = [r for r in regs if r.get("kind", "").startswith("addFunctionCase")]
        if not regs:
            continue
        record = {
            "name": func["name"],
            "qualified_name": func.get("qualified_name") or func["name"],
            "namespace": func.get("namespace", ""),
            "test_type": "function_based",
            "test_level": "function",
            "file_path": str(cpp_path.relative_to(GL_MODULES_ROOT)),
            "code": func.get("code", ""),
            "base_classes": [],
            "registrations": regs,
        }
        function_tests.append(record)

    all_records = real_class_tests + function_tests
    for r in all_records:
        inject_4_fields(r, cpp_path, ast_utils, side="gl", repo_root=VK_GL_CTS)

    return {
        "cpp": str(cpp_path),
        "side": "gl",
        "category": cpp_path.relative_to(GL_MODULES_ROOT).parts[0],
        "file_basename": cpp_path.stem,
        "class_records": real_class_tests,
        "function_records": function_tests,
    }


def _find_associated_test_instance_qname(cls: Dict, analysis: Dict) -> str | None:
    """找同 cpp 内继承 TestInstance 且被 createInstance() 返回的类（简化版：返回 None，等阶段 4 实现）。"""
    return None
