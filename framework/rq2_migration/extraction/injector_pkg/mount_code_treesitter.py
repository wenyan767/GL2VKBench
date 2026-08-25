"""
mount_code_treesitter.py
========================
用 tree-sitter-cpp 语法树找 target_line 所在函数体。
作为 mount_code_extractor.py 的 fallback（在 libclang 失败时使用）。

API 与 mount_code_extractor.find_enclosing_function_extent 兼容：
  find_enclosing_function_treesitter(cpp_path, target_line) -> (start_line, end_line) | None

为什么用 tree-sitter 而不是纯文本 brace-balance：
  - libclang 在某些复杂 cpp（大型 + 模板深）上死锁
  - 文本 brace-balance 启发式不准（"找最近含(无;行"对 if/switch 误判）
  - tree-sitter 提供精确的 function_definition 节点，10-100x 比 libclang 快

tree-sitter 不做的事（已知 trade-off）：
  - 不展开宏：宏调用如 tcu::TestCaseGroup::CreateXxx 可能判不到函数体
  - 不解析 symbol：不影响 mount_code 抽取（mount_code 只关心语法结构）
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

try:
    from tree_sitter import Language, Parser
    from tree_sitter_cpp import language as cpp_language
    _LANG = Language(cpp_language())
    _PARSER = Parser(_LANG)
    _AVAILABLE = True
except (ImportError, Exception):
    _LANG = None
    _PARSER = None
    _AVAILABLE = False


# 候选函数定义节点类型（cpp grammar 中多种形式）
_FN_NODE_TYPES = {
    "function_definition",          # void foo() { ... }
    "function_template",           # template<...> void foo() { ... }
    "template_declaration",        # 模板声明包裹函数
    "constructor_definition",      # Foo::Foo() { ... }
    "destructor_definition",       # Foo::~Foo() { ... }
}

# 容器节点（递归找 func body 时走过，但自身不是函数定义）
_CONTAINER_TYPES = {
    "translation_unit", "namespace_definition", "declaration_list", "linkage_specification",
    "template_declaration", "function_template", "concept_definition",
    "class_specifier", "struct_specifier", "union_specification",
}


def is_available() -> bool:
    """tree-sitter 是否可用（runtime 时检查）。"""
    return _AVAILABLE


def find_enclosing_function_treesitter(
    cpp_path: Path,
    target_line: int,
) -> Optional[Tuple[int, int]]:
    """返回 target_line 所在最近的 function_definition 的 (start_line, end_line)。

    Args:
        cpp_path: 源 cpp 文件绝对路径
        target_line: 注册点所在行号（1-based）

    Returns:
        (start_line, end_line) — 都是 1-based 行号
        或 None（找不到 / tree-sitter 不可用）
    """
    if not _AVAILABLE or _PARSER is None:
        return None
    try:
        src = cpp_path.read_bytes()
    except Exception:
        return None
    if not src:
        return None

    try:
        tree = _PARSER.parse(src)
    except Exception:
        return None

    root = tree.root_node
    if root is None or root.start_point is None or root.end_point is None:
        return None

    return _find_smallest_covering_fn(root, target_line)


def _find_smallest_covering_fn(node, target_line: int) -> Optional[Tuple[int, int]]:
    """递归找覆盖 target_line 的最小 function_definition 节点。
    策略：找包含 target_line 的最小 function_definition（嵌套 lambda 也覆盖）。
    """
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1

    if not (start_line <= target_line <= end_line):
        return None

    # 当前节点是函数定义？取它
    if node.type in _FN_NODE_TYPES:
        return (start_line, end_line)

    # 递归找子树中更紧的覆盖节点
    for child in node.children:
        if child is None:
            continue
        r = _find_smallest_covering_fn(child, target_line)
        if r is not None:
            return r

    # 当前节点是容器（如 namespace/class body），函数体外
    # 但 target_line 在它的范围内 —— 此时返回 None 因为没有内层函数定义
    # 这种情况 cpp 是全局作用域代码
    return None


# 缓存 parsed tree（同一 cpp 多次调用时复用）
_TREE_CACHE: dict[Tuple[str, int], object] = {}
_TREE_CACHE_LIMIT = 200


def find_enclosing_function_treesitter_cached(
    cpp_path: Path,
    target_line: int,
) -> Optional[Tuple[int, int]]:
    """带 LRU cache 的版本（同一 cpp 多次调用不重新 parse）。

    cache key: (str(cpp_path), len(text)) — 文件改变后 mtime 可能不同，但简化起见
    用 size 做 key 也够（同一 cpp 在同次运行中 size 不会变）。
    """
    if not is_available():
        return None
    try:
        size = cpp_path.stat().st_size
    except Exception:
        return None
    key = (str(cpp_path), size)
    tree = _TREE_CACHE.get(key)
    if tree is None:
        try:
            src = cpp_path.read_bytes()
            tree = _PARSER.parse(src)
        except Exception:
            return None
        # 简单 LRU：超过限制时全清
        if len(_TREE_CACHE) > _TREE_CACHE_LIMIT:
            _TREE_CACHE.clear()
        _TREE_CACHE[key] = tree

    return _find_smallest_covering_fn(tree.root_node, target_line)


# ===== 公开 API：批量 mount_code 抽取 =====
# 一条 cpp 内可能有多条 registration，每条都要找一次 enclosing function。
# 同 cpp 内调用树相同（嵌套优化），但 line 各异。
# 实际用：先 parse cpp 一次，然后对多个 line 复用 parse tree。

def extract_mount_codes_for_cpp(
    cpp_path: Path,
    line_to_reg: dict[int, list[Tuple[dict, dict]]],  # line -> [(record, reg), ...]
) -> dict[int, Tuple[int, int, str]]:
    """对单 cpp 内的多条 registration 批量找 mount_code。

    Args:
        cpp_path: cpp 路径
        line_to_reg: {target_line: [(record_dict, reg_dict), ...]}

    Returns:
        {target_line: (start, end, body)} — body 是 mount_code 内容
        只包含成功找到的 line。
    """
    out = {}
    if not is_available():
        return out
    try:
        src = cpp_path.read_bytes()
    except Exception:
        return out
    if not src:
        return out

    try:
        tree = _PARSER.parse(src)
    except Exception:
        return out

    for target_line in line_to_reg.keys():
        ext = _find_smallest_covering_fn(tree.root_node, target_line)
        if ext is None:
            continue
        start, end = ext
        try:
            text_bytes = src.decode("utf-8", errors="replace")
            lines = text_bytes.splitlines()
            body = "\n".join(lines[start - 1:end])
            out[target_line] = (start, end, body)
        except Exception:
            continue
    return out
