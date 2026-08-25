"""
registration_classifier.py
=========================
根据 libclang call_cursor 把 addChild / addFunctionCase* 调用的 4 大类
（直接构造 / 非模板构造 / 模板构造 / 函数模式）打标签。

判断规则（按优先顺序）：
  1. 函数模式：call_name ∈ {addFunctionCase, addFunctionCaseWithPrograms, addFunctionCaseInNewSubgroup}
  2. 模板构造：call_expr 下第一个 CXX_NEW_EXPR，其 TYPE_REF 是 TEMPLATE_REF
     例: addChild(new LimitCase<GLint>(...)) 或 TestSubcase::Create<X>(...)
  3. 非模板构造：CXX_NEW_EXPR 的 CXXConstructExpr args[0] 是已知测试类名（payload）
     例: addChild(new GroupCase(testCtx, SomeOtherTest, params))
  4. 直接构造：CXX_NEW_EXPR 的 CXXConstructExpr args[0] 是 testCtx / context / m_context 等
     例: addChild(new BlitImageTestCase(testCtx, "name", params))

回退（libclang 拿不到 children 或解析失败）→ "未分类"
"""
from __future__ import annotations

import os
import re
from typing import Optional


# env 必须干净
for name in ("INCLUDE", "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
    os.environ.pop(name, None)


# 直接构造 / 函数模式识别关键字
_DIRECT_CTOR_CONTEXT_NAMES = {
    "testCtx", "context", "m_context", "m_testCtx", "ctx",
    "this", "testContext", "m_ctx", "group", "parent", "m_parent",
}

# 函数模式注册点（call 名本身就是函数模式）
_FUNCTION_MODE_CALLS = {
    "addFunctionCase",
    "addFunctionCaseWithPrograms",
    "addFunctionCaseInNewSubgroup",
}


# ---------- 基于 call 文本的 ast-free 分类（v2.1 7+1 正则复用）----------

# 模板构造: new X<...>(...)   or   类名后跟 <...>
_TEMPLATE_CTOR_RE = re.compile(r"\bnew\s+\w+\s*<")
# TestSubcase::Create<X> 形式（GL 模块特有，addChild(new TestSubcase(ctx, name, TestSubcase::Create<X>))）
_GL_TESTSUBCASE_CREATE_RE = re.compile(r"TestSubcase::Create\s*<")
# 直接构造（花括号初始化）
_BRACE_INIT_RE = re.compile(r"\bnew\s+\w+\s*\{")
# 抽 addChild(new X(first_arg, ...)) 的 first_arg
_FIRST_CTOR_ARG_RE = re.compile(r"\bnew\s+\w+(?:<[^>]*>)?\s*\(\s*(\w+)")
# 抽 addChild(new X(first_arg, second_arg, ...)) 的 second_arg
_SECOND_CTOR_ARG_RE = re.compile(r"\bnew\s+\w+(?:<[^>]*>)?\s*\(\s*[^,\(\)]+,\s*(\*?\s*\w+)")
# 函数模式：call name 本身
_FUNCTION_NAME_RE = re.compile(r"addFunctionCase(?:WithPrograms|InNewSubgroup)?\s*\(")


def _classify_by_call_text(kind: str, call: str) -> str:
    """基于 call 文本的 ast-free 分类。

    v3.5 bug 修复：原 _quick_classify 硬编码 addChild → 直接构造。
    现改为先看 call 文本判 4 大类（模板/非模板/直接构造），仅在没有 call 时回退到默认。
    """
    if not call:
        return _legacy_quick_classify(kind)

    # 函数模式（call 文本里有 addFunctionCase* 形式）
    if _FUNCTION_NAME_RE.search(call):
        return "函数模式"

    # addChild(new X(...)) 系列
    if "addChild" in call:
        # 1. 模板构造：new X<...>(...)  —— 包括 GL TestSubcase::Create<X>
        if _TEMPLATE_CTOR_RE.search(call):
            return "模板构造"
        # 1b. GL TestSubcase::Create<X>(...) 形式
        #   addChild(new TestSubcase(ctx, name, TestSubcase::Create<X>))
        #   上面的 _TEMPLATE_CTOR_RE 不会匹配（因为 outer new 后没有 <）
        if _GL_TESTSUBCASE_CREATE_RE.search(call):
            return "模板构造"
        # 2. 直接构造（花括号初始化）
        if _BRACE_INIT_RE.search(call):
            return "直接构造"
        # 3. 看 args[0]
        m = _FIRST_CTOR_ARG_RE.search(call)
        second_arg_m = _SECOND_CTOR_ARG_RE.search(call)
        if m:
            first = m.group(1)
            if first in _DIRECT_CTOR_CONTEXT_NAMES:
                # 第一参是 ctx（测试上下文）—— 检查第二参是否 Wrapper 的 payload
                # #4 Wrapper 模式：ctx + testCase* + name
                if second_arg_m:
                    second = second_arg_m.group(1).strip().rstrip('*').strip()
                    payload_names = {"testCase", "m_testCase", "caseInst", "m_caseInst",
                                     "testCaseInst", "test", "m_test", "case_impl",
                                     "payload", "m_payload"}
                    # 第二参是 testCase 类指针 → #4 Wrapper
                    if second in payload_names or second.lower() in {n.lower() for n in payload_names}:
                        return "非模板构造"
                    # 第二参是大写开头的 TestCase 类名（payload 类）
                    if (second and second[0].isupper() and any(
                            kw in second for kw in ("Test", "Case", "Group", "Instance"))):
                        return "非模板构造"
                return "直接构造"
            # 第一参已是大写 TestCase 类名 → #4 Wrapper（payload 先行）
            if first and first[0].isupper() and any(
                kw in first for kw in ("Test", "Case", "Group", "Instance", "Subcase")
            ):
                return "非模板构造"
        # 4. 默认：直接构造
        return "直接构造"

    # TestSubcase::Create<X>(...) 作为外部 addChild 时的等价形式（备用）
    if _GL_TESTSUBCASE_CREATE_RE.search(call):
        return "模板构造"

    return "未分类"


def _legacy_quick_classify(kind: str) -> str:
    """原始 _quick_classify 行为（用于 call 文本缺失时回退）。"""
    if kind in _FUNCTION_MODE_CALLS:
        return "函数模式"
    if kind == "TestSubcaseCreate":
        return "模板构造"
    if kind == "addChild":
        return "直接构造"
    return "未分类"


def quick_classify_by_call(kind: str, call: str) -> str:
    """公开 API：基于 call 文本快速分类（不调 libclang）。

    用于 injector._quick_classify 和 reclassify 脚本。
    与 _classify_by_call_text 等价，但作为 module 公共函数导出。
    """
    return _classify_by_call_text(kind, call)


def classify_registration(
    call_cursor,
    call_name: str,
    file_path,
    ast_utils=None,
) -> str:
    """4 大类标签之一。"""
    # 1. 函数模式
    if call_name in _FUNCTION_MODE_CALLS:
        return "函数模式"

    # 2-4: 类注册（addChild）
    # 找第一个 CXX_NEW_EXPR
    new_expr = _find_first_new_expr(call_cursor)
    if new_expr is None:
        # GL TestSubcase::Create<X>(...) 也是模板构造
        if call_name in ("TestSubcaseCreate", "addCases", "newTestCase"):
            return "模板构造"
        return "未分类"

    # 2. 模板构造：CXX_NEW_EXPR 子节点含 TEMPLATE_REF
    for child in new_expr.get_children():
        kind = getattr(child, "kind", None)
        if kind is not None and _is_template_ref(kind):
            return "模板构造"

    # 3-4: 看 CXXConstructExpr 的 args[0]
    args = _get_ctor_first_args(new_expr)
    if args is None:
        return "直接构造"  # 默认 fallback

    arg_name = args.strip()
    if not arg_name:
        return "直接构造"

    # 3. 非模板构造：args[0] 是已知测试类名（payload 类）
    if _looks_like_test_class_name(arg_name):
        return "非模板构造"

    # 4. 直接构造：args[0] 是上下文变量
    if arg_name in _DIRECT_CTOR_CONTEXT_NAMES:
        return "直接构造"

    # 启发式 fallback：args[0] 是 testCtx-like 或单字母/驼峰 → 直接构造
    if arg_name.lower().endswith("ctx") or arg_name.lower().endswith("context"):
        return "直接构造"

    # 默认：直接构造（覆盖大多数情况）
    return "直接构造"


def _find_first_new_expr(cursor):
    """递归找 call_cursor 子树中第一个 CXX_NEW_EXPR。"""
    from clang.cindex import CursorKind
    target_kind = CursorKind.CXX_NEW_EXPR
    # 先看直接子节点
    for child in cursor.get_children():
        if getattr(child, "kind", None) == target_kind:
            return child
    # 再 BFS 一层
    for child in cursor.get_children():
        for sub in child.get_children():
            if getattr(sub, "kind", None) == target_kind:
                return sub
    return None


def _is_template_ref(kind) -> bool:
    from clang.cindex import CursorKind
    return kind in (CursorKind.TEMPLATE_REF,)


def _get_ctor_first_args(new_expr_cursor) -> Optional[str]:
    """取 CXX_NEW_EXPR 第一个 CXXConstructExpr 的 args[0] 的源码/spelling。"""
    from clang.cindex import CursorKind
    ctor_kind = CursorKind.CXX_CONSTRUCT_EXPR
    for child in new_expr_cursor.get_children():
        if getattr(child, "kind", None) != ctor_kind:
            continue
        # 找 DECL_REF_EXPR
        for arg in child.get_children():
            arg_kind = getattr(arg, "kind", None)
            if arg_kind is None:
                continue
            from clang.cindex import CursorKind as CK
            if arg_kind in (CK.DECL_REF_EXPR, CK.MEMBER_REF_EXPR, CK.UNEXPOSED_EXPR):
                # 优先 spelling
                sp = getattr(arg, "spelling", "")
                if sp:
                    return sp
                # 用 displayname
                dn = getattr(arg, "displayname", "")
                if dn:
                    return dn
                # 再退到 type.spelling（unexposed expr 常见）
                t = getattr(arg, "type", None)
                if t:
                    return getattr(t, "spelling", "") or getattr(t, "displayname", "")
            # 未识别 kind：尝试 spell
            sp = getattr(arg, "spelling", "")
            if sp:
                return sp
        # CXXConstructExpr 没 args → 可能是 default ctor → 直接构造
        return None
    return None


def _looks_like_test_class_name(name: str) -> bool:
    """判断 identifier 是否像测试类名（含 Test / Case / Group / Instance / Subcase 等）。"""
    if not name or not name[0].isupper():
        return False
    keywords = ("Test", "Case", "Group", "Instance", "Subcase", "Tests", "Suite")
    for kw in keywords:
        if kw in name:
            return True
    return False