"""GL2VKFrame 核心库：数据加载、代码清洗、结构分类、模板组装、结构校验。

由原工程 9 个模块机械合并而成（execution_context / input_loader /
template_manager / source_cleaner / code_classifier / structure_validator /
feedback_schema / build_lock / prepare_agent_case），保留各自的真实实现，
只做三类改动：

  * 路径重锚到 paths.py，删除全部 `parents[3]` 仓库根依赖。
  * 删除跨种子静默回退（曾致 5 例「产物齐全、无报错、方法学错误」）。
  * 删除 stage2 LLM 池专用的 llm_channel / llm_worker_index override。

合并时有两个符号在两个源模块里重名，处置如下：
  * FREE_TEST_FUNC_PATTERN —— 保留 code_classifier 版（带 name/tail 命名组，
    分类逻辑取 .group("name") 依赖它）；删除 structure_validator 版（不带组的
    宽松探测，仅用于 bool 判断，被前者完全覆盖）。
  * REPO_ROOT —— 两份定义全删，改为从 paths.py 取。
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from typing import Iterator
import argparse
import json
import os
import random
import re
import time

from paths import (
    DEFAULT_RUNS_DIR,
    TEMPLATE_DIR,
    seed_case_dir,
)

# ========================================================================
# 源自 execution_context.py
# ========================================================================

_OVERRIDES: dict[str, Any] = {
    "package_name": None,
    "cts_root": None,
    "config_dir": None,
    "output_root": None,
    "run_record_path": None,
    "row_ids": None,
    "gl_sim_vk_path": None,
    "candidate_root": None,
    "defer_compile": False,
    "run_after_compile": False,
    "runtime_timeout_seconds": 300,
}

def set_execution_overrides(
    *,
    package_name: str | None = None,
    cts_root: str | Path | None = None,
    config_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    run_record_path: str | Path | None = None,
    row_ids: list[int] | tuple[int, ...] | set[int] | None = None,
    gl_sim_vk_path: str | Path | None = None,
    candidate_root: str | Path | None = None,
    defer_compile: bool = False,
    run_after_compile: bool = False,
    runtime_timeout_seconds: int = 300,
) -> None:
    _OVERRIDES["package_name"] = package_name or None
    _OVERRIDES["cts_root"] = Path(cts_root) if cts_root else None
    _OVERRIDES["config_dir"] = Path(config_dir) if config_dir else None
    _OVERRIDES["output_root"] = Path(output_root) if output_root else None
    _OVERRIDES["run_record_path"] = Path(run_record_path) if run_record_path else None
    _OVERRIDES["row_ids"] = sorted(int(v) for v in row_ids) if row_ids else None
    _OVERRIDES["gl_sim_vk_path"] = Path(gl_sim_vk_path) if gl_sim_vk_path else None
    _OVERRIDES["candidate_root"] = Path(candidate_root) if candidate_root else None
    _OVERRIDES["defer_compile"] = bool(defer_compile)
    _OVERRIDES["run_after_compile"] = bool(run_after_compile)
    _OVERRIDES["runtime_timeout_seconds"] = max(1, int(runtime_timeout_seconds))

def clear_execution_overrides() -> None:
    set_execution_overrides()

def get_execution_overrides() -> dict[str, Any]:
    return dict(_OVERRIDES)

def resolve_package_name(mode_name: str, model_name: str) -> str:
    package_name = _OVERRIDES.get("package_name")
    if package_name:
        return str(package_name)
    return f"{mode_name.replace('-', '_')}_{model_name}"

def resolve_target_name(mode_name: str, model_name: str) -> str:
    return f"deqp-vk-{resolve_package_name(mode_name, model_name).replace('_', '-')}"

def resolve_runtime_root_group(mode_name: str, model_name: str) -> str:
    # dEQP case path 使用 vktTestPackage.cpp 传给 createTests() 的组名；模块 target
    # 使用连字符，但运行时 root group 与 C++ namespace/package 一致，保留下划线。
    return resolve_package_name(mode_name, model_name)

def resolve_cts_root_override() -> Path | None:
    value = _OVERRIDES.get("cts_root")
    return Path(value) if value else None

def resolve_config_dir_override() -> Path | None:
    value = _OVERRIDES.get("config_dir")
    return Path(value) if value else None

def resolve_output_root_override() -> Path | None:
    value = _OVERRIDES.get("output_root")
    return Path(value) if value else None

def resolve_run_record_path_override() -> Path | None:
    value = _OVERRIDES.get("run_record_path")
    return Path(value) if value else None

def resolve_row_ids_override() -> list[int] | None:
    value = _OVERRIDES.get("row_ids")
    return list(value) if value else None

def resolve_gl_sim_vk_path_override() -> Path | None:
    value = _OVERRIDES.get("gl_sim_vk_path")
    return Path(value) if value else None

def resolve_candidate_root_override() -> Path | None:
    value = _OVERRIDES.get("candidate_root")
    return Path(value) if value else None

def resolve_defer_compile() -> bool:
    return bool(_OVERRIDES.get("defer_compile", False))

def resolve_run_after_compile() -> bool:
    return bool(_OVERRIDES.get("run_after_compile", False))

def resolve_runtime_timeout_seconds() -> int:
    return max(1, int(_OVERRIDES.get("runtime_timeout_seconds", 300)))


# ========================================================================
# 源自 input_loader.py
# ========================================================================


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def resolve_gl_sim_vk_path() -> Path:
    override = resolve_gl_sim_vk_path_override()
    if override is not None:
        return override
    # 环境变量仅作为独立入口的显式外部配置；公开包不携带固定输入。
    env_path = os.environ.get("AGENT_GL_SIM_VK_PATH")
    if env_path:
        return Path(env_path)
    raise ValueError(
        "Framework input is required; pass --input-json/--gl-sim-vk-path "
        "or set AGENT_GL_SIM_VK_PATH"
    )

def load_gl_sim_vk_row(row_id: int) -> dict[str, Any]:
    gl_sim_vk_path = resolve_gl_sim_vk_path()
    rows = load_json(gl_sim_vk_path)
    for row in rows:
        if int(row["row_id"]) == int(row_id):
            return row
    raise ValueError(f"row_id={row_id} not found in {gl_sim_vk_path}")

def resolve_seed_kind(mode_name: str) -> str:
    """把调用方传的 mode 名归一成种子种类。

    调用方有两种传法：Step 2 直接传种子名（non_cot / cot），Step 3 传方法名
    （methods.Method 的 key / slug / legacy 任一种）。两者都要落到同一份种子上 ——
    「同一份种子只编一次、判据共用」这条就靠这里成立。

    未知名字直接抛错。原实现在这里有 mode→目录映射加环境变量兜底，找不到就换
    另一种种子，产出过 5 例「文件齐全、没有报错、方法学错误」的结果。
    """
    from methods import METHODS, SEED_DISPLAY, SEED_KINDS, SEED_ALIAS

    if mode_name in SEED_KINDS:
        return mode_name
    for seed_kind, display_name in SEED_DISPLAY.items():
        if mode_name in (display_name, SEED_ALIAS.get(seed_kind)):
            return seed_kind
    if mode_name == "non-cot":
        return "non_cot"
    for method in METHODS.values():
        if mode_name in (method.key, method.slug, method.legacy, method.display):
            return method.seed
    known = list(SEED_KINDS) + sorted(
        {name for m in METHODS.values() for name in (m.key, m.slug, m.legacy)})
    raise ValueError(
        f"无法判定 {mode_name!r} 该用哪条种子线；已知名字: {', '.join(known)}。\n"
        f"本包不猜测种子来源：猜错会拿另一条线的种子去修，结果看起来齐全但方法学是错的。")


def resolve_candidate_case_dir(mode_name: str, model_name: str, row_id: int) -> Path:
    """种子目录。mode_name 可以是种子名，也可以是方法名（见 resolve_seed_kind）。

    与原实现的区别：没有 mode→目录映射，也没有环境变量兜底。种子位置只由
    产物根 + seed_kind 决定，换根用 --output-root。
    """
    override = resolve_candidate_root_override()
    if override is not None:
        return override / model_name / f"case_{row_id}"
    return seed_case_dir(resolve_seed_kind(mode_name), model_name, row_id)


def resolve_candidate_paths(mode_name: str, model_name: str, row_id: int) -> tuple[Path, Path]:
    """返回 (种子目录, 种子 cpp 路径)。不做任何跨种子回退。

    mode_name 可以是种子名，也可以是方法名（见 resolve_seed_kind）。

    原实现在 only-fix 找不到 cot 种子时会回退到 non-cot 种子目录，产出「文件
    齐全、没有报错、方法学错误」的结果，实测发生 5 例。这里不回退：缺就是缺，
    由 load_initial_candidate 抛错。
    """
    case_dir = resolve_candidate_case_dir(mode_name, model_name, row_id)
    return case_dir, case_dir / "raw_generated.cpp"


def load_initial_candidate(mode_name: str, model_name: str, row_id: int) -> dict[str, Any]:
    case_dir, generated_cpp = resolve_candidate_paths(mode_name, model_name, row_id)

    if not generated_cpp.exists():
        raise FileNotFoundError(
            f"缺少种子文件: {generated_cpp}\n"
            f"请先跑 Step 1 生成 {resolve_seed_kind(mode_name)} 种子；"
            f"本包不会回退到另一种种子。")

    return {
        "source_case_dir": str(case_dir),
        "generated_cpp_path": str(generated_cpp),
        "generated_cpp": generated_cpp.read_text(encoding="utf-8"),
    }


# ========================================================================
# 源自 template_manager.py
# ========================================================================




def _read_template(name: str) -> str:
    return (TEMPLATE_DIR / name).read_text(encoding="utf-8")

def _sanitize_leaf_name(name: str | None, fallback: str) -> str:
    if not isinstance(name, str):
        return fallback
    cleaned = name.strip().replace(".", "_").replace("-", "_")
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned.replace(" ", "_")
    return cleaned or fallback

def _build_class_register_body(info: ClassifiedCode) -> str:
    lines: list[str] = []
    register_cases = info.register_cases or []
    if register_cases and info.params_type:
        for index, case in enumerate(register_cases, start=1):
            var_name = f"params_{index}"
            case_name = _sanitize_leaf_name(case.get("case_name"), info.case_name)
            params_init = case.get("params_init", "{}")
            lines.append(f"    const {info.params_type} {var_name} = {params_init};")
            lines.append(
                f'    group->addChild(new {info.class_name}(group->getTestContext(), "{case_name}", {var_name}));'
            )
            if index != len(register_cases):
                lines.append("")
        return "\n".join(lines)

    if info.has_params and info.params_type:
        lines.append(f"    const {info.params_type} params = {{}};")
        lines.append("")
        lines.append(
            f'    group->addChild(new {info.class_name}(group->getTestContext(), "{info.case_name}", params));'
        )
    elif info.class_ctor_arity == 2:
        lines.append(
            f'    group->addChild(new {info.class_name}(group->getTestContext(), "{info.case_name}"));'
        )
    else:
        lines.append(
            f'    group->addChild(new {info.class_name}(group->getTestContext(), "{info.case_name}", {{}}));'
        )
    return "\n".join(lines)

def _function_params_init(info: ClassifiedCode) -> str:
    """function 的参数初值。

    只取 register_cases 第一项的 params_init —— function 恒定挂一个叶子, 所以多余项
    没有落点; 每项的 case_name 也被忽略, 叶子名一律用脚本的 info.case_name, 因为
    stage3 按 case_{row_id} 拼测试路径, 放模型自选名会导致跑 0 个 test。
    多叶子 function 要等 stage3 的叶子名口径定了再开。
    """
    for case in info.register_cases or []:
        if not isinstance(case, dict):
            continue
        params_init = str(case.get("params_init") or "").strip()
        if params_init:
            return params_init
    return "{}"

def _build_function_register_body(info: ClassifiedCode) -> str:
    lines: list[str] = []
    if info.has_params and info.params_type:
        lines.append(f"    const {info.params_type} params = {_function_params_init(info)};")
        lines.append("")

    if info.register_style == "addFunctionCaseWithPrograms":
        if info.has_params and info.has_check_support:
            call = (
                f'    addFunctionCaseWithPrograms<{info.params_type}>'
                f'(group, "{info.case_name}", { _check_support_name(info) }, { _init_programs_name(info) }, '
                f"{info.function_name}, params);"
            )
        elif info.has_params:
            call = (
                f'    addFunctionCaseWithPrograms<{info.params_type}>'
                f'(group, "{info.case_name}", { _init_programs_name(info) }, {info.function_name}, params);'
            )
        elif info.has_check_support:
            call = (
                f'    addFunctionCaseWithPrograms(group, "{info.case_name}", '
                f"{ _check_support_name(info) }, { _init_programs_name(info) }, {info.function_name});"
            )
        else:
            call = (
                f'    addFunctionCaseWithPrograms(group, "{info.case_name}", '
                f"{ _init_programs_name(info) }, {info.function_name});"
            )
    else:
        if info.has_params and info.has_check_support:
            call = (
                f'    addFunctionCase<{info.params_type}>'
                f'(group, "{info.case_name}", { _check_support_name(info) }, {info.function_name}, params);'
            )
        elif info.has_params:
            call = (
                f'    addFunctionCase<{info.params_type}>'
                f'(group, "{info.case_name}", {info.function_name}, params);'
            )
        elif info.has_check_support:
            call = (
                f'    addFunctionCase(group, "{info.case_name}", '
                f"{ _check_support_name(info) }, {info.function_name});"
            )
        else:
            call = f'    addFunctionCase(group, "{info.case_name}", {info.function_name});'

    lines.append(call)
    return "\n".join(lines)

def _check_support_name(info: ClassifiedCode) -> str:
    return _find_named_symbol(info.code, "checkSupport") or "checkSupport"

def _init_programs_name(info: ClassifiedCode) -> str:
    return _find_named_symbol(info.code, "initPrograms") or "initPrograms"

def _find_named_symbol(code: str, needle: str) -> str | None:
    import re

    match = re.search(rf"\b(\w*{needle}\w*)\s*\(", code)
    return match.group(1) if match else None

def assemble_cpp(info: ClassifiedCode) -> str:
    if info.template_type == "class":
        template = _read_template("class_case.cpp.tpl")
        rendered = template.replace("{{CLASS_CODE}}", info.code)
        rendered = rendered.replace("{{REGISTER_NAME}}", info.register_name)
        rendered = rendered.replace("{{REGISTER_BODY}}", _build_class_register_body(info))
        return rendered

    template = _read_template("function_case.cpp.tpl")
    rendered = template.replace(
        "{{CHECK_SUPPORT_BLOCK}}",
        "" if not info.has_check_support else "",
    )
    rendered = rendered.replace(
        "{{INIT_PROGRAMS_BLOCK}}",
        "" if not info.has_init_programs else "",
    )
    rendered = rendered.replace("{{FUNCTION_CODE}}", info.code)
    rendered = rendered.replace("{{REGISTER_NAME}}", info.register_name)
    rendered = rendered.replace("{{REGISTER_BODY}}", _build_function_register_body(info))
    return rendered


# ========================================================================
# 源自 source_cleaner.py
# ========================================================================

INCLUDE_LINE_PATTERN = re.compile(r"^\s*#\s*include\s+[<\"].*[>\"]\s*$", re.MULTILINE)

USING_NAMESPACE_PATTERN = re.compile(r"^\s*using\s+namespace\s+.+?;\s*$", re.MULTILINE)

CODE_FENCE_PATTERN = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*$", re.MULTILINE)

def clean_generated_code(raw_code: str, *, decode_escaped_newlines: bool = True) -> dict[str, object]:
    code = raw_code.replace("\r\n", "\n")
    code = _extract_embedded_code_payload(code)
    if decode_escaped_newlines:
        code = _decode_common_escaped_newlines(code)
    code = _strip_code_fences(code)
    # Note: _fix_string_literal_newlines is not needed if decode_escaped_newlines=False

    removed_include_count = len(INCLUDE_LINE_PATTERN.findall(code))
    removed_using_namespace_count = len(USING_NAMESPACE_PATTERN.findall(code))

    code = INCLUDE_LINE_PATTERN.sub("", code)
    code = USING_NAMESPACE_PATTERN.sub("", code)
    code = _strip_leading_explanatory_text(code)
    code, removed_namespace_layers = _strip_outer_namespaces(code)
    code = _collapse_blank_lines(code).strip()

    return {
        "cleaned_code": code,
        "removed_include_count": removed_include_count,
        "removed_using_namespace_count": removed_using_namespace_count,
        "removed_namespace_layers": removed_namespace_layers,
    }

def _strip_code_fences(code: str) -> str:
    return CODE_FENCE_PATTERN.sub("", code)

def _extract_embedded_code_payload(code: str) -> str:
    stripped = code.strip()
    if not stripped:
        return code

    # Some upstream outputs wrap the actual C++ body in a JSON object and add
    # long explanatory prose before/after it. Extract the first complete JSON
    # object first, then decode only that fragment.
    json_text = _extract_first_json_object_text(stripped)
    if json_text:
        try:
            payload = json.loads(json_text)
            if isinstance(payload, dict):
                for key in ("vulkan_test_class_code", "class_code", "function_code", "code"):
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        return value
        except Exception:
            pass

    for key in ("vulkan_test_class_code", "class_code", "function_code", "code"):
        value = _extract_string_field(code, key)
        if value.strip():
            return value

    return code

def _extract_first_json_object_text(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None

def _extract_string_field(text: str, key: str) -> str:
    marker = f'"{key}"'
    key_index = text.find(marker)
    if key_index == -1:
        return ""

    colon_index = text.find(":", key_index + len(marker))
    if colon_index == -1:
        return ""

    value_start = _find_next_non_whitespace(text, colon_index + 1)
    if value_start == -1 or text[value_start] != '"':
        return ""

    chars: list[str] = []
    escape = False
    for index in range(value_start + 1, len(text)):
        ch = text[index]
        if escape:
            chars.append("\\" + ch)
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            return "".join(chars)
        chars.append(ch)
    return "".join(chars)

def _find_next_non_whitespace(text: str, start: int) -> int:
    for index in range(start, len(text)):
        if not text[index].isspace():
            return index
    return -1

def _decode_common_escaped_newlines(code: str) -> str:
    if "\\n" not in code and "\\t" not in code and '\\"' not in code:
        return code

    # Decode common escaped formatting without trying to interpret arbitrary text
    # as a full string literal.
    decoded = (
        code.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )
    return decoded

def _strip_leading_explanatory_text(code: str) -> str:
    lines = code.splitlines()
    start_index = 0
    starter_patterns = (
        "struct ",
        "class ",
        "enum ",
        "template<",
        "template <",
        "static ",
        "inline ",
        "tcu::TestStatus ",
        "void ",
        "bool ",
        "int ",
        "namespace ",
        "typedef ",
        "using ",
        "{",
    )
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('"') and index + 1 < len(lines):
            continue
        if stripped.startswith(starter_patterns):
            start_index = index
            break
    return "\n".join(lines[start_index:])

def _strip_outer_namespaces(code: str) -> tuple[str, int]:
    stripped = code.strip()
    removed = 0
    while True:
        match = re.match(
            r"^\s*namespace\s+\w+\s*\{\s*(?P<body>.*)\s*\}\s*(?://.*)?\s*$",
            stripped,
            re.DOTALL,
        )
        if not match:
            break
        body = match.group("body").strip()
        if not body:
            break
        stripped = body
        removed += 1
    return stripped, removed

def _collapse_blank_lines(code: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", code)

def _fix_string_literal_newlines(code: str) -> str:
    """
    Fix bare newlines in C++ string literals and adjacent string concatenation.

    Patterns like:
        "text
    "

    Are converted to:
        "text\\n"

    And patterns like:
        "vec" "4"

    Are converted to:
        "vec4"

    This handles common errors where string literals contain actual newlines
    instead of \\n escape sequences, or where strings are incorrectly split.
    """
    # First, fix adjacent string literals (e.g., "vec" "4" -> "vec4")
    # Pattern: quote, content, quote, optional whitespace, quote, content, quote
    adjacent_pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"\s+"([^"\\]*(?:\\.[^"\\]*)*)"'

    def merge_adjacent(match):
        first = match.group(1)
        second = match.group(2)
        return f'"{first}{second}"'

    # Apply multiple times to handle chains like "a" "b" "c"
    prev_code = None
    max_iterations = 10
    iteration = 0

    while prev_code != code and iteration < max_iterations:
        prev_code = code
        code = re.sub(adjacent_pattern, merge_adjacent, code)
        iteration += 1

    # Second, fix bare newlines in string literals
    # Pattern: quote, content, newline, optional whitespace, quote
    newline_pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)\n\s*"'

    def fix_newline(match):
        content = match.group(1)
        # Add \n escape sequence and close/reopen the string
        return f'"{content}\\n"\n"'

    # Apply the fix multiple times to handle nested cases
    prev_code = None
    iteration = 0

    while prev_code != code and iteration < max_iterations:
        prev_code = code
        code = re.sub(newline_pattern, fix_newline, code)
        iteration += 1

    return code


# ========================================================================
# 源自 code_classifier.py
# ========================================================================

CLASS_CASE_PATTERN = re.compile(
    r"class\s+(?P<name>\w+)\s*:\s*public\s+(?:\w+::)?TestCase\b"
)

GENERIC_CLASS_PATTERN = re.compile(
    r"class\s+(?P<name>\w+)\s*:\s*public\s+(?P<base>[\w:]+)\b"
)

INSTANCE_CASE_PATTERN = re.compile(
    r"class\s+(?P<name>\w+)\s*:\s*public\s+(?:\w+::)?TestInstance\b"
)

GENERIC_INSTANCE_PATTERN = re.compile(
    r"class\s+(?P<name>\w+)\s*:\s*public\s+(?P<base>[\w:]*Instance)\b"
)

FREE_TEST_FUNC_PATTERN = re.compile(
    r"tcu::TestStatus\s+(?P<name>\w+)\s*\(\s*(?:const\s+)?(?:vkt::)?Context\s*&\s*\w+(?P<tail>[^)]*)\)",
    re.MULTILINE,
)

GENERIC_FREE_FUNCTION_PATTERN = re.compile(
    r"(?<!if\s)(?<!for\s)(?<!while\s)(?<!switch\s)(?<!return\s)"
    r"\b(?P<return_type>[\w:<>~*&\s]+?)\s+"
    r"(?P<name>\w+)\s*\((?P<params>[^;{}()]*(?:\([^)]*\)[^;{}()]*)*)\)\s*\{",
    re.MULTILINE,
)

FREE_CHECK_SUPPORT_PATTERN = re.compile(
    r"void\s+(?P<name>\w*checkSupport\w*)\s*\(",
    re.MULTILINE,
)

FREE_INIT_PROGRAMS_PATTERN = re.compile(
    r"void\s+(?P<name>\w*initPrograms\w*)\s*\(",
    re.MULTILINE,
)

STRUCT_PATTERN = re.compile(r"struct\s+(?P<name>\w+)\s*\{", re.MULTILINE)

PARAM_COMMENT_PATTERN = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)

CTOR_PATTERN = (
    r"{class_name}\s*\(\s*tcu::TestContext\s*&\s*\w+\s*,\s*"
    r"(?:const\s+)?std::string\s*&?\s*\w+(?P<tail>[^)]*)\)"
)

@dataclass
class ClassifiedCode:
    template_type: str
    class_name: str | None
    instance_name: str | None
    function_name: str | None
    register_name: str
    case_name: str
    has_check_support: bool
    has_init_programs: bool
    has_params: bool
    params_type: str | None
    class_ctor_arity: int | None
    register_style: str
    register_cases: list[dict] | None
    code: str

    def to_dict(self) -> dict:
        return asdict(self)

def infer_class_name_from_member_definitions(code: str) -> str | None:
    candidates = re.findall(r"\b([A-Za-z_]\w*)::[A-Za-z_]\w*\s*\(", code)
    if not candidates:
        return None
    for candidate in candidates:
        if candidate not in {"vk", "vkt", "tcu", "glu", "std", "de"}:
            return candidate
    return candidates[0]

def infer_function_name_from_body(code: str) -> str | None:
    strict_match = re.search(
        r"\btcu::TestStatus\s+(?P<name>\w+)\s*\((?P<params>[^;{}()]*(?:\([^)]*\)[^;{}()]*)*)\)",
        code,
        re.MULTILINE,
    )
    if strict_match:
        return strict_match.group("name")

    relaxed_match = re.search(
        r"(?<!if\s)(?<!for\s)(?<!while\s)(?<!switch\s)(?<!return\s)"
        r"\b[\w:<>~*&\s]+?\s+(?P<name>\w+)\s*\((?P<params>[^;{}()]*(?:\([^)]*\)[^;{}()]*)*)\)",
        code,
        re.MULTILINE,
    )
    if relaxed_match:
        candidate = relaxed_match.group("name")
        if candidate not in {"if", "for", "while", "switch", "return"}:
            return candidate
    return None

def _extract_params_type_from_function_signature(code: str, function_name: str) -> str | None:
    pattern = re.compile(
        rf"tcu::TestStatus\s+{re.escape(function_name)}\s*\(\s*(?:const\s+)?(?:vkt::)?Context\s*&\s*\w+\s*,\s*(?P<params>[^)]+)\)"
    )
    match = pattern.search(code)
    if not match:
        return None
    params_text = match.group("params").strip()
    # Only a single trailing params object fits the registration contract; a
    # multi-parameter signature would otherwise yield a truncated type.
    if len(_split_top_level_params(params_text)) != 1:
        return None
    params_text = _strip_param_name(params_text)
    return _normalize_params_type(params_text)

def _extract_params_type_from_generic_function_signature(code: str, function_name: str) -> str | None:
    pattern = re.compile(
        rf"\b[\w:<>~*&\s]+?\s+{re.escape(function_name)}\s*\((?P<params>[^;{{}}()]*(?:\([^)]*\)[^;{{}}()]*)*)\)\s*\{{",
        re.MULTILINE,
    )
    match = pattern.search(code)
    if not match:
        return None
    params_text = match.group("params").strip()
    if not params_text:
        return None
    top_level_params = _split_top_level_params(params_text)
    if len(top_level_params) != 2:
        return None
    tail = top_level_params[1].strip()
    if not tail:
        return None
    return _normalize_params_type(_strip_param_name(tail))

def _extract_params_type_from_ctor(code: str, class_name: str) -> tuple[str | None, int]:
    ctor_pattern = re.compile(CTOR_PATTERN.format(class_name=re.escape(class_name)))
    match = ctor_pattern.search(code)
    if not match:
        return None, 2
    tail = match.group("tail").strip()
    if not tail:
        return None, 2
    tail = tail.lstrip(",").strip()
    if not tail:
        return None, 2
    if len(_split_top_level_params(tail)) > 1:
        # Constructor-heavy CTS framework subclasses are still useful to compile,
        # but they do not fit the single-params-object registration contract.
        # Let them pass as class candidates and surface the real constructor mismatch
        # during compile/feedback instead of failing early in prepare/classify.
        return None, 2
    params_text = _strip_param_name(tail)
    return _normalize_params_type(params_text), 3

def _strip_param_name(params_text: str) -> str:
    text = PARAM_COMMENT_PATTERN.sub(" ", params_text).strip()
    if not text:
        return text

    # Remove a trailing default value without disturbing the type.
    default_index = _find_top_level_char(text, "=")
    if default_index != -1:
        text = text[:default_index].rstrip()

    candidate = text.rstrip()
    if not candidate:
        return candidate

    # Remove a trailing identifier only when there is clear evidence it is a variable name.
    # The separator may include `&`/`*` glued to the name (`Foo &params`, `Foo *params`),
    # which is why they are part of the separator class rather than the type.
    match = re.search(
        r"^(?P<type>.*?)(?P<sep>[\s&*]*\s[\s&*]*|[&*]+)(?P<name>[A-Za-z_]\w*)$", candidate
    )
    if not match:
        return candidate

    type_part = match.group("type").rstrip()
    name_part = match.group("name")
    if not type_part:
        return candidate

    # A trailing builtin keyword is part of a multi-word type (`unsigned int`,
    # `long long`), never a parameter name.
    builtin_type_words = {
        "int",
        "char",
        "short",
        "long",
        "float",
        "double",
        "bool",
        "void",
        "signed",
        "unsigned",
        "wchar_t",
    }
    if name_part in builtin_type_words:
        return candidate

    # These tokens can never stand alone as a type, so keep the text intact rather
    # than reducing `struct Foo` to `struct`.
    dangling_type_tokens = {"const", "volatile", "struct", "class", "enum", "union"}
    if type_part.split()[-1] in dangling_type_tokens:
        return candidate

    # Everything reaching here is a parameter declaration, so a trailing identifier
    # after a complete type is always the parameter name and must be dropped.
    return type_part

def _find_top_level_char(text: str, target: str) -> int:
    depth_angle = 0
    depth_paren = 0
    depth_brace = 0
    depth_bracket = 0
    for index, char in enumerate(text):
        if char == "<":
            depth_angle += 1
        elif char == ">":
            depth_angle = max(0, depth_angle - 1)
        elif char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren = max(0, depth_paren - 1)
        elif char == "{":
            depth_brace += 1
        elif char == "}":
            depth_brace = max(0, depth_brace - 1)
        elif char == "[":
            depth_bracket += 1
        elif char == "]":
            depth_bracket = max(0, depth_bracket - 1)
        elif (
            char == target
            and depth_angle == 0
            and depth_paren == 0
            and depth_brace == 0
            and depth_bracket == 0
        ):
            return index
    return -1

def _split_top_level_params(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth_angle = 0
    depth_paren = 0
    depth_brace = 0
    depth_bracket = 0

    for index, char in enumerate(text):
        if char == "<":
            depth_angle += 1
        elif char == ">":
            depth_angle = max(0, depth_angle - 1)
        elif char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren = max(0, depth_paren - 1)
        elif char == "{":
            depth_brace += 1
        elif char == "}":
            depth_brace = max(0, depth_brace - 1)
        elif char == "[":
            depth_bracket += 1
        elif char == "]":
            depth_bracket = max(0, depth_bracket - 1)
        elif (
            char == ","
            and depth_angle == 0
            and depth_paren == 0
            and depth_brace == 0
            and depth_bracket == 0
        ):
            part = text[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1

    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts

def _guess_params_type_from_struct(code: str) -> str | None:
    for match in STRUCT_PATTERN.finditer(code):
        name = match.group("name")
        if "param" in name.lower() or "case" in name.lower():
            return name
    return None

def _normalize_params_type(params_type: str | None) -> str | None:
    if not params_type:
        return None
    normalized = params_type.strip()
    normalized = re.sub(r"^\s*const\s+", "", normalized)
    normalized = re.sub(r"\s*[&*]\s*$", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or None

def _build_case_name(row_id: int) -> str:
    return f"case_{row_id}"

def classify_candidate_code(code: str, mode_name: str, model_name: str, row_id: int) -> ClassifiedCode:
    register_name = f"register_{mode_name.replace('-', '_')}_{model_name}_{row_id}"
    case_name = _build_case_name(row_id)

    class_match = CLASS_CASE_PATTERN.search(code)
    instance_match = INSTANCE_CASE_PATTERN.search(code)
    if not class_match:
        generic_class_match = GENERIC_CLASS_PATTERN.search(code)
        generic_instance_match = GENERIC_INSTANCE_PATTERN.search(code)
        if generic_class_match:
            class_match = generic_class_match
            if not instance_match:
                instance_match = generic_instance_match
    if not class_match:
        inferred_class_name = infer_class_name_from_member_definitions(code)
        if inferred_class_name:
            params_type, ctor_arity = _extract_params_type_from_ctor(code, inferred_class_name)
            if not params_type:
                params_type = _guess_params_type_from_struct(code)
            return ClassifiedCode(
                template_type="class",
                class_name=inferred_class_name,
                instance_name=instance_match.group("name") if instance_match else None,
                function_name=None,
                register_name=register_name,
                case_name=case_name,
                has_check_support="checkSupport" in code,
                has_init_programs="initPrograms" in code,
                has_params=params_type is not None,
                params_type=params_type,
                class_ctor_arity=ctor_arity,
                register_style="addChild",
                register_cases=None,
                code=code.strip(),
            )
    if class_match:
        class_name = class_match.group("name")
        instance_name = instance_match.group("name") if instance_match else None
        params_type, ctor_arity = _extract_params_type_from_ctor(code, class_name)
        if not params_type:
            params_type = _guess_params_type_from_struct(code)
        return ClassifiedCode(
            template_type="class",
            class_name=class_name,
            instance_name=instance_name,
            function_name=None,
            register_name=register_name,
            case_name=case_name,
            has_check_support="checkSupport" in code,
            has_init_programs="initPrograms" in code,
            has_params=params_type is not None,
            params_type=params_type,
            class_ctor_arity=ctor_arity,
            register_style="addChild",
            register_cases=None,
            code=code.strip(),
        )

    function_match = FREE_TEST_FUNC_PATTERN.search(code)
    function_name = function_match.group("name") if function_match else None
    if function_name is None:
        generic_function_match = GENERIC_FREE_FUNCTION_PATTERN.search(code)
        function_name = generic_function_match.group("name") if generic_function_match else None
    if function_name is None:
        function_name = infer_function_name_from_body(code)
    if function_name is None:
        raise ValueError("unable to classify candidate as class or function")

    params_type = _extract_params_type_from_function_signature(code, function_name)
    if not params_type:
        params_type = _extract_params_type_from_generic_function_signature(code, function_name)
    if not params_type:
        params_type = _guess_params_type_from_struct(code)

    has_check_support = FREE_CHECK_SUPPORT_PATTERN.search(code) is not None
    has_init_programs = FREE_INIT_PROGRAMS_PATTERN.search(code) is not None
    register_style = "addFunctionCaseWithPrograms" if has_init_programs else "addFunctionCase"

    return ClassifiedCode(
        template_type="function",
        class_name=None,
        instance_name=None,
        function_name=function_name,
        register_name=register_name,
        case_name=case_name,
        has_check_support=has_check_support,
        has_init_programs=has_init_programs,
        has_params=params_type is not None,
        params_type=params_type,
        class_ctor_arity=None,
        register_style=register_style,
        register_cases=None,
        code=code.strip(),
    )


# ========================================================================
# 源自 structure_validator.py
# ========================================================================

INCLUDE_PATTERN = re.compile(r"^\s*#\s*include\b", re.MULTILINE)

NAMESPACE_PATTERN = re.compile(r"^\s*namespace\s+\w+\s*\{", re.MULTILINE)

FENCE_PATTERN = re.compile(r"```")

REGISTER_PATTERN = re.compile(r"\b(addChild|addFunctionCase(?:WithPrograms)?)\s*\(")

TESTCASE_CLASS_PATTERN = re.compile(r"class\s+\w+\s*:\s*public\s+(?:\w+::)?TestCase\b")

TESTINSTANCE_CLASS_PATTERN = re.compile(r"class\s+\w+\s*:\s*public\s+(?:\w+::)?TestInstance\b")

CREATE_INSTANCE_PATTERN = re.compile(r"\bcreateInstance\s*\(")

ITERATE_PATTERN = re.compile(r"\biterate\s*\(")

FREE_TEST_FUNC_PATTERN = re.compile(
    r"\btcu::TestStatus\s+\w+\s*\(\s*(?:const\s+)?(?:vkt::)?Context\s*&\s*\w+",
    re.MULTILINE,
)

CHECK_SUPPORT_PATTERN = re.compile(r"\b\w*checkSupport\w*\s*\(")

INIT_PROGRAMS_PATTERN = re.compile(r"\b\w*initPrograms\w*\s*\(")

def _brace_balance(code: str) -> int:
    balance = 0
    for ch in code:
        if ch == "{":
            balance += 1
        elif ch == "}":
            balance -= 1
    return balance

def validate_structure(code: str, classification: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if INCLUDE_PATTERN.search(code):
        errors.append("cleaned body still contains include directives")
    if NAMESPACE_PATTERN.search(code):
        errors.append("cleaned body still contains namespace wrappers")
    if FENCE_PATTERN.search(code):
        errors.append("cleaned body still contains markdown fences")
    if REGISTER_PATTERN.search(code):
        errors.append("cleaned body still contains registration calls")

    brace_balance = _brace_balance(code)
    if brace_balance != 0:
        errors.append(f"brace balance is not zero: {brace_balance}")

    template_type = classification.get("template_type")
    if template_type == "class":
        if not TESTCASE_CLASS_PATTERN.search(code):
            errors.append("class candidate is missing a TestCase-derived class")
        if not CREATE_INSTANCE_PATTERN.search(code):
            errors.append("class candidate is missing createInstance()")
        if not TESTINSTANCE_CLASS_PATTERN.search(code):
            warnings.append("class candidate does not contain an explicit TestInstance-derived class")
        if not ITERATE_PATTERN.search(code):
            warnings.append("class candidate does not contain iterate()")
    elif template_type == "function":
        if TESTCASE_CLASS_PATTERN.search(code):
            warnings.append("function candidate still contains a TestCase-derived class")
        if not FREE_TEST_FUNC_PATTERN.search(code):
            errors.append("function candidate is missing a free tcu::TestStatus test function")
        if classification.get("has_check_support") and not CHECK_SUPPORT_PATTERN.search(code):
            errors.append("function candidate declares has_check_support=true but no checkSupport function was found")
        if classification.get("has_init_programs") and not INIT_PROGRAMS_PATTERN.search(code):
            errors.append("function candidate declares has_init_programs=true but no initPrograms function was found")
    else:
        errors.append(f"unsupported template_type in classification: {template_type}")

    params_type = classification.get("params_type")
    has_params = bool(classification.get("has_params"))
    if has_params and not params_type:
        errors.append("has_params=true but params_type is empty")
    if not has_params and params_type:
        warnings.append("has_params=false but params_type is populated")

    return {
        "valid": not errors,
        "template_type": template_type,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "signals": {
            "contains_testcase_class": bool(TESTCASE_CLASS_PATTERN.search(code)),
            "contains_testinstance_class": bool(TESTINSTANCE_CLASS_PATTERN.search(code)),
            "contains_create_instance": bool(CREATE_INSTANCE_PATTERN.search(code)),
            "contains_iterate": bool(ITERATE_PATTERN.search(code)),
            "contains_free_test_function": bool(FREE_TEST_FUNC_PATTERN.search(code)),
            "contains_check_support": bool(CHECK_SUPPORT_PATTERN.search(code)),
            "contains_init_programs": bool(INIT_PROGRAMS_PATTERN.search(code)),
            "brace_balance": brace_balance,
        },
    }

def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def validate_case_artifact(case_dir: Path, attempt_index: int | None = None) -> dict[str, Any]:
    if attempt_index is None:
        body_path = case_dir / "normalized" / "cleaned_candidate_body.cpp"
        classification_path = case_dir / "meta" / "classification.json"
        output_path = case_dir / "meta" / "structure_validation.json"
    else:
        body_path = case_dir / "attempts" / f"attempt_{attempt_index}" / "normalized" / "cleaned_candidate_body.cpp"
        classification_path = case_dir / "attempts" / f"attempt_{attempt_index}" / "meta" / "classification.json"
        output_path = case_dir / "attempts" / f"attempt_{attempt_index}" / "meta" / "structure_validation.json"

    if not body_path.exists():
        raise FileNotFoundError(f"missing cleaned candidate body: {body_path}")
    if not classification_path.exists():
        raise FileNotFoundError(f"missing classification json: {classification_path}")

    result = validate_structure(
        code=body_path.read_text(encoding="utf-8"),
        classification=_load_json(classification_path),
    )
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# ========================================================================
# 源自 feedback_schema.py
# ========================================================================

DIRECTOR_DECISION_SCHEMA = {
    "type": "object",
    "required": [
        "use_agents",
        "reason",
    ],
    "properties": {
        "use_agents": {"type": "array"},
        "reason": {"type": "string"},
    },
}

EXPLAINER_SCHEMA = {
    "type": "object",
    "required": [
        "test_objective",
        "pipeline_structure",
        "resource_and_state_requirements",
        "semantic_invariants",
        "pass_criteria",
        "fail_criteria",
        "base_class_obligations",
    ],
    "properties": {
        "test_objective": {"type": "string"},
        "pipeline_structure": {"type": "array"},
        "resource_and_state_requirements": {"type": "array"},
        "semantic_invariants": {"type": "array"},
        "pass_criteria": {"type": "array"},
        "fail_criteria": {"type": "array"},
        "base_class_obligations": {"type": "array"},
    },
}

VERIFIER_SCHEMA = {
    "type": "object",
    "required": [
        "semantics_aligned",
        "summary",
        "preserved_points",
        "drift_points",
        "required_fixes",
    ],
    "properties": {
        "semantics_aligned": {"type": "boolean"},
        "summary": {"type": "string"},
        "preserved_points": {"type": "array"},
        "drift_points": {"type": "array"},
        "required_fixes": {"type": "array"},
    },
}

CHECKER_SCHEMA = {
    "type": "object",
    "required": [
        "error_groups",
        "template_risk",
        "priority_fixes",
    ],
    "properties": {
        "error_groups": {"type": "array"},
        "template_risk": {
            "type": "string",
            "enum": ["class_still_valid", "function_still_valid", "structure_mismatch", "uncertain"],
        },
        "priority_fixes": {"type": "array"},
    },
}

STRUCTURE_REJUDGE_SCHEMA = {
    "type": "object",
    "required": [
        "template_type",
        "has_params",
        "has_check_support",
        "has_init_programs",
        "params_type",
        "reason",
    ],
    "properties": {
        "template_type": {"type": "string", "enum": ["class", "function"]},
        "has_params": {"type": "boolean"},
        "has_check_support": {"type": "boolean"},
        "has_init_programs": {"type": "boolean"},
        "params_type": {"type": ["string", "null"]},
        "reason": {"type": "string"},
    },
}

FUNCTION_REPAIR_SCHEMA = {
    "type": "object",
    "required": [
        "template_type",
        "function_code",
        "has_params",
        "has_check_support",
        "has_init_programs",
    ],
    "properties": {
        "template_type": {"type": "string", "enum": ["function"]},
        "function_code": {"type": "string"},
        "check_support_code": {"type": "string"},
        "init_programs_code": {"type": "string"},
        "params_decl_code": {"type": "string"},
        "has_params": {"type": "boolean"},
        "has_check_support": {"type": "boolean"},
        "has_init_programs": {"type": "boolean"},
        "params_type": {"type": ["string", "null"]},
        # 与 class 同名同形。function 只消费第一项的 params_init 用来给参数初值;
        # case_name 被忽略,叶子名恒为脚本的 case_{row_id}(stage3 按该名拼测试路径)。
        "register_cases": {"type": "array"},
    },
}

CLASS_REPAIR_SCHEMA = {
    "type": "object",
    "required": [
        "template_type",
        "class_code",
        "has_params",
    ],
    "properties": {
        "template_type": {"type": "string", "enum": ["class"]},
        "class_code": {"type": "string"},
        "params_decl_code": {"type": "string"},
        "has_params": {"type": "boolean"},
        "params_type": {"type": ["string", "null"]},
        "register_cases": {"type": "array"},
    },
}

REGENERATION_CLASS_SCHEMA = CLASS_REPAIR_SCHEMA

REGENERATION_FUNCTION_SCHEMA = FUNCTION_REPAIR_SCHEMA


# ========================================================================
# 源自 build_lock.py
# ========================================================================

try:
    import psutil
except ImportError:  # pragma: no cover - 当前运行环境已安装；缺失时采取保守策略。
    psutil = None

BUILD_LOCK_DIR = DEFAULT_RUNS_DIR / "_build_lock"

BUILD_LOCK_WAIT_MIN_SECONDS = 4.0

BUILD_LOCK_WAIT_MAX_SECONDS = 12.0

def _pid_is_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        # Windows 的 os.kill(pid, 0) **不是** POSIX 探活：Python 会调用
        # TerminateProcess(handle, 0)，从而真的以 returncode=0 杀掉持锁 worker。
        # psutil 只读查询进程；权限不足、依赖缺失或查询异常时保守判为 alive，
        # 宁可等待人工清锁，也绝不能把仍在注入/编译窗口的锁误删。
        if psutil is None:
            return True
        try:
            process = psutil.Process(int(pid))
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False
        except psutil.AccessDenied:
            return True
        except Exception:
            return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return True
    return True

def _load_lock_payload(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _sanitize_lock_component(value: str) -> str:
    sanitized = []
    for ch in value:
        if ch.isalnum():
            sanitized.append(ch.lower())
        else:
            sanitized.append("_")
    text = "".join(sanitized).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    return text or "default"

def _lock_path_for_scope(scope: str) -> Path:
    return BUILD_LOCK_DIR / f"{_sanitize_lock_component(scope)}.lock"

def build_lock_scope_for_cts_root(cts_root: Path | str) -> str:
    return str((Path(cts_root) / "build").resolve())

@contextmanager
def acquire_build_lock(scope: str, owner: str) -> Iterator[None]:
    BUILD_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path_for_scope(scope)
    pid = os.getpid()
    while True:
        payload = {
            "scope": scope,
            "owner": owner,
            "pid": pid,
            "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError):
            existing = _load_lock_payload(lock_path)
            existing_pid = existing.get("pid")
            if isinstance(existing_pid, int) and not _pid_is_alive(existing_pid):
                try:
                    lock_path.unlink()
                    print(
                        f"[Build Lock] Removed stale lock scope={scope} from pid={existing_pid} "
                        f"owner={existing.get('owner')}"
                    )
                    continue
                except FileNotFoundError:
                    continue
                except OSError:
                    pass
            wait_seconds = random.uniform(BUILD_LOCK_WAIT_MIN_SECONDS, BUILD_LOCK_WAIT_MAX_SECONDS)
            lock_owner = existing.get("owner") if isinstance(existing, dict) else None
            print(
                f"[Build Lock] Waiting for build lock scope={scope} "
                f"(owner={lock_owner}, pid={existing_pid}) for {wait_seconds:.1f}s"
            )
            time.sleep(wait_seconds)
            continue
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            yield
        finally:
            existing = _load_lock_payload(lock_path)
            if existing.get("pid") == pid:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
        return


# ========================================================================
# 源自 prepare_agent_case.py
# ========================================================================

AGENT_DATA_DIR = DEFAULT_RUNS_DIR

AGENT_TEST_DATA_RUNS_DIR = DEFAULT_RUNS_DIR

LEGACY_ROOT_FILES = (
    "raw_generated.cpp",
    "cleaned_candidate_body.cpp",
    "assembled_candidate.cpp",
    "cleaning_report.json",
    "classification.json",
    "input_snapshot.json",
    "status.json",
)

def get_agent_output_root(develop: bool = False) -> Path:
    override = resolve_output_root_override()
    if override is not None:
        return override
    return AGENT_TEST_DATA_RUNS_DIR if develop else AGENT_DATA_DIR

def build_output_dir(mode_name: str, model_name: str, row_id: int, develop: bool = False) -> Path:
    output_root = get_agent_output_root(develop=develop)
    override = resolve_output_root_override()
    if override is not None:
        return output_root / f"case_{row_id}"
    return output_root / mode_name / model_name / f"case_{row_id}"

def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def cleanup_legacy_root_files(output_dir: Path) -> None:
    for name in LEGACY_ROOT_FILES:
        path = output_dir / name
        if path.exists():
            path.unlink()

def _fallback_classification_from_generated_code(
    cleaned_code: str,
    *,
    mode_name: str,
    model_name: str,
    row_id: int,
) -> ClassifiedCode:
    register_name = f"register_{mode_name.replace('-', '_')}_{model_name}_{row_id}"
    case_name = f"case_{row_id}"
    class_name = infer_class_name_from_member_definitions(cleaned_code)
    function_name = infer_function_name_from_body(cleaned_code)

    if class_name:
        return ClassifiedCode(
            template_type="class",
            class_name=class_name,
            instance_name=None,
            function_name=None,
            register_name=register_name,
            case_name=case_name,
            has_check_support="checkSupport" in cleaned_code,
            has_init_programs="initPrograms" in cleaned_code,
            has_params=False,
            params_type=None,
            class_ctor_arity=2,
            register_style="addChild",
            register_cases=None,
            code=cleaned_code.strip(),
        )

    if function_name:
        has_init_programs = "initPrograms" in cleaned_code
        return ClassifiedCode(
            template_type="function",
            class_name=None,
            instance_name=None,
            function_name=function_name,
            register_name=register_name,
            case_name=case_name,
            has_check_support="checkSupport" in cleaned_code,
            has_init_programs=has_init_programs,
            has_params=False,
            params_type=None,
            class_ctor_arity=None,
            register_style="addFunctionCaseWithPrograms" if has_init_programs else "addFunctionCase",
            register_cases=None,
            code=cleaned_code.strip(),
        )

    raise ValueError("unable to classify candidate as class or function")

def build_prepared_case_payload(mode_name: str, model_name: str, row_id: int) -> dict:
    """Build deterministic attempt-1 preparation data without creating output directories."""
    gl_row = load_gl_sim_vk_row(row_id)
    initial_candidate = load_initial_candidate(mode_name, model_name, row_id)
    # Don't decode escaped newlines - JSON parser already did it, and we need to preserve
    # C++ string literal escape sequences like \n, \t, etc.
    cleaned = clean_generated_code(initial_candidate["generated_cpp"], decode_escaped_newlines=False)
    try:
        classified = classify_candidate_code(
            cleaned["cleaned_code"],
            mode_name=mode_name,
            model_name=model_name,
            row_id=row_id,
        )
    except ValueError:
        classified = _fallback_classification_from_generated_code(
            cleaned["cleaned_code"],
            mode_name=mode_name,
            model_name=model_name,
            row_id=row_id,
        )
    assembled_cpp = assemble_cpp(classified)
    return {
        "gl_row": gl_row,
        "initial_candidate": initial_candidate,
        "cleaned": cleaned,
        "classified": classified,
        "assembled_cpp": assembled_cpp,
        "structure_validation": validate_structure(
            cleaned["cleaned_code"], classified.to_dict()
        ),
    }

def prepare_case(mode_name: str, model_name: str, row_id: int, force: bool, develop: bool = False) -> Path:
    prepared = build_prepared_case_payload(mode_name, model_name, row_id)
    gl_row = prepared["gl_row"]
    initial_candidate = prepared["initial_candidate"]
    cleaned = prepared["cleaned"]
    classified = prepared["classified"]
    assembled_cpp = prepared["assembled_cpp"]

    output_dir = build_output_dir(mode_name, model_name, row_id, develop=develop)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs_dir = output_dir / "inputs"
    normalized_dir = output_dir / "normalized"
    meta_dir = output_dir / "meta"

    inputs_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    raw_cpp_path = inputs_dir / "raw_generated.cpp"
    cleaned_cpp_path = normalized_dir / "cleaned_candidate_body.cpp"
    assembled_cpp_path = normalized_dir / "assembled_candidate.cpp"
    cleaning_report_path = meta_dir / "cleaning_report.json"
    classification_path = meta_dir / "classification.json"
    structure_validation_path = meta_dir / "structure_validation.json"
    input_snapshot_path = meta_dir / "input_snapshot.json"
    status_path = meta_dir / "status.json"

    if not force and assembled_cpp_path.exists():
        raise FileExistsError(f"{assembled_cpp_path} already exists, use --force to overwrite")
    if force:
        cleanup_legacy_root_files(output_dir)

    raw_cpp_path.write_text(initial_candidate["generated_cpp"], encoding="utf-8")
    cleaned_cpp_path.write_text(str(cleaned["cleaned_code"]), encoding="utf-8")
    assembled_cpp_path.write_text(assembled_cpp, encoding="utf-8")
    write_json(cleaning_report_path, dict(cleaned))
    write_json(classification_path, classified.to_dict())
    write_json(structure_validation_path, prepared["structure_validation"])
    write_json(
        input_snapshot_path,
        {
            "row_id": row_id,
            "mode_name": mode_name,
            "model_name": model_name,
            "gl_subclass": gl_row.get("gl_subclass"),
            "gl_base_classes": gl_row.get("gl_base_classes"),
            "similar_vulkan_case": gl_row.get("similar_vulkan_case"),
            "source_case_dir": initial_candidate["source_case_dir"],
            "generated_cpp_path": initial_candidate["generated_cpp_path"],
        },
    )
    write_json(
        status_path,
        {
            "row_id": row_id,
            "mode_name": mode_name,
            "model_name": model_name,
            "stage": "prepared",
            "template_type": classified.template_type,
            "template_type_source": "actual_generated_candidate",
            "has_params": classified.has_params,
            "has_check_support": classified.has_check_support,
            "has_init_programs": classified.has_init_programs,
            "cleaned_before_compile": True,
        },
    )
    return output_dir


# ========================================================================
# 源自 repair_payload_applier.py —— 修复载荷校验与应用
# ========================================================================

FUNCTION_REQUIRED_FIELDS = (
    "template_type",
    "function_code",
    "has_params",
    "has_check_support",
    "has_init_programs",
)

CLASS_REQUIRED_FIELDS = (
    "template_type",
    "class_code",
    "has_params",
)

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def _normalize_slot_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = clean_generated_code(text, decode_escaped_newlines=False)
    return str(cleaned["cleaned_code"]).strip()

def _infer_template_type_from_payload(payload: dict[str, Any]) -> str | None:
    template_type = payload.get("template_type")
    if template_type in {"class", "function"}:
        return str(template_type)

    class_code = payload.get("class_code")
    if isinstance(class_code, str) and class_code.strip():
        return "class"

    function_code = payload.get("function_code")
    if isinstance(function_code, str) and function_code.strip():
        return "function"

    register_cases = payload.get("register_cases")
    if isinstance(register_cases, (list, dict)) and register_cases:
        return "class"

    return None

def _extract_json_object(raw_text: str) -> dict[str, Any]:
    stripped = raw_text.strip()
    if not stripped:
        raise ValueError("empty payload")

    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_text, re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    first = raw_text.find("{")
    last = raw_text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("unable to locate JSON object in payload")
    return json.loads(raw_text[first : last + 1])

def load_payload(payload_path: Path) -> dict[str, Any]:
    return _extract_json_object(_read_text(payload_path))

def _require_fields(payload: dict[str, Any], required_fields: tuple[str, ...]) -> None:
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(f"payload missing required fields: {missing}")

def _validate_common_fields(payload: dict[str, Any]) -> None:
    if payload["template_type"] not in {"class", "function"}:
        raise ValueError(f"unsupported template_type={payload['template_type']}")

    for field in ("has_params",):
        if not isinstance(payload[field], bool):
            raise ValueError(f"{field} must be boolean")

    if "params_type" in payload and payload["params_type"] is not None and not isinstance(payload["params_type"], str):
        raise ValueError("params_type must be string or null")

def validate_payload(payload: dict[str, Any]) -> None:
    payload = _normalize_payload_consistency(payload)
    _validate_common_fields(payload)

    if payload["template_type"] == "function":
        _require_fields(payload, FUNCTION_REQUIRED_FIELDS)
        for field in ("has_check_support", "has_init_programs"):
            if not isinstance(payload[field], bool):
                raise ValueError(f"{field} must be boolean")
        if not isinstance(payload.get("function_code"), str):
            payload["function_code"] = str(payload.get("function_code") or "")
    else:
        _require_fields(payload, CLASS_REQUIRED_FIELDS)
        if not isinstance(payload.get("class_code"), str):
            payload["class_code"] = str(payload.get("class_code") or "")
        if payload.get("register_cases") is not None:
            payload["register_cases"] = _normalize_register_cases(payload.get("register_cases"))
            for index, case in enumerate(payload["register_cases"]):
                if not isinstance(case, dict):
                    raise ValueError(f"register_cases[{index}] must be object")
                if "case_name" not in case or "params_init" not in case:
                    raise ValueError(f"register_cases[{index}] must contain case_name and params_init")
                if not isinstance(case["case_name"], str) or not case["case_name"].strip():
                    raise ValueError(f"register_cases[{index}].case_name must be non-empty string")
                if not isinstance(case["params_init"], str) or not case["params_init"].strip():
                    raise ValueError(f"register_cases[{index}].params_init must be non-empty string")

def _normalize_payload_consistency(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    inferred_template_type = _infer_template_type_from_payload(normalized)
    if inferred_template_type is not None:
        normalized["template_type"] = inferred_template_type

    if normalized.get("template_type") != "class":
        normalized["function_code"] = str(normalized.get("function_code") or "")
        normalized["check_support_code"] = str(normalized.get("check_support_code") or "")
        normalized["init_programs_code"] = str(normalized.get("init_programs_code") or "")
        normalized["params_decl_code"] = str(normalized.get("params_decl_code") or "")
        if "has_params" not in normalized or not isinstance(normalized.get("has_params"), bool):
            normalized["has_params"] = bool(
                normalized.get("params_type")
                or str(normalized.get("params_decl_code") or "").strip()
            )
        if "has_check_support" not in normalized or not isinstance(normalized.get("has_check_support"), bool):
            normalized["has_check_support"] = bool(str(normalized.get("check_support_code") or "").strip())
        if "has_init_programs" not in normalized or not isinstance(normalized.get("has_init_programs"), bool):
            normalized["has_init_programs"] = bool(str(normalized.get("init_programs_code") or "").strip())
        # function 挂一个叶子, 所以只留第一项 —— 它的 params_init 用来给参数初值,
        # case_name 被 template_manager 忽略。整体丢弃会让参数退化成全零的 {}。
        normalized["register_cases"] = _normalize_register_cases(normalized.get("register_cases"))[:1]
        return normalized

    normalized["class_code"] = str(normalized.get("class_code") or "")
    normalized["params_decl_code"] = str(normalized.get("params_decl_code") or "")
    if "has_params" not in normalized or not isinstance(normalized.get("has_params"), bool):
        normalized["has_params"] = bool(
            normalized.get("params_type")
            or str(normalized.get("params_decl_code") or "").strip()
            or normalized.get("register_cases")
        )

    if not normalized.get("has_params", False):
        normalized["params_type"] = None
        normalized["params_decl_code"] = ""
        normalized["register_cases"] = []
        return normalized

    normalized["register_cases"] = _normalize_register_cases(normalized.get("register_cases"))
    return normalized

def _build_repaired_body(payload: dict[str, Any]) -> str:
    segments: list[str] = []

    params_decl_code = _normalize_slot_text(payload.get("params_decl_code", ""))
    class_code = _normalize_slot_text(payload.get("class_code", "")) if payload["template_type"] == "class" else ""
    params_decl_code = _dedupe_class_params_decl(
        params_decl_code=params_decl_code,
        class_code=class_code,
        params_type=payload.get("params_type"),
        template_type=payload["template_type"],
    )
    if params_decl_code:
        segments.append(params_decl_code)

    if payload["template_type"] == "class":
        if class_code:
            segments.append(class_code)
        return "\n\n".join(segments).strip()

    check_support_code = _normalize_slot_text(payload.get("check_support_code", ""))
    init_programs_code = _normalize_slot_text(payload.get("init_programs_code", ""))
    function_code = _normalize_slot_text(payload["function_code"])

    if payload.get("has_check_support") and check_support_code:
        segments.append(check_support_code)
    if payload.get("has_init_programs") and init_programs_code:
        segments.append(init_programs_code)
    segments.append(function_code)
    return "\n\n".join(segment for segment in segments if segment).strip()

def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def _normalize_register_cases(raw_cases: Any) -> list[dict[str, str]]:
    if raw_cases is None:
        return []
    if isinstance(raw_cases, dict):
        raw_cases = [raw_cases]
    if not isinstance(raw_cases, list):
        return []

    normalized_cases: list[dict[str, str]] = []
    for index, case in enumerate(raw_cases, start=1):
        if isinstance(case, str):
            params_init = case.strip()
            if params_init:
                normalized_cases.append(
                    {
                        "case_name": f"case_{index}",
                        "params_init": params_init,
                    }
                )
            continue
        if not isinstance(case, dict):
            continue
        case_name = str(case.get("case_name") or f"case_{index}").strip()
        params_init = str(case.get("params_init") or "").strip()
        if not params_init:
            continue
        normalized_cases.append(
            {
                "case_name": case_name or f"case_{index}",
                "params_init": params_init,
            }
        )
    return normalized_cases

def _dedupe_class_params_decl(
    *,
    params_decl_code: str,
    class_code: str,
    params_type: Any,
    template_type: str,
) -> str:
    if template_type != "class":
        return params_decl_code
    if not params_decl_code or not class_code or not isinstance(params_type, str) or not params_type.strip():
        return params_decl_code

    type_name = re.escape(params_type.strip())
    struct_pattern = re.compile(rf"\bstruct\s+{type_name}\b")
    class_pattern = re.compile(rf"\bclass\s+{type_name}\b")
    decl_has_type = bool(struct_pattern.search(params_decl_code) or class_pattern.search(params_decl_code))
    class_has_type = bool(struct_pattern.search(class_code) or class_pattern.search(class_code))
    if not decl_has_type or not class_has_type:
        return params_decl_code

    normalized_decl = _normalize_space(params_decl_code)
    normalized_class = _normalize_space(class_code)
    if normalized_decl and normalized_decl in normalized_class:
        return ""
    return params_decl_code

def _load_base_status(case_dir: Path) -> dict[str, Any]:
    status_path = case_dir / "meta" / "status.json"
    if status_path.exists():
        return json.loads(_read_text(status_path))
    return {}

def _merge_status(base_status: dict[str, Any], classification: ClassifiedCode, attempt_index: int) -> dict[str, Any]:
    merged = dict(base_status)
    merged.update(
        {
            "stage": "repair_applied",
            "attempt_index": attempt_index,
            "template_type": classification.template_type,
            "has_params": classification.has_params,
            "has_check_support": classification.has_check_support,
            "has_init_programs": classification.has_init_programs,
            "cleaned_before_compile": True,
        }
    )
    return merged

def _build_locked_classification(
    *,
    mode_name: str,
    model_name: str,
    row_id: int,
    payload: dict[str, Any],
    repaired_body: str,
) -> ClassifiedCode:
    register_name = f"register_{mode_name.replace('-', '_')}_{model_name}_{row_id}"
    case_name = f"case_{row_id}"
    template_type = str(payload["template_type"])

    if template_type == "class":
        class_match = CLASS_CASE_PATTERN.search(repaired_body) or GENERIC_CLASS_PATTERN.search(repaired_body)
        instance_match = INSTANCE_CASE_PATTERN.search(repaired_body) or GENERIC_INSTANCE_PATTERN.search(repaired_body)
        class_name = class_match.group("name") if class_match else infer_class_name_from_member_definitions(repaired_body)
        if not class_name:
            class_name = f"generated_{mode_name.replace('-', '_')}_{model_name}_{row_id}"
        instance_name = instance_match.group("name") if instance_match else None
        params_type = payload.get("params_type")
        ctor_params_type, ctor_arity = _extract_params_type_from_ctor(repaired_body, class_name)
        if not isinstance(params_type, str) or not params_type.strip():
            params_type = ctor_params_type or _guess_params_type_from_struct(repaired_body)
        has_params = bool(payload.get("has_params"))
        if params_type:
            has_params = True
        elif not has_params:
            params_type = None
        return ClassifiedCode(
            template_type="class",
            class_name=class_name,
            instance_name=instance_name,
            function_name=None,
            register_name=register_name,
            case_name=case_name,
            has_check_support="checkSupport" in repaired_body,
            has_init_programs="initPrograms" in repaired_body,
            has_params=has_params,
            params_type=params_type,
            class_ctor_arity=ctor_arity,
            register_style="addChild",
            register_cases=payload.get("register_cases") or [],
            code=repaired_body.strip(),
        )

    function_match = FREE_TEST_FUNC_PATTERN.search(repaired_body) or GENERIC_FREE_FUNCTION_PATTERN.search(repaired_body)
    function_name = function_match.group("name") if function_match else infer_function_name_from_body(repaired_body)
    if not function_name:
        function_name = f"generated_{mode_name.replace('-', '_')}_{model_name}_{row_id}"
    params_type = payload.get("params_type")
    if not isinstance(params_type, str) or not params_type.strip():
        params_type = _extract_params_type_from_function_signature(repaired_body, function_name)
    if not params_type:
        params_type = _extract_params_type_from_generic_function_signature(repaired_body, function_name)
    if not params_type:
        params_type = _guess_params_type_from_struct(repaired_body)
    has_check_support = bool(payload.get("has_check_support"))
    has_init_programs = bool(payload.get("has_init_programs"))
    if FREE_CHECK_SUPPORT_PATTERN.search(repaired_body):
        has_check_support = True
    if FREE_INIT_PROGRAMS_PATTERN.search(repaired_body):
        has_init_programs = True
    has_params = bool(payload.get("has_params"))
    if params_type:
        has_params = True
    elif not has_params:
        params_type = None
    return ClassifiedCode(
        template_type="function",
        class_name=None,
        instance_name=None,
        function_name=function_name,
        register_name=register_name,
        case_name=case_name,
        has_check_support=has_check_support,
        has_init_programs=has_init_programs,
        has_params=has_params,
        params_type=params_type,
        class_ctor_arity=None,
        register_style="addFunctionCaseWithPrograms" if has_init_programs else "addFunctionCase",
        # 只有第一项会被消费(取 params_init 给参数初值), 叶子名仍用 case_name
        register_cases=(payload.get("register_cases") or [])[:1],
        code=repaired_body.strip(),
    )

def apply_repair_payload(
    mode_name: str,
    model_name: str,
    row_id: int,
    payload: dict[str, Any],
    attempt_index: int,
    develop: bool = False,
) -> dict[str, str]:
    payload = _normalize_payload_consistency(payload)
    validate_payload(payload)

    case_dir = build_output_dir(mode_name, model_name, row_id, develop=develop)
    if not case_dir.exists():
        raise FileNotFoundError(f"case directory not found: {case_dir}")

    feedback_dir = case_dir / "feedback" / f"attempt_{attempt_index}"
    attempt_dir = case_dir / "attempts" / f"attempt_{attempt_index}"
    normalized_dir = attempt_dir / "normalized"
    meta_dir = attempt_dir / "meta"

    feedback_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    repaired_body = _build_repaired_body(payload)
    classification = _build_locked_classification(
        mode_name=mode_name,
        model_name=model_name,
        row_id=row_id,
        payload=payload,
        repaired_body=repaired_body,
    )
    assembled_cpp = assemble_cpp(classification)

    write_json(feedback_dir / "repair_payload.normalized.json", payload)
    (normalized_dir / "cleaned_candidate_body.cpp").write_text(repaired_body, encoding="utf-8")
    (normalized_dir / "assembled_candidate.cpp").write_text(assembled_cpp, encoding="utf-8")
    write_json(meta_dir / "classification.json", classification.to_dict())
    write_json(meta_dir / "structure_validation.json", validate_structure(repaired_body, classification.to_dict()))
    write_json(
        meta_dir / "repair_application.json",
        {
            "row_id": row_id,
            "mode_name": mode_name,
            "model_name": model_name,
            "attempt_index": attempt_index,
            "template_type": classification.template_type,
            "payload_template_type": payload["template_type"],
            "register_cases_count": len(payload.get("register_cases", [])),
        },
    )
    write_json(
        meta_dir / "status.json",
        _merge_status(_load_base_status(case_dir), classification, attempt_index),
    )

    return {
        "case_dir": str(case_dir),
        "feedback_dir": str(feedback_dir),
        "attempt_dir": str(attempt_dir),
        "cleaned_candidate_body_path": str(normalized_dir / "cleaned_candidate_body.cpp"),
        "assembled_candidate_path": str(normalized_dir / "assembled_candidate.cpp"),
        "classification_path": str(meta_dir / "classification.json"),
    }

def apply_repair_payload_file(
    mode_name: str,
    model_name: str,
    row_id: int,
    payload_path: Path,
    attempt_index: int,
    develop: bool = False,
) -> dict[str, str]:
    case_dir = build_output_dir(mode_name, model_name, row_id, develop=develop)
    feedback_dir = case_dir / "feedback" / f"attempt_{attempt_index}"
    feedback_dir.mkdir(parents=True, exist_ok=True)

    raw_text = _read_text(payload_path)
    (feedback_dir / "repair_payload.raw.txt").write_text(raw_text, encoding="utf-8")
    payload = _extract_json_object(raw_text)
    result = apply_repair_payload(
        mode_name=mode_name,
        model_name=model_name,
        row_id=row_id,
        payload=payload,
        attempt_index=attempt_index,
        develop=develop,
    )
    write_json(feedback_dir / "repair_payload.extracted.json", payload)
    return result
