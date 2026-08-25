"""方法与种子注册表 —— 全包唯一的方法名来源。

路径、C++ 符号、构建目标、日志标签一律从这里取值；别处不出现方法名字面量。
改名时只改本文件的 slug/display 两个字段，加一次 runs/ 目录重命名。

为什么 slug 与 display 必须分开：`w/o cot` 带斜杠和空格，而落盘路径和 C++
标识符不能有。原实现的 token 净化只把 `-` 换成 `_`，直接拿 display 去拼会产出
非法标识符 `populate_w/o cot_claude_44`。论文表格用 display，落盘和符号用 slug。

消融语义（组件递进，非 2×2 网格）：director 与 cot 是互替的推理机制，不叠加。
所以 multi_agent 用 non_cot 种子不是「漏了 cot」，是 cot 被 director 取代。

    方法              组件    种子      修复策略
    non-cot           ①      -         -
    cot               ②      -         -          (= w/o feedback)
    w/o cot           ①+③    non_cot   反馈
    single-agent      ②+③    cot       反馈
    multi-agent       ①+④    non_cot   director
"""

from __future__ import annotations

from dataclasses import dataclass

# ── 种子 ────────────────────────────────────────────────────────────────
# 两条种子线本身就是基线 ① 和 ②。cot 种子另有一个展示别名 `w/o feedback`：
# 它不是独立运行，而是「cot + 反馈修复」去掉反馈后的那一格，已验证与
# single-agent 的 attempt_1 输入字节全等，因此不单独建目录。

SEED_KINDS = ("non_cot", "cot")

SEED_DISPLAY = {
    "non_cot": "non-cot",
    "cot": "woFeedback",
}

# The CoT seed is reported as woFeedback because it is the initial-generation
# condition without compiler-feedback repair.
SEED_ALIAS = {
    "cot": "woFeedback",
}

SEED_INPUT_NAMES = tuple(sorted({
    *SEED_KINDS,
    "non-cot",
    *SEED_DISPLAY.values(),
    *SEED_ALIAS.values(),
}))

SEED_PROMPT_FILE = {
    "non_cot": "non_cot_prompt.txt",
    "cot": "cot_prompt.txt",
}


@dataclass(frozen=True)
class Method:
    """Step 3 的一条修复线。

    slug     落盘目录名与 C++ 符号用；必须是合法标识符片段。
    display  论文/报告用名。
    legacy   原工程的目录名，用于读取既有产物；新产物不写这个名字。
    seed     种子种类，决定 attempt_1 从哪来。
    repair   修复策略：feedback = 单轮反馈修复；director = 多智能体编排。
    """

    key: str
    slug: str
    display: str
    legacy: str
    seed: str
    repair: str


# 键名即最终语义。slug/display 已改成论文口径；legacy 保持旧目录名不动 ——
# 它有两个真实用途：读既有产物，以及 cts.py 里按 `only-fix` 判物理模块布局
# （那一支要多注入一对 SmokeTests 文件）。把 legacy 一起改了会同时打断这两处。
METHODS: dict[str, Method] = {
    "wo_cot": Method(
        key="wo_cot", slug="wo_cot", display="woCot",
        legacy="shot-only-fix", seed="non_cot", repair="feedback",
    ),
    "single_agent": Method(
        key="single_agent", slug="single_agent", display="singleAgent",
        legacy="only-fix", seed="cot", repair="feedback",
    ),
    "multi_agent": Method(
        key="multi_agent", slug="multi_agent", display="multiAgent",
        legacy="director", seed="non_cot", repair="director",
    ),
}

METHOD_INPUT_NAMES = tuple(sorted({
    name
    for method in METHODS.values()
    for name in (method.key, method.slug, method.legacy, method.display)
}))

# 修复循环预算：attempt_2 起最多两轮，与原实现一致。
MAX_REPAIR_ATTEMPTS = 2
FIRST_REPAIR_ATTEMPT = 2

__all__ = [
    "SEED_KINDS", "SEED_DISPLAY", "SEED_ALIAS", "SEED_INPUT_NAMES", "SEED_PROMPT_FILE",
    "Method", "METHODS", "METHOD_INPUT_NAMES",
    "MAX_REPAIR_ATTEMPTS", "FIRST_REPAIR_ATTEMPT",
    "get_method", "resolve_repair_strategy", "method_display_name",
    "seed_display_name",
    "safe_token", "package_name", "target_name",
    "namespace_name", "populate_symbol", "case_basename",
    "seed_package_name",
]


def get_method(key: str) -> Method:
    for method in METHODS.values():
        if key in (method.key, method.slug, method.legacy, method.display):
            return method
    if key not in METHODS:
        raise ValueError(
            f"Unknown method {key!r}; available: {', '.join(METHOD_INPUT_NAMES)}")
    return METHODS[key]


def method_display_name(value: str) -> str:
    return get_method(value).display


def seed_display_name(value: str) -> str:
    if value not in SEED_KINDS:
        raise ValueError(f"Unknown seed kind: {value!r}")
    return SEED_DISPLAY[value]


def resolve_repair_strategy(mode_name: str) -> str:
    """给任一方法别名返回修复策略：feedback 或 director。

    原实现在修复循环里用 `mode_name == "only-fix"` 判分支，因为 agent_runner 当时
    只跑 single-agent 与 multi-agent 两条线，w/o cot 由独立 runner 驱动。三条线合进
    一个模块后，那个字面量会把同样走反馈修复的 w/o cot 误判进 director 分支。
    """
    for method in METHODS.values():
        if mode_name in (method.key, method.slug, method.legacy, method.display):
            return method.repair
    raise ValueError(
        f"无法判定 {mode_name!r} 的修复策略；可用: "
        f"{', '.join(sorted({n for m in METHODS.values() for n in (m.key, m.slug, m.legacy)}))}。")


def safe_token(value: str) -> str:
    """把名字压成合法的 C++ 标识符片段。

    原实现只替换 `-`。这里连非字母数字一律换成 `_`，否则改名后带斜杠或空格的
    display 一旦误传进来会产出无法编译的符号。
    """
    return "".join(ch if ch.isalnum() else "_" for ch in value)


def package_name(slug: str, model: str) -> str:
    """CTS 包名。种子编译用中性包名，见 seed_package_name。"""
    return f"{safe_token(slug)}_{safe_token(model)}"


def target_name(slug: str, model: str) -> str:
    return f"deqp-vk-{package_name(slug, model).replace('_', '-')}"


def namespace_name(slug: str, model: str) -> str:
    parts = package_name(slug, model).split("_")
    return "".join(part[:1].upper() + part[1:] for part in parts if part)


def populate_symbol(slug: str, model: str, row_id: int) -> str:
    return f"populate_{package_name(slug, model)}_{row_id}"


def case_basename(slug: str, model: str, row_id: int) -> str:
    return f"vktCustomCase_{package_name(slug, model)}_{row_id}"


def seed_package_name(seed_kind: str, model: str) -> str:
    """种子编译的包名。

    刻意与方法名解耦：同一份种子被多条线共用，只编一次。已验证包名不影响
    编译结论（同一份种子在两个不同包名下编译，2019 例结论完全一致），但错误
    文本里含包名，因此新跑结果与旧产物不是逐字节一致的。
    """
    return f"seed_{safe_token(seed_kind)}_{safe_token(model)}"
