"""包内唯一路径锚点。

原实现里每个模块各写一次 `Path(__file__).resolve().parents[3]`，锚在仓库根上；
任何目录深度变化都会静默指向错误位置。本包把锚点收在这里，只依赖「本文件与
包根同级」这一条事实。

CTS 树的位置不在这里 —— 它必须由 `--cts-root` 显式传入，没有默认值。原实现
有三张硬编码的 CTS 根映射表（`E:\\VK-GL-CTS_02` 等），那些树同时被其它流程占用，
猜错会把代码注入正在使用的树。
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent

PROMPT_DIR = PACKAGE_ROOT / "prompts"
GENERATION_PROMPT_DIR = PROMPT_DIR / "generation"
FEEDBACK_PROMPT_DIR = PROMPT_DIR / "feedback"
DIRECTOR_PROMPT_DIR = PROMPT_DIR / "director"

TEMPLATE_DIR = PACKAGE_ROOT / "pattern" / "templates"

# 运行时产物。--output-root 可覆盖。
DEFAULT_RUNS_DIR = PACKAGE_ROOT / "runs"

__all__ = [
    "PACKAGE_ROOT", "PROMPT_DIR",
    "GENERATION_PROMPT_DIR", "FEEDBACK_PROMPT_DIR", "DIRECTOR_PROMPT_DIR",
    "TEMPLATE_DIR", "DEFAULT_RUNS_DIR",
    "runs_root", "seed_case_dir", "seed_compile_dir", "method_case_dir",
    "index_path",
]

_runs_root: Path = DEFAULT_RUNS_DIR


def runs_root(override: Path | None = None) -> Path:
    """产物根。传入 override 时同时设为后续默认值。"""
    global _runs_root
    if override is not None:
        _runs_root = Path(override).resolve()
    return _runs_root


def seed_case_dir(seed_kind: str, model: str, row_id: int) -> Path:
    """Step 1/2 的种子目录。seed_kind ∈ {non_cot, cot}。

    两条种子各自只生成一次、只编译一次；Step 3 的三条线引用它们，不复制。
    """
    return runs_root() / "seeds" / seed_kind / model / f"case_{row_id}"


def seed_compile_dir(seed_kind: str, model: str, row_id: int) -> Path:
    """Step 2 的种子编译目录 —— 刻意与种子目录分开。

    prepare_case(force=True) 会清掉 case 根的 LEGACY_ROOT_FILES，其首项正是
    `raw_generated.cpp`。原工程里种子住在另一棵树（rag_generation/）所以无碍；
    本包把两者收进 runs/ 之后若共用一个 case 目录，Step 2 会删掉 Step 1 的种子。
    种子目录因此保持只读，编译产物落在这里。
    """
    return runs_root() / "compile" / seed_kind / model / f"case_{row_id}"


def method_case_dir(method_slug: str, model: str, row_id: int) -> Path:
    """Step 3 的方法目录。attempt 从 2 起，attempt_1 即对应种子。"""
    return runs_root() / method_slug / model / f"case_{row_id}"


def index_path() -> Path:
    """全局索引，取代原实现的 stage2 sqlite 表。"""
    return runs_root() / "_index.json"
