"""config.py — 全流水线唯一的路径 / 环境配置入口。

设计目标：把原来散落在各脚本里的硬编码盘符路径全部收敛到这里，
并允许用环境变量覆盖，方便别人 clone 到自己机器上直接跑。

覆盖方式（环境变量优先级最高）：
  CTS_SOURCE_ROOT   VK-GL-CTS 源码树根
  CTS_BUILD_ROOT    CMake/MSBuild 生成目录（用于推导 compile_commands / .vcxproj）
  CTS_OUTPUT_ROOT   Vulkan 数据集输出根
  LIBCLANG_PATH     libclang.dll 绝对路径（Windows 下 ast_engine 会优先用它）

不改任何被收编模块（ast_engine.py / injector_pkg / classifier.py）的内部逻辑，
只在这里集中赋默认值并把 LIBCLANG_PATH 注入 os.environ。
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_path(key: str, default: str | None) -> Path:
    val = os.environ.get(key)
    if val:
        return Path(val).resolve()
    return Path(default).resolve() if default else Path('__UNSET__').resolve()


# --- 源码树 ---------------------------------------------------------------
# VK-GL-CTS 根目录。
SOURCE_ROOT: Path = _env_path("CTS_SOURCE_ROOT", None)

VK_MODULES_ROOT: Path = SOURCE_ROOT / "external" / "vulkancts" / "modules"

# 构建目录（ast_engine 从 .vcxproj / compile_commands 推导编译参数）。
# 构建目录由 CTS_BUILD_ROOT 显式提供。
BUILD_ROOT: Path = _env_path("CTS_BUILD_ROOT", None)

# --- 输出 -----------------------------------------------------------------
OUTPUT_ROOT: Path = _env_path(
    "CTS_OUTPUT_ROOT",
    str(Path(__file__).resolve().parent / "output"),
)
VK_OUTPUT_ROOT: Path = OUTPUT_ROOT / "vulkan_dataset_v2"

# --- libclang -------------------------------------------------------------
# 若显式设了 LIBCLANG_PATH，透传给 ast_engine（它已支持读该 env）。
# 未设则 ast_engine 会回退到其内置候选路径 + 系统自动发现。
_LIBCLANG = os.environ.get("LIBCLANG_PATH")
if _LIBCLANG:
    os.environ["LIBCLANG_PATH"] = str(Path(_LIBCLANG).resolve())

# --- MSBuild 配置偏好（compile args 推导用） ------------------------------
PREFERRED_CONFIGURATIONS = [
    "Debug|x64",
    "RelWithDebInfo|x64",
    "Release|x64",
    "MinSizeRel|x64",
]


def ensure_output_dirs() -> None:
    """创建输出根目录（幂等）。"""
    VK_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def ensure_vk_prerequisites() -> None:
    """Validate the user-owned CTS checkout before any AST work starts."""
    missing = []
    if not os.environ.get("CTS_SOURCE_ROOT"):
        missing.append("CTS_SOURCE_ROOT")
    elif not SOURCE_ROOT.is_dir():
        raise RuntimeError(f"CTS_SOURCE_ROOT does not exist: {SOURCE_ROOT}")
    if not os.environ.get("CTS_BUILD_ROOT"):
        missing.append("CTS_BUILD_ROOT")
    elif not BUILD_ROOT.is_dir():
        raise RuntimeError(f"CTS_BUILD_ROOT does not exist: {BUILD_ROOT}")
    if missing:
        raise RuntimeError("Missing required environment variable(s): " + ", ".join(missing))
    if not VK_MODULES_ROOT.is_dir():
        raise RuntimeError(
            f"Vulkan CTS modules directory not found: {VK_MODULES_ROOT}. "
            "Check that CTS_SOURCE_ROOT points to the VK-GL-CTS checkout root."
        )
    nested_vk_root = VK_MODULES_ROOT / "vulkan"
    if not nested_vk_root.is_dir():
        raise RuntimeError(
            f"Vulkan module root not found: {nested_vk_root}. "
            "The extractor expects external/vulkancts/modules/vulkan."
        )


def describe() -> str:
    """打印当前生效配置，便于排查路径问题。"""
    return (
        f"SOURCE_ROOT      = {SOURCE_ROOT}\n"
        f"VK_MODULES_ROOT  = {VK_MODULES_ROOT}  (exists={VK_MODULES_ROOT.exists()})\n"
        f"BUILD_ROOT       = {BUILD_ROOT}  (exists={BUILD_ROOT.exists() if BUILD_ROOT else None})\n"
        f"OUTPUT_ROOT      = {OUTPUT_ROOT}\n"
        f"LIBCLANG_PATH    = {os.environ.get('LIBCLANG_PATH', '(auto-detect)')}"
    )


if __name__ == "__main__":
    print(describe())
