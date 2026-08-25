"""在 CTS 树里创建自定义模块骨架，并登记进父 CMakeLists。

为什么需要这一步：`cts.py` 注入候选代码之前会先 `_clean_old_test_cases`，
而那一步要求模块目录已经存在 —— 不存在直接 `FileNotFoundError: module dir
not found`。原工程里模块是人工建好的（`only_fix_*` / `director_*` 五套已在树里），
只有 shot-only-fix 那五个包由 `stage2/tool/ensure_shot_only_fix_packages.py`
生成。那个脚本靠克隆 `only_fix_{model}` 来造包，还带三条硬编码 CTS 根，两点
都不能随包发布：拿到一棵干净 VK-GL-CTS 检出的人树里没有任何 `only_fix_*`。

所以这里改成**从模板渲染**，不依赖树里已有任何自定义包。渲染出的 7 个文件与
既有包逐字段同构（已比对 `only_fix_claude` 与 `director_claude`，两者结构一致）：

    CMakeLists.txt
    vkt{Prefix}Tests.{hpp,cpp}              createTests 入口
    group/vkt{Prefix}Group.{hpp,cpp}        createGroups
    test_case/vkt{Prefix}SmokeTests.{hpp,cpp}   一个恒过的 smoke case

`cts.py` 的两个 CMake 渲染函数无条件把 `test_case/vkt{Prefix}SmokeTests.cpp`
列进 SRCS，所以 smoke 这一对文件是必需的，不是可选装饰。

写 CTS 树是有副作用的操作：默认只做检查，`--apply` 才落盘。父 CMakeLists 在
修改前备份为 `CMakeLists.txt.gl2vkframe.bak`，`--revert` 按备份还原并删除本
工具建的包目录。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import methods

MODELS = ("claude", "deepseek", "gemini", "glm", "gpt")

CMAKE_BACKUP_NAME = "CMakeLists.txt.gl2vkframe.bak"
REGISTRY_BACKUP_NAME = "vktTestPackage.cpp.gl2vkframe.bak"
STATE_NAME = ".gl2vkframe_state.json"

# 本工具建过的包在目录里留一个标记，--revert 只删自己建的，不碰人工建的包。
MARKER_NAME = ".gl2vkframe_created"


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── 命名 ────────────────────────────────────────────────────────────────
# 与 methods.py 同源：包名走 methods.package_name，类前缀是它的 CamelCase。
# 这里不再自己拼方法名字面量，否则统一改名时会漏掉这一处。

def package_name(slug: str, model: str) -> str:
    return methods.package_name(slug, model)


def class_prefix(slug: str, model: str) -> str:
    parts = package_name(slug, model).split("_")
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def target_name(slug: str, model: str) -> str:
    return methods.target_name(slug, model)


# ── 渲染 ────────────────────────────────────────────────────────────────

def _render_cmake(slug: str, model: str) -> str:
    prefix = class_prefix(slug, model)
    var = prefix.upper()
    target = target_name(slug, model)
    target_sc = target.replace("deqp-vk-", "deqp-vksc-")
    return f"""include_directories(
    ..
    .
    group
    test_case
    ${{DEQP_INL_DIR}}
    )

set({var}_SRCS
    vkt{prefix}Tests.cpp
    group/vkt{prefix}Group.cpp
    test_case/vkt{prefix}SmokeTests.cpp
    )

PCH({var}_SRCS ../pch.cpp)

add_library({target} STATIC ${{{var}_SRCS}})
target_link_libraries({target} PUBLIC tcutil vkutil)

add_library({target_sc} STATIC ${{{var}_SRCS}})
target_include_directories({target_sc} BEFORE PUBLIC ${{CMAKE_SOURCE_DIR}}/external/vulkancts/framework/vulkan/generated/vulkansc)
target_link_libraries({target_sc} PUBLIC deqp-vksc-util tcutil vkscutil)
"""


def _render_tests_hpp(slug: str, model: str) -> str:
    prefix = class_prefix(slug, model)
    ns = package_name(slug, model)
    guard = f"_VKT{prefix.upper()}TESTS_HPP"
    return f"""#ifndef {guard}
#define {guard}

#include "tcuDefs.hpp"
#include "tcuTestCase.hpp"

namespace tcu
{{
class TestCaseGroup;
class TestContext;
}} // namespace tcu

namespace vkt
{{
namespace {ns}
{{

tcu::TestCaseGroup* createTests(tcu::TestContext& testCtx, const std::string& name);

}} // namespace {ns}
}} // namespace vkt

#endif // {guard}
"""


def _render_tests_cpp(slug: str, model: str) -> str:
    prefix = class_prefix(slug, model)
    ns = package_name(slug, model)
    return f"""#include "vkt{prefix}Tests.hpp"

#include "group/vkt{prefix}Group.hpp"
#include "vktTestGroupUtil.hpp"

namespace vkt
{{
namespace {ns}
{{

tcu::TestCaseGroup* createTests(tcu::TestContext& testCtx, const std::string& name)
{{
    return createTestGroup(testCtx, name, create{prefix}Groups);
}}

}} // namespace {ns}
}} // namespace vkt
"""


def _render_group_hpp(slug: str, model: str) -> str:
    prefix = class_prefix(slug, model)
    ns = package_name(slug, model)
    guard = f"_VKT{prefix.upper()}GROUP_HPP"
    return f"""#ifndef {guard}
#define {guard}

#include "tcuDefs.hpp"
#include "tcuTestCase.hpp"

namespace tcu
{{
class TestCaseGroup;
}} // namespace tcu

namespace vkt
{{
namespace {ns}
{{

void create{prefix}Groups(tcu::TestCaseGroup* group);

}} // namespace {ns}
}} // namespace vkt

#endif // {guard}
"""


def _render_group_cpp(slug: str, model: str) -> str:
    """初始 group.cpp。

    SmokeTests.cpp 仍然保留在 CMake 源文件列表中，作为 package 骨架的编译
    锚点；但不能把 smoke group 注册到 root。runtime 会把生成的 leaf case
    直接挂到这个 root，而 CTS 要求同一 TestCaseGroup 的所有 child 属于同一
    node class（group 与 executable case 不能混用）。
    """
    prefix = class_prefix(slug, model)
    ns = package_name(slug, model)
    return f"""#include "vkt{prefix}Group.hpp"
#include "vkt{prefix}SmokeTests.hpp"
#include "vktTestGroupUtil.hpp"

namespace vkt
{{
namespace {ns}
{{

void create{prefix}Groups(tcu::TestCaseGroup* group)
{{
    DE_UNREF(group);
}}

}} // namespace {ns}
}} // namespace vkt
"""


def _render_smoke_hpp(slug: str, model: str) -> str:
    prefix = class_prefix(slug, model)
    ns = package_name(slug, model)
    guard = f"_VKT{prefix.upper()}SMOKETESTS_HPP"
    return f"""#ifndef {guard}
#define {guard}

#include "tcuDefs.hpp"
#include "tcuTestCase.hpp"

namespace tcu
{{
class TestCaseGroup;
}} // namespace tcu

namespace vkt
{{
class Context;

namespace {ns}
{{

void populate{prefix}SmokeTests(tcu::TestCaseGroup* group);
tcu::TestStatus {ns}_smoke(Context& context);

}} // namespace {ns}
}} // namespace vkt

#endif // {guard}
"""


def _render_smoke_cpp(slug: str, model: str) -> str:
    prefix = class_prefix(slug, model)
    ns = package_name(slug, model)
    return f"""#include "test_case/vkt{prefix}SmokeTests.hpp"

#include "deDefs.hpp"
#include "vktTestCaseUtil.hpp"

namespace vkt
{{
namespace {ns}
{{

void populate{prefix}SmokeTests(tcu::TestCaseGroup* group)
{{
    addFunctionCase(group, "basic", {ns}_smoke);
}}

tcu::TestStatus {ns}_smoke(Context& context)
{{
    DE_UNREF(context);
    return tcu::TestStatus::pass("{ns}_smoke");
}}

}} // namespace {ns}
}} // namespace vkt
"""


def module_files(slug: str, model: str) -> dict[str, str]:
    prefix = class_prefix(slug, model)
    return {
        "CMakeLists.txt": _render_cmake(slug, model),
        f"vkt{prefix}Tests.hpp": _render_tests_hpp(slug, model),
        f"vkt{prefix}Tests.cpp": _render_tests_cpp(slug, model),
        f"group/vkt{prefix}Group.hpp": _render_group_hpp(slug, model),
        f"group/vkt{prefix}Group.cpp": _render_group_cpp(slug, model),
        f"test_case/vkt{prefix}SmokeTests.hpp": _render_smoke_hpp(slug, model),
        f"test_case/vkt{prefix}SmokeTests.cpp": _render_smoke_cpp(slug, model),
    }


# ── CTS 树 ──────────────────────────────────────────────────────────────

def vulkan_dir(cts_root: Path) -> Path:
    return cts_root / "external" / "vulkancts" / "modules" / "vulkan"


def check_cts_root(cts_root: Path) -> Path:
    """确认这是一棵可用的 VK-GL-CTS 检出，且已 configure 过。

    宁可在这里报错，也不要在半程失败后留下写坏的树。
    """
    vk = vulkan_dir(cts_root)
    if not (vk / "CMakeLists.txt").is_file():
        raise SystemExit(
            f"不是有效的 VK-GL-CTS 检出：缺少 {vk / 'CMakeLists.txt'}")
    if not (vk / "vktTestPackage.cpp").is_file():
        raise SystemExit(
            f"不是有效的 VK-GL-CTS 检出：缺少 {vk / 'vktTestPackage.cpp'}")
    return cts_root


def _write_exact(path: Path, text: str) -> None:
    """按原样写盘，不做换行转换 —— CTS 的 CMakeLists 有自己的换行风格。"""
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def _packages(slugs: Iterable[str], models: Iterable[str]) -> list[tuple[str, str]]:
    return [(slug, model) for slug in slugs for model in models]


def _insert_after_last(lines: list[str], predicate, additions: list[str]) -> list[str]:
    """插在最后一个锚点之后。找不到锚点就报错，不猜位置。"""
    indexes = [i for i, line in enumerate(lines) if predicate(line)]
    if not indexes:
        raise RuntimeError("在父 CMakeLists 里找不到插入锚点")
    at = indexes[-1] + 1
    return lines[:at] + additions + lines[at:]


def patch_parent_cmake(text: str, pkgs: list[tuple[str, str]]) -> str:
    """把包登记进父 CMakeLists 的四个位置：子目录、include、VK 库、VKSC 库。

    四处都是幂等的：已存在的行不重复插入。锚点用 CTS 自带的 `custom_migration`
    / `api` 等固有条目，不依赖任何自定义包 —— 干净检出上也能找到。
    """
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    indent = "\t"

    names = [package_name(slug, model) for slug, model in pkgs]
    targets = [target_name(slug, model) for slug, model in pkgs]

    missing_subdirs = [f"add_subdirectory({n})" for n in names
                       if f"add_subdirectory({n})" not in text]
    if missing_subdirs:
        lines = _insert_after_last(
            lines,
            lambda ln: bool(re.fullmatch(r"add_subdirectory\([A-Za-z0-9_]+\)", ln.strip())),
            missing_subdirs,
        )

    stripped = {ln.strip() for ln in lines}
    missing_includes = [f"{indent}{n}" for n in names if n not in stripped]
    if missing_includes:
        lines = _insert_after_last(
            lines,
            lambda ln: ln.startswith((indent, "    ")) and ln.strip() == "api",
            missing_includes,
        )

    missing_vk = [f"{indent}{t}" for t in targets if t not in text]
    if missing_vk:
        marker = next(
            (i for i, ln in enumerate(lines) if ln.strip() == "list(APPEND DEQP_VK_LIBS"),
            None)
        if marker is None:
            raise RuntimeError("在父 CMakeLists 里找不到 list(APPEND DEQP_VK_LIBS")
        lines[marker + 1:marker + 1] = missing_vk

    sc_targets = [t.replace("deqp-vk-", "deqp-vksc-") for t in targets]
    missing_sc = [f"{indent}{t}" for t in sc_targets if t not in text]
    if missing_sc:
        marker = next(
            (i for i, ln in enumerate(lines) if ln.strip() == "set(DEQP_VKSC_LIBS"),
            None)
        if marker is None:
            raise RuntimeError("在父 CMakeLists 里找不到 set(DEQP_VKSC_LIBS")
        lines[marker + 1:marker + 1] = missing_sc

    return newline.join(lines) + newline


def patch_test_package(text: str, pkgs: list[tuple[str, str]]) -> str:
    """Persist package includes and root registrations in vktTestPackage.cpp."""
    patched = text
    newline = "\r\n" if "\r\n" in patched else "\n"
    include_matches = list(re.finditer(
        r'^#include\s+"vkt[A-Za-z0-9_]+Tests\.hpp"[ \t]*$',
        patched,
        flags=re.MULTILINE,
    ))
    if not include_matches:
        raise RuntimeError(
            "vktTestPackage.cpp has no Vulkan test header include anchor"
        )

    include_lines: list[str] = []
    registration_lines: list[str] = []
    for slug, model in pkgs:
        name = package_name(slug, model)
        prefix = class_prefix(slug, model)
        include_line = f'#include "vkt{prefix}Tests.hpp"'
        registration_line = (
            f'    addRootChild("{name}", m_caseListFilter, '
            f'{name}::createTests);'
        )
        if include_line not in patched:
            include_lines.append(include_line)
        if registration_line not in patched:
            registration_lines.append(registration_line)

    if include_lines:
        anchor = include_matches[-1]
        patched = (
            patched[:anchor.end()]
            + newline
            + newline.join(include_lines)
            + patched[anchor.end():]
        )
    if registration_lines:
        # `info` is registered in both the Vulkan and Vulkan SC branches. Add the
        # new roots after every active occurrence so the shared source remains
        # valid in both configurations without depending on a version-specific
        # test group near the end of either branch.
        info_pattern = re.compile(
            r'^(?P<line>[ \t]*addRootChild\("info"[^\r\n]*\);)[ \t]*$',
            flags=re.MULTILINE,
        )
        anchors = list(info_pattern.finditer(patched))
        if not anchors:
            raise RuntimeError(
                "vktTestPackage.cpp has no active info addRootChild anchor"
            )
        additions = newline.join(registration_lines)
        patched = info_pattern.sub(
            lambda match: match.group("line") + newline + additions,
            patched,
        )
    return patched


def inspect(cts_root: Path, pkgs: list[tuple[str, str]]) -> dict[str, Any]:
    vk = vulkan_dir(cts_root)
    text = (vk / "CMakeLists.txt").read_text(encoding="utf-8")
    registry_text = (vk / "vktTestPackage.cpp").read_text(encoding="utf-8")
    info_registration_count = len(re.findall(
        r'^\s*addRootChild\(\"info\"[^\r\n]*\);\s*$',
        registry_text,
        flags=re.MULTILINE,
    ))
    records = []
    for slug, model in pkgs:
        name = package_name(slug, model)
        target = target_name(slug, model)
        module_dir = vk / name
        expected = module_files(slug, model)
        present = module_dir.is_dir()
        missing = sorted(f for f in expected if not (module_dir / f).is_file()) if present else sorted(expected)
        include_line = f'#include "vkt{class_prefix(slug, model)}Tests.hpp"'
        registration_line = (
            f'addRootChild("{name}", m_caseListFilter, {name}::createTests);'
        )
        records.append({
            "package": name,
            "method_slug": slug,
            "model": model,
            "dir_exists": present,
            "missing_files": missing,
            "created_by_this_tool": (module_dir / MARKER_NAME).is_file(),
            "cmake_subdir": f"add_subdirectory({name})" in text,
            "cmake_subdir_count": text.count(f"add_subdirectory({name})"),
            "cmake_include": any(ln.strip() == name for ln in text.splitlines()),
            "cmake_include_count": sum(ln.strip() == name for ln in text.splitlines()),
            "cmake_vk_lib": target in text,
            "cmake_vk_lib_count": text.count(target),
            "cmake_vksc_lib": target.replace("deqp-vk-", "deqp-vksc-") in text,
            "cmake_vksc_lib_count": text.count(target.replace("deqp-vk-", "deqp-vksc-")),
            "registry_include": include_line in registry_text,
            "registry_include_count": registry_text.count(include_line),
            "registry_registration": registration_line in registry_text,
            "registry_registration_count": registry_text.count(registration_line),
            "expected_registration_count": info_registration_count,
        })
        record = records[-1]
        chain_values = [
            record["dir_exists"],
            not record["missing_files"],
            record["cmake_subdir"],
            record["cmake_include"],
            record["cmake_vk_lib"],
            record["cmake_vksc_lib"],
            record["registry_include"],
            record["registry_registration"],
        ]
        duplicate_registration = (
            record["cmake_subdir_count"] > 1
            or record["cmake_include_count"] > 1
            or record["cmake_vk_lib_count"] > 1
            or record["cmake_vksc_lib_count"] > 1
            or record["registry_include_count"] != 1
            or (
                info_registration_count > 0
                and record["registry_registration_count"]
                not in {0, info_registration_count}
            )
        )
        if not any(chain_values):
            record["state"] = "absent"
        elif duplicate_registration:
            record["state"] = "conflict"
            record["conflict_reason"] = (
                "CTS parent/package registration is duplicated or inconsistent: "
                f"subdir={record['cmake_subdir_count']} "
                f"include={record['cmake_include_count']} "
                f"vk_lib={record['cmake_vk_lib_count']} "
                f"vksc_lib={record['cmake_vksc_lib_count']} "
                f"registry_include={record['registry_include_count']} "
                f"registration={record['registry_registration_count']} "
                f"expected={info_registration_count}"
            )
        elif all(chain_values) and record["created_by_this_tool"]:
            record["state"] = "managed_ready"
        elif all(chain_values):
            record["state"] = "conflict"
            record["conflict_reason"] = (
                "同名 CTS 包及其注册结构已存在，但没有本工具 marker；"
                "为保护已有代码，拒绝覆盖或复用"
            )
        elif record["dir_exists"] and not record["created_by_this_tool"]:
            record["state"] = "conflict"
        else:
            record["state"] = "partial"
    ready = all(
        r["state"] in {"absent", "managed_ready"}
        and r["dir_exists"] and not r["missing_files"] and r["cmake_subdir"]
        and r["cmake_include"] and r["cmake_vk_lib"] and r["cmake_vksc_lib"]
        and r["registry_include"] and r["registry_registration"]
        for r in records
    )
    return {"cts_root": str(cts_root), "ready": ready, "packages": records}


def apply(cts_root: Path, pkgs: list[tuple[str, str]]) -> dict[str, Any]:
    """建包 + 登记。任一步失败就把树还原到调用前的样子。"""
    vk = vulkan_dir(cts_root)
    cmake = vk / "CMakeLists.txt"
    registry = vk / "vktTestPackage.cpp"
    preflight = inspect(cts_root, pkgs)
    # 建包是“从零开始”的事务：只允许目标完全不存在。即使是本工具
    # 以前留下的 managed_ready 包，也不自动复用，避免把旧状态误当成
    # 当前事务的输入；调用方应先显式 --revert，再重新 apply。
    blocked = [
        record for record in preflight["packages"]
        if record["state"] != "absent"
    ]
    if blocked:
        raise RuntimeError(
            "refusing to modify existing or partial/conflicting CTS packages: "
            + json.dumps(blocked, ensure_ascii=False)
        )
    before = cmake.read_text(encoding="utf-8")
    registry_before = registry.read_text(encoding="utf-8")

    backup = cmake.with_name(CMAKE_BACKUP_NAME)
    if not backup.exists():
        _write_exact(backup, before)
    registry_backup = registry.with_name(REGISTRY_BACKUP_NAME)
    if not registry_backup.exists():
        _write_exact(registry_backup, registry_before)
    state_path = vk / STATE_NAME
    previous_state = state_path.read_text(encoding="utf-8") if state_path.exists() else None

    created: list[Path] = []
    staging: list[Path] = []
    try:
        for slug, model in pkgs:
            destination = vk / package_name(slug, model)
            if destination.exists():
                continue
            # 先写进同盘临时目录再原子改名：中途失败不会留下半个包，
            # 而半个包会被 cts.py 当成可用模块然后编译失败。
            tmp = Path(tempfile.mkdtemp(
                prefix=f".{package_name(slug, model)}.", dir=str(vk)))
            staging.append(tmp)
            for relative, text in module_files(slug, model).items():
                target = tmp / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
            (tmp / MARKER_NAME).write_text(
                "created by GL2VKFrame ctspkg.py --apply\n", encoding="utf-8")
            os.replace(tmp, destination)
            staging.remove(tmp)
            created.append(destination)

        patched = patch_parent_cmake(before, pkgs)
        if patched != before:
            tmp_cmake = cmake.with_name("CMakeLists.txt.gl2vkframe.tmp")
            _write_exact(tmp_cmake, patched)
            os.replace(tmp_cmake, cmake)

        registry_patched = patch_test_package(registry_before, pkgs)
        if registry_patched != registry_before:
            tmp_registry = registry.with_name("vktTestPackage.cpp.gl2vkframe.tmp")
            _write_exact(tmp_registry, registry_patched)
            os.replace(tmp_registry, registry)

        state_path.write_text(
            json.dumps(
                {
                    "cmake_managed_sha256": _text_sha256(cmake.read_text(encoding="utf-8")),
                    "registry_managed_sha256": _text_sha256(registry.read_text(encoding="utf-8")),
                    "packages": [package_name(slug, model) for slug, model in pkgs],
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        state = inspect(cts_root, pkgs)
        if not state["ready"]:
            raise RuntimeError(f"后置条件未满足: {json.dumps(state, ensure_ascii=False)}")
        state["created_packages"] = [p.name for p in created]
        return state
    except Exception:
        _write_exact(cmake, before)
        _write_exact(registry, registry_before)
        for path in reversed(created):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
        if previous_state is None:
            state_path.unlink(missing_ok=True)
        else:
            state_path.write_text(previous_state, encoding="utf-8")
        raise
    finally:
        for path in staging:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)


def revert(cts_root: Path, pkgs: list[tuple[str, str]]) -> dict[str, Any]:
    """按备份还原父 CMakeLists，并只删本工具建的包目录。"""
    vk = vulkan_dir(cts_root)
    cmake = vk / "CMakeLists.txt"
    registry = vk / "vktTestPackage.cpp"
    backup = cmake.with_name(CMAKE_BACKUP_NAME)
    registry_backup = registry.with_name(REGISTRY_BACKUP_NAME)
    state_path = vk / STATE_NAME

    if backup.is_file() or registry_backup.is_file():
        if not state_path.is_file():
            raise RuntimeError(
                "refusing to restore backups without the GL2VKFrame managed-state file"
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        current_hashes = {
            "cmake_managed_sha256": _text_sha256(cmake.read_text(encoding="utf-8")),
            "registry_managed_sha256": _text_sha256(registry.read_text(encoding="utf-8")),
        }
        changed = [
            name for name, value in current_hashes.items()
            if state.get(name) != value
        ]
        if changed:
            raise RuntimeError(
                "refusing to revert because managed CTS parent files changed after apply: "
                + ", ".join(changed)
            )

    removed, kept = [], []
    for slug, model in pkgs:
        module_dir = vk / package_name(slug, model)
        if not module_dir.is_dir():
            continue
        if (module_dir / MARKER_NAME).is_file():
            shutil.rmtree(module_dir, ignore_errors=True)
            removed.append(module_dir.name)
        else:
            # 人工建的包不动：删掉别人正在用的模块比留下多余登记严重得多。
            kept.append(module_dir.name)

    restored = False
    if backup.is_file():
        _write_exact(cmake, backup.read_text(encoding="utf-8"))
        backup.unlink()
        restored = True

    registry_restored = False
    if registry_backup.is_file():
        _write_exact(registry, registry_backup.read_text(encoding="utf-8"))
        registry_backup.unlink()
        registry_restored = True

    if not backup.exists() and not registry_backup.exists():
        state_path.unlink(missing_ok=True)

    return {
        "cts_root": str(cts_root),
        "cmake_restored_from_backup": restored,
        "registry_restored_from_backup": registry_restored,
        "removed_packages": removed,
        "kept_not_ours": kept,
    }


# ── CLI ─────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在 CTS 树里创建自定义模块骨架并登记进父 CMakeLists。")
    parser.add_argument(
        "--cts-root", required=True,
        help="VK-GL-CTS 检出路径。必需，没有默认值。")
    parser.add_argument(
        "--methods", nargs="+", default=sorted(methods.METHODS),
        choices=methods.METHOD_INPUT_NAMES,
        help="要建包的修复线，默认全部三条。")
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument(
        "--seeds", action="store_true",
        help="同时为两条种子线建包（Step 2 编译种子时需要）。")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--apply", action="store_true", help="真正写 CTS 树。")
    group.add_argument("--revert", action="store_true",
                       help="还原父 CMakeLists 并删除本工具建的包。")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cts_root = check_cts_root(Path(args.cts_root).resolve())
    slugs = [methods.get_method(key).slug for key in args.methods]
    pkgs = _packages(slugs, args.models)
    if args.seeds:
        pkgs += [
            (f"seed_{kind}", model)
            for kind in methods.SEED_KINDS
            for model in args.models
        ]

    if args.revert:
        result = revert(cts_root, pkgs)
    elif args.apply:
        result = apply(cts_root, pkgs)
    else:
        result = inspect(cts_root, pkgs)
        result["note"] = "只做检查，未写盘。加 --apply 才落盘。"

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.revert and not result.get("ready", True):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
