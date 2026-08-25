"""GL2VKFrame 唯一入口。

三步流程：

    Step 1  generate   两条种子线（non_cot / cot）—— 基线 ① 与 ②
    Step 2  compile    每条种子只编一次，判据落盘供下游共用
    Step 3  repair     三条修复线（w/o cot / single-agent / multi-agent）
    可选    runtime    挂载进 CTS 树、构建、执行、按 qpa 判定、还原

依赖顺序由本文件强制：repair 的 attempt_1 就是对应种子，缺种子直接报错，
不会静默回退到另一种种子。

    python pipeline.py --stage generate --model claude --rows 44,59
    python pipeline.py --stage compile  --model claude --cts-root <PATH>
    python pipeline.py --stage repair   --model claude --methods multi_agent --cts-root <PATH>
    python pipeline.py --stage all      --model claude --cts-root <PATH>

--cts-root 没有默认值。Step 2/3/runtime 都要它：修复提示词的输入就是真实编译
错误，所以「只用 API key 就能跑」仅对 Step 1 成立。

并发：--stage all 默认串行。三条修复线不能共用一棵 CTS 树（挂载会互相覆盖），
要并行就为每条线各给一棵树，本文件不会自行寻找第二棵。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import core
import methods
import paths
import model_adapter
from input_contract import load_records
from model_adapter import MODELS
from methods import (
    METHODS, METHOD_INPUT_NAMES, SEED_INPUT_NAMES, SEED_KINDS,
    get_method, seed_display_name,
)

STAGES = ("generate", "compile", "repair", "runtime", "all")


def log(message: str) -> None:
    print(message, flush=True)


def parse_rows(spec: str | None) -> list[int] | None:
    """解析 --rows。支持 44,59 与 100-120 混合。"""
    if not spec:
        return None
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def load_dataset_rows(input_path: Path) -> list[int]:
    rows = load_records(input_path)
    return sorted(int(r["row_id"]) for r in rows)


def resolve_rows(args: argparse.Namespace) -> list[int]:
    """确定要跑的 row_id 集合，并校验都在数据集内。"""
    input_path = Path(args.input_json).resolve()
    available = load_dataset_rows(input_path)
    requested = parse_rows(args.rows)
    if requested is None:
        rows = available
    else:
        unknown = [r for r in requested if r not in set(available)]
        if unknown:
            raise SystemExit(
                f"以下 row_id 不在 {input_path.name} 中: {unknown}\n"
                f"数据集共 {len(available)} 行，范围 {available[0]}..{available[-1]}。")
        rows = requested
    if args.limit:
        rows = rows[: args.limit]
    return rows


def require_cts_root(args: argparse.Namespace, stage: str) -> Path:
    if not args.cts_root:
        raise SystemExit(
            f"--stage {stage} 需要 --cts-root <VK-GL-CTS 检出路径>。\n"
            f"本包不猜测 CTS 位置：那些树可能正被其它流程占用，"
            f"猜错会把代码注入正在使用的树。")
    root = Path(args.cts_root).resolve()
    if not root.is_dir():
        raise SystemExit(f"--cts-root 不存在或不是目录: {root}")
    return root


def require_cts_packages(cts_root: Path, slugs: list[str],
                         model: str) -> None:
    """确认要用的 CTS 模块骨架已存在。

    cts.py 注入候选前会先重置模块，那一步要求模块目录已经存在，否则抛
    `module dir not found`。原工程里这些模块是预先建好的（`only_fix_*` /
    `director_*` 在树里，shot-only-fix 那五个由 stage2 的一个工具生成），
    干净检出上一个都没有。与其让用户撞上那句底层报错，这里先检查并给出
    建包命令。
    """
    import ctspkg

    ctspkg.check_cts_root(cts_root)
    pkgs = [(slug, model) for slug in slugs]
    state = ctspkg.inspect(cts_root, pkgs)
    if state["ready"]:
        return
    missing = [r["package"] for r in state["packages"]
               if not r["dir_exists"] or r["missing_files"]
               or not (r["cmake_subdir"] and r["cmake_include"]
                       and r["cmake_vk_lib"] and r["cmake_vksc_lib"])]
    raise SystemExit(
        f"CTS 树里缺少所需模块骨架: {', '.join(missing)}\n"
        f"先建包（会写 CTS 树，默认只检查）：\n"
        f"    python ctspkg.py --cts-root {cts_root} --models {model} --seeds --apply\n"
        f"结束后可用 --revert 还原。")


# ── Step 1 ──────────────────────────────────────────────────────────────

def stage_generate(args: argparse.Namespace, rows: list[int]) -> int:
    """两条种子线各生成一次。不需要 CTS。"""
    import generate as gen_mod

    failed = 0
    for seed_name in args.seeds:
        seed_kind = core.resolve_seed_kind(seed_name)
        mode = "non-cot" if seed_kind == "non_cot" else "cot"
        log(f"\n=== Step 1 生成  method={seed_display_name(seed_kind)}  model={args.model}  n={len(rows)} ===")
        argv = ["--mode", mode, "--model", args.model,
                "--input-json", str(Path(args.input_json).resolve())]
        if args.dry_run:
            argv.append("--dry-run")
        for row_id in rows:
            rc = _run_module_main(gen_mod, argv + ["--row-id", str(row_id)])
            if rc:
                failed += 1
    return 1 if failed else 0


def _run_module_main(module: Any, argv: list[str]) -> int:
    """用给定 argv 调用子模块的 main()。子模块自己解析参数。"""
    saved = sys.argv
    try:
        sys.argv = [module.__name__] + argv
        rc = module.main()
        return int(rc or 0)
    except SystemExit as exc:
        return int(exc.code or 0)
    finally:
        sys.argv = saved


# ── Step 2 ──────────────────────────────────────────────────────────────

def stage_compile(args: argparse.Namespace, rows: list[int]) -> int:
    """编译两条种子。同一份种子只编一次，判据供三条修复线共用。

    判据以 error_summary.json 的 errors 是否为空为准 —— 已在 4274 条历史记录上
    验证与编译通过完全等价、零交叉，且不依赖 compile_record.json 是否存在。
    """
    import cts
    from methods import seed_package_name

    cts_root = require_cts_root(args, "compile")
    # 种子编译用中性包名 seed_{kind}_{model}，所以要检查的是那几个包。
    if not args.dry_run:
        require_cts_packages(
            cts_root, [f"seed_{core.resolve_seed_kind(kind)}" for kind in args.seeds], args.model)
    failed = 0
    for seed_name in args.seeds:
        seed_kind = core.resolve_seed_kind(seed_name)
        log(f"\n=== Step 2 编译种子  seed={seed_kind}  model={args.model}  n={len(rows)} ===")
        for row_id in rows:
            seed_cpp = paths.seed_case_dir(seed_kind, args.model, row_id) / "raw_generated.cpp"
            if not seed_cpp.is_file():
                log(f"  [SKIP] row {row_id}: 缺少种子 {seed_cpp}，先跑 --stage generate")
                failed += 1
                continue
            log(f"  [COMPILE] {seed_kind}/{args.model}/case_{row_id}")
            if args.dry_run:
                continue
            # 种子编译用中性包名，不挪用任何一条方法线的包名。cts_root / 包名 / 目录
            # 都走 override：run_case_compile 没有对应形参，内部靠 resolve_* 读取。
            #
            # 读种子用 candidate_root，写产物用 output_root，两者必须分开：
            # prepare_case(force=True) 会删 case 根的 raw_generated.cpp，指同一个
            # 目录就会吃掉 Step 1 的种子。
            #
            # 两个 override 的层级不一样，别对齐着写：
            #   candidate_root -> <root>/{model}/case_N   给到 seed_kind 层
            #   output_root    -> <root>/case_N           给到 model 层
            core.set_execution_overrides(
                cts_root=cts_root,
                package_name=seed_package_name(seed_kind, args.model),
                candidate_root=paths.seed_case_dir(seed_kind, args.model, row_id).parent.parent,
                output_root=paths.seed_compile_dir(seed_kind, args.model, row_id).parent,
                gl_sim_vk_path=Path(args.input_json).resolve(),
            )
            try:
                # 编译要 normalized/assembled_candidate.cpp 与 meta/classification.json，
                # 而 Step 1 只落 raw_generated.cpp —— prepare_case 补齐这一层。
                core.prepare_case(mode_name=seed_kind, model_name=args.model,
                                  row_id=row_id, force=True)
                compile_result = cts.run_case_compile(
                    mode_name=seed_kind,
                    model_name=args.model,
                    row_id=row_id,
                    attempt_index=1,
                )
                summary = cts.summarize_compile_logs(
                    compile_result.stdout_path,
                    compile_result.stderr_path,
                )
                summary["build_returncode"] = int(compile_result.returncode)
                summary_path = Path(compile_result.stdout_path).parent / "error_summary.json"
                cts.write_summary(summary, summary_path)
                log(
                    f"    [RESULT] returncode={compile_result.returncode} "
                    f"errors={cts.summary_error_count(summary)} summary={summary_path}"
                )
            except Exception as exc:  # 单例失败不终止整批
                log(f"    [ERROR] {type(exc).__name__}: {exc}")
                failed += 1
            finally:
                core.clear_execution_overrides()
    return 1 if failed else 0


# ── Step 3 ──────────────────────────────────────────────────────────────

def stage_repair(args: argparse.Namespace, rows: list[int]) -> int:
    """三条修复线。attempt 从 2 起；attempt_1 是对应种子。"""
    import agent as agent_mod

    cts_root = require_cts_root(args, "repair")
    if not args.dry_run:
        require_cts_packages(
            cts_root, [get_method(k).slug for k in args.methods], args.model)
    failed = 0
    for method_key in args.methods:
        method = get_method(method_key)
        log(f"\n=== Step 3 修复  method={method.display}  seed={seed_display_name(method.seed)}  "
            f"repair={method.repair}  model={args.model}  n={len(rows)} ===")
        for row_id in rows:
            seed_cpp = paths.seed_case_dir(method.seed, args.model, row_id) / "raw_generated.cpp"
            if not seed_cpp.is_file():
                log(f"  [SKIP] row {row_id}: 缺少 {method.seed} 种子；"
                    f"本包不回退到另一种种子")
                failed += 1
                continue
            log(f"  [REPAIR] {method.slug}/{args.model}/case_{row_id}")
            if args.dry_run:
                continue
            # 与 Step 2 同理：run_agent_case 内部也会调 prepare_case，读种子与写
            # 产物必须分指两处，否则 force_prepare 时会删掉种子。产物目录用 slug
            # 而非 legacy —— legacy 只用于读旧产物。
            core.set_execution_overrides(
                cts_root=cts_root,
                package_name=methods.package_name(method.slug, args.model),
                candidate_root=paths.seed_case_dir(method.seed, args.model, row_id).parent.parent,
                output_root=paths.method_case_dir(method.slug, args.model, row_id).parent,
                gl_sim_vk_path=Path(args.input_json).resolve(),
            )
            try:
                agent_mod.run_agent_case(
                    mode_name=method.legacy, model_name=args.model, row_id=row_id,
                    max_attempts=args.max_attempts)
            except Exception as exc:
                log(f"    [ERROR] {type(exc).__name__}: {exc}")
                failed += 1
            finally:
                core.clear_execution_overrides()
    return 1 if failed else 0


# ── 可选：运行期 ────────────────────────────────────────────────────────

def stage_runtime(args: argparse.Namespace, rows: list[int]) -> int:
    """把编译通过的用例真的跑一遍，取 Pass/Fail/NotSupported。

    编译通过不等于用例正确：编译只进静态库、不链接，声明而未定义也能通过。
    既有产物里，编译通过的候选含无条件 pass("Pass") 的比例按模型为
    57.4%~90.7%（gemini 最高、claude 最低），编译率高的模型空壳占比也最高。

    这里只跑 plan（扫描 + 落计划，不写 CTS 树）。真正的挂载/构建/执行请直接用
    runtime.py —— 那几步会改 CTS 树，需要显式确认，不该被 --stage all 顺带触发。
    """
    import runtime as runtime_mod

    cts_root = require_cts_root(args, "runtime")
    log(f"\n=== 可选 运行期验证（plan）  cts_root={cts_root} ===")

    ns = argparse.Namespace(
        cts_root=str(cts_root),
        modes=[get_method(k).slug for k in args.methods],
        models=[args.model],
        rows=args.rows,
        limit=args.limit,
    )
    rc = runtime_mod.command_plan(ns)
    log("  挂载/构建/执行会往 CTS 树里写文件，需显式执行：")
    log(f"    python runtime.py --stage mount   --cts-root {cts_root}")
    log(f"    python runtime.py --stage build   --cts-root {cts_root}")
    log(f"    python runtime.py --stage execute --cts-root {cts_root}")
    log(f"    python runtime.py --restore   # 结束后按台账还原")
    return rc


# ── CLI ─────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GL2VKFrame：OpenGL CTS -> Vulkan CTS 迁移流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--stage", choices=STAGES, default="all",
                   help="要执行的阶段（默认 all，串行）")
    p.add_argument("--model", required=True, choices=sorted(MODELS),
                   help="论文模型键；实际模型 ID 由 --model-id 记录")
    p.add_argument("--input-json", required=True, type=Path,
                   help="retrieval/build_gl_sim_vk.py 生成的 Framework 输入")
    p.add_argument("--rows", default=None,
                   help="row_id 列表，如 44,59 或 100-120；缺省跑输入文件全部记录")
    p.add_argument("--limit", type=int, default=None, help="截取前 N 个 row")
    p.add_argument("--seeds", nargs="*", choices=SEED_INPUT_NAMES, default=list(SEED_KINDS),
                   help="Step 1/2 要处理的种子线")
    p.add_argument("--methods", nargs="*", choices=METHOD_INPUT_NAMES,
                   default=sorted(METHODS), help="Step 3 要跑的修复线")
    p.add_argument("--cts-root", default=None,
                   help="VK-GL-CTS 检出路径。Step 2/3/runtime 必需，无默认值")
    p.add_argument("--output-root", default=None,
                   help="产物根，缺省为包内 runs/")
    p.add_argument("--max-attempts", type=int, default=2,
                   help="修复轮数上限（attempt_2 起）")
    p.add_argument("--dry-run", action="store_true",
                   help="只走流程不调模型、不编译")
    model_adapter.add_runner_arguments(p)
    return p


def main() -> int:
    args = build_parser().parse_args()

    model_adapter.configure_from_args(
        args,
        required=args.stage in ("generate", "repair", "all") and not args.dry_run,
    )

    if args.output_root:
        paths.runs_root(Path(args.output_root))
    log(f"产物根: {paths.runs_root()}")

    rows = resolve_rows(args)
    log(f"模型: {MODELS[args.model]} ({args.model})   row 数: {len(rows)}")

    stage = args.stage
    rc = 0
    if stage == "generate":
        rc |= stage_generate(args, rows)
    elif stage == "compile":
        rc |= stage_compile(args, rows)
    elif stage == "repair":
        rc |= stage_repair(args, rows)
    elif stage == "runtime":
        rc |= stage_runtime(args, rows)
    elif stage == "all":
        rc = stage_generate(args, rows)
        if not rc:
            rc = stage_compile(args, rows)
        if not rc:
            rc = stage_repair(args, rows)

    log(f"\n完成，退出码 {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
