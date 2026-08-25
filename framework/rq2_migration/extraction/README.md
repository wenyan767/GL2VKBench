# Vulkan CTS Extraction

This package provides the reproducible Vulkan CTS candidate-extraction procedure for RQ2. It requires the user to obtain and build VK-GL-CTS locally; no CTS source tree, generated dataset, model client, or credentials are distributed here.

## Prerequisites

- Python 3.10 or newer
- A VK-GL-CTS checkout containing `external/vulkancts/modules/vulkan`
- A completed CTS build directory usable for compiler-argument discovery. Build VK-GL-CTS before running extraction; follow the official Vulkan CTS build instructions at [KhronosGroup/VK-GL-CTS `external/vulkancts/README.md`](https://github.com/KhronosGroup/VK-GL-CTS/blob/main/external/vulkancts/README.md).
- LLVM/libclang compatible with the pinned Python packages
- Windows users should use a native shell and ensure the compiler toolchain is available

Install Python dependencies from this directory:

```text
python -m pip install -r requirements.txt
```

Set paths before running. PowerShell example:

```powershell
$env:CTS_SOURCE_ROOT = 'D:\src\VK-GL-CTS'
$env:CTS_BUILD_ROOT = 'D:\src\VK-GL-CTS\build'
$env:CTS_OUTPUT_ROOT = 'D:\vk-cts-extracted' # optional
# $env:LIBCLANG_PATH = 'C:\Program Files\LLVM\bin\libclang.dll' # optional
```

`CTS_SOURCE_ROOT` and `CTS_BUILD_ROOT` are mandatory. The program validates both directories and `external/vulkancts/modules/vulkan` before starting AST work.

The extractor first checks `LIBCLANG_PATH`, then the `clang/native/` directory of the installed Python package, and finally conventional system LLVM locations. If the installed wheel is not detected, obtain its exact DLL path with the same Python interpreter used for extraction and set it explicitly:

```powershell
$env:LIBCLANG_PATH = & python -c "from pathlib import Path; import clang.cindex; print(Path(clang.cindex.__file__).resolve().parent / 'native' / 'libclang.dll')"
```

For a Visual Studio or Windows SDK installed outside the conventional location, run from a Visual Studio Developer PowerShell or set `VSINSTALLDIR` and `WindowsSdkDir` explicitly. The public extractor does not contain author-machine fallback paths.

## Filesystem Safety

The extraction commands read CTS source and build metadata but do not modify the CTS checkout. They do write derived data under `CTS_OUTPUT_ROOT/vulkan_dataset_v2/`:

- `extract.py` creates directories and replaces same-named JSON, case-source, and registration files for every selected module. `--resume` skips modules that already have a class or function record; without it, selected modules are rewritten. Stale files that are no longer emitted are not deleted.
- `base_class.py` without `--write` is a no-write inspection. `base_class.py --write` creates or replaces `base_class.json` and rewrites sibling `child_class.json`/`function_tests.json` files when backfilling base-class IDs.

Use a new user-owned output directory or preserve a snapshot before rerunning. Do not point `CTS_OUTPUT_ROOT` at the CTS checkout, repository root, or a directory containing unrelated files.

## Extraction

Run from this directory or pass an absolute script path:

```text
python extract.py
```

Useful bounded/debug runs:

```text
python extract.py --limit 5
python extract.py --source draw/vktDrawMultipleInterpolationTests.cpp
python extract.py --source draw/vktDrawMultipleInterpolationTests.cpp --source pipeline/vktPipelineImageViewTests.cpp
python extract.py --resume
python extract.py --batch 80
python config.py
```

Use repeatable `--source` arguments for a partial test that must target exact Vulkan files. Each value is relative to `external/vulkancts/modules/vulkan`; absolute paths outside that module root and non-`.cpp` inputs are rejected. `--source` and `--limit` are mutually exclusive so the selected scope is unambiguous.

The extractor is intentionally Vulkan-only. It writes under `CTS_OUTPUT_ROOT/vulkan_dataset_v2/` (or the package-local `output/` when `CTS_OUTPUT_ROOT` is set to that location). Each module may contain:

- `child_class.json`: class-based test cases;
- `function_tests.json`: function-based test cases;
- `case_sources/`: self-contained extracted case source files;
- `<module>_registrations.cpp`: source text plus registration annotations.

After extraction, generate base-class definitions and IDs:

```text
python base_class.py --write
```

This command reads the extracted Vulkan output and writes `base_class.json` beside each module's case records. Without `--write`, it performs a dry run.

## Postconditions and Next Stage

Before semantic-description generation, verify that each candidate has its case code, registration data, and, when applicable, `base_class.json`. Extraction outputs candidate records; it does **not** create `semantic_description`.

Generate descriptions for the benchmark OpenGL rows, their ground-truth Vulkan rows, and the extracted candidates with the shared [`semantic_description_generation.txt`](../../../benchmark/prompts/semantic_description_generation.txt) prompt. Each description is derived from the test-case code plus applicable base-class code and is stored only in user-owned runtime records.

The next stage, [BGE retrieval](../retrieval/README.md), consumes those semantic-enriched records, excludes only the current row's own `vk_recommended.class_id`, selects Top-1, and exports the explicit framework input. Migration generation, CTS compilation, and feedback repair are outside this extraction package.

## Reproducibility and Hygiene

Do not commit extracted output, logs, `__pycache__`, model responses, build products, or API keys. Keep the public repository limited to the rules and scripts needed to reproduce the candidate pool.
