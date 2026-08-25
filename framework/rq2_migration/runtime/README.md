# RQ2 Runtime Validation and API Collection

This directory documents the RQ2 runtime stage. It has two public entry points:

- [`../runtime.py`](../runtime.py) performs package-level runtime validation for candidates that have already passed the compilation stage. It plans, mounts, builds, discovers executable CTS cases, executes them, and restores CTS source when requested or when its `all` transaction completes.
- [`api_sequence_runner.py`](api_sequence_runner.py) executes or parses one CTS case with the LunarG API dump layer and writes an ordered Vulkan API sequence.

Neither entry point calls a model. Both operate on user-owned runtime artifacts and, where applicable, a user-supplied VK-GL-CTS checkout.

## Pipeline Boundary

`pipeline.py --stage runtime` calls only the planning operation from `runtime.py`. It scans eligible compiled candidates and writes `_runtime/state/plan.json` under the selected output root; it does not mount packages, build `deqp-vk`, or execute CTS cases.

Likewise, `pipeline.py --stage all` covers generation, compilation, and repair only. Run `runtime.py` explicitly for runtime validation after those stages complete.

## Package-Level Runtime Validation

### Prerequisites

- A user-owned RQ2 output root containing candidates that passed the pipeline compilation or repair stage.
- A compatible VK-GL-CTS checkout passed explicitly with `--cts-root`; its selected experiment packages must already be created and registered through [`../ctspkg.py`](../ctspkg.py).
- A configured CTS build tree capable of building `deqp-vk`.
- A Vulkan-capable execution environment and the LunarG API dump layer. `runtime.py` reads `VULKAN_SDK` to locate the layer.

Use a dedicated CTS checkout or dedicated experiment packages. The runtime command scans the selected output root; use `--modes`, `--models`, `--rows`, or `--limit` to restrict a local run.

### Stages

| Command | Actual behavior | CTS effect |
| --- | --- | --- |
| `--stage plan` | Scans compiled candidates and writes `_runtime/state/plan.json`. | Does not modify CTS source. |
| `--stage mount` | Verifies a clean baseline build, then mounts planned candidate files into the selected registered packages and records a ledger. | Modifies CTS source; run `--restore` afterward. |
| `--stage build` | Builds `deqp-vk`. When a batch link failure is attributable to specific mounted candidates, removes those candidates and retries up to `--max-build-rounds`. | CTS mounts remain in place. |
| `--stage execute` | Resolves each mounted leaf from the real `deqp-vk` case list, then executes it and records CTS/QPA/API-dump evidence. | Does not add mounts, but writes runtime artifacts. |
| `--stage all` | Runs plan, baseline verification, mount, build, case discovery, and execution as one transaction. | Restores ledger-managed CTS source in `finally`, then verifies that no managed mount residue remains. |

`--restore` restores ledger-managed CTS source changes and removes ledger-listed created files. `--verify-clean` checks the selected CTS checkout for managed runtime mount residue. Both are explicit recovery/verification commands rather than stages.

### Run A Package-Level Workflow

Use one explicit `--output-root` for every command in the same runtime run:

```text
python framework/rq2_migration/runtime.py \
  --stage plan \
  --cts-root path/to/VK-GL-CTS \
  --output-root path/to/local-rq2-run \
  --models claude \
  --rows 44,59

python framework/rq2_migration/runtime.py \
  --stage mount \
  --cts-root path/to/VK-GL-CTS \
  --output-root path/to/local-rq2-run \
  --models claude \
  --rows 44,59

python framework/rq2_migration/runtime.py \
  --stage build \
  --cts-root path/to/VK-GL-CTS \
  --output-root path/to/local-rq2-run \
  --models claude \
  --rows 44,59

python framework/rq2_migration/runtime.py \
  --stage execute \
  --cts-root path/to/VK-GL-CTS \
  --output-root path/to/local-rq2-run \
  --models claude \
  --rows 44,59
```

For a single local transaction, use `--stage all` with the same selection arguments. It restores mounted CTS source in `finally`; build products, logs, QPA files, API dumps, and result records remain under the output root.

```text
python framework/rq2_migration/runtime.py \
  --stage all \
  --cts-root path/to/VK-GL-CTS \
  --output-root path/to/local-rq2-run \
  --models claude \
  --rows 44,59
```

### Recovery and Filesystem Safety

`--stage mount` changes CTS source and a standalone `mount`/`build`/`execute` sequence does not restore it automatically. Keep the runtime ledger under the same output root and recover with:

```text
python framework/rq2_migration/runtime.py \
  --output-root path/to/local-rq2-run \
  --restore

python framework/rq2_migration/runtime.py \
  --cts-root path/to/VK-GL-CTS \
  --output-root path/to/local-rq2-run \
  --verify-clean
```

Do not manually remove mounted files when the ledger or backups are missing. Preserve the state and inspect the reported error. Recovery restores only ledger-managed CTS source; it does not remove the CTS build, logs, QPA files, API dumps, or other runtime artifacts.

## Runtime Evidence

Package-level execution resolves exact leaf paths from the built `deqp-vk` case list rather than synthesizing them from `row_id`. It records the command result, stdout/stderr, QPA path, exact case path, API dump path, and a per-case `run_summary.json`.

A usable CTS result requires a complete QPA test-case result with `StatusCode`. Normal CTS verdicts are recorded distinctly. A missing or truncated QPA, filter miss, crash, timeout, or result without `StatusCode` is incomplete execution evidence and must not be treated as a successful run. API collection supplements the CTS verdict; it does not replace it.

For the next metrics stage, take compilation status from the generation/repair outcome and take migrated CTS verdict evidence from this runtime output. Reference CTS verdict evidence must be collected for the selected retrieval reference separately. API-sequence collection below supplies the corresponding reference and migrated sequence fields.

## API Sequence Collection

`api_sequence_runner.py` is the focused entry point for collecting one API sequence. It can either execute a supplied CTS case with `VK_LAYER_LUNARG_api_dump` enabled or parse an existing `vk_apidump.txt` with `--dump-only`.

### Prerequisites

- A local Vulkan CTS build containing `deqp-vk` (or `deqp-vk.exe`) for execution mode.
- The LunarG API dump layer available through `VULKAN_SDK` or `--layer-path`.
- An exact Vulkan CTS case name for execution mode, or an existing API dump for `--dump-only`.
- A dedicated writable per-case `--output-dir`.

### Execute and Parse

```powershell
python framework/rq2_migration/runtime/api_sequence_runner.py `
  --executable C:/path/to/deqp-vk.exe `
  --case dEQP-VK.example.case `
  --output-dir C:/path/to/rq2-artifacts/reference-case `
  --layer-path C:/VulkanSDK/1.4.x/Bin `
  --row-id 1 `
  --class-id vktExampleCase `
  --test-path external/vulkancts/modules/vulkan/example/vktExample.cpp `
  --runtime-name vk_gl_cts
```

`--layer-path` is optional when `VULKAN_SDK` is set. Pass `--extra-arg` repeatedly for additional `deqp-vk` arguments. The command replaces its fixed output filenames in `--output-dir`; preserve prior output or use a new directory for each case.

```powershell
python framework/rq2_migration/runtime/api_sequence_runner.py `
  --dump-only `
  --api-dump C:/path/to/vk_apidump.txt `
  --output-dir C:/path/to/rq2-artifacts/reference-case `
  --row-id 1 `
  --class-id vktExampleCase `
  --execution-status passed
```

## API Output Contract

`api_sequence.json` retains supplied traceability metadata and records the ordered sequence, length, hash, and per-call counts:

```json
{
  "api_sequence": ["vkCreateInstance", "vkEnumeratePhysicalDevices"],
  "api_sequence_length": 2,
  "api_sequence_hash": "...",
  "api_call_counts": {
    "vkCreateInstance": 1,
    "vkEnumeratePhysicalDevices": 1
  }
}
```

For each benchmark row, collect one sequence from the reference Vulkan CTS case and one from the migrated case. Copy them into `reference_api_sequence` and `migrated_api_sequence` in the local input consumed by [`../metrics/migration_metrics.py`](../metrics/migration_metrics.py). This supports `APISim`; the other RQ2 metrics use the associated code and CTS outcomes.
