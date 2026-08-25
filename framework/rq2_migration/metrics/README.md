# RQ2 Metrics

## Prerequisites

- Python 3.10 or newer.
- A user-owned JSON list containing one complete runtime record per case; see the required fields and minimal input shape below.
- Reference and migrated code strings, compile/execution evidence, normalized or raw CTS verdicts, and API sequences must come from the same experiment provenance. Do not combine runtime evidence from a previously built package with clean-build failure evidence from a separate build for the same case.
- The optional codebleu package is required unless the command is run with --skip-codebleu.
- Metrics calculation itself does not require CTS, a GPU, Vulkan SDK, a model runner, or network access.

## Runtime Evidence Rules

CTS verdicts Pass, Fail, and NotSupported are explicitly normalized to Passed, Failed, and Unsupported. execution_success=true means CTS produced one of those three normal verdicts; it does not mean that the test verdict was Pass. Compile or link failures, filter misses including 0/0 runs, missing or truncated QPA files, crashes, timeouts, internal errors, and warnings without a defined mapping must use execution_success=false and an empty migrated_result. Contradictory records and duplicate row_id values are rejected.

All supplied RQ2 cases form the denominator. B4 and CB4 remain static code metrics over every supplied artifact, including code that later fails to compile. Missing or failed compilation/execution contributes zero to CSR, ESR, and APISim. BLEU-4 uses add-one modified-precision smoothing and macro-averages per-case scores, matching the existing local static-metrics implementation. CodeBLEU uses C++ with equal weights for its four components.

These model-independent calculators follow the paper definitions and consume user-owned runtime artifacts. They do not call a model or ship experiment outputs. Scores are fractions in `[0, 1]`; multiply by 100 for percentages.

## Metrics Input Assembly

Assemble these fields into one metrics input record per benchmark case before invoking the metrics script. The calculator validates the assembled record; it does not collect data from retrieval, generation, runtime, or API artifacts automatically.

| Metrics field | Source |
| --- | --- |
| `reference_code` | The selected Vulkan reference test in the retrieval output. |
| `migrated_code` | The generated candidate after the selected compilation/feedback-repair path. |
| `compilation_success` | The compilation outcome for that migrated candidate. |
| `execution_success` | The migrated candidate's package-level CTS runtime evidence. |
| `reference_result` | CTS runtime evidence collected for the selected Vulkan reference case. |
| `migrated_result` | CTS runtime evidence collected for the migrated candidate. |
| `reference_api_sequence` | `api_sequence.json` collected for the selected Vulkan reference case. |
| `migrated_api_sequence` | `api_sequence.json` collected for the migrated candidate. |

For migration, provide a local JSON list with `row_id`, `reference_code`, `migrated_code`, `compilation_success`, `execution_success`, `reference_result`, `migrated_result`, `reference_api_sequence`, and `migrated_api_sequence`, then run `migration_metrics.py`. The calculator validates these fields and fails on missing or wrongly typed values; it does not silently replace missing data with zeros. `compilation_success` and `execution_success` must be JSON booleans. Result values are `Passed`, `Failed`, or `Unsupported` (an empty migrated result is allowed only for a failed execution).

The API arrays can be produced by [`../runtime/api_sequence_runner.py`](../runtime/api_sequence_runner.py), which runs a CTS case under `VK_LAYER_LUNARG_api_dump` and parses the dump in the same invocation. It also accepts an existing dump with `--dump-only`. The calculator reports the six RQ2 migration metrics: `B4`, `CB4`, `CSR`, `ESR`, `TRC` (macro-F1 over `Passed`, `Failed`, `Unsupported`), and `APISim` (BLEU-4 over instrumented API sequences, zero on compile/execute failure). `CB4` requires the optional `codebleu` package. RQ1 negative cases are not included in this migration denominator.

## Run

Run from the repository root with a user-owned runtime record file and output path:

```text
python framework/rq2_migration/metrics/migration_metrics.py \
  --input path/to/rq2_runtime_records.json \
  --output path/to/rq2_metrics.json
```

Use `--skip-codebleu` when the optional CodeBLEU dependency is not installed. The command creates the parent directory of `--output` when needed and replaces an existing file at that exact path. It does not modify the input records, CTS tree, model configuration, or API dumps.

Minimal input shape:

```json
[{
  "row_id": 1,
  "reference_code": "...",
  "migrated_code": "...",
  "compilation_success": true,
  "execution_success": true,
  "reference_result": "Passed",
  "migrated_result": "Passed",
  "reference_api_sequence": ["vkCreateInstance"],
  "migrated_api_sequence": ["vkCreateInstance"]
}]
```
