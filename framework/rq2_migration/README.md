# RQ2 Migration

This directory is the main navigation and reproducibility entry point for RQ2. The workflow starts with the benchmark and an extracted Vulkan CTS candidate pool, retrieves a relevant Vulkan CTS reference test, generates a migrated Vulkan test, refines it through compiler feedback, and optionally validates it through CTS execution before calculating metrics.

Run the commands below from the repository root.

## Overview

RQ2 consumes the retrieval output explicitly through `--input-json`. The retrieval result supplies the OpenGL source context and the selected Vulkan reference test for each benchmark row. Generated code, model responses, logs, builds, and CTS working copies are user-owned runtime artifacts and are not part of the published artifact.

## Workflow

```text
Benchmark
    -> Vulkan CTS extraction
    -> semantic-description generation
    -> semantic retrieval
    -> migration generation
    -> compilation / feedback repair
    -> optional runtime validation
    -> API sequence collection
    -> metrics
```

The shared semantic-description prompt is [`../../benchmark/prompts/semantic_description_generation.txt`](../../benchmark/prompts/semantic_description_generation.txt). Extraction, retrieval, generation, runtime/API collection, and metric calculation each have their own authoritative README under this directory.

## Stage Handoffs

The following handoffs make the complete RQ2 path explicit. All derived records are user-owned runtime artifacts; the released benchmark remains unchanged.

| Stage | Input | Output and next handoff |
| --- | --- | --- |
| Benchmark | Released benchmark records, including test-case and applicable base-class code. | Runtime OpenGL records to be given semantic descriptions. |
| Vulkan CTS extraction | A local VK-GL-CTS checkout and build metadata. | Vulkan candidate records, case code, registration data, and applicable base-class records. |
| Semantic-description generation | Each OpenGL or Vulkan test-case code block plus its applicable base-class code, rendered with the shared prompt. | Runtime benchmark and candidate records with non-empty `semantic_description`. |
| Semantic retrieval | Semantic-enriched benchmark rows and eligible Vulkan candidates. | Retrieval-result JSON, passed explicitly to the next stage as `--input-json`. |
| Migration generation | A retrieval-result row and the selected initial seed. | CoT or non-CoT migrated candidate for compilation and configuration-specific repair. |
| Compilation / feedback repair | Generated candidate, selected CTS package, and compiler feedback. | Compilation and repair artifacts; eligible candidates may proceed to optional runtime validation. |
| Runtime validation | Compiled candidates, a CTS checkout, and a runtime plan. | CTS verdict/evidence for each migrated case. |
| API sequence collection | One reference CTS case and its corresponding migrated case. | `reference_api_sequence` and `migrated_api_sequence` for the metrics record. |
| Metrics | One assembled record per benchmark case containing code, compilation, runtime, and API evidence. | Aggregate RQ2 metrics. |

Extraction does not create `semantic_description`. Generate it after extraction from test-case code plus applicable base-class code, then run retrieval only on the semantic-enriched runtime records. The [extraction README](extraction/README.md) and [retrieval README](retrieval/README.md) define the respective detailed contracts.

Runtime validation is optional only when a run ends after generation and repair. Calculating the complete RQ2 metrics requires the migrated runtime evidence, reference CTS evidence, and both API sequences described in the [runtime README](runtime/README.md) and assembled in the [metrics README](metrics/README.md).

## External Requirements

The complete workflow uses user-managed resources:

- Python 3.10 or newer and the dependencies required by the selected stage;
- runtime semantic descriptions for the retrieval query records and eligible Vulkan candidates;
- a local BGE model for retrieval;
- an external model runner for generation and repair;
- a VK-GL-CTS checkout and compatible build toolchain for CTS-facing stages; and
- a dedicated writable output root outside the CTS checkout and published source data.

Runtime execution and API collection additionally require a built `deqp-vk`, a compatible Vulkan environment, and the LunarG API dump layer. See the relevant submodule README for stage-specific prerequisites.

## RQ2 Configuration Mapping

The four configurations reported in the RQ2 experiment map to the repository workflow as follows:

| Paper configuration | Initial-generation seed | Subsequent processing | Repository execution path |
| --- | --- | --- | --- |
| **Multi-agent** | `non_cot` | Director-selected Explainer, Verifier, and Checker analysis, followed by Generator repair | `multi_agent` method, `director` repair strategy |
| **Single-agent** | `cot` | Feedback repair with structural re-evaluation and compile-only repair | `single_agent` method, `feedback` repair strategy |
| **w/o CoT** | `non_cot` | Feedback repair with structural re-evaluation and compile-only repair | `wo_cot` method, `feedback` repair strategy |
| **w/o Feedback** | `cot` | No compiler-feedback repair; use the initial CoT candidate from seed `cot` | `cot` seed at `attempt_1`, also used as the `single_agent` input |

The **w/o Feedback** configuration is the CoT initial candidate before feedback repair. It therefore reuses the `cot` seed that supplies `attempt_1` for **Single-agent**, so the comparison changes the feedback component while keeping the initial CoT generation fixed.

```text
Initial generation
|- CoT seed
|  |- w/o Feedback -> no feedback repair
|  `- Single-agent -> single-agent feedback repair
|
`- non-CoT seed
   |- w/o CoT -> single-agent feedback repair
   `- Multi-agent -> Director-coordinated multi-agent repair
```

## Main Entry Point

[`pipeline.py`](pipeline.py) is the main RQ2 orchestration entry point. It accepts `generate`, `compile`, `repair`, `runtime`, and `all` stages. Use `--rows` or `--limit` for a bounded run and pass the same user-owned `--output-root` throughout one experiment.

```text
python framework/rq2_migration/pipeline.py \
  --stage all \
  --model claude \
  --model-id your-exact-model-id \
  --model-runner path/to/model_runner.py \
  --input-json path/to/rq2_retrieval_results.json \
  --cts-root path/to/VK-GL-CTS \
  --output-root path/to/local-rq2-run \
  --rows 44,59
```

Before CTS-facing compilation, repair, or runtime work, inspect and, when needed, create the selected dedicated CTS packages with [`ctspkg.py`](ctspkg.py):

```text
python framework/rq2_migration/ctspkg.py \
  --cts-root path/to/VK-GL-CTS \
  --models claude \
  --seeds \
  --apply
```

Use `pipeline.py --stage generate` for generation only, `--stage compile` for compilation, and `--stage repair` for compiler-feedback repair. `pipeline.py --stage runtime` prepares the runtime plan only; it does not mount packages, build `deqp-vk`, or execute CTS cases. `--stage all` covers generation, compilation, and repair only. Invoke [`runtime.py`](runtime.py) explicitly for runtime execution. The stage-specific contracts and options are defined by the corresponding entry point and submodule README.

## Model Runner Contract

The repository is provider-neutral. [`model_adapter.py`](model_adapter.py) sends one JSON request to a user-provided runner over stdin:

```json
{"prompt":"...","model_key":"claude","model_id":"user-recorded-id","description":"...","meta":{}}
```

The runner must return one JSON object on stdout:

```json
{"response":"raw model response","usage":{}}
```

Select the runner and exact model identity with `--model-runner` and `--model-id`, or with `GL2VK_MODEL_RUNNER` and `GL2VK_MODEL_ID`. `usage` is optional. Credentials, endpoints, and provider SDKs remain outside this repository. Model responses are runtime artifacts and are not part of the published artifact.

## Generation and Repair

Extracted records may identify a class-form test as `child` or `class`; the framework normalizes `child` to internal `class`. A function-form record uses `function`.

The generation prompt strictly requires the implementation form of the retrieved Vulkan reference test case. If a generated candidate has a different form, the framework records the detected form and continues with that form rather than rejecting the candidate. Accordingly, `template_type_matches_reference` is record-only: `false` does not by itself invalidate a candidate.

For Multi-agent repair, the Director may selectively invoke the Explainer, Verifier, and Checker to analyze source semantics, semantic drift, and compilation failures. Their outputs and the Director decision are passed to the Generator, which always produces the next structured repair payload. Single-agent and w/o CoT feedback repair instead use structural re-evaluation followed by compile-only repair; this is not part of the Multi-agent flow.

## Running RQ2

Reproduce the methodology in this order:

1. Start from the shared [benchmark](../../benchmark/README.md), extract the Vulkan CTS candidate pool, and generate semantic descriptions for both sides with the shared prompt. Follow [extraction](extraction/README.md).
2. Run BGE retrieval on those semantic-enriched runtime records. Its result is the explicit RQ2 input JSON. Follow [retrieval](retrieval/README.md).
3. Use `pipeline.py --stage all` with that `--input-json` for initial generation, compilation, and the selected repair configuration. Create or inspect CTS packages with `ctspkg.py` before CTS-facing stages.
4. For optional runtime validation, use `pipeline.py --stage runtime` only to prepare the plan, then use [`runtime.py`](runtime.py) explicitly for mount, build, execution, and restoration. Use [`runtime/api_sequence_runner.py`](runtime/api_sequence_runner.py) to collect reference and migrated API sequences. Follow [runtime](runtime/README.md).
5. Assemble one local metrics input record per benchmark case from the retrieval, generation/repair, runtime, and API outputs, then calculate metrics. Follow [metrics](metrics/README.md).

[`generation/prepare_generation_prompt.py`](generation/prepare_generation_prompt.py) is an optional prompt-rendering utility for inspecting a retrieval record. It does not replace `generate.py` or `pipeline.py` for the experimental workflow.

## Filesystem Safety: Read Before Running

Some RQ2 operations modify a user-supplied VK-GL-CTS checkout. In particular, package creation and reversion, compilation/repair in dedicated experiment packages, and runtime mount/build/execute operations can write CTS source, package state, or build artifacts. Use an isolated checkout or dedicated packages and a separate user-owned output root.

Before the first CTS write, read the relevant instructions in [this workflow](#main-entry-point), [runtime](runtime/README.md), and the `ctspkg.py`/`runtime.py` command help. The required recovery entry points are:

```text
python framework/rq2_migration/runtime.py --output-root path/to/local-rq2-run --restore
python framework/rq2_migration/runtime.py --cts-root path/to/VK-GL-CTS --output-root path/to/local-rq2-run --verify-clean
python framework/rq2_migration/ctspkg.py --cts-root path/to/VK-GL-CTS --models claude --seeds --revert
```

Use the same output root and package selection for recovery as for the corresponding write operation. The detailed write, replacement, deletion, mount, build, execution, and evidence rules belong to the stage documentation and command help.

## Submodules

| Submodule | Authoritative documentation |
| --- | --- |
| Candidate extraction and base-class recovery | [extraction/README.md](extraction/README.md) |
| BGE retrieval and Top-1 selection | [retrieval/README.md](retrieval/README.md) |
| Prompt rendering utility | [generation/README.md](generation/README.md) |
| Runtime validation and API collection | [runtime/README.md](runtime/README.md) |
| B4, CB4, CSR, ESR, TRC, and APISim | [metrics/README.md](metrics/README.md) |

These submodule README files define their own prerequisites, command options, input/output contracts, runtime evidence, metrics, and filesystem effects. This README only defines how they fit into the RQ2 reproduction workflow.
