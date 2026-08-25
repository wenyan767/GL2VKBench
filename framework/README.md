# Framework

`framework/` contains the executable workflows for the two research questions. Shared benchmark data and the semantic-description prompt remain in [`../benchmark/`](../benchmark/) and are not duplicated here.

## RQ1 Compatibility

[`rq1_compatibility/`](rq1_compatibility/) implements this chain:

```text
benchmark_data.json + rq1_compatibility/inputs/neg_406.json
        ↓
shared semantic_description generation
        ↓
compatibility judgment prompt
        ↓
response parsing
        ↓
Accuracy / Precision / Recall / F1 / confusion matrix
```

The positive and negative sources remain separate. Missing semantic descriptions and invalid model outputs are explicit failures/statuses.

## RQ2 Migration

[`rq2_migration/`](rq2_migration/) contains the complete experiment chain:

1. extract Vulkan CTS candidates and required class/registration context;
2. generate semantic descriptions in user-managed working data;
3. retrieve with BGE, exclude each row's Ground Truth, and select Top-1;
4. validate the retrieval result supplied through explicit `--input-json`;
5. generate migration seeds for the woFeedback condition and the non-CoT baseline through a user-provided external model runner;
6. create CTS module skeletons, inject candidates, and compile them in the user's CTS tree;
7. execute the woCot, singleAgent, and multiAgent compiler-feedback repair lines;
8. optionally run CTS/API-dump validation and calculate B4, CB4, CSR, ESR, TRC, and APISim.

The repository provides a vendor-neutral process adapter, not provider clients. Model identities are recorded with `--model-id`.

RQ2 never reads a fixed `framework/data/` file. The only sample input is an explicitly marked test fixture. Read [`rq2_migration/README.md`](rq2_migration/README.md) before running any command that modifies a CTS tree.

## External Requirements

Each workflow documents its own external conditions. Use the subdirectory README as the authoritative checklist rather than treating this page as a combined installation guide:

- [RQ1 prerequisites](rq1_compatibility/README.md#preconditions): Python standard library, user-generated semantic descriptions, and a user-managed model-call mechanism.
- [RQ2 overall conditions](rq2_migration/README.md#external-conditions): explicit retrieval input, external model runner, CTS checkout, isolated output root, and optional Vulkan runtime environment.
- [RQ2 extraction](rq2_migration/extraction/README.md#prerequisites): CTS source/build roots, LLVM/libclang, and extraction dependencies.
- [RQ2 retrieval](rq2_migration/retrieval/README.md#prerequisites): semantic-enriched benchmark/candidate data, local BGE model, and retrieval dependencies.
- [RQ2 runtime](rq2_migration/runtime/README.md#prerequisites): built deqp-vk, Vulkan SDK/API dump layer, and exact case names.
- [RQ2 metrics](rq2_migration/metrics/README.md#prerequisites): runtime-record JSON and optional CodeBLEU dependency.

Generated outputs and local configurations are user-owned and must not be uploaded.
