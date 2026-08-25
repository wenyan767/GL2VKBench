# GL2VK Research Artifacts

This repository is the reproducibility artifact accompanying the paper *Benchmarking LLM-Based Test Migration from Legacy to Modern Graphics APIs: A Case Study of OpenGL-to-Vulkan*. It provides the benchmark assets and executable workflows needed to reproduce the paper's methodology and experimental workflow.

The [supplementary materials](supplementary_materials.pdf) provide simplified prompt templates for compatibility analysis and chain-of-thought migration, together with CTS test context construction, registration, and build configuration templates.

## Overview

- **RQ1 — Compatibility analysis:** Determine whether the functionality tested by an OpenGL CTS test case is supported by Vulkan and therefore migratable.
- **RQ2 — Test migration:** Retrieve a relevant Vulkan CTS reference test, generate a migrated Vulkan test case, refine it through CTS compiler feedback, and optionally validate it through CTS execution.

## Repository Structure

```text
benchmark/
framework/
├── rq1_compatibility/
└── rq2_migration/
```

- [`benchmark/`](benchmark/) contains the shared benchmark data, data dictionary, and semantic-description prompt used by both research questions.
- [`framework/rq1_compatibility/`](framework/rq1_compatibility/) contains the RQ1 compatibility-judgment and evaluation workflow.
- [`framework/rq2_migration/`](framework/rq2_migration/) contains the RQ2 retrieval, migration, CTS validation, and evaluation workflow.

## Reproducing the Method

Follow the documentation in this order:

1. [`benchmark/README.md`](benchmark/README.md) — prepare and validate the shared benchmark inputs.
2. [`framework/rq1_compatibility/README.md`](framework/rq1_compatibility/README.md) — reproduce the RQ1 compatibility-analysis workflow.
3. [`framework/rq2_migration/README.md`](framework/rq2_migration/README.md) — reproduce the RQ2 migration workflow.

Each entry-point README is authoritative for its prerequisites, commands, input and output contracts, and validation steps. Runtime-generated model outputs, logs, builds, and CTS working copies are not part of this repository.

## Important Note

Some RQ2 stages modify a user-supplied VK-GL-CTS checkout. Read the [filesystem-safety instructions in the RQ2 README](framework/rq2_migration/README.md#filesystem-safety-read-before-running) before execution.

## License

Original project materials are licensed under the [Apache License 2.0](LICENSE). Materials derived from [KhronosGroup VK-GL-CTS](https://github.com/KhronosGroup/VK-GL-CTS) remain subject to their upstream copyright and license terms; see the included [upstream license](LICENSES/Apache-2.0.txt), [NOTICE](LICENSES/VK-GL-CTS-NOTICE.txt), and [source attribution](LICENSES/VK-GL-CTS-SOURCE-ATTRIBUTION.txt).
