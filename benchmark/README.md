# Shared Benchmark Assets

This directory contains the benchmark data and shared semantic-description prompt used by the RQ1 compatibility analysis and RQ2 test-migration workflows.

## Contents

- [`benchmark_data.json`](benchmark_data.json) contains the 406 benchmark records used by the study, including OpenGL CTS test information, inheritance context, and matched Vulkan CTS reference information required by the experimental workflows.
- [`data_dictionary.md`](data_dictionary.md) defines the benchmark fields and their validation rules.
- [`prompts/semantic_description_generation.txt`](prompts/semantic_description_generation.txt) is the shared prompt used to generate semantic descriptions at runtime.

## Semantic Descriptions

The prompt uses only test-case code and base-class code through `{{TEST_CASE_CODE}}` and `{{BASE_CLASS_CODE}}`. Registration metadata remains part of the benchmark's source and provenance information but is not an input to semantic-description generation.

A semantic description expresses the tested functionality or object, the relevant condition and execution stage, the specific behavior under test, and the expected result relationship at an API-independent level. It describes test semantics rather than source-code structure, API calls, or a compatibility conclusion.

## Use in RQ1

RQ1 uses the OpenGL benchmark records together with the negative cases defined by the RQ1 workflow to evaluate compatibility classification. See the [`framework/rq1_compatibility/README.md`](../framework/rq1_compatibility/README.md) for the complete procedure.

## Use in RQ2

RQ2 uses the benchmark records as OpenGL source tests and matched Vulkan references. Semantic descriptions are generated for the relevant benchmark test data and the extracted Vulkan candidate pool before semantic retrieval and test migration. See the [`framework/rq2_migration/README.md`](../framework/rq2_migration/README.md) for the complete procedure.
