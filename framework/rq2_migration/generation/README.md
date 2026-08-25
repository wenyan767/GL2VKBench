# Prompt Preparation

This directory contains a prompt-rendering utility for inspecting one retrieval record or preparing a migration prompt for a user-managed model runner. It does not call a model, compile CTS code, or execute tests.

The initial non-CoT and woFeedback generation paths use [`../generate.py`](../generate.py) or [`../pipeline.py`](../pipeline.py). Those entry points pass the resulting candidate into CTS compilation and compiler-feedback repair.

## Usage

The input must be retrieval output produced by [`../retrieval/build_gl_sim_vk.py`](../retrieval/build_gl_sim_vk.py):

```text
python framework/rq2_migration/generation/prepare_generation_prompt.py \
  --input path/to/rq2_retrieval_results.json \
  --output-dir path/to/runtime/migration_prompts
```

Use `--row-id 44` to render one record. The rendered prompt and `manifest.json` are user-owned runtime artifacts and should not be committed.
