# BGE Retrieval and Top-1 Selection

This module implements the RQ2 retrieval step. BGE embedding and Top-1 selection are performed together because they form one reproducibility step. This module does not call a generative model and does not create `semantic_description`; users generate that field beforehand from test-case code plus applicable base-class code with the shared benchmark prompt.

## Prerequisites

- Python 3.10 or newer.
- The dependencies in requirements.txt installed in the interpreter used for retrieval.
- A runtime copy of benchmark_data.json in which every selected OpenGL row has a non-empty semantic_description.
- An extraction output root containing eligible Vulkan candidate records, with non-empty candidate semantic_description values and the required source/code fields.
- A local copy or cache of BAAI/bge-base-zh-v1.5 (or another explicitly selected local model path). The script uses local_files_only=True and does not download model files.
- Enough RAM/VRAM for the selected BGE model and the requested retrieval scope.

## Inputs

The script requires three user-controlled paths:

1. --benchmark: a runtime copy of benchmark_data.json with non-empty semantic_description values;
2. --vulkan-root: the output root from extraction/extract.py, after the candidate records also have semantic_description values;
3. --output: a new JSON file for BGE retrieval results.

The published benchmark_data.json is not modified. Runtime semantic descriptions are derived inputs and should remain outside the public base data.

## Model and Scoring Contract

- Default model: BAAI/bge-base-zh-v1.5.
- The model must be available locally because loading uses local_files_only=True.
- CUDA is used when available, otherwise CPU.
- Tokenizer truncation uses max_length=512.
- Mean pooling uses the attention mask.
- Embeddings are L2-normalized.
- Similarity is the dot product of normalized embeddings, equivalent to cosine similarity.
- Candidate and query text contain semantic_description.
- base_class.json is excluded from the candidate pool.
- Records without a non-empty `semantic_description` are excluded.

For every benchmark row, candidates are ranked by decreasing similarity. Only the current row's vk_recommended.class_id is excluded. The next highest-scoring candidate becomes similar_vulkan_case. No global exclusion list is applied.

## Filesystem Safety

The retrieval script reads `--benchmark`, `--vulkan-root`, and the local BGE model without modifying them. It creates the parent directory of `--output` when needed and unconditionally replaces an existing file at that exact path. Use a new derived-output path or copy an existing result before rerunning; never set `--output` to the published benchmark or an extracted candidate JSON.

## Run

Install dependencies:

    python -m pip install -r framework/rq2_migration/retrieval/requirements.txt

Run from the repository root:

    python framework/rq2_migration/retrieval/build_gl_sim_vk.py --benchmark path/to/runtime_benchmark_with_semantics.json --vulkan-root path/to/vulkan_dataset_v2 --output path/to/rq2_retrieval_results.json --model-name path/to/local/BAAI-bge-base-zh-v1.5

No network download is attempted by this script. The model must already be present in the local cache or supplied as a local directory.

## Output

Each output row contains row_id, ground_truth_vk_class_id, the OpenGL query context under gl_subclass, gl_base_classes, and similar_vulkan_case including the selected candidate class_id, source JSON, semantic description, code, and similarity_score.

The output is a derived RQ2 retrieval artifact. Pass it unchanged as `--input-json` to [`../pipeline.py`](../pipeline.py) or [`../generate.py`](../generate.py) for migration generation; it is not the public benchmark file and must remain a user-owned runtime result.

## Preconditions and Postconditions

Before running, verify that the benchmark query rows and eligible Vulkan candidates have non-empty semantic_description. After running, verify that every row has either a selected candidate or explicit null when no eligible candidate remains, and that the selected candidate's class_id is not the row's ground_truth_vk_class_id.
