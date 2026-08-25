import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, AutoTokenizer


MODEL_NAME = "BAAI/bge-base-zh-v1.5"


@dataclass
class VulkanCandidate:
    benchmark_id: Optional[str]
    class_id: Optional[str]
    qualified_name: Optional[str]
    class_type: Optional[str]
    file_path: Optional[str]
    code_file: Optional[str]
    semantic_description: Any
    code: str
    source_json: str
    text: str


class BGEEmbedder:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = model_name if Path(model_name).exists() else snapshot_download(
            repo_id=model_name, local_files_only=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = self._load_local_model(model_path).to(self.device)
        self.model.eval()

    @staticmethod
    def _load_local_model(model_path: str) -> AutoModel:
        try:
            return AutoModel.from_pretrained(
                model_path,
                local_files_only=True,
                use_safetensors=True,
            )
        except Exception:
            config = AutoConfig.from_pretrained(model_path, local_files_only=True)
            model = AutoModel.from_config(config)
            checkpoint_path = Path(model_path) / "pytorch_model.bin"
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            return model

    @staticmethod
    def _mean_pool(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)
                pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                normalized = F.normalize(pooled, p=2, dim=1)
            embeddings.append(normalized.cpu().numpy())
        return np.vstack(embeddings)


def load_json(path: Path) -> Any:
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    return json.loads(path.read_text(encoding="latin-1"))


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(safe_text(item) for item in value if safe_text(item))
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            item_text = safe_text(item)
            if item_text:
                parts.append(f"{key}: {item_text}")
        return "\n".join(parts)
    return str(value).strip()


def build_match_text(semantic_description: Any) -> str:
    semantic_text = safe_text(semantic_description)
    return f"semantic_description:\n{semantic_text}" if semantic_text else ""


def infer_class_type(record: dict[str, Any], source_json_name: str) -> Optional[str]:
    class_type = record.get("class_type")
    if class_type:
        return class_type
    if source_json_name == "function_tests.json":
        return "function"
    return None


def load_code(module_dir: Path, code_file: Optional[str]) -> str:
    if not code_file:
        return ""
    code_path = module_dir / code_file
    if not code_path.exists():
        return ""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return code_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return ""


def record_code(module_dir: Path, record: dict[str, Any]) -> str:
    """Read code from the extractor's inline schema or legacy code_file schema."""
    inline = safe_text(record.get("code"))
    return inline or load_code(module_dir, record.get("code_file"))


def should_skip_candidate(record: dict[str, Any], source_json_name: str) -> bool:
    if source_json_name == "base_class.json":
        return True
    semantic_text = safe_text(record.get("semantic_description"))
    return not semantic_text


def load_vulkan_candidates(vulkan_dataset_root: Path) -> list[VulkanCandidate]:
    candidates: list[VulkanCandidate] = []
    for json_path in vulkan_dataset_root.rglob("*.json"):
        if json_path.name == "base_class.json":
            continue
        payload = load_json(json_path)
        if not isinstance(payload, list):
            continue
        module_dir = json_path.parent
        for record in payload:
            if not isinstance(record, dict):
                continue
            if should_skip_candidate(record, json_path.name):
                continue
            code = record_code(module_dir, record)
            if not safe_text(record.get("class_id")) or not code.strip():
                continue
            text = build_match_text(
                record.get("semantic_description"),
            )
            if not text:
                continue
            candidates.append(
                VulkanCandidate(
                    benchmark_id=record.get("benchmark_id"),
                    class_id=record.get("class_id"),
                    qualified_name=record.get("qualified_name"),
                    class_type=infer_class_type(record, json_path.name),
                    file_path=record.get("file_path"),
                    code_file=record.get("code_file"),
                    semantic_description=record.get("semantic_description"),
                    code=code,
                    source_json=str(json_path.relative_to(vulkan_dataset_root)).replace("\\", "/"),
                    text=text,
                )
            )
    return candidates


def select_best_non_gt_candidate(
    scores: np.ndarray,
    candidates: list[VulkanCandidate],
    gt_vk_class_id: Optional[str],
) -> tuple[Optional[VulkanCandidate], Optional[float], Optional[int]]:
    ranked_indices = np.argsort(-scores)
    for source_rank, index in enumerate(ranked_indices, start=1):
        candidate = candidates[int(index)]
        if gt_vk_class_id and candidate.class_id == gt_vk_class_id:
            continue
        return candidate, float(scores[int(index)]), source_rank
    return None, None, None


def build_output_records(
    gl_cases: list[dict[str, Any]],
    candidates: list[VulkanCandidate],
    candidate_embeddings: np.ndarray,
    embedder: BGEEmbedder,
    query_batch_size: int,
    model_name: str,
) -> list[dict[str, Any]]:
    gl_texts = [
        build_match_text(
            item.get("gl_subclass", {}).get("semantic_description"),
        )
        for item in gl_cases
    ]
    missing_queries = [
        str(gl_cases[index].get("row_id", index))
        for index, item in enumerate(gl_cases)
        if not safe_text(item.get("gl_subclass", {}).get("semantic_description"))
    ]
    if missing_queries:
        raise ValueError(
            "Missing semantic_description for benchmark rows: "
            + ", ".join(missing_queries)
        )
    gl_embeddings = embedder.encode(gl_texts, batch_size=query_batch_size)

    output: list[dict[str, Any]] = []
    for index, case in enumerate(gl_cases):
        gl_subclass = case.get("gl_subclass", {})
        gt_vk_class_id = case.get("vk_recommended", {}).get("class_id")
        scores = np.matmul(candidate_embeddings, gl_embeddings[index])
        best_candidate, similarity, source_rank = select_best_non_gt_candidate(
            scores=scores,
            candidates=candidates,
            gt_vk_class_id=gt_vk_class_id,
        )
        record = {
                "row_id": case.get("row_id"),
                "ground_truth_vk_class_id": gt_vk_class_id,
                "gl_subclass": {
                    "class_id": gl_subclass.get("class_id"),
                    "qualified_name": gl_subclass.get("qualified_name"),
                    "file_path": gl_subclass.get("file_path"),
                    "code": gl_subclass.get("code"),
                    "semantic_description": gl_subclass.get("semantic_description"),
                },
                "gl_base_classes": case.get("gl_base_classes", []),
                "similar_vulkan_case": None
                if best_candidate is None
                else {
                    "class_id": best_candidate.class_id,
                    "qualified_name": best_candidate.qualified_name,
                    "class_type": best_candidate.class_type,
                    "file_path": best_candidate.file_path,
                    "code_file": best_candidate.code_file,
                    "semantic_description": best_candidate.semantic_description,
                    "code": best_candidate.code,
                    "source_json": best_candidate.source_json,
                    "similarity_score": similarity,
                    "rank": 1,
                    "source_rank_before_ground_truth_exclusion": source_rank,
                },
                "retrieval": {
                    "model": Path(model_name).name if Path(model_name).exists() else model_name,
                    "pooling": "attention_mask_mean",
                    "l2_normalized": True,
                    "max_length": 512,
                    "similarity": "cosine",
                    "selection": "exclude_current_ground_truth_then_top1",
                },
            }
        output.append(record)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build similar Vulkan test cases for GL validation samples."
    )
    parser.add_argument("--benchmark", required=True, type=Path,
                        help="Runtime benchmark JSON containing semantic_description fields.")
    parser.add_argument("--vulkan-root", required=True, type=Path,
                        help="Root of extracted Vulkan candidate output.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output JSON path for BGE + Top-1 results.")
    parser.add_argument("--model-name", default=MODEL_NAME,
                        help="BGE model id or local model directory.")
    parser.add_argument("--candidate-batch", type=int, default=32)
    parser.add_argument("--query-batch", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gl_cases = load_json(args.benchmark)
    if not isinstance(gl_cases, list):
        raise ValueError(f"{args.benchmark} must contain a JSON list.")

    candidates = load_vulkan_candidates(args.vulkan_root)
    if not candidates:
        raise ValueError("No eligible Vulkan candidates were found.")

    embedder = BGEEmbedder(args.model_name)
    candidate_embeddings = embedder.encode([item.text for item in candidates], batch_size=args.candidate_batch)
    output_records = build_output_records(
        gl_cases=gl_cases,
        candidates=candidates,
        candidate_embeddings=candidate_embeddings,
        embedder=embedder,
        query_batch_size=args.query_batch,
        model_name=args.model_name,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(output_records)} records to {output_path}")


if __name__ == "__main__":
    main()
