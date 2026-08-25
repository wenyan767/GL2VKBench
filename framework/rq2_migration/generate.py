"""Step 1 —— 种子生成（基线 ① non-cot 与 ② cot）。

源自 run_rag_cot_generation.py。改动：

  * Prompt 与数据集路径改为包内相对路径。
  * 客户端改为用户提供的外部 JSON runner，不发布厂商 SDK、凭据或配置加载器。
  * 输出落到 runs/seeds/{non_cot,cot}/{model}/case_{row}/。

两条种子线各生成一次。它们本身就是基线 ① 和 ②；cot 那条在论文表格里同时
充当 w/o feedback（cot 去掉反馈修复的那一格），不单独跑。
"""

import argparse
import json
import re
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


from paths import (
    GENERATION_PROMPT_DIR,
    seed_case_dir,
)
from methods import SEED_PROMPT_FILE
from input_contract import load_records, reference_template_type
import model_adapter
import core

PROMPT_TEMPLATE_PATH = GENERATION_PROMPT_DIR / SEED_PROMPT_FILE["cot"]
NON_COT_PROMPT_TEMPLATE_PATH = GENERATION_PROMPT_DIR / SEED_PROMPT_FILE["non_cot"]
def log(message: str) -> None:
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        target_encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe_message = message.encode(target_encoding, errors="backslashreplace").decode(
            target_encoding,
            errors="replace",
        )
        print(safe_message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RQ2 non-CoT/CoT migration seeds from an explicit retrieval result."
    )
    parser.add_argument(
        "--mode",
        choices=["cot", "non-cot", "woFeedback"],
        default="woFeedback",
        help="Prompt mode. 'woFeedback' (or legacy 'cot') uses the CoT template; 'non-cot' uses the pure RAG template.",
    )
    parser.add_argument("--model", choices=sorted(model_adapter.MODELS), default=None,
                        help="Single paper model key")
    parser.add_argument("--models", nargs="*", choices=sorted(model_adapter.MODELS), default=None,
                        help="Multiple paper model keys")
    parser.add_argument("--input-json", required=True, type=Path,
                        help="User-generated RQ2 retrieval output JSON")
    parser.add_argument("--template-path", default=None, help="Prompt template path")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--row-id", type=int, default=None, help="Only process one row_id")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N records")
    parser.add_argument("--dry-run", action="store_true", help="Only build prompt files without calling the LLM")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing response files")
    parser.add_argument("--prompt-budget", type=int, default=46000, help="Estimated input token budget")
    parser.add_argument("--max-code-chars-subclass", type=int, default=48000, help="Max chars for gl_subclass code before fallback compression")
    parser.add_argument("--max-code-chars-similar", type=int, default=36000, help="Max chars for similar_vulkan_case code before fallback compression")
    parser.add_argument("--max-base-code-chars", type=int, default=5000, help="Total max chars reserved for all base-class code excerpts")
    parser.add_argument("--stats-file", default="usage_stats.jsonl", help="JSONL usage stats file name inside output dir")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Resume completed cases and rerun incomplete/failed ones")
    model_adapter.add_runner_arguments(parser)
    return parser.parse_args()


def resolve_mode_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    # mode 名与 seed_kind 的对应：non-cot -> non_cot, cot -> cot
    seed_kind = "non_cot" if args.mode == "non-cot" else "cot"
    if args.mode == "non-cot":
        default_template = NON_COT_PROMPT_TEMPLATE_PATH
    else:
        default_template = PROMPT_TEMPLATE_PATH
    # 取到 runs/seeds/{seed_kind} 这一层；逐 case 的 model/case_{row} 由下游拼。
    # seed_case_dir 返回 .../seeds/{kind}/{model}/case_{row}，上溯两级即所需。
    default_output = seed_case_dir(seed_kind, "_", 0).parent.parent

    template_path = Path(args.template_path) if args.template_path else default_template
    output_dir = Path(args.output_dir) if args.output_dir else default_output
    return template_path, output_dir


def resolve_models(args: argparse.Namespace) -> List[str]:
    if args.models:
        return [item.lower() for item in args.models]
    if args.model:
        return [args.model.lower()]
    return ["gpt", "gemini", "claude", "glm", "deepseek"]


def load_json(path: Path) -> Any:
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    return json.loads(path.read_text(encoding="latin-1"))


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken  # type: ignore

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def strip_md_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def strip_think_blocks(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"^\s*<think>.*?</think>\s*", "", stripped, flags=re.S)
    return stripped.strip()


def normalize_repeated_fenced_json(text: str) -> str:
    stripped = text.strip()
    fenced_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S)
    if len(fenced_blocks) < 2:
        return stripped

    normalized_blocks = []
    seen = set()
    for block in fenced_blocks:
        compact = re.sub(r"\s+", "", block)
        if compact in seen:
            continue
        seen.add(compact)
        normalized_blocks.append(block.strip())

    if len(normalized_blocks) == 1:
        return normalized_blocks[0]
    if normalized_blocks:
        return "\n\n".join(normalized_blocks)
    return stripped


def extract_balanced_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def compress_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_code_skeleton(code: str, max_chars: int) -> str:
    code = compress_whitespace(code)
    if len(code) <= max_chars:
        return code

    lines = code.splitlines()
    kept: List[str] = []
    patterns = [
        re.compile(r"^\s*(class|struct)\s+\w+"),
        re.compile(r"^\s*[\w:<>~]+\s*\("),
        re.compile(r"\b(iterate|init|deinit|checkSupport|initPrograms|createInstance)\b"),
        re.compile(r"^\s*(public|protected|private)\s*:"),
        re.compile(r"^\s*[\w:<>,~*&\s]+\s+\w+\s*;"),
    ]
    for line in lines:
        if any(pattern.search(line) for pattern in patterns):
            kept.append(line)
    skeleton = compress_whitespace("\n".join(kept))
    if not skeleton:
        skeleton = code[:max_chars]
    if len(skeleton) > max_chars:
        skeleton = skeleton[:max_chars].rstrip() + "\n...<truncated>"
    else:
        skeleton += "\n...<truncated>"
    return skeleton


def build_base_class_summary(base_class: Dict[str, Any], excerpt_chars: int) -> Dict[str, Any]:
    code = base_class.get("code") or ""
    excerpt = extract_code_skeleton(code, excerpt_chars) if code else ""
    key_methods = sorted(set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", excerpt)))[:20]
    key_members = sorted(set(re.findall(r"\b([A-Za-z_]\w*)\s*;", excerpt)))[:20]
    return {
        "class_id": base_class.get("class_id"),
        "name": base_class.get("name"),
        "relationship": base_class.get("relationship"),
        "qualified_name": base_class.get("qualified_name"),
        "namespace": base_class.get("namespace"),
        "file_path": base_class.get("file_path"),
        "base_class_summary": {
            "role": "OpenGL base class context for subclass migration",
            "key_methods": key_methods,
            "key_members": key_members,
        },
        "code_excerpt": excerpt,
    }


def reduce_base_classes(base_classes: List[Dict[str, Any]], total_chars: int) -> List[Dict[str, Any]]:
    if not base_classes:
        return []
    per_class = max(800, total_chars // max(1, len(base_classes)))
    return [build_base_class_summary(item, per_class) for item in base_classes]


def build_case_payload(case: Dict[str, Any], args: argparse.Namespace, compact: bool) -> Dict[str, Any]:
    payload = deepcopy(case)
    gl_subclass = payload.get("gl_subclass") or {}
    similar = payload.get("similar_vulkan_case") or {}
    base_classes = payload.get("gl_base_classes") or []

    subclass_chars = args.max_code_chars_subclass if not compact else max(12000, args.max_code_chars_subclass // 2)
    similar_chars = args.max_code_chars_similar if not compact else max(10000, args.max_code_chars_similar // 2)
    base_chars = args.max_base_code_chars if not compact else max(1500, args.max_base_code_chars // 2)

    gl_subclass["code"] = extract_code_skeleton(gl_subclass.get("code") or "", subclass_chars)
    similar["code"] = extract_code_skeleton(similar.get("code") or "", similar_chars)
    payload["gl_base_classes"] = reduce_base_classes(base_classes, base_chars)
    payload["gl_subclass"] = gl_subclass
    payload["similar_vulkan_case"] = similar
    return payload


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def render_prompt(template: str, case_payload: Dict[str, Any]) -> str:
    case_json = json.dumps(case_payload, ensure_ascii=False, indent=2)
    return template.replace("{{CASE_JSON}}", case_json)


def build_budgeted_prompt(case: Dict[str, Any], template: str, args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    full_payload = build_case_payload(case, args, compact=False)
    prompt = render_prompt(template, full_payload)
    est_tokens = estimate_tokens(prompt)
    mode = "full"
    used_payload = full_payload

    if est_tokens > args.prompt_budget:
        compact_payload = build_case_payload(case, args, compact=True)
        compact_prompt = render_prompt(template, compact_payload)
        compact_tokens = estimate_tokens(compact_prompt)
        prompt = compact_prompt
        est_tokens = compact_tokens
        mode = "compact"
        used_payload = compact_payload

    metadata = {
        "row_id": case.get("row_id"),
        "mode": mode,
        "estimated_prompt_tokens": est_tokens,
        "prompt_budget": args.prompt_budget,
        "gl_base_classes_count": len(used_payload.get("gl_base_classes") or []),
        "subclass_code_chars": len((used_payload.get("gl_subclass") or {}).get("code") or ""),
        "similar_code_chars": len((used_payload.get("similar_vulkan_case") or {}).get("code") or ""),
    }
    return prompt, metadata


def parse_response_content(content: str) -> Dict[str, Any]:
    normalized = normalize_repeated_fenced_json(content)
    stripped = strip_md_fence(normalized)
    stripped = strip_think_blocks(stripped)

    balanced_candidate = extract_balanced_json_object(stripped)
    if balanced_candidate:
        try:
            parsed = json.loads(balanced_candidate)
            if isinstance(parsed, dict) and parsed:
                return parsed
        except Exception:
            pass

    for candidate in [stripped]:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and parsed:
                return parsed
        except Exception:
            pass

    fenced_json_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S)
    for candidate in fenced_json_blocks:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and parsed:
                return parsed
        except Exception:
            pass

    decoder = json.JSONDecoder()
    for start_match in re.finditer(r"\{", stripped):
        start = start_match.start()
        try:
            parsed, end = decoder.raw_decode(stripped[start:])
            if isinstance(parsed, dict) and parsed:
                return parsed
        except Exception:
            continue

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and parsed:
                return parsed
        except Exception:
            pass

    return {"raw_response": stripped}


def extract_cpp_from_raw_text(raw_text: str) -> Optional[str]:
    if not isinstance(raw_text, str):
        return None
    text = strip_think_blocks(raw_text).strip()
    if not text:
        return None

    fenced_patterns = [
        r"```cpp\s*(.*?)\s*```",
        r"```c\+\+\s*(.*?)\s*```",
        r"```cxx\s*(.*?)\s*```",
        r"```cc\s*(.*?)\s*```",
        r"```\s*(#include .*?)\s*```",
    ]
    for pattern in fenced_patterns:
        matches = re.findall(pattern, text, flags=re.S | re.I)
        for match in matches:
            candidate = match.strip()
            if candidate:
                return candidate

    for key in ("vulkan_test_class_code", "class_code", "function_code", "code"):
        candidate = extract_json_like_code_field(text, key)
        if candidate:
            return candidate

    marker_patterns = [
        r"(#include\s+[<\"].*?(?:\n.*)*)",
        r"((?:static\s+)?void\s+\w+\s*\([^)]*\)\s*\{.*)",
        r"(class\s+\w+.*?\{.*)",
    ]
    for pattern in marker_patterns:
        match = re.search(pattern, text, flags=re.S)
        if not match:
            continue
        candidate = match.group(1).strip()
        if candidate:
            return candidate

    return None


def _find_next_non_whitespace(text: str, start: int) -> int:
    for index in range(start, len(text)):
        if not text[index].isspace():
            return index
    return -1


def extract_json_like_code_field(text: str, key: str) -> Optional[str]:
    marker = f'"{key}"'
    key_index = text.find(marker)
    if key_index == -1:
        return None

    colon_index = text.find(":", key_index + len(marker))
    if colon_index == -1:
        return None

    value_start = _find_next_non_whitespace(text, colon_index + 1)
    if value_start == -1 or text[value_start] != '"':
        return None

    chars: List[str] = []
    escape = False
    for index in range(value_start + 1, len(text)):
        ch = text[index]
        if escape:
            chars.append("\\" + ch)
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            # 未转义的 " 不一定是值的结尾 —— 模型会把 C++ 字符串字面量原样写进 JSON
            # 字符串值,例如 << "#version 460\n"。实测 gpt 三个 case 里两个这样,
            # 于是原实现在第一个 << " 处提前收尾:14103 字节的产出只留下 803 字节
            # (丢 94%),而 status 仍写 "ok",只有 cpp_extraction_method=regex_fallback
            # 留下痕迹 —— 坏数据看起来是正常数据,这是最需要防的形态。
            #
            # 判据分两层,只看后一个字符不够:
            #   1. 结尾引号后面(跳过空白)必须是 JSON 结构符 , } ]
            #   2. 但 C++ 里 "1d"   }, 和 "}\n"; 也满足第 1 层(实测 gpt 两个 case
            #      的第二个断点正是这两种),所以还要求此刻代码的花括号是配平的 ——
            #      值真的结束时,类体的 { } 必然已经成对。
            # claude/gemini 走结构化路径,不经过这里,不受影响。
            nxt = _find_next_non_whitespace(text, index + 1)
            if nxt != -1 and text[nxt] not in ',}]':
                chars.append(ch)
                continue
            pending = "".join(chars)
            if pending.count("{") != pending.count("}"):
                chars.append(ch)
                continue
            raw = pending
            try:
                decoded = json.loads(f'"{raw}"')
            except Exception:
                # raw 里现在可能含未转义的裸 " ,json.loads 会失败;逐个转义再解一次,
                # 失败才退到 unicode_escape(它不认 \\" 且会毁掉非 ASCII)。
                try:
                    decoded = json.loads('"' + raw.replace('"', '\\"') + '"')
                except Exception:
                    decoded = raw.encode("utf-8").decode("unicode_escape")
            decoded = decoded.strip()
            return decoded or None
        chars.append(ch)

    raw = "".join(chars)
    raw = raw.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
    raw = raw.strip()
    return raw or None


def extract_cpp_code_with_method(parsed_response: Dict[str, Any], raw_text: Optional[str] = None) -> Tuple[Optional[str], str]:
    for key in ("vulkan_test_class_code", "class_code", "function_code", "code"):
        code = parsed_response.get(key)
        if isinstance(code, str) and code.strip():
            return code.strip(), f"structured:{key}"
    response_obj = parsed_response.get("response")
    if isinstance(response_obj, dict):
        for key in ("vulkan_test_class_code", "class_code", "function_code", "code"):
            code = response_obj.get(key)
            if isinstance(code, str) and code.strip():
                return code.strip(), f"structured:response.{key}"
    if isinstance(raw_text, str):
        code = extract_cpp_from_raw_text(raw_text)
        if code:
            return code, "regex_fallback:full_response_text"
    raw_response = parsed_response.get("raw_response")
    if isinstance(raw_response, str):
        code = extract_cpp_from_raw_text(raw_response)
        if code:
            return code, "regex_fallback:raw_response"
    return None, "missing"


def iter_cases(cases: List[Dict[str, Any]], row_id: Optional[int], limit: Optional[int]) -> List[Dict[str, Any]]:
    selected = cases
    if row_id is not None:
        selected = [item for item in cases if item.get("row_id") == row_id]
    if limit is not None:
        selected = selected[:limit]
    return selected


def extract_cpp_code(parsed_response: Dict[str, Any]) -> Optional[str]:
    code, _ = extract_cpp_code_with_method(parsed_response)
    return code


def is_valid_response_payload(payload: Dict[str, Any]) -> bool:
    response = payload.get("response")
    return isinstance(response, dict) and bool(extract_cpp_code(response))


def load_existing_payload(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = load_json(path)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def load_existing_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def has_nonempty_text(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return bool(path.read_text(encoding="utf-8").strip())
    except Exception:
        return False


def is_unrecoverable_completed_non_cpp_case(
    response_payload: Optional[Dict[str, Any]],
    status_payload: Optional[Dict[str, Any]],
    raw_response_path: Path,
    cpp_path: Path,
) -> bool:
    return False


def case_paths(case_dir: Path) -> Dict[str, Path]:
    return {
        "response_json": case_dir / "response.json",
        "response_raw": case_dir / "response_raw.txt",
        "generated_cpp": case_dir / "generated.cpp",
        # Step 2 consumes this stable seed filename through core.prepare_case().
        # Keep generated.cpp as the user-facing artifact and write both together.
        "raw_generated_cpp": case_dir / "raw_generated.cpp",
        "prompt": case_dir / "migration_task.txt",
        "opengl_source": case_dir / "opengl_test_code.cpp",
        "opengl_base_classes": case_dir / "opengl_base_classes.json",
        "retry_prompt": case_dir / "retry_missing_cpp_prompt.txt",
        "retry_response_raw": case_dir / "retry_missing_cpp_response_raw.txt",
        "truncated_retry_prompt": case_dir / "retry_truncated_prompt.txt",
        "truncated_retry_response_raw": case_dir / "retry_truncated_response_raw.txt",
        # 采纳重试会覆盖 response_raw.txt,首发原文的落点就丢了。留一份副本,
        # 才能分辨截断是稳定边界(每次同一字节)还是随机丢流(落点分散)。
        "first_response_raw": case_dir / "first_response_raw.txt",
    }


def write_generated_cpp_artifacts(paths: Dict[str, Path], code: str) -> None:
    text = code.rstrip() + "\n"
    paths["generated_cpp"].write_text(text, encoding="utf-8")
    paths["raw_generated_cpp"].write_text(text, encoding="utf-8")


def clear_case_dir(case_dir: Path) -> None:
    if not case_dir.exists():
        return
    for child in case_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def write_case_status(case_dir: Path, payload: Dict[str, Any]) -> None:
    status_path = case_dir / "status.json"
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_case_context_artifacts(case_dir: Path, case: Dict[str, Any], prompt: str) -> None:
    paths = case_paths(case_dir)
    gl_subclass = case.get("gl_subclass") or {}
    gl_base_classes = case.get("gl_base_classes") or []
    source_code = (gl_subclass.get("code") or "").strip()
    if source_code:
        paths["opengl_source"].write_text(source_code + "\n", encoding="utf-8")
    paths["opengl_base_classes"].write_text(
        json.dumps(gl_base_classes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["prompt"].write_text(prompt, encoding="utf-8")


def strip_literals_and_comments(text: str) -> str:
    """去掉字符串/字符字面量和注释,只留真正的代码括号。

    换行不终止字面量:模型会把整段 GLSL 写成跨行字符串
    (<< "#version 450 后面直接真换行),里面的花括号不是代码。
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            i = text.find("\n", i)
            if i == -1:
                break
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        if ch in ('"', "'"):
            quote, i = ch, i + 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def response_looks_truncated(
    raw_text: Optional[str], cpp_code: Optional[str] = None
) -> bool:
    """模型原文是否在半句里被切断。

    为什么需要:流式读取只累加 text_delta,不看 stop_reason —— SSE 流提前断开时,
    已收到的部分被当成正常成功返回,status 照写 "ok"。实测 rq4 全量里 claude 31/100、
    gpt 1/100 是这样,产物停在 `val = (a == 1)`、`vk::VK_` 这种半句上。
    这种种子必然编译失败,若直接进 ③④⑤,记到账上的是传输缺陷而不是模型的迁移能力。

    判据用原文尾部而不是花括号配平:配平会被字面量里的 GLSL 花括号骗
    (<< "void main() {\\n" 单边出现),实测会误报 16 个完整 case。
    完整回包收在 JSON 的 } 或 ``` 上;截断的收在代码中间。

    实测这不是固定上限:同样的 case 重跑,22.7KB 的原文能拿到 29.0KB 且合法收尾,
    所以重试是有效的补救手段。

    尾部判据会漏一类:截断正好停在嵌套初始化列表的 } 上。实测 claude case_28 原文末尾
    是 `{ vk::VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u }\\n        }`,尾字符合法而
    花括号深度 +2、语句没有分号。所以传入 cpp_code 时补一条"缺闭合"判据:剥掉字面量
    后深度为正 = 少写闭合括号 = 截断。深度为负(多写闭合)不算 —— 那是模型自己的笔误,
    该交给 ③④⑤ 的编译修复,不该在这里重生成掉。
    """
    if not raw_text:
        return False
    tail = raw_text.rstrip()
    if not tail:
        return False
    if not (tail.endswith("}") or tail.endswith("```")):
        return True
    if not cpp_code:
        return False
    code = strip_literals_and_comments(cpp_code)
    return (code.count("{") - code.count("}")) > 0 or (
        code.count("(") - code.count(")")
    ) > 0


def build_truncated_retry_prompt(base_prompt: str) -> str:
    return (
        base_prompt.rstrip()
        + "\n\n---\n\n"
        + "Previous response was cut off mid-output and could not be used.\n"
        + "Retry the same migration now and return the complete result.\n"
        + "Return strict JSON only, fully closed.\n"
        + "Ensure the JSON object is terminated properly.\n"
    )
    # 刻意不加"写紧凑些/别写注释"之类的额外要求:基础模板(rag_prompt_template
    # _gl_sim_vk.txt:93-94)已经禁了注释和 TODO,再强调一遍是重复;而任何只有
    # claude 收到、gemini/gpt 没收到的额外约束都会让三方不可比 —— 截断是传输问题,
    # 补救手段不该顺带改变提示词。所以这里只是"原样再问一次,把话说完"。


def build_missing_cpp_retry_prompt(base_prompt: str) -> str:
    return (
        base_prompt.rstrip()
        + "\n\n---\n\n"
        + "Previous response did not yield usable C++ code.\n"
        + "Retry the same migration now.\n"
        + "Return strict JSON only.\n"
        + "Do not include analysis prose outside JSON.\n"
        + "At minimum, provide one non-empty code field.\n"
        + "Preferred field order:\n"
        + '1. "vulkan_test_class_code"\n'
        + '2. "class_code"\n'
        + '3. "function_code"\n'
        + "If the full JSON cannot be perfectly completed, still return the code-bearing field with the migrated C++ body filled in.\n"
        + "Do not leave the code field empty.\n"
    )


def expected_template_type(case: Dict[str, Any]) -> str:
    similar_case = case.get("similar_vulkan_case") or {}
    return reference_template_type(similar_case.get("class_type"))


def generated_template_type(
    cpp_code: Optional[str], *, mode_name: str, model_name: str, row_id: int
) -> Optional[str]:
    if not cpp_code or not cpp_code.strip():
        return None
    try:
        return core.classify_candidate_code(
            cpp_code,
            mode_name=mode_name,
            model_name=model_name,
            row_id=row_id,
        ).template_type
    except ValueError:
        return None


def scan_model_output_dir(model_output_dir: Path, model_name: str) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    if not model_output_dir.exists():
        return summary

    for case_dir in sorted(model_output_dir.glob("case_*")):
        if not case_dir.is_dir():
            continue
        status_path = case_dir / "status.json"
        response_path = case_dir / "response.json"
        cpp_path = case_dir / "generated.cpp"

        if status_path.exists():
            payload = load_existing_payload(status_path)
            if isinstance(payload, dict):
                summary.append(payload)
                continue

        row_text = case_dir.name.replace("case_", "")
        try:
            row_id = int(row_text)
        except ValueError:
            row_id = row_text

        status = "missing_response"
        if response_path.exists() and cpp_path.exists():
            status = "ok"
        elif response_path.exists():
            status = "response_without_cpp"

        summary.append(
            {
                "row_id": row_id,
                "model": model_name,
                "status": status,
                "case_dir": str(case_dir),
                "response_file": str(response_path) if response_path.exists() else None,
                "raw_response_file": str(case_dir / "response_raw.txt") if (case_dir / "response_raw.txt").exists() else None,
                "cpp_file": str(cpp_path) if cpp_path.exists() else None,
                "cpp_extraction_method": None,
            }
        )

    return summary


def build_run_summary_metrics(run_summary: List[Dict[str, Any]], mode_name: str, model_name: str) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {}
    extraction_method_counts: Dict[str, int] = {}
    regex_fallback_cases: List[int] = []
    missing_cpp_cases: List[int] = []

    for item in run_summary:
        status = str(item.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

        method = str(item.get("cpp_extraction_method") or "none")
        extraction_method_counts[method] = extraction_method_counts.get(method, 0) + 1

        row_id = item.get("row_id")
        if method.startswith("regex_fallback:") and isinstance(row_id, int):
            regex_fallback_cases.append(row_id)
        if status in {"response_without_cpp", "skipped_unparseable_completed_response"} and isinstance(row_id, int):
            missing_cpp_cases.append(row_id)

    return {
        "mode": mode_name,
        "model": model_name,
        "total_cases": len(run_summary),
        "status_counts": status_counts,
        "cpp_extraction_method_counts": extraction_method_counts,
        "regex_fallback_case_count": len(regex_fallback_cases),
        "regex_fallback_cases": sorted(regex_fallback_cases),
        "missing_cpp_case_count": len(missing_cpp_cases),
        "missing_cpp_cases": sorted(missing_cpp_cases),
    }


def scan_batch_output_dir(root_output_dir: Path) -> List[Dict[str, Any]]:
    batch_summary: List[Dict[str, Any]] = []
    for model_dir in sorted(root_output_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        run_summary_path = model_dir / "run_summary.json"
        if run_summary_path.exists():
            payload = load_existing_json(run_summary_path)
            if isinstance(payload, list):
                failed_rows = [
                    item.get("row_id") for item in payload
                    if not generation_case_succeeded(item, dry_run=False)
                ]
                batch_summary.append(
                    {
                        "model": model_name,
                        "status": "ok" if payload and not failed_rows else "failed",
                        "output_dir": str(model_dir),
                        "cases_processed": len(payload),
                        "failed_rows": failed_rows,
                    }
                )
                continue
        batch_summary.append(
            {
                "model": model_name,
                "status": "no_summary",
                "output_dir": str(model_dir),
            }
        )
    return batch_summary


GENERATION_SUCCESS_STATUSES = {
    "ok",
    "skipped_completed",
    "recovered_cpp_from_existing_json",
}


def generation_case_succeeded(item: Dict[str, Any], *, dry_run: bool) -> bool:
    """Return whether one requested case produced the artifact Step 2 consumes."""
    status = str(item.get("status") or "")
    if dry_run:
        return status == "prompt_built"
    if status not in GENERATION_SUCCESS_STATUSES:
        return False
    case_dir = Path(str(item.get("case_dir") or ""))
    return has_nonempty_text(case_dir / "generated.cpp") and has_nonempty_text(
        case_dir / "raw_generated.cpp"
    )


def run_for_model(
    model_name: str,
    args: argparse.Namespace,
    selected_cases: List[Dict[str, Any]],
    template: str,
    root_output_dir: Path,
) -> Dict[str, Any]:
    model_output_dir = root_output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = model_output_dir / args.stats_file
    log(f"[MODEL] {model_name}: output_dir={model_output_dir}")

    client = None
    if not args.dry_run:
        log(f"[CLIENT] {model_name}: creating client")
        # 私有模型配置完全由外部 runner 管理；Framework 不读凭据或厂商配置。
        client = model_adapter.create_client(model_name, stats_path=stats_path)
        log(f"[CLIENT] {model_name}: client ready")
    else:
        log(f"[DRYRUN] {model_name}: prompt build only")

    total_cases = len(selected_cases)
    for index, case in enumerate(selected_cases, start=1):
        row_id = case.get("row_id")
        row_id_int = int(row_id)
        reference_template_type_value = expected_template_type(case)
        case_dir = model_output_dir / f"case_{row_id}"
        case_dir.mkdir(parents=True, exist_ok=True)
        paths = case_paths(case_dir)
        status_path = case_dir / "status.json"
        log(f"[CASE] {model_name} [{index}/{total_cases}] row_id={row_id}")
        existing_payload = load_existing_payload(paths["response_json"]) if args.resume and not args.overwrite else None
        existing_status = load_existing_payload(status_path) if args.resume and not args.overwrite else None
        # 完成判据以 generated.cpp 为准,不再用 is_valid_response_payload 把门。
        # 原因:那个判据只看 response.json 的 response 字段,且内部 raw_text=None,
        # 走 regex_fallback:full_response_text 抽出来的 case(response 字段里只剩
        # checklist 布尔,代码在 response_raw.txt 里)会被判成未完成,短路掉下面
        # 读 response_raw.txt 的逻辑,最终落到 clear_case_dir 删掉有效 cpp 重跑。
        # 实测 96 个 case 命中(gemini 11 / glm 81 / gpt 3 / deepseek 1)。
        # 下游 ③④⑤ 只读 generated.cpp,它非空就说明这个 case 可用。
        if existing_payload and not args.dry_run:
            cpp_exists = has_nonempty_text(paths["generated_cpp"])
            # 已有产物但原文是半句 -> 不算完成,重跑。
            # 否则续跑会永久跳过这些 case:判据只看 generated.cpp 非空,
            # 而截断的产物恰好非空。实测重跑能救回大部分(22.7KB -> 29.0KB 合法收尾)。
            if cpp_exists and paths["response_raw"].exists():
                if response_looks_truncated(
                    paths["response_raw"].read_text(encoding="utf-8", errors="replace"),
                    paths["generated_cpp"].read_text(encoding="utf-8", errors="replace"),
                ):
                    log(f"[CASE] {model_name} row_id={row_id}: existing response truncated, regenerating")
                    cpp_exists = False
            if cpp_exists:
                if not has_nonempty_text(paths["raw_generated_cpp"]):
                    write_generated_cpp_artifacts(
                        paths,
                        paths["generated_cpp"].read_text(encoding="utf-8", errors="replace"),
                    )
                existing_response = existing_payload.get("response", {})
                raw_text = None
                if paths["response_raw"].exists():
                    raw_text = paths["response_raw"].read_text(encoding="utf-8")
                _, extraction_method = extract_cpp_code_with_method(
                    existing_response if isinstance(existing_response, dict) else {},
                    raw_text=raw_text,
                )
                log(f"[CASE] {model_name} row_id={row_id}: skip completed")
                write_case_status(
                    case_dir,
                    {
                        "row_id": row_id,
                        "model": model_name,
                        "status": "skipped_completed",
                        "case_dir": str(case_dir),
                        "response_file": str(paths["response_json"]),
                        "cpp_file": str(paths["generated_cpp"]),
                        "cpp_extraction_method": extraction_method,
                    },
                )
                continue
            existing_response = existing_payload.get("response", {})
            raw_text = None
            if paths["response_raw"].exists():
                raw_text = paths["response_raw"].read_text(encoding="utf-8")
            existing_cpp, extraction_method = extract_cpp_code_with_method(
                existing_response if isinstance(existing_response, dict) else {},
                raw_text=raw_text,
            )
            # 原文截断时不走"从已有 response.json 里捞回来":捞回来的就是那份半句代码,
            # 等于把上面刚判定要重跑的结论撤销掉。让它落下去重新调模型。
            if existing_cpp and response_looks_truncated(raw_text, existing_cpp):
                existing_cpp = None
            if existing_cpp:
                existing_template_type = generated_template_type(
                    existing_cpp,
                    mode_name=args.mode,
                    model_name=model_name,
                    row_id=row_id_int,
                )
                write_generated_cpp_artifacts(paths, existing_cpp)
                log(
                    f"[CASE] {model_name} row_id={row_id}: recovered cpp from existing response.json "
                    f"via {extraction_method} (actual template_type={existing_template_type!r})"
                )
                write_case_status(
                    case_dir,
                    {
                        "row_id": row_id,
                        "model": model_name,
                        "status": "recovered_cpp_from_existing_json",
                        "case_dir": str(case_dir),
                        "response_file": str(paths["response_json"]),
                        "cpp_file": str(paths["generated_cpp"]),
                        "cpp_extraction_method": extraction_method,
                        "reference_template_type": reference_template_type_value,
                        "generated_template_type": existing_template_type,
                    },
                )
                continue
        if (
            args.resume
            and not args.overwrite
            and not args.dry_run
            and is_unrecoverable_completed_non_cpp_case(
                response_payload=existing_payload,
                status_payload=existing_status,
                raw_response_path=paths["response_raw"],
                cpp_path=paths["generated_cpp"],
            )
        ):
            log(
                f"[CASE] {model_name} row_id={row_id}: skip rerun because a complete raw response "
                f"already exists but no usable cpp could be extracted"
            )
            write_case_status(
                case_dir,
                {
                    "row_id": row_id,
                    "model": model_name,
                    "status": "skipped_unparseable_completed_response",
                    "case_dir": str(case_dir),
                    "response_file": str(paths["response_json"]) if paths["response_json"].exists() else None,
                    "raw_response_file": str(paths["response_raw"]),
                    "cpp_file": None,
                    "cpp_extraction_method": "missing",
                },
            )
            continue

        if any(case_dir.iterdir()):
            log(f"[CASE] {model_name} row_id={row_id}: clearing existing case directory before rerun")
            clear_case_dir(case_dir)
            paths = case_paths(case_dir)

        log(f"[PROMPT] {model_name} row_id={row_id}: building prompt")
        prompt, metadata = build_budgeted_prompt(case, template, args)
        metadata["reference_template_type"] = reference_template_type_value
        log(
            f"[PROMPT] {model_name} row_id={row_id}: "
            f"mode={metadata['mode']}, est_tokens={metadata['estimated_prompt_tokens']}, "
            f"base_classes={metadata['gl_base_classes_count']}"
        )
        write_case_context_artifacts(case_dir, case, prompt)

        if args.dry_run:
            log(f"[DRYRUN] {model_name} row_id={row_id}: prompt built")
            write_case_status(
                case_dir,
                {
                    "row_id": row_id,
                    "model": model_name,
                    "status": "prompt_built",
                    "case_dir": str(case_dir),
                    "cpp_extraction_method": None,
                    **metadata,
                },
            )
            continue

        log(f"[LLM] {model_name} row_id={row_id}: sending request")
        try:
            response_text = client.generate(
                prompt,
                description=f"rag_cot_generation_row_{row_id}",
                meta={"row_id": row_id, "model": model_name, **metadata},
            )
        except Exception as exc:
            log(f"[LLM] {model_name} row_id={row_id}: request failed: {exc}")
            write_case_status(
                case_dir,
                {
                    "row_id": row_id,
                    "model": model_name,
                    "status": "llm_failed",
                    "case_dir": str(case_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                    **metadata,
                },
            )
            continue
        log(f"[LLM] {model_name} row_id={row_id}: response received, saving raw output")
        paths["response_raw"].write_text(response_text, encoding="utf-8")
        log(f"[PARSE] {model_name} row_id={row_id}: parsing model response")
        parsed_response = parse_response_content(response_text)
        cpp_code, extraction_method = extract_cpp_code_with_method(parsed_response, raw_text=response_text)
        retry_used = False
        retry_error = None
        truncated_retry_used = False
        truncated_retry_error = None
        response_truncated = response_looks_truncated(response_text, cpp_code)
        # 抽出了代码但原文是半句 —— 这是最需要拦的形态:status 会写 ok,
        # 产物看起来正常,实际必然编译失败。重试一次;两次都截断就保留更长的那份
        # 并在 extraction_method 上留标记,让下游能筛掉,而不是静默当好数据。
        if cpp_code and response_truncated:
            truncated_prompt = build_truncated_retry_prompt(prompt)
            paths["truncated_retry_prompt"].write_text(truncated_prompt, encoding="utf-8")
            log(f"[LLM] {model_name} row_id={row_id}: response truncated mid-output, retrying once")
            try:
                retry_text = client.generate(
                    truncated_prompt,
                    description=f"rag_cot_generation_row_{row_id}_retry_truncated",
                    meta={"row_id": row_id, "model": model_name, "retry_reason": "truncated", **metadata},
                )
                truncated_retry_used = True
                paths["truncated_retry_response_raw"].write_text(retry_text, encoding="utf-8")
                retry_parsed = parse_response_content(retry_text)
                retry_cpp, retry_method = extract_cpp_code_with_method(retry_parsed, raw_text=retry_text)
                retry_truncated = response_looks_truncated(retry_text, retry_cpp)
                # 采纳条件:重试完整,或重试同样截断但内容更长(更长=拿到更多有效代码)
                if retry_cpp and (not retry_truncated or len(retry_cpp) > len(cpp_code)):
                    if not paths["first_response_raw"].exists():
                        paths["first_response_raw"].write_text(
                            response_text, encoding="utf-8")
                    response_text = retry_text
                    parsed_response = retry_parsed
                    cpp_code = retry_cpp
                    response_truncated = retry_truncated
                    extraction_method = f"retry_truncated:{retry_method}"
                    log(f"[LLM] {model_name} row_id={row_id}: retry accepted "
                        f"(truncated={retry_truncated})")
                else:
                    log(f"[LLM] {model_name} row_id={row_id}: retry not better, keeping first response")
            except Exception as exc:
                truncated_retry_error = f"{type(exc).__name__}: {exc}"
                log(f"[LLM] {model_name} row_id={row_id}: truncated retry failed: {exc}")
        if response_truncated and cpp_code:
            extraction_method = f"{extraction_method}|TRUNCATED"
        if not cpp_code:
            retry_prompt = build_missing_cpp_retry_prompt(prompt)
            paths["retry_prompt"].write_text(retry_prompt, encoding="utf-8")
            log(f"[LLM] {model_name} row_id={row_id}: no cpp extracted, retrying once with stricter JSON/code constraints")
            try:
                retry_response_text = client.generate(
                    retry_prompt,
                    description=f"rag_cot_generation_row_{row_id}_retry_missing_cpp",
                    meta={"row_id": row_id, "model": model_name, "retry_reason": "missing_cpp", **metadata},
                )
                retry_used = True
                paths["retry_response_raw"].write_text(retry_response_text, encoding="utf-8")
                retry_parsed_response = parse_response_content(retry_response_text)
                retry_cpp_code, retry_extraction_method = extract_cpp_code_with_method(
                    retry_parsed_response,
                    raw_text=retry_response_text,
                )
                if retry_cpp_code:
                    response_text = retry_response_text
                    parsed_response = retry_parsed_response
                    cpp_code = retry_cpp_code
                    extraction_method = f"retry_missing_cpp:{retry_extraction_method}"
                else:
                    extraction_method = f"{extraction_method}|retry_missing_cpp:missing"
            except Exception as exc:
                retry_error = f"{type(exc).__name__}: {exc}"
                log(f"[LLM] {model_name} row_id={row_id}: retry for missing cpp failed: {exc}")
        actual_template_type = generated_template_type(
            cpp_code,
            mode_name=args.mode,
            model_name=model_name,
            row_id=row_id_int,
        )
        if cpp_code:
            write_generated_cpp_artifacts(paths, cpp_code)
            log(
                f"[CPP] {model_name} row_id={row_id}: generated.cpp and "
                f"raw_generated.cpp saved via {extraction_method}; "
                f"actual template_type={actual_template_type!r}, "
                f"reference template_type={reference_template_type_value!r}"
            )
        else:
            log(f"[CPP] {model_name} row_id={row_id}: no cpp extracted from response")
        response_payload = {
            "row_id": row_id,
            "model": model_name,
            "case_dir": str(case_dir),
            "metadata": metadata,
            "response": parsed_response,
            "cpp_extraction_method": extraction_method,
            "retry_for_missing_cpp_used": retry_used,
            "retry_for_missing_cpp_error": retry_error,
            "response_truncated": response_truncated,
            "retry_for_truncated_used": truncated_retry_used,
            "retry_for_truncated_error": truncated_retry_error,
            "reference_template_type": reference_template_type_value,
            "generated_template_type": actual_template_type,
            "template_type_matches_reference": actual_template_type == reference_template_type_value,
        }
        paths["response_json"].write_text(
            json.dumps(response_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # 截断且重试后仍截断 -> 单独的状态,不能写 "ok"。
        # 写 ok 就是让坏数据装成好数据:下游只看 status 会照单全收,
        # 而这种种子必然编译失败,最后污染的是模型的编译率。
        if not cpp_code:
            final_status = "response_without_cpp"
        elif response_truncated:
            final_status = "ok_truncated"
        else:
            final_status = "ok"
        log(f"[CASE] {model_name} row_id={row_id}: status={final_status}")
        write_case_status(
            case_dir,
            {
                "row_id": row_id,
                "model": model_name,
                "status": final_status,
                "case_dir": str(case_dir),
                "response_file": str(paths["response_json"]),
                "raw_response_file": str(paths["response_raw"]),
                "cpp_file": str(paths["generated_cpp"]) if cpp_code else None,
                "cpp_extraction_method": extraction_method,
                "retry_for_missing_cpp_used": retry_used,
                "retry_for_missing_cpp_error": retry_error,
                "response_truncated": response_truncated,
                "retry_for_truncated_used": truncated_retry_used,
                "retry_for_truncated_error": truncated_retry_error,
                "reference_template_type": reference_template_type_value,
                "generated_template_type": actual_template_type,
                "template_type_matches_reference": actual_template_type == reference_template_type_value,
                "migration_task_file": str(paths["prompt"]),
                "opengl_source_file": str(paths["opengl_source"]) if paths["opengl_source"].exists() else None,
                **metadata,
            },
        )

    summary_path = model_output_dir / "run_summary.json"
    run_summary = scan_model_output_dir(model_output_dir, model_name)
    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path = model_output_dir / "run_summary_metrics.json"
    summary_metrics = build_run_summary_metrics(run_summary, args.mode, model_name)
    metrics_path.write_text(json.dumps(summary_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[SUMMARY] {model_name}: {len(run_summary)} case(s), {summary_path}")
    log(
        f"[METRICS] {model_name}: structured/regex/missing="
        f"{summary_metrics['cpp_extraction_method_counts']}, {metrics_path}"
    )
    requested_row_ids = {str(case.get("row_id")) for case in selected_cases}
    requested_summary = [
        item for item in run_summary if str(item.get("row_id")) in requested_row_ids
    ]
    failed_rows = [
        item.get("row_id") for item in requested_summary
        if not generation_case_succeeded(item, dry_run=args.dry_run)
    ]
    missing_rows = sorted(
        requested_row_ids - {str(item.get("row_id")) for item in requested_summary}
    )
    succeeded = bool(selected_cases) and not failed_rows and not missing_rows
    return {
        "model": model_name,
        "status": "ok" if succeeded else "failed",
        "output_dir": str(model_output_dir),
        "cases_processed": len(requested_summary),
        "failed_rows": failed_rows,
        "missing_rows": missing_rows,
    }


def main() -> int:
    args = parse_args()
    model_adapter.configure_from_args(args, required=not args.dry_run)
    input_path = args.input_json.resolve()
    template_path, root_output_dir = resolve_mode_paths(args)
    root_output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_records(input_path)
    selected_cases = iter_cases(cases, args.row_id, args.limit)
    template = load_prompt_template(template_path)
    models = resolve_models(args)

    model_results = []
    for model_name in models:
        model_results.append(run_for_model(
            model_name=model_name,
            args=args,
            selected_cases=selected_cases,
            template=template,
            root_output_dir=root_output_dir,
        ))

    batch_summary_path = root_output_dir / "batch_run_summary.json"
    batch_summary_path.write_text(
        json.dumps(model_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"[BATCH] summary saved to {batch_summary_path}")
    return 0 if model_results and all(item["status"] == "ok" for item in model_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
