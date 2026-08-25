"""Vendor-neutral model runner boundary for the public RQ2 Framework.

The Framework does not ship provider clients, credentials, endpoints, or token
pools.  Instead, a user-owned executable receives one JSON request on stdin and
returns one JSON response on stdout.

Request::

    {"prompt": "...", "model_key": "claude", "model_id": "...",
     "description": "...", "meta": {...}}

Response::

    {"response": "raw model response", "usage": {...}}

``usage`` is optional.  The runner may obtain credentials from its own process
environment; this module never reads or records them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional


MODEL_DISPLAY_NAMES: dict[str, str] = {
    "claude": "Claude-4.6",
    "gemini": "Gemini-3.1-pro",
    "gpt": "GPT-5.4",
    "deepseek": "DeepSeek-v3.2",
    "glm": "GLM-5",
}

# Kept as a mapping because the original pipeline uses ``sorted(MODELS)`` and
# displays ``MODELS[key]``.  Values are paper-facing names, not provider IDs.
MODELS = dict(MODEL_DISPLAY_NAMES)
TIMEOUT_SECONDS = 3600


@dataclass
class RunnerConfig:
    runner: Path | None = None
    runner_args: list[str] = field(default_factory=list)
    model_id: str = ""
    timeout_seconds: float = float(TIMEOUT_SECONDS)


_CONFIG = RunnerConfig()


def add_runner_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-runner",
        type=Path,
        help="User-owned executable or Python script implementing the JSON stdin/stdout contract.",
    )
    parser.add_argument(
        "--model-runner-arg",
        action="append",
        default=[],
        help="Argument passed to the external model runner; repeat as needed.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Actual provider model identifier recorded in runner requests.",
    )
    parser.add_argument(
        "--model-timeout-seconds",
        type=float,
        default=None,
        help=f"Per-call external runner timeout (default {TIMEOUT_SECONDS}s).",
    )


def configure_runner(
    *,
    runner: str | Path | None,
    runner_args: list[str] | None = None,
    model_id: str | None = None,
    timeout_seconds: float | None = None,
) -> RunnerConfig:
    resolved_runner = Path(runner).expanduser().resolve() if runner else None
    if resolved_runner is not None and not resolved_runner.is_file():
        raise ValueError(f"model runner does not exist or is not a file: {resolved_runner}")
    timeout = float(timeout_seconds if timeout_seconds is not None else TIMEOUT_SECONDS)
    if timeout <= 0:
        raise ValueError("model runner timeout must be positive")
    _CONFIG.runner = resolved_runner
    _CONFIG.runner_args = list(runner_args or [])
    _CONFIG.model_id = str(model_id or "").strip()
    _CONFIG.timeout_seconds = timeout
    return RunnerConfig(
        runner=_CONFIG.runner,
        runner_args=list(_CONFIG.runner_args),
        model_id=_CONFIG.model_id,
        timeout_seconds=_CONFIG.timeout_seconds,
    )


def configure_from_args(args: argparse.Namespace, *, required: bool) -> RunnerConfig:
    runner = (
        getattr(args, "model_runner", None)
        or os.environ.get("GL2VK_MODEL_RUNNER")
        or _CONFIG.runner
    )
    runner_args = list(getattr(args, "model_runner_arg", None) or [])
    if not runner_args:
        raw_args = os.environ.get("GL2VK_MODEL_RUNNER_ARGS_JSON", "").strip()
        if raw_args:
            parsed = json.loads(raw_args)
            if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
                raise ValueError("GL2VK_MODEL_RUNNER_ARGS_JSON must be a JSON list of strings")
            runner_args = parsed
        elif _CONFIG.runner_args:
            runner_args = list(_CONFIG.runner_args)
    model_id = (
        getattr(args, "model_id", None)
        or os.environ.get("GL2VK_MODEL_ID", "")
        or _CONFIG.model_id
    )
    timeout = getattr(args, "model_timeout_seconds", None)
    if timeout is None:
        raw_timeout = os.environ.get("GL2VK_MODEL_TIMEOUT_SECONDS", "").strip()
        timeout = float(raw_timeout) if raw_timeout else float(_CONFIG.timeout_seconds)
    config = configure_runner(
        runner=runner,
        runner_args=runner_args,
        model_id=model_id,
        timeout_seconds=timeout,
    )
    if required and config.runner is None:
        raise ValueError(
            "model execution requires --model-runner (or GL2VK_MODEL_RUNNER); "
            "use --dry-run to prepare prompts without a model call"
        )
    return config


class ExternalModelClient:
    def __init__(
        self,
        model_key: str,
        *,
        stats_path: Path | None = None,
        stats_lock: threading.Lock | None = None,
        usage_hook: Callable[[dict[str, Any]], Any] | None = None,
        retry: int = 1,
        timeout: float | None = None,
    ) -> None:
        if model_key not in MODEL_DISPLAY_NAMES:
            raise ValueError(
                f"unknown model key {model_key!r}; available: {', '.join(sorted(MODEL_DISPLAY_NAMES))}"
            )
        if _CONFIG.runner is None:
            raise RuntimeError("external model runner is not configured")
        self.model_key = model_key
        self.stats_path = Path(stats_path) if stats_path else None
        self.stats_lock = stats_lock or threading.Lock()
        self.usage_hook = usage_hook
        self.retry = max(1, int(retry))
        self.timeout = float(timeout if timeout is not None else _CONFIG.timeout_seconds)

    def _command(self) -> list[str]:
        assert _CONFIG.runner is not None
        prefix = [sys.executable, str(_CONFIG.runner)] if _CONFIG.runner.suffix.lower() == ".py" else [str(_CONFIG.runner)]
        return prefix + list(_CONFIG.runner_args)

    def _invoke(self, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        completed = subprocess.run(
            self._command(),
            input=json.dumps(request, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=self.timeout,
            check=False,
        )
        if completed.returncode != 0:
            error = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"model runner exited with {completed.returncode}: {error}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("model runner stdout must be one JSON object") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("response"), str):
            raise RuntimeError("model runner response must contain a string field named 'response'")
        usage = payload.get("usage")
        return payload["response"], usage if isinstance(usage, dict) else {}

    def generate(self, prompt: str, description: str = "", meta: Optional[Dict[str, Any]] = None) -> str:
        request = {
            "prompt": prompt,
            "model_key": self.model_key,
            "model_id": _CONFIG.model_id,
            "description": description,
            "meta": dict(meta or {}),
        }
        started = time.time()
        last_error: Exception | None = None
        response = ""
        usage: dict[str, Any] = {}
        for attempt in range(1, self.retry + 1):
            try:
                response, usage = self._invoke(request)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < self.retry:
                    time.sleep(min(16, 2 ** (attempt - 1)))
        record: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "model_key": self.model_key,
            "model_display_name": MODEL_DISPLAY_NAMES[self.model_key],
            "model_id": _CONFIG.model_id,
            "description": description,
            "elapsed_seconds": round(time.time() - started, 2),
            "success": last_error is None,
            "prompt_chars": len(prompt),
            "response_chars": len(response),
            **dict(meta or {}),
        }
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if key in usage:
                try:
                    record[key] = int(usage[key])
                except (TypeError, ValueError):
                    pass
        if last_error is not None:
            record["error"] = str(last_error)
        if self.stats_path:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            with self.stats_lock, self.stats_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.usage_hook:
            try:
                self.usage_hook(record)
            except Exception:
                pass
        if last_error is not None:
            raise RuntimeError(f"external model runner failed: {last_error}") from last_error
        return response


def create_client(
    model_key: str,
    *,
    stats_path: Path | None = None,
    stats_lock: threading.Lock | None = None,
    usage_hook: Callable[[dict[str, Any]], Any] | None = None,
    retry: int = 1,
    timeout: float | None = None,
    **_: Any,
) -> ExternalModelClient:
    return ExternalModelClient(
        model_key,
        stats_path=stats_path,
        stats_lock=stats_lock,
        usage_hook=usage_hook,
        retry=retry,
        timeout=timeout,
    )


__all__ = [
    "MODELS",
    "MODEL_DISPLAY_NAMES",
    "TIMEOUT_SECONDS",
    "ExternalModelClient",
    "add_runner_arguments",
    "configure_from_args",
    "configure_runner",
    "create_client",
]
