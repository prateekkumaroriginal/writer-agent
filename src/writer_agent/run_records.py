"""JSON run records keyed by durable LangGraph thread ID."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from os import getenv, replace
from pathlib import Path
from typing import Any

DEFAULT_RUN_RECORD_DIR = (
    Path(__file__).resolve().parents[2] / "tests" / "runs"
)
SAFE_THREAD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")


def record_run(
    thread_id: str,
    state: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> Path:
    """Atomically write the latest run state to ``<thread_id>.json``."""
    normalized = thread_id.strip()
    if not SAFE_THREAD_ID.fullmatch(normalized):
        raise ValueError("thread_id is not safe for a run-record filename")

    target_dir = Path(
        output_dir
        or getenv("WRITER_AGENT_RUN_DIR")
        or DEFAULT_RUN_RECORD_DIR
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{normalized}.json"
    temporary = target_dir / f".{normalized}.tmp"
    payload = {**dict(state), "thread_id": normalized}
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    replace(temporary, target)
    return target
