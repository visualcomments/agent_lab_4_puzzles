"""Lightweight persistence utilities for AgentLaboratory.

Why this exists
--------------
Some LLM backends (especially local / wrapper providers) may crash the Python
process under high memory pressure. When that happens, in-memory conversation
history is lost and the agent restarts without context.

This module provides small, dependency-free helpers to persist agent state and
history to disk in JSON + JSONL, so a restarted process can restore context.

Design goals
------------
- Atomic writes (write temp then replace)
- Append-only history log (JSONL)
- Defensive parsing (corrupt/partial files won't crash the run)
- Text compaction helpers to keep prompts bounded
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically to avoid partial/corrupt writes on crash."""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def safe_read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")


def truncate_middle(text: str, max_chars: int) -> str:
    """Keep head+tail so errors/tails remain visible."""
    if text is None:
        return ""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = "\n...[truncated]...\n"
    if max_chars <= len(marker) + 10:
        return text[:max_chars]
    head = (max_chars - len(marker)) // 2
    tail = max_chars - len(marker) - head
    return text[:head] + marker + text[-tail:]


def compact_lines(lines: Iterable[str], max_lines: int, max_line_chars: int = 400) -> str:
    """Compact a list of lines for prompt use."""
    if max_lines <= 0:
        return ""
    lines = list(lines)
    n = len(lines)
    if n <= max_lines:
        kept = lines
    else:
        head = max_lines // 2
        tail = max_lines - head
        kept = lines[:head] + [f"... [omitted {n - max_lines} lines] ..."] + lines[-tail:]
    out = []
    for ln in kept:
        ln = ln if isinstance(ln, str) else str(ln)
        out.append(truncate_middle(ln, max_line_chars))
    return "\n".join(out)


@dataclass
class JsonStateStore:
    """Tiny helper for JSON state + JSONL event log."""

    state_path: Path
    log_path: Optional[Path] = None

    def load(self) -> Dict[str, Any]:
        return safe_read_json(self.state_path, default={})

    def save(self, state: Dict[str, Any]) -> None:
        atomic_write_json(self.state_path, state)

    def log(self, event: Dict[str, Any]) -> None:
        if self.log_path is None:
            return
        append_jsonl(self.log_path, event)
