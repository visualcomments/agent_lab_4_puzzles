"""Running-summary utilities for long agent sessions.

Why
---
Even with bounded prompt history, long-lived agent runs can lose important
context when older turns are dropped. A common strategy is to maintain a
"running summary" that periodically compresses older dialogue into a compact
representation that can be re-injected into future prompts.

This module provides a small helper that updates a running summary using the
same LLM backend as the rest of AgentLaboratory (via `query_model`).

Notes
-----
- We do *not* attempt to enforce strict token limits (backend-dependent). We
  instead (1) keep the input chunk bounded, (2) instruct the model to stay
  short, and (3) post-truncate the result defensively.
- If the LLM call fails, we fall back to a lightweight heuristic summary.
"""

from __future__ import annotations

from typing import List

from inference import query_model
from persistence import truncate_middle


SUMMARY_SYSTEM_PROMPT = (
    "You are a summarization engine for an autonomous LLM agent. "
    "Your job is to maintain a running summary of the *older* conversation so "
    "future steps can continue consistently even when detailed history is dropped. "
    "Preserve decisions, key facts, constraints, data, file names/paths, "
    "commands executed, outputs/results, and open TODOs. "
    "Write in the same language as the input."
)


def _heuristic_fallback(existing_summary: str, chunk_text: str, max_chars: int) -> str:
    """Fallback summary if LLM summarization fails."""
    lines = [ln.strip() for ln in chunk_text.splitlines() if ln.strip()]
    head = "\n".join(lines[:10])
    out = (existing_summary or "").strip()
    if out:
        out += "\n\n"
    out += "[Heuristic fold]\n" + head
    return truncate_middle(out, max_chars)


def update_running_summary(
    *,
    model: str,
    openai_api_key: str | None,
    existing_summary: str,
    chunk_items: List[str] | str,
    max_input_chars: int = 12000,
    max_output_chars: int = 6000,
    temp: float = 0.0,
) -> str:
    """Update a running summary with a new chunk of historical items."""

    if isinstance(chunk_items, list):
        chunk_text = "\n".join([x for x in chunk_items if x])
    else:
        chunk_text = str(chunk_items or "")

    chunk_text = truncate_middle(chunk_text, max_input_chars)
    existing_summary = (existing_summary or "").strip()

    user_prompt = (
        "Update the running summary by merging the NEW chunk into the EXISTING summary.\n\n"
        "EXISTING SUMMARY (may be empty):\n"
        f"{existing_summary}\n\n"
        "NEW CHUNK (older conversation snippets to compress):\n"
        f"{chunk_text}\n\n"
        "OUTPUT FORMAT (keep it short):\n"
        "- 5–12 bullet points max\n"
        "- keep important concrete details (numbers, file names, constraints)\n"
        "- include an 'Open questions / TODO' bullet list if relevant\n"
        "Hard limit: aim for <= 1200 words."
    )

    try:
        out = query_model(
            model_str=model,
            system_prompt=SUMMARY_SYSTEM_PROMPT,
            prompt=user_prompt,
            temp=temp,
            openai_api_key=openai_api_key,
        )
        out = (out or "").strip()
        if not out:
            return _heuristic_fallback(existing_summary, chunk_text, max_output_chars)
        return truncate_middle(out, max_output_chars)
    except Exception:
        return _heuristic_fallback(existing_summary, chunk_text, max_output_chars)
