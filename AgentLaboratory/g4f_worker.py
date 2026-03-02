#!/usr/bin/env python3
"""Isolated g4f worker process.

This module is executed as a standalone Python process to run exactly one g4f request.
It exists to prevent RAM growth across long-running sessions when some providers keep
large internal caches / session objects.

Protocol:
  argv[1] = req.json
  argv[2] = resp.json

req.json keys:
  - model: str
  - messages: list[dict]
  - timeout: float
  - provider_name: str (optional)
  - api_key: str (optional)
  - max_tokens: int|null (optional)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


def _g4f_to_text(resp: Any) -> str:
    if isinstance(resp, str):
        return resp
    if resp is not None and hasattr(resp, "__iter__"):
        # Some g4f providers stream token-like chunks.
        parts: list[str] = []
        try:
            for ch in resp:
                if isinstance(ch, str):
                    parts.append(ch)
        except Exception:
            pass
        return "".join(parts)
    return ""


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: g4f_worker.py <req.json> <resp.json>", file=sys.stderr)
        return 2

    req_path, resp_path = sys.argv[1], sys.argv[2]
    try:
        with open(req_path, "r", encoding="utf-8") as f:
            req = json.load(f)
    except Exception as e:
        with open(resp_path, "w", encoding="utf-8") as f:
            json.dump({"ok": False, "error": f"bad request json: {e}"}, f)
        return 1

    # Import g4f. Prefer installed package, but fall back to the vendored checkout in ../gpt4free.
    try:
        import g4f  # type: ignore
    except Exception:
        try:
            from pathlib import Path

            root = Path(__file__).resolve().parents[1]
            vendor = root / "gpt4free"
            if vendor.exists():
                sys.path.insert(0, str(vendor))
            import g4f  # type: ignore
        except Exception as e:
            with open(resp_path, "w", encoding="utf-8") as f:
                json.dump({"ok": False, "error": f"could not import g4f: {e}"}, f)
            return 1

    model = str(req.get("model") or "").strip()
    messages = req.get("messages") or []
    timeout = float(req.get("timeout") or 20.0)
    provider_name = str(req.get("provider_name") or "").strip()
    api_key = str(req.get("api_key") or "").strip()
    max_tokens = req.get("max_tokens")

    kwargs: dict[str, Any] = {}

    # Only pass supported kwargs for the installed g4f.
    try:
        import inspect

        sig = inspect.signature(g4f.ChatCompletion.create)  # type: ignore
        if "api_key" in sig.parameters and api_key:
            kwargs["api_key"] = api_key
        if "provider" in sig.parameters and provider_name:
            try:
                kwargs["provider"] = getattr(g4f.Provider, provider_name)  # type: ignore
            except Exception:
                pass
        if "max_tokens" in sig.parameters and isinstance(max_tokens, int) and max_tokens > 0:
            kwargs["max_tokens"] = max_tokens
    except Exception:
        pass

    try:
        resp = g4f.ChatCompletion.create(
            model=model,
            messages=messages,
            timeout=int(max(1, timeout)),
            **kwargs,
        )
        answer = _g4f_to_text(resp).strip()
        with open(resp_path, "w", encoding="utf-8") as f:
            json.dump({"ok": True, "answer": answer}, f, ensure_ascii=False)
        return 0
    except Exception as e:
        with open(resp_path, "w", encoding="utf-8") as f:
            json.dump({"ok": False, "error": str(e)}, f, ensure_ascii=False)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
