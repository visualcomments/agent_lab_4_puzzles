"""
examples/agentlab_sort/solver.py

Row-wise solver adapter for llm-puzzles universal_adapter.

It expects each CSV row to contain a JSON array in one of the columns (e.g. "vector", "a", "state", "initial_state").
You can force a specific column via env var VECTOR_COL.

Returns:
- list[str] of moves (universal_adapter will join using cfg.move_joiner)
"""
from __future__ import annotations
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from .solve_module import solve

CANDIDATE_COLS = ["vector", "a", "arr", "array", "state", "initial_state", "input", "puzzle"]

def _try_parse_json_list(s: str) -> Optional[List[Any]]:
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return v
    except Exception:
        return None
    return None

def extract_vector(row: Dict[str, str]) -> List[Any]:
    # explicit override
    col = os.getenv("VECTOR_COL", "").strip()
    if col and col in row:
        v = _try_parse_json_list(row[col])
        if v is not None:
            return v

    for c in CANDIDATE_COLS:
        if c in row and row[c]:
            v = _try_parse_json_list(row[c])
            if v is not None:
                return v

    # last resort: try any column that looks like a JSON list
    for k, val in row.items():
        if not val:
            continue
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            v = _try_parse_json_list(val)
            if v is not None:
                return v

    raise ValueError(f"Could not find a JSON list vector in row. Columns: {list(row.keys())}")

def solve_row(row: Dict[str, str], cfg) -> List[str]:
    vec = extract_vector(row)
    moves, sorted_array = solve(vec)
    # universal_adapter accepts list[str], or dict with moves key; return moves list
    return moves
