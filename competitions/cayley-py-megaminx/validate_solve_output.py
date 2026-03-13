#!/usr/bin/env python3
"""Validator for CayleyPy Megaminx solver output.

Contract (same as other competitions in this repo):
- run solver as:  python solve_module.py "[ ... ]"
- solver prints JSON: {"moves": ..., "sorted_array": ...}

Validation performed:
- solver runs and outputs a JSON object with keys "moves" and "sorted_array"
- moves is either:
    * list[str] of generator names
    * a '.'-separated string of generator names
    * the string 'UNSOLVED' (accepted for template/baseline fallback)
- applying the generators to the input must reach the puzzle's central_state
  (loaded from data/puzzle_info.json)
- sorted_array must equal the final state after applying the moves

Permutation application convention:
- generator perm is a list p where: new[i] = old[p[i]]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _env_float(name: str, default: float) -> float:
    try:
        return float((os.environ.get(name, str(default)) or str(default)).strip())
    except Exception:
        return default


def _load_puzzle_info() -> Dict[str, Any]:
    here = Path(__file__).resolve().parent
    p = here / "data" / "puzzle_info.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing puzzle_info.json at {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _apply_perm(state: List[int], perm: List[int]) -> List[int]:
    return [state[j] for j in perm]


def _parse_moves(moves: Any) -> List[str] | None:
    if isinstance(moves, str):
        s = moves.strip()
        if s.upper() == "UNSOLVED":
            return None
        if not s:
            return []
        return s.split(".")
    if isinstance(moves, list) and all(isinstance(m, str) for m in moves):
        return moves
    raise TypeError("moves must be list[str], a dot-separated string, or 'UNSOLVED'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", required=True)
    ap.add_argument("--vector", required=True)
    args = ap.parse_args()

    solver = Path(args.solver)
    vec = json.loads(args.vector)

    solver_timeout_s = _env_float("AGENTLAB_SOLVER_TIMEOUT_S", 15.0)
    try:
        completed = subprocess.run(
            [sys.executable, str(solver), json.dumps(vec)],
            capture_output=True,
            text=True,
            timeout=solver_timeout_s if solver_timeout_s > 0 else None,
        )
    except subprocess.TimeoutExpired:
        print(f"[!] solver timed out after {solver_timeout_s:.1f}s", file=sys.stderr)
        raise SystemExit(124)

    if completed.returncode != 0:
        print(f"[!] solver process failed with exit code {completed.returncode}", file=sys.stderr)
        if completed.stdout:
            print(completed.stdout, file=sys.stderr)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        raise SystemExit(completed.returncode or 1)

    out = completed.stdout
    try:
        data = json.loads(out)
    except Exception:
        print("[!] solver output is not valid JSON", file=sys.stderr)
        print(out, file=sys.stderr)
        raise SystemExit(1)

    if not isinstance(data, dict):
        print("[!] solver output must be a JSON object", file=sys.stderr)
        raise SystemExit(1)

    if "moves" not in data or "sorted_array" not in data:
        print("[!] JSON must contain keys: moves, sorted_array", file=sys.stderr)
        raise SystemExit(1)

    sorted_array = data["sorted_array"]
    if not isinstance(sorted_array, list) or not all(isinstance(x, int) for x in sorted_array):
        print("[!] sorted_array must be a list[int]", file=sys.stderr)
        raise SystemExit(1)

    puzzle = _load_puzzle_info()
    central_state = list(puzzle["central_state"])
    generators: Dict[str, List[int]] = {str(k): list(v) for k, v in dict(puzzle["generators"]).items()}

    moves_list = _parse_moves(data["moves"])
    if moves_list is None:
        print("[validate] moves = UNSOLVED (accepted template baseline).")
        raise SystemExit(0)

    state = list(vec)
    for m in moves_list:
        if m not in generators:
            print(f"[!] invalid move token: {m}", file=sys.stderr)
            raise SystemExit(1)
        state = _apply_perm(state, generators[m])

    if state != central_state:
        print("[!] applying moves does not reach central_state", file=sys.stderr)
        raise SystemExit(1)

    if sorted_array != state:
        print("[!] sorted_array must equal the state after applying moves", file=sys.stderr)
        raise SystemExit(1)

    print(f"[validate] OK moves={len(moves_list)}")


if __name__ == "__main__":
    main()
