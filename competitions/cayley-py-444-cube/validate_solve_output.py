#!/usr/bin/env python3
"""Validator for CayleyPy 444 Cube solver output.

This repo uses a *generic* solver contract across competitions:
- run solver as:  python solve_module.py "[ ... ]"
- solver prints JSON: {"moves": ..., "sorted_array": ...}

For the 444-cube competitions, Kaggle submissions expect a "path" string
of generator names separated by '.', but inside this repo we validate the
more structured JSON output.

Validation performed:
- solver runs and outputs valid JSON object
- keys "moves" and "sorted_array" exist
- moves is either:
    * list[str] of generator names
    * a '.'-separated string of generator names
    * the string 'UNSOLVED' (accepted for template/baseline)
- applying the generators to the input must reach the puzzle's central_state
  (loaded from data/puzzle_info.json)
- sorted_array must equal the final state.

Permutation application convention:
- generator perm is a list p where: new[i] = old[p[i]]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_puzzle_info() -> Dict[str, Any]:
    here = Path(__file__).resolve().parent
    p = here / "data" / "puzzle_info.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing puzzle_info.json at {p}")
    return json.loads(p.read_text())


def _apply_perm(state: List[int], perm: List[int]) -> List[int]:
    # new[i] = old[perm[i]]
    return [state[j] for j in perm]


def _parse_moves(moves: Any) -> List[str] | None:
    if isinstance(moves, str):
        s = moves.strip()
        if s.upper() == "UNSOLVED":
            return None
        if not s:
            return []
        # Allow either a single move "f1" or a dot-separated path "-d3.-r3"
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

    out = subprocess.check_output([sys.executable, str(solver), json.dumps(vec)], text=True)
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
    central_state = puzzle["central_state"]
    generators: Dict[str, List[int]] = puzzle["generators"]

    moves_list = _parse_moves(data["moves"])
    if moves_list is None:
        # Template/baseline path: accept UNSOLVED.
        print("[validate] moves = UNSOLVED (accepted template baseline).")
        raise SystemExit(0)

    # Apply moves.
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
