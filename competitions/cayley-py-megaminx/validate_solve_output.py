#!/usr/bin/env python3
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
    return json.loads(p.read_text(encoding="utf-8"))


def _apply_perm(state: List[int], perm: List[int]) -> List[int]:
    return [state[i] for i in perm]


def _parse_moves(moves: Any) -> List[str] | None:
    if isinstance(moves, str):
        s = moves.strip()
        if s.upper() == "UNSOLVED":
            return None
        if not s:
            return []
        return [tok for tok in s.split(".") if tok]
    if isinstance(moves, list) and all(isinstance(m, str) for m in moves):
        return moves
    raise TypeError("moves must be list[str], a dot-separated string, or 'UNSOLVED'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", required=True)
    ap.add_argument("--vector", required=True)
    args = ap.parse_args()

    solver = Path(args.solver)
    if not solver.exists():
        print(f"[!] solver not found: {solver}", file=sys.stderr)
        raise SystemExit(2)

    vec = json.loads(args.vector)
    if not isinstance(vec, list) or not all(isinstance(x, int) for x in vec):
        print("[!] --vector must be a JSON list[int]", file=sys.stderr)
        raise SystemExit(2)

    try:
        out = subprocess.check_output([sys.executable, str(solver), json.dumps(vec)], text=True)
    except subprocess.CalledProcessError as e:
        print("[!] solver crashed", file=sys.stderr)
        print(e.output, file=sys.stderr)
        raise SystemExit(1)

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
    generators: Dict[str, List[int]] = puzzle["generators"]

    if len(vec) != len(central_state):
        print("[!] vector length does not match puzzle central_state length", file=sys.stderr)
        raise SystemExit(1)

    moves_list = _parse_moves(data["moves"])
    if moves_list is None:
        print("[validate] moves = UNSOLVED (accepted template baseline).")
        raise SystemExit(0)

    state = list(vec)
    for step, m in enumerate(moves_list, start=1):
        if m not in generators:
            print(f"[!] invalid move token at step {step}: {m}", file=sys.stderr)
            raise SystemExit(1)
        perm = generators[m]
        if len(perm) != len(state):
            print(f"[!] generator length mismatch for move {m}", file=sys.stderr)
            raise SystemExit(1)
        state = _apply_perm(state, perm)

    if state != central_state:
        print("[!] applying moves does not reach central_state", file=sys.stderr)
        raise SystemExit(1)

    if sorted_array != state:
        print("[!] sorted_array must equal the state after applying moves", file=sys.stderr)
        raise SystemExit(1)

    print(f"[validate] OK moves={len(moves_list)} len={len(state)}")


if __name__ == "__main__":
    main()
