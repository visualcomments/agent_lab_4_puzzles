#!/usr/bin/env python3
"""demo-bubble-sort/validate_solve_output.py

Validator used by pipeline_cli.py:
    python validate_solve_output.py --solver path/to/solve_module.py --vector "[3,2,1]"

Checks:
- solver exits with code 0
- stdout parses as JSON dict with keys "moves" and "sorted_array"
- moves tokens are valid swaps "S{j}"
- applying moves to input yields sorted_array
- sorted_array is non-decreasing and a permutation of input
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from typing import Any, List, Sequence

PYTHON = sys.executable


def apply_moves(vec: Sequence[int], moves: Sequence[str]) -> List[int]:
    a = list(vec)
    n = len(a)
    for mv in moves:
        if not isinstance(mv, str) or not mv.startswith("S"):
            raise AssertionError(f"Bad move token: {mv!r}")
        j_str = mv[1:]
        if not j_str.isdigit():
            raise AssertionError(f"Bad swap index in move: {mv!r}")
        j = int(j_str)
        if j < 0 or j + 1 >= n:
            raise AssertionError(f"Swap index out of range in move: {mv!r}")
        a[j], a[j + 1] = a[j + 1], a[j]
    return a


def is_sorted(a: Sequence[int]) -> bool:
    return all(a[i] <= a[i + 1] for i in range(len(a) - 1))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--solver", required=True)
    p.add_argument("--vector", required=True)
    args = p.parse_args()

    vec = json.loads(args.vector)
    proc = subprocess.run([PYTHON, args.solver, json.dumps(vec)], capture_output=True, text=True)
    if proc.returncode != 0:
        print("Solver failed:", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        return 1

    try:
        out = json.loads(proc.stdout.strip())
    except Exception as e:
        print(f"Solver output is not valid JSON: {e}", file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
        return 1

    if not isinstance(out, dict):
        print("Solver output must be a JSON object (dict).", file=sys.stderr)
        return 1

    if "moves" not in out or "sorted_array" not in out:
        print("Solver output must contain keys: moves, sorted_array", file=sys.stderr)
        return 1

    moves = out["moves"]
    sorted_array = out["sorted_array"]

    if not isinstance(moves, list):
        print("moves must be a list", file=sys.stderr)
        return 1
    if not isinstance(sorted_array, list):
        print("sorted_array must be a list", file=sys.stderr)
        return 1

    applied = apply_moves(vec, moves)
    if applied != sorted_array:
        print("Applying moves does not match sorted_array.", file=sys.stderr)
        print("input:", vec, file=sys.stderr)
        print("applied:", applied, file=sys.stderr)
        print("sorted_array:", sorted_array, file=sys.stderr)
        return 1

    if not is_sorted(sorted_array):
        print("sorted_array is not sorted non-decreasing.", file=sys.stderr)
        print("sorted_array:", sorted_array, file=sys.stderr)
        return 1

    if Counter(sorted_array) != Counter(vec):
        print("sorted_array is not a permutation of input.", file=sys.stderr)
        return 1

    print("[ok] demo-bubble-sort validator passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
