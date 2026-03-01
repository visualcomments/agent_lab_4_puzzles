#!/usr/bin/env python3
"""validate_solve_output.py (cayleypy-glushkov)

Checks that the solver sorts using only moves L and X.

- L: cyclic left shift by 1
- X: swap positions 0 and 1

Accepts moves='UNSOLVED' as a valid (penalized) output.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

ALLOWED = {"L", "X"}


def _apply_L(a: List[int]) -> None:
    n = len(a)
    if n <= 1:
        return
    first = a[0]
    i = 0
    while i < n - 1:
        a[i] = a[i + 1]
        i += 1
    a[n - 1] = first


def _apply_X(a: List[int]) -> None:
    if len(a) >= 2:
        a[0], a[1] = a[1], a[0]


def _apply_move(a: List[int], m: str) -> None:
    if m == "L":
        _apply_L(a)
    elif m == "X":
        _apply_X(a)
    else:
        raise ValueError(m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", required=True)
    ap.add_argument("--vector", required=True)
    args = ap.parse_args()

    solver = Path(args.solver)
    vec = json.loads(args.vector)

    out = subprocess.check_output([sys.executable, str(solver), json.dumps(vec)], text=True)
    data = json.loads(out)

    moves = data.get("moves")
    sorted_array = data.get("sorted_array")

    if isinstance(moves, str) and moves.strip().upper() == "UNSOLVED":
        print("[validate] moves = UNSOLVED (accepted; Kaggle will penalize).")
        raise SystemExit(0)

    if not isinstance(moves, list) or not all(isinstance(m, str) for m in moves):
        print("[!] moves must be list[str]", file=sys.stderr)
        raise SystemExit(1)

    bad = [m for m in moves if m not in ALLOWED]
    if bad:
        print(f"[!] illegal moves: {bad[:10]}", file=sys.stderr)
        raise SystemExit(1)

    a = list(vec)
    for m in moves:
        _apply_move(a, m)

    expected = sorted(vec)
    if a != expected:
        print("[!] final array is not sorted", file=sys.stderr)
        raise SystemExit(1)

    if sorted_array != a:
        print("[!] sorted_array mismatch", file=sys.stderr)
        raise SystemExit(1)

    print(f"[validate] OK n={len(vec)} moves={len(moves)}")


if __name__ == "__main__":
    main()
