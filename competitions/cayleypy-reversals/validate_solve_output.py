#!/usr/bin/env python3
"""validate_solve_output.py (cayleypy-reversals)

Validates a solver for the CayleyPy Reversals-style problems.
Move format:
- R[i,j] where 0 <= i < j < n, reverses a[i..j] inclusive.

Accepts moves='UNSOLVED' as valid (penalized).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

RE_MOVE = re.compile(r"^R\[(\d+),(\d+)\]$")


def _reverse_segment(a: List[int], i: int, j: int) -> None:
    while i < j:
        a[i], a[j] = a[j], a[i]
        i += 1
        j -= 1


def _parse_move(m: str) -> Tuple[int, int]:
    mm = RE_MOVE.match(m)
    if not mm:
        raise ValueError(f"Bad move token: {m!r}")
    return int(mm.group(1)), int(mm.group(2))


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

    a = list(vec)
    n = len(a)
    for m in moves:
        i, j = _parse_move(m)
        if i < 0 or j < 0 or i >= n or j >= n or i >= j:
            print(f"[!] illegal indices in move {m!r} for n={n}", file=sys.stderr)
            raise SystemExit(1)
        _reverse_segment(a, i, j)

    expected = sorted(vec)
    if a != expected:
        print("[!] final array is not sorted", file=sys.stderr)
        raise SystemExit(1)

    if sorted_array != a:
        print("[!] sorted_array mismatch", file=sys.stderr)
        raise SystemExit(1)

    print(f"[validate] OK n={n} moves={len(moves)}")


if __name__ == "__main__":
    main()
