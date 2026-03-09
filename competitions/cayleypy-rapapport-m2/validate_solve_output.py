#!/usr/bin/env python3
"""validate_solve_output.py (cayleypy-rapapport-m2)

Validates that a solver script:
- Accepts a JSON list via argv[1]
- Prints JSON with keys: moves, sorted_array
- Uses only moves in {S,I,K}
- Actually sorts the vector under the competition move semantics

Competition semantics:
- S swaps positions 0 and 1
- I swaps (0,1), (2,3), (4,5), ...
- K swaps (1,2), (3,4), (5,6), ...
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from typing import List
ALLOWED = {"S", "I", "K"}

def _apply_S(a: List[int]) -> None:
    if len(a) >= 2:
        a[0], a[1] = a[1], a[0]

def _apply_I(a: List[int]) -> None:
    i = 0
    while i + 1 < len(a):
        a[i], a[i + 1] = a[i + 1], a[i]
        i += 2

def _apply_K(a: List[int]) -> None:
    i = 1
    while i + 1 < len(a):
        a[i], a[i + 1] = a[i + 1], a[i]
        i += 2

def _apply_move(a: List[int], m: str) -> None:
    if m == "S":
        _apply_S(a)
    elif m == "I":
        _apply_I(a)
    elif m == "K":
        _apply_K(a)
    else:
        raise ValueError(f"Illegal move {m!r}")

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
        raise SystemExit(1)
    bad = [m for m in moves if m not in ALLOWED]
    if bad:
        raise SystemExit(1)
    a = list(vec)
    for m in moves:
        _apply_move(a, m)
    if a != sorted(vec):
        print(f"expected: {sorted(vec)}")
        print(f"got     : {a}")
        raise SystemExit(1)
    if sorted_array != a:
        raise SystemExit(1)
    print(f"[validate] OK. n={len(vec)} moves={len(moves)}")

if __name__ == "__main__":
    main()
