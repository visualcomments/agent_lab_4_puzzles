#!/usr/bin/env python3
"""validate_solve_output.py (cayleypy-rapapport-m2)

Validates that a solver script:
- Accepts a JSON list via argv[1]
- Prints JSON with keys: moves, sorted_array
- Uses only moves in {I,S,K}
- Actually sorts the vector (sorted_array must equal the true result)

This validator is intended for AgentLaboratory perm_pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

ALLOWED = {"I", "S", "K"}


def _apply_I(a: List[int]) -> None:
    if len(a) >= 2:
        a[0], a[1] = a[1], a[0]


def _apply_S(a: List[int]) -> None:
    n = len(a)
    i = 0
    while i + 1 < n:
        a[i], a[i + 1] = a[i + 1], a[i]
        i += 2


def _apply_K(a: List[int]) -> None:
    n = len(a)
    i = 1
    while i + 1 < n:
        a[i], a[i + 1] = a[i + 1], a[i]
        i += 2


def _apply_move(a: List[int], m: str) -> None:
    if m == "I":
        _apply_I(a)
    elif m == "S":
        _apply_S(a)
    elif m == "K":
        _apply_K(a)
    else:
        raise ValueError(f"Illegal move {m!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", required=True)
    ap.add_argument("--vector", required=True, help="JSON list, e.g. '[3,0,1,4,2]' ")
    args = ap.parse_args()

    solver = Path(args.solver)
    if not solver.exists():
        print(f"[!] solver not found: {solver}", file=sys.stderr)
        raise SystemExit(2)

    vec = json.loads(args.vector)
    if not isinstance(vec, list):
        print("[!] --vector must be a JSON list", file=sys.stderr)
        raise SystemExit(2)

    # Run solver
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
        print(out, file=sys.stderr)
        raise SystemExit(1)

    moves = data.get("moves")
    sorted_array = data.get("sorted_array")

    if moves is None or sorted_array is None:
        print("[!] JSON must contain keys: moves, sorted_array", file=sys.stderr)
        print(out, file=sys.stderr)
        raise SystemExit(1)

    # Allow UNSOLVED (Kaggle allows it but with penalty). For local validation we accept it.
    if isinstance(moves, str) and moves.strip().upper() == "UNSOLVED":
        print("[validate] moves = UNSOLVED (accepted; Kaggle will penalize).")
        raise SystemExit(0)

    if not isinstance(moves, list) or not all(isinstance(m, str) for m in moves):
        print("[!] moves must be a list[str] (or 'UNSOLVED')", file=sys.stderr)
        print(out, file=sys.stderr)
        raise SystemExit(1)

    bad = [m for m in moves if m not in ALLOWED]
    if bad:
        print(f"[!] illegal moves: {bad[:10]} (allowed: {sorted(ALLOWED)})", file=sys.stderr)
        raise SystemExit(1)

    a = list(vec)
    for m in moves:
        _apply_move(a, m)

    expected = sorted(vec)
    if a != expected:
        print("[!] final array is not sorted", file=sys.stderr)
        print(f"expected: {expected}")
        print(f"got     : {a}")
        raise SystemExit(1)

    if sorted_array != a:
        print("[!] sorted_array does not match the simulated final state", file=sys.stderr)
        print(f"sorted_array: {sorted_array}")
        print(f"final       : {a}")
        raise SystemExit(1)

    print(f"[validate] OK. n={len(vec)} moves={len(moves)}")


if __name__ == "__main__":
    main()
