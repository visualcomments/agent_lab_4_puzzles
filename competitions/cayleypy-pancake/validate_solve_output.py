#!/usr/bin/env python3
"""validate_solve_output.py (CayleyPy-pancake)

Validates that a solver script:
- Accepts a JSON list via argv[1]
- Prints JSON with keys: moves, sorted_array
- moves is either:
    - 'UNSOLVED' (accepted; Kaggle penalizes)
    - list[str] where each move is 'Rk' for 2<=k<=n
- Applying the moves sorts the vector

Intended for AgentLaboratory perm_pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List

RE_MOVE = re.compile(r"^R(\d+)$")


def _flip_prefix(a: List[int], k: int) -> None:
    i = 0
    j = k - 1
    while i < j:
        a[i], a[j] = a[j], a[i]
        i += 1
        j -= 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", required=True)
    ap.add_argument("--vector", required=True, help="JSON list")
    args = ap.parse_args()

    solver = Path(args.solver)
    if not solver.exists():
        print(f"[!] solver not found: {solver}", file=sys.stderr)
        raise SystemExit(2)

    vec = json.loads(args.vector)
    if not isinstance(vec, list):
        print("[!] --vector must be a JSON list", file=sys.stderr)
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

    moves = data.get("moves")
    sorted_array = data.get("sorted_array")

    if moves is None or sorted_array is None:
        print("[!] JSON must contain keys: moves, sorted_array", file=sys.stderr)
        raise SystemExit(1)

    # UNSOLVED is allowed in Kaggle format (penalty). Accept it here too.
    if isinstance(moves, str) and moves.strip().upper() == "UNSOLVED":
        print("[validate] moves = UNSOLVED (accepted; Kaggle will penalize).")
        raise SystemExit(0)

    if not isinstance(moves, list) or not all(isinstance(m, str) for m in moves):
        print("[!] moves must be a list[str] (or 'UNSOLVED')", file=sys.stderr)
        raise SystemExit(1)

    a = list(vec)
    n = len(a)

    for m in moves:
        mm = RE_MOVE.match(m)
        if not mm:
            print(f"[!] illegal move token: {m!r} (expected 'Rk')", file=sys.stderr)
            raise SystemExit(1)
        k = int(mm.group(1))
        if k < 2 or k > n:
            print(f"[!] illegal flip length k={k} for n={n}", file=sys.stderr)
            raise SystemExit(1)
        _flip_prefix(a, k)

    expected = sorted(vec)
    if a != expected:
        print("[!] final array is not sorted", file=sys.stderr)
        print(f"expected: {expected}")
        print(f"got     : {a}")
        raise SystemExit(1)

    if sorted_array != a:
        print("[!] sorted_array does not match the simulated final state", file=sys.stderr)
        raise SystemExit(1)

    print(f"[validate] OK. n={n} moves={len(moves)}")


if __name__ == "__main__":
    main()
