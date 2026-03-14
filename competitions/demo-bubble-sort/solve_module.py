#!/usr/bin/env python3
"""demo-bubble-sort/solve_module.py

A tiny demo competition to sanity-check the pipeline:
- baseline solver works (no LLM)
- AgentLaboratory can generate a correct solver with either g4f:* or local:* models

Puzzle:
Given an integer list `vector`, return a list of moves that sorts it in non-decreasing order.

Move format:
- Each move is a string "S{j}" meaning: swap positions j and j+1 (0-indexed).

Required solve() signature for pipeline_cli.py:
    solve(vector) -> (moves: List[str], sorted_array: List[int])

The script can also be executed directly:
    python solve_module.py "[3,2,1]"
and prints JSON:
    {"moves": [...], "sorted_array": [...]}
"""

from __future__ import annotations

import json
import sys
from typing import Any, List, Sequence, Tuple


def apply_moves(vec: Sequence[int], moves: Sequence[str]) -> List[int]:
    a = list(vec)
    n = len(a)
    for mv in moves:
        if not isinstance(mv, str) or not mv.startswith("S"):
            raise ValueError(f"Bad move token: {mv!r}")
        j_str = mv[1:]
        if not j_str.isdigit():
            raise ValueError(f"Bad swap index in move: {mv!r}")
        j = int(j_str)
        if j < 0 or j + 1 >= n:
            raise ValueError(f"Swap index out of range in move: {mv!r}")
        a[j], a[j + 1] = a[j + 1], a[j]
    return a


def solve(vector: Sequence[int]) -> Tuple[List[str], List[int]]:
    a = list(vector)
    n = len(a)
    moves: List[str] = []

    # Classic bubble sort (with early-stop optimization)
    for i in range(n):
        swapped = False
        for j in range(0, n - 1 - i):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                moves.append(f"S{j}")
                swapped = True
        if not swapped:
            break

    return moves, a


def _main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("Usage: python solve_module.py \"[3,2,1]\"", file=sys.stderr)
        return 2

    vec = json.loads(argv[1])
    moves, sorted_array = solve(vec)
    out = {"moves": moves, "sorted_array": sorted_array}
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
