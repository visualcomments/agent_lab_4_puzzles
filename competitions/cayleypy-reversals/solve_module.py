"""cayleypy-reversals baseline solver

Kaggle competition: https://www.kaggle.com/competitions/cayleypy-reversals

Allowed moves are reversals of a contiguous segment:
- R[i,j] : reverse the subarray a[i..j] (inclusive).

Goal (typical for these CayleyPy distance puzzles): reach the identity / sorted permutation.

We implement a simple constructive algorithm for permutations of 0..n-1:
for i=0..n-1:
  locate value i, reverse segment [i, idx] to bring it to position i.

This is O(n^2) moves and valid (not optimal).

The module exposes solve(vec)->(moves,sorted_array) and prints JSON in script mode.
"""

from __future__ import annotations

import json
import sys
from typing import List, Sequence, Tuple

Move = str


def _reverse_segment(a: List[int], i: int, j: int) -> None:
    while i < j:
        a[i], a[j] = a[j], a[i]
        i += 1
        j -= 1


def solve(vec: Sequence[int]) -> Tuple[List[Move], List[int]]:
    a = list(vec)
    n = len(a)
    moves: List[Move] = []

    i = 0
    while i < n:
        # find idx of value i
        idx = i
        while idx < n and a[idx] != i:
            idx += 1
        if idx == n:
            # Not a 0..n-1 permutation; fall back to numeric selection sort via reversals.
            # Find minimum in suffix.
            min_idx = i
            k = i + 1
            while k < n:
                if a[k] < a[min_idx]:
                    min_idx = k
                k += 1
            idx = min_idx

        if idx != i:
            _reverse_segment(a, i, idx)
            moves.append(f"R[{i},{idx}]")
        i += 1

    return moves, a


def _main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python solve_module.py '[3,0,2,1]'", file=sys.stderr)
        raise SystemExit(2)

    vec = json.loads(sys.argv[1])
    if not isinstance(vec, list):
        raise SystemExit("Input must be a JSON list")

    moves, sorted_array = solve(vec)
    print(json.dumps({"moves": moves, "sorted_array": sorted_array}))


if __name__ == "__main__":
    _main()
