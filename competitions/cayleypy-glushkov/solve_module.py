"""cayleypy-glushkov baseline solver

Kaggle competition: https://www.kaggle.com/competitions/cayleypy-glushkov

Allowed moves:
- L : cyclic left shift by 1 position
- X : swap positions 0 and 1

Goal: sort the permutation into ascending order.

We implement a simple constructive selection-sort using an adjacent-swap primitive.
Adjacent swap of positions i and i+1 can be realized by conjugation with L:
  swap(i,i+1) = L^i  X  L^{n-i}

This is polynomial-time but not score-optimal.

The module exposes solve(vec)->(moves,sorted_array) and prints JSON in script mode.
"""

from __future__ import annotations

import json
import sys
from typing import List, Sequence, Tuple

Move = str


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


def _do_L(a: List[int], moves: List[Move], times: int) -> None:
    # times may be larger than n, but we keep it simple
    _ = 0
    while _ < times:
        _apply_L(a)
        moves.append("L")
        _ += 1


def _adjacent_swap(a: List[int], moves: List[Move], i: int) -> None:
    """Swap a[i] and a[i+1] using only L and X."""
    n = len(a)
    if n <= 1 or i < 0 or i >= n - 1:
        return

    # L^i
    _do_L(a, moves, i)

    # X
    _apply_X(a)
    moves.append("X")

    # L^{n-i} (inverse of L^i)
    _do_L(a, moves, n - i)


def solve(vec: Sequence[int]) -> Tuple[List[Move], List[int]]:
    a = list(vec)
    n = len(a)
    moves: List[Move] = []

    # Selection sort via repeated adjacent swaps
    j = 0
    while j < n:
        # find index of minimum element in a[j:]
        min_idx = j
        k = j + 1
        while k < n:
            if a[k] < a[min_idx]:
                min_idx = k
            k += 1

        # move it left to position j
        while min_idx > j:
            _adjacent_swap(a, moves, min_idx - 1)
            min_idx -= 1

        j += 1

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
