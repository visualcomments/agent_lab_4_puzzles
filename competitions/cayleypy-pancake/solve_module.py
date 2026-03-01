"""cayleypy-pancake baseline solver

Kaggle competition: https://www.kaggle.com/competitions/CayleyPy-pancake

Allowed moves are prefix reversals ("pancake flips"):
- Rk : reverse the first k elements, for k = 2..n

Goal: sort the permutation into ascending order.

We implement the classic constructive Pancake Sort (Bill Gates algorithm):
for size=n..2:
  bring the largest remaining element to front with one flip, then flip it into place.

Not optimal, but valid and polynomial.

The module exposes:
    solve(vec) -> (moves, sorted_array)

Script mode:
    python solve_module.py "[3,1,2,0]"
prints JSON {"moves": [...], "sorted_array": [...]}
"""

from __future__ import annotations

import json
import sys
from typing import List, Sequence, Tuple

Move = str


def _flip_prefix(a: List[int], k: int) -> None:
    """Reverse first k elements in-place WITHOUT slicing."""
    i = 0
    j = k - 1
    while i < j:
        a[i], a[j] = a[j], a[i]
        i += 1
        j -= 1


def solve(vec: Sequence[int]) -> Tuple[List[Move], List[int]]:
    a = list(vec)
    n = len(a)

    moves: List[Move] = []

    # Target order is numeric ascending.
    # We implement the standard pancake sort that assumes values are 0..n-1.
    # If values are distinct but not 0..n-1, the algorithm still sorts by numeric value.
    for size in range(n, 1, -1):
        # Find index of the maximum value within a[0:size]
        max_idx = 0
        i = 1
        while i < size:
            if a[i] > a[max_idx]:
                max_idx = i
            i += 1

        # Already in place
        if max_idx == size - 1:
            continue

        # Bring it to front
        if max_idx != 0:
            k1 = max_idx + 1
            _flip_prefix(a, k1)
            moves.append(f"R{k1}")

        # Flip it into its final position
        _flip_prefix(a, size)
        moves.append(f"R{size}")

    return moves, a


def _main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python solve_module.py '[3,1,2,0]'", file=sys.stderr)
        raise SystemExit(2)

    vec = json.loads(sys.argv[1])
    if not isinstance(vec, list):
        raise SystemExit("Input must be a JSON list")

    moves, sorted_array = solve(vec)
    print(json.dumps({"moves": moves, "sorted_array": sorted_array}))


if __name__ == "__main__":
    _main()
