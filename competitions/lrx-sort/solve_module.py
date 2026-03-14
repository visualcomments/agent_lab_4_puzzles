#!/usr/bin/env python3
"""
solve_module.py

Constructive polynomial-time sorting using ONLY allowed moves:
- L: left cyclic shift of the whole array by 1
- R: right cyclic shift of the whole array by 1
- X: swap the first two elements

Algorithm:
- Simulate a standard bubble-sort, but whenever we need an adjacent swap at positions (i, i+1),
  we realize it with:
      L repeated i times  -> bring a[i] to front
      X                   -> swap first two
      R repeated i times  -> restore orientation

This is constructive and runs in polynomial time:
- Bubble sort: O(n^2) adjacent swaps (worst-case)
- Each adjacent swap costs O(n) primitive moves (L/R/X)
=> O(n^3) primitive moves in the worst-case, polynomial.

CLI:
    python solve_module.py "[3,1,2,5,4]"
Outputs JSON:
    {"moves": [...], "sorted_array": [...]}

No external dependencies.
"""
from __future__ import annotations
import json
import sys
from typing import List, Tuple, Any


def _apply_L(a: List[Any]) -> None:
    """Left cyclic shift by 1 (in-place) WITHOUT slicing."""
    n = len(a)
    if n <= 1:
        return
    first = a[0]
    i = 0
    while i < n - 1:
        a[i] = a[i + 1]
        i += 1
    a[n - 1] = first


def _apply_R(a: List[Any]) -> None:
    """Right cyclic shift by 1 (in-place) WITHOUT slicing."""
    n = len(a)
    if n <= 1:
        return
    last = a[n - 1]
    i = n - 1
    while i > 0:
        a[i] = a[i - 1]
        i -= 1
    a[0] = last


def _apply_X(a: List[Any]) -> None:
    """Swap a[0] and a[1] (in-place)."""
    if len(a) >= 2:
        a[0], a[1] = a[1], a[0]


def _rot_left(a: List[Any], k: int, moves: List[str]) -> None:
    """Apply L k times; append moves immediately."""
    n = len(a)
    if n <= 1:
        return
    k = k % n
    i = 0
    while i < k:
        _apply_L(a)
        moves.append("L")
        i += 1


def _rot_right(a: List[Any], k: int, moves: List[str]) -> None:
    """Apply R k times; append moves immediately."""
    n = len(a)
    if n <= 1:
        return
    k = k % n
    i = 0
    while i < k:
        _apply_R(a)
        moves.append("R")
        i += 1


def _adjacent_swap(a: List[Any], i: int, moves: List[str]) -> None:
    """Swap positions (i, i+1) using only L/R/X; append moves immediately."""
    n = len(a)
    if n < 2:
        return
    if not (0 <= i < n - 1):
        raise ValueError(f"adjacent_swap index out of range: i={i}, n={n}")

    # Bring i to front, swap, restore
    _rot_left(a, i, moves)
    _apply_X(a)
    moves.append("X")
    _rot_right(a, i, moves)


def _is_sorted(a: List[Any]) -> bool:
    i = 0
    while i < len(a) - 1:
        if a[i] > a[i + 1]:
            return False
        i += 1
    return True


def solve(vec: List[Any]) -> Tuple[List[str], List[Any]]:
    """
    Return (moves, sorted_array).
    The returned moves, when applied sequentially to a copy of vec, produce sorted_array.
    """
    a = list(vec)  # allowed copy to simulate sorting
    moves: List[str] = []
    n = len(a)
    if n <= 1:
        return moves, a

    # Bubble-sort via constructive adjacent swaps
    pass_idx = 0
    while pass_idx < n:
        swapped = False
        i = 0
        while i < n - 1:
            if a[i] > a[i + 1]:
                _adjacent_swap(a, i, moves)
                swapped = True
            i += 1
        if not swapped:
            break
        pass_idx += 1

    return moves, a


def _main() -> None:
    if len(sys.argv) > 1:
        try:
            input_vector = json.loads(sys.argv[1])
        except json.JSONDecodeError:
            input_vector = [3, 1, 2]
    else:
        input_vector = [3, 1, 2]

    if not isinstance(input_vector, list):
        raise SystemExit("Input must be a JSON array, e.g. \"[3,1,2]\"")

    moves, sorted_array = solve(input_vector)
    print(json.dumps({"moves": moves, "sorted_array": sorted_array}, ensure_ascii=False))


if __name__ == "__main__":
    _main()
