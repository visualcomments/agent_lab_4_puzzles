from typing import List, Tuple
from .movesets import Move, Moveset

def bubble_baseline(p: List[int], moveset: Moveset, budget: int) -> List[Move]:
    """Return a sequence of adjacent swaps that sorts p (if budget allows).
    Classic bubble sort; stops early if budget exhausted.
    """
    n = len(p)
    moves: List[Move] = []
    # We'll simulate on a local copy to count, but return moves, not modify caller p
    arr = p[:]
    for pass_ in range(n):
        swapped = False
        for i in range(n-1):
            if arr[i] > arr[i+1]:
                moves.append(("adj_swap", (i,)))
                # apply to local copy to keep correctness of comparisons
                arr[i], arr[i+1] = arr[i+1], arr[i]
                swapped = True
                if len(moves) >= budget:
                    return moves
        if not swapped:
            break
    return moves
