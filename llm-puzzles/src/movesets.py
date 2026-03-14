from typing import List, Tuple
from copy import deepcopy

Move = Tuple[str, Tuple[int, ...]]  # (op_name, params)

class Moveset:
    def __init__(self, n: int):
        self.n = n

    # --- bubble-like adjacent swap ---
    def move_adjacent_swap(self, p: List[int], i: int) -> None:
        assert 0 <= i < self.n - 1
        p[i], p[i+1] = p[i+1], p[i]

    # --- cyclic shift of a window [i, i+r] by one to the right ---
    def move_cyclic_shift_right(self, p: List[int], i: int, r: int) -> None:
        assert 0 <= i <= self.n - (r + 1)
        j = i + r
        last = p[j]
        for k in range(j, i, -1):
            p[k] = p[k-1]
        p[i] = last

    # apply a Move to a permutation in-place
    def apply_inplace(self, p: List[int], m: Move) -> None:
        op, params = m
        if op == "adj_swap":
            (i,) = params
            self.move_adjacent_swap(p, i)
        elif op == "cyc_right":
            i, r = params
            self.move_cyclic_shift_right(p, i, r)
        else:
            raise ValueError(f"Unknown move {op}")

    # enumerate legal moves for current p (simple bounded set for search)
    def legal_moves(self, p: List[int], r: int) -> List[Move]:
        L: List[Move] = []
        for i in range(self.n - 1):
            L.append(("adj_swap", (i,)))
        for i in range(self.n - (r + 1) ):
            L.append(("cyc_right", (i, r)))
        return L

def make_moveset(name: str, n: int, **params) -> Moveset:
    if name == "cyclic_coxeter":
        return Moveset(n)
    elif name == "bubble_only":
        return Moveset(n)
    else:
        raise ValueError(f"Unknown moveset {name}")
