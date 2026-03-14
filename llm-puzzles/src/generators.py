from typing import List, Dict, Tuple
import random

def identity(n: int) -> List[int]:
    return list(range(n))

def random_perm(n: int, rng: random.Random) -> List[int]:
    p = list(range(n))
    rng.shuffle(p)
    return p

def structured_perms(n: int, rng: random.Random) -> List[List[int]]:
    perms = []
    # reversed
    perms.append(list(reversed(range(n))))
    # almost sorted: swap a few adjacent pairs
    p = list(range(n))
    for i in range(0, min(n-2, 6), 2):
        p[i], p[i+1] = p[i+1], p[i]
    perms.append(p)
    # blocky: two blocks reversed
    k = n//2
    p2 = list(range(n))
    p2[:k] = reversed(p2[:k])
    p2[k:] = reversed(p2[k:])
    perms.append(p2)
    return perms

def make_split(n: int, size: int, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    L: List[List[int]] = []
    # include structured cases
    L.extend(structured_perms(n, rng))
    # fill with random perms
    while len(L) < size:
        L.append(random_perm(n, rng))
    return L[:size]
