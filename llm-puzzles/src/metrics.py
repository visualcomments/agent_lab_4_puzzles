from typing import List, Tuple, Dict
from math import comb

def inversions_count(p: List[int]) -> int:
    # simple O(n^2) for clarity; n<=64 typical for our splits
    n = len(p)
    inv = 0
    for i in range(n):
        pi = p[i]
        for j in range(i+1, n):
            if pi > p[j]:
                inv += 1
    return inv

def kendall_tau_distance(p: List[int]) -> int:
    # distance to identity permutation (0..n-1) equals number of inversions
    return inversions_count(p)

def success_at_n(sorted_flags: List[bool]) -> float:
    if not sorted_flags:
        return 0.0
    return sum(1 for x in sorted_flags if x) / len(sorted_flags)

def normalized(value: float, max_value: float) -> float:
    if max_value <= 0: 
        return 0.0
    return max(0.0, min(1.0, value / max_value))

def report_metrics(before_after: List[Tuple[int, int]], success_flags: List[bool]) -> Dict[str, float]:
    # before_after: list of (kendall_before, kendall_after)
    n = len(before_after)
    if n == 0:
        return {"Success@N": 0.0, "E[ΔKendallTau]": 0.0, "E[ΔInversions]": 0.0}
    deltas = [ (b - a) for (b, a) in before_after ]
    avg_delta = sum(deltas)/n
    success = success_at_n(success_flags)
    # same as kendall for our case
    return {
        "Success@N": success,
        "E[ΔKendallTau]": avg_delta,
        "E[ΔInversions]": avg_delta,
    }
