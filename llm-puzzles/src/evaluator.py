from typing import Callable, Dict, List, Tuple
from copy import deepcopy
from .movesets import Moveset, make_moveset, Move
from .metrics import kendall_tau_distance, report_metrics
from .generators import make_split
from .tasks import Task

def apply_moves(p: List[int], moveset: Moveset, seq: List[Move], budget: int) -> List[int]:
    q = p[:]
    steps = 0
    for m in seq:
        if steps >= budget:
            break
        moveset.apply_inplace(q, m)
        steps += 1
    return q

def evaluate_candidate(task: Task, candidate_fn, *, r: int) -> Dict[str, Dict[str, float]]:
    moveset = make_moveset(task.moveset_name, task.n, **task.moveset_params)
    results: Dict[str, Dict[str, float]] = {}
    for split_name, size in task.splits.items():
        data = make_split(task.n, size, seed=hash((task.seed, split_name)) & 0xffffffff)
        before_after = []
        success_flags = []
        for p in data:
            kb = kendall_tau_distance(p)
            # candidate proposes sequence of moves
            seq = candidate_fn(p[:], moveset, task.budget_steps)
            q = apply_moves(p, moveset, seq, task.budget_steps)
            ka = kendall_tau_distance(q)
            before_after.append((kb, ka))
            success_flags.append(ka == 0)
        results[split_name] = report_metrics(before_after, success_flags)
    return results
