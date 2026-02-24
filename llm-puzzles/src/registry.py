from typing import Dict, Callable
from .tasks import Task

# ---- Tasks ----
def task_cyclic_coxeter_n16_r3() -> Task:
    return Task(
        name="cyclic-coxeter-n16-r3",
        n=16,
        moveset_name="cyclic_coxeter",
        moveset_params={"r": 3},
        budget_steps=200,
        splits={"fast": 256, "control": 512},
        seed=42
    )

TASKS: Dict[str, Callable[[], Task]] = {
    "cyclic-coxeter-n16-r3": task_cyclic_coxeter_n16_r3,
}

# ---- Candidates ----
from .baselines import bubble_baseline
from examples.candidates.heuristic1 import solve as heuristic1

CANDIDATES: Dict[str, Callable] = {
    "baseline": bubble_baseline,
    "heuristic1": heuristic1,
}
