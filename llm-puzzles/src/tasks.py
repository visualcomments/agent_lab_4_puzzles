from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
import random

Move = Tuple[str, Tuple[int, ...]]  # (op_name, params)

@dataclass
class Task:
    name: str
    n: int
    moveset_name: str
    moveset_params: Dict[str, int]
    budget_steps: int
    splits: Dict[str, int]  # split_name -> size
    seed: int = 42

    def rng(self) -> random.Random:
        return random.Random(self.seed)

