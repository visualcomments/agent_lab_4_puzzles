# examples/cayleypy_solver/baseline.py
# Universal baseline solver: returns empty moves for every row (solves only already-solved cases).
# Replace with your logic; you receive both the row (dict) and the CompConfig.
from typing import Dict, List, Union
from ...src.comp_registry import CompConfig

def solve_row(row: dict, cfg: CompConfig) -> Union[str, List[str]]:
    # Example: you might inspect row fields like "scramble", "allowed_moves", etc.
    # Return a string "a.b.c" or a list ["a","b","c"].
    return ""  # TODO: produce an actual move string/list per competition format
