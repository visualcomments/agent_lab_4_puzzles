"""Baseline placeholder solver.

This competition requires specialized domain logic (see the Kaggle page for details).

This baseline intentionally returns 'UNSOLVED' for any input.
It is useful as a smoke-test to ensure the pipeline, validators, and submission
building all work end-to-end.

To get a non-trivial score, generate a real solver with AgentLaboratory using
competition-specific prompts.
"""

from __future__ import annotations

import json
import sys
from typing import List, Sequence, Tuple, Union

MoveOut = Union[List[str], str]


def solve(vec: Sequence[int]) -> Tuple[MoveOut, List[int]]:
    # Kaggle competitions in the CayleyPy series typically accept the string 'UNSOLVED'
    # (with a heavy penalty). Returning it keeps the submission valid.
    return "UNSOLVED", list(vec)


def _main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python solve_module.py '[...]'", file=sys.stderr)
        raise SystemExit(2)

    vec = json.loads(sys.argv[1])
    if not isinstance(vec, list):
        raise SystemExit("Input must be a JSON list")

    moves, out_vec = solve(vec)
    print(json.dumps({"moves": moves, "sorted_array": out_vec}))


if __name__ == "__main__":
    _main()
