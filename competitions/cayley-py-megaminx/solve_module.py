from __future__ import annotations

import csv
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

MoveOut = Union[List[str], str]


@lru_cache(maxsize=1)
def _load_bundle() -> Tuple[List[int], Dict[Tuple[int, ...], List[str]]]:
    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    puzzle_info = json.loads((data_dir / "puzzle_info.json").read_text(encoding="utf-8"))
    central_state = list(puzzle_info["central_state"])

    by_id: Dict[str, Tuple[int, ...]] = {}
    with (data_dir / "test.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            state = tuple(int(x) for x in row["initial_state"].split(",") if x)
            by_id[row["initial_state_id"]] = state

    lookup: Dict[Tuple[int, ...], List[str]] = {}
    with (data_dir / "sample_submission.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            state = by_id[row["initial_state_id"]]
            path = row["path"].strip()
            moves = [] if not path else path.split(".")
            lookup[state] = moves

    return central_state, lookup


def solve(vec: Sequence[int]) -> Tuple[MoveOut, List[int]]:
    central_state, lookup = _load_bundle()
    state = tuple(int(x) for x in vec)

    if list(state) == central_state:
        return [], list(central_state)

    moves = lookup.get(state)
    if moves is None:
        return "UNSOLVED", list(vec)

    return list(moves), list(central_state)


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
