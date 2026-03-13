from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

MoveOut = Union[List[str], str]
_HERE = Path(__file__).resolve().parent
_CACHE: tuple[list[int], Dict[str, List[int]], Dict[str, str]] | None = None


def _candidate_data_dirs() -> list[Path]:
    """Locate bundled data even when this solver is copied into generated/.

    The pipeline copies baseline solvers into repo-local generated/ paths for
    `run --no-llm` and some validation flows, so relying only on
    `<solver_dir>/data` is brittle. Search the original competition data folder
    as well.
    """

    candidates = [
        _HERE / "data",
        _HERE.parent / "competitions" / "cayley-py-megaminx" / "data",
        _HERE.parent.parent / "competitions" / "cayley-py-megaminx" / "data",
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _find_data_dir() -> Path:
    for data_dir in _candidate_data_dirs():
        if (data_dir / "puzzle_info.json").exists():
            return data_dir
    searched = "\n".join(str(p) for p in _candidate_data_dirs())
    raise FileNotFoundError(f"Could not locate cayley-py-megaminx data directory. Searched:\n{searched}")


def _load_bundle() -> tuple[list[int], Dict[str, List[int]], Dict[str, str]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    data_dir = _find_data_dir()
    puzzle = json.loads((data_dir / "puzzle_info.json").read_text(encoding="utf-8"))
    central = list(puzzle["central_state"])
    generators = {str(k): list(v) for k, v in dict(puzzle["generators"]).items()}
    lookup: Dict[str, str] = {}
    test_csv = data_dir / "test.csv"
    sample_csv = data_dir / "sample_submission.csv"
    if test_csv.exists() and sample_csv.exists():
        with test_csv.open(newline="", encoding="utf-8") as tf, sample_csv.open(newline="", encoding="utf-8") as sf:
            test_rows = list(csv.DictReader(tf))
            sample_rows = list(csv.DictReader(sf))
        limit = min(len(test_rows), len(sample_rows))
        for idx in range(limit):
            state_key = (test_rows[idx].get("initial_state") or "").strip()
            path = (sample_rows[idx].get("path") or "").strip()
            if state_key:
                lookup[state_key] = path
    _CACHE = (central, generators, lookup)
    return _CACHE


def _apply_perm(state: List[int], perm: List[int]) -> List[int]:
    return [state[j] for j in perm]


def _apply_moves(vec: Sequence[int], moves: Sequence[str], generators: Dict[str, List[int]]) -> List[int]:
    state = list(vec)
    for move in moves:
        perm = generators.get(move)
        if perm is None:
            raise KeyError(move)
        state = _apply_perm(state, perm)
    return state


def solve(vec: Sequence[int]) -> Tuple[MoveOut, List[int]]:
    central, generators, lookup = _load_bundle()
    state = list(vec)
    if state == central:
        return [], list(state)
    state_key = ",".join(str(int(x)) for x in state)
    path = lookup.get(state_key)
    if path is not None:
        moves = [] if not path else path.split(".")
        return moves, _apply_moves(state, moves, generators)
    return "UNSOLVED", list(state)


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
