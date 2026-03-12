from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import pipeline_cli  # type: ignore


def _load_sample_row(initial_state_id: str) -> tuple[list[int], list[str]]:
    data_dir = ROOT / 'competitions' / 'cayley-py-megaminx' / 'data'
    with (data_dir / 'test.csv').open(newline='', encoding='utf-8') as f:
        state_row = next(row for row in csv.DictReader(f) if row['initial_state_id'] == initial_state_id)
    with (data_dir / 'sample_submission.csv').open(newline='', encoding='utf-8') as f:
        path_row = next(row for row in csv.DictReader(f) if row['initial_state_id'] == initial_state_id)
    state = [int(x) for x in state_row['initial_state'].split(',') if x]
    moves = [] if not path_row['path'] else path_row['path'].split('.')
    return state, moves


def test_megaminx_solver_replays_official_sample_path_for_known_state():
    solver_path = ROOT / 'competitions' / 'cayley-py-megaminx' / 'solve_module.py'
    solve = pipeline_cli._load_solve_fn(solver_path)
    vec, expected_moves = _load_sample_row('2')
    moves, sorted_array = solve(vec)
    assert moves == expected_moves
    assert sorted_array == list(range(120))


def test_megaminx_validator_accepts_known_sample_path(tmp_path):
    data_dir = ROOT / 'competitions' / 'cayley-py-megaminx' / 'data'
    solver_path = ROOT / 'competitions' / 'cayley-py-megaminx' / 'solve_module.py'
    validator_path = ROOT / 'competitions' / 'cayley-py-megaminx' / 'validate_solve_output.py'
    vec, _ = _load_sample_row('1')

    import subprocess
    import sys as _sys

    subprocess.check_call([
        _sys.executable,
        str(validator_path),
        '--solver',
        str(solver_path),
        '--vector',
        json.dumps(vec),
    ])
