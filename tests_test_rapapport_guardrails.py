from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'llm-puzzles'))
sys.path.insert(0, str(ROOT))

import pipeline_cli  # type: ignore
from src import kaggle_utils  # type: ignore


def test_validate_submission_move_tokens_rejects_h(tmp_path):
    submission = tmp_path / 'submission.csv'
    submission.write_text(
        'id,permutation,solution\n'
        '0,"2,1,0",I.S.K\n'
        '1,"2,0,1",I.H.K\n',
        encoding='utf-8',
    )

    try:
        pipeline_cli._validate_submission_move_tokens(
            submission_csv=submission,
            move_column='solution',
            allowed_moves={'I', 'S', 'K'},
            joiner='.',
        )
    except ValueError as exc:
        msg = str(exc)
        assert "Row 2" in msg
        assert "'H'" in msg
    else:
        raise AssertionError('expected ValueError for illegal move token')


def test_finalize_submission_output_preserves_backup(tmp_path):
    out_csv = tmp_path / 'submission.csv'
    candidate = pipeline_cli._candidate_output_path(out_csv)
    out_csv.write_text('old\n', encoding='utf-8')
    candidate.write_text('new\n', encoding='utf-8')

    pipeline_cli._finalize_submission_output(candidate, out_csv)

    assert out_csv.read_text(encoding='utf-8') == 'new\n'
    assert (tmp_path / 'submission.csv.bak').read_text(encoding='utf-8') == 'old\n'
    assert not candidate.exists()


def test_discover_default_kaggle_credentials_path_prefers_existing_files(tmp_path, monkeypatch):
    fake_home = tmp_path / 'home'
    kaggle_dir = fake_home / '.kaggle'
    kaggle_dir.mkdir(parents=True)
    token_file = kaggle_dir / 'access_token'
    token_file.write_text('secret-token', encoding='utf-8')

    monkeypatch.setattr(kaggle_utils.Path, 'home', staticmethod(lambda: fake_home))

    discovered = kaggle_utils._discover_default_credentials_path()
    assert discovered == str(token_file)

    env = kaggle_utils.build_kaggle_env()
    assert env['KAGGLE_CONFIG_DIR']
    assert env['KAGGLE_API_TOKEN'] == 'secret-token'


def test_rapapport_solver_only_emits_isk_and_sorts():
    solver_path = ROOT / 'competitions' / 'cayleypy-rapapport-m2' / 'solve_module.py'
    solve = pipeline_cli._load_solve_fn(solver_path)
    vec = [3, 0, 1, 4, 2]
    moves, sorted_array = solve(vec)
    assert set(moves) <= {'I', 'S', 'K'}
    assert sorted_array == [0, 1, 2, 3, 4]



def test_rapapport_sample_solution_matches_competition_semantics():
    vec = [3, 0, 1, 4, 2]
    for m in ['S', 'K', 'I', 'S']:
        if m == 'S':
            vec[0], vec[1] = vec[1], vec[0]
        elif m == 'I':
            for i in range(0, len(vec) - 1, 2):
                vec[i], vec[i + 1] = vec[i + 1], vec[i]
        elif m == 'K':
            for i in range(1, len(vec) - 1, 2):
                vec[i], vec[i + 1] = vec[i + 1], vec[i]
    assert vec == [0, 1, 2, 3, 4]
