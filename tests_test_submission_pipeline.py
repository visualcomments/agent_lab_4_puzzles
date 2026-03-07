from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "llm-puzzles"))
sys.path.insert(0, str(ROOT))

from src.comp_registry import get_config  # type: ignore
import pipeline_cli  # type: ignore


def test_current_cayleypy_path_schema_configs():
    for slug in ["cayleypy-pancake", "cayleypy-glushkov", "cayleypy-rapapport-m2", "CayleyPy-pancake"]:
        cfg = get_config(slug)
        assert cfg.submission_headers == ["initial_state_id", "path"]
        assert cfg.header_keys == ["id", "moves"]
        assert cfg.puzzles_id_field == "id"


def test_preferred_kaggle_cli_submit_cmd_uses_positional_competition(tmp_path):
    out = tmp_path / "submission.csv"
    cmd = pipeline_cli._preferred_kaggle_cli_submit_cmd("cayleypy-rapapport-m2", out, "msg")
    assert cmd[:4] == ["kaggle", "competitions", "submit", "cayleypy-rapapport-m2"]
    assert "-c" not in cmd


def test_bundled_sample_submission_matches_current_path_schema():
    for comp in ["cayleypy-pancake", "cayleypy-glushkov", "cayleypy-rapapport-m2"]:
        sp = ROOT / "competitions" / comp / "data" / "sample_submission.csv"
        with sp.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            row = next(reader)
        assert header == ["initial_state_id", "path"]
        assert row[0].isdigit()
