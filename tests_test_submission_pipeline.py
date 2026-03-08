from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "llm-puzzles"))
sys.path.insert(0, str(ROOT))

from src.comp_registry import get_config  # type: ignore
import pipeline_cli  # type: ignore


def test_current_cayleypy_sample_backed_schema_configs():
    for slug in ["cayleypy-pancake", "cayleypy-glushkov", "cayleypy-rapapport-m2", "CayleyPy-pancake"]:
        cfg = get_config(slug)
        assert cfg.submission_headers == ["id", "permutation", "solution"]
        assert cfg.header_keys == ["id", "permutation", "moves"]
        assert cfg.puzzles_id_field == "id"


def test_preferred_kaggle_cli_submit_cmd_uses_positional_competition(tmp_path):
    out = tmp_path / "submission.csv"
    cmd = pipeline_cli._preferred_kaggle_cli_submit_cmd("cayleypy-rapapport-m2", out, "msg")
    assert cmd[:4] == ["kaggle", "competitions", "submit", "cayleypy-rapapport-m2"]
    assert "-c" not in cmd


def test_bundled_sample_submission_matches_competition_zip_schema():
    for comp in ["cayleypy-pancake", "cayleypy-glushkov", "cayleypy-rapapport-m2"]:
        sp = ROOT / "competitions" / comp / "data" / "sample_submission.csv"
        with sp.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            row = next(reader)
        assert header == ["id", "permutation", "solution"]
        assert row[0].isdigit()
        assert "," in row[1]


def test_infer_format_slug_from_sample_header(tmp_path):
    sample = tmp_path / "sample_submission.csv"
    sample.write_text("id,permutation,solution\n0,3,UNSOLVED\n", encoding="utf-8")
    assert pipeline_cli._infer_format_slug_from_sample(sample) == "format/id+permutation+solution"
