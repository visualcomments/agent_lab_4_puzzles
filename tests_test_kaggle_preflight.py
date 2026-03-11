from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'llm-puzzles'))
sys.path.insert(0, str(ROOT))

from src import kaggle_utils  # type: ignore
import pipeline_cli  # type: ignore



def test_parse_kaggle_version_extracts_semver_from_cli_output():
    assert kaggle_utils._parse_kaggle_version('Kaggle API 1.6.6') == (1, 6, 6)
    assert kaggle_utils._parse_kaggle_version('kaggle==1.5.16.post1') == (1, 5, 16)


def test_installed_kaggle_package_version_uses_metadata_without_importing_kaggle(monkeypatch):
    monkeypatch.setattr(kaggle_utils.importlib_metadata, 'version', lambda name: '2.0.0')
    assert kaggle_utils.installed_kaggle_package_version() == (2, 0, 0)



def test_preflight_api_rejects_old_submit_client(monkeypatch):
    monkeypatch.setattr(kaggle_utils, 'installed_kaggle_package_version', lambda: (1, 4, 9))

    try:
        kaggle_utils.preflight_submit_via_api('demo-comp')
    except kaggle_utils.KagglePreflightError as exc:
        assert 'too old' in str(exc)
        assert '1.5.0' in str(exc)
    else:
        raise AssertionError('expected KagglePreflightError for an old Kaggle API package')



def test_preflight_cli_uses_positional_competition_for_submissions(monkeypatch):
    monkeypatch.setattr(
        kaggle_utils,
        'get_kaggle_cli_version',
        lambda env=None: {'available': True, 'version': (1, 6, 6), 'raw': 'Kaggle API 1.6.6'},
    )

    calls = []

    def fake_run(cmd, capture_output, text, env, check):
        calls.append(cmd)

        class Proc:
            returncode = 0
            stdout = ''
            stderr = ''

        return Proc()

    report = kaggle_utils.preflight_submit_via_cli('cayleypy-rapapport-m2', runner=fake_run)

    assert report['command'][:4] == ['kaggle', 'competitions', 'submissions', 'cayleypy-rapapport-m2']
    assert calls[0][:4] == ['kaggle', 'competitions', 'submissions', 'cayleypy-rapapport-m2']
    assert '-c' not in calls[0]



def test_preflight_cli_reports_rules_acceptance_problem(monkeypatch):
    monkeypatch.setattr(
        kaggle_utils,
        'get_kaggle_cli_version',
        lambda env=None: {'available': True, 'version': (1, 6, 6), 'raw': 'Kaggle API 1.6.6'},
    )

    def fake_run(cmd, capture_output, text, env, check):
        class Proc:
            returncode = 1
            stdout = ''
            stderr = "403 - Forbidden - Permission 'competitions.participate' was denied"

        return Proc()

    try:
        kaggle_utils.preflight_submit_via_cli('cayleypy-rapapport-m2', runner=fake_run)
    except kaggle_utils.KagglePreflightError as exc:
        msg = str(exc)
        assert 'accept the competition rules' in msg
        assert 'joined the competition' in msg
    else:
        raise AssertionError('expected KagglePreflightError for missing rules acceptance / competition join')



def test_cmd_kaggle_preflight_json_reports_success(monkeypatch, capsys):
    monkeypatch.setattr(
        kaggle_utils,
        'preflight_submit_via_api',
        lambda competition, kaggle_json_path=None, config_dir=None: {
            'mode': 'api',
            'competition': competition,
            'client_version': '1.6.6',
            'access': {'can_list_submissions': True, 'rules_accepted_or_joined': True},
        },
    )
    monkeypatch.setattr(
        kaggle_utils,
        'preflight_submit_via_cli',
        lambda competition, credentials_path=None, config_dir=None: {
            'mode': 'cli',
            'competition': competition,
            'client_version': '1.6.6',
            'access': {'can_list_submissions': True, 'rules_accepted_or_joined': True},
        },
    )

    args = argparse.Namespace(
        competition='cayleypy-rapapport-m2',
        kaggle_json=None,
        kaggle_config_dir=None,
        submit_via='auto',
        submit_competition=None,
        json=True,
    )
    pipeline_cli.cmd_kaggle_preflight(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload['ok'] is True
    assert payload['competition'] == 'cayleypy-rapapport-m2'
    assert payload['results']['api']['client_version'] == '1.6.6'
    assert payload['results']['cli']['client_version'] == '1.6.6'
