from pathlib import Path
import json
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'AgentLaboratory'))
sys.path.insert(0, str(ROOT / 'AgentLaboratory' / 'perm_pipeline'))

import run_perm_pipeline as rpp  # type: ignore


def test_extract_python_from_fenced_block():
    text = 'before\n```python\ndef solve(vec):\n    return [], list(vec)\n```\nafter'
    code = rpp.extract_python(text)
    assert code is not None
    assert 'def solve' in code


def test_compile_python_reports_syntax_error():
    ok, reason = rpp.compile_python('def solve(:\n    pass')
    assert ok is False
    assert 'SyntaxError' in reason


def test_rank_models_prefers_stronger_code_models():
    ranked = rpp.rank_models_for_codegen([
        'g4f:aria',
        'g4f:gpt-4o-mini',
        'g4f:command-r',
    ])
    assert ranked[0] == 'g4f:gpt-4o-mini'


def test_default_max_rss_mb_is_disabled_outside_explicit_or_colab(monkeypatch):
    monkeypatch.delenv('AGENTLAB_MAX_RSS_MB', raising=False)
    monkeypatch.delenv('COLAB_GPU', raising=False)
    monkeypatch.delenv('COLAB_RELEASE_TAG', raising=False)
    monkeypatch.setattr(rpp, '_system_total_mb', lambda: 10000.0)
    assert rpp._default_max_rss_mb() == 0


def test_default_max_rss_mb_uses_explicit_env_or_colab_default(monkeypatch):
    monkeypatch.setenv('AGENTLAB_MAX_RSS_MB', '4096')
    assert rpp._default_max_rss_mb() == 4096

    monkeypatch.delenv('AGENTLAB_MAX_RSS_MB', raising=False)
    monkeypatch.setenv('COLAB_GPU', '1')
    monkeypatch.setattr(rpp, '_system_total_mb', lambda: 10000.0)
    assert rpp._default_max_rss_mb() == 7200


def test_query_model_stable_uses_worker_result(monkeypatch, tmp_path):
    monkeypatch.setenv('AGENTLAB_REMOTE_SUBPROCESS', '1')

    def fake_worker(**kwargs):
        kwargs['out_json'].write_text(json.dumps({'ok': True, 'answer': '```python\ndef solve(vec):\n    return [], list(vec)\n```'}), encoding='utf-8')
        return {'ok': True, 'answer': '```python\ndef solve(vec):\n    return [], list(vec)\n```'}

    monkeypatch.setattr(rpp, '_run_json_worker_subprocess', fake_worker)
    answer = rpp._query_model_stable('g4f:gpt-4', 'prompt', 'system', tries=1, timeout=5.0)
    assert 'def solve' in answer


def test_query_model_stable_bypasses_worker_for_local(monkeypatch):
    monkeypatch.setenv('AGENTLAB_REMOTE_SUBPROCESS', '1')

    def fake_query_model(model, prompt, system_prompt, **kwargs):
        return 'LOCAL_OK'

    monkeypatch.setattr(rpp, 'query_model', fake_query_model)
    answer = rpp._query_model_stable('local:demo-model', 'prompt', 'system')
    assert answer == 'LOCAL_OK'


def test_attempt_recovery_rounds_retries_models(monkeypatch, tmp_path):
    reports = [
        'g4f:gpt-4: fixer iteration 44 did not return a python file',
        'g4f:gpt-4: baseline-patcher did not return a python file',
    ]
    monkeypatch.setenv('AGENTLAB_G4F_RECOVERY_ROUNDS', '1')
    monkeypatch.setenv('AGENTLAB_G4F_RECOVERY_MAX_ITERS', '1')
    monkeypatch.setenv('AGENTLAB_G4F_RECOVERY_SLEEP_S', '0')

    calls = []

    def fake_try_generate_with_model(**kwargs):
        calls.append((kwargs['model'], kwargs['stage_label'], kwargs['plan'], kwargs['baseline_code']))
        return True, 'g4f:gpt-4: recovery round 1 output validated immediately'

    monkeypatch.setattr(rpp, 'try_generate_with_model', fake_try_generate_with_model)
    monkeypatch.setattr(rpp, '_best_effort_release_memory', lambda clear_local_cache=True: None)

    ok, report = rpp.attempt_recovery_rounds(
        recovery_models=['g4f:gpt-4'],
        fixer_models=['g4f:gpt-4'],
        user_prompt='solve it',
        plan='initial plan',
        prompts={'coder': 'coder', 'fixer': 'fixer'},
        out_path=tmp_path / 'solve.py',
        validator_path=tmp_path / 'validator.py',
        tests=[[1, 2, 3]],
        max_iters=4,
        baseline_code='def solve(vec):\n    return [], list(vec)\n',
        generation_reports=reports,
    )

    assert ok is True
    assert 'recovery round 1' in report
    assert calls
    assert calls[0][1] == 'recovery round 1'
    assert 'RECOVERY MODE' in calls[0][2]
    assert 'did not return a python file' in calls[0][2]


def test_attempt_recovery_rounds_skips_nonrecoverable_reports(monkeypatch, tmp_path):
    monkeypatch.setenv('AGENTLAB_G4F_RECOVERY_ROUNDS', '1')
    called = []

    def fake_try_generate_with_model(**kwargs):
        called.append(kwargs)
        return False, 'should not run'

    monkeypatch.setattr(rpp, 'try_generate_with_model', fake_try_generate_with_model)
    ok, report = rpp.attempt_recovery_rounds(
        recovery_models=['g4f:gpt-4'],
        fixer_models=['g4f:gpt-4'],
        user_prompt='solve it',
        plan='plan',
        prompts={'coder': 'coder', 'fixer': 'fixer'},
        out_path=tmp_path / 'solve.py',
        validator_path=tmp_path / 'validator.py',
        tests=[[1, 2, 3]],
        max_iters=4,
        baseline_code='def solve(vec):\n    return [], list(vec)\n',
        generation_reports=['compile check failed'],
    )
    assert ok is False
    assert report is None
    assert called == []
