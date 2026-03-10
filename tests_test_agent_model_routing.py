from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'AgentLaboratory'))
sys.path.insert(0, str(ROOT / 'AgentLaboratory' / 'perm_pipeline'))
sys.path.insert(0, str(ROOT / 'llm-puzzles'))
sys.path.insert(0, str(ROOT))

import pipeline_cli  # type: ignore
import run_perm_pipeline as rpp  # type: ignore


def test_parse_agent_model_overrides_and_resolve():
    overrides = rpp.parse_agent_model_overrides('planner=gpt-4; coder=local:Qwen/demo | gpt-4o-mini\nfixer=g4f:deepseek-chat')
    assert overrides['planner'] == ['g4f:gpt-4']
    assert overrides['coder'] == ['local:Qwen/demo', 'g4f:gpt-4o-mini']
    assert overrides['fixer'] == ['g4f:deepseek-chat']
    assert rpp.resolve_agent_models('planner', ['g4f:aria'], overrides) == ['g4f:gpt-4']
    assert rpp.resolve_agent_models('reviewer', ['g4f:aria'], overrides) == ['g4f:aria']


def test_try_generate_with_model_uses_separate_fixer_models(monkeypatch, tmp_path):
    calls = []

    def fake_query_model_stable(model, prompt, system_prompt, **kwargs):
        calls.append(model)
        if model == 'g4f:coder-main':
            return '```python\ndef solve(:\n```'
        if model == 'g4f:fixer-a':
            return 'not python'
        if model == 'g4f:fixer-b':
            return '```python\ndef solve(vec):\n    return [], list(vec)\n```'
        raise AssertionError(model)

    monkeypatch.setattr(rpp, '_query_model_stable', fake_query_model_stable)
    monkeypatch.setattr(rpp, 'validate_solver_suite', lambda *args, **kwargs: (True, 'OK'))
    monkeypatch.setattr(rpp, '_memory_limit_exceeded', lambda: (False, 0.0, 0))
    monkeypatch.setattr(rpp, '_best_effort_release_memory', lambda clear_local_cache=False: None)

    out_path = tmp_path / 'solver.py'
    ok, report = rpp.try_generate_with_model(
        model='g4f:coder-main',
        fixer_models=['g4f:fixer-a', 'g4f:fixer-b'],
        user_prompt='task',
        plan='plan',
        prompts={'coder': 'coder-sys', 'fixer': 'fixer-sys'},
        out_path=out_path,
        validator_path=tmp_path / 'validator.py',
        tests=[[1, 2, 3]],
        max_iters=1,
    )

    assert ok is True
    assert 'validated after fixer iteration 1' in report
    assert calls == ['g4f:coder-main', 'g4f:fixer-a', 'g4f:fixer-a', 'g4f:fixer-b']
    assert 'def solve(vec)' in out_path.read_text(encoding='utf-8')



def test_run_agent_laboratory_passes_agent_model_flags(monkeypatch, tmp_path):
    captured = {}

    def fake_check_call(cmd, cwd=None, env=None):
        captured['cmd'] = cmd
        captured['cwd'] = cwd
        captured['env'] = env
        return 0

    monkeypatch.setattr(pipeline_cli.subprocess, 'check_call', fake_check_call)

    pipeline_cli._run_agent_laboratory(
        prompt_file=tmp_path / 'prompt.txt',
        out_path=tmp_path / 'out.py',
        validator=tmp_path / 'validator.py',
        baseline=tmp_path / 'baseline.py',
        llm='gpt-4o-mini',
        agent_models='planner=gpt-4;coder=local:Qwen/demo',
        planner_models='gpt-4',
        coder_models='local:Qwen/demo',
        fixer_models='gpt-4o-mini',
        max_iters=3,
        g4f_switch_sleep=2.5,
        g4f_repeat_model_sleep=0.5,
        print_generation=True,
        print_generation_max_chars=321,
    )

    cmd = captured['cmd']
    assert '--agent-models' in cmd
    assert '--planner-models' in cmd
    assert '--coder-models' in cmd
    assert '--fixer-models' in cmd
    assert captured['env']['AGENTLAB_G4F_MODEL_SWITCH_SLEEP_S'] == '2.5'
    assert captured['env']['AGENTLAB_G4F_REPEAT_MODEL_SLEEP_S'] == '0.5'
    assert captured['env']['AGENTLAB_PRINT_GENERATION'] == '1'
    assert captured['env']['AGENTLAB_PRINT_GENERATION_MAX_CHARS'] == '321'



def test_build_parser_supports_agent_model_flags_for_generate_and_run():
    parser = pipeline_cli.build_parser()
    gen_args = parser.parse_args([
        'generate-solver',
        '--competition', 'cayleypy-rapapport-m2',
        '--out', 'generated/x.py',
        '--agent-models', 'planner=gpt-4;coder=local:Qwen/demo',
        '--planner-models', 'gpt-4',
        '--coder-models', 'local:Qwen/demo',
        '--fixer-models', 'gpt-4o-mini',
        '--g4f-switch-sleep', '2.0',
        '--g4f-repeat-model-sleep', '0.4',
        '--print-generation',
        '--print-generation-max-chars', '555',
    ])
    assert gen_args.agent_models.startswith('planner=')
    assert gen_args.planner_models == 'gpt-4'
    assert gen_args.coder_models == 'local:Qwen/demo'
    assert gen_args.fixer_models == 'gpt-4o-mini'
    assert gen_args.g4f_switch_sleep == 2.0
    assert gen_args.g4f_repeat_model_sleep == 0.4
    assert gen_args.print_generation is True
    assert gen_args.print_generation_max_chars == 555

    run_args = parser.parse_args([
        'run',
        '--competition', 'cayleypy-rapapport-m2',
        '--output', 'submissions/out.csv',
        '--agent-models', 'planner=gpt-4;coder=local:Qwen/demo',
        '--print-generation',
    ])
    assert run_args.agent_models.startswith('planner=')
    assert run_args.print_generation is True
