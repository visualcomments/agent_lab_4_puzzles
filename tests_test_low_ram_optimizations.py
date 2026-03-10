from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "AgentLaboratory"))
sys.path.insert(0, str(ROOT / "llm-puzzles"))

import inference  # type: ignore
import CallLLM  # type: ignore


def test_g4f_to_text_bounds_stream(monkeypatch):
    chunks = ("abc", "def", "ghi")
    out = inference._g4f_to_text(chunks, max_chars=5)
    assert out == "abcde"


def test_query_model_stable_uses_worker_result(monkeypatch):
    monkeypatch.setenv("AGENTLAB_REMOTE_SUBPROCESS", "1")

    def fake_worker(**kwargs):
        kwargs["out_json"].write_text(json.dumps({"ok": True, "answer": "OK"}), encoding="utf-8")
        return {"ok": True, "answer": "OK"}

    monkeypatch.setattr(inference, "_run_json_worker_subprocess", fake_worker)
    assert inference.query_model_stable("g4f:gpt-4o-mini", "p", "s", tries=1, timeout=5.0) == "OK"


def test_callllm_iter_to_text_bounds_stream():
    out = CallLLM._iter_to_text(iter(["123", "456", "789"]), max_chars=7)
    assert out == "1234567"



def test_g4f_to_text_stops_after_python_fence():
    chunks = iter(["prefix\n```python\n", "def solve(vec):\n    return [], vec\n", "```\nignore me forever"])
    out = inference._g4f_to_text(chunks, max_chars=1000, stop_at_python_fence=True)
    assert "ignore me forever" not in out
    assert out.rstrip().endswith("```")


def test_base_agent_spills_large_artifacts(tmp_path, monkeypatch):
    import types
    import sys as _sys
    import json as _json
    import re as _re
    from datetime import datetime as _datetime

    fake_utils = types.ModuleType("utils")
    fake_utils.json = _json
    fake_utils.re = _re
    fake_tools = types.ModuleType("tools")

    monkeypatch.setitem(_sys.modules, "utils", fake_utils)
    monkeypatch.setitem(_sys.modules, "tools", fake_tools)
    from agents import BaseAgent

    class DemoAgent(BaseAgent):
        def context(self, phase):
            return ""
        def phase_prompt(self, phase):
            return ""
        def role_description(self):
            return "demo"
        def command_descriptions(self, phase):
            return ""
        def example_command(self, phase):
            return ""

    import os
    os.environ["AGENTLAB_ARTIFACT_SPILL_CHARS"] = "20"
    agent = DemoAgent(memory_dir=tmp_path, run_id="t")
    big = "A" * 200
    agent.report = big
    assert agent.report == big
    assert agent._artifact_paths.get("report")
    assert Path(agent._artifact_paths["report"]).exists()

def test_g4f_to_text_stops_on_idle_stream_timeout():
    def slow_chunks():
        yield "abc"
        time.sleep(0.25)
        yield "def"

    out = inference._g4f_to_text(
        slow_chunks(),
        max_chars=100,
        stream_timeout_s=1.0,
        stream_idle_timeout_s=0.05,
    )
    assert out == "abc"



def test_terminate_process_tree_can_avoid_killpg(monkeypatch):
    events = []

    class DummyProc:
        pid = 123
        def terminate(self):
            events.append('terminate')
        def kill(self):
            events.append('kill')
        def wait(self, timeout=None):
            events.append(f'wait:{timeout}')

    monkeypatch.setenv('AGENTLAB_WORKER_KILL_PROCESS_GROUP', '0')
    monkeypatch.setattr(inference.os, 'name', 'posix')

    def fail_killpg(pid, sig):
        events.append(f'killpg:{pid}:{sig}')
        raise AssertionError('killpg should not be used when disabled')

    monkeypatch.setattr(inference.os, 'killpg', fail_killpg)
    inference._terminate_process_tree(DummyProc())
    assert events[0] == 'terminate'
    assert not any(evt.startswith('killpg:') for evt in events)


def test_terminate_process_tree_uses_killpg_by_default(monkeypatch):
    events = []

    class DummyProc:
        pid = 321
        def kill(self):
            events.append('kill')
        def wait(self, timeout=None):
            events.append(f'wait:{timeout}')

    monkeypatch.delenv('AGENTLAB_WORKER_KILL_PROCESS_GROUP', raising=False)
    monkeypatch.setattr(inference.os, 'name', 'posix')

    def fake_killpg(pid, sig):
        events.append(f'killpg:{pid}:{sig}')
        if sig == inference.signal.SIGTERM:
            raise RuntimeError('force escalation')

    monkeypatch.setattr(inference.os, 'killpg', fake_killpg)
    inference._terminate_process_tree(DummyProc())
    assert any(evt.startswith('killpg:321:') for evt in events)


def test_query_model_prefers_g4f_async_client(monkeypatch):
    class FakeChoices:
        def __init__(self):
            self.message = type('Msg', (), {'content': 'ASYNC_OK'})()

    class FakeResponse:
        choices = [FakeChoices()]

    class FakeCompletions:
        async def create(self, **kwargs):
            return FakeResponse()

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            self.chat = type('Chat', (), {'completions': FakeCompletions()})()

    class FakeG4F:
        class ChatCompletion:
            @staticmethod
            def create(*args, **kwargs):
                raise AssertionError('sync g4f path should not be used when async is enabled')

    monkeypatch.setenv('AGENTLAB_G4F_USE_ASYNC', '1')
    monkeypatch.setenv('AGENTLAB_G4F_ASYNC_FALLBACK_TO_SYNC', '0')
    monkeypatch.setattr(inference, '_load_g4f_module', lambda: FakeG4F)
    monkeypatch.setattr(inference, '_load_g4f_async_client_class', lambda: FakeAsyncClient)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)

    out = inference.query_model('g4f:gpt-4o-mini', 'ping', 'sys', tries=1, timeout=5.0)
    assert out == 'ASYNC_OK'
