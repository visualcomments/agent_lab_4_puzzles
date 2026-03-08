from pathlib import Path
import json
import sys

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

    def fake_run(cmd, capture_output, text, env, timeout):
        out_json = Path(cmd[cmd.index("--out-json") + 1])
        out_json.write_text(json.dumps({"ok": True, "answer": "OK"}), encoding="utf-8")
        class CP:
            stdout = ""
            stderr = ""
        return CP()

    monkeypatch.setattr(inference.subprocess, "run", fake_run)
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
