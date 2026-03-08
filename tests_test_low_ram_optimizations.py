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
