import asyncio
import json
from types import SimpleNamespace

import pipeline_cli


def test_build_parser_registers_check_g4f_models_command():
    parser = pipeline_cli.build_parser()
    args = parser.parse_args(["check-g4f-models", "--list-only"])
    assert args.func is pipeline_cli.cmd_check_g4f_models
    assert args.list_only is True
    assert args.concurrency == 5
    assert args.probe_mode == "pipeline"


def test_discover_g4f_candidate_models_filters_media_and_keeps_text_models(monkeypatch):
    class BaseModel:
        def __init__(self, name: str):
            self.name = name

    class ImageModel(BaseModel):
        pass

    class AudioModel(BaseModel):
        pass

    class VideoModel(BaseModel):
        pass

    class VisionModel(BaseModel):
        pass

    class Registry:
        @staticmethod
        def all_models():
            return {
                "gpt-4": BaseModel("gpt-4"),
                "gpt-4o": VisionModel("gpt-4o"),
                "flux": ImageModel("flux"),
                "whisper": AudioModel("whisper"),
                "sora": VideoModel("sora"),
            }

    fake_module = SimpleNamespace(
        ModelRegistry=Registry,
        ImageModel=ImageModel,
        AudioModel=AudioModel,
        VideoModel=VideoModel,
    )
    monkeypatch.setattr(pipeline_cli, "_load_g4f_models_module", lambda: fake_module)

    models = pipeline_cli._discover_g4f_candidate_models()
    assert models == ["gpt-4", "gpt-4o"]




def test_probe_g4f_model_pipeline_uses_agentlab_query_model_stable(monkeypatch):
    captured = {}

    fake_module = SimpleNamespace(
        query_model_stable=lambda **kwargs: captured.update(kwargs) or "pong"
    )
    monkeypatch.setattr(pipeline_cli, "_load_agentlab_inference_module", lambda: fake_module)
    monkeypatch.delenv("G4F_PROVIDER", raising=False)

    ok, info, elapsed = pipeline_cli._probe_g4f_model_pipeline(
        model="gpt-4o-mini",
        timeout=7.5,
        prompt="ping",
        system_prompt="Return a short reply.",
        provider_name="Blackbox",
    )

    assert ok is True
    assert info == "pong"
    assert elapsed >= 0
    assert captured["model_str"] == "g4f:gpt-4o-mini"
    assert captured["tries"] == 1
    assert captured["timeout"] == 7.5
    assert "G4F_PROVIDER" not in __import__("os").environ


def test_probe_g4f_model_async_uses_async_client_and_extracts_content(monkeypatch):
    captured = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="pong"))]
            )

    class FakeAsyncClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(pipeline_cli, "_load_g4f_async_client_class", lambda: FakeAsyncClient)

    ok, info, elapsed = asyncio.run(
        pipeline_cli._probe_g4f_model_async(
            model="g4f:gpt-4o-mini",
            timeout=1.0,
            prompt="ping",
            system_prompt="Return a very short plain-text reply.",
            provider_name="Blackbox",
        )
    )

    assert ok is True
    assert info == "pong"
    assert elapsed >= 0
    assert captured["model"] == "gpt-4o-mini"
    assert captured["provider"] == "Blackbox"
    assert captured["web_search"] is False
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][1]["content"] == "ping"



def test_probe_g4f_models_async_honors_concurrency_and_preserves_order(monkeypatch):
    in_flight = 0
    max_seen = 0
    completion_order = []

    async def fake_probe(model, timeout, prompt, system_prompt, provider_name=None):
        nonlocal in_flight, max_seen
        in_flight += 1
        max_seen = max(max_seen, in_flight)
        await asyncio.sleep(0.01 if model == "b" else 0)
        completion_order.append(model)
        in_flight -= 1
        return model != "c", f"detail-{model}", 0.01

    monkeypatch.setattr(pipeline_cli, "_probe_g4f_model_async", fake_probe)

    seen_callbacks = []

    def on_result(idx, total, result):
        seen_callbacks.append((idx, total, result["model"]))

    results = asyncio.run(
        pipeline_cli._probe_g4f_models_async(
            ["a", "b", "c"],
            timeout=1.0,
            prompt="ping",
            system_prompt="sys",
            provider_name=None,
            concurrency=2,
            on_result=on_result,
        )
    )

    assert [r["model"] for r in results] == ["a", "b", "c"]
    assert [r["ok"] for r in results] == [True, True, False]
    assert max_seen <= 2
    assert completion_order[0] in {"a", "b"}
    assert seen_callbacks and all(total == 3 for _, total, _ in seen_callbacks)



def test_cmd_check_g4f_models_prints_working_models(monkeypatch, capsys):
    monkeypatch.setattr(
        pipeline_cli,
        "_discover_g4f_candidate_models",
        lambda backend_api_url=None: ["gpt-4o-mini", "command-r", "aria"],
    )

    def fake_probe_many(candidates, **kwargs):
        return [
            {"model": "gpt-4o-mini", "ok": True, "detail": "OK", "elapsed_s": 0.1},
            {"model": "command-r", "ok": False, "detail": "bad gateway", "elapsed_s": 0.3},
            {"model": "aria", "ok": True, "detail": "OK", "elapsed_s": 0.2},
        ]

    monkeypatch.setattr(pipeline_cli, "_probe_g4f_models_sync", fake_probe_many)

    args = pipeline_cli.build_parser().parse_args(["check-g4f-models", "--max-models", "3"])
    args.func(args)

    out = capsys.readouterr().out
    assert "Working g4f models:" in out
    assert "gpt-4o-mini" in out
    assert "aria" in out



def test_cmd_check_g4f_models_list_only_prints_only_models_that_answered(monkeypatch, capsys):
    monkeypatch.setattr(
        pipeline_cli,
        "_discover_g4f_candidate_models",
        lambda backend_api_url=None: ["gpt-4o-mini", "command-r", "aria"],
    )

    def fake_probe_many(candidates, **kwargs):
        assert kwargs["prompt"] == "ping"
        return [
            {"model": "gpt-4o-mini", "ok": True, "detail": "pong", "elapsed_s": 0.1},
            {"model": "command-r", "ok": False, "detail": "timeout", "elapsed_s": 0.2},
            {"model": "aria", "ok": True, "detail": "pong", "elapsed_s": 0.1},
        ]

    monkeypatch.setattr(pipeline_cli, "_probe_g4f_models_sync", fake_probe_many)

    args = pipeline_cli.build_parser().parse_args(["check-g4f-models", "--list-only", "--max-models", "3"])
    args.func(args)

    out_lines = [line.strip() for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert out_lines == ["gpt-4o-mini", "aria"]



def test_cmd_check_g4f_models_json_list_only_reports_only_working_subset(monkeypatch, capsys):
    def fake_probe_many(candidates, **kwargs):
        return [
            {"model": "gpt-4o-mini", "ok": True, "detail": "OK", "elapsed_s": 0.1},
            {"model": "aria", "ok": False, "detail": "err", "elapsed_s": 0.1},
        ]

    monkeypatch.setattr(pipeline_cli, "_probe_g4f_models_sync", fake_probe_many)

    args = pipeline_cli.build_parser().parse_args(
        ["check-g4f-models", "--list-only", "--json", "--models", "g4f:gpt-4o-mini,gpt-4o-mini,aria"]
    )
    args.func(args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["working_models"] == ["gpt-4o-mini"]
    assert payload["working_count"] == 1
    assert payload["checked_count"] == 2
    assert payload["probe_prompt"] == "ping"



def test_cmd_check_g4f_models_discover_only_json_preserves_candidate_listing(capsys):
    args = pipeline_cli.build_parser().parse_args(
        ["check-g4f-models", "--discover-only", "--json", "--models", "g4f:gpt-4o-mini,gpt-4o-mini,aria"]
    )
    args.func(args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["discovered_models"] == ["gpt-4o-mini", "aria"]
    assert payload["discovered_count"] == 2


def test_cmd_check_g4f_models_async_mode_uses_async_probe(monkeypatch, capsys):
    monkeypatch.setattr(
        pipeline_cli,
        "_discover_g4f_candidate_models",
        lambda backend_api_url=None: ["gpt-4o-mini", "aria"],
    )

    async def fake_probe_many(candidates, **kwargs):
        return [
            {"model": "gpt-4o-mini", "ok": True, "detail": "OK", "elapsed_s": 0.1},
            {"model": "aria", "ok": False, "detail": "bad gateway", "elapsed_s": 0.2},
        ]

    monkeypatch.setattr(pipeline_cli, "_probe_g4f_models_async", fake_probe_many)

    args = pipeline_cli.build_parser().parse_args(["check-g4f-models", "--probe-mode", "async", "--max-models", "2"])
    args.func(args)

    out = capsys.readouterr().out
    assert "Working g4f models:" in out
    assert "gpt-4o-mini" in out


def test_memory_env_for_codegen_defaults_to_async_and_no_clip(monkeypatch):
    monkeypatch.delenv('AGENTLAB_G4F_USE_ASYNC', raising=False)
    monkeypatch.delenv('AGENTLAB_MAX_RESPONSE_CHARS', raising=False)
    monkeypatch.delenv('AGENTLAB_G4F_STOP_AT_PYTHON_FENCE', raising=False)
    env = pipeline_cli._memory_env_for_codegen('gpt-4o-mini')
    assert env['AGENTLAB_G4F_USE_ASYNC'] == '1'
    assert env['AGENTLAB_MAX_RESPONSE_CHARS'] == '0'
    assert env['AGENTLAB_G4F_STOP_AT_PYTHON_FENCE'] == '1'



def test_memory_env_for_codegen_defaults_to_stop_at_python_fence(monkeypatch):
    monkeypatch.delenv('AGENTLAB_G4F_STOP_AT_PYTHON_FENCE', raising=False)
    env = pipeline_cli._memory_env_for_codegen('g4f:gpt-4o-mini')
    assert env['AGENTLAB_G4F_STOP_AT_PYTHON_FENCE'] == '1'
