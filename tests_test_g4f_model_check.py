import json
from types import SimpleNamespace

import pipeline_cli


def test_build_parser_registers_check_g4f_models_command():
    parser = pipeline_cli.build_parser()
    args = parser.parse_args(["check-g4f-models", "--list-only"])
    assert args.func is pipeline_cli.cmd_check_g4f_models
    assert args.list_only is True


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


def test_cmd_check_g4f_models_prints_working_models(monkeypatch, capsys):
    monkeypatch.setattr(
        pipeline_cli,
        "_discover_g4f_candidate_models",
        lambda backend_api_url=None: ["gpt-4o-mini", "command-r", "aria"],
    )

    def fake_probe(model, timeout, prompt, system_prompt):
        if model == "gpt-4o-mini":
            return True, "OK", 0.1
        if model == "aria":
            return True, "OK", 0.2
        return False, "bad gateway", 0.3

    monkeypatch.setattr(pipeline_cli, "_probe_g4f_model", fake_probe)

    args = pipeline_cli.build_parser().parse_args(["check-g4f-models", "--max-models", "3"])
    args.func(args)

    out = capsys.readouterr().out
    assert "Working g4f models:" in out
    assert "gpt-4o-mini" in out
    assert "aria" in out
    assert "command-r" in out  # shown in per-model status lines


def test_cmd_check_g4f_models_json_list_only_dedupes_prefixes(capsys):
    args = pipeline_cli.build_parser().parse_args(
        ["check-g4f-models", "--list-only", "--json", "--models", "g4f:gpt-4o-mini,gpt-4o-mini,aria"]
    )
    args.func(args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["models"] == ["gpt-4o-mini", "aria"]
    assert payload["count"] == 2
