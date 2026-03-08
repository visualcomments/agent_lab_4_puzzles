from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'AgentLaboratory'))

import inference  # type: ignore


def test_local_model_runtime_config_reflects_env(monkeypatch):
    monkeypatch.setenv('AGENTLAB_LOCAL_QUANT', '4bit')
    monkeypatch.setenv('AGENTLAB_ATTN_IMPL', 'flash_attention_2')
    monkeypatch.setenv('AGENTLAB_LOCAL_DTYPE', 'bf16')
    monkeypatch.setenv('AGENTLAB_LOCAL_MAX_MEMORY', 'cuda:6GiB,cpu:24GiB')
    monkeypatch.setenv('AGENTLAB_ENABLE_OFFLOAD', '1')
    cfg = inference.local_model_runtime_config()
    assert cfg['quant'] == '4bit'
    assert cfg['attn_impl'] == 'flash_attention_2'
    assert cfg['dtype'] == 'bf16'
    assert cfg['offload_enabled'] is True
    assert cfg['max_memory'] == {'cuda': '6GiB', 'cpu': '24GiB'}


def test_local_transformers_load_applies_quant_attn_and_offload(monkeypatch):
    inference._LOCAL_LM_CACHE.clear()

    captured = {}

    fake_torch = types.ModuleType('torch')
    fake_torch.float16 = 'float16'
    fake_torch.bfloat16 = 'bfloat16'
    class FakeCuda:
        @staticmethod
        def is_available():
            return True
    fake_torch.cuda = FakeCuda()
    def fake_compile(model, mode=None):
        captured['compile_mode'] = mode
        return model
    fake_torch.compile = fake_compile

    class FakeTokenizer:
        pad_token_id = None
        eos_token_id = 42
        @classmethod
        def from_pretrained(cls, model_id, use_fast=True):
            captured['tok_model_id'] = model_id
            return cls()

    class FakeModel:
        def eval(self):
            captured['eval_called'] = True
            return self

    class FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            captured['model_id'] = model_id
            captured['kwargs'] = kwargs
            return FakeModel()

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_transformers = types.ModuleType('transformers')
    fake_transformers.AutoTokenizer = FakeTokenizer
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    fake_transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig

    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'transformers', fake_transformers)
    monkeypatch.setenv('AGENTLAB_DEVICE', 'cuda')
    monkeypatch.setenv('AGENTLAB_LOCAL_QUANT', '4bit')
    monkeypatch.setenv('AGENTLAB_ATTN_IMPL', 'flash_attention_2')
    monkeypatch.setenv('AGENTLAB_LOCAL_DTYPE', 'bf16')
    monkeypatch.setenv('AGENTLAB_ENABLE_OFFLOAD', '1')
    monkeypatch.setenv('AGENTLAB_OFFLOAD_DIR', '/tmp/offload-test')
    monkeypatch.setenv('AGENTLAB_LOCAL_MAX_MEMORY', 'cuda:6GiB,cpu:24GiB')
    monkeypatch.setenv('AGENTLAB_TORCH_COMPILE', '1')

    tok, model = inference._local_transformers_load('local-demo-model')
    assert tok is not None and model is not None
    assert captured['kwargs']['attn_implementation'] == 'flash_attention_2'
    assert captured['kwargs']['torch_dtype'] == 'bfloat16'
    assert captured['kwargs']['device_map'] == 'auto'
    assert captured['kwargs']['offload_folder'] == '/tmp/offload-test'
    assert captured['kwargs']['max_memory'] == {'cuda': '6GiB', 'cpu': '24GiB'}
    assert captured['compile_mode'] == 'reduce-overhead'
    assert captured['kwargs']['quantization_config'].kwargs['load_in_4bit'] is True


class _FakeTensor:
    def __init__(self, values=None, device='cpu'):
        self.values = values or [1, 2, 3]
        self.device = device

    def to(self, device):
        return _FakeTensor(self.values, device=device)

    @property
    def shape(self):
        return (1, len(self.values))


class _FakeInferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeModelForChat:
    def parameters(self):
        yield _FakeTensor(device='cpu')

    def generate(self, **kwargs):
        return [[1, 2, 3, 4, 5]]


def test_local_transformers_chat_uses_input_truncation(monkeypatch):
    fake_torch = types.ModuleType('torch')
    fake_torch.inference_mode = lambda: _FakeInferenceMode()
    class FakeCuda:
        @staticmethod
        def is_available():
            return False
    fake_torch.cuda = FakeCuda()
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)

    calls = {}

    class FakeTokenizer:
        eos_token_id = 99
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return 'PROMPT'
        def __call__(self, text, **kwargs):
            calls['tok_kwargs'] = kwargs
            return {'input_ids': _FakeTensor([1, 2, 3])}
        def decode(self, tokens, skip_special_tokens=True):
            return 'decoded'

    monkeypatch.setattr(inference, '_local_transformers_load', lambda model_id: (FakeTokenizer(), _FakeModelForChat()))
    monkeypatch.setenv('AGENTLAB_LOCAL_MAX_INPUT_TOKENS', '128')

    out = inference._local_transformers_chat('local-demo', 'user prompt', 'system prompt')
    assert out == 'decoded'
    assert calls['tok_kwargs']['truncation'] is True
    assert calls['tok_kwargs']['max_length'] == 128
