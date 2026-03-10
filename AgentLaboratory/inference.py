import time
import asyncio
import os
import json
import math
import gc
import io
import queue
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from collections import OrderedDict

try:
    import ctypes
except Exception:
    ctypes = None  # type: ignore

# ------------------------------
# Optional local (on-device) LLM
# ------------------------------
#
# By default this repo uses g4f (provider endpoints) or API-backed models.
# Those calls run remotely and will not use your local GPU.
#
# If you want *local* inference that can use CUDA, pass model strings like:
#   local:Qwen/Qwen2.5-0.5B-Instruct
#
# This backend is intentionally lightweight and only activates when requested.

_LOCAL_LM_CACHE = OrderedDict()


def _best_effort_release_memory(clear_local_cache: bool = False) -> None:
    """Best-effort RAM/VRAM cleanup for long-running notebook/Colab sessions."""
    if clear_local_cache:
        try:
            _LOCAL_LM_CACHE.clear()
        except Exception:
            pass

    try:
        gc.collect()
    except Exception:
        pass

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass

    if ctypes is not None:
        try:
            libc = ctypes.CDLL("libc.so.6")
            if hasattr(libc, "malloc_trim"):
                libc.malloc_trim(0)
        except Exception:
            pass


def _get_agentlab_device() -> str:
    """Resolve the intended compute device for local inference."""
    dev = (os.getenv("AGENTLAB_DEVICE") or "").strip()
    if dev:
        return dev
    use_gpu = (os.getenv("AGENTLAB_USE_GPU") or "").strip()
    if use_gpu.lower() in {"1", "true", "yes", "on"}:
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    return "cpu"


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.getenv(name) or str(default)).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float((os.getenv(name) or str(default)).strip())
    except Exception:
        return default


def _local_cache_key(model_id: str):
    device = _get_agentlab_device()
    return (
        model_id,
        device,
        (os.getenv("AGENTLAB_LOCAL_QUANT") or "none").strip().lower(),
        (os.getenv("AGENTLAB_ATTN_IMPL") or "").strip().lower(),
        (os.getenv("AGENTLAB_LOCAL_DTYPE") or "").strip().lower(),
        (os.getenv("AGENTLAB_LOCAL_MAX_MEMORY") or "").strip(),
        (os.getenv("AGENTLAB_OFFLOAD_DIR") or "").strip(),
        (os.getenv("AGENTLAB_TORCH_COMPILE") or "0").strip(),
    )


def _local_cache_get(key):
    ttl_s = _env_int("AGENTLAB_LOCAL_CACHE_TTL_S", 3600)
    try:
        item = _LOCAL_LM_CACHE.pop(key)
    except KeyError:
        return None
    value, ts = item
    if ttl_s > 0 and (time.time() - ts) > ttl_s:
        try:
            del value
        except Exception:
            pass
        return None
    _LOCAL_LM_CACHE[key] = (value, ts)
    return value


def _local_cache_set(key, value) -> None:
    max_items = max(1, _env_int("AGENTLAB_LOCAL_CACHE_MAX_ITEMS", 1))
    try:
        _LOCAL_LM_CACHE.pop(key, None)
        _LOCAL_LM_CACHE[key] = (value, time.time())
        while len(_LOCAL_LM_CACHE) > max_items:
            _, (old_value, _) = _LOCAL_LM_CACHE.popitem(last=False)
            try:
                del old_value
            except Exception:
                pass
    except Exception:
        _LOCAL_LM_CACHE[key] = (value, time.time())


def _resolve_torch_dtype(torch_mod, device: str):
    raw = (os.getenv("AGENTLAB_LOCAL_DTYPE") or "").strip().lower()
    if raw in {"float16", "fp16", "half"}:
        return getattr(torch_mod, "float16", None)
    if raw in {"bfloat16", "bf16"}:
        return getattr(torch_mod, "bfloat16", None)
    if raw in {"float32", "fp32"}:
        return getattr(torch_mod, "float32", None)
    if device.startswith("cuda"):
        return getattr(torch_mod, "float16", None)
    return None


def _parse_max_memory() -> dict | None:
    raw = (os.getenv("AGENTLAB_LOCAL_MAX_MEMORY") or "").strip()
    if not raw:
        return None
    out = {}
    for part in raw.split(','):
        if ':' not in part:
            continue
        name, value = part.split(':', 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            continue
        out[name] = value
    return out or None


def _local_transformers_load(model_id: str):
    """Load and cache a HF Transformers CausalLM model (lazy import)."""
    key = _local_cache_key(model_id)
    cached = _local_cache_get(key)
    if cached is not None:
        return cached

    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as e:
        raise ImportError(
            "Local inference requires 'torch' and 'transformers'. "
            "Install AgentLaboratory/requirements.txt (or at least torch+transformers+accelerate)."
        ) from e

    device = _get_agentlab_device()
    quant = (os.getenv("AGENTLAB_LOCAL_QUANT") or "none").strip().lower()
    attn_impl = (os.getenv("AGENTLAB_ATTN_IMPL") or "sdpa").strip().lower()
    offload_dir = (os.getenv("AGENTLAB_OFFLOAD_DIR") or "./.offload").strip()
    offload_enabled = _env_truthy("AGENTLAB_ENABLE_OFFLOAD", default=False) or bool((os.getenv("AGENTLAB_LOCAL_MAX_MEMORY") or "").strip())

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if getattr(tok, 'pad_token_id', None) is None and getattr(tok, 'eos_token_id', None) is not None:
        try:
            tok.pad_token_id = tok.eos_token_id
        except Exception:
            pass

    model_kwargs = {"low_cpu_mem_usage": True}

    torch_dtype = _resolve_torch_dtype(torch, device)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    if attn_impl and attn_impl not in {"auto", "default"}:
        model_kwargs["attn_implementation"] = attn_impl

    if quant in {"8bit", "int8", "8"}:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
        except Exception as e:
            raise ImportError(
                "AGENTLAB_LOCAL_QUANT=8bit requires bitsandbytes support in transformers. "
                "Install bitsandbytes (GPU Linux) and a recent transformers build."
            ) from e
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quant in {"4bit", "nf4", "4"}:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
        except Exception as e:
            raise ImportError(
                "AGENTLAB_LOCAL_QUANT=4bit requires bitsandbytes support in transformers. "
                "Install bitsandbytes (GPU Linux) and a recent transformers build."
            ) from e
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype or getattr(torch, "float16", None),
        )

    max_memory = _parse_max_memory()
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory

    if device.startswith("cuda"):
        model_kwargs["device_map"] = "auto"
        if offload_enabled:
            model_kwargs["offload_folder"] = offload_dir
            model_kwargs["offload_state_dict"] = _env_truthy("AGENTLAB_OFFLOAD_STATE_DICT", default=True)
            model_kwargs["offload_buffers"] = _env_truthy("AGENTLAB_OFFLOAD_BUFFERS", default=False)
    else:
        model_kwargs["device_map"] = {"": "cpu"}

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    if _env_truthy("AGENTLAB_TORCH_COMPILE", default=False):
        compile_mode = (os.getenv("AGENTLAB_TORCH_COMPILE_MODE") or "reduce-overhead").strip()
        try:
            model = torch.compile(model, mode=compile_mode)
        except Exception:
            pass

    value = (tok, model)
    _local_cache_set(key, value)
    return value


def _local_transformers_chat(model_id: str, prompt: str, system_prompt: str, temp=None, timeout: float = 20.0) -> str:
    """Run a small local chat completion with Transformers."""

    tok, model = _local_transformers_load(model_id)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    if hasattr(tok, "apply_chat_template"):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{prompt}\n\nASSISTANT:\n"

    max_new_tokens = _env_int("AGENTLAB_LOCAL_MAX_NEW_TOKENS", 768)
    temperature = _env_float("AGENTLAB_LOCAL_TEMPERATURE", float(temp if temp is not None else 0.2))
    max_input_tokens = _env_int("AGENTLAB_LOCAL_MAX_INPUT_TOKENS", 0)

    inputs = None
    out = None
    gen = None
    answer = ""
    try:
        import torch  # type: ignore

        tok_kwargs = {"return_tensors": "pt"}
        if max_input_tokens > 0:
            tok_kwargs["truncation"] = True
            tok_kwargs["max_length"] = max_input_tokens
        inputs = tok(text, **tok_kwargs)
        # With device_map="auto", model spans devices; push inputs to the first param device.
        try:
            first_param = next(model.parameters())
            inputs = {k: v.to(first_param.device) for k, v in inputs.items()}
        except Exception:
            pass

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature,
                pad_token_id=getattr(tok, "eos_token_id", None),
                eos_token_id=getattr(tok, "eos_token_id", None),
            )

        gen = out[0]
        prompt_len = inputs["input_ids"].shape[-1]
        answer = tok.decode(gen[prompt_len:], skip_special_tokens=True).strip()
        return answer
    finally:
        del inputs, out, gen, messages, text, answer
        _best_effort_release_memory(clear_local_cache=False)

def local_model_runtime_config() -> dict:
    """Return the active local-model memory knobs for debugging / tests."""
    return {
        "device": _get_agentlab_device(),
        "quant": (os.getenv("AGENTLAB_LOCAL_QUANT") or "none").strip().lower(),
        "attn_impl": (os.getenv("AGENTLAB_ATTN_IMPL") or "sdpa").strip().lower(),
        "dtype": (os.getenv("AGENTLAB_LOCAL_DTYPE") or "auto").strip().lower() or "auto",
        "offload_enabled": _env_truthy("AGENTLAB_ENABLE_OFFLOAD", default=False) or bool((os.getenv("AGENTLAB_LOCAL_MAX_MEMORY") or "").strip()),
        "offload_dir": (os.getenv("AGENTLAB_OFFLOAD_DIR") or "./.offload").strip(),
        "torch_compile": _env_truthy("AGENTLAB_TORCH_COMPILE", default=False),
        "max_input_tokens": _env_int("AGENTLAB_LOCAL_MAX_INPUT_TOKENS", 0),
        "cache_max_items": _env_int("AGENTLAB_LOCAL_CACHE_MAX_ITEMS", 1),
        "cache_ttl_s": _env_int("AGENTLAB_LOCAL_CACHE_TTL_S", 3600),
        "max_memory": _parse_max_memory(),
    }

# Optional dependencies: keep g4f-only usage lightweight.
try:
    import openai  # type: ignore
    from openai import OpenAI  # type: ignore
except Exception:
    openai = None  # type: ignore
    OpenAI = None  # type: ignore

try:
    import anthropic  # type: ignore
except Exception:
    anthropic = None  # type: ignore

genai = None  # type: ignore
_TIKTOKEN_UNAVAILABLE = False
_G4F_MODULE = None
_G4F_IMPORT_ERROR = None
_G4F_ASYNC_CLIENT_CLASS = None
_G4F_ASYNC_CLIENT_IMPORT_ERROR = None


def _load_g4f_module():
    global _G4F_MODULE, _G4F_IMPORT_ERROR
    if _G4F_MODULE is not None:
        return _G4F_MODULE
    if _G4F_IMPORT_ERROR is not None:
        return None
    try:
        import g4f  # type: ignore
        _G4F_MODULE = g4f
        return _G4F_MODULE
    except Exception:
        try:
            _ROOT = Path(__file__).resolve().parents[1]
            _VENDOR = _ROOT / "gpt4free"
            if _VENDOR.exists() and str(_VENDOR) not in sys.path:
                sys.path.insert(0, str(_VENDOR))
            import g4f  # type: ignore
            _G4F_MODULE = g4f
            return _G4F_MODULE
        except Exception as exc:
            _G4F_IMPORT_ERROR = exc
            return None

def _load_g4f_async_client_class():
    global _G4F_ASYNC_CLIENT_CLASS, _G4F_ASYNC_CLIENT_IMPORT_ERROR
    if _G4F_ASYNC_CLIENT_CLASS is not None:
        return _G4F_ASYNC_CLIENT_CLASS
    if _G4F_ASYNC_CLIENT_IMPORT_ERROR is not None:
        return None
    try:
        from g4f.client import AsyncClient  # type: ignore
        _G4F_ASYNC_CLIENT_CLASS = AsyncClient
        return _G4F_ASYNC_CLIENT_CLASS
    except Exception:
        try:
            _ROOT = Path(__file__).resolve().parents[1]
            _VENDOR = _ROOT / "gpt4free"
            if _VENDOR.exists() and str(_VENDOR) not in sys.path:
                sys.path.insert(0, str(_VENDOR))
            from g4f.client import AsyncClient  # type: ignore
            _G4F_ASYNC_CLIENT_CLASS = AsyncClient
            return _G4F_ASYNC_CLIENT_CLASS
        except Exception as exc:
            _G4F_ASYNC_CLIENT_IMPORT_ERROR = exc
            return None


def _g4f_api_key_from_env() -> str | None:
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "TOGETHER_API_KEY", "GEMINI_API_KEY"):
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return None


def _g4f_async_enabled() -> bool:
    return _env_truthy("AGENTLAB_G4F_USE_ASYNC", default=True)


def _g4f_async_stream_enabled() -> bool:
    return _env_truthy("AGENTLAB_G4F_ASYNC_STREAM", default=False)


async def _async_iter_stream_with_timeouts(resp, *, total_timeout_s: float | None = None, idle_timeout_s: float | None = None):
    iterator = resp.__aiter__()
    deadline = time.monotonic() + float(total_timeout_s) if total_timeout_s is not None and total_timeout_s > 0 else None
    while True:
        wait_s = None
        if idle_timeout_s is not None and idle_timeout_s > 0:
            wait_s = float(idle_timeout_s)
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait_s = remaining if wait_s is None else min(wait_s, remaining)
        try:
            if wait_s is None:
                chunk = await iterator.__anext__()
            else:
                chunk = await asyncio.wait_for(iterator.__anext__(), timeout=max(0.01, wait_s))
        except (StopAsyncIteration, asyncio.TimeoutError):
            break
        yield chunk


async def _g4f_async_to_text(
    resp,
    *,
    max_chars: int | None = None,
    stop_at_python_fence: bool = False,
    stream_timeout_s: float | None = None,
    stream_idle_timeout_s: float | None = None,
) -> str:
    if isinstance(resp, str):
        text = resp[:max_chars] if (max_chars is not None and max_chars > 0) else resp
        if stop_at_python_fence:
            text = _trim_after_python_fence(text)
        return text

    if resp is None:
        return ""

    if hasattr(resp, "choices") and getattr(resp, "choices", None):
        try:
            message = resp.choices[0].message
            content = getattr(message, "content", None)
            if isinstance(content, str):
                text = content[:max_chars] if (max_chars is not None and max_chars > 0) else content
                if stop_at_python_fence:
                    text = _trim_after_python_fence(text)
                return text
        except Exception:
            pass

    if not hasattr(resp, "__aiter__"):
        text = _chunk_to_text(resp)
        if max_chars is not None and max_chars > 0:
            text = text[:max_chars]
        if stop_at_python_fence:
            text = _trim_after_python_fence(text)
        return text

    buf = io.StringIO()
    size = 0
    window = ""
    async for ch in _async_iter_stream_with_timeouts(
        resp,
        total_timeout_s=stream_timeout_s,
        idle_timeout_s=stream_idle_timeout_s,
    ):
        txt = _chunk_to_text(ch)
        if not txt:
            continue
        if max_chars is not None and max_chars > 0:
            remaining = max_chars - size
            if remaining <= 0:
                break
            if len(txt) > remaining:
                txt = txt[:remaining]
        buf.write(txt)
        size += len(txt)
        if stop_at_python_fence:
            window = (window + txt)[-4096:]
            lowered = window.lower()
            open_idx = lowered.find("```python")
            if open_idx != -1:
                close_idx = lowered.rfind("```")
                if close_idx > open_idx:
                    break
        if max_chars is not None and max_chars > 0 and size >= max_chars:
            break

    text = buf.getvalue()
    if stop_at_python_fence:
        return _trim_after_python_fence(text)
    return text


def _run_coro_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    holder = {}

    def _runner():
        try:
            holder["result"] = asyncio.run(coro)
        except BaseException as exc:
            holder["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True, name="agentlab-async-runner")
    thread.start()
    thread.join()
    if "error" in holder:
        raise holder["error"]
    return holder.get("result")


async def _g4f_async_create_text(
    *,
    model_str: str,
    messages: list[dict],
    provider_name: str | None,
    timeout_s: float,
    stream_timeout_s: float | None,
    stream_idle_timeout_s: float | None,
    max_resp_chars: int | None,
    stop_at_python_fence: bool,
) -> str:
    AsyncClient = _load_g4f_async_client_class()
    if AsyncClient is None:
        raise RuntimeError(f"g4f AsyncClient is unavailable: {_G4F_ASYNC_CLIENT_IMPORT_ERROR}")

    client_kwargs = {}
    api_key = _g4f_api_key_from_env()
    if api_key:
        client_kwargs["api_key"] = api_key
    client = AsyncClient(**client_kwargs)

    request_kwargs = {
        "model": model_str,
        "messages": messages,
        "provider": provider_name or None,
        "web_search": False,
        "stream": _g4f_async_stream_enabled(),
    }
    max_tokens = _env_int("AGENTLAB_G4F_MAX_TOKENS", 0)
    if max_tokens > 0:
        request_kwargs["max_tokens"] = max_tokens

    response = await asyncio.wait_for(
        client.chat.completions.create(**request_kwargs),
        timeout=max(1.0, float(timeout_s)),
    )
    text = await _g4f_async_to_text(
        response,
        max_chars=max_resp_chars,
        stop_at_python_fence=stop_at_python_fence,
        stream_timeout_s=stream_timeout_s if request_kwargs["stream"] else None,
        stream_idle_timeout_s=stream_idle_timeout_s if request_kwargs["stream"] else None,
    )
    return text.strip() if isinstance(text, str) else ""



def _get_tiktoken_module():
    global _TIKTOKEN_UNAVAILABLE
    if _TIKTOKEN_UNAVAILABLE:
        return None
    try:
        import tiktoken  # type: ignore
        return tiktoken
    except Exception:
        _TIKTOKEN_UNAVAILABLE = True
        return None


def _chunk_to_text(chunk) -> str:
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        for key in ("content", "text"):
            val = chunk.get(key)
            if isinstance(val, str):
                return val
        try:
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str):
                    return content
        except Exception:
            return ""
        return ""
    for attr in ("content", "text"):
        try:
            val = getattr(chunk, attr)
            if isinstance(val, str):
                return val
        except Exception:
            pass
    try:
        choices = getattr(chunk, "choices")
        if choices:
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None)
            if isinstance(content, str):
                return content
    except Exception:
        pass
    return ""


def _iter_stream_in_background(resp, out_q) -> None:
    try:
        for ch in resp:
            out_q.put(("chunk", ch))
    except BaseException as exc:  # pragma: no cover - defensive against provider quirks
        out_q.put(("error", exc))
    finally:
        out_q.put(("done", None))


def _iter_stream_with_timeouts(resp, *, total_timeout_s: float | None = None, idle_timeout_s: float | None = None):
    if total_timeout_s is None and idle_timeout_s is None:
        yield from resp
        return

    out_q = queue.Queue(maxsize=128)
    producer = threading.Thread(
        target=_iter_stream_in_background,
        args=(resp, out_q),
        daemon=True,
        name="agentlab-g4f-stream",
    )
    producer.start()

    deadline = time.monotonic() + float(total_timeout_s) if total_timeout_s is not None and total_timeout_s > 0 else None
    idle_wait = float(idle_timeout_s) if idle_timeout_s is not None and idle_timeout_s > 0 else None

    while True:
        wait_s = idle_wait
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait_s = remaining if wait_s is None else min(wait_s, remaining)

        try:
            if wait_s is None:
                kind, payload = out_q.get()
            else:
                kind, payload = out_q.get(timeout=max(0.01, wait_s))
        except queue.Empty:
            break

        if kind == "chunk":
            yield payload
            continue
        if kind in {"done", "error"}:
            break


def _g4f_to_text(
    resp,
    *,
    max_chars: int | None = None,
    stop_at_python_fence: bool = False,
    stream_timeout_s: float | None = None,
    stream_idle_timeout_s: float | None = None,
) -> str:
    """g4f may return a string or an iterator (stream). Convert to string safely.

    For streamed providers we optionally consume chunks from a background thread with
    wall-clock and idle deadlines so a provider that stalls after opening the stream
    cannot block the whole pipeline forever. For codegen workloads we optionally stop
    right after a completed ```python fenced block arrives, which avoids reading long
    trailing explanations into RAM.
    """
    if isinstance(resp, str):
        text = resp[:max_chars] if (max_chars is not None and max_chars > 0) else resp
        if stop_at_python_fence:
            text = _trim_after_python_fence(text)
        return text
    if resp is None or not hasattr(resp, "__iter__"):
        text = _chunk_to_text(resp)
        if max_chars is not None and max_chars > 0:
            text = text[:max_chars]
        if stop_at_python_fence:
            text = _trim_after_python_fence(text)
        return text

    iterable = _iter_stream_with_timeouts(
        resp,
        total_timeout_s=stream_timeout_s,
        idle_timeout_s=stream_idle_timeout_s,
    )

    buf = io.StringIO()
    size = 0
    window = ""
    try:
        for ch in iterable:
            txt = _chunk_to_text(ch)
            if not txt:
                continue
            if max_chars is not None and max_chars > 0:
                remaining = max_chars - size
                if remaining <= 0:
                    break
                if len(txt) > remaining:
                    txt = txt[:remaining]
            buf.write(txt)
            size += len(txt)
            if stop_at_python_fence:
                window = (window + txt)[-4096:]
                lowered = window.lower()
                open_idx = lowered.find("```python")
                if open_idx != -1:
                    close_idx = lowered.rfind("```")
                    if close_idx > open_idx:
                        break
            if max_chars is not None and max_chars > 0 and size >= max_chars:
                break
        text = buf.getvalue()
        if stop_at_python_fence:
            return _trim_after_python_fence(text)
        return text
    except Exception:
        text = buf.getvalue()
        if stop_at_python_fence:
            return _trim_after_python_fence(text)
        return text


def _trim_after_python_fence(text: str) -> str:
    if not text:
        return text
    lowered = text.lower()
    open_idx = lowered.find("```python")
    if open_idx == -1:
        return text
    close_idx = lowered.find("```", open_idx + len("```python"))
    if close_idx == -1:
        return text
    return text[: close_idx + 3]


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_remote_model(model_str: str | None) -> bool:
    return not str(model_str or "").strip().startswith("local:")


def _use_remote_subprocess_isolation(model_str: str | None) -> bool:
    if not _is_remote_model(model_str):
        return False
    return _env_truthy("AGENTLAB_REMOTE_SUBPROCESS", default=True)


def _g4f_supports_stream_flag(g4f_mod=None) -> bool:
    g4f_mod = g4f_mod or _load_g4f_module()
    if g4f_mod is None:
        return False
    try:
        import inspect

        sig = inspect.signature(g4f_mod.ChatCompletion.create)  # type: ignore[attr-defined]
        if "stream" in sig.parameters:
            return True
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except Exception:
        return True


def _should_count_tokens(using_g4f_backend: bool, *, total_text_chars: int = 0) -> bool:
    if _get_tiktoken_module() is None:
        return False
    if _env_truthy("AGENTLAB_DISABLE_TOKEN_COUNT", default=False):
        return False
    if using_g4f_backend and not _env_truthy("AGENTLAB_ENABLE_TOKEN_COUNT_FOR_G4F", default=False):
        return False
    max_chars = _env_int("AGENTLAB_TOKEN_COUNT_MAX_CHARS", 12000)
    if max_chars > 0 and total_text_chars > max_chars:
        return False
    return True


def _tiktoken_encoding_for_model(model_str: str):
    tiktoken_mod = _get_tiktoken_module()
    if tiktoken_mod is None:
        return None
    if model_str in ["o1-preview", "o1-mini", "claude-3.5-sonnet", "o1", "o3-mini"]:
        return tiktoken_mod.encoding_for_model("gpt-4o")
    if model_str in ["deepseek-chat"]:
        return tiktoken_mod.get_encoding("cl100k_base")
    try:
        return tiktoken_mod.encoding_for_model(model_str)
    except Exception:
        try:
            return tiktoken_mod.get_encoding("cl100k_base")
        except Exception:
            return None


def _read_text_tail(path: Path, max_chars: int = 8000) -> str:
    try:
        size = path.stat().st_size
    except Exception:
        return ""
    if size <= 0:
        return ""
    max_bytes = max(1024, int(max_chars * 4))
    try:
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
            data = fh.read()
        text = data.decode("utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[-max_chars:]
        return text.strip()
    except Exception:
        return ""


def _remote_worker_per_attempt_budget(timeout: float, *, model: str | None = None) -> float:
    base_timeout = max(1.0, float(timeout))
    if not _is_remote_model(model):
        return base_timeout
    stream_timeout_s = _env_float("AGENTLAB_G4F_STREAM_TIMEOUT_S", max(3.0, base_timeout + 5.0))
    idle_timeout_s = _env_float("AGENTLAB_G4F_STREAM_IDLE_TIMEOUT_S", max(5.0, min(15.0, base_timeout)))
    budget = base_timeout
    if stream_timeout_s > 0:
        budget = max(budget, float(stream_timeout_s))
    if idle_timeout_s > 0:
        budget = max(budget, float(idle_timeout_s))
    return budget


def _remote_worker_timeout_s(*, tries: int, timeout: float, model: str | None = None) -> int:
    explicit = _env_float("AGENTLAB_REMOTE_WORKER_TIMEOUT_S", 0.0)
    if explicit > 0:
        return max(30, int(math.ceil(explicit)))
    attempts = max(1, int(tries))
    per_attempt_budget = _remote_worker_per_attempt_budget(timeout, model=model)
    per_attempt_buffer = max(5.0, _env_float("AGENTLAB_REMOTE_WORKER_ATTEMPT_BUFFER_S", 5.0))
    startup_buffer = max(10.0, _env_float("AGENTLAB_REMOTE_WORKER_STARTUP_BUFFER_S", 10.0))
    total_budget = startup_buffer + attempts * (per_attempt_budget + per_attempt_buffer)
    return max(30, int(math.ceil(total_budget)))


def _worker_log_excerpt(stdout_path: Path, stderr_path: Path, *, max_chars: int = 8000) -> str:
    parts = []
    err_tail = _read_text_tail(stderr_path, max_chars=max_chars // 2)
    out_tail = _read_text_tail(stdout_path, max_chars=max_chars // 2)
    if err_tail:
        parts.append(f"stderr tail:\n{err_tail}")
    if out_tail:
        parts.append(f"stdout tail:\n{out_tail}")
    return "\n\n".join(parts).strip()


def _worker_kill_process_group_enabled() -> bool:
    return _env_truthy("AGENTLAB_WORKER_KILL_PROCESS_GROUP", default=True)


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    kill_group = _worker_kill_process_group_enabled()
    if os.name != "nt" and kill_group:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=2)
            return
        except Exception:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    else:
        try:
            proc.terminate()
            proc.wait(timeout=2)
            return
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def _run_json_worker_subprocess(
    *,
    cmd: list[str],
    env: dict[str, str],
    proc_timeout: int,
    out_json: Path,
    tmpdir_path: Path,
    model_label: str,
):
    stdout_path = tmpdir_path / "worker_stdout.log"
    stderr_path = tmpdir_path / "worker_stderr.log"
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_f, stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_f:
        popen_kwargs = {
            "stdout": stdout_f,
            "stderr": stderr_f,
            "env": env,
        }
        if os.name != "nt" and _worker_kill_process_group_enabled():
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(cmd, **popen_kwargs)
        try:
            proc.wait(timeout=proc_timeout)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_tree(proc)
            excerpt = _worker_log_excerpt(stdout_path, stderr_path)
            detail = f" {excerpt}" if excerpt else ""
            raise RuntimeError(f"{model_label}: remote worker timed out after {proc_timeout}s.{detail}".strip()) from exc

    if not out_json.exists():
        excerpt = _worker_log_excerpt(stdout_path, stderr_path)
        detail = f" {excerpt}" if excerpt else ""
        raise RuntimeError(f"{model_label}: remote worker did not produce a result file.{detail}".strip())

    try:
        return json.loads(out_json.read_text(encoding="utf-8"))
    except Exception as exc:
        excerpt = _worker_log_excerpt(stdout_path, stderr_path)
        detail = f" {excerpt}" if excerpt else ""
        raise RuntimeError(f"{model_label}: failed to parse remote worker output ({exc}).{detail}".strip()) from exc


def query_model_stable(
    model_str,
    prompt,
    system_prompt,
    openai_api_key=None,
    gemini_api_key=None,
    anthropic_api_key=None,
    tries=5,
    timeout=20.0,
    temp=None,
    print_cost=True,
    version="1.5",
):
    if not _use_remote_subprocess_isolation(model_str):
        return query_model(
            model_str=model_str,
            prompt=prompt,
            system_prompt=system_prompt,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            anthropic_api_key=anthropic_api_key,
            tries=tries,
            timeout=timeout,
            temp=temp,
            print_cost=print_cost,
            version=version,
        )

    worker_path = Path(__file__).resolve().with_name("query_model_worker.py")
    if not worker_path.exists():
        return query_model(
            model_str=model_str,
            prompt=prompt,
            system_prompt=system_prompt,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            anthropic_api_key=anthropic_api_key,
            tries=tries,
            timeout=timeout,
            temp=temp,
            print_cost=print_cost,
            version=version,
        )

    with tempfile.TemporaryDirectory(prefix="agentlab_query_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        prompt_file = tmpdir_path / "prompt.txt"
        system_file = tmpdir_path / "system.txt"
        out_json = tmpdir_path / "result.json"
        prompt_file.write_text(prompt, encoding="utf-8")
        system_file.write_text(system_prompt, encoding="utf-8")

        cmd = [
            sys.executable,
            str(worker_path),
            "--model",
            str(model_str),
            "--prompt-file",
            str(prompt_file),
            "--system-file",
            str(system_file),
            "--out-json",
            str(out_json),
            "--tries",
            str(int(tries)),
            "--timeout",
            str(float(timeout)),
            "--version",
            str(version),
        ]
        if print_cost:
            cmd.append("--print-cost")
        if temp is not None:
            cmd.extend(["--temp", str(temp)])

        env = dict(os.environ)
        env["AGENTLAB_REMOTE_SUBPROCESS"] = "0"
        proc_timeout = _remote_worker_timeout_s(tries=tries, timeout=timeout, model=str(model_str))
        payload = _run_json_worker_subprocess(
            cmd=cmd,
            env=env,
            proc_timeout=proc_timeout,
            out_json=out_json,
            tmpdir_path=tmpdir_path,
            model_label=str(model_str),
        )
        if payload.get("ok"):
            answer = payload.get("answer", "")
            return answer if isinstance(answer, str) else ""

        error = str(payload.get("error", "") or "").strip()
        error_type = str(payload.get("error_type", "") or "").strip()
        if error_type == "MissingLLMCredentials":
            raise MissingLLMCredentials(error or "credentials required")
        raise RuntimeError(error or f"{model_str}: remote worker failed")


class MissingLLMCredentials(RuntimeError):
    """Raised when the selected backend requires credentials that are not provided."""


_FATAL_AUTH_MARKERS = (
    'Add a "api_key"',
    "MissingAuthError",
    "Add a .har file",
    'Add a "api_key" or a .har file',
)


def _looks_like_missing_auth(err: Exception) -> bool:
    msg = str(err)
    return any(m in msg for m in _FATAL_AUTH_MARKERS)


TOKENS_IN = dict()
TOKENS_OUT = dict()


def curr_cost_est():
    costmap_in = {
        "gpt-4o": 2.50 / 1000000,
        "gpt-4o-mini": 0.150 / 1000000,
        "o1-preview": 15.00 / 1000000,
        "o1-mini": 3.00 / 1000000,
        "claude-3-5-sonnet": 3.00 / 1000000,
        "deepseek-chat": 1.00 / 1000000,
        "o1": 15.00 / 1000000,
        "o3-mini": 1.10 / 1000000,
    }
    costmap_out = {
        "gpt-4o": 10.00/ 1000000,
        "gpt-4o-mini": 0.6 / 1000000,
        "o1-preview": 60.00 / 1000000,
        "o1-mini": 12.00 / 1000000,
        "claude-3-5-sonnet": 12.00 / 1000000,
        "deepseek-chat": 5.00 / 1000000,
        "o1": 60.00 / 1000000,
        "o3-mini": 4.40 / 1000000,
    }
    return sum([costmap_in[_]*TOKENS_IN[_] for _ in TOKENS_IN]) + sum([costmap_out[_]*TOKENS_OUT[_] for _ in TOKENS_OUT])

def query_model(
    model_str,
    prompt,
    system_prompt,
    openai_api_key=None,
    gemini_api_key=None,
    anthropic_api_key=None,
    tries=5,
    timeout=20.0,
    temp=None,
    print_cost=True,
    version="1.5",
):
    """Query an LLM backend.

    Robustness notes:
    - g4f providers can be slow/unstable. The previous default timeout (5s) caused frequent
      false timeouts. We default to 20s.
    - You can override retries/timeouts via env vars without touching code:
        AGENTLAB_TRIES=3 AGENTLAB_TIMEOUT=60
      (G4F_TRIES/G4F_TIMEOUT are supported as aliases)
    """

    # Allow env overrides (useful in Colab / CI where providers are flaky)
    try:
        tries = int(os.getenv("AGENTLAB_TRIES", os.getenv("G4F_TRIES", str(tries))))
    except Exception:
        pass
    try:
        timeout = float(os.getenv("AGENTLAB_TIMEOUT", os.getenv("G4F_TIMEOUT", str(timeout))))
    except Exception:
        pass
    # Allow a simple "fallback list" syntax: model_a|model_b|model_c
    if isinstance(model_str, str) and ("|" in model_str):
        last_err = None
        for _m in [x.strip() for x in model_str.split("|") if x.strip()]:
            try:
                ans = query_model(_m, prompt, system_prompt, openai_api_key=openai_api_key,
                                  gemini_api_key=gemini_api_key, anthropic_api_key=anthropic_api_key,
                                  tries=tries, timeout=timeout, temp=temp, print_cost=print_cost, version=version)
                if ans:
                    return ans
            except Exception as e:
                last_err = e
                continue
        if last_err is not None:
            raise last_err
        raise Exception("No model produced an answer (fallback list empty or all failed).")

    # --- Local transformers backend ---
    # Use: local:<hf_model_id>
    if isinstance(model_str, str) and model_str.startswith("local:"):
        model_id = model_str.split(":", 1)[1].strip() or os.getenv(
            "AGENTLAB_LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
        )
        return _local_transformers_chat(model_id, prompt, system_prompt, temp=temp, timeout=timeout)

    # If prefixed with 'g4f:' we force GPT4Free backend (no API key needed)
    force_g4f = isinstance(model_str, str) and model_str.startswith("g4f:")
    if force_g4f:
        model_str = model_str.split(":", 1)[1].strip()
    preloaded_api = os.getenv('OPENAI_API_KEY')
    if openai_api_key is None and preloaded_api is not None:
        openai_api_key = preloaded_api
    if openai_api_key is not None and openai is None:
        raise ImportError("openai package is required for OpenAI-backed models; install it or use g4f:")
    if anthropic_api_key is not None and anthropic is None:
        raise ImportError("anthropic package is required for Claude-backed models")
    # Gemini SDK is imported lazily only if a Gemini model is actually requested.
    g4f_mod = _load_g4f_module()
    if openai_api_key is None and anthropic_api_key is None and gemini_api_key is None and g4f_mod is None:
        raise Exception("No API key provided and g4f is not available in query_model")
    if openai_api_key is not None:
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if anthropic_api_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    if gemini_api_key is not None:
        os.environ["GEMINI_API_KEY"] = gemini_api_key
    for _ in range(tries):
        try:            # --- g4f backend ---
            if force_g4f or (g4f_mod is not None and openai_api_key is None and anthropic_api_key is None and gemini_api_key is None):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                kwargs = {}
                provider_name = os.getenv("G4F_PROVIDER", "").strip() or None
                max_resp_chars = _env_int("AGENTLAB_MAX_RESPONSE_CHARS", 60000)
                if max_resp_chars <= 0:
                    max_resp_chars = None
                stop_at_python_fence = _env_truthy("AGENTLAB_G4F_STOP_AT_PYTHON_FENCE", default=False)
                stream_timeout_s = _env_float("AGENTLAB_G4F_STREAM_TIMEOUT_S", max(3.0, float(timeout) + 5.0))
                stream_idle_timeout_s = _env_float("AGENTLAB_G4F_STREAM_IDLE_TIMEOUT_S", max(5.0, min(15.0, float(timeout))))
                request_timeout_s = _env_float(
                    "AGENTLAB_G4F_REQUEST_TIMEOUT_S",
                    max(float(timeout), _remote_worker_per_attempt_budget(timeout, model=model_str)),
                )
                if stream_timeout_s <= 0:
                    stream_timeout_s = None
                if stream_idle_timeout_s <= 0:
                    stream_idle_timeout_s = None
                if request_timeout_s <= 0:
                    request_timeout_s = max(1.0, float(timeout))

                if _g4f_async_enabled():
                    try:
                        answer = _run_coro_sync(
                            _g4f_async_create_text(
                                model_str=model_str,
                                messages=messages,
                                provider_name=provider_name,
                                timeout_s=request_timeout_s,
                                stream_timeout_s=stream_timeout_s,
                                stream_idle_timeout_s=stream_idle_timeout_s,
                                max_resp_chars=max_resp_chars,
                                stop_at_python_fence=stop_at_python_fence,
                            )
                        )
                        if isinstance(answer, str):
                            answer = answer.strip()
                        if answer:
                            _best_effort_release_memory(clear_local_cache=False)
                            return answer
                    except Exception:
                        if not _env_truthy("AGENTLAB_G4F_ASYNC_FALLBACK_TO_SYNC", default=True):
                            raise

                # Some g4f providers require credentials (API key or .har). If the user provided
                # one via env vars, pass it through (only if supported by this installed g4f).
                try:
                    import inspect
                    sig = inspect.signature(g4f_mod.ChatCompletion.create)  # type: ignore
                    if "api_key" in sig.parameters:
                        api_key = _g4f_api_key_from_env()
                        if api_key:
                            kwargs["api_key"] = api_key
                except Exception:
                    pass

                if provider_name:
                    try:
                        prov = getattr(g4f_mod.Provider, provider_name)  # type: ignore
                        kwargs["provider"] = prov
                    except Exception:
                        kwargs["provider"] = provider_name
                if _g4f_supports_stream_flag(g4f_mod):
                    kwargs["stream"] = True
                resp = g4f_mod.ChatCompletion.create(
                    model=model_str,
                    messages=messages,
                    # Keep the provider request timeout aligned with the outer stream/worker budget.
                    timeout=int(max(1, math.ceil(request_timeout_s))),
                    **kwargs,
                )
                answer = _g4f_to_text(
                    resp,
                    max_chars=max_resp_chars,
                    stop_at_python_fence=stop_at_python_fence,
                    stream_timeout_s=stream_timeout_s,
                    stream_idle_timeout_s=stream_idle_timeout_s,
                )
                if isinstance(answer, str):
                    answer = answer.strip()
                if answer:
                    _best_effort_release_memory(clear_local_cache=False)
                    return answer

            if model_str == "gpt-4o-mini" or model_str == "gpt4omini" or model_str == "gpt-4omini" or model_str == "gpt4o-mini":
                model_str = "gpt-4o-mini"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp
                        )
                else:
                    client = OpenAI()
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content

            elif model_str == "gemini-2.0-pro":
                try:
                    import google.generativeai as genai  # type: ignore
                except Exception as e:
                    raise ImportError("Gemini backend requires 'google-generativeai' (legacy) or update this code to google-genai.") from e
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel(model_name="gemini-2.0-pro-exp-02-05", system_instruction=system_prompt)
                answer = model.generate_content(prompt).text
            elif model_str == "gemini-1.5-pro":
                try:
                    import google.generativeai as genai  # type: ignore
                except Exception as e:
                    raise ImportError("Gemini backend requires 'google-generativeai' (legacy) or update this code to google-genai.") from e
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel(model_name="gemini-1.5-pro", system_instruction=system_prompt)
                answer = model.generate_content(prompt).text
            elif model_str == "o3-mini":
                model_str = "o3-mini"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o3-mini-2025-01-31", messages=messages)
                answer = completion.choices[0].message.content

            elif model_str == "claude-3.5-sonnet":
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}])
                answer = json.loads(message.to_json())["content"][0]["text"]
            elif model_str == "gpt4o" or model_str == "gpt-4o":
                model_str = "gpt-4o"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp)
                else:
                    client = OpenAI()
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o-2024-08-06", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-2024-08-06", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "deepseek-chat":
                model_str = "deepseek-chat"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    raise Exception("Please upgrade your OpenAI version to use DeepSeek client")
                else:
                    deepseek_client = OpenAI(
                        api_key=os.getenv('DEEPSEEK_API_KEY'),
                        base_url="https://api.deepseek.com/v1"
                    )
                    if temp is None:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages)
                    else:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages,
                            temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "o1-mini":
                model_str = "o1-mini"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-mini-2024-09-12", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str == "o1":
                model_str = "o1"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model="o1-2024-12-17",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-2024-12-17", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str == "o1-preview":
                model_str = "o1-preview"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-preview", messages=messages)
                answer = completion.choices[0].message.content

            try:
                using_g4f_backend = bool(force_g4f or (g4f_mod is not None and openai_api_key is None and anthropic_api_key is None and gemini_api_key is None))
                total_text_chars = len(system_prompt or "") + len(prompt or "") + len(answer or "")
                if _should_count_tokens(using_g4f_backend, total_text_chars=total_text_chars):
                    encoding = _tiktoken_encoding_for_model(model_str)
                    if encoding is not None:
                        if model_str not in TOKENS_IN:
                            TOKENS_IN[model_str] = 0
                            TOKENS_OUT[model_str] = 0
                        TOKENS_IN[model_str] += len(encoding.encode(system_prompt + prompt))
                        TOKENS_OUT[model_str] += len(encoding.encode(answer))
                        if print_cost:
                            print(f"Current experiment cost = ${curr_cost_est()}, ** Approximate values, may not reflect true cost")
                elif print_cost and using_g4f_backend:
                    print("Token counting skipped for g4f low-RAM mode. Set AGENTLAB_ENABLE_TOKEN_COUNT_FOR_G4F=1 to enable.")
            except Exception as e:
                if print_cost: print(f"Cost approximation has an error? {e}")
            _best_effort_release_memory(clear_local_cache=False)
            return answer
        except Exception as e:
            # Fail fast on missing credentials (common with g4f providers that need api_key or .har)
            if _looks_like_missing_auth(e):
                raise MissingLLMCredentials(
                    "g4f provider requires credentials (api_key or .har). "
                    "Set OPENROUTER_API_KEY / OPENAI_API_KEY (or other provider key), or place a .har/.json in ./har_and_cookies, "
                    "or run with --no-llm to use the offline baseline solver. "
                    f"Original error: {e}"
                ) from e

            print("Inference Exception:", e)
            # Don't sleep for the whole request timeout; use a small backoff.
            try:
                backoff = min(2.0, max(0.1, timeout * 0.25))
            except Exception:
                backoff = 1.0
            time.sleep(backoff)
            continue
    _best_effort_release_memory(clear_local_cache=False)
    raise Exception("Max retries: timeout")


#print(query_model(model_str="o1-mini", prompt="hi", system_prompt="hey"))
