import time
import os
import json
import sys
import tempfile
import subprocess

# Optional dependencies: keep g4f-only usage lightweight.
try:
    import openai  # type: ignore
    from openai import OpenAI  # type: ignore
except Exception:
    openai = None  # type: ignore
    OpenAI = None  # type: ignore

try:
    import tiktoken  # type: ignore
    encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    tiktoken = None  # type: ignore
    encoding = None

try:
    import anthropic  # type: ignore
except Exception:
    anthropic = None  # type: ignore

genai = None  # type: ignore

# g4f (GPT4Free) backend (optional)
try:
    import g4f  # type: ignore
except Exception:
    g4f = None
    # Optional vendor fallback: if this repo contains a bundled gpt4free checkout at ../gpt4free
    try:
        import sys
        from pathlib import Path

        _ROOT = Path(__file__).resolve().parents[1]
        _VENDOR = _ROOT / "gpt4free"
        if _VENDOR.exists():
            sys.path.insert(0, str(_VENDOR))
            import g4f  # type: ignore
    except Exception:
        g4f = None


def _g4f_to_text(resp):
    """g4f may return a string or an iterator (stream). Convert to string safely."""
    if isinstance(resp, str):
        return resp
    if resp is not None and hasattr(resp, "__iter__"):
        try:
            parts = []
            for ch in resp:
                if isinstance(ch, str):
                    parts.append(ch)
            return "".join(parts)
        except Exception:
            return ""
    return ""


def _g4f_call_isolated(*, model: str, messages, timeout_s: float, provider_name: str | None, api_key: str | None, max_tokens: int | None) -> str:
    """Run g4f in a short-lived subprocess to avoid RAM growth/leaks.

    Some g4f providers/versions can steadily grow memory usage across many calls.
    Spawning a new Python process per request guarantees memory is returned to the OS
    when the process exits.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "g4f_worker.py")
    if not os.path.exists(worker):
        raise FileNotFoundError(worker)

    with tempfile.TemporaryDirectory(prefix="agentlab_g4f_") as td:
        req_path = os.path.join(td, "req.json")
        resp_path = os.path.join(td, "resp.json")
        with open(req_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": model,
                    "messages": messages,
                    "timeout": float(timeout_s),
                    "provider_name": provider_name or "",
                    "api_key": api_key or "",
                    "max_tokens": max_tokens,
                },
                f,
                ensure_ascii=False,
            )

        cmd = [sys.executable, worker, req_path, resp_path]
        # Do NOT capture stdout/stderr: provider debug output can be large and capturing it may increase RAM.
        p = subprocess.run(cmd, cwd=here)
        if p.returncode != 0:
            raise RuntimeError(f"g4f worker failed with returncode={p.returncode}")

        try:
            with open(resp_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            raise RuntimeError(f"g4f worker produced no readable response: {e}")

        if not payload.get("ok"):
            raise RuntimeError(payload.get("error") or "g4f worker error")
        return (payload.get("answer") or "").strip()


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

if encoding is None and tiktoken is not None:
    encoding = tiktoken.get_encoding("cl100k_base")

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
    if openai_api_key is None and anthropic_api_key is None and gemini_api_key is None and g4f is None:
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
            if force_g4f or (g4f is not None and openai_api_key is None and anthropic_api_key is None and gemini_api_key is None):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                api_key = (
                    os.getenv("OPENROUTER_API_KEY")
                    or os.getenv("OPENAI_API_KEY")
                    or os.getenv("GROQ_API_KEY")
                    or os.getenv("TOGETHER_API_KEY")
                    or os.getenv("GEMINI_API_KEY")
                )
                provider_name = os.getenv("G4F_PROVIDER", "").strip() or None

                # Cap output where possible (many providers respect max_tokens).
                try:
                    max_tokens_env = os.getenv("AGENTLAB_MAX_TOKENS_OUT", os.getenv("G4F_MAX_TOKENS_OUT", ""))
                    max_tokens = int(max_tokens_env) if max_tokens_env else 4096
                except Exception:
                    max_tokens = 4096

                # By default, isolate g4f calls in a subprocess to avoid RAM growth during long runs.
                isolate = os.getenv("AGENTLAB_G4F_ISOLATE", "1").strip().lower() not in ("0", "false", "no")

                if isolate:
                    answer = _g4f_call_isolated(
                        model=model_str,
                        messages=messages,
                        timeout_s=float(timeout),
                        provider_name=provider_name,
                        api_key=api_key,
                        max_tokens=max_tokens,
                    )
                else:
                    kwargs = {}
                    try:
                        import inspect
                        sig = inspect.signature(g4f.ChatCompletion.create)  # type: ignore
                        if "api_key" in sig.parameters and api_key:
                            kwargs["api_key"] = api_key
                        if "max_tokens" in sig.parameters and max_tokens:
                            kwargs["max_tokens"] = max_tokens
                        if "provider" in sig.parameters and provider_name:
                            try:
                                kwargs["provider"] = getattr(g4f.Provider, provider_name)  # type: ignore
                            except Exception:
                                pass
                    except Exception:
                        pass

                    resp = g4f.ChatCompletion.create(
                        model=model_str,
                        messages=messages,
                        timeout=int(max(1, timeout)),
                        **kwargs,
                    )
                    answer = _g4f_to_text(resp)
                if isinstance(answer, str):
                    answer = answer.strip()
                if answer:
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

            # Token/cost accounting is optional and can be memory-heavy for large prompts.
            # Disable by default for g4f (no reliable pricing anyway) and allow an env toggle.
            try:
                disable_token_count = os.getenv("AGENTLAB_DISABLE_TOKEN_COUNT", "").strip().lower() in ("1", "true", "yes")
                using_g4f = force_g4f or (g4f is not None and openai_api_key is None and anthropic_api_key is None and gemini_api_key is None)
                if (not disable_token_count) and (not using_g4f) and tiktoken is not None:
                    # Avoid huge transient allocations: fall back to a coarse estimate if text is very large.
                    max_chars = int(os.getenv("AGENTLAB_TOKEN_COUNT_MAX_CHARS", "200000"))
                    sp = system_prompt or ""
                    pr = prompt or ""
                    an = answer or ""
                    if model_str not in TOKENS_IN:
                        TOKENS_IN[model_str] = 0
                        TOKENS_OUT[model_str] = 0

                    if len(sp) + len(pr) <= max_chars:
                        if model_str in ["o1-preview", "o1-mini", "claude-3.5-sonnet", "o1", "o3-mini"]:
                            enc = tiktoken.encoding_for_model("gpt-4o")
                        elif model_str in ["deepseek-chat"]:
                            enc = tiktoken.encoding_for_model("cl100k_base")
                        else:
                            enc = tiktoken.encoding_for_model(model_str)
                        TOKENS_IN[model_str] += len(enc.encode(sp + pr))
                    else:
                        # Very rough heuristic (~4 chars/token in practice for many English-like texts).
                        TOKENS_IN[model_str] += int((len(sp) + len(pr)) / 4)

                    if len(an) <= max_chars:
                        try:
                            enc_out = tiktoken.encoding_for_model(model_str)
                        except Exception:
                            enc_out = tiktoken.get_encoding("cl100k_base")
                        TOKENS_OUT[model_str] += len(enc_out.encode(an))
                    else:
                        TOKENS_OUT[model_str] += int(len(an) / 4)

                    if print_cost:
                        print(f"Current experiment cost = ${curr_cost_est()}, ** Approximate values, may not reflect true cost")
            except Exception as e:
                if print_cost:
                    print(f"Cost approximation has an error? {e}")
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
    raise Exception("Max retries: timeout")


#print(query_model(model_str="o1-mini", prompt="hi", system_prompt="hey"))
