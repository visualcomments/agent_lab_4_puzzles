import os, time, json, queue, requests
from typing import Dict, List, Optional

try:
    import g4f
    from g4f.Provider import Any as Provider
    try:
        from g4f.errors import ModelNotFoundError  # type: ignore
    except Exception:
        class ModelNotFoundError(Exception):
            pass
except Exception as e:
    g4f, Provider = None, None
    class ModelNotFoundError(Exception):
        pass

try:
    import g4f.providers.retry_provider as retry_mod
    OriginalRotatedProvider = retry_mod.RotatedProvider
    class TrackedRotated(OriginalRotatedProvider):
        pass
    retry_mod.RotatedProvider = TrackedRotated
except Exception:
    pass

CONFIG: Dict = {
    "CONSTANTS": {"REQUEST_TIMEOUT": 60, "MODEL_TYPE_TEXT": "text", "DELIMITER_MODEL": "|", "MAX_WORKERS": 8},
    "URLS": {"WORKING_RESULTS": "https://raw.githubusercontent.com/xtekky/gpt4free/refs/heads/main/docs/WORKING_RESULTS.txt"}
}

def _iter_to_text(resp) -> str:
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

def quick_selfcheck(models: List[str], prompt: str = "ok?", max_models: int = 2, timeout: int = 8) -> List[str]:
    ok: List[str] = []
    if g4f is None or Provider is None:
        return ok
    test_subset = [m for m in models if isinstance(m, str) and m.strip()][:max_models]
    for m in test_subset:
        try:
            resp = g4f.ChatCompletion.create(model=m, messages=[{"role": "user", "content": prompt}], provider=Provider, timeout=timeout)
            text = _iter_to_text(resp)
            if isinstance(text, str) and text.strip():
                ok.append(m)
        except Exception:
            continue
    return ok

def get_models_list(config: Dict) -> List[str]:
    url_txt = config.get('URLS', {}).get('WORKING_RESULTS')
    delimiter = config.get('CONSTANTS', {}).get('DELIMITER_MODEL', '|')
    text_type = config.get('CONSTANTS', {}).get('MODEL_TYPE_TEXT', 'text')
    timeout = config.get('CONSTANTS', {}).get('REQUEST_TIMEOUT', 60)

    working_models: List[str] = []
    if url_txt:
        try:
            resp = requests.get(url_txt, timeout=timeout)
            resp.raise_for_status()
            for line in resp.text.splitlines():
                if delimiter in line:
                    parts = [p.strip() for p in line.split(delimiter)]
                    if len(parts) == 3 and parts[2] == text_type:
                        name = parts[1]
                        low = name.lower()
                        if 'flux' not in low and not any(sub in low for sub in ['image','vision','audio','video']):
                            working_models.append(name)
        except Exception:
            pass

    g4f_models: List[str] = []
    try:
        import importlib
        gm = importlib.import_module('g4f.models')
        names = []
        try:
            names = list(getattr(gm, '__all__', []))
        except Exception:
            names = []
        if not names:
            for k in dir(gm):
                if k.startswith('_'):
                    continue
                try:
                    v = getattr(gm, k)
                    if isinstance(v, str):
                        names.append(v)
                except Exception:
                    continue
        for n in names:
            if isinstance(n, str) and n.strip():
                s = n.strip(); low = s.lower()
                if 'flux' not in low and not any(sub in low for sub in ['image','vision','audio','video']):
                    g4f_models.append(s)
    except Exception:
        pass

    merged: List[str] = []
    seen = set()
    for m in working_models + g4f_models:
        if isinstance(m, str):
            m2 = m.strip()
            if m2 and m2 not in seen:
                seen.add(m2); merged.append(m2)

    env_wl = os.getenv('G4F_MODELS', '').strip()
    if env_wl:
        wl = [x.strip() for x in env_wl.split(',') if x.strip()]
        if wl:
            merged = wl

    if os.getenv('G4F_SELFCHECK', '1').lower() in {'1','true','yes'}:
        top = int(os.getenv('G4F_SELFCHECK_TOP', '2') or '2')
        pr = os.getenv('G4F_SELFCHECK_PROMPT', 'ok?')
        ok = quick_selfcheck(merged, prompt=pr, max_models=top, timeout=8)
        rest = [m for m in merged if m not in ok]
        merged = ok + rest

    if not merged:
        merged = ['aria','command-r','command-a']
    return merged

def llm_query(model: str, prompt: str, retries_config: Dict, config: Dict, progress_queue: queue.Queue, stage: str = None) -> Optional[str]:
    if os.getenv('LLM_OFFLINE', '').lower() in {'1','true','yes'}:
        return None
    if g4f is None or Provider is None:
        return None
    request_timeout = config.get('CONSTANTS', {}).get('REQUEST_TIMEOUT', 60)
    max_retries = int(retries_config.get('max_retries', 0))
    backoff = float(retries_config.get('backoff_factor', 1.0))
    for attempt in range(max_retries + 1):
        try:
            resp = g4f.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}], provider=Provider, timeout=request_timeout)
            text = _iter_to_text(resp)
            if isinstance(text, str) and text.strip():
                return text.strip()
        except Exception:
            time.sleep(backoff * (2 ** attempt) if attempt < max_retries else 0)
    return None

def main():
    models = get_models_list(CONFIG)
    print(json.dumps({'models': models[:10]}, ensure_ascii=False))

if __name__ == "__main__":
    main()
