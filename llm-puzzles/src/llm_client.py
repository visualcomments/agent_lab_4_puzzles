"""
Thin client around CallLLM.py for ad-hoc prompts.
Not used by universal_adapter directly; the examples/llm_solver/solver.py calls CallLLM itself.
"""
from typing import Optional
import queue

try:
    from CallLLM import CONFIG, llm_query, get_models_list
except Exception:
    CONFIG, llm_query, get_models_list = None, None, None

def ask(prompt: str, stage: str = "adhoc") -> Optional[str]:
    if CONFIG is None or llm_query is None or get_models_list is None:
        return None
    try:
        models = get_models_list(CONFIG)
    except Exception:
        models = ["gpt-4o-mini"]
    q = queue.Queue()
    for m in models:
        try:
            resp = llm_query(m, prompt, {'max_retries':1,'backoff_factor':1.0}, CONFIG, q, stage=stage)
            if resp:
                return resp
        except Exception:
            continue
    return None