from typing import Dict, Any, List, Optional
import os, re, json, csv, queue

try:
    from CallLLM import CONFIG, llm_query, get_models_list, quick_selfcheck
except Exception:
    CONFIG, llm_query, get_models_list = {'CONSTANTS': {'REQUEST_TIMEOUT': 60}}, None, (lambda c: [])
    def quick_selfcheck(models: List[str], prompt: str = "ok?", max_models: int = 2, timeout: int = 8) -> List[str]:
        return []

PROMPT_TEMPLATE = (
    "You are given a puzzle row as JSON. Return ONLY a single line listing moves, "
    "joined by '{joiner}'. Valid tokens look like '-d3', 'f1', 'r0', etc. No extra text.\n"
    "{row_json}"
)

def _iter_to_text(resp) -> str:
    if isinstance(resp, str):
        return resp
    if resp is not None and hasattr(resp, '__iter__'):
        try:
            parts = []
            for ch in resp:
                if isinstance(ch, str):
                    parts.append(ch)
            return ''.join(parts)
        except Exception:
            return ''
    return ''

def _extract_moves_only(text: str, joiner: str = '.') -> str:
    if not text:
        return ''
    t = str(text).strip()
    t = re.sub(r'^```[a-zA-Z0-9]*\s*|\s*```$', '', t, flags=re.MULTILINE)
    tokens = re.findall(r'-?[A-Za-z]\d+', t)
    if not tokens:
        return ''
    return joiner.join(tokens)

_SAMPLE_MAP: Optional[Dict[int, str]] = None

def _load_sample_map() -> Dict[int, str]:
    global _SAMPLE_MAP
    if _SAMPLE_MAP is not None:
        return _SAMPLE_MAP
    puzzles_csv = os.getenv('PUZZLES_CSV', '') or ''
    if not puzzles_csv:
        _SAMPLE_MAP = {}
        return _SAMPLE_MAP
    base_dir = os.path.dirname(puzzles_csv)
    sample_path = os.path.join(base_dir, 'sample_submission.csv')
    mp: Dict[int, str] = {}
    try:
        with open(sample_path, 'r', newline='') as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                rid = row.get('initial_state_id')
                pth = row.get('path', '')
                if rid is None:
                    continue
                try:
                    rid_int = int(str(rid))
                except Exception:
                    continue
                mp[rid_int] = str(pth or '')
    except Exception:
        mp = {}
    _SAMPLE_MAP = mp
    return _SAMPLE_MAP

def _sample_path_for(row: Dict[str, Any]) -> str:
    try:
        rid = int(str(row.get('initial_state_id')))
    except Exception:
        return ''
    mp = _load_sample_map()
    return mp.get(rid, '')

def _llm_ask(prompt: str, stage: str = "perm_llm_submission") -> str:
    if os.getenv('LLM_OFFLINE', '').lower() in {'1','true','yes'}:
        return ''
    if llm_query is None or get_models_list is None:
        return ''
    try:
        models = get_models_list(CONFIG)
    except Exception:
        models = []

    if os.getenv('G4F_SELFCHECK','1').lower() in {'1','true','yes'}:
        try:
            top = int(os.getenv('G4F_SELFCHECK_TOP','2') or '2')
        except Exception:
            top = 2
        ok = quick_selfcheck(models, prompt=os.getenv('G4F_SELFCHECK_PROMPT','ok?'), max_models=top, timeout=8)
        tail = [m for m in models if m not in ok]
        models = ok + tail

    preferred = [m for m in models if any(k in m.lower() for k in ['gpt-4o','gpt-4','gpt-3.5','claude','llama'])]
    remaining = [m for m in models if m not in preferred]
    ordered = preferred + remaining

    retries_cfg = {'max_retries': 2, 'backoff_factor': 1.0}
    q = queue.Queue()
    for model in (ordered or ['gpt-4o-mini']):
        try:
            resp = llm_query(model=model, prompt=prompt, retries_config=retries_cfg, config=CONFIG, progress_queue=q, stage=stage)
            text = _iter_to_text(resp)
            if text and text.strip():
                return text.strip()
        except Exception:
            continue
    return ''

def solve_row(row: Dict[str, Any], cfg) -> str:
    joiner = getattr(cfg, 'move_joiner', '.') if hasattr(cfg, 'move_joiner') else '.'
    strict = os.getenv('STRICT_LLM', '').lower() in {'1','true','yes'}
    prefer_sample = os.getenv('PREFER_SAMPLE', '').lower() in {'1','true','yes'}

    compact: Dict[str, str] = {}
    for k, v in row.items():
        s = str(v)
        if len(s) > 2000:
            s = s[:2000] + '...'
        compact[k] = s
    row_json = json.dumps(compact, ensure_ascii=False)
    prompt = PROMPT_TEMPLATE.format(row_json=row_json, joiner=joiner)

    if strict:
        raw = _llm_ask(prompt)
        moves = _extract_moves_only(raw, joiner=joiner)
        return moves or ''

    if prefer_sample:
        smp = _sample_path_for(row)
        if smp:
            return smp

    raw = _llm_ask(prompt)
    moves = _extract_moves_only(raw, joiner=joiner)
    if moves:
        return moves

    smp = _sample_path_for(row)
    if smp:
        return smp

    return ''
