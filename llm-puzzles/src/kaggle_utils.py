from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except Exception:  # pragma: no cover
    KaggleApi = None  # type: ignore


def prepare_kaggle_config(kaggle_json_path: str, target_dir: Optional[str] = None) -> str:
    """Prepare a Kaggle config directory that contains a `kaggle.json`.

    Kaggle tools look for credentials in either:
    - env vars `KAGGLE_USERNAME` / `KAGGLE_KEY`, OR
    - `~/.kaggle/kaggle.json`, OR
    - `$KAGGLE_CONFIG_DIR/kaggle.json`.

    This helper copies the given `kaggle.json` into a dedicated directory and returns
    that directory path (so the caller can set `KAGGLE_CONFIG_DIR`).

    Notes:
    - We do NOT print the credentials.
    - On Unix we chmod 600 for compatibility with Kaggle CLI.
    """

    src = Path(kaggle_json_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(str(src))

    # Basic JSON sanity check
    obj = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or "username" not in obj or "key" not in obj:
        raise ValueError('kaggle.json must contain "username" and "key"')

    if target_dir is None:
        target_dir = tempfile.mkdtemp(prefix="kaggle_cfg_")

    dst_dir = Path(target_dir).expanduser().resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst = dst_dir / "kaggle.json"
    shutil.copyfile(src, dst)

    # Best-effort chmod for unix
    try:
        os.chmod(dst, 0o600)
    except Exception:
        pass

    return str(dst_dir)


def ensure_auth(kaggle_json_path: Optional[str] = None, config_dir: Optional[str] = None):
    """Authenticate with Kaggle API using env vars or a kaggle.json.

    If `kaggle_json_path` is provided, we set `KAGGLE_CONFIG_DIR` to a directory
    containing that file before initializing `KaggleApi`.
    """

    if KaggleApi is None:
        raise ImportError("kaggle package is not installed. Run: pip install kaggle")

    if kaggle_json_path:
        cfg_dir = prepare_kaggle_config(kaggle_json_path, target_dir=config_dir)
        os.environ["KAGGLE_CONFIG_DIR"] = cfg_dir

    api = KaggleApi()
    api.authenticate()
    return api


def submit_file(api, competition: str, filepath: str, message: str = "auto-submit") -> dict:
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    api.competition_submit(file_name=filepath, message=message, competition=competition)
    return {"competition": competition, "file": filepath, "message": message}


def list_submissions(api, competition: str):
    return api.competition_submissions(competition)


def latest_scored_submission(api, competition: str) -> Optional[dict]:
    try:
        subs = list_submissions(api, competition) or []
    except Exception:
        return None

    def _as_dict(s):
        d = {}
        for k in dir(s):
            if k.startswith("_"):
                continue
            try:
                v = getattr(s, k)
            except Exception:
                continue
            if isinstance(v, (str, int, float, bool)) or v is None:
                d[k] = v
        if "publicScore" in d and "public_score" not in d:
            d["public_score"] = d["publicScore"]
        if "privateScore" in d and "private_score" not in d:
            d["private_score"] = d["privateScore"]
        return d

    for s in subs:
        d = _as_dict(s)
        ps = d.get("public_score") or d.get("publicScore")
        prs = d.get("private_score") or d.get("privateScore")
        if ps not in (None, "", "None") or prs not in (None, "", "None"):
            return d

    return None


def download_leaderboard(api, competition: str, path: str = ".", **kwargs) -> str:
    os.makedirs(path, exist_ok=True)
    api.competition_leaderboard_download(competition, path=path)
    return os.path.join(path, "leaderboard.csv")


def _submission_to_dict(s) -> dict:
    d = {}
    for k in dir(s):
        if k.startswith('_'):
            continue
        try:
            v = getattr(s, k)
        except Exception:
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            d[k] = v
    alias_pairs = {
        'publicScore': 'public_score',
        'privateScore': 'private_score',
        'errorDescription': 'error_description',
        'errorDescriptionNullable': 'error_description',
        'date': 'date',
        'description': 'description',
        'status': 'status',
        'state': 'state',
        'ref': 'ref',
    }
    for src, dst in alias_pairs.items():
        if src in d and dst not in d:
            d[dst] = d[src]
    return d


def latest_submission(api, competition: str) -> Optional[dict]:
    try:
        subs = list_submissions(api, competition) or []
    except Exception:
        return None
    if not subs:
        return None
    return _submission_to_dict(subs[0])


def wait_for_submission_result(api, competition: str, target_ref: str | int | None = None, wait_seconds: int = 45, poll_every: float = 3.0) -> Optional[dict]:
    """Best-effort wait for the latest submission to get a score or an error.

    Kaggle usually accepts the file immediately, then evaluates it asynchronously.
    Polling recent submissions helps surface format/scoring errors that explain why
    a submission does not appear on the leaderboard.
    """
    import time

    deadline = time.time() + max(0, wait_seconds)
    target_ref = str(target_ref) if target_ref is not None else None

    last_seen = None
    while True:
        sub = latest_submission(api, competition)
        if sub:
            sid = sub.get('id') or sub.get('ref')
            sid_str = str(sid) if sid is not None else None
            if target_ref is None or sid_str == target_ref:
                last_seen = sub
                status = (sub.get('status') or sub.get('state') or '').lower()
                has_score = (sub.get('public_score') not in (None, '', 'None')) or (sub.get('private_score') not in (None, '', 'None'))
                has_error = sub.get('error_description') not in (None, '', 'None')
                if has_score or has_error or status in {'complete', 'error', 'failed'}:
                    return sub
        if time.time() >= deadline:
            return last_seen
        time.sleep(max(0.5, poll_every))
