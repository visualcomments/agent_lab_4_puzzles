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


def _chmod_private(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def _load_kaggle_credentials(credentials_path: str) -> dict:
    src = Path(credentials_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(str(src))

    raw = src.read_text(encoding="utf-8").strip()
    parsed = None
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        if parsed.get("username") and parsed.get("key"):
            return {
                "kind": "legacy_json",
                "source": str(src),
                "username": str(parsed["username"]),
                "key": str(parsed["key"]),
            }
        for token_key in ("api_token", "access_token", "token"):
            token = parsed.get(token_key)
            if token:
                return {
                    "kind": "access_token",
                    "source": str(src),
                    "token": str(token),
                }

    # Support passing a plain access_token file as well.
    if raw and "\n" not in raw and not raw.startswith("{"):
        return {
            "kind": "access_token",
            "source": str(src),
            "token": raw,
        }

    raise ValueError(
        "Unsupported Kaggle credentials file. Expected kaggle.json with username/key, "
        "a JSON file with api_token/access_token/token, or a plain access_token file."
    )




def _discover_default_credentials_path(config_dir: Optional[str] = None) -> Optional[str]:
    candidates = []
    cfg_dir = Path(config_dir).expanduser() if config_dir else None
    env_cfg = Path(os.environ["KAGGLE_CONFIG_DIR"]).expanduser() if os.environ.get("KAGGLE_CONFIG_DIR") else None
    for base in [cfg_dir, env_cfg, Path.home() / ".kaggle"]:
        if base is None:
            continue
        candidates.append(base / "kaggle.json")
        candidates.append(base / "access_token")
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def build_kaggle_env(credentials_path: Optional[str] = None, config_dir: Optional[str] = None) -> dict[str, str]:
    env: dict[str, str] = {}
    credentials_path = credentials_path or _discover_default_credentials_path(config_dir)
    if not credentials_path:
        return env

    creds = _load_kaggle_credentials(credentials_path)
    cfg_dir = prepare_kaggle_config(credentials_path, target_dir=config_dir)
    env["KAGGLE_CONFIG_DIR"] = cfg_dir

    if creds["kind"] == "legacy_json":
        env["KAGGLE_USERNAME"] = creds["username"]
        env["KAGGLE_KEY"] = creds["key"]
    else:
        env["KAGGLE_API_TOKEN"] = creds["token"]

    return env


def prepare_kaggle_config(kaggle_json_path: str, target_dir: Optional[str] = None) -> str:
    """Prepare a Kaggle config directory for either legacy or token-style auth.

    Supported input file formats:
    - legacy `kaggle.json` with `username` / `key`
    - JSON with `api_token` / `access_token` / `token`
    - plain-text access token file
    """

    creds = _load_kaggle_credentials(kaggle_json_path)
    if target_dir is None:
        target_dir = tempfile.mkdtemp(prefix="kaggle_cfg_")

    dst_dir = Path(target_dir).expanduser().resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    if creds["kind"] == "legacy_json":
        src = Path(creds["source"])
        dst = dst_dir / "kaggle.json"
        shutil.copyfile(src, dst)
        _chmod_private(dst)
    else:
        dst = dst_dir / "access_token"
        dst.write_text(creds["token"], encoding="utf-8")
        _chmod_private(dst)

    return str(dst_dir)


def ensure_auth(kaggle_json_path: Optional[str] = None, config_dir: Optional[str] = None):
    """Authenticate with Kaggle API using env vars or an explicit credentials file."""

    if KaggleApi is None:
        raise ImportError("kaggle package is not installed. Run: pip install kaggle")

    env_updates = build_kaggle_env(kaggle_json_path, config_dir=config_dir)
    if env_updates:
        os.environ.update(env_updates)

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
    """Best-effort wait for the latest submission to get a score or an error."""
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
