from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

KaggleApi = None  # type: ignore


MIN_KAGGLE_SUBMIT_VERSION: tuple[int, int, int] = (1, 5, 0)


class KagglePreflightError(RuntimeError):
    """Raised when a submission preflight check fails."""


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


def _parse_kaggle_version(raw: str) -> Optional[tuple[int, ...]]:
    text = (raw or "").strip()
    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d+)+)", text)
    if not match:
        return None
    parts = [p for p in match.group(1).split(".") if p != ""]
    if not parts:
        return None
    try:
        return tuple(int(p) for p in parts)
    except Exception:
        return None



def _normalize_version_tuple(version: Sequence[int], width: int) -> tuple[int, ...]:
    padded = list(version[:width])
    if len(padded) < width:
        padded.extend([0] * (width - len(padded)))
    return tuple(padded)



def _version_is_at_least(version: Sequence[int], minimum: Sequence[int]) -> bool:
    width = max(len(version), len(minimum))
    return _normalize_version_tuple(version, width) >= _normalize_version_tuple(minimum, width)



def _format_version(version: Sequence[int] | None) -> str:
    if not version:
        return "unknown"
    return ".".join(str(x) for x in version)



def installed_kaggle_package_version() -> Optional[tuple[int, ...]]:
    try:
        raw = str(importlib_metadata.version("kaggle") or "")
    except Exception:
        return None
    return _parse_kaggle_version(raw)



def get_kaggle_cli_version(env: Optional[dict[str, str]] = None) -> dict[str, Any]:
    exe = shutil.which("kaggle")
    if not exe:
        return {
            "available": False,
            "version": None,
            "raw": "",
            "command": None,
        }

    try:
        proc = subprocess.run(
            [exe, "--version"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    except Exception as exc:
        return {
            "available": True,
            "version": None,
            "raw": str(exc),
            "command": [exe, "--version"],
        }

    raw = "\n".join(part for part in [proc.stdout, proc.stderr] if part).strip()
    return {
        "available": True,
        "version": _parse_kaggle_version(raw),
        "raw": raw,
        "command": [exe, "--version"],
        "returncode": int(proc.returncode),
    }



def _looks_like_rules_or_join_error(message: str) -> bool:
    low = (message or "").lower()
    needles = [
        "competitions.participate",
        "accept competition rules",
        "accept the competition rules",
        "accept the rules",
        "rules have not been accepted",
        "rules were not accepted",
        "join the competition",
        "joined the competition",
        "permission 'competitions.participate' was denied",
        "permission was denied",
        "unauthorized",
        "forbidden",
    ]
    return any(needle in low for needle in needles)



def _rules_url_for_competition(competition: str) -> str:
    slug = str(competition or "").strip().strip("/")
    return f"https://www.kaggle.com/competitions/{slug}/rules"



def _rules_or_join_error_message(competition: str, detail: str) -> str:
    rules_url = _rules_url_for_competition(competition)
    detail = (detail or "").strip()
    suffix = f" Raw Kaggle message: {detail}" if detail else ""
    return (
        f"Kaggle preflight failed for '{competition}': the account cannot access competition submissions yet. "
        f"Open {rules_url}, accept the competition rules, and make sure this account has joined the competition."
        f"{suffix}"
    )



def _ensure_submit_version_or_raise(version: Optional[Sequence[int]], source: str) -> tuple[int, ...]:
    if version is None:
        raise KagglePreflightError(
            f"Could not determine the installed Kaggle version for {source}. "
            f"Competition submissions may not work on versions below {_format_version(MIN_KAGGLE_SUBMIT_VERSION)}."
        )
    version_tuple = tuple(int(x) for x in version)
    if not _version_is_at_least(version_tuple, MIN_KAGGLE_SUBMIT_VERSION):
        raise KagglePreflightError(
            f"Kaggle {source} version {_format_version(version_tuple)} is too old for reliable competition submission. "
            f"Install/upgrade to at least {_format_version(MIN_KAGGLE_SUBMIT_VERSION)}."
        )
    return version_tuple



def _get_kaggle_api_class():
    global KaggleApi
    if KaggleApi is not None:
        return KaggleApi
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi as _KaggleApi  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("kaggle package is not installed. Run: pip install kaggle") from exc
    KaggleApi = _KaggleApi
    return KaggleApi


def ensure_auth(kaggle_json_path: Optional[str] = None, config_dir: Optional[str] = None):
    """Authenticate with Kaggle API using env vars or an explicit credentials file."""

    env_updates = build_kaggle_env(kaggle_json_path, config_dir=config_dir)
    if env_updates:
        os.environ.update(env_updates)

    api_cls = _get_kaggle_api_class()
    api = api_cls()
    api.authenticate()
    return api



def _probe_api_submit_access(api, competition: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "competition": competition,
        "can_list_files": False,
        "can_list_submissions": False,
        "rules_accepted_or_joined": None,
    }

    try:
        files = api.competition_list_files(competition) or []
        report["can_list_files"] = True
        report["file_count"] = len(files)
    except Exception as exc:
        report["files_error"] = str(exc)

    try:
        subs = api.competition_submissions(competition) or []
        report["can_list_submissions"] = True
        report["rules_accepted_or_joined"] = True
        report["submission_count"] = len(subs)
    except Exception as exc:
        msg = str(exc)
        report["submissions_error"] = msg
        if _looks_like_rules_or_join_error(msg):
            report["rules_accepted_or_joined"] = False
            raise KagglePreflightError(_rules_or_join_error_message(competition, msg)) from exc
        raise KagglePreflightError(
            f"Kaggle API preflight failed while checking submission access for '{competition}': {msg}"
        ) from exc

    return report



def preflight_submit_via_api(
    competition: str,
    kaggle_json_path: Optional[str] = None,
    config_dir: Optional[str] = None,
) -> dict[str, Any]:
    version_tuple = _ensure_submit_version_or_raise(installed_kaggle_package_version(), "API package")
    api = ensure_auth(kaggle_json_path=kaggle_json_path, config_dir=config_dir)
    access = _probe_api_submit_access(api, competition)
    return {
        "mode": "api",
        "competition": competition,
        "client_version": _format_version(version_tuple),
        "access": access,
    }



def preflight_submit_via_cli(
    competition: str,
    credentials_path: Optional[str] = None,
    config_dir: Optional[str] = None,
    runner: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(build_kaggle_env(credentials_path, config_dir=config_dir))

    info = get_kaggle_cli_version(env=env)
    if not info.get("available"):
        raise KagglePreflightError(
            "Kaggle CLI is not installed or not on PATH. Install it with `pip install kaggle` "
            "before using --submit-via cli."
        )

    version_tuple = _ensure_submit_version_or_raise(info.get("version"), "CLI")
    run = runner or subprocess.run
    cmd = ["kaggle", "competitions", "submissions", competition, "-q"]
    try:
        proc = run(cmd, capture_output=True, text=True, env=env, check=False)
    except FileNotFoundError as exc:
        raise KagglePreflightError(
            "Kaggle CLI is not installed or not on PATH. Install it with `pip install kaggle` "
            "before using --submit-via cli."
        ) from exc
    except Exception as exc:
        raise KagglePreflightError(f"Kaggle CLI preflight failed for '{competition}': {exc}") from exc

    raw = "\n".join(part for part in [getattr(proc, "stdout", ""), getattr(proc, "stderr", "")] if part).strip()
    rc = int(getattr(proc, "returncode", 1))
    if rc != 0:
        if _looks_like_rules_or_join_error(raw):
            raise KagglePreflightError(_rules_or_join_error_message(competition, raw))
        raise KagglePreflightError(
            f"Kaggle CLI preflight failed while checking submission access for '{competition}': "
            f"exit code {rc}. {raw or 'No output from kaggle CLI.'}"
        )

    return {
        "mode": "cli",
        "competition": competition,
        "client_version": _format_version(version_tuple),
        "access": {
            "can_list_submissions": True,
            "rules_accepted_or_joined": True,
        },
        "command": cmd,
    }



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
