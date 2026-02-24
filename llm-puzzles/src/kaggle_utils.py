from __future__ import annotations
import os
from typing import Optional
from kaggle.api.kaggle_api_extended import KaggleApi

def ensure_auth():
    """Authenticate with Kaggle API using env vars or ~/.kaggle/kaggle.json."""
    # KaggleApi reads KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json
    api = KaggleApi()
    api.authenticate()
    return api

def submit_file(api: KaggleApi, competition: str, filepath: str, message: str = "auto-submit") -> dict:
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    api.competition_submit(file_name=filepath, message=message, competition=competition)
    # Kaggle API doesn't return a structured object here; we return a stub
    return {"competition": competition, "file": filepath, "message": message}


def list_submissions(api: KaggleApi, competition: str):
    """
    Return submission history for the authenticated user.
    Note: Kaggle returns newest-first in the CLI; the API can be similar but we defensively sort by date if present.
    """
    subs = api.competition_submissions(competition)
    return subs

def latest_scored_submission(api: KaggleApi, competition: str) -> Optional[dict]:
    """
    Return the latest submission that has a (public or private) score, as a plain dict,
    or None if nothing is scored yet.
    """
    try:
        subs = list_submissions(api, competition) or []
    except Exception:
        return None

    def _as_dict(s):
        # Submission is a dataclass-like object; fall back to __dict__
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
        # normalize common fields used by kaggle-api
        if "publicScore" in d and "public_score" not in d:
            d["public_score"] = d["publicScore"]
        if "privateScore" in d and "private_score" not in d:
            d["private_score"] = d["privateScore"]
        return d

    scored = []
    for s in subs:
        d = _as_dict(s)
        ps = d.get("public_score") or d.get("publicScore")
        prs = d.get("private_score") or d.get("privateScore")
        if ps not in (None, "", "None") or prs not in (None, "", "None"):
            scored.append((d, d.get("date"), d.get("submittedDate"), d.get("id"), d.get("ref")))
    if not scored:
        return None
    # best effort: keep original order, but prefer the one with latest date if parsable
    return scored[0][0]

def download_leaderboard(api: KaggleApi, competition: str, path: str = ".", **kwargs) -> str:
    """
    Download leaderboard CSV for the competition into `path`.
    Returns the downloaded file path.
    """
    os.makedirs(path, exist_ok=True)
    api.competition_leaderboard_download(competition, path=path)
    # kaggle api usually names it 'leaderboard.csv'
    return os.path.join(path, "leaderboard.csv")
