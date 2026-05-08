from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def safe_read_json(path: Optional[Path]) -> Any:
    if path is None:
        return None
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _split_path(path: Any) -> list[str]:
    text = str(path or "").strip()
    if not text or text.upper() == "UNSOLVED":
        return []
    if "." in text:
        return [p for p in text.split(".") if p]
    return [p for p in text.split() if p]


def _score_path(path: Any) -> int:
    return len(_split_path(path))


def _path_motifs(path: Any, *, n: int = 2, limit: int = 6) -> list[dict[str, Any]]:
    toks = _split_path(path)
    counts: dict[tuple[str, ...], int] = {}
    if n <= 0:
        return []
    for idx in range(0, max(0, len(toks) - n + 1)):
        key = tuple(toks[idx:idx + n])
        counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return [{"motif": ".".join(key), "count": count} for key, count in ranked if count > 1]


def _rewrite_windows(path: Any, *, window: int = 8, limit: int = 4) -> list[dict[str, Any]]:
    toks = _split_path(path)
    out: list[dict[str, Any]] = []
    if not toks:
        return out
    starts = [0, max(0, len(toks)//3 - window//2), max(0, 2*len(toks)//3 - window//2), max(0, len(toks) - window)]
    seen: set[int] = set()
    for start in starts:
        if start in seen:
            continue
        seen.add(start)
        chunk = toks[start:start + window]
        if chunk:
            out.append({"start": start, "end": start + len(chunk), "moves": ".".join(chunk)})
        if len(out) >= limit:
            break
    return out


def _read_submission_rows(path: Optional[Path]) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            out = []
            for idx, row in enumerate(csv.DictReader(f)):
                out.append({
                    "row_id": row.get("initial_state_id") or idx,
                    "current_best_len": _score_path(row.get("path")),
                    "path": row.get("path") or "",
                    "path_motifs": _path_motifs(row.get("path")),
                    "rewrite_windows": _rewrite_windows(row.get("path")),
                })
            return out
    except Exception:
        return []


def find_row_profile_file(*, explicit_path: Optional[Path] = None, output_dir: Optional[Path] = None, baseline_solver_path: Optional[Path] = None) -> Optional[Path]:
    candidates: list[Optional[Path]] = [explicit_path]
    if output_dir is not None:
        candidates.extend([
            output_dir / "profiles.json",
            output_dir / "search_profiles.json",
            output_dir / "row_profiles.json",
            output_dir.parent / "profiles.json",
            output_dir.parent / "search_profiles.json",
        ])
    comp_dir = baseline_solver_path.parent if baseline_solver_path is not None else None
    if comp_dir is not None:
        candidates.extend([
            comp_dir / "submissions" / "optimized_submission.v3.profiles.json",
            comp_dir / "submissions" / "submission_search_improved_v3_top300.profiles.json",
            comp_dir / "submissions" / "_smoke_hybrid.profiles.json",
            comp_dir / "profiles.json",
            comp_dir / "search_profiles.json",
        ])
    for path in candidates:
        if path is not None and path.exists():
            return path
    return None


def find_submission_file(*, baseline_solver_path: Optional[Path] = None) -> Optional[Path]:
    if baseline_solver_path is None:
        return None
    comp_dir = baseline_solver_path.parent
    for path in [
        comp_dir / "submissions" / "optimized_submission.csv",
        comp_dir / "data" / "sample_submission.csv",
    ]:
        if path.exists():
            return path
    return None


def _iter_rows(raw: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(raw, dict):
        for key in ("rows", "profiles", "row_profiles", "results", "per_row"):
            value = raw.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
        for row_id, value in raw.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("row_id", row_id)
                yield row


def _row_id(row: Dict[str, Any], fallback: int) -> Any:
    for key in ("row_id", "initial_state_id", "id", "index", "row", "puzzle_id"):
        if key in row:
            return row.get(key)
    return fallback


def _num(row: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return default


def summarize_row_profiles(raw: Any, *, top_n: int = 12, fallback_submission_rows: Optional[list[dict[str, Any]]] = None) -> Dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(_iter_rows(raw)):
        current_best_len = _num(row, "current_best_len", "best_len", "path_len", "length", "len", default=0.0)
        baseline_len = _num(row, "baseline_len", "original_len", default=current_best_len)
        saved_moves = _num(row, "saved_moves", "preopt_saved", "improvement", default=max(0.0, baseline_len - current_best_len))
        delta_moves = _num(row, "delta_moves", "net_delta_moves", "score_delta", default=0.0)
        runtime_s = _num(row, "runtime_s", "seconds", "elapsed_s", default=0.0)
        path_value = row.get("path") or row.get("moves") or row.get("solution") or ""
        rows.append({
            "row_id": _row_id(row, idx),
            "current_best_len": int(current_best_len),
            "baseline_len": int(baseline_len),
            "saved_moves": int(saved_moves),
            "delta_moves": int(delta_moves),
            "runtime_s": runtime_s,
            "saved_moves_per_second": saved_moves / max(runtime_s, 1e-9),
            "regressed": bool(row.get("regressed") or delta_moves > 0),
            "path": path_value,
            "path_motifs": row.get("path_motifs") or _path_motifs(path_value),
            "rewrite_windows": row.get("rewrite_windows") or _rewrite_windows(path_value),
        })
    if not rows and fallback_submission_rows:
        rows = [dict(row) for row in fallback_submission_rows]
    if not rows:
        return {
            "available": False,
            "rows": 0,
            "top_hard_rows": [],
            "top_improved_rows": [],
            "top_regressed_rows": [],
            "path_length_buckets": {},
            "hardest_tail": [],
            "hard_row_ids": [],
            "routing_hint": "no row-level profile data available",
        }
    for row in rows:
        row.setdefault("baseline_len", row.get("current_best_len", 0))
        row.setdefault("saved_moves", 0)
        row.setdefault("delta_moves", 0)
        row.setdefault("runtime_s", 0.0)
        row.setdefault("saved_moves_per_second", 0.0)
        row.setdefault("regressed", False)
    top_hard_rows = sorted(rows, key=lambda r: int(r.get("current_best_len") or 0), reverse=True)[:top_n]
    top_improved_rows = sorted(rows, key=lambda r: int(r.get("saved_moves") or 0), reverse=True)[:top_n]
    top_regressed_rows = sorted([r for r in rows if r.get("regressed") or int(r.get("delta_moves") or 0) > 0], key=lambda r: (int(r.get("delta_moves") or 0), int(r.get("current_best_len") or 0)), reverse=True)[:top_n]
    buckets = {
        "short_or_solved_<=560": sum(1 for r in rows if int(r.get("current_best_len") or 0) <= 560),
        "mid_561_700": sum(1 for r in rows if 560 < int(r.get("current_best_len") or 0) <= 700),
        "hard_>700": sum(1 for r in rows if int(r.get("current_best_len") or 0) > 700),
        "hard_>800": sum(1 for r in rows if int(r.get("current_best_len") or 0) > 800),
    }
    hard_row_ids = [r.get("row_id") for r in top_hard_rows]
    hardest_tail = [{
        "row_id": r.get("row_id"),
        "current_best_len": int(r.get("current_best_len") or 0),
        "saved_moves": int(r.get("saved_moves") or 0),
        "saved_moves_per_second": round(float(r.get("saved_moves_per_second") or 0.0), 4),
    } for r in top_hard_rows]
    hard_row_micro_pack = [{
        "row_id": r.get("row_id"),
        "baseline_len": int(r.get("current_best_len") or 0),
        "target_len": max(0, int(r.get("current_best_len") or 0) - 1),
        "path_prefix": ".".join(_split_path(r.get("path"))[:24]),
        "path_suffix": ".".join(_split_path(r.get("path"))[-24:]),
        "repeated_motifs": r.get("path_motifs") or [],
        "rewrite_windows": r.get("rewrite_windows") or [],
        "contract": "shorten this row by >=1 move under exact replay or return no_candidate for this lane",
    } for r in top_hard_rows[:min(8, top_n)]]
    return {
        "available": True,
        "rows": len(rows),
        "top_hard_rows": top_hard_rows,
        "top_improved_rows": top_improved_rows,
        "top_regressed_rows": top_regressed_rows,
        "path_length_buckets": buckets,
        "hardest_tail": hardest_tail,
        "hard_row_micro_pack": hard_row_micro_pack,
        "hard_row_ids": hard_row_ids,
        "routing_hint": "focus generated code on hard-tail row ids first; improve >=1 listed row with zero regressions, otherwise fail explicitly",
    }


def load_row_profile_summary(*, explicit_path: Optional[Path] = None, output_dir: Optional[Path] = None, baseline_solver_path: Optional[Path] = None, top_n: int = 12) -> Dict[str, Any]:
    profile_path = find_row_profile_file(explicit_path=explicit_path, output_dir=output_dir, baseline_solver_path=baseline_solver_path)
    submission_path = find_submission_file(baseline_solver_path=baseline_solver_path)
    raw = safe_read_json(profile_path)
    fallback_rows = _read_submission_rows(submission_path)
    summary = summarize_row_profiles(raw, top_n=top_n, fallback_submission_rows=fallback_rows)
    summary["source_path"] = str(profile_path) if profile_path is not None else None
    summary["fallback_submission_path"] = str(submission_path) if submission_path is not None else None
    return summary


def row_profile_prompt_block(summary: Dict[str, Any]) -> str:
    if not summary.get("available"):
        return "Row-level exact-search memory:\n- no profiles.json or submission fallback was available; generated code must still expose per-row decision trace"
    hard = summary.get("hardest_tail", [])[:10]
    improved = summary.get("top_improved_rows", [])[:6]
    regressed = summary.get("top_regressed_rows", [])[:6]
    buckets = summary.get("path_length_buckets", {})
    hard_ids = summary.get("hard_row_ids", [])[:10]
    micro_pack = summary.get("hard_row_micro_pack", [])[:6]
    return "\n".join([
        "Row-level exact-search memory and hard-row target contract:",
        f"- source_path: {summary.get('source_path')}",
        f"- fallback_submission_path: {summary.get('fallback_submission_path')}",
        f"- rows: {summary.get('rows')}",
        f"- path_length_buckets: {buckets}",
        f"- target_hard_row_ids: {hard_ids}",
        f"- hardest_tail: {hard}",
        f"- hard_row_micro_pack: {micro_pack}",
        f"- top_improved_rows_from_profile: {improved}",
        f"- top_regressed_rows_from_profile: {regressed}",
        "- mandatory next-candidate goal: improve at least one target_hard_row_id by exact replay, regress zero rows, and expose selected_lane/baseline_len/candidate_len/rollback_reason.",
        "- do not submit an optimized_submission replay wrapper as improvement; a no-op candidate must fail explicitly.",
    ])
