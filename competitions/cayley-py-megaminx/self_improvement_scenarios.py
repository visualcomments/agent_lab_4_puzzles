"""Reusable self-improvement scenario primitives for the Megaminx pipeline.

The basic scenario hardens the outer loop: provider failures and baseline-identical
fallbacks are failed generation attempts, not candidates.  The advanced scenario
adds an evaluator-driven lane contract that prompts and generated solvers can use
to make progress on hard rows instead of repeatedly replaying the incumbent.
"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

BASIC_MAX_FAILURES_BEFORE_LIVE_CANDIDATE = 3

ADVANCED_LANE_IDS = {
    "lane_patch",
    "lane_fresh",
    "lane_params",
    "lane_hard_row_micro",
    "lane_portfolio",
}


@dataclass(frozen=True)
class BasicScenarioPolicy:
    """Fail-fast policy for self-improving code generation rounds."""

    strict_codegen: bool = True
    reject_baseline_identical_solver: bool = True
    reject_baseline_identical_submission: bool = True
    require_per_row_improvement: bool = True
    max_codegen_failures_before_live_candidate: int = BASIC_MAX_FAILURES_BEFORE_LIVE_CANDIDATE
    archive_attempt_artifacts: bool = True


@dataclass(frozen=True)
class CandidateManifest:
    """Machine-readable contract every advanced candidate should expose."""

    lane_id: str
    changed_mechanism: str
    target_rows: List[Any]
    expected_improved_rows: int
    fallback_policy: str
    novelty_claim: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def basic_policy_for_round(*, keep_improving: bool, self_improve_prompts: bool, reject_identical_candidates: bool) -> BasicScenarioPolicy:
    """Return the default basic hardening policy for an outer-loop round."""

    strict = bool(keep_improving or self_improve_prompts or reject_identical_candidates)
    return BasicScenarioPolicy(strict_codegen=strict)


def validate_candidate_manifest(payload: Any) -> tuple[bool, List[str]]:
    """Validate the advanced lane manifest without executing generated code."""

    errors: List[str] = []
    if not isinstance(payload, dict):
        return False, ["manifest must be a dict"]

    required = [
        "lane_id",
        "changed_mechanism",
        "target_rows",
        "expected_improved_rows",
        "fallback_policy",
        "novelty_claim",
    ]
    for key in required:
        if key not in payload:
            errors.append(f"missing {key}")

    lane_id = str(payload.get("lane_id") or "")
    if lane_id not in ADVANCED_LANE_IDS:
        errors.append(f"lane_id must be one of {sorted(ADVANCED_LANE_IDS)}")

    changed_mechanism = str(payload.get("changed_mechanism") or "").strip()
    if len(changed_mechanism) < 12:
        errors.append("changed_mechanism must describe a real algorithmic/search change")
    banned_mechanism_markers = ("comment", "rename", "format", "wrapper only", "lookup only", "baseline replay")
    if any(marker in changed_mechanism.lower() for marker in banned_mechanism_markers):
        errors.append("changed_mechanism cannot be a cosmetic/wrapper/lookup-only change")

    target_rows = payload.get("target_rows")
    if not isinstance(target_rows, list):
        errors.append("target_rows must be a list")

    try:
        expected = int(payload.get("expected_improved_rows"))
        if expected < 1:
            errors.append("expected_improved_rows must be >= 1")
    except Exception:
        errors.append("expected_improved_rows must be an integer")

    fallback_policy = str(payload.get("fallback_policy") or "").lower()
    if "rollback" not in fallback_policy and "fallback" not in fallback_policy:
        errors.append("fallback_policy must mention rollback/fallback on regression or replay failure")

    novelty_claim = str(payload.get("novelty_claim") or "").strip()
    if len(novelty_claim) < 12:
        errors.append("novelty_claim must explain why this is not incumbent replay")

    return not errors, errors


def build_hard_row_micro_pack(row_profile_summary: Dict[str, Any], *, limit: int = 6) -> List[Dict[str, Any]]:
    """Extract compact hard-row tasks for lane_hard_row_micro prompts."""

    micro_pack = row_profile_summary.get("hard_row_micro_pack")
    if isinstance(micro_pack, list) and micro_pack:
        return [dict(item) for item in micro_pack[:limit] if isinstance(item, dict)]

    hard_rows = row_profile_summary.get("hardest_tail") or row_profile_summary.get("top_hard_rows") or []
    out: List[Dict[str, Any]] = []
    for row in hard_rows[:limit]:
        if not isinstance(row, dict):
            continue
        baseline_len = int(row.get("current_best_len") or row.get("baseline_len") or 0)
        out.append(
            {
                "row_id": row.get("row_id"),
                "baseline_len": baseline_len,
                "target_len": max(0, baseline_len - 1),
                "contract": "shorten this row by >=1 move under exact replay or return no_candidate for this lane",
            }
        )
    return out


def append_candidate_archive_entry(archive_jsonl: Path, entry: Dict[str, Any]) -> None:
    """Append a structured candidate record for later prompt evolution."""

    archive_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with archive_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def rejection_bucket_from_history_entry(entry: Dict[str, Any]) -> str:
    """Normalize noisy outer-loop history into stable self-improvement buckets."""

    reasons = entry.get("rejection_reasons") or []
    reason_text = " ".join(str(r) for r in reasons).lower()
    error_text = str(entry.get("error") or "").lower()
    combined = f"{reason_text} {error_text}"
    if "credential" in combined or "ratelimit" in combined or "rate limit" in combined or "provider" in combined or "timeout" in combined:
        return "provider_or_backend_failure"
    if "identical_solver" in combined or "identical_submission" in combined or "baseline" in combined:
        return "no_novelty_or_baseline_fallback"
    if "per_row" in combined or "improved_rows" in combined:
        return "no_per_row_improvement"
    if "score_not_improved" in combined:
        return "score_not_improved"
    if entry.get("accepted") is True:
        return "accepted"
    return "unknown_failure"


__all__ = [
    "ADVANCED_LANE_IDS",
    "BasicScenarioPolicy",
    "CandidateManifest",
    "append_candidate_archive_entry",
    "basic_policy_for_round",
    "build_hard_row_micro_pack",
    "rejection_bucket_from_history_entry",
    "validate_candidate_manifest",
]
